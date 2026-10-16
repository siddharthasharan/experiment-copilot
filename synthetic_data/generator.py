"""Synthetic social-media experiment data generator.

Generates realistic A/B-test datasets for social platforms (Facebook, Instagram,
TikTok, Reddit, LinkedIn) across three metric perspectives:

  1. Flagship (core product health): reach, users, views, time on app, engagement
  2. Ads business: ad impressions, CTR, ad revenue
  3. Trust & Safety: reports, takedowns

Each experiment also bakes in *known ground truth*, including optional trap
scenarios (SRM, novelty decay, guardrail violations, whale variance) used by
the eval harness to grade whether the analysis agent catches them.

Two views are produced per experiment:
  - analyst view  -> assignments.csv + metrics_daily.csv (what the agent sees)
  - answer key    -> experiment.json (ground truth; NEVER shown to the agent)

Usage:
    python generator.py --platform instagram --scenario clean_win --n-users 10000 --days 14
    python generator.py --list-platforms
    python generator.py --list-scenarios
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Platform profiles: daily per-user base rates, tuned to each product's shape.
# TikTok  = view-heavy, long sessions.  LinkedIn = infrequent but high-value.
# Reddit  = comment-heavy, lower daily frequency.  Instagram = reels/stories.
# Facebook ("blue app") = broad feed mix, highest daily reach.
# ---------------------------------------------------------------------------


@dataclass
class PlatformProfile:
    name: str
    # --- flagship ---
    p_active: float            # P(user active on a given day)
    sessions_lambda: float     # mean sessions | active
    time_median: float          # median minutes | active
    time_sigma: float           # lognormal sigma for time spent
    views_median: float         # median content views | active
    views_sigma: float          # lognormal sigma for views
    like_rate: float            # P(like | view)
    comment_rate: float         # P(comment | view)
    share_rate: float           # P(share | view)
    # --- ads business ---
    ad_load: float              # fraction of views that are ad impressions
    ecpm: float                 # effective $ per 1000 ad impressions
    ctr: float                  # P(click | ad impression)
    # --- trust & safety ---
    p_report: float             # P(report | active user-day)
    p_takedown_given_report: float


PLATFORMS: dict[str, PlatformProfile] = {
    "facebook": PlatformProfile(
        name="facebook", p_active=0.62, sessions_lambda=3.2,
        time_median=28.0, time_sigma=0.9, views_median=140.0, views_sigma=0.9,
        like_rate=0.045, comment_rate=0.008, share_rate=0.004,
        ad_load=0.14, ecpm=9.50, ctr=0.011,
        p_report=0.0016, p_takedown_given_report=0.42,
    ),
    "instagram": PlatformProfile(
        name="instagram", p_active=0.58, sessions_lambda=4.1,
        time_median=32.0, time_sigma=0.95, views_median=170.0, views_sigma=0.95,
        like_rate=0.062, comment_rate=0.006, share_rate=0.007,
        ad_load=0.12, ecpm=8.20, ctr=0.009,
        p_report=0.0013, p_takedown_given_report=0.45,
    ),
    "tiktok": PlatformProfile(
        name="tiktok", p_active=0.55, sessions_lambda=5.4,
        time_median=52.0, time_sigma=0.85, views_median=320.0, views_sigma=0.85,
        like_rate=0.085, comment_rate=0.011, share_rate=0.014,
        ad_load=0.09, ecpm=6.10, ctr=0.008,
        p_report=0.0011, p_takedown_given_report=0.48,
    ),
    "reddit": PlatformProfile(
        name="reddit", p_active=0.34, sessions_lambda=2.2,
        time_median=24.0, time_sigma=1.0, views_median=85.0, views_sigma=1.0,
        like_rate=0.030, comment_rate=0.022, share_rate=0.002,
        ad_load=0.07, ecpm=4.40, ctr=0.006,
        p_report=0.0024, p_takedown_given_report=0.55,
    ),
    "linkedin": PlatformProfile(
        name="linkedin", p_active=0.28, sessions_lambda=1.6,
        time_median=14.0, time_sigma=0.9, views_median=45.0, views_sigma=0.9,
        like_rate=0.028, comment_rate=0.009, share_rate=0.003,
        ad_load=0.10, ecpm=22.00, ctr=0.007,
        p_report=0.0006, p_takedown_given_report=0.38,
    ),
}

# ---------------------------------------------------------------------------
# Scenarios: treatment effects + injected traps. `lifts` are multiplicative
# effects on treatment vs control. `expected_verdict` is the answer key for
# the eval harness: what a rigorous analyst should conclude.
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    hypothesis: str
    control_frac: float = 0.5
    lifts: dict[str, float] = field(default_factory=dict)
    novelty_tau_days: float | None = None   # effect decays as exp(-day/tau)
    whale_users: int = 0                   # # treatment users with 40x activity
    whale_multiplier: float = 40.0
    expected_verdict: str = ""


SCENARIOS: dict[str, Scenario] = {
    "clean_win": Scenario(
        name="clean_win",
        hypothesis="Ranking change lifts engagement without harming ads or safety.",
        lifts={"time_spent_min": 1.04, "views": 1.05, "likes": 1.06,
               "comments": 1.05, "shares": 1.05},
        expected_verdict="SHIP: significant engagement lift, no guardrail breach.",
    ),
    "srm": Scenario(
        name="srm",
        hypothesis="Same as clean_win, but a logging bug skews randomization 55/45.",
        control_frac=0.45,  # bug: treatment gets 55%
        lifts={"time_spent_min": 1.04, "views": 1.05},
        expected_verdict="INVALID: SRM detected (split != 50/50). Do not interpret results.",
    ),
    "novelty_decay": Scenario(
        name="novelty_decay",
        hypothesis="UI refresh gets a novelty bump that fades within two weeks.",
        lifts={"time_spent_min": 1.09, "views": 1.10, "likes": 1.08},
        novelty_tau_days=5.0,
        expected_verdict="HOLD: early lift is novelty-driven and decaying; extend run.",
    ),
    "guardrail_violation": Scenario(
        name="guardrail_violation",
        hypothesis="Engagement win that degrades trust & safety.",
        lifts={"time_spent_min": 1.05, "views": 1.06, "likes": 1.07,
               "reports": 1.35},
        expected_verdict="DO NOT SHIP: engagement up but reports/user up 35% (guardrail breach).",
    ),
    "whale_variance": Scenario(
        name="whale_variance",
        hypothesis="Apparent win driven entirely by a handful of outlier users.",
        lifts={"time_spent_min": 1.03},
        whale_users=8,
        expected_verdict="INCONCLUSIVE: lift driven by outliers; winsorize/segment before deciding.",
    ),
    "null_result": Scenario(
        name="null_result",
        hypothesis="Change has no real effect; noise only.",
        lifts={},
        expected_verdict="NO EFFECT: no significant movement; do not ship on noise.",
    ),
}

PRE_PERIOD_DAYS = 7  # pre-period for CUPED-style covariate adjustment


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _clip(p: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    return np.clip(p, lo, hi)


def generate_experiment(
    platform: str,
    scenario: str,
    n_users: int = 10_000,
    days: int = 14,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate one experiment. Returns (assignments, metrics_daily, answer_key)."""
    rng = np.random.default_rng(seed)
    prof = PLATFORMS[platform]
    scen = SCENARIOS[scenario]
    user_ids = np.arange(n_users)

    # --- randomization (with optional SRM bug) ---
    variant = np.where(rng.random(n_users) < scen.control_frac, "control", "treatment")

    # --- latent user quality: correlates pre-period with experiment period ---
    user_effect = rng.normal(0, 1, n_users)

    # --- pre-period aggregates (covariates for CUPED) ---
    pre_time = np.zeros(n_users)
    pre_views = np.zeros(n_users)
    pre_eng = np.zeros(n_users)
    for _ in range(PRE_PERIOD_DAYS):
        a = rng.random(n_users) < _clip(prof.p_active + 0.08 * user_effect)
        t = np.where(a, rng.lognormal(np.log(prof.time_median) + 0.35 * user_effect,
                                      prof.time_sigma), 0.0)
        v = np.where(a, rng.lognormal(np.log(prof.views_median) + 0.30 * user_effect,
                                      prof.views_sigma), 0.0)
        pre_time += t
        pre_views += v
        pre_eng += rng.poisson(v * (prof.like_rate + prof.comment_rate + prof.share_rate))

    assignments = pd.DataFrame({
        "user_id": user_ids,
        "variant": variant,
        "pre_time_spent_min": np.round(pre_time, 2),
        "pre_views": pre_views.astype(int),
        "pre_engagements": pre_eng.astype(int),
    })

    # --- experiment period, daily grain ---
    rows = []
    treat = (variant == "treatment")
    for day in range(days):
        # novelty decay: lift fades as exp(-day / tau)
        if scen.novelty_tau_days:
            decay = float(np.exp(-day / scen.novelty_tau_days))
        else:
            decay = 1.0

        active = rng.random(n_users) < _clip(prof.p_active + 0.08 * user_effect)
        sessions = np.where(active, 1 + rng.poisson(prof.sessions_lambda - 1, n_users), 0)

        time_spent = np.where(
            active,
            rng.lognormal(np.log(prof.time_median) + 0.35 * user_effect, prof.time_sigma),
            0.0,
        )
        views = np.where(
            active,
            rng.lognormal(np.log(prof.views_median) + 0.30 * user_effect, prof.views_sigma),
            0.0,
        )
        likes = rng.poisson(views * prof.like_rate)
        comments = rng.poisson(views * prof.comment_rate)
        shares = rng.poisson(views * prof.share_rate)

        # apply treatment lifts (flagship)
        for metric, lift in scen.lifts.items():
            if metric in ("time_spent_min", "views", "likes", "comments", "shares"):
                eff = 1.0 + (lift - 1.0) * decay
                arr = {"time_spent_min": time_spent, "views": views, "likes": likes,
                       "comments": comments, "shares": shares}[metric]
                if metric == "time_spent_min":
                    time_spent = np.where(treat, arr * eff, arr)
                elif metric == "views":
                    views = np.where(treat, arr * eff, arr)
                elif metric == "likes":
                    likes = np.where(treat, rng.poisson(arr * eff), arr)
                elif metric == "comments":
                    comments = np.where(treat, rng.poisson(arr * eff), arr)
                elif metric == "shares":
                    shares = np.where(treat, rng.poisson(arr * eff), arr)

        # whale trap: a few treatment users with extreme activity
        if scen.whale_users and day == 0:
            whales = rng.choice(np.where(treat)[0],
                                size=min(scen.whale_users, treat.sum()), replace=False)
        if scen.whale_users:
            time_spent = time_spent.copy()
            views = views.copy()
            time_spent[whales] *= scen.whale_multiplier
            views[whales] *= scen.whale_multiplier

        # --- ads business ---
        ad_impr = rng.binomial(np.round(views).astype(int), prof.ad_load)
        ad_clicks = rng.binomial(ad_impr, prof.ctr)
        ad_revenue = ad_impr / 1000.0 * prof.ecpm * rng.lognormal(0, 0.1, n_users)

        # --- trust & safety ---
        p_rep = _clip(prof.p_report * (1 + 0.2 * user_effect))
        rep_lift = scen.lifts.get("reports", 1.0)
        p_rep = np.where(treat, _clip(p_rep * rep_lift), p_rep)
        reports = (rng.random(n_users) < p_rep).astype(int)
        takedowns = (rng.random(n_users) < prof.p_takedown_given_report).astype(int) * reports

        day_df = pd.DataFrame({
            "user_id": user_ids, "day": day, "variant": variant,
            "active": active.astype(int), "sessions": sessions,
            "time_spent_min": np.round(time_spent, 2),
            "views": np.round(views).astype(int),
            "likes": likes, "comments": comments, "shares": shares,
            "ad_impressions": ad_impr, "ad_clicks": ad_clicks,
            "ad_revenue_usd": np.round(ad_revenue, 4),
            "reports": reports, "takedowns": takedowns,
        })
        rows.append(day_df)

    metrics_daily = pd.concat(rows, ignore_index=True)

    answer_key = {
        "platform": platform,
        "scenario": scenario,
        "hypothesis": scen.hypothesis,
        "n_users": n_users,
        "days": days,
        "seed": seed,
        "control_frac_intended": 0.5,
        "control_frac_actual": float(scen.control_frac),
        "true_lifts": scen.lifts,
        "novelty_tau_days": scen.novelty_tau_days,
        "whale_users": scen.whale_users,
        "injected_traps": [
            t for t, on in {
                "srm": scen.control_frac != 0.5,
                "novelty_decay": scen.novelty_tau_days is not None,
                "guardrail_violation": scen.lifts.get("reports", 1.0) > 1.0,
                "whale_variance": scen.whale_users > 0,
            }.items() if on
        ],
        "expected_verdict": scen.expected_verdict,
    }
    return assignments, metrics_daily, answer_key


def save_experiment(assignments: pd.DataFrame, metrics: pd.DataFrame,
                     answer_key: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out / "assignments.csv", index=False)
    metrics.to_csv(out / "metrics_daily.csv", index=False)
    (out / "experiment.json").write_text(json.dumps(answer_key, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic social-media experiment generator")
    ap.add_argument("--platform", default="instagram", choices=list(PLATFORMS))
    ap.add_argument("--scenario", default="clean_win", choices=list(SCENARIOS))
    ap.add_argument("--n-users", type=int, default=10_000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    ap.add_argument("--list-platforms", action="store_true")
    ap.add_argument("--list-scenarios", action="store_true")
    args = ap.parse_args()

    if args.list_platforms:
        for name, p in PLATFORMS.items():
            print(f"{name:10s} p_active={p.p_active:.2f} views/day~{p.views_median:.0f} "
                  f"ecpm=${p.ecpm:.2f}")
        return
    if args.list_scenarios:
        for name, s in SCENARIOS.items():
            print(f"{name:20s} traps={s.expected_verdict[:60]}")
            print(f"{'':20s} hypothesis: {s.hypothesis}")
        return

    exp_id = f"{args.platform}_{args.scenario}_s{args.seed}"
    assignments, metrics, key = generate_experiment(
        args.platform, args.scenario, args.n_users, args.days, args.seed)
    out = save_experiment(assignments, metrics, key, Path(args.out) / exp_id)

    split = assignments["variant"].value_counts(normalize=True)
    m = metrics.groupby("variant")["time_spent_min"].mean()
    print(f"Wrote {out}  ({len(metrics):,} user-days)")
    print(f"Split: {split.to_dict()}")
    print(f"Mean time_spent_min/day: {m.to_dict()}")
    print(f"Injected traps: {key['injected_traps'] or 'none'}")


if __name__ == "__main__":
    main()
