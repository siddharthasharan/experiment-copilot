"""
v2 synthetic data generator: Meta-scale baseline metrics at per-user grain.

Design decisions (see README "v2 data design"):
  - Per-user rows, NOT daily rows: 400k rows instead of ~17M. Daily grain
    exists only where a trap needs temporal structure, encoded via
    `enrollment_day` (peeking, novelty) and `report_day` (guardrail).
  - Heavy-tailed PRIMARY for the whale trap: revenue_per_user is lognormal
    with sigma=1.6 (CV~3.5, mean ~$2.23/28d). The v1 trap was dishonest —
    its bomb landed on a secondary metric while the headline ran on
    `converted`. Here the whales go into the primary metric itself.
  - Whale count scales with n: 0.125% of the treatment arm (250 at n=400k),
    set to a fixed $400-800 spend tier (vs ~$2.23 mean). Fixed values keep
    the std-ratio stable (~3) across seeds; random-pick x-multiplier designs
    had 20% seed-miss rates. Targeting std-ratio ~3 for ~100% detection.
  - Pre-period covariates (28d) correlated with experiment metrics at
    rho=0.75, targeting CUPED R^2 ~ 0.5-0.6 on engagement and revenue.
  - Novelty is cross-sectional by enrollment cohort: lift decays with days
    since enrollment. Stylized (real novelty decays within-user), but it
    produces exactly the analytical signature — headline significant,
    long-exposure cohort null, early/late heterogeneity significant.
  - Peeking is a net-zero transient: +4% for enrollment_day<7, -1.33%
    after, so the full sample is null but an early stopper ships it.

Everything has known ground truth: each scenario writes an experiment.json
answer key with true lifts, trap flags, expected verdict, and per-metric
power adequacy at the chosen n.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .power import (achieved_power, achieved_power_binary, apply_cuped,
                    required_n, required_n_binary)

N_USERS = 400_000
EXP_DAYS = 28
PRE_DAYS = 28

# --- metric design constants ------------------------------------------------
ENG_CV = 1.0
ENG_S = math.sqrt(math.log(2.0))          # lognormal s giving CV=1
ENG_MEAN = 100.0                          # engagement index, 28d
REV_SIGMA = 1.6                           # heavy tail: CV ~= 3.45
REV_LOGMEAN = -0.48                       # -> mean ~$2.23 per 28d
CONV_BASE = 0.30
REPORTS_BASE = 0.001
PRE_RHO = 0.75                            # pre/exp correlation (engagement)
# Revenue needs higher log-space rho: for heavy-tailed lognormal(sigma=1.6)
# the LEVELS correlation is attenuated — corr_levels ~= (exp(rho*s^2)-1)/(exp(s^2)-1).
# rho=0.75 gives levels R^2 ~ 0.24; rho=0.89 gives ~0.55. Spend habits are
# persistent, so this is realistic, and CUPED on revenue is load-bearing.
PRE_RHO_REV = 0.89
WHALE_FRAC = 0.00125                       # of treatment arm (250 at n=400k)
# Whales get FIXED uniform values (not a multiplier): random selection +
# multiplier has high seed variance (sometimes the picked users are small).
# Fixed $400-800 values give a stable std-ratio ~3 across seeds and mirror
# reality — whales are a distinct spend tier, not scaled-up average users.
WHALE_LO = 400.0
WHALE_HI = 800.0


@dataclass
class V2Scenario:
    """A fully specified v2 experiment scenario (baseline metrics)."""

    name: str
    seed: int = 42
    n_users: int = N_USERS
    engagement_lift: float = 0.0   # relative lift on engagement (treatment)
    revenue_lift: float = 0.0      # relative lift on revenue (treatment)
    reports_lift: float = 0.0      # relative lift on reports (treatment)
    traps: tuple[str, ...] = ()    # srm | peeking | novelty | whale | guardrail
    allocation: float = 0.5        # intended treatment fraction
    expected_verdict: str = "don't launch"
    expected_flag: str = ""


@dataclass
class V2Experiment:
    df: pd.DataFrame
    ground_truth: dict = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _apply_srm(n: int, rng: np.random.Generator, severity: float = 0.55) -> np.ndarray:
    """Re-assign users so the split is exactly (1-severity, severity)."""
    n_treat = int(n * severity)
    idx = rng.choice(n, size=n_treat, replace=False)
    groups = np.zeros(n, dtype=np.int8)
    groups[idx] = 1
    return groups


def _segments(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    countries = rng.choice(
        ["US", "IN", "BR", "GB", "DE"], size=n, p=[0.40, 0.20, 0.15, 0.10, 0.15])
    platforms = rng.choice(
        ["ios", "android", "web"], size=n, p=[0.45, 0.45, 0.10])
    tenure = np.where(rng.random(n) < 0.30, "new", "returning")
    return {"country": countries, "platform": platforms, "tenure": tenure}


def generate_v2(spec: V2Scenario) -> V2Experiment:
    """Generate one v2 scenario at per-user grain. Fully vectorized."""
    rng = _rng(spec.seed)
    n = spec.n_users

    # --- assignment ------------------------------------------------------
    if "srm" in spec.traps:
        groups = _apply_srm(n, rng, severity=0.55)
    else:
        groups = (rng.random(n) < spec.allocation).astype(np.int8)
    treat = groups == 1
    treat_idx = np.flatnonzero(treat)

    segs = _segments(n, rng)
    enrollment_day = rng.integers(0, EXP_DAYS, size=n).astype(np.int16)

    # --- latents ----------------------------------------------------------
    eta = rng.standard_normal(n)                       # engagement propensity
    zeta = 0.5 * eta + math.sqrt(0.75) * rng.standard_normal(n)  # revenue, corr w/ eta
    eps = math.sqrt(1.0 - PRE_RHO**2)

    # --- engagement: lognormal-ish, CV ~= 1 -------------------------------
    pre_eng = np.exp(math.log(ENG_MEAN) + ENG_S * eta)
    exp_eng = np.exp(math.log(ENG_MEAN)
                     + ENG_S * (PRE_RHO * eta + eps * rng.standard_normal(n)))

    # treatment multipliers by scenario / trap
    mult = np.ones(n)
    if spec.engagement_lift:
        mult[treat] *= 1.0 + spec.engagement_lift
    if "peeking" in spec.traps:
        early = enrollment_day < 7
        mult[treat & early] *= 1.04
        mult[treat & ~early] *= 1.0 - 0.04 * 7 / 21   # -1.333%: net zero
    if "novelty" in spec.traps:
        days_since = EXP_DAYS - enrollment_day
        mult[treat] *= 1.0 + 0.06 * np.exp(-days_since[treat] / 7.0)
    exp_eng = exp_eng * mult

    # --- revenue: heavy-tailed primary, lognormal sigma=1.6 ----------------
    pre_rev = np.exp(REV_LOGMEAN + REV_SIGMA * zeta)
    eps_rev = math.sqrt(1.0 - PRE_RHO_REV**2)
    exp_rev = np.exp(REV_LOGMEAN
                     + REV_SIGMA * (PRE_RHO_REV * zeta + eps_rev * rng.standard_normal(n)))
    if spec.revenue_lift:
        exp_rev[treat] *= 1.0 + spec.revenue_lift

    n_whales = 0
    if "whale" in spec.traps:
        # HONEST trap: whales go into the primary metric itself, as a fixed
        # high-spend tier ($400-800 vs ~$2.23 mean). 250 whales ≈ +34% on the
        # headline mean — the trap has teeth — while the std-ratio (~3)
        # reliably flags the variance as whale-driven.
        n_whales = max(1, int(treat.sum() * WHALE_FRAC))
        chosen = rng.choice(treat_idx, size=n_whales, replace=False)
        exp_rev[chosen] = rng.uniform(WHALE_LO, WHALE_HI, size=n_whales)

    # --- converted: binary flagship secondary ------------------------------
    logit_p = math.log(CONV_BASE / (1 - CONV_BASE)) + 0.4 * eta
    p_conv = 1.0 / (1.0 + np.exp(-logit_p))
    converted = (rng.random(n) < p_conv).astype(np.int8)
    p_pre = 1.0 / (1.0 + np.exp(-(math.log(0.012 / 0.988) + 0.4 * eta)))
    pre_conv = rng.binomial(PRE_DAYS, np.clip(p_pre, 0, 1))

    # --- reports / takedowns: rare guardrail -------------------------------
    p_rep = np.full(n, REPORTS_BASE)
    p_rep[treat] *= 1.0 + spec.reports_lift
    reports = (rng.random(n) < p_rep).astype(np.int8)
    report_day = np.full(n, -1, dtype=np.int16)
    rep_idx = np.flatnonzero(reports == 1)
    report_day[rep_idx] = rng.integers(0, EXP_DAYS, size=len(rep_idx)).astype(np.int16)
    takedowns = np.zeros(n, dtype=np.int8)
    takedowns[rep_idx] = (rng.random(len(rep_idx)) < 0.6).astype(np.int8)
    pre_reports = rng.binomial(PRE_DAYS, REPORTS_BASE / PRE_DAYS, size=n).astype(np.int16)

    df = pd.DataFrame({
        "user_id": np.arange(n, dtype=np.int64),
        "group": np.where(treat, "treatment", "control"),
        "country": segs["country"],
        "platform": segs["platform"],
        "tenure": segs["tenure"],
        "enrollment_day": enrollment_day,
        "pre_engagement_28d": pre_eng,
        "pre_revenue_28d": pre_rev,
        "pre_converted_28d": pre_conv.astype(np.int16),
        "pre_reports_28d": pre_reports,
        "engagement": exp_eng,
        "revenue": exp_rev,
        "converted": converted,
        "reports": reports,
        "report_day": pd.array(np.where(reports == 1, report_day, pd.NA),
                              dtype="Int16"),
        "takedowns": takedowns,
    })

    # --- answer key ---------------------------------------------------------
    n_arm = n / 2
    truth: dict = {
        "scenario": spec.name,
        "n_users": n,
        "seed": spec.seed,
        "intended_allocation": spec.allocation,
        "actual_allocation": round(float(treat.mean()), 4),
        "traps": list(spec.traps),
        "true_lifts": {
            "engagement": spec.engagement_lift,
            "revenue": spec.revenue_lift,
            "reports": spec.reports_lift,
        },
        "expected_verdict": spec.expected_verdict,
        "expected_flag": spec.expected_flag,
        "design_notes": {
            "grain": "per-user (not daily rows)",
            "pre_period_days": PRE_DAYS,
            "exp_days": EXP_DAYS,
            "whale_trap": ({"n_whales": n_whales,
                             "value_range": [WHALE_LO, WHALE_HI],
                             "target_metric": "revenue (PRIMARY)"}
                            if "whale" in spec.traps else None),
        },
        "power_adequacy": {
            "engagement": _adequacy("continuous", ENG_CV, 0.02, n_arm),
            "revenue": _adequacy("continuous", 3.45, 0.02, n_arm),
            "converted": _adequacy("binary", CONV_BASE, 0.02, n_arm),
            "reports": _adequacy("binary", REPORTS_BASE, 0.35, n_arm),
        },
    }
    return V2Experiment(df=df, ground_truth=truth)


def _adequacy(kind: str, cv_or_p: float, lift: float, n_arm: float) -> dict:
    if kind == "continuous":
        need = required_n(cv_or_p, lift)
        need_c = apply_cuped(need, 0.55)
        pw = achieved_power(n_arm, cv_or_p, lift)
        pw_c = achieved_power(n_arm / 0.45, cv_or_p, lift)
    else:
        need = required_n_binary(cv_or_p, lift)
        need_c = need
        pw = achieved_power_binary(n_arm, cv_or_p, lift)
        pw_c = pw
    return {
        "mde": lift,
        "required_n_per_arm": round(need),
        "required_n_per_arm_cuped": round(need_c),
        "power_at_design_n": round(pw, 3),
        "power_at_design_n_cuped": round(pw_c, 3),
        "adequate": bool(pw >= 0.8 or pw_c >= 0.8),
    }


def v2_scenarios(seed_offset: int = 0) -> list[V2Scenario]:
    """The six v2 baseline scenarios. Realistic 1-2% MDEs, uniform n=400k."""
    return [
        V2Scenario("clean_win", seed=101 + seed_offset,
                   engagement_lift=0.02,
                   expected_verdict="launch",
                   expected_flag=""),
        V2Scenario("srm_trap", seed=102 + seed_offset,
                   engagement_lift=0.02, traps=("srm",),
                   expected_verdict="invalid",
                   expected_flag="SRM detected — do not trust any lift"),
        V2Scenario("peeking_trap", seed=103 + seed_offset,
                   traps=("peeking",),
                   expected_verdict="inconclusive",
                   expected_flag="early-cohort significance washes out full-sample; "
                                 "interim looks without correction"),
        V2Scenario("novelty_decay_trap", seed=104 + seed_offset,
                   traps=("novelty",),
                   expected_verdict="inconclusive",
                   expected_flag="lift concentrated in recent enrollees; "
                                 "long-exposure cohort shows ~null"),
        V2Scenario("guardrail_violation_trap", seed=105 + seed_offset,
                   engagement_lift=0.02, reports_lift=0.35,
                   traps=("guardrail",),
                   expected_verdict="don't launch",
                   expected_flag="reports up ~35% on treatment — guardrail veto"),
        V2Scenario("whale_variance_trap", seed=106 + seed_offset,
                   revenue_lift=0.02, traps=("whale",),
                   expected_verdict="inconclusive",
                   expected_flag="revenue lift driven by whale outliers — "
                                 "mean is fragile, investigate before launching"),
        V2Scenario("null_result", seed=107 + seed_offset,
                   expected_verdict="don't launch",
                   expected_flag=""),
    ]


def write_scenario(spec: V2Scenario, outdir: str,
                   sample_n: int | None = None,
                   sample_dir: str | None = None) -> V2Experiment:
    """Generate, write parquet + experiment.json. Optionally also write a
    fixed-seed sample for the repo."""
    exp = generate_v2(spec)
    os.makedirs(outdir, exist_ok=True)
    exp.df.to_parquet(os.path.join(outdir, f"{spec.name}.parquet"), index=False)
    with open(os.path.join(outdir, f"{spec.name}.experiment.json"), "w") as f:
        json.dump(exp.ground_truth, f, indent=2)
    if sample_n and sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
        rng = np.random.default_rng(999)
        idx = rng.choice(len(exp.df), size=sample_n, replace=False)
        sample = exp.df.iloc[np.sort(idx)].reset_index(drop=True)
        sample.to_parquet(os.path.join(sample_dir, f"{spec.name}.parquet"),
                          index=False)
        key = dict(exp.ground_truth)
        key["n_users"] = sample_n
        key["note"] = (f"20k-user committed sample of the full n={spec.n_users} "
                       "run; power adequacy figures refer to the full n.")
        with open(os.path.join(sample_dir, f"{spec.name}.experiment.json"), "w") as f:
            json.dump(key, f, indent=2)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser(description="v2 baseline-metrics generator")
    ap.add_argument("--scenario", default="all",
                    help="scenario name or 'all'")
    ap.add_argument("--n", type=int, default=N_USERS)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--outdir", default="eval_data_v2")
    ap.add_argument("--sample-n", type=int, default=20_000)
    ap.add_argument("--sample-dir", default="data_v2_sample")
    args = ap.parse_args()

    specs = v2_scenarios(seed_offset=args.seed_offset)
    if args.scenario != "all":
        specs = [s for s in specs if s.name == args.scenario]
        assert specs, f"unknown scenario {args.scenario}"
    for spec in specs:
        spec.n_users = args.n
        write_scenario(spec, args.outdir,
                       sample_n=args.sample_n, sample_dir=args.sample_dir)
        print(f"wrote {spec.name}: n={args.n:,}")


if __name__ == "__main__":
    main()
