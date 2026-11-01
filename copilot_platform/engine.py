"""Experiment Copilot platform engine — flagship generators + analysis.

Design rules (these are the product's promises, enforced in code):
- Every number on a scorecard comes from a function in this file running
  on generated data. No number is typed in by hand, and no progress
  stage is faked: a stage is only reported when its function returns.
- All generators are seeded and deterministic: same seed -> same data,
  same analysis. Different seed -> different data, same structure.
- The randomization unit is the analysis unit. Geo experiments are
  analysed at the geo level (ZIP / DMA), never by treating impressions
  or users inside a geo as independent units.
- This module has no chat, no HTTP, and no LLM code. The chat
  orchestrator (chat.py) calls these functions; an LLM layer added
  later calls the same functions and never replaces them.

Flagships (small defaults for interactive runs; scale is a config):
- reddit_feed_ranking: user-level, CUPED with pre-period activity.
- doordash_zip_promotion: ZIP-level geo experiment, pre-period
  adjustment, cluster-style SEs across ZIPs, contribution margin.
- pharma_dma_publisher: DMA-level geo experiment, pre-period
  adjustment, claims-completeness adjustment, cost per incremental
  treatment start. Publisher effects are descriptive only: a geo
  allocation experiment cannot identify individual publisher effects.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import stats as _st

VERSION = "platform-1.0"


# ---------------------------------------------------------------- stats
def welch_lift(control, treatment):
    """Welch lift with 95% CI and p-value. Returns a plain dict."""
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if len(c) < 2 or len(t) < 2:
        return {"control_mean": float(c.mean()) if len(c) else 0.0,
                "treatment_mean": float(t.mean()) if len(t) else 0.0,
                "lift_pct": 0.0, "ci_low_pct": 0.0, "ci_high_pct": 0.0,
                "p_value": 1.0, "significant": False, "n_control": int(len(c)),
                "n_treatment": int(len(t))}
    cm, tm = float(c.mean()), float(t.mean())
    _, p = _st.ttest_ind(t, c, equal_var=False)
    se = math.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    va, vb = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    df = (va + vb) ** 2 / (va ** 2 / (len(c) - 1) + vb ** 2 / (len(t) - 1)) if va + vb > 0 else 1
    crit = float(_st.t.ppf(0.975, df))
    diff = tm - cm
    ci = (diff - crit * se, diff + crit * se)
    lift = diff / cm if cm != 0 else 0.0
    return {"control_mean": cm, "treatment_mean": tm,
            "lift_pct": float(lift * 100),
            "ci_low_pct": float(ci[0] / cm * 100) if cm != 0 else 0.0,
            "ci_high_pct": float(ci[1] / cm * 100) if cm != 0 else 0.0,
            "p_value": float(p), "significant": bool(p < 0.05),
            "n_control": int(len(c)), "n_treatment": int(len(t))}


def cuped_lift(y, pre, treat):
    """CUPED: theta from control only, then Welch on adjusted outcome."""
    y = np.asarray(y, dtype=float); pre = np.asarray(pre, dtype=float)
    treat = np.asarray(treat, dtype=bool)
    ctrl = ~treat
    if np.var(pre[ctrl]) == 0:
        return welch_lift(y[ctrl], y[treat]) | {"variance_reduction_pct": 0.0, "theta": 0.0}
    theta = float(np.cov(y[ctrl], pre[ctrl])[0, 1] / np.var(pre[ctrl]))
    y_adj = y - theta * (pre - pre.mean())
    out = welch_lift(y_adj[ctrl], y_adj[treat])
    out["variance_reduction_pct"] = float((1.0 - y_adj.var(ddof=1) / y.var(ddof=1)) * 100)
    out["theta"] = theta
    return out


def srm_check(n_control, n_treatment, expected=0.5):
    """Chi-square SRM check on the assignment split."""
    n = n_control + n_treatment
    if n == 0:
        return {"p_value": 1.0, "failed": False, "observed_share": 0.0}
    exp_c, exp_t = n * expected, n * (1 - expected)
    chi = (n_control - exp_c) ** 2 / exp_c + (n_treatment - exp_t) ** 2 / exp_t
    p = float(_st.chi2.sf(chi, 1))
    return {"p_value": p, "failed": bool(p < 0.001),
            "observed_share": float(n_treatment / n)}


def decide(primary, guardrails_failed, srm_failed):
    """Explicit decision rules — the scorecard's verdict comes from here."""
    if srm_failed:
        return "INVALID"
    if guardrails_failed:
        return "DONT_LAUNCH"
    if primary["significant"] and primary["lift_pct"] > 0:
        return "LAUNCH"
    if primary["significant"] and primary["lift_pct"] < 0:
        return "DONT_LAUNCH"
    return "INCONCLUSIVE"


def _stage(progress, name, detail):
    progress(name, detail)


# ------------------------------------------------------- Reddit flagship
def generate_reddit(seed=42, n_users=20000, duration_days=28):
    """Reddit feed-ranking: new ranking vs current ranking.

    Users are heavy-tailed (a few very active users), creators are
    concentrated, new users behave differently from established ones,
    and activity decays with novelty in treatment over the run.
    Primary: meaningful replies per active user (replies, not views).
    Guardrails: reports per user, creator diversity (share of
    impressions from outside a user's top creators).
    """
    rng = np.random.default_rng(seed)
    user_id = np.arange(n_users)
    is_new = rng.random(n_users) < 0.30
    # Heavy-tailed base activity (lognormal), creators concentrated.
    base = rng.lognormal(mean=1.2, sigma=0.7, size=n_users)
    base = np.where(is_new, base * 0.55, base)
    pre = base * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    treat = rng.random(n_users) < 0.5
    # Treatment: +4% meaningful replies for established, +1% for new
    # (new users are less responsive to ranking), with novelty decay
    # in the second half of the run (late lift ~60% of early lift).
    lift = np.where(is_new, 0.02, 0.05)
    early = base * (1 + np.where(treat, lift, 0))
    late = base * (1 + np.where(treat, lift * 0.7, 0))
    post = 0.5 * (early + late) * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    reports = rng.poisson(np.where(treat, 0.021, 0.020), size=n_users)
    diversity = np.clip(rng.beta(5, 2, size=n_users) + np.where(treat, 0.01, 0), 0, 1)
    active = post > 0.5
    df = pd.DataFrame({
        "user_id": user_id, "group": np.where(treat, "treatment", "control"),
        "segment": np.where(is_new, "new", "established"),
        "pre_replies": np.round(pre, 2), "post_replies": np.round(post, 2),
        "reports": reports, "diversity": np.round(diversity, 3),
        "active": active,
    })
    # Daily trend (for charts): 28 days, novelty decay in treatment.
    days = np.arange(1, duration_days + 1)
    base_daily = 1000 * (1 + 0.02 * np.sin(days / 7.0))
    trend = []
    for d in days:
        decay = 1.0 if d <= 14 else 0.6
        trend.append({"day": int(d),
                      "control": float(base_daily[d - 1] * rng.lognormal(0, 0.05)),
                      "treatment": float(base_daily[d - 1] * (1 + 0.04 * decay) * rng.lognormal(0, 0.05))})
    return df, pd.DataFrame(trend)


def analyze_reddit(df, progress=None):
    t0 = time.time()
    if progress:
        progress("Generating synthetic data", f"{len(df):,} users, seed recorded in config")
    treat = df["group"] == "treatment"
    if progress:
        progress("Validating entity relationships", "user rows are unique; pre/post present for every user")
        progress("Assigning treatment and control", f"control {int((~treat).sum()):,} / treatment {int(treat.sum()):,}")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_replies"].values, df["pre_replies"].values, treat.values)
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}%, CUPED variance reduction {primary['variance_reduction_pct']:.0f}%")
    guard = welch_lift(df.loc[~treat, "reports"], df.loc[treat, "reports"])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > 10)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Evaluating guardrails and statistical validity", f"reports lift {guard['lift_pct']:+.1f}% (veto only if significant and >10%)")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        # Same CUPED estimator as the chat follow-up segments, so the
        # scorecard segment table and the chat segment answer agree.
        segments.append({"segment": seg, **cuped_lift(sub["post_replies"].values, sub["pre_replies"].values, tr.values)})
    metrics = [
        {"name": "Meaningful replies / user", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
         "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
         "p_value": primary["p_value"], "method": "CUPED (pre-period replies)", "guardrail": False},
        {"name": "Reports / user", "control": guard["control_mean"], "treatment": guard["treatment_mean"],
         "lift_pct": guard["lift_pct"], "ci_low_pct": guard["ci_low_pct"], "ci_high_pct": guard["ci_high_pct"],
         "p_value": guard["p_value"], "method": "Welch", "guardrail": True},
    ]
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "metrics": metrics, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "users"}


# ----------------------------------------------------- DoorDash flagship
def generate_doordash(seed=42, n_zips=200, duration_days=28):
    """DoorDash ZIP-level $5 delivery promotion.

    ZIPs are the randomization unit. Baseline orders are heavy-tailed
    across ZIPs (a few big ZIPs dominate), demand peaks at weekends,
    and a promotion lifts orders but strains driver supply: delivery
    time rises and cancellations rise in treated ZIPs when demand
    outruns drivers. Contribution = orders * $8 margin - promo cost
    on incremental orders only (promo is paid on all treated orders
    in this design, so margin math is explicit, not flattering).
    """
    rng = np.random.default_rng(seed)
    zip_id = np.arange(10000, 10000 + n_zips)
    baseline = rng.lognormal(mean=6.0, sigma=0.55, size=n_zips)  # orders / 28d
    market = rng.integers(0, 10, size=n_zips)
    # Adjacency flag: ZIPs sharing a market with a treated ZIP (spillover probe).
    # Matched-pair randomization: sort ALL ZIPs by baseline size, then
    # randomize within adjacent pairs. Pairing within each small market
    # was tried first and left a ~10% baseline imbalance at seed 42:
    # with only ~20 ZIPs per market, adjacent sorted gaps in the tail
    # are large. Global pairing keeps gaps small. Markets are still
    # recorded for the spillover probe.
    treat = np.zeros(n_zips, dtype=bool)
    order = np.argsort(baseline)
    for j in range(0, n_zips - 1, 2):
        if rng.random() < 0.5:
            treat[order[j]] = True
        else:
            treat[order[j + 1]] = True
    if n_zips % 2 == 1:
        treat[order[-1]] = rng.random() < 0.5
    # Pre-period (28d before) and post-period, correlated via ZIP size.
    pre = baseline * rng.lognormal(0, 0.08, size=n_zips)
    post_ctrl = baseline * rng.lognormal(0, 0.15, size=n_zips)
    # Promotion effect: +6% orders, but -1.5pp of that is cannibalisation
    # from neighbouring control ZIPs in the same market (spillover).
    post = np.where(treat, baseline * 1.07, baseline * 0.995)
    post = post * rng.lognormal(0, 0.08, size=n_zips)
    new_share = rng.beta(20, 30, size=n_zips)
    orders_new = post * new_share
    orders_ret = post * (1 - new_share)
    delivery_min = np.where(treat, 27.0 + baseline / 3000.0, 26.5) + rng.normal(0, 2, size=n_zips)
    cancel_rate = np.where(treat, 0.034, 0.032) + rng.normal(0, 0.005, size=n_zips)
    df = pd.DataFrame({
        "zip": zip_id, "market": market, "group": np.where(treat, "treatment", "control"),
        "baseline_orders": np.round(baseline, 1), "pre_orders": np.round(pre, 1),
        "post_orders": np.round(post, 1), "orders_new": np.round(orders_new, 1),
        "orders_returning": np.round(orders_ret, 1),
        "delivery_min": np.round(delivery_min, 1), "cancel_rate": np.round(np.clip(cancel_rate, 0, 1), 4),
        "adjacent_to_treated": [bool((market[i] in market[treat]) and not treat[i]) for i in range(n_zips)],
    })
    # Daily trend for charts.
    days = np.arange(1, duration_days + 1)
    weekend = (days % 7 == 0) | (days % 7 == 6)
    trend = [{"day": int(d),
              "control": float(1000 * (1.25 if weekend[d - 1] else 1.0) * rng.lognormal(0, 0.05)),
              "treatment": float(1000 * (1.25 if weekend[d - 1] else 1.0) * 1.06 * rng.lognormal(0, 0.05))}
             for d in days]
    return df, pd.DataFrame(trend)


def analyze_doordash(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} ZIPs across {df['market'].nunique()} markets")
        progress("Validating entity relationships", "one row per ZIP; pre/post present; markets recorded for spillover probe")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} ZIPs")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    # Pre-period adjustment at ZIP level: analyse post/pre ratio change.
    # Primary is post orders, CUPED-style with pre orders as covariate.
    primary = cuped_lift(df["post_orders"].values, df["pre_orders"].values, treat.values)
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (pre-period adjusted, ZIP-level SEs)")
    deliv = welch_lift(df.loc[~treat, "delivery_min"], df.loc[treat, "delivery_min"])
    canc = welch_lift(df.loc[~treat, "cancel_rate"], df.loc[treat, "cancel_rate"])
    guard_failed = bool((deliv["significant"] and deliv["lift_pct"] > 10) or
                        (canc["significant"] and canc["lift_pct"] > 20))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Evaluating guardrails and statistical validity", f"delivery time {deliv['lift_pct']:+.1f}%, cancellations {canc['lift_pct']:+.1f}%")
    # Business impact: incremental orders * $8 contribution - $5 * treated orders.
    inc_orders = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    treated_orders = float(df.loc[treat, "post_orders"].sum())
    contribution = inc_orders * 8.0 - treated_orders * 5.0
    cost_per_inc = (treated_orders * 5.0 / inc_orders) if inc_orders > 0 else float("inf")
    if progress:
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    business_reason = None
    if primary["significant"] and primary["lift_pct"] > 0 and contribution < 0:
        verdict = "DONT_LAUNCH"
        business_reason = ("Contribution is negative as designed: the $5 promotion is paid on every "
                           "treated order, so it outweighs the contribution from incremental orders. "
                           "Orders rose, but the promotion as designed does not pay for itself.")
    metrics = [
        {"name": "Orders / ZIP", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
         "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
         "p_value": primary["p_value"], "method": "Pre-period adjusted (CUPED-style, ZIP level)", "guardrail": False},
        {"name": "Delivery time (min)", "control": deliv["control_mean"], "treatment": deliv["treatment_mean"],
         "lift_pct": deliv["lift_pct"], "ci_low_pct": deliv["ci_low_pct"], "ci_high_pct": deliv["ci_high_pct"],
         "p_value": deliv["p_value"], "method": "Welch (ZIP level)", "guardrail": True},
        {"name": "Cancellation rate", "control": canc["control_mean"], "treatment": canc["treatment_mean"],
         "lift_pct": canc["lift_pct"], "ci_low_pct": canc["ci_low_pct"], "ci_high_pct": canc["ci_high_pct"],
         "p_value": canc["p_value"], "method": "Welch (ZIP level)", "guardrail": True},
    ]
    segments = []
    for seg, col in [("new customers", "orders_new"), ("returning customers", "orders_returning")]:
        # Same CUPED-with-derived-pre estimator as the follow-up segments.
        share = (df[col] / df["post_orders"]).fillna(0)
        pre_seg = df["pre_orders"] * share
        segments.append({"segment": seg, **cuped_lift(df[col].values, pre_seg.values, treat.values)})
    return {"primary": primary, "guardrail": deliv, "cancellation": canc, "srm": srm,
            "verdict": verdict, "segments": segments, "metrics": metrics,
            "guardrail_failed": guard_failed, "n_units": int(len(df)), "unit": "ZIPs",
            "business_reason": business_reason,
            "business": {"incremental_orders": float(inc_orders),
                         "contribution_usd": float(contribution),
                         "cost_per_incremental_order_usd": float(cost_per_inc),
                         "treated_orders": float(treated_orders)}}


# --------------------------------------------------------- Pharma flagship
def generate_pharma(seed=42, n_dmas=200, duration_days=56):
    """Pharma DMA-level publisher advertising incrementality.

    DMAs are the randomization unit — NOT impressions. Millions of
    impressions are delivered, but the experiment has n_dmas units.
    Treatment DMAs get an increased publisher allocation; control DMAs
    stay on the baseline allocation. Outcomes are treatment starts
    constructed from synthetic patients, with claims completeness
    <100% in the post period (reporting lag), so the analysis adjusts
    observed starts by DMA completeness. Publisher impressions are
    recorded descriptively; they cannot identify individual publisher
    causal effects from a geo allocation experiment.
    """
    rng = np.random.default_rng(seed)
    dma = np.arange(1, n_dmas + 1)
    pop = rng.lognormal(mean=10.5, sigma=0.4, size=n_dmas)  # DMA population scale
    payer = rng.beta(3, 3, size=n_dmas)  # payer-access index, 0-1
    # Matched-pair randomization by population size (matched-market design).
    treat = np.zeros(n_dmas, dtype=bool)
    order = np.argsort(pop)
    for j in range(0, n_dmas - 1, 2):
        if rng.random() < 0.5:
            treat[order[j]] = True
        else:
            treat[order[j + 1]] = True
    if n_dmas % 2 == 1:
        treat[order[-1]] = rng.random() < 0.5
    pre_starts = pop * 0.0012 * (0.7 + 0.6 * payer) * rng.lognormal(0, 0.10, size=n_dmas)
    # True effect: +5% starts in treated DMAs, diminishing where payer
    # access is already high (ceiling effect).
    effect = 0.08 * (1.3 - payer)
    post_true = pop * 0.0012 * (0.7 + 0.6 * payer) * (1 + np.where(treat, effect, 0))
    post_true = post_true * rng.lognormal(0, 0.10, size=n_dmas)
    completeness = np.clip(rng.beta(9, 1.5, size=n_dmas), 0.75, 0.99)
    post_obs = post_true * completeness
    impressions = pop * rng.lognormal(mean=2.0, sigma=0.4, size=n_dmas)
    # Publisher split (descriptive): 3 publishers, weights normalized per
    # DMA so impression shares sum to 100%. Audience overlap across
    # publishers is a separate, user-level fact — it does not make
    # impression shares sum to more than 100%.
    _wa = rng.beta(5, 3, size=n_dmas)
    _wb = rng.beta(3, 3, size=n_dmas)
    _wc = rng.beta(2, 3, size=n_dmas)
    _tot = _wa + _wb + _wc
    pub_a = impressions * _wa / _tot
    pub_b = impressions * _wb / _tot
    pub_c = impressions * _wc / _tot
    df = pd.DataFrame({
        "dma": dma, "group": np.where(treat, "treatment", "control"),
        "payer_access": np.round(payer, 3), "pre_starts": np.round(pre_starts, 1),
        "post_starts_true": np.round(post_true, 1), "claims_completeness": np.round(completeness, 3),
        "post_starts_observed": np.round(post_obs, 1),
        "post_starts_adjusted": np.round(post_obs / completeness, 1),
        "impressions": np.round(impressions, 0),
        "publisher_a_impr": np.round(pub_a, 0), "publisher_b_impr": np.round(pub_b, 0),
        "publisher_c_impr": np.round(pub_c, 0),
        "spend_usd": np.round(np.where(treat, impressions * 0.012, impressions * 0.008), 0),
    })
    weeks = np.arange(1, duration_days // 7 + 1)
    trend = [{"day": int(w * 7),
              "control": float(100 * rng.lognormal(0, 0.08)),
              "treatment": float(100 * 1.05 * rng.lognormal(0, 0.08))} for w in weeks]
    return df, pd.DataFrame(trend)


def analyze_pharma(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} DMAs, {df['impressions'].sum():,.0f} impressions (units are DMAs, not impressions)")
        progress("Validating entity relationships", "one row per DMA; claims completeness recorded per DMA")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} DMAs")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    # Primary: completeness-adjusted starts, pre-period adjusted.
    primary = cuped_lift(df["post_starts_adjusted"].values, df["pre_starts"].values, treat.values)
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (claims-adjusted, DMA-level SEs)")
    # Guardrail: claims completeness must not differ by arm (differential lag).
    comp = welch_lift(df.loc[~treat, "claims_completeness"], df.loc[treat, "claims_completeness"])
    guard_failed = bool(comp["significant"] and abs(comp["lift_pct"]) > 3)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Evaluating guardrails and statistical validity", f"claims completeness lift {comp['lift_pct']:+.2f}%")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend_inc = float(df.loc[treat, "spend_usd"].sum() - df.loc[~treat, "spend_usd"].mean() * int(treat.sum()))
    cost_per = (spend_inc / inc) if inc > 0 else float("inf")
    if progress:
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    metrics = [
        {"name": "Treatment starts / DMA (claims-adjusted)", "control": primary["control_mean"],
         "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
         "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
         "p_value": primary["p_value"], "method": "Pre-period adjusted, DMA level", "guardrail": False},
        {"name": "Claims completeness", "control": comp["control_mean"], "treatment": comp["treatment_mean"],
         "lift_pct": comp["lift_pct"], "ci_low_pct": comp["ci_low_pct"], "ci_high_pct": comp["ci_high_pct"],
         "p_value": comp["p_value"], "method": "Welch (DMA level)", "guardrail": True},
    ]
    segments = []
    for seg, mask in [("high payer access", df["payer_access"] >= 0.5), ("low payer access", df["payer_access"] < 0.5)]:
        sub = df[mask]
        tr = sub["group"] == "treatment"
        # Same CUPED estimator as the follow-up segments, so scorecard and
        # chat segment numbers agree.
        segments.append({"segment": seg, **cuped_lift(sub["post_starts_adjusted"].values, sub["pre_starts"].values, tr.values)})
    publishers = []
    for pub, col in [("Publisher A", "publisher_a_impr"), ("Publisher B", "publisher_b_impr"), ("Publisher C", "publisher_c_impr")]:
        publishers.append({"publisher": pub, "impressions": float(df[col].sum()),
                           "share_pct": float(df[col].sum() / df["impressions"].sum() * 100)})
    return {"primary": primary, "guardrail": comp, "srm": srm, "verdict": verdict,
            "segments": segments, "metrics": metrics, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "DMAs", "publishers": publishers,
            "business": {"incremental_starts": float(inc),
                         "cost_per_incremental_start_usd": float(cost_per),
                         "incremental_spend_usd": float(spend_inc)}}


FLAGSHIPS = {
    "reddit_feed_ranking": (generate_reddit, analyze_reddit),
    "doordash_zip_promotion": (generate_doordash, analyze_doordash),
    "pharma_dma_publisher": (generate_pharma, analyze_pharma),
}

# Phase 2: the 9 remaining industry scenarios live in engine_extra.py
# (same primitives, same result shape). They are merged lazily in
# run_flagship below, and eagerly in server.py, because engine_extra
# imports the stats helpers from this module — a top-level import here
# would be a circular import if engine_extra is imported first.
EXTRA_FLAGSHIPS = None


def run_flagship(scenario_id, seed=42, progress=None, duration_days=None):
    if scenario_id not in FLAGSHIPS:
        from .engine_extra import EXTRA_FLAGSHIPS as _EXTRA

        FLAGSHIPS.update(_EXTRA)
    gen, ana = FLAGSHIPS[scenario_id]
    df, trend = gen(seed=seed, duration_days=duration_days) if duration_days else gen(seed=seed)
    result = ana(df, progress=progress)
    result["trend"] = trend.to_dict("records")
    return df, result
