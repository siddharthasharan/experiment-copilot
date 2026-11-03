"""Phase 2 experiment generators — the 9 remaining industry scenarios.

Same promises as engine.py, same primitives (welch_lift, cuped_lift,
srm_check, decide). Every number on a scorecard comes from a function
here running on generated data; no number is typed in by hand.

Randomization unit = analysis unit, always:
- Social + HCP: users / HCPs, CUPED with a pre-period covariate.
- Geo / market: DMAs, ZIPs, cities, territories — never impressions,
  users, or orders inside a geo treated as independent units.
- Geo randomization uses global matched pairs by baseline size (the
  DoorDash lesson: pairing within small markets left ~10% baseline
  imbalance at seed 42, so pairs are formed on the globally sorted
  baseline, with the market/region still recorded for probes).

Each analyze_* returns the same result shape as the flagships, plus
a "followup_spec" dict that followups.py uses for generic segment and
exclude-top recomputations, so the chat segment answers and the
scorecard segment tables use the same estimator on the same columns.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scipy import stats as _st

from .engine import cuped_lift, decide, srm_check, welch_lift


def _matched_pairs(rng, sort_values):
    """Global matched-pair randomization on a sorted baseline."""
    n = len(sort_values)
    treat = np.zeros(n, dtype=bool)
    order = np.argsort(sort_values)
    for j in range(0, n - 1, 2):
        if rng.random() < 0.5:
            treat[order[j]] = True
        else:
            treat[order[j + 1]] = True
    if n % 2 == 1:
        treat[order[-1]] = rng.random() < 0.5
    return treat


def _daily_trend(rng, duration_days, lift):
    days = np.arange(1, duration_days + 1)
    base = 1000 * (1 + 0.02 * np.sin(days / 7.0))
    return pd.DataFrame([{"day": int(d),
                          "control": float(base[d - 1] * rng.lognormal(0, 0.05)),
                          "treatment": float(base[d - 1] * (1 + lift) * rng.lognormal(0, 0.05))}
                         for d in days])


def _social_df(rng, n_users, effect_new, effect_est, guard_ctrl, guard_trt,
               pre_name, post_name, guard_name, decay=None):
    """Shared user-level generator. decay: late-period lift multiplier."""
    is_new = rng.random(n_users) < 0.30
    base = rng.lognormal(mean=1.2, sigma=0.7, size=n_users)
    base = np.where(is_new, base * 0.55, base)
    pre = base * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    treat = rng.random(n_users) < 0.5
    lift = np.where(is_new, effect_new, effect_est)
    if decay is None:
        post = base * (1 + np.where(treat, lift, 0)) * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    else:
        early = base * (1 + np.where(treat, lift, 0))
        late = base * (1 + np.where(treat, lift * decay, 0))
        post = 0.5 * (early + late) * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    guard = rng.poisson(np.where(treat, guard_trt, guard_ctrl), size=n_users)
    return pd.DataFrame({
        "user_id": np.arange(n_users), "group": np.where(treat, "treatment", "control"),
        "segment": np.where(is_new, "new", "established"),
        pre_name: np.round(pre, 2), post_name: np.round(post, 2),
        guard_name: guard,
    })


def _social_analyze(df, progress, pre_col, post_col, guard_col, guard_veto_pct,
                    n_label, unit_label="users"):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df):,} {n_label}, seed recorded in config")
        progress("Validating entity relationships", "user rows are unique; pre/post present for every user")
        progress("Assigning treatment and control", f"control {int((~treat).sum()):,} / treatment {int(treat.sum()):,}")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df[post_col].values, df[pre_col].values, treat.values)
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}%, CUPED variance reduction {primary['variance_reduction_pct']:.0f}%")
    guard = welch_lift(df.loc[~treat, guard_col], df.loc[treat, guard_col])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > guard_veto_pct)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Evaluating guardrails and statistical validity", f"guardrail lift {guard['lift_pct']:+.1f}% (veto only if significant and >{guard_veto_pct}%)")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub[post_col].values, sub[pre_col].values, tr.values)})
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": unit_label,
            "followup_spec": {"post_col": post_col, "pre_col": pre_col,
                              "segment_col": "segment",
                              "segment_labels": {"new": "new users", "established": "established users"}}}


# ------------------------------------------------------- Social: Meta
def generate_meta(seed=42, duration_days=28):
    """Meta creator recommendations: new ranking surfaces smaller creators
    to viewers. Unit: viewers (users). Primary: creator content sessions
    per user. Established viewers respond more than new viewers (cold
    start); guardrail: low-quality reports per user."""
    rng = np.random.default_rng(seed)
    df = _social_df(rng, 20000, effect_new=0.01, effect_est=0.04,
                    guard_ctrl=0.020, guard_trt=0.021,
                    pre_name="pre_sessions", post_name="post_sessions",
                    guard_name="reports")
    return df, _daily_trend(rng, duration_days, 0.035)


def analyze_meta(df, progress=None):
    out = _social_analyze(df, progress, "pre_sessions", "post_sessions", "reports", 10, "users")
    out["metrics"] = [
        {"name": "Creator content sessions / user", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "CUPED (pre-period sessions)", "guardrail": False},
        {"name": "Reports / user", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch", "guardrail": True},
    ]
    return out


# ----------------------------------------------------- Social: TikTok
def generate_tiktok(seed=42, duration_days=28):
    """TikTok personalized notifications: push timing/content personalized
    per user. Unit: users. Primary: sessions per user. Strong novelty
    decay by design: early lift ~7%, late lift ~1.5% of base — the trend
    chart, not the headline alone, shows whether the effect persists.
    Guardrail: notification opt-outs per user."""
    rng = np.random.default_rng(seed)
    df = _social_df(rng, 20000, effect_new=0.07, effect_est=0.07,
                    guard_ctrl=0.015, guard_trt=0.017,
                    pre_name="pre_sessions", post_name="post_sessions",
                    guard_name="opt_outs", decay=0.2)
    days = np.arange(1, duration_days + 1)
    base = 1000 * (1 + 0.02 * np.sin(days / 7.0))
    trend = pd.DataFrame([{"day": int(d),
                           "control": float(base[d - 1] * rng.lognormal(0, 0.05)),
                           "treatment": float(base[d - 1] * (1 + (0.07 if d <= 14 else 0.015)) * rng.lognormal(0, 0.05))}
                          for d in days])
    return df, trend


def analyze_tiktok(df, progress=None):
    out = _social_analyze(df, progress, "pre_sessions", "post_sessions", "opt_outs", 10, "users")
    out["metrics"] = [
        {"name": "Sessions / user", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "CUPED (pre-period sessions)", "guardrail": False},
        {"name": "Notification opt-outs / user", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch", "guardrail": True},
    ]
    return out


# --------------------------------------------------- Social: LinkedIn
def generate_linkedin(seed=42, duration_days=28):
    """LinkedIn connection recommendations: suggested connections ranked
    by shared context. Unit: members (users). Primary: accepted
    connections per user. New members benefit most (empty network);
    guardrail: ignored / spam-flagged invites per user."""
    rng = np.random.default_rng(seed)
    df = _social_df(rng, 20000, effect_new=0.08, effect_est=0.05,
                    guard_ctrl=0.030, guard_trt=0.031,
                    pre_name="pre_connections", post_name="post_connections",
                    guard_name="spam_flags")
    return df, _daily_trend(rng, duration_days, 0.06)


def analyze_linkedin(df, progress=None):
    out = _social_analyze(df, progress, "pre_connections", "post_connections", "spam_flags", 10, "users")
    out["metrics"] = [
        {"name": "Accepted connections / user", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "CUPED (pre-period connections)", "guardrail": False},
        {"name": "Spam-flagged invites / user", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch", "guardrail": True},
    ]
    return out


# ------------------------------------------- Marketplace: DMA advertising
def generate_dma_ads(seed=42, duration_days=28):
    """DMA-level digital advertising incrementality for a marketplace
    brand. Unit: DMAs (matched pairs by population). Treatment DMAs get
    increased digital ad spend; primary: orders per DMA, pre-period
    adjusted. Guardrail: refund rate. Spend is recorded per DMA so the
    cost per incremental order is computed, not assumed."""
    rng = np.random.default_rng(seed)
    n = 200
    pop = rng.lognormal(mean=10.5, sigma=0.4, size=n)
    treat = _matched_pairs(rng, pop)
    baseline = pop * 0.02 * rng.lognormal(0, 0.10, size=n)
    pre = baseline * rng.lognormal(0, 0.08, size=n)
    post = baseline * (1 + np.where(treat, 0.06, 0)) * rng.lognormal(0, 0.10, size=n)
    spend = np.where(treat, pop * 0.010, pop * 0.006)
    refund = np.clip(np.where(treat, 0.045, 0.044) + rng.normal(0, 0.008, size=n), 0, 1)
    seg = np.where(baseline >= np.median(baseline), "large", "small")
    df = pd.DataFrame({"dma": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "segment": seg, "baseline_orders": np.round(baseline, 1),
                       "pre_orders": np.round(pre, 1), "post_orders": np.round(post, 1),
                       "spend_usd": np.round(spend, 0), "refund_rate": np.round(refund, 4)})
    return df, _daily_trend(rng, duration_days, 0.06)


def analyze_dma_ads(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} DMAs, seed recorded in config")
        progress("Validating entity relationships", "one row per DMA; pre/post present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} DMAs")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_orders"].values, df["pre_orders"].values, treat.values)
    guard = welch_lift(df.loc[~treat, "refund_rate"], df.loc[treat, "refund_rate"])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > 10)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (pre-period adjusted, DMA-level SEs)")
        progress("Evaluating guardrails and statistical validity", f"refund rate lift {guard['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend_inc = float(df.loc[treat, "spend_usd"].sum() - df.loc[~treat, "spend_usd"].mean() * int(treat.sum()))
    cost_per = (spend_inc / inc) if inc > 0 else float("inf")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_orders"].values, sub["pre_orders"].values, tr.values)})
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "DMAs",
            "metrics": [
                {"name": "Orders / DMA", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
                 "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "Pre-period adjusted, DMA level", "guardrail": False},
                {"name": "Refund rate", "control": guard["control_mean"], "treatment": guard["treatment_mean"],
                 "lift_pct": guard["lift_pct"], "ci_low_pct": guard["ci_low_pct"], "ci_high_pct": guard["ci_high_pct"],
                 "p_value": guard["p_value"], "method": "Welch (DMA level)", "guardrail": True}],
            "business": {"incremental_orders": float(inc), "incremental_spend_usd": float(spend_inc),
                         "cost_per_incremental_order_usd": float(cost_per),
                         "impact_text": (f"Incremental orders: {inc:,.0f}. Incremental ad spend: "
                                         f"${spend_inc:,.0f}. Cost per incremental order: ${cost_per:,.2f}. "
                                         f"Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_orders", "pre_col": "pre_orders",
                              "base_col": "baseline_orders", "unit_col": "dma",
                              "segment_col": "segment",
                              "segment_labels": {"large": "large DMAs", "small": "small DMAs"}}}


# --------------------------------------------- Marketplace: Uber incentives
def generate_uber(seed=42, duration_days=28):
    """Uber driver incentives: bonus for completing a trip target in
    treated cities. Unit: cities (n=60, matched pairs by baseline
    trips) — driver-level analysis would overstate precision because
    drivers in one city share the same market conditions. Primary:
    completed trips per city. Guardrail: cancellation rate. Incentive
    spend is recorded so cost per incremental trip is computed."""
    rng = np.random.default_rng(seed)
    n = 60
    baseline = rng.lognormal(mean=9.0, sigma=0.5, size=n)
    treat = _matched_pairs(rng, baseline)
    pre = baseline * rng.lognormal(0, 0.08, size=n)
    post = baseline * (1 + np.where(treat, 0.12, 0)) * rng.lognormal(0, 0.10, size=n)
    driver_hours = post * 0.45 * rng.lognormal(0, 0.05, size=n)
    utilization = post / np.maximum(driver_hours, 1)
    cancel = np.clip(np.where(treat, 0.052, 0.050) + rng.normal(0, 0.008, size=n), 0, 1)
    seg = np.where(baseline >= np.median(baseline), "large", "small")
    df = pd.DataFrame({"city": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "segment": seg, "baseline_trips": np.round(baseline, 1),
                       "pre_trips": np.round(pre, 1), "post_trips": np.round(post, 1),
                       "driver_hours": np.round(driver_hours, 1),
                       "utilization": np.round(utilization, 3),
                       "cancel_rate": np.round(cancel, 4),
                       "incentive_spend_usd": np.round(np.where(treat, driver_hours * 3.0, 0), 0)})
    return df, _daily_trend(rng, duration_days, 0.12)


def analyze_uber(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} cities, seed recorded in config")
        progress("Validating entity relationships", "one row per city; pre/post present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} cities")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_trips"].values, df["pre_trips"].values, treat.values)
    util = welch_lift(df.loc[~treat, "utilization"], df.loc[treat, "utilization"])
    canc = welch_lift(df.loc[~treat, "cancel_rate"], df.loc[treat, "cancel_rate"])
    guard_failed = bool(canc["significant"] and canc["lift_pct"] > 20)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (pre-period adjusted, city-level SEs)")
        progress("Evaluating guardrails and statistical validity", f"cancellations {canc['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend = float(df.loc[treat, "incentive_spend_usd"].sum())
    cost_per = (spend / inc) if inc > 0 else float("inf")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_trips"].values, sub["pre_trips"].values, tr.values)})
    return {"primary": primary, "guardrail": canc, "utilization": util, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "Cities",
            "metrics": [
                {"name": "Completed trips / city", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
                 "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "Pre-period adjusted, city level", "guardrail": False},
                {"name": "Driver utilization (trips / driver-hour)", "control": util["control_mean"], "treatment": util["treatment_mean"],
                 "lift_pct": util["lift_pct"], "ci_low_pct": util["ci_low_pct"], "ci_high_pct": util["ci_high_pct"],
                 "p_value": util["p_value"], "method": "Welch (city level)", "guardrail": False},
                {"name": "Cancellation rate", "control": canc["control_mean"], "treatment": canc["treatment_mean"],
                 "lift_pct": canc["lift_pct"], "ci_low_pct": canc["ci_low_pct"], "ci_high_pct": canc["ci_high_pct"],
                 "p_value": canc["p_value"], "method": "Welch (city level)", "guardrail": True}],
            "business": {"incremental_trips": float(inc), "incentive_spend_usd": spend,
                         "cost_per_incremental_trip_usd": float(cost_per),
                         "impact_text": (f"Incremental completed trips: {inc:,.0f}. Incentive spend: "
                                         f"${spend:,.0f}. Cost per incremental trip: ${cost_per:,.2f}. "
                                         f"Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_trips", "pre_col": "pre_trips",
                              "base_col": "baseline_trips", "unit_col": "city",
                              "segment_col": "segment",
                              "segment_labels": {"large": "large cities", "small": "small cities"}}}


# --------------------------------------- Marketplace: delivery-fee test
def generate_fee(seed=42, duration_days=28):
    """Delivery-fee optimization: treated ZIPs get a $2 lower delivery
    fee. Unit: ZIPs (global matched pairs by baseline orders, as in
    DoorDash). Primary: orders per ZIP. The economics question is the
    point: orders rise, but each treated order gives up $2 of fee, so
    contribution can fall even as orders rise — computed explicitly."""
    rng = np.random.default_rng(seed)
    n = 200
    baseline = rng.lognormal(mean=6.0, sigma=0.55, size=n)
    treat = _matched_pairs(rng, baseline)
    pre = baseline * rng.lognormal(0, 0.08, size=n)
    post = baseline * (1 + np.where(treat, 0.09, 0)) * rng.lognormal(0, 0.08, size=n)
    new_share = rng.beta(20, 30, size=n)
    delivery_min = np.where(treat, 26.8, 26.5) + rng.normal(0, 2, size=n)
    cancel_rate = np.where(treat, 0.033, 0.032) + rng.normal(0, 0.005, size=n)
    df = pd.DataFrame({"zip": np.arange(10000, 10000 + n), "group": np.where(treat, "treatment", "control"),
                       "baseline_orders": np.round(baseline, 1), "pre_orders": np.round(pre, 1),
                       "post_orders": np.round(post, 1), "orders_new": np.round(post * new_share, 1),
                       "orders_returning": np.round(post * (1 - new_share), 1),
                       "delivery_min": np.round(delivery_min, 1),
                       "cancel_rate": np.round(np.clip(cancel_rate, 0, 1), 4)})
    return df, _daily_trend(rng, duration_days, 0.09)


def analyze_fee(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} ZIPs, seed recorded in config")
        progress("Validating entity relationships", "one row per ZIP; pre/post present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} ZIPs")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_orders"].values, df["pre_orders"].values, treat.values)
    deliv = welch_lift(df.loc[~treat, "delivery_min"], df.loc[treat, "delivery_min"])
    canc = welch_lift(df.loc[~treat, "cancel_rate"], df.loc[treat, "cancel_rate"])
    guard_failed = bool((deliv["significant"] and deliv["lift_pct"] > 10) or
                        (canc["significant"] and canc["lift_pct"] > 20))
    verdict = decide(primary, guard_failed, srm["failed"])
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    treated_orders = float(df.loc[treat, "post_orders"].sum())
    contribution = inc * 6.0 - treated_orders * 2.0
    cost_per = (treated_orders * 2.0 / inc) if inc > 0 else float("inf")
    business_reason = None
    if primary["significant"] and primary["lift_pct"] > 0 and contribution < 0:
        verdict = "DONT_LAUNCH"
        business_reason = ("Contribution is negative as designed: the $2 fee cut is given up on "
                           "every treated order, so it outweighs the contribution from incremental "
                           "orders. Orders rose, but the fee cut as designed does not pay for itself.")
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (pre-period adjusted, ZIP-level SEs)")
        progress("Evaluating guardrails and statistical validity", f"delivery time {deliv['lift_pct']:+.1f}%, cancellations {canc['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, col in [("new customers", "orders_new"), ("returning customers", "orders_returning")]:
        share = (df[col] / df["post_orders"]).fillna(0)
        pre_seg = df["pre_orders"] * share
        segments.append({"segment": seg, **cuped_lift(df[col].values, pre_seg.values, treat.values)})
    return {"primary": primary, "guardrail": deliv, "cancellation": canc, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "ZIPs", "business_reason": business_reason,
            "metrics": [
                {"name": "Orders / ZIP", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
                 "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "Pre-period adjusted (CUPED-style, ZIP level)", "guardrail": False},
                {"name": "Delivery time (min)", "control": deliv["control_mean"], "treatment": deliv["treatment_mean"],
                 "lift_pct": deliv["lift_pct"], "ci_low_pct": deliv["ci_low_pct"], "ci_high_pct": deliv["ci_high_pct"],
                 "p_value": deliv["p_value"], "method": "Welch (ZIP level)", "guardrail": True},
                {"name": "Cancellation rate", "control": canc["control_mean"], "treatment": canc["treatment_mean"],
                 "lift_pct": canc["lift_pct"], "ci_low_pct": canc["ci_low_pct"], "ci_high_pct": canc["ci_high_pct"],
                 "p_value": canc["p_value"], "method": "Welch (ZIP level)", "guardrail": True}],
            "business": {"incremental_orders": float(inc), "contribution_usd": float(contribution),
                         "cost_per_incremental_order_usd": float(cost_per),
                         "treated_orders": float(treated_orders)},
            "followup_spec": {"post_col": "post_orders", "pre_col": "pre_orders",
                              "base_col": "baseline_orders", "unit_col": "zip",
                              "segment_col": None}}


# ------------------------------------------------- Pharma: HCP omnichannel
def generate_hcp(seed=42, duration_days=28):
    """HCP omnichannel engagement: treated HCPs get coordinated email +
    portal + rep outreach vs email-only. Unit: HCPs (individual
    professionals), CUPED with pre-period engagement. Primary: engaged
    interactions per HCP. Less-engaged HCPs respond most. Guardrail:
    opt-out rate. Outreach cost is recorded per treated HCP."""
    rng = np.random.default_rng(seed)
    n = 8000
    prior = rng.lognormal(mean=1.0, sigma=0.6, size=n)
    low_prior = prior < np.median(prior)
    pre = prior * rng.lognormal(mean=0, sigma=0.3, size=n)
    treat = rng.random(n) < 0.5
    lift = np.where(low_prior, 0.09, 0.03)
    post = prior * (1 + np.where(treat, lift, 0)) * rng.lognormal(mean=0, sigma=0.3, size=n)
    opt_out = rng.poisson(np.where(treat, 0.012, 0.011), size=n)
    df = pd.DataFrame({"hcp_id": np.arange(n), "group": np.where(treat, "treatment", "control"),
                       "segment": np.where(low_prior, "low", "high"),
                       "pre_interactions": np.round(pre, 2), "post_interactions": np.round(post, 2),
                       "opt_outs": opt_out})
    return df, _daily_trend(rng, duration_days, 0.06)


def analyze_hcp(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df):,} HCPs, seed recorded in config")
        progress("Validating entity relationships", "one row per HCP; pre/post present")
        progress("Assigning treatment and control", f"control {int((~treat).sum()):,} / treatment {int(treat.sum()):,}")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_interactions"].values, df["pre_interactions"].values, treat.values)
    guard = welch_lift(df.loc[~treat, "opt_outs"], df.loc[treat, "opt_outs"])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > 10)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}%, CUPED variance reduction {primary['variance_reduction_pct']:.0f}%")
        progress("Evaluating guardrails and statistical validity", f"opt-out lift {guard['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend = float(int(treat.sum()) * 45.0)
    cost_per = (spend / inc) if inc > 0 else float("inf")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_interactions"].values, sub["pre_interactions"].values, tr.values)})
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "HCPs",
            "metrics": [
                {"name": "Engaged interactions / HCP", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
                 "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED (pre-period engagement)", "guardrail": False},
                {"name": "Opt-outs / HCP", "control": guard["control_mean"], "treatment": guard["treatment_mean"],
                 "lift_pct": guard["lift_pct"], "ci_low_pct": guard["ci_low_pct"], "ci_high_pct": guard["ci_high_pct"],
                 "p_value": guard["p_value"], "method": "Welch", "guardrail": True}],
            "business": {"incremental_interactions": float(inc), "outreach_spend_usd": spend,
                         "cost_per_incremental_interaction_usd": float(cost_per),
                         "impact_text": (f"Incremental engaged interactions: {inc:,.0f}. Outreach spend: "
                                         f"${spend:,.0f} ($45 per treated HCP). Cost per incremental interaction: "
                                         f"${cost_per:,.2f}. Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_interactions", "pre_col": "pre_interactions",
                              "segment_col": "segment",
                              "segment_labels": {"low": "low prior engagement HCPs", "high": "high prior engagement HCPs"}}}


# ------------------------------------------- Pharma: field visits + sampling
def generate_field(seed=42, duration_days=56):
    """Field visits and sampling effectiveness: treated territories get
    an added rep-visit + sampling cycle. Unit: territories (n=120,
    matched pairs by baseline prescriptions). Primary: new prescriptions
    per territory, pre-period adjusted. Guardrail: HCP opt-out rate.
    Visit cost is recorded per treated territory."""
    rng = np.random.default_rng(seed)
    n = 120
    baseline = rng.lognormal(mean=7.0, sigma=0.5, size=n)
    treat = _matched_pairs(rng, baseline)
    pre = baseline * rng.lognormal(0, 0.08, size=n)
    post = baseline * (1 + np.where(treat, 0.07, 0)) * rng.lognormal(0, 0.10, size=n)
    opt_out = np.clip(np.where(treat, 0.021, 0.020) + rng.normal(0, 0.005, size=n), 0, 1)
    seg = np.where(baseline >= np.median(baseline), "large", "small")
    df = pd.DataFrame({"territory": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "segment": seg, "baseline_rx": np.round(baseline, 1),
                       "pre_rx": np.round(pre, 1), "post_rx": np.round(post, 1),
                       "opt_out_rate": np.round(opt_out, 4)})
    weeks = np.arange(1, duration_days // 7 + 1)
    trend = pd.DataFrame([{"day": int(w * 7), "control": float(100 * rng.lognormal(0, 0.08)),
                           "treatment": float(100 * 1.07 * rng.lognormal(0, 0.08))} for w in weeks])
    return df, trend


def analyze_field(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} territories, seed recorded in config")
        progress("Validating entity relationships", "one row per territory; pre/post present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} territories")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_rx"].values, df["pre_rx"].values, treat.values)
    guard = welch_lift(df.loc[~treat, "opt_out_rate"], df.loc[treat, "opt_out_rate"])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > 10)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (pre-period adjusted, territory-level SEs)")
        progress("Evaluating guardrails and statistical validity", f"opt-out rate lift {guard['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend = float(int(treat.sum()) * 1800.0)
    cost_per = (spend / inc) if inc > 0 else float("inf")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_rx"].values, sub["pre_rx"].values, tr.values)})
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "Territories",
            "metrics": [
                {"name": "New prescriptions / territory", "control": primary["control_mean"], "treatment": primary["treatment_mean"],
                 "lift_pct": primary["lift_pct"], "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "Pre-period adjusted, territory level", "guardrail": False},
                {"name": "HCP opt-out rate", "control": guard["control_mean"], "treatment": guard["treatment_mean"],
                 "lift_pct": guard["lift_pct"], "ci_low_pct": guard["ci_low_pct"], "ci_high_pct": guard["ci_high_pct"],
                 "p_value": guard["p_value"], "method": "Welch (territory level)", "guardrail": True}],
            "business": {"incremental_rx": float(inc), "visit_spend_usd": spend,
                         "cost_per_incremental_rx_usd": float(cost_per),
                         "impact_text": (f"Incremental new prescriptions: {inc:,.0f}. Visit + sampling spend: "
                                         f"${spend:,.0f} ($1,800 per treated territory). Cost per incremental prescription: "
                                         f"${cost_per:,.2f}. Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_rx", "pre_col": "pre_rx",
                              "base_col": "baseline_rx", "unit_col": "territory",
                              "segment_col": "segment",
                              "segment_labels": {"large": "large territories", "small": "small territories"}}}


# -------------------------------------- Pharma: cross-publisher budget
def generate_cross_publisher(seed=42, duration_days=56):
    """Cross-publisher marketing budget allocation: treated DMAs shift
    20% of budget from Publisher A to a B/C mix, at the same total
    spend. Unit: DMAs (matched pairs by population). Primary: treatment
    starts per DMA, claims-adjusted, as in the Pharma flagship — but
    the intervention is a reallocation, not a spend increase, so the
    economics question is efficiency, not scale. Publisher shares are
    descriptive only: a geo allocation experiment cannot identify
    individual publisher causal effects."""
    rng = np.random.default_rng(seed)
    n = 200
    pop = rng.lognormal(mean=10.5, sigma=0.4, size=n)
    payer = rng.beta(3, 3, size=n)
    treat = _matched_pairs(rng, pop)
    pre_starts = pop * 0.0012 * (0.7 + 0.6 * payer) * rng.lognormal(0, 0.10, size=n)
    effect = 0.04 * (1.3 - payer)
    post_true = pop * 0.0012 * (0.7 + 0.6 * payer) * (1 + np.where(treat, effect, 0))
    post_true = post_true * rng.lognormal(0, 0.10, size=n)
    completeness = np.clip(rng.beta(9, 1.5, size=n), 0.75, 0.99)
    post_obs = post_true * completeness
    impressions = pop * rng.lognormal(mean=2.0, sigma=0.4, size=n)
    _wa = rng.beta(5, 3, size=n); _wb = rng.beta(3, 3, size=n); _wc = rng.beta(2, 3, size=n)
    _tot = _wa + _wb + _wc
    spend = impressions * 0.010
    df = pd.DataFrame({"dma": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "payer_access": np.round(payer, 3), "pre_starts": np.round(pre_starts, 1),
                       "post_starts_true": np.round(post_true, 1),
                       "claims_completeness": np.round(completeness, 3),
                       "post_starts_observed": np.round(post_obs, 1),
                       "post_starts_adjusted": np.round(post_obs / completeness, 1),
                       "impressions": np.round(impressions, 0),
                       "publisher_a_impr": np.round(impressions * _wa / _tot, 0),
                       "publisher_b_impr": np.round(impressions * _wb / _tot, 0),
                       "publisher_c_impr": np.round(impressions * _wc / _tot, 0),
                       "spend_usd": np.round(spend, 0)})
    weeks = np.arange(1, duration_days // 7 + 1)
    trend = pd.DataFrame([{"day": int(w * 7), "control": float(100 * rng.lognormal(0, 0.08)),
                           "treatment": float(100 * 1.04 * rng.lognormal(0, 0.08))} for w in weeks])
    return df, trend


def analyze_cross_publisher(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df)} DMAs, {df['impressions'].sum():,.0f} impressions (units are DMAs, not impressions)")
        progress("Validating entity relationships", "one row per DMA; claims completeness recorded per DMA")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} DMAs")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_starts_adjusted"].values, df["pre_starts"].values, treat.values)
    comp = welch_lift(df.loc[~treat, "claims_completeness"], df.loc[treat, "claims_completeness"])
    guard_failed = bool(comp["significant"] and abs(comp["lift_pct"]) > 3)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (claims-adjusted, DMA-level SEs)")
        progress("Evaluating guardrails and statistical validity", f"claims completeness lift {comp['lift_pct']:+.2f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    inc = (primary["treatment_mean"] - primary["control_mean"]) * int(treat.sum())
    spend_inc = float(df.loc[treat, "spend_usd"].sum() - df.loc[~treat, "spend_usd"].mean() * int(treat.sum()))
    cost_per = (spend_inc / inc) if inc > 0 else float("inf")
    segments = []
    for seg, mask in [("high payer access", df["payer_access"] >= 0.5), ("low payer access", df["payer_access"] < 0.5)]:
        sub = df[mask]
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_starts_adjusted"].values, sub["pre_starts"].values, tr.values)})
    publishers = []
    for pub, col in [("Publisher A", "publisher_a_impr"), ("Publisher B", "publisher_b_impr"), ("Publisher C", "publisher_c_impr")]:
        publishers.append({"publisher": pub, "impressions": float(df[col].sum()),
                           "share_pct": float(df[col].sum() / df["impressions"].sum() * 100)})
    return {"primary": primary, "guardrail": comp, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "DMAs", "publishers": publishers,
            "metrics": [
                {"name": "Treatment starts / DMA (claims-adjusted)", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "Pre-period adjusted, DMA level", "guardrail": False},
                {"name": "Claims completeness", "control": comp["control_mean"], "treatment": comp["treatment_mean"],
                 "lift_pct": comp["lift_pct"], "ci_low_pct": comp["ci_low_pct"], "ci_high_pct": comp["ci_high_pct"],
                 "p_value": comp["p_value"], "method": "Welch (DMA level)", "guardrail": True}],
            "business": {"incremental_starts": float(inc), "incremental_spend_usd": float(spend_inc),
                         "cost_per_incremental_start_usd": float(cost_per),
                         "impact_text": (f"Incremental treatment starts: {inc:,.0f}. Spend change vs control-mean "
                                         f"baseline: ${spend_inc:,.0f} (a reallocation, not a spend increase — "
                                         f"so cost per incremental start is ${cost_per:,.0f} on that small spend "
                                         f"difference, and the decision rests on the starts lift itself, not on spend scale). "
                                         f"Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_starts_adjusted", "pre_col": "pre_starts",
                              "base_col": "pre_starts", "unit_col": "dma",
                              "segment_col": None}}


# --------------------------------------- LinkedIn: creator crowding-out
def generate_crowding_out(seed=42, duration_days=28):
    """LinkedIn creator crowding-out: does a ranking change that is
    meant to surface creators actually help small creators, or do top
    creators absorb the gain?

    Honesty note, stated in the code because it decides what this
    experiment can and cannot claim: randomization is at VIEWER level
    (users), not creator level. The data are per-viewer tier aggregates
    — the share of a viewer's creator sessions going to each follower
    tier — not individual creators. This design therefore answers the
    EXPOSURE question (what viewers see: does the small tier's share
    rise or fall?) and cannot answer a creator-outcome question (what
    small creators earn, or whether a specific creator was harmed) —
    that would need creator-level randomization. Tiers: top = 100,000+
    followers, mid = 10,000-100,000, small = 1,000-10,000.

    Ground truth at seed 42 (deliberate, and stated, not hidden): the
    ranking lifts total sessions (+5.7%), but the gain goes to the top
    tier — small share falls 16.1% to 14.8% (-1.3pp) while top share
    rises +1.6pp. That IS the crowding-out finding: total engagement
    up, small creators crowded out. A different seed can soften or
    reverse it; the scorecard reports the computed result either way.
    """
    rng = np.random.default_rng(seed)
    n_users = 20000
    is_new = rng.random(n_users) < 0.30
    base = rng.lognormal(mean=1.2, sigma=0.7, size=n_users)
    base = np.where(is_new, base * 0.55, base)
    pre_total = base * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    treat = rng.random(n_users) < 0.5
    # Pre-period tier shares per viewer (Dirichlet: top-heavy feed).
    shares = rng.dirichlet([5.2, 3.0, 1.6], size=n_users)  # top, mid, small
    pre_top = shares[:, 0] * 100
    pre_small = shares[:, 2] * 100
    # Post-period shares: control drifts slightly; treated shifts toward
    # the top tier and away from the small tier (crowding-out).
    post_top = pre_top + np.where(treat, 2.5, 0) + rng.normal(0, 1.0, size=n_users)
    post_small = pre_small + np.where(treat, -1.8, 0) + rng.normal(0, 1.0, size=n_users)
    post_top = np.clip(post_top, 30, 70)
    post_small = np.clip(post_small, 5, 35)
    post_mid = np.clip(100 - post_top - post_small, 1, 60)
    tot = post_top + post_mid + post_small
    post_top, post_mid, post_small = post_top / tot * 100, post_mid / tot * 100, post_small / tot * 100
    post_total = base * (1 + np.where(treat, 0.04, 0)) * rng.lognormal(mean=0, sigma=0.35, size=n_users)
    reports = rng.poisson(np.where(treat, 0.021, 0.020), size=n_users)
    df = pd.DataFrame({
        "user_id": np.arange(n_users), "group": np.where(treat, "treatment", "control"),
        "segment": np.where(is_new, "new", "established"),
        "pre_sessions": np.round(pre_total, 2), "post_sessions": np.round(post_total, 2),
        "pre_top_share_pct": np.round(pre_top, 2), "post_top_share_pct": np.round(post_top, 2),
        "pre_small_share_pct": np.round(pre_small, 2), "post_small_share_pct": np.round(post_small, 2),
        "post_mid_share_pct": np.round(post_mid, 2),
        "small_sessions": np.round(post_total * post_small / 100, 2),
        "top_sessions": np.round(post_total * post_top / 100, 2),
        "reports": reports,
    })
    # Trend is the small-tier share trajectory — the crowding-out story
    # itself — not total sessions (which rise in both arms' story).
    days = np.arange(1, duration_days + 1)
    trend = pd.DataFrame([{"day": int(d),
                           "control": float(16.0 * rng.lognormal(0, 0.03)),
                           "treatment": float(14.2 * rng.lognormal(0, 0.03))} for d in days])
    return df, trend


def analyze_crowding_out(df, progress=None):
    t0 = time.time()
    treat = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{len(df):,} viewers, seed recorded in config")
        progress("Validating entity relationships", "viewer rows are unique; pre/post tier shares present for every viewer")
        progress("Assigning treatment and control", f"control {int((~treat).sum()):,} / treatment {int(treat.sum()):,}")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    # Primary: small-creator share (percentage points in the data, so the
    # control/treatment means read as % and the lift is relative).
    primary = cuped_lift(df["post_small_share_pct"].values, df["pre_small_share_pct"].values, treat.values)
    total = welch_lift(df.loc[~treat, "post_sessions"], df.loc[treat, "post_sessions"])
    top = welch_lift(df.loc[~treat, "post_top_share_pct"], df.loc[treat, "post_top_share_pct"])
    guard = welch_lift(df.loc[~treat, "reports"], df.loc[treat, "reports"])
    guard_failed = bool(guard["significant"] and guard["lift_pct"] > 10)
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"small-creator share lift {primary['lift_pct']:+.2f}%, CUPED variance reduction {primary['variance_reduction_pct']:.0f}%")
        progress("Evaluating guardrails and statistical validity", f"reports lift {guard['lift_pct']:+.1f}% (veto only if significant and >10%)")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    small_pp = primary["treatment_mean"] - primary["control_mean"]
    top_pp = top["treatment_mean"] - top["control_mean"]
    if primary["significant"] and primary["lift_pct"] < 0:
        business_reason = ("Crowding-out, as the question asked: total creator sessions rose "
                           f"{total['lift_pct']:+.1f}%, but the small tier's share fell {small_pp:+.1f}pp "
                           f"({primary['control_mean']:.1f}% to {primary['treatment_mean']:.1f}%) while the top "
                           f"tier's share rose {top_pp:+.1f}pp. The engagement gain went to top creators, "
                           "not to the small creators this ranking was meant to surface — so do not launch "
                           "it as a small-creator intervention.")
        impact = (f"Small-creator share fell {small_pp:+.1f}pp ({primary['control_mean']:.1f}% to "
                  f"{primary['treatment_mean']:.1f}%), top-tier share rose {top_pp:+.1f}pp, and total "
                  f"creator sessions rose {total['lift_pct']:+.1f}%. Those figures come from the generated "
                  f"data, not from the headline lift alone — a rising total with a falling small share "
                  f"is exactly what crowding-out looks like.")
    elif primary["significant"] and primary["lift_pct"] > 0:
        business_reason = ("No crowding-out in this run: the small tier's share rose "
                           f"{small_pp:+.1f}pp while total sessions rose {total['lift_pct']:+.1f}%. "
                           "The ranking surfaced small creators as intended.")
        impact = (f"Small-creator share rose {small_pp:+.1f}pp ({primary['control_mean']:.1f}% to "
                  f"{primary['treatment_mean']:.1f}%), so small creators gained exposure, not just total sessions.")
    else:
        business_reason = ("No clear crowding-out verdict in this run: the small tier's share did not "
                           "move significantly. Do not claim crowding-out — or surfacing — from this run.")
        impact = (f"Small-creator share moved {small_pp:+.1f}pp ({primary['control_mean']:.1f}% to "
                  f"{primary['treatment_mean']:.1f}%), not significant — neither crowding-out nor surfacing is established.")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = sub["group"] == "treatment"
        segments.append({"segment": seg, **cuped_lift(sub["post_small_share_pct"].values, sub["pre_small_share_pct"].values, tr.values)})
    return {"primary": primary, "guardrail": guard, "total": total, "top_share": top, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "users", "business_reason": business_reason,
            "metrics": [
                {"name": "Small-creator share of creator sessions (%)", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED (pre-period small share)", "guardrail": False},
                {"name": "Top-creator share of creator sessions (%)", "control": top["control_mean"],
                 "treatment": top["treatment_mean"], "lift_pct": top["lift_pct"],
                 "ci_low_pct": top["ci_low_pct"], "ci_high_pct": top["ci_high_pct"],
                 "p_value": top["p_value"], "method": "Welch", "guardrail": False},
                {"name": "Creator sessions / user", "control": total["control_mean"],
                 "treatment": total["treatment_mean"], "lift_pct": total["lift_pct"],
                 "ci_low_pct": total["ci_low_pct"], "ci_high_pct": total["ci_high_pct"],
                 "p_value": total["p_value"], "method": "Welch", "guardrail": False},
                {"name": "Reports / user", "control": guard["control_mean"],
                 "treatment": guard["treatment_mean"], "lift_pct": guard["lift_pct"],
                 "ci_low_pct": guard["ci_low_pct"], "ci_high_pct": guard["ci_high_pct"],
                 "p_value": guard["p_value"], "method": "Welch", "guardrail": True}],
            "business": {"impact_text": impact},
            "followup_spec": {"post_col": "post_small_share_pct", "pre_col": "pre_small_share_pct",
                              "segment_col": "segment",
                              "segment_labels": {"new": "new users", "established": "established users"}}}


# ------------------------------------------------- Switchback experiments
# Switchback design (DoorDash approach, stated because it decides what
# the analysis may claim):
# - Randomization unit = region x time window. Each region-window is
#   assigned independently (coin flip per region-window), so a region
#   switches between arms over the run. This is the design for cases
#   where user-level randomization is invalid because units share a
#   fleet, a market, or a feed: the shared resource is inside the
#   region-window, not across arms.
# - Analysis unit = the same region-window. The primary estimate is
#   the simple (unweighted) average of unit values by arm — NOT a
#   volume-weighted average of individual deliveries/trips. Weighting
#   by volume would let a few huge windows dominate and would answer
#   a different question (the average delivery's experience, not the
#   average region-window's). The stored frame keeps one row per
#   region-window so that choice is checkable, not asserted.
# - Standard errors are cluster-robust by region (sandwich/CR0 on the
#   OLS fit y ~ treat, t with regions-1 degrees of freedom). Windows
#   in the same region are correlated (same fleet, same demand regime),
#   so treating windows as independent would overstate precision.
# - Carryover / washout: the first window after a switch is excluded
#   from the primary analysis, because the system has not settled into
#   the new arm yet (couriers in transit, prices mid-adjustment). That
#   costs roughly half the windows under independent per-window
#   randomization — the price of carryover protection — and it is
#   reported, not hidden. A sensitivity including washout windows is
#   computed alongside the primary, so a reader can see whether the
#   washout choice drives the conclusion. The washout rule assumes
#   carryover lasts at most one window; if it lasts longer, the
#   primary is still contaminated, and the sensitivity does not fix
#   that — it only shows the washout's own effect.
# - SRM is checked on the assignment split of ALL randomized windows
#   (washout windows included): washout is an analysis exclusion, not
#   an assignment failure, so excluding it from the SRM check would
#   hide a randomization problem.
def cluster_robust_lift(values, treat, clusters):
    """Difference in unit means with cluster-robust (CR0) SEs.

    Returns the same headline keys as welch_lift (control_mean,
    treatment_mean, lift_pct, ci_low_pct, ci_high_pct, p_value,
    significant, n_control, n_treatment) plus se and n_clusters, so
    scorecard and follow-up code can use it interchangeably.
    """
    import numpy as np

    y = np.asarray(values, dtype=float)
    t = np.asarray(treat, dtype=float)
    c = np.asarray(clusters)
    n = len(y)
    n_t, n_c = int(t.sum()), int(n - t.sum())
    cm = float(y[t == 0].mean()) if n_c else 0.0
    tm = float(y[t == 1].mean()) if n_t else 0.0
    if n_t < 2 or n_c < 2:
        return {"control_mean": cm, "treatment_mean": tm, "lift_pct": 0.0,
                "ci_low_pct": 0.0, "ci_high_pct": 0.0, "p_value": 1.0,
                "significant": False, "n_control": n_c, "n_treatment": n_t,
                "se": 0.0, "n_clusters": int(len(np.unique(c)))}
    x = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    meat = np.zeros((2, 2))
    for g in np.unique(c):
        m = c == g
        s = x[m].T @ resid[m]
        meat += np.outer(s, s)
    var = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(max(var[1, 1], 0.0)))
    dfree = max(int(len(np.unique(c))) - 1, 1)
    diff = tm - cm
    crit = float(_st.t.ppf(0.975, dfree))
    p = float(2 * _st.t.sf(abs(diff / se), dfree)) if se > 0 else 1.0
    return {"control_mean": cm, "treatment_mean": tm,
            "lift_pct": float(diff / cm * 100) if cm != 0 else 0.0,
            "ci_low_pct": float((diff - crit * se) / cm * 100) if cm != 0 else 0.0,
            "ci_high_pct": float((diff + crit * se) / cm * 100) if cm != 0 else 0.0,
            "p_value": p, "significant": bool(p < 0.05),
            "n_control": n_c, "n_treatment": n_t, "se": se,
            "n_clusters": int(len(np.unique(c)))}


def _switchback_frame(rng, n_regions, region_col, base_volume, region_effect_sd,
                      base_value, effect, noise_sd, duration_days, windows_per_day=4):
    """Shared region x window frame. One row per region-window.

    base_volume: requests/deliveries scale per region (lognormal draws
    are made by the caller and passed in, so each scenario's scale is
    explicit). base_value: per-region baseline of the primary metric
    (a rate or a mean, never a raw count, so unit means are comparable
    across regions of very different size). effect: additive treatment
    effect on the primary metric. Weekend windows shift by -0.02 in
    relative terms (demand regime, not treatment).
    """
    n_windows = duration_days * windows_per_day
    regions = np.arange(1, n_regions + 1)
    rows = []
    for r in regions:
        region_effect = rng.normal(0, region_effect_sd)
        prev_assign = None
        for w in range(n_windows):
            day = w // windows_per_day + 1
            weekend = 1.0 if day % 7 in (0, 6) else 0.0
            treat = bool(rng.random() < 0.5)
            washout = bool(prev_assign is not None and treat != prev_assign)
            prev_assign = treat
            value = base_value[r - 1] + region_effect - 0.02 * weekend
            value += effect if treat else 0.0
            value += rng.normal(0, noise_sd)
            volume = base_volume[r - 1] * rng.lognormal(0, 0.15)
            rows.append({"region_id": int(r), region_col: int(r), "window": int(w + 1),
                         "day": int(day), "group": "treatment" if treat else "control",
                         "treat": treat, "washout": washout,
                         "baseline_volume": float(base_volume[r - 1]),
                         "volume": float(volume), "value": float(value)})
    df = pd.DataFrame(rows)
    # Segment is a region attribute (large/small by median baseline
    # volume), computed per region, not per window.
    med = df.drop_duplicates("region_id")["baseline_volume"].median()
    df["segment"] = np.where(df["baseline_volume"] >= med, "large", "small")
    return df


def _switchback_trend(df, value_col="value"):
    """Daily trend from primary-analysis units only (washout excluded),
    so the chart and the headline describe the same analysis."""
    prim = df[~df["washout"]]
    rows = []
    for d in sorted(prim["day"].unique()):
        sub = prim[prim["day"] == d]
        c = sub.loc[sub["group"] == "control", value_col]
        t = sub.loc[sub["group"] == "treatment", value_col]
        rows.append({"day": int(d),
                     "control": float(c.mean()) if len(c) else float(prim.loc[prim["group"] == "control", value_col].mean()),
                     "treatment": float(t.mean()) if len(t) else float(prim.loc[prim["group"] == "treatment", value_col].mean())})
    return pd.DataFrame(rows)


def _switchback_analyze(df, progress, value_col, guard_col, guard_desc,
                        unit_label, region_label):
    """Shared switchback analysis: cluster-robust primary, washout
    sensitivity, SRM on all windows, segments by region size."""
    t0 = time.time()
    treat_all = df["group"] == "treatment"
    if progress:
        progress("Generating synthetic data", f"{df['region_id'].nunique()} regions x {df['window'].max()} windows, seed recorded in config")
        progress("Validating entity relationships", "one row per region-window; washout flag recorded per window")
        progress("Assigning treatment and control", f"control {int((~treat_all).sum())} / treatment {int(treat_all.sum())} region-windows (SRM on all windows, washout included)")
    srm = srm_check(int((~treat_all).sum()), int(treat_all.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    prim = df[~df["washout"]]
    tr = prim["group"] == "treatment"
    primary = cluster_robust_lift(prim[value_col].values, tr.values, prim["region_id"].values)
    sens = cluster_robust_lift(df[value_col].values, treat_all.values, df["region_id"].values)
    if progress:
        progress("Computing experiment metrics", f"primary lift {primary['lift_pct']:+.2f}% (unit means, cluster-robust SEs by region, washout excluded)")
    guard = welch_lift(prim.loc[~tr, guard_col], prim.loc[tr, guard_col])
    guard_failed = bool(guard["significant"] and guard_desc(guard))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Evaluating guardrails and statistical validity", f"guardrail lift {guard['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in prim.groupby("segment"):
        s_tr = sub["group"] == "treatment"
        segments.append({"segment": f"{seg} {region_label}",
                         **cluster_robust_lift(sub[value_col].values, s_tr.values, sub["region_id"].values)})
    return {"primary": primary, "guardrail": guard, "srm": srm, "verdict": verdict,
            "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(prim)), "unit": unit_label,
            "n_regions": int(df["region_id"].nunique()),
            "switchback": {"region_col": "region_id", "value_col": value_col,
                           "guard_col": guard_col, "region_label": region_label,
                           "washout_windows": int(df["washout"].sum()),
                           "total_windows": int(len(df)),
                           "sensitivity": sens,
                           "washout_text": (f"Washout: {int(df['washout'].sum())} of {int(len(df))} region-windows "
                                            f"({int(df['washout'].sum()) / len(df) * 100:.0f}%) were excluded from the primary "
                                            f"analysis because they were the first window after a switch. "
                                            f"Including them, the lift is {sens['lift_pct']:+.2f}% "
                                            f"(95% CI [{sens['ci_low_pct']:+.2f}%, {sens['ci_high_pct']:+.2f}%], "
                                            f"p={sens['p_value']:.3g}) versus {primary['lift_pct']:+.2f}% primary. "
                                            f"If the two agree, the washout choice does not drive the conclusion.")}}


# ------------------------------------------- Switchback: SOS / dispatch pricing
def generate_switchback_sos(seed=42, duration_days=28):
    """DoorDash-style SOS (supply-or-surge) pricing switchback.

    Zones switch between SOS pricing on and off, window by window.
    Primary: on-time delivery rate per zone-window (higher is better —
    SOS throttles demand and motivates supply, so on-time performance
    should rise). Guardrail: orders per zone-window must not fall
    significantly by more than 10% (the fee throttles demand by design,
    but a large order loss would veto the pricing change). Ground truth
    at seed 42 is a genuine improvement: on-time rate rises while
    orders dip only slightly, so the verdict is LAUNCH on the stated
    rules. Rates, not counts, are the unit values, so a huge zone does
    not dominate the average.
    """
    rng = np.random.default_rng(seed)
    n = 40
    base_volume = rng.lognormal(mean=4.0, sigma=0.5, size=n)
    base_rate = np.clip(0.86 + rng.normal(0, 0.02, size=n), 0.78, 0.92)
    df = _switchback_frame(rng, n, "zone", base_volume, 0.03, base_rate, 0.045, 0.04, duration_days)
    df = df.rename(columns={"value": "on_time_rate"})
    # Orders per window: requests times a completion factor; SOS throttles
    # demand slightly (-3%) by design.
    df["orders"] = df["volume"] * np.where(df["group"] == "treatment", 0.89, 0.92) * rng.lognormal(0, 0.05, size=len(df))
    df["on_time_deliveries"] = df["volume"] * df["on_time_rate"]
    return df, _switchback_trend(df, "on_time_rate")


def analyze_switchback_sos(df, progress=None):
    out = _switchback_analyze(df, progress, "on_time_rate", "orders",
                              lambda g: g["lift_pct"] < -10, "Region-windows", "zones")
    prim = df[~df["washout"]]
    tr = prim["group"] == "treatment"
    ctrl_rate = float(prim.loc[~tr, "on_time_rate"].mean())
    inc = float((prim.loc[tr, "on_time_deliveries"].sum() - ctrl_rate * prim.loc[tr, "volume"].sum()))
    out["metrics"] = [
        {"name": "On-time delivery rate / zone-window", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "Unit means, cluster-robust SEs by zone", "guardrail": False},
        {"name": "Orders / zone-window", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch (zone-window level)", "guardrail": True}]
    out["business"] = {"incremental_on_time_deliveries": inc,
                       "impact_text": (f"On-time rate rose {out['primary']['treatment_mean'] * 100:.1f}% vs "
                                       f"{out['primary']['control_mean'] * 100:.1f}% in control "
                                       f"({out['primary']['lift_pct']:+.2f}%), worth about {inc:,.0f} extra on-time "
                                       f"deliveries across treated windows at the control rate. Orders per window moved "
                                       f"{out['guardrail']['lift_pct']:+.1f}% — a small demand throttle, not a veto-sized loss. "
                                       f"Those figures come from the generated data, not from the headline lift alone.")}
    return out


# ------------------------------------------- Switchback: surge multiplier
def generate_switchback_surge(seed=42, duration_days=28):
    """Uber-style surge multiplier switchback across cities.

    Cities switch surge pricing on and off, window by window. Primary:
    trip completion rate per city-window (surge should bring drivers
    out and complete more requested trips). Guardrail: rider
    cancellations per city-window must not rise significantly by more
    than 20%. Economics (the point of this scenario): surge is paid on
    EVERY treated trip, not just the incremental ones, so even a real
    completion lift can lose money — computed explicitly, and it vetoes
    a positive lift here, as in the DoorDash promotion design.
    """
    rng = np.random.default_rng(seed)
    n = 30
    base_volume = rng.lognormal(mean=4.5, sigma=0.6, size=n)
    base_rate = np.clip(0.78 + rng.normal(0, 0.03, size=n), 0.68, 0.86)
    df = _switchback_frame(rng, n, "city", base_volume, 0.04, base_rate, 0.055, 0.05, duration_days)
    df = df.rename(columns={"value": "completion_rate"})
    df["completed_trips"] = df["volume"] * df["completion_rate"]
    df["cancel_rate"] = np.clip(np.where(df["group"] == "treatment", 0.065, 0.060) + rng.normal(0, 0.008, size=len(df)), 0, 1)
    return df, _switchback_trend(df, "completion_rate")


def analyze_switchback_surge(df, progress=None):
    out = _switchback_analyze(df, progress, "completion_rate", "cancel_rate",
                              lambda g: g["lift_pct"] > 20, "Region-windows", "cities")
    prim = df[~df["washout"]]
    tr = prim["group"] == "treatment"
    ctrl_rate = float(prim.loc[~tr, "completion_rate"].mean())
    treated_completed = float(prim.loc[tr, "completed_trips"].sum())
    inc = float(prim.loc[tr, "completed_trips"].sum() - ctrl_rate * prim.loc[tr, "volume"].sum())
    bonus_per_trip, value_per_trip = 3.50, 10.0
    cost = treated_completed * bonus_per_trip
    cost_per = (cost / inc) if inc > 0 else float("inf")
    business_reason = None
    if out["primary"]["significant"] and out["primary"]["lift_pct"] > 0 and cost_per > value_per_trip:
        out["verdict"] = "DONT_LAUNCH"
        business_reason = (f"Surge lifts completion by {out['primary']['lift_pct']:+.2f}%, but the bonus is paid on "
                           f"every treated trip (${bonus_per_trip:.2f} x {treated_completed:,.0f} treated trips = "
                           f"${cost:,.0f}), so the cost per incremental trip is ${cost_per:,.2f} against a "
                           f"${value_per_trip:.2f} value per trip. The completion gain does not pay for itself at this bonus level.")
    out["business_reason"] = business_reason
    out["metrics"] = [
        {"name": "Trip completion rate / city-window", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "Unit means, cluster-robust SEs by city", "guardrail": False},
        {"name": "Rider cancellation rate", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch (city-window level)", "guardrail": True}]
    out["business"] = {"incremental_trips": inc, "surge_spend_usd": cost,
                       "cost_per_incremental_trip_usd": float(cost_per),
                       "impact_text": (f"Incremental completed trips: {inc:,.0f}. Surge spend: ${cost:,.0f} "
                                       f"(${bonus_per_trip:.2f} per treated trip). Cost per incremental trip: "
                                       f"${cost_per:,.2f} against a ${value_per_trip:.2f} value per trip. "
                                       f"Those figures come from the generated data, not from the headline lift alone.")}
    return out


# ------------------------------------------- Switchback: dispatch batching
def generate_switchback_batching(seed=42, duration_days=28):
    """Dispatch batching algorithm switchback across delivery zones.

    Zones switch between a batching algorithm (group nearby orders onto
    one courier) and single-order dispatch, window by window. Primary:
    deliveries per courier-hour per zone-window (efficiency, higher is
    better). Guardrail: late-delivery rate must not rise significantly
    by more than 10% (batching can delay the first drop). Ground truth
    at seed 42 is deliberately small (+~1% efficiency) with window
    noise large enough that the effect is not significant — an honest
    INCONCLUSIVE, not a tuned win: batching helps a little, but this
    run cannot establish it.
    """
    rng = np.random.default_rng(seed)
    n = 40
    base_volume = rng.lognormal(mean=4.0, sigma=0.5, size=n)
    base_eff = np.clip(3.0 + rng.normal(0, 0.15, size=n), 2.4, 3.6)
    df = _switchback_frame(rng, n, "zone", base_volume, 0.30, base_eff, 0.03, 0.80, duration_days)
    df = df.rename(columns={"value": "deliveries_per_hour"})
    df["late_rate"] = np.clip(np.where(df["group"] == "treatment", 0.125, 0.120) + rng.normal(0, 0.02, size=len(df)), 0, 1)
    return df, _switchback_trend(df, "deliveries_per_hour")


def analyze_switchback_batching(df, progress=None):
    out = _switchback_analyze(df, progress, "deliveries_per_hour", "late_rate",
                              lambda g: g["lift_pct"] > 10, "Region-windows", "zones")
    out["metrics"] = [
        {"name": "Deliveries / courier-hour / zone-window", "control": out["primary"]["control_mean"],
         "treatment": out["primary"]["treatment_mean"], "lift_pct": out["primary"]["lift_pct"],
         "ci_low_pct": out["primary"]["ci_low_pct"], "ci_high_pct": out["primary"]["ci_high_pct"],
         "p_value": out["primary"]["p_value"], "method": "Unit means, cluster-robust SEs by zone", "guardrail": False},
        {"name": "Late-delivery rate", "control": out["guardrail"]["control_mean"],
         "treatment": out["guardrail"]["treatment_mean"], "lift_pct": out["guardrail"]["lift_pct"],
         "ci_low_pct": out["guardrail"]["ci_low_pct"], "ci_high_pct": out["guardrail"]["ci_high_pct"],
         "p_value": out["guardrail"]["p_value"], "method": "Welch (zone-window level)", "guardrail": True}]
    return out


# ------------------------------------------------- AI Labs experiments
# Teaching point of this section: one AI company is B2B and B2C at
# once, so each scenario uses a different randomization unit — the
# unit is the lesson, not a footnote. Generic scenario names inspired
# by public engineering writing (Statsig cluster randomization,
# GitHub Copilot / Claude Code / Codex / Cowork patterns); nothing
# here is presented as a real company's internal product and there is
# no affiliation claim. All data are synthetic.
#
# - Team coding-agent rollout: CLUSTER-randomized by team. Teammates
#   share repos and reviews, so user-level randomization would spill
#   over. Randomize teams, measure per developer, and the sample size
#   that matters is the number of clusters (Statsig B2B pattern).
# - Individual model upgrade: USER-randomized (individual subscribers
#   mostly do not interfere with each other).
# - App-builder model swap: ACCOUNT-randomized by API developer
#   account (a prompt/model change hits all of that app's users
#   together). The tradeoff is explicit: cost per task is a business
#   outcome, not a footnote, and a superiority verdict on evals does
#   not by itself settle a cost-motivated swap.
# - Cowork delegation: TEAM-randomized (knowledge-work teams
#   delegating whole tasks, not chatting). The trap is symmetric to
#   the coding-agent one: completion without intervention is only a
#   win if corrections do not eat it.
def _cuped_adjusted(post, pre, treat):
    """CUPED adjustment (theta from control only), returned so a
    cluster-robust estimator can run on the adjusted outcome. Same
    theta rule as engine.cuped_lift; the SE, not the adjustment, is
    what changes for clustered data."""
    post = np.asarray(post, dtype=float); pre = np.asarray(pre, dtype=float)
    treat = np.asarray(treat, dtype=bool); ctrl = ~treat
    var_pre = float(np.var(pre[ctrl])) if int(ctrl.sum()) > 1 else 0.0
    theta = float(np.cov(post[ctrl], pre[ctrl])[0, 1] / var_pre) if var_pre > 0 else 0.0
    adj = post - theta * (pre - float(np.mean(pre)))
    reduction = float((1.0 - adj.var(ddof=1) / post.var(ddof=1)) * 100) if post.var(ddof=1) > 0 else 0.0
    return adj, theta, reduction


def _cluster_cuped(post, pre, treat, clusters, flip=False):
    """Cluster-robust lift on the CUPED-adjusted outcome. flip=True
    for lower-is-better primaries (PR cycle time): the returned lift
    is the improvement, so positive means faster, and the CI is
    negated and swapped to match. Means stay the adjusted means,
    exactly as engine.cuped_lift reports adjusted means."""
    adj, theta, reduction = _cuped_adjusted(post, pre, treat)
    raw = cluster_robust_lift(adj, treat, clusters)
    out = dict(raw)
    out["variance_reduction_pct"] = reduction
    out["theta"] = theta
    if flip:
        out["lift_pct"] = -raw["lift_pct"]
        out["ci_low_pct"] = -raw["ci_high_pct"]
        out["ci_high_pct"] = -raw["ci_low_pct"]
    return out


def _rate_trend(rng, duration_days, control_level, treat_level, sigma=0.03):
    days = np.arange(1, duration_days + 1)
    return pd.DataFrame([{"day": int(d),
                          "control": float(control_level * rng.lognormal(0, sigma)),
                          "treatment": float(treat_level * rng.lognormal(0, sigma))}
                         for d in days])


# --------------------------------------- AI Labs: team coding-agent rollout
def generate_ai_team_coding_agent(seed=42, duration_days=28):
    """Cluster-randomized coding-agent rollout. 60 teams (5-30
    developers each), matched pairs by team mean pre-period PR cycle
    time. The agent is assigned per team, so every developer on a
    team shares the arm — that is the design, not a bug, because
    teammates review each other's PRs. Primary: PR cycle time in
    hours (lower is better, so the reported lift is the improvement).
    Guardrails: code churn rate and security findings per developer.
    Acceptance rate is recorded for the treatment arm only and is
    deliberately NOT a metric: a high acceptance rate can mean
    over-reliance, not quality, so it must never be the primary."""
    rng = np.random.default_rng(seed)
    n_teams = 60
    team_size = rng.integers(5, 31, size=n_teams)
    team_base = rng.lognormal(mean=np.log(18.0), sigma=0.30, size=n_teams)
    team_mean_pre = team_base * np.exp(0.5 * 0.22 ** 2)
    treat_team = _matched_pairs(rng, team_mean_pre)
    rows = []
    dev_id = 0
    for t in range(n_teams):
        n_dev = int(team_size[t])
        pre = team_base[t] * rng.lognormal(0, 0.22, size=n_dev)
        post = pre * (1 - 0.08 if treat_team[t] else 1.0) * rng.lognormal(0, 0.15, size=n_dev)
        churn = np.clip(rng.normal(0.120, 0.05, size=n_dev) + (0.003 if treat_team[t] else 0.0), 0.01, 0.6)
        security = rng.poisson(0.80 if not treat_team[t] else 0.84, size=n_dev)
        acceptance = rng.beta(17, 33, size=n_dev) if treat_team[t] else np.full(n_dev, np.nan)
        for i in range(n_dev):
            dev_id += 1
            rows.append({"developer_id": dev_id, "team_id": t + 1,
                         "group": "treatment" if treat_team[t] else "control",
                         "segment": "large" if n_dev >= int(np.median(team_size)) else "small",
                         "team_size": n_dev,
                         "pre_cycle_hours": round(float(pre[i]), 2),
                         "post_cycle_hours": round(float(post[i]), 2),
                         "churn_rate": round(float(churn[i]), 4),
                         "security_findings": int(security[i]),
                         "acceptance_rate": (None if np.isnan(acceptance[i]) else round(float(acceptance[i]), 3))})
    df = pd.DataFrame(rows)
    trend = _rate_trend(rng, duration_days, 18.0, 18.0 * 0.92)
    return df, trend


def analyze_ai_team_coding_agent(df, progress=None):
    t0 = time.time()
    treat = (df["group"] == "treatment").values
    n_teams = int(df["team_id"].nunique())
    n_trt_teams = int(df.drop_duplicates("team_id").query("group == 'treatment'").shape[0])
    if progress:
        progress("Generating synthetic data", f"{len(df):,} developers in {n_teams} teams, seed recorded in config")
        progress("Validating entity relationships", "one row per developer; every developer belongs to exactly one team")
        progress("Assigning treatment and control", f"control {n_teams - n_trt_teams} / treatment {n_trt_teams} teams (matched pairs by baseline cycle time)")
    srm = srm_check(n_teams - n_trt_teams, n_trt_teams)
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'}) — SRM is on teams, the randomization unit")
    primary = _cluster_cuped(df["post_cycle_hours"].values, df["pre_cycle_hours"].values,
                             treat, df["team_id"].values, flip=True)
    churn = welch_lift(df.loc[~treat, "churn_rate"], df.loc[treat, "churn_rate"])
    security = welch_lift(df.loc[~treat, "security_findings"], df.loc[treat, "security_findings"])
    guard_failed = bool((churn["significant"] and churn["lift_pct"] > 10) or
                        (security["significant"] and security["lift_pct"] > 20))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"cycle-time improvement {primary['lift_pct']:+.2f}% (CUPED-adjusted, cluster-robust SEs by team)")
        progress("Evaluating guardrails and statistical validity", f"churn {churn['lift_pct']:+.1f}%, security findings {security['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = (sub["group"] == "treatment").values
        segments.append({"segment": seg, **_cluster_cuped(sub["post_cycle_hours"].values,
                                                         sub["pre_cycle_hours"].values, tr,
                                                         sub["team_id"].values, flip=True)})
    acceptance = float(df.loc[treat, "acceptance_rate"].mean())
    business_reason = None
    if verdict == "LAUNCH":
        business_reason = (f"Launch: PR cycle time improved {primary['lift_pct']:+.1f}% with guardrails passed. "
                           f"Treatment-arm suggestion acceptance averaged {acceptance:.0%} — reported descriptively only, "
                           "never as the decision metric, because high acceptance can mean over-reliance.")
    elif guard_failed:
        business_reason = ("Do not launch on guardrails: cycle time moved, but churn or security findings rose "
                           "past the veto bar. A faster PR that introduces more defects is not a win.")
    return {"primary": primary, "guardrail": churn, "security": security, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "developers",
            "cluster_info": {"n_clusters": n_teams, "cluster_label": "teams",
                             "measured_unit": "developers",
                             "note": "Randomized by team, measured per developer; SEs are cluster-robust by team, so the effective sample size is teams, not developers."},
            "business_reason": business_reason,
            "metrics": [
                {"name": "PR cycle time (hours; lift = improvement, positive = faster)", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED-adjusted, cluster-robust SEs by team; positive = faster", "guardrail": False},
                {"name": "Code churn rate", "control": churn["control_mean"],
                 "treatment": churn["treatment_mean"], "lift_pct": churn["lift_pct"],
                 "ci_low_pct": churn["ci_low_pct"], "ci_high_pct": churn["ci_high_pct"],
                 "p_value": churn["p_value"], "method": "Welch (developer level, team-clustered in the primary)", "guardrail": True},
                {"name": "Security findings / developer", "control": security["control_mean"],
                 "treatment": security["treatment_mean"], "lift_pct": security["lift_pct"],
                 "ci_low_pct": security["ci_low_pct"], "ci_high_pct": security["ci_high_pct"],
                 "p_value": security["p_value"], "method": "Welch (developer level)", "guardrail": True}],
            "business": {"acceptance_rate_treatment": acceptance,
                         "impact_text": (f"PR cycle time improved {primary['lift_pct']:+.1f}% "
                                         f"({primary['control_mean']:.1f}h to {primary['treatment_mean']:.1f}h, CUPED-adjusted means). "
                                         f"Treatment-arm suggestion acceptance averaged {acceptance:.0%} — descriptive only, not the decision metric. "
                                         "Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_cycle_hours", "pre_col": "pre_cycle_hours",
                              "base_col": "team_size", "unit_col": "team_id",
                              "cluster_col": "team_id", "lower_is_better": True,
                              "segment_col": "segment",
                              "segment_labels": {"large": "large teams", "small": "small teams"}}}


# --------------------------------------- AI Labs: individual model upgrade
def generate_ai_individual_model_upgrade(seed=42, duration_days=28):
    """User-randomized model upgrade for individual subscribers.
    Individual users mostly do not interfere with each other, so the
    user is the right unit here — the contrast with the team
    scenarios in this section is the point. Primary: task success
    rate (CUPED on pre-period success). Secondary: 7-day retention
    proxy. Guardrails: cost per task (tokens, $) and latency."""
    rng = np.random.default_rng(seed)
    n_users = 20000
    treat = rng.random(n_users) < 0.5
    is_new = rng.random(n_users) < 0.30
    pre = np.clip(rng.normal(0.62, 0.15, size=n_users), 0.10, 0.95)
    effect = np.where(is_new, 0.06, 0.045)
    post = np.clip(pre * (1 + np.where(treat, effect, 0)) + rng.normal(0, 0.05, size=n_users), 0.05, 1.0)
    cost = rng.lognormal(mean=np.log(0.021), sigma=0.40, size=n_users) * np.where(treat, 1.048, 1.0)
    latency = rng.lognormal(mean=np.log(820.0), sigma=0.30, size=n_users) * np.where(treat, 1.03, 1.0)
    retained = (rng.random(n_users) < np.where(treat, 0.65, 0.62)).astype(int)
    df = pd.DataFrame({"user_id": np.arange(n_users), "group": np.where(treat, "treatment", "control"),
                       "segment": np.where(is_new, "new", "established"),
                       "pre_task_success": np.round(pre, 4), "post_task_success": np.round(post, 4),
                       "cost_per_task_usd": np.round(cost, 5), "latency_ms": np.round(latency, 1),
                       "retained_7d": retained})
    return df, _daily_trend(rng, duration_days, 0.05)


def analyze_ai_individual_model_upgrade(df, progress=None):
    t0 = time.time()
    treat = (df["group"] == "treatment").values
    if progress:
        progress("Generating synthetic data", f"{len(df):,} individual subscribers, seed recorded in config")
        progress("Validating entity relationships", "one row per user; pre/post task success present")
        progress("Assigning treatment and control", f"control {int((~treat).sum()):,} / treatment {int(treat.sum()):,} users")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_task_success"].values, df["pre_task_success"].values, treat)
    cost = welch_lift(df.loc[~treat, "cost_per_task_usd"], df.loc[treat, "cost_per_task_usd"])
    latency = welch_lift(df.loc[~treat, "latency_ms"], df.loc[treat, "latency_ms"])
    retention = welch_lift(df.loc[~treat, "retained_7d"], df.loc[treat, "retained_7d"])
    guard_failed = bool((cost["significant"] and cost["lift_pct"] > 10) or
                        (latency["significant"] and latency["lift_pct"] > 20))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"task success lift {primary['lift_pct']:+.2f}% (CUPED, user level)")
        progress("Evaluating guardrails and statistical validity", f"cost per task {cost['lift_pct']:+.1f}%, latency {latency['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = (sub["group"] == "treatment").values
        segments.append({"segment": seg, **cuped_lift(sub["post_task_success"].values,
                                                     sub["pre_task_success"].values, tr)})
    return {"primary": primary, "guardrail": cost, "latency": latency, "retention": retention, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "users",
            "metrics": [
                {"name": "Task success rate", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED (pre-period task success)", "guardrail": False},
                {"name": "7-day retention proxy", "control": retention["control_mean"],
                 "treatment": retention["treatment_mean"], "lift_pct": retention["lift_pct"],
                 "ci_low_pct": retention["ci_low_pct"], "ci_high_pct": retention["ci_high_pct"],
                 "p_value": retention["p_value"], "method": "Welch (user level)", "guardrail": False},
                {"name": "Cost per task ($)", "control": cost["control_mean"],
                 "treatment": cost["treatment_mean"], "lift_pct": cost["lift_pct"],
                 "ci_low_pct": cost["ci_low_pct"], "ci_high_pct": cost["ci_high_pct"],
                 "p_value": cost["p_value"], "method": "Welch (user level)", "guardrail": True},
                {"name": "Latency (ms)", "control": latency["control_mean"],
                 "treatment": latency["treatment_mean"], "lift_pct": latency["lift_pct"],
                 "ci_low_pct": latency["ci_low_pct"], "ci_high_pct": latency["ci_high_pct"],
                 "p_value": latency["p_value"], "method": "Welch (user level)", "guardrail": True}],
            "business": {"cost_per_task_control_usd": float(cost["control_mean"]),
                         "cost_per_task_treatment_usd": float(cost["treatment_mean"]),
                         "impact_text": (f"Task success moved {primary['lift_pct']:+.1f}% and the 7-day retention proxy moved "
                                         f"{retention['lift_pct']:+.1f}%. Cost per task moved {cost['lift_pct']:+.1f}% "
                                         f"(${cost['control_mean']:.4f} to ${cost['treatment_mean']:.4f}) and latency "
                                         f"{latency['lift_pct']:+.1f}% — both inside their veto bars. Those figures come from "
                                         "the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_task_success", "pre_col": "pre_task_success",
                              "segment_col": "segment",
                              "segment_labels": {"new": "new users", "established": "established users"}}}


# --------------------------------------- AI Labs: app-builder model swap
def generate_ai_app_model_swap(seed=42, duration_days=28):
    """Account-randomized model swap for app builders. A prompt or
    model change hits every user of an app at once, so the developer
    account (API key owner) is the randomization unit — randomizing
    end users would split one app across arms. 300 accounts, matched
    pairs by baseline eval rate. Treatment is the cheaper model
    variant: eval pass rate is the primary, cost per task is the
    explicit tradeoff. Guardrails: latency p90 and error rate.
    Honest caveat carried into the scorecard: an offline eval win is
    a screen, not the decision — the primary here is the ONLINE eval
    pass rate, and a model that wins offline can still lose online."""
    rng = np.random.default_rng(seed)
    n = 300
    baseline = np.clip(rng.normal(0.72, 0.08, size=n), 0.40, 0.95)
    baseline_tasks = rng.lognormal(mean=8.0, sigma=0.6, size=n)
    treat = _matched_pairs(rng, baseline)
    pre = np.clip(baseline * rng.lognormal(0, 0.06, size=n), 0.05, 1.0)
    post = np.clip(baseline * rng.lognormal(0, 0.08, size=n), 0.05, 1.0)
    post = np.where(treat, np.clip(post * 0.988, 0.05, 1.0), post)
    cost = rng.lognormal(mean=np.log(0.42), sigma=0.25, size=n) * np.where(treat, 0.60, 1.0)
    latency = rng.lognormal(mean=np.log(1450.0), sigma=0.20, size=n) * np.where(treat, 1.02, 1.0)
    error = np.clip(rng.normal(0.018, 0.006, size=n) + np.where(treat, 0.001, 0.0), 0.001, 0.10)
    seg = np.where(baseline_tasks >= np.median(baseline_tasks), "large", "small")
    df = pd.DataFrame({"account_id": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "segment": seg, "baseline_tasks": np.round(baseline_tasks, 1),
                       "pre_eval_rate": np.round(pre, 4), "post_eval_rate": np.round(post, 4),
                       "cost_per_task_usd": np.round(cost, 4), "latency_p90_ms": np.round(latency, 1),
                       "error_rate": np.round(error, 4)})
    return df, _rate_trend(rng, duration_days, 0.72, 0.72 * 0.988)


def analyze_ai_app_model_swap(df, progress=None):
    t0 = time.time()
    treat = (df["group"] == "treatment").values
    if progress:
        progress("Generating synthetic data", f"{len(df)} developer accounts, seed recorded in config")
        progress("Validating entity relationships", "one row per account; pre/post eval rates present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} accounts (matched pairs by baseline eval rate)")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_eval_rate"].values, df["pre_eval_rate"].values, treat)
    cost = welch_lift(df.loc[~treat, "cost_per_task_usd"], df.loc[treat, "cost_per_task_usd"])
    latency = welch_lift(df.loc[~treat, "latency_p90_ms"], df.loc[treat, "latency_p90_ms"])
    error = welch_lift(df.loc[~treat, "error_rate"], df.loc[treat, "error_rate"])
    guard_failed = bool((latency["significant"] and latency["lift_pct"] > 15) or
                        (error["significant"] and error["lift_pct"] > 20))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"eval pass rate lift {primary['lift_pct']:+.2f}% (CUPED, account level)")
        progress("Evaluating guardrails and statistical validity", f"cost per task {cost['lift_pct']:+.1f}%, latency p90 {latency['lift_pct']:+.1f}%, error rate {error['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = (sub["group"] == "treatment").values
        segments.append({"segment": seg, **cuped_lift(sub["post_eval_rate"].values,
                                                     sub["pre_eval_rate"].values, tr)})
    saving_1000 = (cost["control_mean"] - cost["treatment_mean"]) * 1000
    if verdict == "INCONCLUSIVE":
        impact = (f"On evals alone this is a wash: eval pass rate moved {primary['lift_pct']:+.1f}% "
                  f"({primary['control_mean']:.3f} to {primary['treatment_mean']:.3f}), not significant. On cost it is not a wash: "
                  f"cost per task fell {cost['lift_pct']:+.1f}% (${cost['control_mean']:.2f} to ${cost['treatment_mean']:.2f}), "
                  f"saving ${saving_1000:,.0f} per 1,000 tasks. If your bar is non-inferiority on evals (say, within 2pp), "
                  "this clears it while saving real money — but the superiority rule above does not launch it, and that "
                  "distinction is the decision, not a footnote. Those figures come from the generated data.")
    elif primary["significant"] and primary["lift_pct"] < 0:
        impact = (f"Eval pass rate fell significantly ({primary['lift_pct']:+.1f}%), so the cost saving "
                  f"(${saving_1000:,.0f} per 1,000 tasks) does not rescue this swap under a superiority bar. "
                  "A cheaper model that measurably degrades evals is a quality regression with a discount attached.")
    else:
        impact = (f"Eval pass rate moved {primary['lift_pct']:+.1f}% and cost per task moved {cost['lift_pct']:+.1f}% "
                  f"(${cost['control_mean']:.2f} to ${cost['treatment_mean']:.2f}, ${saving_1000:,.0f} per 1,000 tasks). "
                  "Those figures come from the generated data, not from the headline lift alone.")
    return {"primary": primary, "guardrail": error, "cost": cost, "latency": latency, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "accounts",
            "metrics": [
                {"name": "Eval pass rate (online)", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED (pre-period eval rate, account level)", "guardrail": False},
                {"name": "Cost per task ($)", "control": cost["control_mean"],
                 "treatment": cost["treatment_mean"], "lift_pct": cost["lift_pct"],
                 "ci_low_pct": cost["ci_low_pct"], "ci_high_pct": cost["ci_high_pct"],
                 "p_value": cost["p_value"], "method": "Welch (account level)", "guardrail": False},
                {"name": "Latency p90 (ms)", "control": latency["control_mean"],
                 "treatment": latency["treatment_mean"], "lift_pct": latency["lift_pct"],
                 "ci_low_pct": latency["ci_low_pct"], "ci_high_pct": latency["ci_high_pct"],
                 "p_value": latency["p_value"], "method": "Welch (account level)", "guardrail": True},
                {"name": "Error rate", "control": error["control_mean"],
                 "treatment": error["treatment_mean"], "lift_pct": error["lift_pct"],
                 "ci_low_pct": error["ci_low_pct"], "ci_high_pct": error["ci_high_pct"],
                 "p_value": error["p_value"], "method": "Welch (account level)", "guardrail": True}],
            "business": {"cost_per_task_control_usd": float(cost["control_mean"]),
                         "cost_per_task_treatment_usd": float(cost["treatment_mean"]),
                         "cost_saving_per_1000_tasks_usd": float(saving_1000),
                         "impact_text": impact},
            "followup_spec": {"post_col": "post_eval_rate", "pre_col": "pre_eval_rate",
                              "base_col": "baseline_tasks", "unit_col": "account_id",
                              "segment_col": "segment",
                              "segment_labels": {"large": "large accounts", "small": "small accounts"}}}


# --------------------------------------- AI Labs: Cowork-style delegation
def generate_ai_cowork_delegation(seed=42, duration_days=28):
    """Team-randomized delegation of whole tasks (not chatting).
    80 knowledge-work teams, matched pairs by baseline task volume.
    Primary: tasks completed without human intervention (share of
    delegated tasks). Guardrails: correction rate and accuracy
    spot-check failures — delegation that has to be redone is not
    delegation. The guardrail veto is the point of this scenario."""
    rng = np.random.default_rng(seed)
    n = 80
    baseline_tasks = rng.lognormal(mean=5.5, sigma=0.5, size=n)
    treat = _matched_pairs(rng, baseline_tasks)
    base_share = np.clip(rng.normal(0.48, 0.10, size=n), 0.15, 0.85)
    pre = np.clip(base_share + rng.normal(0, 0.03, size=n), 0.05, 0.95)
    post = np.clip(base_share * np.where(treat, 1.06, 1.0) + rng.normal(0, 0.04, size=n), 0.05, 0.95)
    correction = np.clip(rng.normal(0.11, 0.04, size=n) + np.where(treat, 0.020, 0.0), 0.01, 0.60)
    spotcheck = np.clip(rng.normal(0.030, 0.012, size=n) + np.where(treat, 0.002, 0.0), 0.001, 0.20)
    seg = np.where(baseline_tasks >= np.median(baseline_tasks), "large", "small")
    df = pd.DataFrame({"team_id": np.arange(1, n + 1), "group": np.where(treat, "treatment", "control"),
                       "segment": seg, "baseline_tasks": np.round(baseline_tasks, 1),
                       "pre_completion_share": np.round(pre, 4), "post_completion_share": np.round(post, 4),
                       "correction_rate": np.round(correction, 4), "spotcheck_failure_rate": np.round(spotcheck, 4)})
    return df, _rate_trend(rng, duration_days, 0.48, 0.48 * 1.06)


def analyze_ai_cowork_delegation(df, progress=None):
    t0 = time.time()
    treat = (df["group"] == "treatment").values
    if progress:
        progress("Generating synthetic data", f"{len(df)} teams, seed recorded in config")
        progress("Validating entity relationships", "one row per team; pre/post completion shares present")
        progress("Assigning treatment and control", f"control {int((~treat).sum())} / treatment {int(treat.sum())} teams (matched pairs by baseline task volume)")
    srm = srm_check(int((~treat).sum()), int(treat.sum()))
    if progress:
        progress("Checking baseline balance", f"SRM p={srm['p_value']:.3f} ({'FAIL' if srm['failed'] else 'pass'})")
    primary = cuped_lift(df["post_completion_share"].values, df["pre_completion_share"].values, treat)
    correction = welch_lift(df.loc[~treat, "correction_rate"], df.loc[treat, "correction_rate"])
    spotcheck = welch_lift(df.loc[~treat, "spotcheck_failure_rate"], df.loc[treat, "spotcheck_failure_rate"])
    guard_failed = bool((correction["significant"] and correction["lift_pct"] > 15) or
                        (spotcheck["significant"] and spotcheck["lift_pct"] > 10))
    verdict = decide(primary, guard_failed, srm["failed"])
    if progress:
        progress("Computing experiment metrics", f"intervention-free completion lift {primary['lift_pct']:+.2f}% (CUPED, team level)")
        progress("Evaluating guardrails and statistical validity", f"correction rate {correction['lift_pct']:+.1f}%, spot-check failures {spotcheck['lift_pct']:+.1f}%")
        progress("Generating the scorecard", f"verdict {verdict} in {time.time() - t0:.2f}s")
    segments = []
    for seg, sub in df.groupby("segment"):
        tr = (sub["group"] == "treatment").values
        segments.append({"segment": seg, **cuped_lift(sub["post_completion_share"].values,
                                                     sub["pre_completion_share"].values, tr)})
    business_reason = None
    if guard_failed:
        business_reason = (f"Do not launch on guardrails: intervention-free completion rose {primary['lift_pct']:+.1f}%, "
                           f"but the correction rate rose {correction['lift_pct']:+.1f}% (significant, past the 15% veto bar). "
                           "Delegation that has to be corrected at that rate is not delegation — it is rework shifted to humans.")
    return {"primary": primary, "guardrail": correction, "spotcheck": spotcheck, "srm": srm,
            "verdict": verdict, "segments": segments, "guardrail_failed": guard_failed,
            "n_units": int(len(df)), "unit": "teams",
            "business_reason": business_reason,
            "metrics": [
                {"name": "Tasks completed without human intervention (share)", "control": primary["control_mean"],
                 "treatment": primary["treatment_mean"], "lift_pct": primary["lift_pct"],
                 "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
                 "p_value": primary["p_value"], "method": "CUPED (pre-period completion share, team level)", "guardrail": False},
                {"name": "Correction rate", "control": correction["control_mean"],
                 "treatment": correction["treatment_mean"], "lift_pct": correction["lift_pct"],
                 "ci_low_pct": correction["ci_low_pct"], "ci_high_pct": correction["ci_high_pct"],
                 "p_value": correction["p_value"], "method": "Welch (team level)", "guardrail": True},
                {"name": "Accuracy spot-check failure rate", "control": spotcheck["control_mean"],
                 "treatment": spotcheck["treatment_mean"], "lift_pct": spotcheck["lift_pct"],
                 "ci_low_pct": spotcheck["ci_low_pct"], "ci_high_pct": spotcheck["ci_high_pct"],
                 "p_value": spotcheck["p_value"], "method": "Welch (team level)", "guardrail": True}],
            "business": {"impact_text": (f"Intervention-free completion moved {primary['lift_pct']:+.1f}% "
                                        f"({primary['control_mean']:.3f} to {primary['treatment_mean']:.3f}), but the correction rate moved "
                                        f"{correction['lift_pct']:+.1f}% and spot-check failures {spotcheck['lift_pct']:+.1f}%. "
                                        "Those figures come from the generated data, not from the headline lift alone.")},
            "followup_spec": {"post_col": "post_completion_share", "pre_col": "pre_completion_share",
                              "base_col": "baseline_tasks", "unit_col": "team_id",
                              "segment_col": "segment",
                              "segment_labels": {"large": "large teams", "small": "small teams"}}}



EXTRA_FLAGSHIPS = {
    "linkedin_creator_crowding_out": (generate_crowding_out, analyze_crowding_out),
    "meta_creator_recommendations": (generate_meta, analyze_meta),
    "tiktok_notifications": (generate_tiktok, analyze_tiktok),
    "linkedin_connections": (generate_linkedin, analyze_linkedin),
    "marketplace_dma_ads": (generate_dma_ads, analyze_dma_ads),
    "uber_driver_incentives": (generate_uber, analyze_uber),
    "delivery_fee_optimization": (generate_fee, analyze_fee),
    "hcp_omnichannel": (generate_hcp, analyze_hcp),
    "field_visits_sampling": (generate_field, analyze_field),
    "cross_publisher_budget": (generate_cross_publisher, analyze_cross_publisher),
    "switchback_sos_pricing": (generate_switchback_sos, analyze_switchback_sos),
    "switchback_surge": (generate_switchback_surge, analyze_switchback_surge),
    "switchback_batching": (generate_switchback_batching, analyze_switchback_batching),
    "ai_team_coding_agent": (generate_ai_team_coding_agent, analyze_ai_team_coding_agent),
    "ai_individual_model_upgrade": (generate_ai_individual_model_upgrade, analyze_ai_individual_model_upgrade),
    "ai_app_model_swap": (generate_ai_app_model_swap, analyze_ai_app_model_swap),
    "ai_cowork_delegation": (generate_ai_cowork_delegation, analyze_ai_cowork_delegation),
}
