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


EXTRA_FLAGSHIPS = {
    "meta_creator_recommendations": (generate_meta, analyze_meta),
    "tiktok_notifications": (generate_tiktok, analyze_tiktok),
    "linkedin_connections": (generate_linkedin, analyze_linkedin),
    "marketplace_dma_ads": (generate_dma_ads, analyze_dma_ads),
    "uber_driver_incentives": (generate_uber, analyze_uber),
    "delivery_fee_optimization": (generate_fee, analyze_fee),
    "hcp_omnichannel": (generate_hcp, analyze_hcp),
    "field_visits_sampling": (generate_field, analyze_field),
    "cross_publisher_budget": (generate_cross_publisher, analyze_cross_publisher),
}
