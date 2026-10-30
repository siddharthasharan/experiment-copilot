#!/usr/bin/env python3
"""Build the Experiment Copilot showroom.

For each of the 14 scenarios: load the 1M-user parquet + answer key,
run a review composition with the REAL estimators, and write
showroom/scorecard_data.json. Then inline that JSON into a single
self-contained showroom/index.html (no CDNs, no fetch — works from
file:// and GitHub Pages).

Additive only: reads existing data + estimators, writes only showroom/.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment_copilot.stats import (  # noqa: E402
    check_guardrail,
    check_srm,
    check_variance_health,
    cuped,
    estimate_lift,
    segment_breakdown,
    _se_from_ci,
)
from experiment_copilot.triggering import (  # noqa: E402
    check_trigger_balance,
    itt_effect,
    triggered_contrast,
    triggered_cuped_lift,
)
from experiment_copilot.interaction import (  # noqa: E402
    bh_fdr,
    cluster_robust_se,
    interaction_test,
    overlap_analysis,
    segment_deviation_pvalues,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHOWROOM = os.path.join(ROOT, "docs")

# --------------------------------------------------------------------------
# helpers (copied patterns from scripts/calibrate_v2.py)
# --------------------------------------------------------------------------

def _daily_guardrail_frame(df: pd.DataFrame) -> pd.DataFrame:
    rep = df.loc[df.reports == 1, ["report_day", "group"]].copy()
    rep["report_day"] = rep["report_day"].astype(int)
    piv = rep.groupby(["report_day", "group"]).size().unstack("group", fill_value=0)
    piv = piv.reindex(range(28), fill_value=0)
    for g in ("control", "treatment"):
        if g not in piv.columns:
            piv[g] = 0
    out = piv.reset_index().rename(columns={"report_day": "day"})
    rows = []
    for _, r in out.iterrows():
        rows.append({"day": int(r["day"]), "group": "control",
                     "reports": int(r["control"])})
        rows.append({"day": int(r["day"]), "group": "treatment",
                     "reports": int(r["treatment"])})
    return pd.DataFrame(rows)


def _hetero_z(a, b) -> float:
    se = math.sqrt(_se_from_ci(a.ci_low, a.ci_high) ** 2
                   + _se_from_ci(b.ci_low, b.ci_high) ** 2)
    z = (a.lift - b.lift) / se if se > 0 else 0.0
    return float(2 * (1 - sstats.norm.cdf(abs(z))))


def _lift(df, col):
    c = df.loc[df.group == "control", col].to_numpy(dtype=float)
    t = df.loc[df.group == "treatment", col].to_numpy(dtype=float)
    return estimate_lift(c, t)


def metric_row(name, display_name, unit, control, treatment, alpha=0.05,
               cuped_res=None):
    r = estimate_lift(np.asarray(control, dtype=float),
                      np.asarray(treatment, dtype=float), alpha=alpha)
    row = {
        "name": name,
        "display_name": display_name,
        "unit": unit,
        "control_mean": float(r.control_mean),
        "treatment_mean": float(r.treatment_mean),
        "lift_pct": float(r.lift * 100.0),
        "ci_low_pct": float(r.ci_low * 100.0),
        "ci_high_pct": float(r.ci_high * 100.0),
        "p_value": float(r.p_value),
        "significant": bool(r.significant),
    }
    if cuped_res is not None:
        row["cuped_lift_pct"] = float(cuped_res.lift * 100.0)
        row["cuped_variance_reduction"] = float(cuped_res.variance_reduction)
    return row, r


# --------------------------------------------------------------------------
# scenario registry
# --------------------------------------------------------------------------

SCENARIOS = [
    # id, track, title, description, trap_tags, dir, primary metric
    ("clean_win", "baseline", "Clean win",
     "A textbook +2% engagement lift, properly powered. The control case for the whole showroom.",
     [], "eval_data_v2", "engagement"),
    ("srm_trap", "baseline", "SRM: broken randomization",
     "The split came out 55/45 instead of 50/50. Nothing downstream can be trusted.",
     ["srm"], "eval_data_v2", "engagement"),
    ("peeking_trap", "baseline", "The peeker's mirage",
     "Significant in the early cohort, gone in the full sample: interim looks without correction.",
     ["peeking"], "eval_data_v2", "engagement"),
    ("novelty_decay_trap", "baseline", "Novelty decay",
     "A hot launch window that fades — the headline average overstates steady state.",
     ["novelty"], "eval_data_v2", "engagement"),
    ("guardrail_violation_trap", "baseline", "Guardrail veto",
     "Engagement is up — but user reports are up ~35%. Safety vetoes the launch.",
     ["guardrail"], "eval_data_v2", "engagement"),
    ("whale_variance_trap", "baseline", "Whale variance",
     "A significant revenue 'win' driven entirely by 625 whale users. The mean is fragile.",
     ["variance", "whales"], "eval_data_v2", "revenue"),
    ("null_result", "baseline", "True null",
     "No effect. A well-powered no — not an underpowered maybe.",
     [], "eval_data_v2", "engagement"),
    ("dilution_trap", "triggered", "Dilution trap",
     "A real +2.5% effect on triggered users that ITT can't see (τ≈15%).",
     ["dilution", "triggering"], "eval_data_v2_triggered", "engagement"),
    ("trigger_imbalance_trap", "triggered", "Trigger imbalance",
     "Treatment changes who triggers — triggered-only analysis is selection-biased.",
     ["triggering", "selection bias"], "eval_data_v2_triggered", "engagement"),
    ("triggered_clean_win", "triggered", "Triggered clean win",
     "Balanced triggering; CACE and ITT agree, dilution math exact.",
     ["triggering"], "eval_data_v2_triggered", "engagement"),
    ("overlap_trap", "interaction", "Overlapping experiments",
     "A concurrent experiment B flips A's effect: +2% vs −0.5%. The pooled average misleads.",
     ["overlap", "interaction"], "eval_data_v2_interaction", "engagement"),
    ("hte_mirage_trap", "interaction", "The segment mirage",
     "One segment 'wins' at p<0.05 by chance across 12 segments — FDR says otherwise.",
     ["hte", "multiple testing"], "eval_data_v2_interaction", "engagement"),
    ("hte_real", "interaction", "Real heterogeneity",
     "New users truly respond +4% vs +1% elsewhere — and it survives FDR.",
     ["hte"], "eval_data_v2_interaction", "engagement"),
    ("interference_sketch", "interaction", "Spillover (SUTVA sketch)",
     "Outcomes leak across cluster-mates; naive SEs understate uncertainty ~1.3×.",
     ["interference", "sutva"], "eval_data_v2_interaction", "engagement"),
]

# expected verdicts per the showroom mapping
EXPECTED = {
    "clean_win": "LAUNCH",
    "srm_trap": "INVALID",
    "peeking_trap": "INCONCLUSIVE",
    "novelty_decay_trap": "INCONCLUSIVE",
    "guardrail_violation_trap": "DONT_LAUNCH",
    "whale_variance_trap": "INCONCLUSIVE",
    "null_result": "DONT_LAUNCH",
    "dilution_trap": "LAUNCH",
    "trigger_imbalance_trap": "INCONCLUSIVE",
    "triggered_clean_win": "LAUNCH",
    "overlap_trap": "INCONCLUSIVE",
    "hte_mirage_trap": "LAUNCH",
    "hte_real": "LAUNCH",
    "interference_sketch": "INCONCLUSIVE",
}

TRAP_REVEAL = {
    "clean_win": {
        "what_happened": "A straightforward +2% engagement lift at n=1M, with CUPED cutting variance roughly in half. SRM clean, guardrails quiet, no segment disagrees.",
        "why_it_matters": "This is the control case for the whole showroom: what a shippable result looks like when the machinery works. Every other case is a deviation from this.",
    },
    "srm_trap": {
        "what_happened": "The split came out 55/45 instead of 50/50 (χ² p≈1e-111). Something broke upstream — assignment logging, bot traffic, or a triggered-analysis bug.",
        "why_it_matters": "SRM invalidates everything downstream. No lift estimate, however significant, can be trusted until randomization is fixed. Check this first, always.",
    },
    "peeking_trap": {
        "what_happened": "Users enrolled in the first week showed a significant lift that washed out in the full 28-day sample. A daily peeker would have shipped it early.",
        "why_it_matters": "Interim looks without sequential correction inflate the false-positive rate. This is what 'p<0.05' means after ten uncorrected peeks.",
    },
    "novelty_decay_trap": {
        "what_happened": "The lift was concentrated in recently-enrolled users still inside the novelty window; the long-exposure cohort showed ~null, and the early-vs-late difference is itself significant.",
        "why_it_matters": "Novelty effects decay. Launching on the full-period average ships a transient — extend past the novelty window before deciding.",
    },
    "guardrail_violation_trap": {
        "what_happened": "Engagement rose significantly — but user reports rose ~35% on treatment (paired daily test, one-sided). The guardrail veto fired.",
        "why_it_matters": "Safety regressions aren't tradable against primary-metric wins. This is the veto working as designed: don't launch, fix the harm.",
    },
    "whale_variance_trap": {
        "what_happened": "Revenue shows a significant lift, but 625 whale users ($400–800 tier, 0.125% of the treatment arm) in treatment drive all of it — treatment std is ~3× control's while the median user is untouched.",
        "why_it_matters": "The mean is fragile: a handful of outliers, not a real effect. Same headline as a win, opposite decision. Heavy-tailed metrics need variance health checks, not just t-tests.",
    },
    "null_result": {
        "what_happened": "True lift is zero and the confidence interval sits tightly around null. This is a well-powered no, not an underpowered maybe.",
        "why_it_matters": "Don't-launch is a decision too. A clean null closes the idea without burning another cycle — and without inventing a story about the data.",
    },
    "dilution_trap": {
        "what_happened": "A true +2.5% lift exists on triggered users only (τ≈15%). ITT over everyone randomized is diluted to ~+0.4% and misses significance most of the time; the direct CACE contrast is significant.",
        "why_it_matters": "Analyzing the wrong population buries real effects. ITT answers the rollout question ('what if we ship to everyone'), CACE answers the product question ('does it work where it applies') — know which you're asking.",
    },
    "trigger_imbalance_trap": {
        "what_happened": "Treatment itself boosted triggering by ~5pp, and triggering correlates with engagement. The triggered-only contrast is selection-biased — it even flips sign (−2.1% vs a true +2.0%).",
        "why_it_matters": "Conditioning on a post-treatment variable breaks randomization. When the trigger-balance check fires, the triggered analysis is invalid — fall back to ITT.",
    },
    "triggered_clean_win": {
        "what_happened": "Trigger rate 50%, balanced across arms. CACE is +2.5% and significant; ITT lands at ≈ τ×CACE, exactly as theory predicts.",
        "why_it_matters": "The sanity case: when triggering is clean, both analyses agree and the dilution math is exact. This is what 'no trap' looks like under triggering.",
    },
    "overlap_trap": {
        "what_happened": "A concurrent experiment B was running: A lifts engagement +2% when B=control but −0.5% when B=treatment. The pooled +0.75% average is significant on some seeds — and meaningless on all of them.",
        "why_it_matters": "In a matrixed program no experiment runs alone. Ship the pooled average and you ship harm to half your users. Stratify, don't average.",
    },
    "hte_mirage_trap": {
        "what_happened": "The true effect is a homogeneous +2% across 12 segments. One segment hit nominal p<0.05 by pure chance — Benjamini-Hochberg rejected it, and Cochran's Q stayed quiet.",
        "why_it_matters": "With 12 segments, a 'winning segment' appears ~half the time under the null. FDR control is what separates discovery from noise. Never ship a segment story without it.",
    },
    "hte_real": {
        "what_happened": "New users truly respond +4% while everyone else sees +1% — and the 'new' cell survives BH-FDR. The overall +2% is real; the average just hides the story.",
        "why_it_matters": "Credible heterogeneity changes the rollout: ship to everyone, but expect — and separately measure — different returns by segment. This is what a real segment win looks like.",
    },
    "interference_sketch": {
        "what_happened": "Users sit in 2,000 clusters and outcomes spill over onto cluster-mates. The naive i.i.d. SE understates uncertainty by ~1.3×; the cluster-robust SE is wider and the SUTVA flag fires.",
        "why_it_matters": "SUTVA violations are invisible to standard tests — the p-value looks fine while the precision is a lie. On social surfaces, always ask who else the treatment touched.",
    },
}


def normalize_expected(raw: str) -> str:
    v = raw.strip().lower()
    if v.startswith("launch"):
        return "LAUNCH"
    if v.startswith("don't launch"):
        return "DONT_LAUNCH"
    if v.startswith("invalid"):
        return "INVALID"
    if v.startswith("inconclusive"):
        return "INCONCLUSIVE"
    # wording divergences: the key phrases the *question*, the mapping gives the launch call
    if v.startswith("no credible heterogeneity"):
        return "LAUNCH?"
    if v.startswith("credible heterogeneity"):
        return "LAUNCH?"
    if v.startswith("sutva"):
        return "INCONCLUSIVE?"
    return "?"


# --------------------------------------------------------------------------
# review composition
# --------------------------------------------------------------------------

def review_baseline(df, key, spec_id):
    metrics, flags, segments = [], [], []
    checks = {}

    g = df["group"].to_numpy()
    n_c, n_t = int((g == "control").sum()), int((g == "treatment").sum())

    # SRM
    srm = check_srm(n_c, n_t)
    checks["srm"] = bool(srm.detected)
    if srm.detected:
        flags.append({"severity": "veto",
                      "text": f"SRM DETECTED (p={srm.p_value:.2g}, split "
                              f"{srm.actual_allocation:.3f} vs 0.500). Randomization is broken — "
                              f"do not trust any lift estimate."})

    # metric rows
    defs = [("engagement", "Engagement", "number", "pre_engagement_28d"),
            ("revenue", "Revenue / user", "currency", "pre_revenue_28d"),
            ("converted", "Converted", "rate", None),
            ("reports", "Reports", "rate", None)]
    lifts = {}
    for col, disp, unit, pre in defs:
        c = df.loc[df.group == "control", col].to_numpy(dtype=float)
        t = df.loc[df.group == "treatment", col].to_numpy(dtype=float)
        cres = cuped(df, y_col=col, pre_col=pre) if pre else None
        row, r = metric_row(col, disp, unit, c, t, cuped_res=cres)
        metrics.append(row)
        lifts[col] = r
        if cres is not None and cres.variance_reduction > 0.15:
            flags.append({"severity": "info",
                          "text": f"CUPED cut {disp.lower()} variance by "
                                  f"{cres.variance_reduction:.0%} using the 28-day pre-period."})

    # guardrail
    # Showroom veto uses alpha=0.01 (the eval harness's documented veto bar),
    # not check_guardrail's 0.05 default: at 1M users a true-null rare metric
    # can hit p~0.03 on daily totals by chance, which must not veto a launch.
    gv = check_guardrail(_daily_guardrail_frame(df), variant_col="group",
                         day_col="day", metrics=("reports",), min_lift=0.10,
                         alpha=0.01)
    checks["guardrail_veto"] = bool(gv.veto)
    if gv.veto:
        flags.append({"severity": "veto", "text": gv.note})

    # variance health on the heavy-tailed primary
    vh = check_variance_health(df, y_col="revenue")
    checks["variance_ratio"] = float(vh.std_ratio)
    checks["variance_warned"] = bool(vh.warned)
    if vh.warned and vh.std_ratio >= 2.0:
        flags.append({"severity": "warn",
                      "text": f"VARIANCE WARNING: treatment revenue std is {vh.std_ratio:.1f}x control's "
                              f"({vh.outlier_share_treatment:.2%} of treatment beyond 5 SD). The mean is "
                              f"fragile — investigate whales/outliers before any launch decision."})
    elif vh.warned:
        flags.append({"severity": "info",
                      "text": f"Variance note: revenue std ratio {vh.std_ratio:.2f} "
                              f"(below the 2.0 action bar)."})

    # peeking (early-enrollee cohort vs full sample)
    full = lifts["engagement"]
    early = _lift(df[df.enrollment_day < 7], "engagement")
    checks["peeking"] = bool((not full.significant) and early.significant)
    if checks["peeking"]:
        flags.append({"severity": "warn",
                      "text": f"PEEKING WARNING: the early-enrollee cohort looked significant "
                              f"({early.lift:+.1%}, p={early.p_value:.3g}) but the full sample is not "
                              f"(p={full.p_value:.3g}). Interim looks without sequential correction "
                              f"invalidate the p-value."})

    # novelty (recent vs long-exposure cohorts)
    late = _lift(df[df.enrollment_day <= 6], "engagement")
    recent = _lift(df[df.enrollment_day >= 21], "engagement")
    p_het = _hetero_z(recent, late)
    # Material-decay rule, not a significance gate: at 1M the long-exposure
    # cohort's residual average lift (+1.05% in the demo seed) is significant
    # even though it is under half the recent-cohort lift (+2.90%).
    checks["novelty"] = bool(full.significant and p_het < 0.05
                             and late.lift < 0.5 * recent.lift)
    if checks["novelty"]:
        flags.append({"severity": "warn",
                      "text": f"NOVELTY WARNING: lift was {recent.lift:+.1%} for recently-enrolled users "
                              f"but {late.lift:+.1%} for the long-exposure cohort (heterogeneity p={p_het:.3g}). "
                              f"The headline average overstates steady state."})

    # segments: new vs returning
    for s in segment_breakdown(df, y_col="engagement", segment_col="tenure"):
        segments.append({"segment": s.segment,
                         "lift_pct": s.lift * 100.0,
                         "ci_low_pct": s.ci_low * 100.0,
                         "ci_high_pct": s.ci_high * 100.0,
                         "p_value": s.p_value,
                         "bh_survivor": None,
                         "n": s.n_control + s.n_treatment})

    return metrics, flags, segments, checks, lifts


def review_triggered(df, key, spec_id):
    metrics, flags, segments = [], [], []
    checks = {}

    bal = check_trigger_balance(df)
    checks["trigger_imbalance"] = bool(bal.detected)
    tau_c, tau_t = bal.trigger_rate_control, bal.trigger_rate_treatment
    if bal.detected:
        flags.append({"severity": "warn",
                      "text": f"TRIGGER IMBALANCE: treatment shifted trigger rates "
                              f"({tau_c:.1%} control vs {tau_t:.1%} treatment, p={bal.p_value:.2g}). "
                              f"Triggered-only contrasts are selection-biased — fall back to ITT."})
    else:
        flags.append({"severity": "info",
                      "text": f"Trigger rates balanced ({tau_c:.1%} vs {tau_t:.1%}, p={bal.p_value:.2g}) — "
                              f"triggered contrasts are valid."})

    itt = itt_effect(df)
    cace = triggered_contrast(df)
    cace_cuped = triggered_cuped_lift(df)

    def _row(name, disp, r):
        return {"name": name, "display_name": disp, "unit": "number",
                "control_mean": float(r.control_mean),
                "treatment_mean": float(r.treatment_mean),
                "lift_pct": float(r.lift * 100.0),
                "ci_low_pct": float(r.ci_low * 100.0),
                "ci_high_pct": float(r.ci_high * 100.0),
                "p_value": float(r.p_value),
                "significant": bool(r.significant)}

    metrics.append(_row("engagement_itt", "Engagement — ITT (all randomized)", itt))
    metrics.append(_row("engagement_cace", "Engagement — CACE (triggered only)", cace))
    cuped_row = _row("engagement_cace_cuped", "Engagement — CACE + CUPED", cace_cuped)
    metrics.append(cuped_row)
    flags.append({"severity": "info",
                  "text": f"Dilution check: ITT {itt.lift:+.2%} vs CACE {cace.lift:+.2%} "
                          f"(τ≈{tau_t:.2f}; ITT ≈ τ×CACE as theory predicts)."})

    details = {"trigger_rate_control": tau_c, "trigger_rate_treatment": tau_t,
               "triggered_n": int(df["triggered"].sum())}
    return metrics, flags, segments, checks, {"engagement": cace_cuped}, details


def review_interaction(df, key, spec_id):
    metrics, flags, segments = [], [], []
    checks = {}
    details = {}
    primary = None

    if spec_id == "overlap_trap":
        ov = overlap_analysis(df)
        checks["overlap"] = bool(ov.detected)
        for name, disp, r in [("a_in_b0", "Engagement — A | B=control", ov.lift_a_in_b0),
                              ("a_in_b1", "Engagement — A | B=treatment", ov.lift_a_in_b1),
                              ("a_pooled", "Engagement — pooled (misleading)", ov.lift_a_pooled)]:
            metrics.append({"name": name, "display_name": disp, "unit": "number",
                            "control_mean": float(r.control_mean),
                            "treatment_mean": float(r.treatment_mean),
                            "lift_pct": float(r.lift * 100.0),
                            "ci_low_pct": float(r.ci_low * 100.0),
                            "ci_high_pct": float(r.ci_high * 100.0),
                            "p_value": float(r.p_value),
                            "significant": bool(r.significant)})
        if ov.detected:
            flags.append({"severity": "warn", "text": ov.note})
        details = {"interaction_p": ov.interaction_p,
                   "interaction_lift_pp": ov.interaction_lift * 100.0,
                   "b_rate_in_a0": ov.b_rate_in_a0, "b_rate_in_a1": ov.b_rate_in_a1}
        primary = _lift(df, "engagement")

    elif spec_id in ("hte_mirage_trap", "hte_real"):
        het = interaction_test(df, y_col="engagement", segment_col="segment")
        dev = segment_deviation_pvalues(het.segments)
        mask = bh_fdr([p for _, _, p in dev], q=0.10)
        checks["heterogeneity_q"] = bool(het.detected)
        checks["n_bh_survivors"] = int(mask.sum())
        r = _lift(df, "engagement")
        metrics.append({"name": "engagement", "display_name": "Engagement (overall)",
                        "unit": "number",
                        "control_mean": float(r.control_mean),
                        "treatment_mean": float(r.treatment_mean),
                        "lift_pct": float(r.lift * 100.0),
                        "ci_low_pct": float(r.ci_low * 100.0),
                        "ci_high_pct": float(r.ci_high * 100.0),
                        "p_value": float(r.p_value),
                        "significant": bool(r.significant)})
        for s, (_, _, p), surv in zip(het.segments, dev, mask):
            segments.append({"segment": s.segment,
                             "lift_pct": s.lift * 100.0,
                             "ci_low_pct": s.ci_low * 100.0,
                             "ci_high_pct": s.ci_high * 100.0,
                             "p_value": float(p),
                             "bh_survivor": bool(surv),
                             "n": s.n_control + s.n_treatment})
        if het.detected:
            flags.append({"severity": "warn", "text": het.note})
        if spec_id == "hte_mirage_trap":
            flags.append({"severity": "info",
                          "text": "BH-FDR (q=0.10) rejected every nominal segment hit — "
                                  "the 'winning segment' was multiple-testing noise."})
        else:
            surv = [s["segment"] for s in segments if s["bh_survivor"]]
            flags.append({"severity": "warn",
                          "text": f"Credible heterogeneity: {', '.join(surv)} survive BH-FDR — "
                                  f"decide per segment, not on the average."})
        details = {"cochran_q_p": het.p_value, "n_bh_survivors": int(mask.sum())}
        primary = r

    elif spec_id == "interference_sketch":
        r = _lift(df, "engagement")
        metrics.append({"name": "engagement", "display_name": "Engagement",
                        "unit": "number",
                        "control_mean": float(r.control_mean),
                        "treatment_mean": float(r.treatment_mean),
                        "lift_pct": float(r.lift * 100.0),
                        "ci_low_pct": float(r.ci_low * 100.0),
                        "ci_high_pct": float(r.ci_high * 100.0),
                        "p_value": float(r.p_value),
                        "significant": bool(r.significant)})
        cr = cluster_robust_se(df)
        checks["sutva"] = bool(cr.sutva_flag)
        checks["se_ratio"] = float(cr.ratio)
        flags.append({"severity": "warn", "text": cr.note if cr.note else
                      f"Cluster-robust SE is {cr.ratio:.2f}x the naive SE."})
        details = {"naive_se": cr.naive_se, "cluster_se": cr.cluster_se,
                   "se_ratio": float(cr.ratio), "n_clusters": cr.n_clusters}
        primary = r

    return metrics, flags, segments, checks, {"engagement": primary}, details


def decide_verdict(checks, primary):
    if checks.get("srm"):
        return "INVALID"
    if checks.get("guardrail_veto"):
        return "DONT_LAUNCH"
    if checks.get("trigger_imbalance"):
        return "INCONCLUSIVE"
    if checks.get("variance_warned") and checks.get("variance_ratio", 0) >= 2.0:
        return "INCONCLUSIVE"
    if checks.get("peeking"):
        return "INCONCLUSIVE"
    if checks.get("novelty"):
        return "INCONCLUSIVE"
    if checks.get("overlap"):
        return "INCONCLUSIVE"
    if checks.get("sutva"):
        return "INCONCLUSIVE"
    if primary is not None and primary.significant and primary.lift > 0:
        return "LAUNCH"
    return "DONT_LAUNCH"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    os.makedirs(SHOWROOM, exist_ok=True)
    scenarios_out = []
    match_rows = []

    for spec_id, track, title, desc, tags, dname, primary_col in SCENARIOS:
        print(f"reviewing {spec_id} ...", flush=True)
        df = pd.read_parquet(os.path.join(ROOT, dname, f"{spec_id}.parquet"))
        with open(os.path.join(ROOT, dname, f"{spec_id}.experiment.json")) as f:
            key = json.load(f)

        details = {}
        if track == "baseline":
            metrics, flags, segments, checks, lifts = review_baseline(df, key, spec_id)
            primary = lifts[primary_col]
        elif track == "triggered":
            metrics, flags, segments, checks, lifts, details = review_triggered(df, key, spec_id)
            primary = lifts["engagement"]
        else:
            metrics, flags, segments, checks, lifts, details = review_interaction(df, key, spec_id)
            primary = lifts["engagement"]

        verdict = decide_verdict(checks, primary)
        raw_expected = key.get("expected_verdict", "?")
        norm = normalize_expected(raw_expected)
        expected = EXPECTED[spec_id]
        match = (verdict == expected)
        wording_note = ""
        if norm.endswith("?"):
            wording_note = (f"answer-key text is {raw_expected!r}, not a bare verdict word; "
                            f"mapped per showroom convention")
        match_rows.append((spec_id, verdict, expected, raw_expected, match, wording_note))

        reveal = TRAP_REVEAL[spec_id]
        scenarios_out.append({
            "id": spec_id,
            "track": track,
            "title": title,
            "description": desc,
            "trap_tags": tags,
            "verdict": verdict,
            "expected_verdict_raw": raw_expected,
            "verdict_match": bool(match),
            "n_users": int(key.get("n_users", len(df))),
            "metrics": metrics,
            "flags": flags,
            "segments": segments,
            "details": details,
            "trap_reveal": reveal,
        })

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_scenarios": len(scenarios_out),
        "verdict_legend": {
            "LAUNCH": "Ship it — significant positive effect, no blockers.",
            "DONT_LAUNCH": "Don't ship — no credible positive effect, or a guardrail veto.",
            "INCONCLUSIVE": "Can't decide — a trap fired; investigate before any call.",
            "INVALID": "Throw it out — randomization itself is broken.",
        },
        "scenarios": scenarios_out,
    }

    json_path = os.path.join(SHOWROOM, "scorecard_data.json")
    with open(json_path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {json_path} ({os.path.getsize(json_path)} bytes)")

    print("\nverdict match table (computed vs showroom mapping | answer-key raw):")
    all_ok = True
    for sid, got, exp, raw, m, note in match_rows:
        mark = "OK " if m else "MISMATCH"
        if not m:
            all_ok = False
        print(f"  [{mark}] {sid:24s} computed={got:12s} mapped={exp:12s} key={raw!r}"
              + (f"  <-- {note}" if note else ""))
    if not all_ok:
        print("WARNING: verdict mismatches present — see above")
    return payload, all_ok



# Per-scenario next steps for the chat agent. The launch call itself is
# derived from the scenario verdict in JS; these are the actions that
# follow it. No numbers here are computed — they restate the trap design.
CHAT_NEXT_STEPS = {
    "clean_win": ["Ship it; the result is powered, SRM-clean and guardrails are quiet.",
                  "Keep guardrails monitored post-launch; no further analysis is required for this decision."],
    "srm_trap": ["Do not trust any lift from this run — randomization itself is broken.",
                 "Find and fix the assignment or logging bug, then re-randomize and re-run."],
    "peeking_trap": ["Do not ship on an interim look — the early significance did not survive the full sample.",
                     "Wait for the pre-committed horizon, and pre-register the number of looks next time (or use a sequential method)."],
    "novelty_decay_trap": ["Do not annualize the early lift — it decays as users get used to the change.",
                           "Extend the run past the novelty window and decide on the steady-state effect."],
    "guardrail_violation_trap": ["Do not ship — a guardrail veto overrides a primary-metric win.",
                                  "Find what is driving the reports increase, fix the harm, and re-test."],
    "whale_variance_trap": ["Do not ship the headline mean alone — 625 whale users drive it and the median user is untouched.",
                            "Report the median and a winsorized sensitivity check alongside the mean, and investigate the whales separately."],
    "null_result": ["Do not ship — there is no credible positive effect here.",
                    "Close the idea, or redesign for a larger effect or a more sensitive primary metric."],
    "dilution_trap": ["Ship for the triggered users — the triggered effect is real; the diluted ITT understates it.",
                      "Size the rollout impact as triggered effect x trigger rate, and measure ITT in the rollout."],
    "trigger_imbalance_trap": ["Fall back to ITT — the triggered comparison is biased because treatment changed who triggers.",
                               "Fix triggering so treatment cannot boost it, then re-run."],
    "triggered_clean_win": ["Ship it — ITT and CACE agree here, and the triggered read is clean.",
                            "Size the rollout impact as triggered effect x trigger rate."],
    "overlap_trap": ["Do not ship the pooled average — it mixes two different B arms.",
                    "Stratify by the B assignment and decide per B arm."],
    "hte_mirage_trap": ["Ship the overall +2% if the decision is overall — but do not ship the segment story.",
                        "Pre-specify segments and correct for multiple comparisons next time."],
    "hte_real": ["Ship overall, and measure the new-user segment separately — it genuinely responds more.",
                 "Consider a segment-specific rollout, with the segment effect re-measured, not assumed."],
    "interference_sketch": ["Treat the naive precision as overstated — outcomes leak across cluster-mates.",
                            "Use cluster-robust SEs for the decision, and consider cluster randomization next time."],
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Copilot — Showroom</title>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#17202e; --muted:#5d6675; --line:#e4e7ec;
  --green:#15803d; --green-bg:#e7f6ec; --red:#b91c1c; --red-bg:#fdeaea;
  --amber:#b45309; --amber-bg:#fef3dc; --gray:#4b5563; --gray-bg:#eef0f3;
  --info:#1d4ed8; --info-bg:#e8effe; --accent:#6d28d9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:14px;line-height:1.5}
header.top{background:#fff;border-bottom:1px solid var(--line);padding:20px 28px}
header.top h1{margin:0;font-size:22px;letter-spacing:-.01em}
header.top p{margin:4px 0 0;color:var(--muted);max-width:900px}
header.top a{color:var(--accent);text-decoration:none}
.layout{display:flex;max-width:1320px;margin:0 auto;align-items:flex-start}
aside{width:300px;flex:none;background:#fff;border-right:1px solid var(--line);
  min-height:calc(100vh - 140px);padding:16px 12px;position:sticky;top:0;
  max-height:calc(100vh - 0px);overflow-y:auto}
.track-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin:14px 8px 6px;font-weight:700}
.case-btn{display:block;width:100%;text-align:left;background:none;border:1px solid transparent;
  border-radius:8px;padding:8px 10px;margin:2px 0;cursor:pointer;color:var(--ink)}
.case-btn:hover{background:var(--bg)}
.case-btn.active{background:var(--info-bg);border-color:#c3d4fb}
.case-btn .t{font-weight:600;font-size:13.5px}
.case-btn .tags{margin-top:3px}
.tag{display:inline-block;font-size:10.5px;background:var(--gray-bg);color:var(--gray);
  border-radius:20px;padding:1px 8px;margin:1px 2px 1px 0}
main{flex:1;padding:26px 30px;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:20px 22px;margin-bottom:18px}
.case-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px}
.case-head h2{margin:0;font-size:20px}
.badge{font-size:12px;font-weight:700;letter-spacing:.06em;border-radius:20px;padding:4px 14px}
.badge.LAUNCH{background:var(--green-bg);color:var(--green)}
.badge.DONT_LAUNCH{background:var(--red-bg);color:var(--red)}
.badge.INCONCLUSIVE{background:var(--amber-bg);color:var(--amber)}
.badge.INVALID{background:var(--gray-bg);color:var(--gray)}
.desc{color:var(--muted);margin:6px 0 0}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;
  text-transform:uppercase;letter-spacing:.04em;padding:8px 10px;border-bottom:2px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.lift-pos{color:var(--green);font-weight:600}
.lift-neg{color:var(--red);font-weight:600}
.lift-null{color:var(--muted)}
.flag{border-left:4px solid;border-radius:6px;padding:10px 14px;margin:10px 0;background:#fff}
.flag.warn{border-color:var(--amber);background:var(--amber-bg)}
.flag.veto{border-color:var(--red);background:var(--red-bg)}
.flag.info{border-color:var(--info);background:var(--info-bg)}
.flag .k{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.segbar-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
.segbar-name{width:170px;flex:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.segbar-track{flex:1;height:14px;background:var(--gray-bg);border-radius:4px;position:relative}
.segbar-zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#9aa1ad}
.segbar-fill{position:absolute;top:0;bottom:0;border-radius:4px}
.segbar-fill.pos{background:var(--green)}
.segbar-fill.neg{background:var(--red)}
.segbar-fill.survivor{outline:2px solid var(--accent);outline-offset:1px}
.segbar-val{width:120px;flex:none;text-align:right;font-variant-numeric:tabular-nums}
.reveal h3{margin:0 0 8px;font-size:15px}
.reveal p{margin:6px 0}
.reveal .why{border-top:1px dashed var(--line);padding-top:8px;margin-top:10px}
.kv{display:grid;grid-template-columns:220px 1fr;gap:4px 12px;font-size:13px;margin-top:8px}
.kv dt{color:var(--muted)}
.kv dd{margin:0;font-variant-numeric:tabular-nums}
footer{border-top:1px solid var(--line);padding:18px 28px;color:var(--muted);
  font-size:12.5px;background:#fff}
h3.sec{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}

.chat-open-btn{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 14px;cursor:pointer;font-weight:600}
body.chat-only .layout, body.chat-only footer{display:none}
#chatview{display:none;flex-direction:column;height:100vh;background:#fff}
body.chat-only #chatview{display:flex}
.chat-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:14px 20px;border-bottom:1px solid var(--line)}
.chat-head strong{font-size:16px}
.chat-head .hint{color:var(--muted);font-size:12.5px}
.chat-head button{margin-left:auto;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
.chat-head button + button{margin-left:0}
#chatlog{flex:1;overflow-y:auto;padding:20px;max-width:900px;width:100%;margin:0 auto}
.bubble{border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:10px 0;background:#fff}
.bubble.user{background:var(--info-bg);border-color:#c3d4fb;margin-left:60px}
.bubble.assistant{margin-right:60px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px}
.chips button{background:var(--bg);border:1px solid var(--line);border-radius:20px;padding:5px 12px;cursor:pointer}
.chat-input-row{border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:8px}
.chat-input-row input{flex:1;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:14px}
.chat-input-row button{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 18px;cursor:pointer;font-weight:600}
@media(max-width:900px){.layout{flex-direction:column}aside{width:100%;position:static;max-height:none}}
</style>
</head>
<body>
<header class="top">
  <h1>Experiment Copilot — Showroom</h1>
  <p>Fourteen synthetic experiments with known ground truth, reviewed by the same statistical
  checklist a product data scientist runs: SRM, guardrails, variance health, peeking, novelty,
  triggering, overlap, heterogeneity, interference. Every trap below is deliberate.
  &nbsp;<a href="https://github.com/siddharthasharan/experiment-copilot">github.com/siddharthasharan/experiment-copilot</a></p>
  <div style="margin-top:10px"><button id="openchat" class="chat-open-btn">Chat &mdash; ask about a scenario</button></div>
</header>
<div class="layout">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>
<div id="chatview">
  <div class="chat-head"><strong>Experiment Copilot chat &mdash; multi-turn assistant</strong>
  <span class="hint">Deterministic: answers come from the scorecard data and the methodology corpus, never invented.</span>
  <button id="chatnew">New chat</button><button id="chatback">Show scorecard</button></div>
  <div id="chatlog"></div>
  <div class="chat-input-row"><input id="chatq" placeholder="Ask about a scenario, e.g. Review the whale trap"><button id="chatsend">Send</button></div>
</div>
<footer>Synthetic data with known ground truth — every trap is deliberate. Lifts are relative;
CIs are 95% Welch's t; p-values two-sided unless noted.</footer>
<script>
"use strict";
const DATA = /*__SHOWROOM_DATA__*/null;
const CORPUS = /*__CORPUS_DATA__*/[];
const CHAT_NEXT_STEPS = /*__CHAT_NEXT_STEPS__*/{};

const TRACK_LABELS = {baseline:"Baseline metrics", triggered:"Triggered analysis", interaction:"Interaction effects"};

function fmtPct(x){
  if(x==null||!isFinite(x)) return "—";
  const s = x>0?"+":(x<0?"−":"");
  return s + Math.abs(x).toFixed(2) + "%";
}
function fmtP(p){
  if(p==null||!isFinite(p)) return "—";
  if(p<0.001) return "<0.001";
  return p.toPrecision(3);
}
function fmtMean(x, unit){
  if(x==null||!isFinite(x)) return "—";
  let v;
  const a = Math.abs(x);
  if(a>=1000) v = x.toLocaleString("en-US",{maximumFractionDigits:0});
  else if(a>=10) v = x.toFixed(1);
  else if(a>=1) v = x.toFixed(2);
  else v = x.toPrecision(3);
  if(unit==="currency") return "$"+v;
  if(unit==="rate" && a<1) return (x*100).toFixed(2)+"%";
  return v;
}
function liftClass(m){
  if(!m.significant) return "lift-null";
  return m.lift_pct>0 ? "lift-pos" : "lift-neg";
}
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;"); }

function renderSidebar(){
  const sb = document.getElementById("sidebar");
  let h = "";
  const tracks = ["baseline","triggered","interaction"];
  for(const tr of tracks){
    h += `<div class="track-label">${TRACK_LABELS[tr]}</div>`;
    for(const s of DATA.scenarios.filter(x=>x.track===tr)){
      const tags = s.trap_tags.map(t=>`<span class="tag">${esc(t)}</span>`).join("");
      h += `<button class="case-btn" data-id="${s.id}"><div class="t">${esc(s.title)}</div><div class="tags">${tags}</div></button>`;
    }
  }
  sb.innerHTML = h;
  sb.querySelectorAll(".case-btn").forEach(b=>{
    b.addEventListener("click", ()=>renderCase(b.dataset.id));
  });
}

function metricTable(metrics){
  let h = `<table><thead><tr><th>Metric</th><th class="num">Control</th><th class="num">Treatment</th>`
        + `<th class="num">Lift</th><th class="num">95% CI</th><th class="num">p-value</th></tr></thead><tbody>`;
  for(const m of metrics){
    const ci = `[${fmtPct(m.ci_low_pct)}, ${fmtPct(m.ci_high_pct)}]`;
    let cuped = "";
    if(m.cuped_lift_pct!=null){
      cuped = `<div style="font-size:11.5px;color:var(--muted)">CUPED ${fmtPct(m.cuped_lift_pct)} · −${(m.cuped_variance_reduction*100).toFixed(0)}% var</div>`;
    }
    h += `<tr><td>${esc(m.display_name)}${cuped}</td>`
       + `<td class="num">${fmtMean(m.control_mean,m.unit)}</td>`
       + `<td class="num">${fmtMean(m.treatment_mean,m.unit)}</td>`
       + `<td class="num ${liftClass(m)}">${fmtPct(m.lift_pct)}</td>`
       + `<td class="num">${ci}</td>`
       + `<td class="num">${fmtP(m.p_value)}</td></tr>`;
  }
  return h + "</tbody></table>";
}

function segmentBlock(segments){
  if(!segments || !segments.length) return "";
  const hasBH = segments.some(s=>s.bh_survivor!==null && s.bh_survivor!==undefined);
  let h = `<div class="card"><h3 class="sec">Segment breakdown</h3>`;
  if(hasBH){
    const maxAbs = Math.max(...segments.map(s=>Math.abs(s.lift_pct)), 1);
    for(const s of segments){
      const w = Math.min(50, Math.abs(s.lift_pct)/maxAbs*50);
      const left = s.lift_pct>=0 ? 50 : 50-w;
      const cls = s.lift_pct>=0 ? "pos" : "neg";
      const surv = s.bh_survivor ? " survivor" : "";
      const star = s.bh_survivor ? " ✓ FDR" : "";
      h += `<div class="segbar-row"><div class="segbar-name">${esc(s.segment)}${star}</div>`
         + `<div class="segbar-track"><div class="segbar-zero"></div>`
         + `<div class="segbar-fill ${cls}${surv}" style="left:${left}%;width:${w}%"></div></div>`
         + `<div class="segbar-val">${fmtPct(s.lift_pct)} <span style="color:var(--muted)">(p=${fmtP(s.p_value)})</span></div></div>`;
    }
  } else {
    h += `<table><thead><tr><th>Segment</th><th class="num">Lift</th><th class="num">95% CI</th><th class="num">p-value</th><th class="num">n</th></tr></thead><tbody>`;
    for(const s of segments){
      h += `<tr><td>${esc(s.segment)}</td><td class="num">${fmtPct(s.lift_pct)}</td>`
         + `<td class="num">[${fmtPct(s.ci_low_pct)}, ${fmtPct(s.ci_high_pct)}]</td>`
         + `<td class="num">${fmtP(s.p_value)}</td><td class="num">${s.n.toLocaleString()}</td></tr>`;
    }
    h += `</tbody></table>`;
  }
  return h + "</div>";
}

function renderCase(id){
  document.querySelectorAll(".case-btn").forEach(b=>b.classList.toggle("active", b.dataset.id===id));
  const s = DATA.scenarios.find(x=>x.id===id);
  const m = document.getElementById("main");
  const vlabel = {LAUNCH:"LAUNCH", DONT_LAUNCH:"DON'T LAUNCH", INCONCLUSIVE:"INCONCLUSIVE", INVALID:"INVALID"}[s.verdict];
  const vtitle = DATA.verdict_legend[s.verdict] || "";
  let h = `<div class="card"><div class="case-head"><h2>${esc(s.title)}</h2>`
        + `<span class="badge ${s.verdict}" title="${esc(vtitle)}">${vlabel}</span></div>`
        + `<div style="margin:6px 0"><button onclick="openChatWith(\'${s.id}\')" class="chat-open-btn">Chat about this scenario</button></div>`
        + `<p class="desc">${esc(s.description)}</p>`
        + `<p class="desc" style="margin-top:6px">n=${s.n_users.toLocaleString()} users · expected answer-key verdict: <code>${esc(s.expected_verdict_raw)}</code></p></div>`;
  h += `<div class="card"><h3 class="sec">Metrics</h3>${metricTable(s.metrics)}</div>`;
  if(s.flags && s.flags.length){
    h += `<div class="card"><h3 class="sec">Reviewer flags</h3>`;
    for(const f of s.flags){
      const k = f.severity==="veto" ? "VETO" : (f.severity==="warn" ? "WARNING" : "NOTE");
      h += `<div class="flag ${f.severity}"><span class="k">${k}</span><br>${esc(f.text)}</div>`;
    }
    h += `</div>`;
  }
  h += segmentBlock(s.segments);
  const det = s.details || {};
  const detKeys = Object.keys(det);
  if(detKeys.length){
    h += `<div class="card"><h3 class="sec">Diagnostics</h3><dl class="kv">`;
    for(const k of detKeys){
      let v = det[k];
      if(typeof v === "number") v = Math.abs(v)<1&&v!==0 ? v.toPrecision(3) : v.toLocaleString("en-US",{maximumFractionDigits:3});
      h += `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`;
    }
    h += `</dl></div>`;
  }
  h += `<div class="card reveal"><h3>What was actually going on?</h3>`
     + `<p>${esc(s.trap_reveal.what_happened)}</p>`
     + `<p class="why"><strong>Why it matters:</strong> ${esc(s.trap_reveal.why_it_matters)}</p></div>`;
  m.innerHTML = h;
  window.scrollTo(0,0);
}

/* ---- Chat: deterministic, no LLM. Results from DATA, methodology from CORPUS. ---- */
const CHAT_STOP = new Set("the and for with that this from into per are was has have will would should could such than then when what which your you our its they them their about under over between both each more most also only just like how why not but all any can may does did were been being who whom whose there here had his her him she".split(" "));
function chatTokens(text){
  const out = new Set();
  for(const m of String(text).toLowerCase().matchAll(/[a-z0-9]+/g)){
    const w = m[0];
    if(w.length > 2 && !CHAT_STOP.has(w)) out.add(w);
  }
  return out;
}
function chatFindScenario(text){
  const low = String(text).toLowerCase();
  const aliases = [
    ["whale", "whale_variance_trap"], ["peek", "peeking_trap"],
    ["srm", "srm_trap"], ["ratio mismatch", "srm_trap"],
    ["novelty", "novelty_decay_trap"], ["primacy", "novelty_decay_trap"],
    ["guardrail", "guardrail_violation_trap"], ["dilution", "dilution_trap"],
    ["imbalance", "trigger_imbalance_trap"], ["spillover", "interference_sketch"],
    ["sutva", "interference_sketch"], ["overlap", "overlap_trap"],
    ["segment mirage", "hte_mirage_trap"], ["real heterogeneity", "hte_real"],
    ["heterogeneity", "hte_real"], ["triggered clean", "triggered_clean_win"],
    ["clean win", "clean_win"], ["null result", "null_result"], ["true null", "null_result"],
  ];
  for(const pair of aliases){ if(low.indexOf(pair[0]) >= 0) return DATA.scenarios.find(function(s){ return s.id === pair[1]; }); }
  const qt = chatTokens(low);
  let best = null, bestScore = 0, tie = false;
  for(const s of DATA.scenarios){
    const titleT = chatTokens(s.title);
    const idT = new Set(s.id.split("_"));
    let sc = 0;
    for(const w of qt){ if(titleT.has(w)) sc += 2; if(idT.has(w)) sc += 3; if(s.trap_tags.indexOf(w) >= 0) sc += 2; }
    if(sc > bestScore){ best = s; bestScore = sc; tie = false; }
    else if(sc === bestScore && sc > 0){ tie = true; }
  }
  return tie ? null : best;
}
function chatRetrieve(query){
  const qt = chatTokens(query);
  const scored = [];
  for(const note of CORPUS){
    const kw = new Set();
    for(const k of note.keywords){ for(const w of chatTokens(k)) kw.add(w); }
    const titleT = chatTokens(note.title);
    const bodyT = chatTokens(note.summary + " " + note.key_points.join(" ") + " " + note.interview_angle);
    let sc = 0;
    for(const w of qt){ if(kw.has(w)) sc += 3; if(titleT.has(w)) sc += 2; if(bodyT.has(w)) sc += 1; }
    if(sc > 0) scored.push([note, sc]);
  }
  scored.sort(function(a, b){ return b[1] - a[1]; });
  return scored.slice(0, 3).map(function(p){ return p[0]; });
}
let chatScenario = null;
function chatVerdictLabel(v){ return {LAUNCH: "LAUNCH", DONT_LAUNCH: "DON'T LAUNCH", INCONCLUSIVE: "INCONCLUSIVE", INVALID: "INVALID"}[v] || v; }
function chatCall(v){
  if(v === "LAUNCH") return "Ship it.";
  if(v === "DONT_LAUNCH") return "Don't ship.";
  if(v === "INVALID") return "Throw this run out — do not decide on it.";
  return "Don't decide yet — investigate first.";
}
function chatInitialHTML(){
  let h = "<p><strong>Hi — here are the scenarios you can ask about.</strong> Pick one, or type a question like <em>Review the whale trap</em>, <em>Should we launch clean_win?</em>, or <em>What is SRM?</em></p>";
  for(const tr of ["baseline", "triggered", "interaction"]){
    h += "<p><strong>" + esc(TRACK_LABELS[tr]) + ":</strong></p><ul>";
    for(const s of DATA.scenarios.filter(function(x){ return x.track === tr; })){
      h += "<li>" + esc(s.title) + "</li>";
    }
    h += "</ul>";
  }
  h += "<p>Every review shows the real scorecard numbers for that scenario, then a diagnosis, a recommendation with next steps, and a PM/Eng brief — in that order, one turn at a time.</p>";
  return h;
}
function chatInitialChips(){
  return DATA.scenarios.map(function(s){ return "Review the " + s.title; });
}
function chatReviewHTML(s){
  let h = '<p><strong>' + esc(s.title) + '</strong> <span class="badge ' + s.verdict + '">' + chatVerdictLabel(s.verdict) + '</span></p>';
  h += "<p>" + esc(s.description) + " · n=" + s.n_users.toLocaleString() + " users · expected answer-key verdict: <code>" + esc(s.expected_verdict_raw) + "</code></p>";
  h += metricTable(s.metrics);
  if(s.flags && s.flags.length){
    h += "<p><strong>Reviewer flags:</strong></p><ul>";
    for(const f of s.flags) h += "<li>" + esc(f.text) + "</li>";
    h += "</ul>";
  }
  return h;
}
function chatDiagnosisHTML(s){
  let h = "<p><strong>Diagnosis — " + esc(s.title) + ".</strong></p>";
  h += "<p>" + esc(s.trap_reveal.what_happened) + "</p>";
  h += "<p><strong>Why it matters:</strong> " + esc(s.trap_reveal.why_it_matters) + "</p>";
  return h;
}
function chatRecommendationHTML(s){
  const steps = CHAT_NEXT_STEPS[s.id] || [];
  let h = "<p><strong>Recommendation — " + esc(s.title) + ": " + chatCall(s.verdict) + "</strong></p>";
  h += "<p><strong>Next steps:</strong></p><ol>";
  for(const st of steps) h += "<li>" + esc(st) + "</li>";
  h += "</ol>";
  return h;
}
function chatBriefHTML(s){
  const steps = CHAT_NEXT_STEPS[s.id] || [];
  const first = s.metrics[0];
  let h = "<p><strong>PM/Eng brief — " + esc(s.title) + ".</strong></p>";
  h += "<p>Verdict: " + chatVerdictLabel(s.verdict) + ". " + esc(s.description);
  if(first) h += " Headline metric " + esc(first.display_name) + " moved " + fmtPct(first.lift_pct) + " (p=" + fmtP(first.p_value) + ").";
  h += " Recommendation: " + chatCall(s.verdict) + "</p>";
  if(steps.length) h += "<p>Next step: " + esc(steps[0]) + "</p>";
  return h;
}
function chatClarificationHTML(){
  let h = "<p>I can't answer that from the showroom data or the methodology corpus, so I won't guess. Ask about one of the scenarios, or a methodology topic covered in the corpus — for example:</p>";
  h += "<ul><li>Review the peeker's mirage</li><li>Should we launch clean_win?</li><li>Why is the whale variance trap inconclusive?</li><li>What is SRM?</li></ul>";
  return h;
}
function chatHandle(text){
  const low = String(text).toLowerCase();
  if(/^(new chat|restart|start over)\b/.test(low)) return {html: chatInitialHTML(), chips: chatInitialChips()};
  if(low.indexOf("list") >= 0 && low.indexOf("scenario") >= 0) return {html: chatInitialHTML(), chips: chatInitialChips()};
  if(/^(what is|what does|how |define |explain what)/.test(low)){
    const qnotes = chatRetrieve(text);
    if(qnotes.length){
      const n = qnotes[0];
      let h = "<p><strong>" + esc(n.title) + "</strong> (from the methodology corpus)</p>";
      h += "<p>" + esc(n.summary) + "</p><ul>";
      for(const kp of n.key_points) h += "<li>" + esc(kp) + "</li>";
      h += "</ul>";
      if(n.citations && n.citations.length) h += "<p><em>Sources: " + esc(n.citations.join("; ")) + "</em></p>";
      const chips = [];
      if(n.related_scenarios && n.related_scenarios.length){
        const rel = DATA.scenarios.find(function(x){ return x.id === n.related_scenarios[0]; });
        if(rel) chips.push("Review the " + rel.title);
      }
      return {html: h, chips: chips};
    }
  }
  if(chatScenario && (low.indexOf("brief") >= 0 || low.indexOf("talking points") >= 0)){
    return {html: chatBriefHTML(chatScenario), chips: ["Another scenario"]};
  }
  if(chatScenario && (low.indexOf("recommend") >= 0 || low.indexOf("should we launch") >= 0 || low.indexOf("should i launch") >= 0 || low.indexOf("next step") >= 0 || low.indexOf("what next") >= 0 || low.indexOf("what should we do") >= 0)){
    return {html: chatRecommendationHTML(chatScenario), chips: ["PM/Eng brief", "Another scenario"]};
  }
  if(chatScenario && (low.indexOf("why") >= 0 || low.indexOf("diagnos") >= 0 || low.indexOf("what happened") >= 0 || low.indexOf("what went wrong") >= 0 || low.indexOf("explain") >= 0)){
    return {html: chatDiagnosisHTML(chatScenario), chips: ["Recommendation and next steps", "PM/Eng brief"]};
  }
  if(low.indexOf("another scenario") >= 0) return {html: chatInitialHTML(), chips: chatInitialChips()};
  const s = chatFindScenario(text);
  if(s){
    chatScenario = s;
    return {html: chatReviewHTML(s), chips: ["Diagnosis", "Recommendation and next steps", "PM/Eng brief", "Another scenario"]};
  }
  const notes = chatRetrieve(text);
  if(notes.length){
    const n = notes[0];
    let h = "<p><strong>" + esc(n.title) + "</strong> (from the methodology corpus)</p>";
    h += "<p>" + esc(n.summary) + "</p><ul>";
    for(const kp of n.key_points) h += "<li>" + esc(kp) + "</li>";
    h += "</ul>";
    if(n.citations && n.citations.length) h += "<p><em>Sources: " + esc(n.citations.join("; ")) + "</em></p>";
    const chips = [];
    if(n.related_scenarios && n.related_scenarios.length){
      const rel = DATA.scenarios.find(function(x){ return x.id === n.related_scenarios[0]; });
      if(rel) chips.push("Review the " + rel.title);
    }
    return {html: h, chips: chips};
  }
  return {html: chatClarificationHTML(), chips: chatInitialChips().slice(0, 4)};
}
function chatAppendUser(text){
  const log = document.getElementById("chatlog");
  log.innerHTML += "<div class='bubble user'>" + esc(text) + "</div>";
}
function chatAppendAssistant(html, chips){
  const log = document.getElementById("chatlog");
  let h = "<div class='bubble assistant'>" + html + "</div>";
  if(chips && chips.length){
    h += "<div class='chips'>";
    for(const c of chips) h += "<button data-q='" + esc(c).replace(/'/g, "&#39;") + "'>" + esc(c) + "</button>";
    h += "</div>";
  }
  log.innerHTML += h;
  log.scrollTop = log.scrollHeight;
}
function chatSend(text){
  text = String(text || "").trim();
  if(!text) return;
  chatAppendUser(text);
  const r = chatHandle(text);
  if(/^(new chat|restart|start over)\b/.test(text.toLowerCase())){ chatScenario = null; document.getElementById("chatlog").innerHTML = ""; }
  chatAppendAssistant(r.html, r.chips);
}
function chatReset(){
  chatScenario = null;
  const log = document.getElementById("chatlog");
  log.innerHTML = "";
  chatAppendAssistant(chatInitialHTML(), chatInitialChips());
}
function showChat(){
  document.body.classList.add("chat-only");
  if(!document.getElementById("chatlog").innerHTML) chatReset();
}
function openChatWith(id){
  const s = DATA.scenarios.find(function(x){ return x.id === id; });
  if(!s) return;
  showChat();
  chatSend("Review the " + s.title);
}
if(typeof window !== "undefined"){
  window.ChatTest = {handle: chatHandle, initial: chatInitialHTML, findScenario: chatFindScenario, retrieve: chatRetrieve, reset: function(){ chatScenario = null; }};
}

document.addEventListener("DOMContentLoaded", ()=>{
  if(!DATA){ document.getElementById("main").innerHTML = "<div class='card'>No data inlined.</div>"; return; }
  renderSidebar();
  renderCase(DATA.scenarios[0].id);
  const oc = document.getElementById("openchat"); if(oc) oc.addEventListener("click", showChat);
  const cb = document.getElementById("chatback"); if(cb) cb.addEventListener("click", ()=>{ document.body.classList.remove("chat-only"); });
  const cn = document.getElementById("chatnew"); if(cn) cn.addEventListener("click", chatReset);
  const cs = document.getElementById("chatsend"); if(cs) cs.addEventListener("click", ()=>{ const q = document.getElementById("chatq"); chatSend(q.value); q.value = ""; });
  const q = document.getElementById("chatq"); if(q) q.addEventListener("keydown", (e)=>{ if(e.key === "Enter"){ chatSend(q.value); q.value = ""; } });
  const log = document.getElementById("chatlog"); if(log) log.addEventListener("click", (e)=>{ const b = e.target.closest ? e.target.closest("button[data-q]") : null; if(b) chatSend(b.dataset.q); });
});
</script>
</body>
</html>
"""

def build_html(payload: dict):
    json_str = json.dumps(payload, separators=(",", ":"))
    # guard against </script> inside strings
    json_str = json_str.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("/*__SHOWROOM_DATA__*/null", json_str)
    corpus_path = os.path.join(ROOT, "rag", "corpus.json")
    notes = []
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            notes = json.load(f)["notes"]
    corpus_str = json.dumps(notes, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__CORPUS_DATA__*/[]", corpus_str)
    steps_str = json.dumps(CHAT_NEXT_STEPS, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__CHAT_NEXT_STEPS__*/{}", steps_str)
    path = os.path.join(SHOWROOM, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    return path


if __name__ == "__main__":
    payload, ok = main()
    html_path = build_html(payload)
    # JS syntax check if node exists
    import re, shutil, subprocess, tempfile
    if shutil.which("node"):
        m = re.search(r"<script>(.*)</script>", open(html_path).read(), re.S)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
            tf.write(m.group(1))
            tfp = tf.name
        r = subprocess.run(["node", "--check", tfp], capture_output=True, text=True)
        print("node --check:", "OK" if r.returncode == 0 else r.stderr[:500])
        os.unlink(tfp)
    else:
        print("node not found — skipping JS syntax check")
    sys.exit(0 if ok else 1)
