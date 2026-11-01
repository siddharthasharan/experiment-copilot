"""Scorecard builder — turns an engine result into the results workspace.

Every field is derived from the result dict. The PM view, DS view and
watch-outs are templates filled with actual numbers, not pre-written
conclusions: change the data and the text changes with it.
"""
from __future__ import annotations


def _pct(x):
    return f"{x:+.2f}%"


def build_scorecard(config, result, trend, seed):
    primary = result["primary"]
    verdict = result["verdict"]
    verdict_label = {"LAUNCH": "Launch", "DONT_LAUNCH": "Do Not Launch",
                     "INCONCLUSIVE": "Inconclusive", "INVALID": "Invalid Experiment"}[verdict]
    business = result.get("business", {})
    # KPIs per scenario.
    kpis = [{"label": "Primary lift", "value": _pct(primary["lift_pct"])},
            {"label": "95% CI", "value": f"[{primary['ci_low_pct']:+.2f}%, {primary['ci_high_pct']:+.2f}%]"},
            {"label": "P-value", "value": f"{primary['p_value']:.3g}"},
            {"label": f"Randomized {result['unit']}", "value": f"{result['n_units']:,}"}]
    if "incremental_orders" in business:
        kpis.append({"label": "Incremental orders", "value": f"{business['incremental_orders']:,.0f}"})
        kpis.append({"label": "Contribution", "value": f"${business['contribution_usd']:,.0f}"})
        kpis.append({"label": "Cost / incremental order", "value": f"${business['cost_per_incremental_order_usd']:,.2f}"})
    if "incremental_starts" in business:
        kpis.append({"label": "Incremental treatment starts", "value": f"{business['incremental_starts']:,.0f}"})
        kpis.append({"label": "Cost / incremental start", "value": f"${business['cost_per_incremental_start_usd']:,.0f}"})
    # Cumulative incremental impact (from trend: treatment-control per period, cumulated).
    cum = []
    total = 0.0
    for row in trend:
        total += row["treatment"] - row["control"]
        cum.append({"day": row["day"], "cumulative": round(total, 1)})
    segments = [{"segment": s["segment"], "lift_pct": s["lift_pct"],
                 "ci_low_pct": s["ci_low_pct"], "ci_high_pct": s["ci_high_pct"],
                 "p_value": s["p_value"]} for s in result.get("segments", [])]
    business_note = result.get("business_reason") or ""
    # PM impact must be self-contained and scenario-specific. The old text
    # pointed at "KPI cards above", which do not exist on the PM tab at all,
    # and claimed contribution/cost figures that Reddit does not have (its
    # business dict is empty — it is an engagement experiment, not a priced
    # promotion). Each branch states its own numbers inline, so the PM tab
    # stands alone on every scenario.
    if "contribution_usd" in business:
        impact = (f"Incremental orders: {business['incremental_orders']:,.0f}. Contribution: "
                  f"${business['contribution_usd']:,.0f}. Cost per incremental order: "
                  f"${business['cost_per_incremental_order_usd']:,.2f}. Those figures come from "
                  f"the generated data, not from the headline lift alone — a positive order lift "
                  f"can still lose money once promotion costs are counted.")
    elif "incremental_starts" in business:
        impact = (f"Incremental treatment starts: {business['incremental_starts']:,.0f}. "
                  f"Cost per incremental start: ${business['cost_per_incremental_start_usd']:,.0f}. "
                  f"Incremental spend: ${business['incremental_spend_usd']:,.0f}. Those figures come "
                  f"from the generated data, not from the headline lift alone.")
    else:
        impact = (f"No contribution or cost figures for this design — it is an engagement "
                  f"experiment, not a priced promotion. The business impact is the primary lift "
                  f"itself: {config['primary']} moved {_pct(primary['lift_pct'])} across "
                  f"{result['n_units']:,} {result['unit']}, with guardrails "
                  f"{'failed' if result['guardrail_failed'] else 'passed'}.")
    pm = {
        "what": f"{config['name']}: {config['hypothesis']} The primary metric, {config['primary']}, "
                f"moved {_pct(primary['lift_pct'])} (95% CI [{primary['ci_low_pct']:+.2f}%, {primary['ci_high_pct']:+.2f}%], "
                f"p={primary['p_value']:.3g}).",
        "why": f"The verdict is {verdict_label} under the explicit decision rules: SRM "
               f"{'failed' if result['srm']['failed'] else 'passed'}, guardrails "
               f"{'failed' if result['guardrail_failed'] else 'passed'}. "
               f"{business_note}",
        "impact": impact,
        "launch": verdict_label + ".",
        "next": ("If launching, monitor the guardrail metrics in the rollout; if not, fix the "
                 "flagged issue and re-run rather than re-reading this run until it looks good."),
    }
    # DS-view fixes, bundled with the PM impact fix (same scorecard.py):
    # - SRM is shown because it checks the randomization split — it does NOT
    #   by itself prove the lift is robust. Saying so in the line itself
    #   stops the "SRM passed, so why check anything else?" confusion.
    # - HTE no longer points at a "charts section" that is not on this tab;
    #   the segment numbers are stated inline, so the DS tab stands alone.
    # - Sensitivity no longer says "re-run": follow-ups are re-analyses of
    #   THIS SAME run, not new experiments and not new data. It is also
    #   scenario-specific: excluding the largest units only exists for geo
    #   experiments — for a user-level experiment it is not an option, and
    #   saying so was wrong.
    seg_text = "; ".join(
        f"{s['segment']}: {s['lift_pct']:+.2f}% (95% CI [{s['ci_low_pct']:+.2f}%, {s['ci_high_pct']:+.2f}%], "
        f"p={s['p_value']:.3g})" for s in segments) or "No segments for this experiment."
    if result["unit"] in ("ZIPs", "DMAs"):
        sensitivity = ("Sensitivity — re-analysis of this same run, NOT a new experiment and NOT new "
                       "data: ask in chat to 'exclude the largest units' or to look at a single "
                       "segment. If the headline conclusion flips under those checks, it was fragile.")
    else:
        sensitivity = ("Sensitivity — re-analysis of this same run, NOT a new experiment and NOT new "
                       "data: ask in chat to look at a single segment (new vs established users). "
                       "Excluding the largest units does not apply here — there are no geo units to "
                       "exclude. If the headline conclusion flips under a segment check, it was fragile.")
    ds = {
        "estimand": f"Average treatment effect on {config['primary']} for the tested population.",
        "unit": config["unit"],
        "estimator": config["method"],
        "se": (f"CUPED-adjusted Welch SEs at the {result['unit']} level; primary n: control "
               f"{primary['n_control']:,}, treatment {primary['n_treatment']:,}.")
              if primary.get("variance_reduction_pct") is not None else
              (f"Welch SEs at the {result['unit']} level; primary n: control "
               f"{primary['n_control']:,}, treatment {primary['n_treatment']:,}."),
        "cuped": (f"CUPED variance reduction {primary.get('variance_reduction_pct', 0):.0f}%."
                  if primary.get("variance_reduction_pct") is not None else "No CUPED adjustment for this design."),
        "srm": (f"SRM p={result['srm']['p_value']:.3f} ({'FAIL' if result['srm']['failed'] else 'pass'}). "
                f"SRM checks only that the assignment split matches the design — a pass does not "
                f"by itself prove the lift is robust. That is what the sensitivity checks below are for."),
        "clustering": ("Geo experiment: analysis is at the geo level, so clustering is handled by "
                       "the unit of analysis itself." if result["unit"] in ("ZIPs", "DMAs")
                       else "User-level randomization; no cluster adjustment required."),
        "interference": ("Spillover is a live risk in geo designs: neighbouring control units can be "
                         "affected. The follow-up 'exclude largest markets' and segment checks are "
                         "sensitivity analyses, not spillover proof."
                         if result["unit"] in ("ZIPs", "DMAs") else
                         "No geographic spillover structure in a user-level design."),
        "hte": (f"Segment effects from this run: {seg_text} Segment results are descriptive "
                f"unless pre-specified."),
        "assumptions": "Random assignment, no differential attrition, and (for pre-period adjustment) that the pre-period covariate is not affected by treatment.",
        "sensitivity": sensitivity,
    }
    watchouts = []
    if result["srm"]["failed"]:
        watchouts.append("Randomization failure (SRM): the assignment split does not match the design. Do not trust any lift from this run.")
    if result["guardrail_failed"]:
        watchouts.append("Guardrail failure: a guardrail metric moved significantly against the launch bar. Guardrails veto a launch even when the primary is significant.")
    if result["unit"] in ("ZIPs", "DMAs"):
        watchouts.append("Geographic spillover: treated and control units in the same market/DMA region can contaminate each other. The analysis unit is the geo, not the user or impression inside it.")
        watchouts.append("Multiple-testing: segment and follow-up checks are exploratory unless pre-specified; do not cherry-pick the segment that looks best.")
    if result["unit"] == "DMAs":
        watchouts.append("Claims reporting delay: post-period starts are completeness-adjusted. If completeness differs by arm, the adjustment itself can bias the result — check the claims-completeness guardrail.")
        watchouts.append("Publisher attribution: this design cannot identify individual publisher causal effects. Publisher figures are descriptive delivery shares only.")
    watchouts.append("Statistical power: a non-significant result is not proof of no effect — check the confidence interval width, not just the p-value.")
    watchouts.append("Novelty: early-period effects can decay. The trend chart shows whether the effect holds across the run.")
    return {
        "experiment_name": config["name"], "industry": config["industry"],
        "status": "Complete", "recommendation": verdict_label, "verdict": verdict,
        "primary_metric": config["primary"], "primary_lift_pct": primary["lift_pct"],
        "ci_low_pct": primary["ci_low_pct"], "ci_high_pct": primary["ci_high_pct"],
        "p_value": primary["p_value"], "seed": seed,
        "guardrail_status": "Failed" if result["guardrail_failed"] else "Passed",
        "srm_status": "Failed" if result["srm"]["failed"] else "Passed",
        "business": business, "kpis": kpis, "trend": trend, "cumulative": cum,
        "segments": segments, "metrics": result["metrics"],
        "pm": pm, "ds": ds, "watchouts": watchouts,
        "publishers": result.get("publishers", []),
        "business_reason": result.get("business_reason"),
    }


def example_scorecard(example):
    """Precomputed example -> same scorecard shape, labelled as such."""
    return {
        "experiment_name": example["title"], "industry": "example",
        "status": "Complete (precomputed demonstration)", "recommendation": example["verdict"],
        "verdict": example["verdict"], "primary_metric": example.get("primary_metric", ""),
        "primary_lift_pct": None, "seed": 42,
        "guardrail_status": "See flags in the showroom record",
        "srm_status": "See flags in the showroom record",
        "business": {}, "kpis": [{"label": "Users", "value": f"{example['n_users']:,}"}],
        "trend": [], "cumulative": [], "segments": [], "metrics": example.get("metrics", []),
        "pm": {"what": example["description"],
               "why": example.get("trap_reveal", {}).get("why_it_matters", ""),
               "impact": "", "launch": example["verdict"], "next": ""},
        "ds": {}, "watchouts": [f["text"] for f in example.get("flags", [])],
        "publishers": [], "precomputed": True,
        "trap_reveal": example.get("trap_reveal", {}),
    }
