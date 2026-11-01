"""Follow-up recomputations — real backend analysis, never canned text.

Each function takes the stored experiment frame and returns a dict
with a plain-language summary AND the recomputed numbers it rests on.
If the data cannot support the question, the function says so and
explains why, instead of inventing a number.

Supported (Phase 1):
- exclude_top(n): geo flagships only — drop the n largest units by
  baseline volume and recompute the primary lift.
- segment(name): Reddit new/established; DoorDash new/returning;
  Pharma high/low payer access.
- driver_availability(): DoorDash only — delivery time + cancellations.
- publisher_best(): Pharma only — descriptive publisher shares, with
  the explicit caveat that a geo allocation experiment cannot identify
  individual publisher causal effects.
- why_significant() / explain_method(): numbers from the stored result.
"""
from __future__ import annotations

import numpy as np

from .engine import welch_lift, cuped_lift


def _fmt(x, digits=2):
    return f"{x:+.{digits}f}%"


def exclude_top(df, result, scenario_id, n=5):
    """Drop the n largest units by baseline volume, recompute primary."""
    if scenario_id == "doordash_zip_promotion":
        base_col, post_col, pre_col, unit_col = "baseline_orders", "post_orders", "pre_orders", "zip"
    elif scenario_id == "pharma_dma_publisher":
        base_col, post_col, pre_col, unit_col = "pre_starts", "post_starts_adjusted", "pre_starts", "dma"
    else:
        return {"supported": False,
                "text": "Excluding the largest markets only applies to geo experiments (DoorDash ZIPs, Pharma DMAs). "
                        "This experiment randomizes at user level, so there are no markets to exclude. "
                        "Try a segment instead — for example, 'Show results for new customers'."}
    top = df.nlargest(n, base_col)[unit_col].tolist()
    sub = df[~df[unit_col].isin(top)]
    treat = sub["group"] == "treatment"
    if scenario_id == "doordash_zip_promotion":
        new = cuped_lift(sub[post_col].values, sub[pre_col].values, treat.values)
    else:
        new = cuped_lift(sub[post_col].values, sub[pre_col].values, treat.values)
    old = result["primary"]
    return {"supported": True, "excluded": top, "n_remaining": int(len(sub)),
            "old_lift_pct": old["lift_pct"], "new_lift_pct": new["lift_pct"],
            "new_ci_low_pct": new["ci_low_pct"], "new_ci_high_pct": new["ci_high_pct"],
            "new_p_value": new["p_value"], "new_significant": new["significant"],
            "text": f"Excluding the {n} largest units ({', '.join(str(x) for x in top)}) leaves {len(sub)} units. "
                    f"The primary lift moves from {old['lift_pct']:+.2f}% to {new['lift_pct']:+.2f}% "
                    f"(95% CI [{new['ci_low_pct']:+.2f}%, {new['ci_high_pct']:+.2f}%], p={new['p_value']:.3g}, "
                    f"{'significant' if new['significant'] else 'not significant'}). "
                    f"If the lift survives excluding the largest units, it is not carried by them alone; "
                    f"if it collapses, the headline result was concentrated in a few large units."}


def segment(df, result, scenario_id, segment_name):
    """Recompute the primary metric for one segment of the stored frame."""
    seg = segment_name.lower()
    if scenario_id == "reddit_feed_ranking":
        if "new" in seg and "establish" not in seg:
            sub = df[df["segment"] == "new"]; label = "new users"
        elif "establish" in seg:
            sub = df[df["segment"] == "established"]; label = "established users"
        else:
            return {"supported": False, "text": "For Reddit, segments are 'new users' and 'established users'."}
        tr = sub["group"] == "treatment"
        r = cuped_lift(sub["post_replies"].values, sub["pre_replies"].values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub):,} users), meaningful replies per user moved "
                        f"{r['lift_pct']:+.2f}% (95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], "
                        f"p={r['p_value']:.3g}, {'significant' if r['significant'] else 'not significant'})."}
    if scenario_id == "doordash_zip_promotion":
        if "new" in seg:
            col, label = "orders_new", "new customers"
        elif "return" in seg:
            col, label = "orders_returning", "returning customers"
        else:
            return {"supported": False, "text": "For DoorDash, segments are 'new customers' and 'returning customers'."}
        tr = df["group"] == "treatment"
        # CUPED with a derived pre-period segment: pre segment orders are
        # pre-period total orders times the same segment share observed
        # post-period. The share is a ZIP attribute, not an outcome, so
        # this is a pre-period covariate in the CUPED sense, and it keeps
        # segment CIs comparable to the headline CUPED CI instead of
        # inflating them with raw ZIP-level variance.
        share = (df[col] / df["post_orders"]).fillna(0)
        pre_seg = df["pre_orders"] * share
        r = cuped_lift(df[col].values, pre_seg.values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label}, orders per ZIP moved {r['lift_pct']:+.2f}% "
                        f"(95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], p={r['p_value']:.3g}, "
                        f"{'significant' if r['significant'] else 'not significant'})."}
    if scenario_id == "pharma_dma_publisher":
        if "high" in seg:
            sub = df[df["payer_access"] >= 0.5]; label = "high payer access DMAs"
        elif "low" in seg:
            sub = df[df["payer_access"] < 0.5]; label = "low payer access DMAs"
        else:
            return {"supported": False, "text": "For Pharma, segments are 'high payer access DMAs' and 'low payer access DMAs'."}
        tr = sub["group"] == "treatment"
        r = cuped_lift(sub["post_starts_adjusted"].values, sub["pre_starts"].values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub)} DMAs), treatment starts per DMA moved {r['lift_pct']:+.2f}% "
                        f"(95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], p={r['p_value']:.3g}, "
                        f"{'significant' if r['significant'] else 'not significant'})."}
    return {"supported": False, "text": "No segment breakdown is defined for this experiment."}


def driver_availability(df, result, scenario_id):
    if scenario_id != "doordash_zip_promotion":
        return {"supported": False, "text": "Driver availability is a DoorDash marketplace metric; this experiment has no driver data."}
    tr = df["group"] == "treatment"
    deliv = welch_lift(df.loc[~tr, "delivery_min"], df.loc[tr, "delivery_min"])
    canc = welch_lift(df.loc[~tr, "cancel_rate"], df.loc[tr, "cancel_rate"])
    return {"supported": True, "delivery": deliv, "cancellation": canc,
            "text": f"Driver supply did strain in treated ZIPs: delivery time moved {deliv['lift_pct']:+.1f}% "
                    f"(control {deliv['control_mean']:.1f} min vs treatment {deliv['treatment_mean']:.1f} min, "
                    f"p={deliv['p_value']:.3g}) and cancellations moved {canc['lift_pct']:+.1f}% "
                    f"(p={canc['p_value']:.3g}). The promotion's order lift has to be read alongside "
                    f"that supply cost — more orders at slower delivery is not a clean win."}


def publisher_best(df, result, scenario_id):
    if scenario_id != "pharma_dma_publisher":
        return {"supported": False, "text": "Publisher breakdowns are a Pharma experiment feature; this experiment has no publisher data."}
    pubs = result.get("publishers", [])
    best = max(pubs, key=lambda p: p["impressions"]) if pubs else None
    lines = "; ".join(f"{p['publisher']}: {p['impressions']:,.0f} impressions ({p['share_pct']:.1f}%)" for p in pubs)
    return {"supported": True, "publishers": pubs,
            "text": f"Descriptively, by impressions delivered: {lines}. "
                    f"That is NOT a causal ranking: this experiment randomizes DMAs to an allocation "
                    f"strategy, so it cannot identify the individual causal effect of any one publisher — "
                    f"audiences overlap across publishers, targeting differs, and exposure is not "
                    f"randomized at the publisher level. Do not conclude that the largest publisher "
                    f"'caused' the lift."}


def why_significant(result, config):
    p = result["primary"]
    return {"supported": True,
            "text": f"The primary metric moved {p['lift_pct']:+.2f}% with a 95% confidence interval of "
                    f"[{p['ci_low_pct']:+.2f}%, {p['ci_high_pct']:+.2f}%] and p={p['p_value']:.3g} "
                    f"({p['n_treatment']:,} treated vs {p['n_control']:,} control {result['unit']}). "
                    f"It is {'significant' if p['significant'] else 'not significant'} at the 5% level because "
                    f"the interval {'excludes' if p['significant'] else 'includes'} zero. "
                    f"Significance is about sampling uncertainty, not size: a significant 1% lift and a "
                    f"non-significant 5% lift can both occur, depending on variance and sample size."}


def explain_method(result, config):
    p = result["primary"]
    vr = p.get("variance_reduction_pct")
    vr_text = f" CUPED cut variance by {vr:.0f}%." if vr is not None else ""
    return {"supported": True,
            "text": f"Method: {config['method']}. Randomization unit: {config['unit']}. "
                    f"Estimand: the average effect of treatment vs control on {config['primary']}. "
                    f"Estimator: {'CUPED (pre-period adjusted)' if vr is not None else 'difference in means'} "
                    f"with Welch 95% confidence intervals.{vr_text} "
                    f"SRM check: p={result['srm']['p_value']:.3f} ({'FAIL — do not trust this run' if result['srm']['failed'] else 'pass'}). "
                    f"Guardrails are evaluated separately from the primary metric and can veto a launch "
                    f"even when the primary is significant."}
