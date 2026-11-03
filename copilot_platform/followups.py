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


_SWITCHBACK_IDS = ("switchback_sos_pricing", "switchback_surge", "switchback_batching")


def _switchback_primary(df, result):
    """Cluster-robust primary on primary-analysis (non-washout) windows."""
    from .engine_extra import cluster_robust_lift

    sb = (result or {}).get("switchback") or {}
    prim = df[~df["washout"]]
    tr = prim["group"] == "treatment"
    return cluster_robust_lift(prim[sb["value_col"]].values, tr.values, prim["region_id"].values)


def switchback_washout(df, result, scenario_id):
    """Carryover/washout explanation for switchback runs only."""
    sb = (result or {}).get("switchback") or {}
    if scenario_id not in _SWITCHBACK_IDS or not sb:
        return {"supported": False,
                "text": "Washout only applies to switchback experiments, where regions switch arms over time. "
                        "This experiment does not switch arms, so there is no carryover window to exclude."}
    return {"supported": True, "text": sb["washout_text"] + (
        " The washout rule assumes carryover lasts at most one window: if the system takes longer "
        "than one window to settle after a switch, the primary can still be contaminated, and the "
        "sensitivity above only shows the washout's own effect, it does not fix longer carryover.")}


def exclude_top(df, result, scenario_id, n=5):
    """Drop the n largest units by baseline volume, recompute primary."""
    if scenario_id in _SWITCHBACK_IDS:
        from .engine_extra import cluster_robust_lift

        sb = (result or {}).get("switchback") or {}
        regions = df.drop_duplicates("region_id").nlargest(n, "baseline_volume")["region_id"].tolist()
        sub = df[(~df["washout"]) & (~df["region_id"].isin(regions))]
        tr = sub["group"] == "treatment"
        new = cluster_robust_lift(sub[sb["value_col"]].values, tr.values, sub["region_id"].values)
        old = result["primary"]
        return {"supported": True, "excluded": regions, "n_remaining": int(len(sub)),
                "old_lift_pct": old["lift_pct"], "new_lift_pct": new["lift_pct"],
                "new_ci_low_pct": new["ci_low_pct"], "new_ci_high_pct": new["ci_high_pct"],
                "new_p_value": new["p_value"], "new_significant": new["significant"],
                "text": f"Excluding the {n} largest {sb.get('region_label', 'regions')} ({', '.join(str(x) for x in regions)}) leaves {len(sub)} region-windows (washout windows already excluded). "
                        f"The primary lift moves from {old['lift_pct']:+.2f}% to {new['lift_pct']:+.2f}% "
                        f"(95% CI [{new['ci_low_pct']:+.2f}%, {new['ci_high_pct']:+.2f}%], p={new['p_value']:.3g}, "
                        f"{'significant' if new['significant'] else 'not significant'}), recomputed with the same "
                        f"cluster-robust-by-region estimator as the headline. "
                        f"If the lift survives excluding the largest regions, it is not carried by them alone."}
    # Cluster-randomized scenarios (AI Labs team coding-agent rollout):
    # the unit is the team, not the developer row, so exclude-top drops
    # whole teams and recomputes with the same CUPED-adjusted
    # cluster-robust estimator as the headline. This branch must come
    # before the generic spec branch, which would otherwise treat
    # developer rows as units and rank them by team size.
    spec = (result or {}).get("followup_spec") or {}
    if spec.get("cluster_col"):
        from .engine_extra import _cluster_cuped

        cluster_col, base_col = spec["cluster_col"], spec["base_col"]
        top = df.drop_duplicates(cluster_col).nlargest(n, base_col)[cluster_col].tolist()
        sub = df[~df[cluster_col].isin(top)]
        tr = sub["group"] == "treatment"
        new = _cluster_cuped(sub[spec["post_col"]].values, sub[spec["pre_col"]].values,
                             tr.values, sub[cluster_col].values,
                             flip=bool(spec.get("lower_is_better")))
        old = result["primary"]
        noun = "teams" if cluster_col == "team_id" else "clusters"
        return {"supported": True, "excluded": top, "n_remaining": int(len(sub)),
                "old_lift_pct": old["lift_pct"], "new_lift_pct": new["lift_pct"],
                "new_ci_low_pct": new["ci_low_pct"], "new_ci_high_pct": new["ci_high_pct"],
                "new_p_value": new["p_value"], "new_significant": new["significant"],
                "text": f"Excluding the {n} largest {noun} ({', '.join(str(x) for x in top)}) leaves {len(sub)} developers in {sub[cluster_col].nunique()} {noun}. "
                        f"The primary lift moves from {old['lift_pct']:+.2f}% to {new['lift_pct']:+.2f}% "
                        f"(95% CI [{new['ci_low_pct']:+.2f}%, {new['ci_high_pct']:+.2f}%], p={new['p_value']:.3g}, "
                        f"{'significant' if new['significant'] else 'not significant'}), recomputed with the same "
                        f"CUPED-adjusted, cluster-robust-by-team estimator as the headline. "
                        f"If the lift survives excluding the largest teams, it is not carried by them alone."}
    if scenario_id == "doordash_zip_promotion":
        base_col, post_col, pre_col, unit_col = "baseline_orders", "post_orders", "pre_orders", "zip"
    elif scenario_id == "pharma_dma_publisher":
        base_col, post_col, pre_col, unit_col = "pre_starts", "post_starts_adjusted", "pre_starts", "dma"
    else:
        # Phase 2 geo scenarios carry their own spec in the result, so
        # exclude-top works for any geo unit (DMA, ZIP, city, territory)
        # without a new hardcoded branch per scenario.
        spec = (result or {}).get("followup_spec") or {}
        if spec.get("base_col") and spec.get("unit_col"):
            base_col, post_col, pre_col, unit_col = (spec["base_col"], spec["post_col"],
                                                     spec["pre_col"], spec["unit_col"])
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


def _geo_cols(df, result, scenario_id):
    """(base_col, unit_col) for a geo scenario, or (None, None).

    Flagships keep their explicit mapping; Phase 2 geo scenarios carry
    the same information in the result's followup_spec. User-level
    scenarios have neither — they have no markets to rank.
    """
    if scenario_id == "doordash_zip_promotion":
        return "baseline_orders", "zip"
    if scenario_id == "pharma_dma_publisher":
        return "pre_starts", "dma"
    spec = (result or {}).get("followup_spec") or {}
    if spec.get("base_col") and spec.get("unit_col"):
        return spec["base_col"], spec["unit_col"]
    return None, None


def largest_units(df, result, scenario_id, n=5, smallest=False):
    """Descriptive, not analytical: list the n largest (or smallest)
    units by baseline volume, with their baseline values and their
    combined share of baseline volume.

    This is the follow-up the chat was missing (2026-10-31): "what are
    the five largest DMAs?" is a question about the data, not a request
    to exclude them, and the old handler only recognised "largest"
    inside "exclude the ... largest". Answering it needs no new
    estimation — just a sorted read of the stored frame — and it pairs
    naturally with the exclude-top check ("these are the units that
    check would drop").
    """
    if scenario_id in _SWITCHBACK_IDS:
        sb = (result or {}).get("switchback") or {}
        regions = df.drop_duplicates("region_id")
        ordered = regions.nsmallest(n, "baseline_volume") if smallest else regions.nlargest(n, "baseline_volume")
        total = float(regions["baseline_volume"].sum())
        share = float(ordered["baseline_volume"].sum() / total * 100) if total else 0.0
        kind = "smallest" if smallest else "largest"
        label = sb.get("region_label", "regions")
        items = "; ".join(f"region {row['region_id']} ({row['baseline_volume']:,.1f})" for _, row in ordered.iterrows())
        text = (f"The {n} {kind} {label} by baseline volume are: {items}. "
                f"Together they account for {share:.1f}% of baseline volume.")
        if not smallest:
            text += f" To see whether the headline result survives without them, ask 'Exclude the {n} largest {label}'."
        return {"supported": True, "units": ordered["region_id"].tolist(), "text": text}
    spec = (result or {}).get("followup_spec") or {}
    if spec.get("cluster_col"):
        cluster_col, base_col = spec["cluster_col"], spec["base_col"]
        per_cluster = df.drop_duplicates(cluster_col)
        ordered = per_cluster.nsmallest(n, base_col) if smallest else per_cluster.nlargest(n, base_col)
        total = float(per_cluster[base_col].sum())
        share = float(ordered[base_col].sum() / total * 100) if total else 0.0
        kind = "smallest" if smallest else "largest"
        items = "; ".join(f"team {row[cluster_col]} ({row[base_col]:,.0f} developers)" for _, row in ordered.iterrows())
        text = (f"The {n} {kind} teams by size are: {items}. "
                f"Together they account for {share:.1f}% of developers.")
        if not smallest:
            text += f" To see whether the headline result survives without them, ask 'Exclude the {n} largest teams'."
        return {"supported": True, "units": ordered[cluster_col].tolist(), "text": text}
    base_col, unit_col = _geo_cols(df, result, scenario_id)
    if base_col is None or unit_col not in df.columns or base_col not in df.columns:
        return {"supported": False,
                "text": "Largest units only applies to geo experiments (ZIPs, DMAs, cities, territories). "
                        "This experiment randomizes at user level, so there are no markets to rank. "
                        "Try a segment instead — for example, 'Show results for new users'."}
    n = max(1, int(n))
    ordered = df.nsmallest(n, base_col) if smallest else df.nlargest(n, base_col)
    total = float(df[base_col].sum())
    share = float(ordered[base_col].sum() / total * 100) if total else 0.0
    kind = "smallest" if smallest else "largest"
    unit_name = {"zip": "ZIPs", "dma": "DMAs", "city": "cities", "territory": "territories"}.get(unit_col, unit_col)
    items = "; ".join(f"{unit_col} {row[unit_col]} ({row[base_col]:,.1f})" for _, row in ordered.iterrows())
    text = (f"The {n} {kind} {unit_name} by baseline volume are: {items}. "
            f"Together they account for {share:.1f}% of baseline volume.")
    if not smallest:
        text += (f" To see whether the headline result survives without them, ask "
                 f"'Exclude the {n} largest {unit_name}'.")
    return {"supported": True, "units": ordered[unit_col].tolist(), "text": text}


def segment(df, result, scenario_id, segment_name):
    """Recompute the primary metric for one segment of the stored frame."""
    seg = segment_name.lower()
    if scenario_id in _SWITCHBACK_IDS:
        sb = (result or {}).get("switchback") or {}
        if "large" in seg:
            match, label = "large", f"large {sb.get('region_label', 'regions')}"
        elif "small" in seg:
            match, label = "small", f"small {sb.get('region_label', 'regions')}"
        else:
            return {"supported": False,
                    "text": f"For this switchback, segments are 'large {sb.get('region_label', 'regions')}' and 'small {sb.get('region_label', 'regions')}'."}
        sub = df[(~df["washout"]) & (df["segment"] == match)]
        tr = sub["group"] == "treatment"
        from .engine_extra import cluster_robust_lift

        r = cluster_robust_lift(sub[sb["value_col"]].values, tr.values, sub["region_id"].values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub):,} region-windows, washout excluded), the primary metric moved "
                        f"{r['lift_pct']:+.2f}% (95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], "
                        f"p={r['p_value']:.3g}, {'significant' if r['significant'] else 'not significant'}), "
                        f"with the same cluster-robust-by-region estimator as the headline."}
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
    if scenario_id == "cross_publisher_budget":
        if "high" in seg:
            sub = df[df["payer_access"] >= 0.5]; label = "high payer access DMAs"
        elif "low" in seg:
            sub = df[df["payer_access"] < 0.5]; label = "low payer access DMAs"
        else:
            return {"supported": False, "text": "For cross-publisher, segments are 'high payer access DMAs' and 'low payer access DMAs'."}
        tr = sub["group"] == "treatment"
        r = cuped_lift(sub["post_starts_adjusted"].values, sub["pre_starts"].values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub)} DMAs), treatment starts per DMA moved {r['lift_pct']:+.2f}% "
                        f"(95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], p={r['p_value']:.3g}, "
                        f"{'significant' if r['significant'] else 'not significant'})."}
    if scenario_id == "delivery_fee_optimization":
        if "new" in seg:
            col, label = "orders_new", "new customers"
        elif "return" in seg:
            col, label = "orders_returning", "returning customers"
        else:
            return {"supported": False, "text": "For the fee experiment, segments are 'new customers' and 'returning customers'."}
        tr = df["group"] == "treatment"
        share = (df[col] / df["post_orders"]).fillna(0)
        pre_seg = df["pre_orders"] * share
        r = cuped_lift(df[col].values, pre_seg.values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label}, orders per ZIP moved {r['lift_pct']:+.2f}% "
                        f"(95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], p={r['p_value']:.3g}, "
                        f"{'significant' if r['significant'] else 'not significant'})."}
    # Cluster-randomized segment branch: same estimator as the
    # scorecard segment table (CUPED-adjusted, cluster-robust), so the
    # two answers agree. Must precede the generic branch below.
    spec = (result or {}).get("followup_spec") or {}
    if spec.get("cluster_col") and spec.get("segment_col") in df.columns:
        labels = spec.get("segment_labels") or {}
        match = None
        for v in df[spec["segment_col"]].unique():
            if str(v).lower() == seg.strip() or str(v).lower() in seg:
                match = v
                break
        if match is None:
            return {"supported": False,
                    "text": "Segments for this experiment are: " + ", ".join(labels.values()) + "."}
        label = labels.get(str(match), str(match))
        sub = df[df[spec["segment_col"]] == match]
        tr = sub["group"] == "treatment"
        from .engine_extra import _cluster_cuped

        r = _cluster_cuped(sub[spec["post_col"]].values, sub[spec["pre_col"]].values,
                           tr.values, sub[spec["cluster_col"]].values,
                           flip=bool(spec.get("lower_is_better")))
        metric_word = "improvement" if spec.get("lower_is_better") else "lift"
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub):,} developers in {sub[spec['cluster_col']].nunique()} teams), the primary {metric_word} is "
                        f"{r['lift_pct']:+.2f}% (95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], "
                        f"p={r['p_value']:.3g}, {'significant' if r['significant'] else 'not significant'}), "
                        f"with the same CUPED-adjusted, cluster-robust-by-team estimator as the headline."}
    # Generic Phase 2 branch: any scenario whose result carries a
    # followup_spec with a segment column uses the same CUPED estimator
    # as its scorecard segment table, so the two answers agree.
    spec = (result or {}).get("followup_spec") or {}
    seg_col = spec.get("segment_col")
    if seg_col and seg_col in df.columns:
        labels = spec.get("segment_labels") or {}
        match = None
        for v in df[seg_col].unique():
            if str(v).lower() == seg.strip() or str(v).lower() in seg:
                match = v
                break
        if match is None:
            return {"supported": False,
                    "text": "Segments for this experiment are: " + ", ".join(labels.values()) + "."}
        label = labels.get(str(match), str(match))
        sub = df[df[seg_col] == match]
        tr = sub["group"] == "treatment"
        r = cuped_lift(sub[spec["post_col"]].values, sub[spec["pre_col"]].values, tr.values)
        return {"supported": True, "segment": label, **r,
                "text": f"For {label} (n={len(sub):,} {result.get('unit', 'units')}), the primary metric moved "
                        f"{r['lift_pct']:+.2f}% (95% CI [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%], "
                        f"p={r['p_value']:.3g}, {'significant' if r['significant'] else 'not significant'})."}
    return {"supported": False, "text": "No segment breakdown is defined for this experiment."}


def driver_availability(df, result, scenario_id):
    if scenario_id == "uber_driver_incentives":
        tr = df["group"] == "treatment"
        util = welch_lift(df.loc[~tr, "utilization"], df.loc[tr, "utilization"])
        canc = welch_lift(df.loc[~tr, "cancel_rate"], df.loc[tr, "cancel_rate"])
        return {"supported": True, "utilization": util, "cancellation": canc,
                "text": f"Driver utilization moved {util['lift_pct']:+.1f}% in treated cities "
                        f"(control {util['control_mean']:.2f} vs treatment {util['treatment_mean']:.2f} trips per driver-hour, "
                        f"p={util['p_value']:.3g}) and cancellations moved {canc['lift_pct']:+.1f}% "
                        f"(p={canc['p_value']:.3g}). If utilization rose without cancellations rising, "
                        f"the bonus bought real supply; if cancellations rose too, the extra trips came with service cost."}
    if scenario_id != "doordash_zip_promotion":
        return {"supported": False, "text": "Driver availability is a marketplace metric; this experiment has no driver data."}
    tr = df["group"] == "treatment"
    deliv = welch_lift(df.loc[~tr, "delivery_min"], df.loc[tr, "delivery_min"])
    canc = welch_lift(df.loc[~tr, "cancel_rate"], df.loc[tr, "cancel_rate"])
    return {"supported": True, "delivery": deliv, "cancellation": canc,
            "text": f"Driver supply did strain in treated ZIPs: delivery time moved {deliv['lift_pct']:+.1f}% "
                    f"(control {deliv['control_mean']:.1f} min vs treatment {deliv['treatment_mean']:.1f} min, "
                    f"p={deliv['p_value']:.3g}) and cancellations moved {canc['lift_pct']:+.1f}% "
                    f"(p={canc['p_value']:.3g}). The promotion's order lift has to be read alongside "
                    f"that supply cost — more orders at slower delivery is not a clean win."}


def tier_shares(df, result, scenario_id):
    """Crowding-out answer: tier shares by arm, with the interpretation
    stated from the numbers, not assumed. Only the crowding-out
    scenario has tier columns; everything else gets an honest refusal."""
    if scenario_id != "linkedin_creator_crowding_out":
        return {"supported": False, "text": "Tier shares only exist for the creator crowding-out experiment; this experiment has no creator tiers."}
    tr = df["group"] == "treatment"
    tiers = [("small creators (1,000-10,000 followers)", "post_small_share_pct"),
             ("mid creators (10,000-100,000 followers)", "post_mid_share_pct"),
             ("top creators (100,000+ followers)", "post_top_share_pct")]
    rows = []
    for label, col in tiers:
        c, t = float(df.loc[~tr, col].mean()), float(df.loc[tr, col].mean())
        rows.append((label, c, t, t - c))
    lines = "; ".join(f"{label}: control {c:.1f}% vs treatment {t:.1f}% ({d:+.1f}pp)" for label, c, t, d in rows)
    small_d = rows[0][3]
    top_d = rows[2][3]
    total = result.get("total", {})
    total_lift = total.get("lift_pct", 0.0) if total else 0.0
    if small_d < 0 and top_d > 0:
        verdict = (f"That is crowding-out: total creator sessions rose {total_lift:+.1f}%, but the "
                   f"small tier's share fell while the top tier's share rose — the gain went to top creators.")
    elif small_d > 0:
        verdict = ("That is surfacing, not crowding-out: the small tier's share rose, so small creators "
                   "gained exposure from this ranking.")
    else:
        verdict = ("No clear crowding-out verdict from shares alone: the small tier's share did not move "
                   "in a single direction against the top tier.")
    return {"supported": True, "tiers": rows,
            "text": f"By arm, share of creator sessions: {lines}. {verdict} "
                    f"Remember the unit: this is viewer exposure (randomized at viewer level), not creator "
                    f"earnings — it shows what viewers saw, not what any individual creator gained or lost."}


def publisher_best(df, result, scenario_id):
    # Any scenario whose result carries publisher shares (Pharma flagship
    # and cross-publisher) gets the descriptive answer with the caveat;
    # scenarios without publisher data get an honest refusal.
    pubs = result.get("publishers", [])
    if not pubs:
        return {"supported": False, "text": "Publisher breakdowns are a Pharma experiment feature; this experiment has no publisher data."}
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
