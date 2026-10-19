"""
Calibrate the v2 generator against the REAL stats.py functions.

For each of the 6 v2 scenarios, over 10 seeds, verify:
  clean_win:              engagement lift significant ~100% (expect t ~= 6.3)
                          CUPED variance reduction ~0.5-0.6 on engagement/revenue
  srm_trap:               SRM detected ~100%
  peeking_trap:           full-sample NOT significant; early-enrollee cohort IS
  novelty_decay_trap:     full-sample significant; long-exposure cohort NOT;
                          early/late heterogeneity significant
  guardrail_violation:    engagement significant AND guardrail veto fires
  whale_variance_trap:    revenue lift significant ~100% AND std-ratio
                          variance flag ~100% (share rule recorded, not required)
  null_result:            engagement NOT significant

If a check fails badly, retune the GENERATOR (not the detector).
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from experiment_copilot.data_v2 import generate_v2, v2_scenarios  # noqa: E402
from experiment_copilot.stats import (  # noqa: E402
    check_guardrail,
    check_srm,
    check_variance_health,
    cuped,
    estimate_lift,
)
from scipy import stats as sstats  # noqa: E402

N_SEEDS = 10


def _lift(df: pd.DataFrame, col: str):
    c = df.loc[df.group == "control", col].to_numpy(float)
    t = df.loc[df.group == "treatment", col].to_numpy(float)
    return estimate_lift(c, t)


def _se_from_ci(res) -> float:
    return (res.ci_high - res.ci_low) / (2 * 1.96)


def _daily_guardrail_frame(df: pd.DataFrame) -> pd.DataFrame:
    rep = df.loc[df.reports == 1, ["report_day", "group"]].copy()
    rep["report_day"] = rep["report_day"].astype(int)
    piv = rep.groupby(["report_day", "group"]).size().unstack("group", fill_value=0)
    piv = piv.reindex(range(28), fill_value=0)
    for g in ("control", "treatment"):
        if g not in piv.columns:
            piv[g] = 0
    out = piv.reset_index().rename(columns={"report_day": "day"})
    out["reports"] = out["control"] + out["treatment"]  # placeholder, regrouped below
    # check_guardrail expects raw event rows OR pre-aggregated; it re-aggregates
    # by (day, variant) with sum, so hand it one row per (day, variant).
    rows = []
    for _, r in out.iterrows():
        rows.append({"day": int(r["day"]), "group": "control",
                     "reports": int(r["control"])})
        rows.append({"day": int(r["day"]), "group": "treatment",
                     "reports": int(r["treatment"])})
    return pd.DataFrame(rows)


def _hetero_z(a, b) -> float:
    """Two-sided p-value for difference of two relative lifts."""
    se = math.sqrt(_se_from_ci(a) ** 2 + _se_from_ci(b) ** 2)
    z = (a.lift - b.lift) / se if se > 0 else 0.0
    return float(2 * (1 - sstats.norm.cdf(abs(z))))


def calibrate() -> dict:
    results: dict[str, dict[str, list]] = {}
    t_gen = 0.0
    for spec in v2_scenarios():
        rec: dict[str, list] = {}
        for k in range(N_SEEDS):
            s = type(spec)(**{**spec.__dict__, "seed": spec.seed + k * 1000})
            t0 = time.time()
            exp = generate_v2(s)
            t_gen += time.time() - t0
            df = exp.df
            name = spec.name

            if name == "clean_win":
                r = _lift(df, "engagement")
                t_stat = r.lift / _se_from_ci(r)
                rec.setdefault("eng_sig", []).append(r.significant)
                rec.setdefault("t_stat", []).append(t_stat)
                ce = cuped(df, y_col="engagement", pre_col="pre_engagement_28d")
                cr = cuped(df, y_col="revenue", pre_col="pre_revenue_28d")
                rec.setdefault("cuped_vr_eng", []).append(ce.variance_reduction)
                rec.setdefault("cuped_vr_rev", []).append(cr.variance_reduction)
            elif name == "srm_trap":
                n_c = int((df.group == "control").sum())
                n_t = int((df.group == "treatment").sum())
                rec.setdefault("srm_detected", []).append(
                    check_srm(n_c, n_t).detected)
            elif name == "peeking_trap":
                rec.setdefault("full_ns", []).append(
                    not _lift(df, "engagement").significant)
                early = df[df.enrollment_day < 7]
                rec.setdefault("early_sig", []).append(
                    _lift(early, "engagement").significant)
            elif name == "novelty_decay_trap":
                rec.setdefault("full_sig", []).append(
                    _lift(df, "engagement").significant)
                late = df[df.enrollment_day <= 6]      # long exposure
                rec.setdefault("longexp_ns", []).append(
                    not _lift(late, "engagement").significant)
                recent = df[df.enrollment_day >= 21]   # still in novelty window
                p_het = _hetero_z(_lift(recent, "engagement"),
                                  _lift(late, "engagement"))
                rec.setdefault("hetero_sig", []).append(p_het < 0.05)
            elif name == "guardrail_violation_trap":
                rec.setdefault("eng_sig", []).append(
                    _lift(df, "engagement").significant)
                g = check_guardrail(_daily_guardrail_frame(df),
                                    variant_col="group", day_col="day",
                                    metrics=("reports",), min_lift=0.10)
                rec.setdefault("veto", []).append(g.veto)
                rec.setdefault("veto_lift", []).append(g.lift)
            elif name == "whale_variance_trap":
                r = _lift(df, "revenue")
                rec.setdefault("rev_sig", []).append(r.significant)
                vh = check_variance_health(df, y_col="revenue")
                rec.setdefault("stdratio_flag", []).append(
                    vh.warned and vh.std_ratio >= 2.0)
                rec.setdefault("std_ratio", []).append(vh.std_ratio)
                rec.setdefault("share_rule_fired", []).append(
                    vh.outlier_share_treatment > 0.001)
                rec.setdefault("share", []).append(vh.outlier_share_treatment)
            elif name == "null_result":
                rec.setdefault("full_ns", []).append(
                    not _lift(df, "engagement").significant)
        results[spec.name] = rec
    results["_gen_seconds_total"] = t_gen
    return results


def _rate(xs) -> float:
    return sum(1 for x in xs if x) / len(xs)


def scorecard(res: dict) -> str:
    L = []
    L.append("=" * 78)
    L.append("V2 CALIBRATION SCORECARD (10 seeds/scenario, real stats.py functions)")
    L.append("=" * 78)

    def row(scen, check, vals, expect, fmt="rate"):
        if fmt == "rate":
            r = _rate(vals)
            ok = "PASS" if r >= expect else "FAIL"
            L.append(f"[{ok}] {scen:<24}{check:<28}{r:>5.0%}  (need >= {expect:.0%})")
        elif fmt == "mean":
            m = float(np.mean(vals))
            lo, hi = expect
            ok = "PASS" if lo <= m <= hi else "FAIL"
            L.append(f"[{ok}] {scen:<24}{check:<28}{m:>8.3f}  (want {lo}-{hi})")

    c = res["clean_win"]
    row("clean_win", "engagement significant", c["eng_sig"], 1.0)
    row("clean_win", "t-stat mean", c["t_stat"], (5.0, 8.0), "mean")
    row("clean_win", "CUPED var-red engagement", c["cuped_vr_eng"], (0.45, 0.70), "mean")
    row("clean_win", "CUPED var-red revenue", c["cuped_vr_rev"], (0.45, 0.70), "mean")

    row("srm_trap", "SRM detected", res["srm_trap"]["srm_detected"], 1.0)

    p = res["peeking_trap"]
    row("peeking_trap", "full-sample NOT significant", p["full_ns"], 0.8)
    row("peeking_trap", "early cohort significant", p["early_sig"], 1.0)

    n = res["novelty_decay_trap"]
    row("novelty_decay_trap", "full-sample significant", n["full_sig"], 0.9)
    row("novelty_decay_trap", "long-exposure NOT significant", n["longexp_ns"], 0.8)
    row("novelty_decay_trap", "early/late heterogeneity", n["hetero_sig"], 0.9)

    g = res["guardrail_violation_trap"]
    row("guardrail_violation", "engagement significant", g["eng_sig"], 1.0)
    row("guardrail_violation", "guardrail veto fires", g["veto"], 0.9)
    row("guardrail_violation", "veto lift mean", g["veto_lift"], (0.2, 0.6), "mean")

    w = res["whale_variance_trap"]
    row("whale_variance_trap", "revenue significant", w["rev_sig"], 1.0)
    row("whale_variance_trap", "std-ratio flag (>=2.0)", w["stdratio_flag"], 1.0)
    row("whale_variance_trap", "std-ratio mean", w["std_ratio"], (2.0, 6.0), "mean")
    L.append(f"      ... share-rule fired {_rate(w['share_rule_fired']):.0%} "
             f"(mean share {np.mean(w['share']):.5f}; informational only)")

    row("null_result", "NOT significant", res["null_result"]["full_ns"], 0.8)

    L.append("-" * 78)
    L.append(f"generation time: {res['_gen_seconds_total']:.1f}s for "
             f"{6 * N_SEEDS} x 400k-user scenarios "
             f"({res['_gen_seconds_total'] / (6 * N_SEEDS):.2f}s/scenario)")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    res = calibrate()
    print(scorecard(res))
