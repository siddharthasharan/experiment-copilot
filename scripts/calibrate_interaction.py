"""Calibrate step-3 interaction scenarios: 10 seeds each, verified against the
real estimators (overlap_analysis, interaction_test, bh_fdr, cluster_robust_se).

Gates:
  overlap_trap:       A×B interaction detected >=9/10; stratified estimates
                      recover +2% / -0.5% within 0.8pp >=9/10. (Pooled naive
                      significance is informational — the trap misleads
                      whenever it fires.)
  hte_mirage_trap:    BH-FDR leaves zero survivors >=8/10; Cochran's Q not
                      significant >=8/10. Raw nominal-hit rate is
                      informational (the mirage appears ~half the seeds).
  hte_real:           pre-specified 'new' segment survives BH-FDR >=9/10.
  interference_sketch: cluster/naive SE ratio > 1.15 in >=9/10; SUTVA flag
                      (ratio > 1.25) fires >=8/10; lambda=0 control ratio in
                      [0.93, 1.10] >=9/10 (the estimator is honest when there
                      is nothing to find).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from experiment_copilot.interaction import (  # noqa: E402
    bh_fdr, cluster_robust_se, interaction_test, overlap_analysis,
    segment_deviation_pvalues)
from experiment_copilot.interaction_data import (  # noqa: E402
    generate_interaction, generate_interference, interaction_scenarios)

N_SEEDS = 10


def _copy_seed(spec, k: int):
    return type(spec)(**{**spec.__dict__, "seed": spec.seed + k * 1000})


def calibrate() -> dict:
    results: dict[str, dict[str, list]] = {}
    t_gen = 0.0
    for spec in interaction_scenarios():
        rec: dict[str, list] = {}
        for k in range(N_SEEDS):
            s = _copy_seed(spec, k)
            t0 = time.time()
            exp = generate_interaction(s)
            t_gen += time.time() - t0
            df = exp.df
            name = spec.name

            if name == "overlap_trap":
                ov = overlap_analysis(df)
                rec.setdefault("interaction_detected", []).append(ov.detected)
                rec.setdefault("interaction_p", []).append(ov.interaction_p)
                rec.setdefault("b0_est", []).append(ov.lift_a_in_b0.lift)
                rec.setdefault("b1_est", []).append(ov.lift_a_in_b1.lift)
                rec.setdefault("b0_ok", []).append(
                    abs(ov.lift_a_in_b0.lift - 0.02) < 0.008)
                rec.setdefault("b1_ok", []).append(
                    abs(ov.lift_a_in_b1.lift + 0.005) < 0.008)
                rec.setdefault("pooled_sig", []).append(
                    ov.lift_a_pooled.significant)
                rec.setdefault("pooled_lift", []).append(ov.lift_a_pooled.lift)
            elif name in ("hte_mirage_trap", "hte_real"):
                het = interaction_test(df)
                dev = segment_deviation_pvalues(het.segments)
                pvals = np.array([p for _, _, p in dev])
                raw_hits = int(np.sum(pvals < 0.05))
                mask = bh_fdr(pvals, q=0.10)
                rec.setdefault("raw_hits", []).append(raw_hits)
                rec.setdefault("bh_survivors", []).append(int(mask.sum()))
                rec.setdefault("bh_clean", []).append(int(mask.sum()) == 0)
                rec.setdefault("q_ns", []).append(not het.detected)
                rec.setdefault("q_p", []).append(het.p_value)
                if name == "hte_real":
                    segs = [sg for sg, _, _ in dev]
                    new_idx = segs.index("new")
                    rec.setdefault("new_survives_bh", []).append(
                        bool(mask[new_idx]))
                    rec.setdefault("new_lift", []).append(
                        het.segments[new_idx].lift)
            elif name == "interference_sketch":
                cr = cluster_robust_se(df)
                rec.setdefault("ratio", []).append(cr.ratio)
                rec.setdefault("ratio_gt_115", []).append(cr.ratio > 1.15)
                rec.setdefault("flag_fires", []).append(cr.sutva_flag)
                # lambda=0 control: same machinery, nothing to find
                s0 = _copy_seed(spec, k)
                exp0 = generate_interference(s0, lam=0.0)
                cr0 = cluster_robust_se(exp0.df)
                rec.setdefault("control_ratio", []).append(cr0.ratio)
                rec.setdefault("control_ok", []).append(
                    0.93 < cr0.ratio < 1.10)
        results[spec.name] = rec
    results["_gen_seconds"] = t_gen
    return results


def _summarize(results: dict) -> list[str]:
    lines: list[str] = []
    ok_all = True

    def gate(name: str, cond: bool, detail: str) -> None:
        nonlocal ok_all
        mark = "PASS" if cond else "FAIL"
        if not cond:
            ok_all = False
        lines.append(f"[{mark}] {name:<22} {detail}")

    r = results["overlap_trap"]
    n = len(r["interaction_detected"])
    gate("overlap_trap",
         sum(r["interaction_detected"]) >= 9,
         f"interaction detected {sum(r['interaction_detected'])}/{n} "
         f"(median p={np.median(r['interaction_p']):.2g})")
    gate("overlap_trap",
         sum(r["b0_ok"]) >= 9 and sum(r["b1_ok"]) >= 9,
         f"stratified recover truth: B0 {np.mean(r['b0_est']):+.3f} "
         f"(true +0.020), B1 {np.mean(r['b1_est']):+.3f} (true -0.005)")
    lines.append(f"      (info) pooled naive significant "
                 f"{sum(r['pooled_sig'])}/{n} "
                 f"(mean {np.mean(r['pooled_lift']):+.3f}) — the trap")

    r = results["hte_mirage_trap"]
    n = len(r["bh_clean"])
    gate("hte_mirage_trap",
         sum(r["bh_clean"]) >= 8,
         f"BH-FDR clean {sum(r['bh_clean'])}/{n} "
         f"(raw nominal hits appeared in "
         f"{sum(1 for h in r['raw_hits'] if h > 0)}/{n} seeds — the mirage)")
    gate("hte_mirage_trap",
         sum(r["q_ns"]) >= 8,
         f"Cochran Q not significant {sum(r['q_ns'])}/{n}")

    r = results["hte_real"]
    n = len(r["new_survives_bh"])
    gate("hte_real",
         sum(r["new_survives_bh"]) >= 9,
         f"'new' survives BH-FDR {sum(r['new_survives_bh'])}/{n} "
         f"(mean lift {np.mean(r['new_lift']):+.3f}, true +0.04); "
         f"mean BH survivors/seed {np.mean(r['bh_survivors']):.2f}")

    r = results["interference_sketch"]
    n = len(r["ratio_gt_115"])
    gate("interference_sketch",
         sum(r["ratio_gt_115"]) >= 9,
         f"cluster/naive SE > 1.15 in {sum(r['ratio_gt_115'])}/{n} "
         f"(median ratio {np.median(r['ratio']):.2f})")
    gate("interference_sketch",
         sum(r["flag_fires"]) >= 8,
         f"SUTVA flag fires {sum(r['flag_fires'])}/{n}")
    gate("interference_sketch",
         sum(r["control_ok"]) >= 9,
         f"lambda=0 control ratio in [0.93,1.10] "
         f"{sum(r['control_ok'])}/{n} "
         f"(median {np.median(r['control_ratio']):.3f})")

    lines.append(f"      generation time: {results['_gen_seconds']:.1f}s total")
    lines.append("ALL GREEN" if ok_all else "FAILURES PRESENT")
    return lines


def main() -> None:
    results = calibrate()
    for line in _summarize(results):
        print(line)


if __name__ == "__main__":
    main()
