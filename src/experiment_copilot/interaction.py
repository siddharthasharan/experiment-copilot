"""
Interaction effects: overlapping experiments, segment HTE with FDR control,
and a network-interference (SUTVA) sketch.

The three ways a clean A/B read goes wrong *after* randomization checks pass:

  overlap:      a concurrent experiment B interacts with A. The pooled
                A-only analysis averages over B arms and misleads.
  hte_mirage:   13 segments, someone goes p-hacking. Nominal hits are
                expected by chance; BH-FDR and a global heterogeneity test
                keep you honest.
  hte_real:     heterogeneity is real and concentrated. The machinery must
                find it while still controlling FDR.
  interference: user-level randomization with cluster spillover. The point
                estimate is fine; the i.i.d. standard error is not.

Design notes:
  - Heterogeneity is tested with Cochran's Q over per-segment Welch lifts
    (the meta-analytic standard; asymptotically the interaction F-test).
    Counting "significant segments" is the wrong test — a small segment can
    miss significance from low power alone while the effect is real.
  - "Which segments differ" uses leave-one-out pooled contrasts + BH-FDR
    at q=0.10 (exploratory standard; Bonferroni is needlessly conservative
    for discovery and would bury real but modest heterogeneity).
  - The cluster-robust SE is the Liang-Zeger sandwich for a difference in
    means: per-cluster influences summed within arm. Sketch-level: it
    assumes clusters are independent (no cross-cluster spillover).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .stats import LiftResult, SegmentResult, estimate_lift, segment_breakdown


# --- heterogeneity ----------------------------------------------------------

@dataclass
class HeterogeneityResult:
    segments: list[SegmentResult]
    cochran_q: float
    q_df: int
    p_value: float
    detected: bool
    note: str = ""


def _se_of(seg: SegmentResult) -> float:
    return (seg.ci_high - seg.ci_low) / (2 * 1.96)


def cochran_q(segments: list[SegmentResult]) -> tuple[float, int, float]:
    """Cochran's Q heterogeneity statistic over per-segment lifts.

    Q = sum w_i (theta_i - theta_bar)^2, w_i = 1/SE_i^2, ~ chi2(k-1)
    under homogeneity. Returns (Q, df, p_value).
    """
    thetas, ses = [], []
    for s in segments:
        se = _se_of(s)
        if se > 0 and np.isfinite(se):
            thetas.append(s.lift)
            ses.append(se)
    k = len(thetas)
    if k < 2:
        return 0.0, 0, 1.0
    thetas = np.array(thetas)
    w = 1.0 / np.array(ses) ** 2
    theta_bar = float(np.sum(w * thetas) / np.sum(w))
    q = float(np.sum(w * (thetas - theta_bar) ** 2))
    df = k - 1
    p = float(1.0 - sstats.chi2.cdf(q, df))
    return q, df, p


def interaction_test(df: pd.DataFrame,
                     group_col: str = "group",
                     y_col: str = "engagement",
                     segment_col: str = "segment",
                     alpha: float = 0.05) -> HeterogeneityResult:
    """Segment x treatment heterogeneity: per-segment lifts + global Q test."""
    segs = segment_breakdown(df, group_col=group_col, y_col=y_col,
                             segment_col=segment_col, alpha=alpha)
    q, df_q, p = cochran_q(segs)
    detected = bool(p < alpha) if df_q > 0 else False
    note = ""
    if detected:
        note = (f"Cochran's Q={q:.1f} on {df_q} df (p={p:.3g}): lifts differ "
                f"across segments — decide per segment, not on the average.")
    return HeterogeneityResult(segments=segs, cochran_q=q, q_df=df_q,
                               p_value=p, detected=detected, note=note)


def segment_deviation_pvalues(
        segments: list[SegmentResult]) -> list[tuple[str, float, float]]:
    """Per-segment deviation from the leave-one-out pooled lift.

    For each segment i: z_i = (theta_i - theta_(-i)) / sqrt(SE_i^2 + SE_(-i)^2),
    theta_(-i) the inverse-variance pooled lift over all *other* segments.
    Returns [(segment, z, p_value)]. Under homogeneity each p ~ Uniform(0,1),
    so with ~13 segments a raw p<0.05 somewhere is expected ~half the time —
    that is the mirage; BH-FDR is the correction.
    """
    out: list[tuple[str, float, float]] = []
    thetas = np.array([s.lift for s in segments])
    ses = np.array([_se_of(s) for s in segments])
    valid = np.isfinite(ses) & (ses > 0)
    for i, s in enumerate(segments):
        mask = valid & (np.arange(len(segments)) != i)
        if mask.sum() == 0 or not valid[i]:
            out.append((s.segment, 0.0, 1.0))
            continue
        w = 1.0 / ses[mask] ** 2
        theta_loo = float(np.sum(w * thetas[mask]) / np.sum(w))
        se_loo = float(np.sqrt(1.0 / np.sum(w)))
        se_diff = float(np.sqrt(ses[i] ** 2 + se_loo ** 2))
        z = float((thetas[i] - theta_loo) / se_diff) if se_diff > 0 else 0.0
        p = float(2 * (1 - sstats.norm.cdf(abs(z))))
        out.append((s.segment, z, p))
    return out


def bh_fdr(p_values, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR control.

    Returns a boolean mask (original order): True where the null is
    rejected while holding the expected false-discovery proportion <= q.
    NaN p-values are treated as 1.0 (never rejected). q=0.10 is the
    exploratory standard — Bonferroni (FWER) is overkill for discovery
    and buries real but modest heterogeneity.
    """
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    k_max = int(np.max(np.where(below)[0]))
    mask = np.zeros(m, dtype=bool)
    mask[order[: k_max + 1]] = True
    return mask


# --- overlapping experiments -------------------------------------------------

@dataclass
class OverlapResult:
    lift_a_in_b0: LiftResult
    lift_a_in_b1: LiftResult
    lift_a_pooled: LiftResult
    interaction_lift: float      # (A|B=1) - (A|B=0), in relative-lift units
    interaction_se: float
    interaction_z: float
    interaction_p: float
    detected: bool
    b_rate_in_a0: float
    b_rate_in_a1: float
    note: str = ""


def _rel_se(r: LiftResult) -> float:
    return (r.ci_high - r.ci_low) / (2 * 1.96)


def overlap_analysis(df: pd.DataFrame,
                     group_col: str = "group",
                     y_col: str = "engagement",
                     b_col: str = "exp_b",
                     alpha: float = 0.05) -> OverlapResult:
    """Estimate A's effect stratified by concurrent experiment B's arm.

    The A×B interaction contrast (A|B=1 minus A|B=0) is the decision
    statistic: if it is significant, the pooled A-only read is misleading
    and effects must be reported per B arm. Also reports B's rate within
    each A arm as an assignment sanity check (independent B => ~50/50).
    """
    b = df[b_col].to_numpy()
    g = df[group_col].to_numpy()
    y = df[y_col].to_numpy(dtype=float)

    def _lift(mask_b: np.ndarray) -> LiftResult:
        ctrl = y[(g == "control") & mask_b]
        treat = y[(g == "treatment") & mask_b]
        return estimate_lift(ctrl, treat, alpha=alpha)

    b0 = b == 0
    b1 = b == 1
    r_b0 = _lift(b0)
    r_b1 = _lift(b1)
    r_pool = _lift(b0 | b1)

    se_b0, se_b1 = _rel_se(r_b0), _rel_se(r_b1)
    inter = float(r_b1.lift - r_b0.lift)
    se_inter = float(np.sqrt(se_b0 ** 2 + se_b1 ** 2))
    z = float(inter / se_inter) if se_inter > 0 else 0.0
    p = float(2 * (1 - sstats.norm.cdf(abs(z))))
    detected = bool(p < alpha)

    a0 = g == "control"
    a1 = g == "treatment"
    note = ""
    if detected:
        note = (f"A×B interaction p={p:.3g}: A lifts {r_b0.lift:+.1%} when "
                f"B=control but {r_b1.lift:+.1%} when B=treatment. The pooled "
                f"{r_pool.lift:+.1%} is misleading — report stratified, do not "
                f"ship on the average.")
    return OverlapResult(
        lift_a_in_b0=r_b0, lift_a_in_b1=r_b1, lift_a_pooled=r_pool,
        interaction_lift=inter, interaction_se=se_inter,
        interaction_z=z, interaction_p=p, detected=detected,
        b_rate_in_a0=float(b[a0].mean()) if a0.sum() else float("nan"),
        b_rate_in_a1=float(b[a1].mean()) if a1.sum() else float("nan"),
        note=note,
    )


# --- interference sketch ------------------------------------------------------

@dataclass
class ClusterSEResult:
    naive_se: float          # i.i.d. Welch SE of the mean difference
    cluster_se: float        # Liang-Zeger cluster-robust SE
    ratio: float             # cluster_se / naive_se
    sutva_flag: bool
    n_clusters: int
    note: str = ""


def cluster_robust_se(df: pd.DataFrame,
                      group_col: str = "group",
                      y_col: str = "engagement",
                      cluster_col: str = "cluster_id",
                      flag_threshold: float = 1.25) -> ClusterSEResult:
    """Sketch-level SUTVA check: cluster-robust vs naive SE.

    Sandwich (Liang-Zeger) variance for a difference in means: within each
    arm, Var(mu_hat) = sum_c psi_gc^2 / N_g^2 with cluster influences
    psi_gc = S_gc - mu_hat_g * n_gc. If outcomes spill over within clusters
    (shared shocks, treated-mate effects), the i.i.d. SE understates
    uncertainty and the ratio exceeds 1. Sketch-level: assumes clusters are
    independent — no cross-cluster spillover modeled.
    """
    g = df[group_col].to_numpy()
    y = df[y_col].to_numpy(dtype=float)
    cl = df[cluster_col].to_numpy()

    yt = y[g == "treatment"]
    yc = y[g == "control"]
    naive = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))

    def _arm_var(mask: np.ndarray) -> float:
        yy, cc = y[mask], cl[mask]
        mu = yy.mean()
        keys, inv = np.unique(cc, return_inverse=True)
        sums = np.bincount(inv, weights=yy)
        ns = np.bincount(inv).astype(float)
        psi = sums - mu * ns
        return float(np.sum(psi ** 2) / len(yy) ** 2)

    var_t = _arm_var(g == "treatment")
    var_c = _arm_var(g == "control")
    cluster_se = float(np.sqrt(var_t + var_c))
    ratio = float(cluster_se / naive) if naive > 0 else float("nan")
    flag = bool(ratio > flag_threshold)
    note = ""
    if flag:
        note = (f"cluster-robust SE is {ratio:.2f}x the naive i.i.d. SE: "
                f"outcomes are correlated within clusters — SUTVA is at risk "
                f"and the naive p-value overstates precision.")
    return ClusterSEResult(
        naive_se=naive, cluster_se=cluster_se, ratio=ratio,
        sutva_flag=flag, n_clusters=int(len(np.unique(cl))), note=note)
