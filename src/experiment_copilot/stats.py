"""
The stats engine: every check a rigorous experiment review runs, as pure
Python functions with no side effects.

Checklist implemented:
    - SRM detection (chi-square goodness-of-fit on the allocation ratio)
    - Lift estimation: Welch's t-test with confidence interval, relative lift
    - CUPED variance reduction using a pre-period covariate
    - Segment breakdowns (does the effect hold across segments?)
    - Sequential-look warning (peeking): daily p-value trajectory, flagging
      when significance appears early and is unstable
    - Power / sample-size analysis for proportions
    - Outlier/variance health check on the metric
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------------

@dataclass
class SRMResult:
    detected: bool
    p_value: float
    actual_allocation: float
    expected_allocation: float
    n_control: int
    n_treatment: int


def check_srm(
    n_control: int,
    n_treatment: int,
    expected_allocation: float = 0.5,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit of the observed split vs the intended split.

    Uses a strict default alpha (0.001): with large samples the test is very
    powerful, and a marginal imbalance rarely invalidates anything — but a
    hard mismatch means something is broken upstream (bad randomization,
    triggered-analysis bugs, bot traffic).
    """
    n = n_control + n_treatment
    observed = np.array([n_control, n_treatment], dtype=float)
    expected = np.array([n * (1 - expected_allocation), n * expected_allocation])
    chi2, p = stats.chisquare(observed, expected)
    return SRMResult(
        detected=bool(p < alpha),
        p_value=float(p),
        actual_allocation=float(n_treatment / n),
        expected_allocation=expected_allocation,
        n_control=n_control,
        n_treatment=n_treatment,
    )


@dataclass
class LiftResult:
    lift: float              # relative lift, treatment vs control
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    control_mean: float
    treatment_mean: float


def estimate_lift(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> LiftResult:
    """Welch's t-test with a confidence interval on the *relative* lift.

    Welch's (unequal-variance) t is the right default — real experiments
    rarely have equal variance, and Student's t silently assumes it.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    cm, tm = control.mean(), treatment.mean()
    t_stat, p = stats.ttest_ind(treatment, control, equal_var=False)
    diff = tm - cm
    se = math_sqrt_var(control, treatment)
    df = _welch_df(control, treatment)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_abs = (diff - t_crit * se, diff + t_crit * se)
    lift = diff / cm if cm != 0 else float("nan")
    ci = (ci_abs[0] / cm, ci_abs[1] / cm) if cm != 0 else (float("nan"), float("nan"))
    return LiftResult(
        lift=float(lift),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        p_value=float(p),
        significant=bool(p < alpha),
        control_mean=float(cm),
        treatment_mean=float(tm),
    )


def math_sqrt_var(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))


@dataclass
class CUPEDResult:
    lift: float
    lift_unadjusted: float
    variance_reduction: float   # 1 - var(adj)/var(raw); >0 means it helped
    theta: float


def cuped(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "converted",
    pre_col: str = "pre_metric",
) -> CUPEDResult:
    """CUPED: regress out the pre-period covariate to shrink variance.

    theta = Cov(Y, X) / Var(X), computed on control (or pooled) so the
    adjustment doesn't absorb the treatment effect itself. Standard practice
    from Deng, Hu, Kohavi et al.
    """
    treat = df[group_col] == "treatment"
    y = df[y_col].to_numpy(dtype=float)
    x = df[pre_col].to_numpy(dtype=float)
    ctrl = ~treat.to_numpy()

    theta = float(np.cov(y[ctrl], x[ctrl])[0, 1] / np.var(x[ctrl]))
    y_adj = y - theta * (x - x.mean())
    raw = estimate_lift(y[ctrl], y[treat.to_numpy()])
    adj = estimate_lift(y_adj[ctrl], y_adj[treat.to_numpy()])
    var_red = 1.0 - (y_adj.var(ddof=1) / y.var(ddof=1))
    return CUPEDResult(
        lift=adj.lift,
        lift_unadjusted=raw.lift,
        variance_reduction=float(var_red),
        theta=theta,
    )


@dataclass
class SegmentResult:
    segment: str
    n_control: int
    n_treatment: int
    lift: float
    ci_low: float
    ci_high: float
    p_value: float


def _se_from_ci(ci_low: float, ci_high: float) -> float:
    return (ci_high - ci_low) / (2 * 1.96)


def check_heterogeneity(segments: list[SegmentResult], alpha: float = 0.05) -> str | None:
    """Interaction test: do any two segments have *significantly different* lifts?

    Counting significant segments is the wrong test — a small segment can fail
    to reach significance from low power alone while the effect is real. What
    matters is whether the lifts themselves disagree.
    """
    if len(segments) < 2:
        return None
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = segments[i], segments[j]
            diff = a.lift - b.lift
            se = float(np.sqrt(_se_from_ci(a.ci_low, a.ci_high) ** 2 + _se_from_ci(b.ci_low, b.ci_high) ** 2))
            if se == 0:
                continue
            p = float(2 * (1 - stats.norm.cdf(abs(diff / se))))
            if p < alpha:
                return (
                    f"lift differs across segments: {a.segment} {a.lift:+.1%} vs "
                    f"{b.segment} {b.lift:+.1%} (interaction p={p:.3g}). "
                    "The average effect hides real disagreement — decide per segment."
                )
    return None


def segment_breakdown(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "converted",
    segment_col: str = "segment",
    alpha: float = 0.05,
) -> list[SegmentResult]:
    """Per-segment lifts. Heterogeneity is where bad launches hide."""
    out: list[SegmentResult] = []
    for seg, sub in df.groupby(segment_col):
        ctrl = sub.loc[sub[group_col] == "control", y_col].to_numpy()
        treat = sub.loc[sub[group_col] == "treatment", y_col].to_numpy()
        r = estimate_lift(ctrl, treat, alpha=alpha)
        out.append(
            SegmentResult(
                segment=str(seg),
                n_control=len(ctrl),
                n_treatment=len(treat),
                lift=r.lift,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=r.p_value,
            )
        )
    return out


@dataclass
class PeekingResult:
    warned: bool
    note: str
    daily_p_values: list[float]
    early_significance_day: int | None
    final_significant: bool


def check_peeking(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "converted",
    day_col: str = "day",
    alpha: float = 0.05,
) -> PeekingResult:
    """Reconstruct what a daily peeker would have seen, day by day.

    Warns when an early interim look would have produced a significant call
    that the full-horizon result does not support — the signature of a
    false ship from uncorrected sequential looks (or, equivalently, a
    transient early effect such as novelty).
    """
    days = sorted(df[day_col].unique())
    p_traj: list[float] = []
    for d in days:
        sub = df[df[day_col] <= d]
        ctrl = sub.loc[sub[group_col] == "control", y_col].to_numpy()
        treat = sub.loc[sub[group_col] == "treatment", y_col].to_numpy()
        if len(ctrl) < 30 or len(treat) < 30:
            p_traj.append(1.0)
            continue
        _, p = stats.ttest_ind(treat, ctrl, equal_var=False)
        p_traj.append(float(p))

    final_sig = p_traj[-1] < alpha
    early_days = [i for i, p in enumerate(p_traj) if p < alpha]
    early_day = early_days[0] if early_days else None
    warned = early_day is not None and not final_sig
    note = ""
    if warned:
        note = (
            f"a daily peeker would have called this significant on day {days[early_day]}, "
            f"but the full-horizon result is not significant"
        )
    elif early_day is not None and final_sig:
        note = (
            f"significant from day {days[early_day]} through the final day — "
            "stable; interim looks would not have changed the call"
        )
    return PeekingResult(
        warned=bool(warned),
        note=note,
        daily_p_values=p_traj,
        early_significance_day=int(days[early_day]) if early_day is not None else None,
        final_significant=bool(final_sig),
    )


@dataclass
class DecayResult:
    warned: bool
    early_lift: float
    late_lift: float
    late_p_value: float


def check_effect_decay(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "converted",
    day_col: str = "day",
    alpha: float = 0.05,
) -> DecayResult:
    """Compare the lift in the first third of days vs the last third.

    A large early lift that collapses to ~null late is the novelty-effect
    signature: the headline number is real but transient, so launching on
    the full-period average would overstate the steady-state impact.
    """
    days = sorted(df[day_col].unique())
    cut = max(1, len(days) // 3)
    early = df[df[day_col].isin(days[:cut])]
    late = df[df[day_col].isin(days[-cut:])]

    def _lift(sub: pd.DataFrame) -> tuple[float, float]:
        ctrl = sub.loc[sub[group_col] == "control", y_col].to_numpy()
        treat = sub.loc[sub[group_col] == "treatment", y_col].to_numpy()
        r = estimate_lift(ctrl, treat, alpha=alpha)
        return r.lift, r.p_value

    early_lift, early_p = _lift(early)
    late_lift, late_p = _lift(late)
    # Require the early lift to be real (significant), not noise, before
    # calling the pattern a decay — otherwise a null experiment can trip
    # the check on a lucky early window.
    warned = bool(
        early_p < alpha and early_lift > 0.03 and late_lift < early_lift / 2
    )
    return DecayResult(
        warned=warned,
        early_lift=float(early_lift),
        late_lift=float(late_lift),
        late_p_value=float(late_p),
    )


@dataclass
class VarianceHealth:
    warned: bool
    control_std: float
    treatment_std: float
    std_ratio: float
    outlier_share_treatment: float


def check_variance_health(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "post_metric",
    std_ratio_threshold: float = 2.0,
    outlier_share_threshold: float = 0.001,
    sd_multiple: float = 5.0,
) -> VarianceHealth:
    """Flag asymmetric heavy tails — the 'variance bomb' signature.

    A treatment std far above control's, driven by a small share of extreme
    values, means the mean is fragile and the experiment isn't decisionable.

    Thresholds are tunable because healthy metrics differ: a lognormal-ish
    engagement metric naturally puts a few tenths of a percent of users
    beyond 5 control-SDs, so the share rule misfires there and the std-ratio
    rule (default bar: 50%+ std inflation) is the reliable signal.
    """
    ctrl = df.loc[df[group_col] == "control", y_col].to_numpy(dtype=float)
    treat = df.loc[df[group_col] == "treatment", y_col].to_numpy(dtype=float)
    ratio = float(treat.std(ddof=1) / ctrl.std(ddof=1)) if ctrl.std() else float("nan")
    # share of treatment values beyond sd_multiple control-SDs from the control mean
    cutoff = ctrl.mean() + sd_multiple * ctrl.std(ddof=1)
    outlier_share = float((treat > cutoff).mean())
    warned = bool(ratio > std_ratio_threshold or outlier_share > outlier_share_threshold)
    return VarianceHealth(
        warned=warned,
        control_std=float(ctrl.std(ddof=1)),
        treatment_std=float(treat.std(ddof=1)),
        std_ratio=ratio,
        outlier_share_treatment=outlier_share,
    )


@dataclass
class GuardrailResult:
    veto: bool
    metric: str | None
    lift: float              # relative lift of treatment vs control on the metric
    p_value: float | None
    note: str


def check_guardrail(
    df: pd.DataFrame,
    variant_col: str = "variant",
    day_col: str = "day",
    metrics: tuple[str, ...] = ("reports", "takedowns"),
    alpha: float = 0.05,
    min_lift: float = 0.0,
) -> GuardrailResult:
    """Veto check for trust & safety (or any 'must not regress') metrics.

    Aggregates each metric to daily per-variant totals and runs a *paired*
    t-test across days (one-sided: treatment worse than control). The daily
    grain is deliberate: guardrail events are rare, and per-user tests on
    sparse Bernoulli outcomes are underpowered with fragile p-values, while
    daily totals are approximately Poisson/normal and pair naturally by day.

    A veto needs both statistical significance (p < alpha) and practical
    significance (lift > min_lift): a veto is a strong claim, and tiny but
    'significant' wiggles in rare-event counts deserve investigation, not an
    automatic block. Regressions above that bar veto the launch — safety is
    not tradable against primary-metric wins.
    """
    days = sorted(df[day_col].unique())
    if len(days) < 3:
        return GuardrailResult(
            veto=False, metric=None, lift=0.0, p_value=None,
            note="too few days for a guardrail read",
        )
    for metric in metrics:
        if metric not in df.columns:
            continue
        piv = df.groupby([day_col, variant_col])[metric].sum().unstack(variant_col)
        if "control" not in piv.columns or "treatment" not in piv.columns:
            continue
        c = piv["control"].to_numpy(dtype=float)
        t = piv["treatment"].to_numpy(dtype=float)
        res = stats.ttest_rel(t, c, alternative="greater")
        p = float(res.pvalue)
        lift = float(t.mean() / c.mean() - 1.0) if c.mean() > 0 else float("nan")
        if p < alpha and lift > min_lift:
            return GuardrailResult(
                veto=True,
                metric=metric,
                lift=lift,
                p_value=p,
                note=(
                    f"GUARDRAIL BREACH: {metric} up {lift:+.1%} on treatment "
                    f"(paired daily t-test, one-sided p={p:.3g}). Safety "
                    "regressions veto the launch regardless of the primary metric."
                ),
            )
    return GuardrailResult(veto=False, metric=None, lift=0.0, p_value=None, note="")


def sample_size_proportions(
    base_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-group sample size for a two-sided test of proportions (normal approx)."""
    p1 = base_rate
    p2 = base_rate * (1 + mde_relative)
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


# ----------------------------------------------------------------------------
# Full review: the checklist, in order
# ----------------------------------------------------------------------------

@dataclass
class ExperimentReview:
    scenario: str | None
    srm: SRMResult
    lift: LiftResult
    cuped: CUPEDResult
    segments: list[SegmentResult] = field(default_factory=list)
    peeking: PeekingResult | None = None
    decay: DecayResult | None = None
    variance_health: VarianceHealth | None = None
    verdict: str = "inconclusive"
    flags: list[str] = field(default_factory=list)


def analyze_experiment(
    df: pd.DataFrame,
    group_col: str = "group",
    y_col: str = "converted",
    pre_col: str = "pre_metric",
    segment_col: str = "segment",
    expected_allocation: float = 0.5,
    scenario_name: str | None = None,
    n_interim_looks: int = 0,
) -> ExperimentReview:
    """Run the full review checklist and return a verdict.

    Order matters: SRM first (nothing downstream is trustworthy if the split
    is broken), then estimation, then stability and health checks. The verdict
    is a recommendation with stated caveats, never a bare 'significant'.

    Pass n_interim_looks when the team monitored the experiment with interim
    decisions — uncorrected sequential looks are flagged explicitly, because
    real peeking is a process fact, not something inferable from the final
    CSV alone.
    """
    n_ctrl = int((df[group_col] == "control").sum())
    n_treat = int((df[group_col] == "treatment").sum())

    srm = check_srm(n_ctrl, n_treat, expected_allocation)
    lift = estimate_lift(
        df.loc[df[group_col] == "control", y_col].to_numpy(),
        df.loc[df[group_col] == "treatment", y_col].to_numpy(),
    )
    cup = cuped(df, group_col, y_col, pre_col)
    segments = segment_breakdown(df, group_col, y_col, segment_col)

    flags: list[str] = []
    verdict = "inconclusive"

    if srm.detected:
        flags.append(
            f"SRM DETECTED (p={srm.p_value:.2e}, "
            f"actual split {srm.actual_allocation:.3f} vs expected {srm.expected_allocation:.3f}). "
            "Randomization is broken — do not trust any lift estimate. Investigate assignment "
            "logging, bot traffic, and triggered-analysis bugs before re-running."
        )
        verdict = "invalid"

    # These checks run even when SRM fired — the flags document everything.
    peeking = check_peeking(df, group_col, y_col)
    if peeking.warned:
        flags.append(
            f"PEEKING WARNING: {peeking.note}. "
            "Interim looks without sequential correction invalidate the p-value; "
            "re-run with a pre-registered horizon or alpha-spending."
        )
    elif peeking.note and not peeking.final_significant:
        flags.append(f"PEEKING NOTE: {peeking.note}.")

    decay = check_effect_decay(df, group_col, y_col)
    if decay.warned:
        flags.append(
            f"NOVELTY WARNING: lift was {decay.early_lift:+.1%} in the first third of "
            f"the run but only {decay.late_lift:+.1%} in the last third "
            f"(p={decay.late_p_value:.3g}). The headline number is transient — "
            "extend the run past the novelty window before deciding."
        )

    if n_interim_looks > 0 and lift.significant:
        # Bonferroni is conservative, which is the point for a first-pass flag:
        # it asks "would this survive even the crudest correction?"
        if lift.p_value * n_interim_looks >= 0.05:
            flags.append(
                f"SEQUENTIAL-TESTING WARNING: significant at p={lift.p_value:.3g}, but "
                f"the team made ~{n_interim_looks} interim looks with no correction. "
                "Not significant under Bonferroni — treat as inconclusive or "
                "re-analyze with alpha-spending."
            )

    var_health = None
    if "post_metric" in df.columns:
        var_health = check_variance_health(df, group_col, "post_metric")
        if var_health.warned:
            flags.append(
                f"VARIANCE WARNING: treatment std is {var_health.std_ratio:.1f}x control's "
                f"({var_health.outlier_share_treatment:.2%} of treatment beyond 5 SD). "
                "The mean is fragile — investigate outliers (logging errors? bots?) "
                "before any launch decision."
            )

    # Segment heterogeneity: test whether lifts actually disagree (interaction),
    # not whether each segment is individually significant — small segments can
    # miss significance from low power alone while the effect is real.
    het = check_heterogeneity(segments)
    if het and lift.significant:
        flags.append(f"HETEROGENEITY: {het}")

    if verdict != "invalid":
        if flags:
            verdict = "inconclusive"
        elif lift.significant and lift.lift > 0 and lift.ci_low > 0:
            verdict = "launch"
        else:
            # Clean run, no evidence to ship — the product decision is "don't".
            # This covers the null and the significant-negative alike.
            verdict = "don't launch"

    # Informational only: a clean null whose CI can't rule out the MDE deserves
    # a longer run, not just a shrug. Appended after the verdict on purpose —
    # it must never flip a decision by itself.
    if verdict == "don't launch" and not lift.significant and lift.ci_high > 0.05:
        flags.append(
            f"NOTE: underpowered for a 5% MDE — the CI admits up to {lift.ci_high:+.1%}. "
            "Don't ship on this, but consider a longer run before closing the idea."
        )

    return ExperimentReview(
        scenario=scenario_name,
        srm=srm,
        lift=lift,
        cuped=cup,
        segments=segments,
        peeking=peeking,
        decay=decay,
        variance_health=var_health,
        verdict=verdict,
        flags=flags,
    )


def review_summary(review: ExperimentReview) -> str:
    """One-page human-readable summary of a review — the memo's fact sheet."""
    lines = [
        f"Scenario: {review.scenario or 'n/a'}",
        f"Verdict: {review.verdict.upper()}",
        "",
        f"SRM: {'DETECTED' if review.srm.detected else 'ok'} "
        f"(p={review.srm.p_value:.2e}, split={review.srm.actual_allocation:.3f})",
        f"Lift: {review.lift.lift:+.2%} "
        f"(95% CI [{review.lift.ci_low:+.2%}, {review.lift.ci_high:+.2%}], "
        f"p={review.lift.p_value:.3g})",
        f"CUPED lift: {review.cuped.lift:+.2%} "
        f"(variance reduction {review.cuped.variance_reduction:+.1%})",
    ]
    if review.decay is not None:
        lines.append(
            f"Early vs late lift: {review.decay.early_lift:+.2%} → {review.decay.late_lift:+.2%}"
        )
    lines += ["", "Segments:"]
    for s in review.segments:
        lines.append(
            f"  - {s.segment}: {s.lift:+.2%} (p={s.p_value:.3g}, "
            f"n={s.n_control + s.n_treatment})"
        )
    if review.flags:
        lines += ["", "Flags:"]
        lines += [f"  ! {f}" for f in review.flags]
    return "\n".join(lines)
