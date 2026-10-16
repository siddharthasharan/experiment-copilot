"""Eval harness: plant known traps, grade the reviewer.

This grades the *platform* generator's trap scenarios
(`synthetic_data/generator.py`), whose answer keys live in `experiment.json`.
(The legacy per-user generator in `src/experiment_copilot/data.py` is graded
by `scripts/quickstart.py`.)

For each (platform, scenario) experiment directory:
    1. load the answer key (expected verdict + injected traps)
    2. aggregate metrics_daily to per-user sums; join pre-period covariates
    3. run the review checklist, reusing the stats engine where the grain fits:
       SRM on the assignment split, Welch lift on per-user time sums, CUPED
       with the pre-period covariate, a daily p-value trajectory (peeking),
       novelty-decay on per-user-day rows, variance health on per-user sums,
       and the guardrail veto on daily trust & safety counts
    4. assign a verdict using the same ordering as `analyze_experiment`:
       broken randomization first, then safety vetoes, then stability flags
    5. grade the verdict and per-trap flag detection against the answer key

The harness never sees the answer key while reviewing — grading happens
after the verdict is assigned, mirroring how the real agent is evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import (
    GuardrailResult,
    check_effect_decay,
    check_guardrail,
    check_srm,
    check_variance_health,
    cuped,
    estimate_lift,
)

# answer-key verdict string -> harness verdict category
EXPECTED_VERDICT_MAP = {
    "SHIP": "launch",
    "INVALID": "invalid",
    "HOLD": "inconclusive",
    "DO NOT SHIP": "don't launch",
    "INCONCLUSIVE": "inconclusive",
    "NO EFFECT": "don't launch",
}


def load_experiment(exp_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load assignments.csv, metrics_daily.csv and the experiment.json answer key."""
    exp_dir = Path(exp_dir)
    assignments = pd.read_csv(exp_dir / "assignments.csv")
    metrics = pd.read_csv(exp_dir / "metrics_daily.csv")
    answer_key = json.loads((exp_dir / "experiment.json").read_text())
    return assignments, metrics, answer_key


def per_user_frame(assignments: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily rows to one row per user with summed metrics.

    The stats engine's core tests (Welch lift, CUPED, variance health) operate
    on per-user outcomes; daily structure is handled separately by the
    trajectory/decay/guardrail checks.
    """
    agg = metrics.groupby(["user_id", "variant"], as_index=False).agg(
        time_sum=("time_spent_min", "sum"),
        views_sum=("views", "sum"),
        eng_sum=("likes", "sum"),  # likes re-aggregated below with the rest
        comments_sum=("comments", "sum"),
        shares_sum=("shares", "sum"),
        revenue_sum=("ad_revenue_usd", "sum"),
        reports_sum=("reports", "sum"),
        takedowns_sum=("takedowns", "sum"),
    )
    agg["eng_sum"] = agg["eng_sum"] + agg["comments_sum"] + agg["shares_sum"]
    out = agg.merge(
        assignments[["user_id", "variant", "pre_time_spent_min", "pre_views", "pre_engagements"]],
        on=["user_id", "variant"],
        how="left",
        validate="one_to_one",
    )
    return out


@dataclass
class PeekingTrajectory:
    warned: bool
    note: str
    daily_p_values: list[float]
    early_significance_day: int | None
    final_significant: bool


def peeking_trajectory(
    metrics: pd.DataFrame,
    variant_col: str = "variant",
    y_col: str = "time_spent_min",
    day_col: str = "day",
    alpha: float = 0.05,
) -> PeekingTrajectory:
    """What a daily peeker would have seen: p-value on cumulative per-user
    sums through each day. Warns when an early interim look would have
    produced a significant call the full-horizon result does not support —
    the signature of a false ship from uncorrected sequential looks (or,
    equivalently, a transient early effect such as novelty)."""
    days = sorted(metrics[day_col].unique())
    p_traj: list[float] = []
    for d in days:
        sub = metrics[metrics[day_col] <= d]
        per_user = sub.groupby(["user_id", variant_col], as_index=False)[y_col].sum()
        ctrl = per_user.loc[per_user[variant_col] == "control", y_col].to_numpy()
        treat = per_user.loc[per_user[variant_col] == "treatment", y_col].to_numpy()
        r = estimate_lift(ctrl, treat, alpha=alpha)
        p_traj.append(r.p_value)

    final_sig = p_traj[-1] < alpha
    early_days = [i for i, p in enumerate(p_traj) if p < alpha]
    early_day = early_days[0] if early_days else None
    warned = early_day is not None and not final_sig
    note = ""
    if warned:
        note = (
            f"a daily peeker would have called this significant on day {days[early_day]}, "
            "but the full-horizon result is not significant"
        )
    elif early_day is not None and final_sig:
        note = (
            f"significant from day {days[early_day]} through the final day — "
            "stable; interim looks would not have changed the call"
        )
    return PeekingTrajectory(
        warned=bool(warned),
        note=note,
        daily_p_values=p_traj,
        early_significance_day=int(days[early_day]) if early_day is not None else None,
        final_significant=bool(final_sig),
    )


@dataclass
class PlatformReview:
    scenario: str
    platform: str
    srm_detected: bool
    srm_p_value: float
    actual_allocation: float
    lift: float
    lift_ci: tuple[float, float]
    lift_p_value: float
    lift_significant: bool
    cuped_lift: float
    cuped_variance_reduction: float
    peeking: PeekingTrajectory
    decay_warned: bool
    decay_early_lift: float
    decay_late_lift: float
    variance_warned: bool
    variance_std_ratio: float
    guardrail: GuardrailResult
    verdict: str
    flags: list[str] = field(default_factory=list)


def review_platform_experiment(
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    scenario_name: str | None = None,
) -> PlatformReview:
    """Run the full review checklist on one platform-generator experiment."""
    platform = scenario_name.split("_")[0] if scenario_name else "n/a"
    users = per_user_frame(assignments, metrics)

    n_ctrl = int((assignments["variant"] == "control").sum())
    n_treat = int((assignments["variant"] == "treatment").sum())
    srm = check_srm(n_ctrl, n_treat, expected_allocation=0.5)

    ctrl = users.loc[users["variant"] == "control", "time_sum"].to_numpy()
    treat = users.loc[users["variant"] == "treatment", "time_sum"].to_numpy()
    lift = estimate_lift(ctrl, treat)

    cup = cuped(users, group_col="variant", y_col="time_sum", pre_col="pre_time_spent_min")

    peeking = peeking_trajectory(metrics)
    decay = check_effect_decay(
        metrics, group_col="variant", y_col="time_spent_min", day_col="day"
    )
    # Platform engagement metrics are heavy-tailed (lognormal), so the
    # default 5-SD share rule misfires on healthy tails (~0.3% beyond 5 SD
    # is normal here). Calibrate to the std-ratio rule instead: 50%+ std
    # inflation on the treatment arm is the reliable whale signature.
    var_health = check_variance_health(
        users,
        group_col="variant",
        y_col="time_sum",
        std_ratio_threshold=1.5,
        outlier_share_threshold=0.01,
    )
    # Veto on the primary safety metric only, requiring both statistical
    # (p < 0.01) and practical (>10% lift) significance: rare-event counts
    # wiggle, and a veto is a strong claim. Takedowns are downstream of
    # reports (conditional on them) — testing both doubles the false veto
    # rate for no extra signal.
    guardrail = check_guardrail(metrics, metrics=("reports",), alpha=0.01, min_lift=0.10)

    flags: list[str] = []
    verdict = "inconclusive"

    # Order matters: broken randomization first, then safety, then stability.
    if srm.detected:
        flags.append(
            f"SRM DETECTED (p={srm.p_value:.2e}, actual treatment share "
            f"{srm.actual_allocation:.3f} vs expected {srm.expected_allocation:.3f}). "
            "Randomization is broken — do not trust any lift estimate."
        )
        verdict = "invalid"

    if guardrail.veto:
        flags.append(guardrail.note)

    if peeking.warned:
        flags.append(f"PEEKING WARNING: {peeking.note}.")
    if decay.warned:
        flags.append(
            f"NOVELTY WARNING: lift was {decay.early_lift:+.1%} in the first third of "
            f"the run but only {decay.late_lift:+.1%} in the last third "
            f"(p={decay.late_p_value:.3g}). The headline number is transient — "
            "extend the run past the novelty window before deciding."
        )
    if var_health.warned:
        flags.append(
            f"VARIANCE WARNING: treatment std is {var_health.std_ratio:.1f}x control's "
            f"({var_health.outlier_share_treatment:.2%} of treatment beyond 5 SD). "
            "The mean is fragile — investigate outliers before any launch decision."
        )

    if verdict != "invalid":
        if guardrail.veto:
            # Safety regressions are not tradable against primary wins.
            verdict = "don't launch"
        elif flags:
            verdict = "inconclusive"
        elif lift.significant and lift.lift > 0 and lift.ci_low > 0:
            verdict = "launch"
        else:
            verdict = "don't launch"

    return PlatformReview(
        scenario=scenario_name or "n/a",
        platform=platform,
        srm_detected=srm.detected,
        srm_p_value=srm.p_value,
        actual_allocation=srm.actual_allocation,
        lift=lift.lift,
        lift_ci=(lift.ci_low, lift.ci_high),
        lift_p_value=lift.p_value,
        lift_significant=lift.significant,
        cuped_lift=cup.lift,
        cuped_variance_reduction=cup.variance_reduction,
        peeking=peeking,
        decay_warned=decay.warned,
        decay_early_lift=decay.early_lift,
        decay_late_lift=decay.late_lift,
        variance_warned=var_health.warned,
        variance_std_ratio=var_health.std_ratio,
        guardrail=guardrail,
        verdict=verdict,
        flags=flags,
    )


# injected trap name -> how to check the review caught it
TRAP_CHECKS = {
    "srm": lambda r: r.srm_detected,
    "novelty_decay": lambda r: r.decay_warned,
    "guardrail_violation": lambda r: r.guardrail.veto,
    "whale_variance": lambda r: r.variance_warned,
}


def expected_verdict_category(answer_key: dict) -> str:
    expected = answer_key.get("expected_verdict", "")
    for prefix, category in EXPECTED_VERDICT_MAP.items():
        if expected.startswith(prefix):
            return category
    return "unknown"


@dataclass
class GradedScenario:
    scenario: str
    platform: str
    verdict: str
    expected_verdict: str
    verdict_ok: bool
    traps: dict[str, bool]          # injected trap -> caught?
    all_traps_caught: bool
    passed: bool
    lift: float
    lift_p_value: float
    n_flags: int


def grade(review: PlatformReview, answer_key: dict) -> GradedScenario:
    expected = expected_verdict_category(answer_key)
    verdict_ok = review.verdict == expected
    traps = {
        trap: bool(TRAP_CHECKS[trap](review))
        for trap in answer_key.get("injected_traps", [])
        if trap in TRAP_CHECKS
    }
    all_traps_caught = all(traps.values()) if traps else True
    return GradedScenario(
        scenario=review.scenario,
        platform=review.platform,
        verdict=review.verdict,
        expected_verdict=expected,
        verdict_ok=verdict_ok,
        traps=traps,
        all_traps_caught=all_traps_caught,
        passed=bool(verdict_ok and all_traps_caught),
        lift=review.lift,
        lift_p_value=review.lift_p_value,
        n_flags=len(review.flags),
    )


def review_summary(review: PlatformReview) -> str:
    lines = [
        f"Scenario: {review.scenario}",
        f"Verdict: {review.verdict.upper()}",
        "",
        f"SRM: {'DETECTED' if review.srm_detected else 'ok'} "
        f"(p={review.srm_p_value:.2e}, treatment share={review.actual_allocation:.3f})",
        f"Lift (time/user): {review.lift:+.2%} "
        f"(95% CI [{review.lift_ci[0]:+.2%}, {review.lift_ci[1]:+.2%}], "
        f"p={review.lift_p_value:.3g})",
        f"CUPED lift: {review.cuped_lift:+.2%} "
        f"(variance reduction {review.cuped_variance_reduction:+.1%})",
        f"Guardrail: {'VETO — ' + (review.guardrail.metric or '') if review.guardrail.veto else 'ok'}",
        f"Peeking: {'WARNED' if review.peeking.warned else 'ok'}; "
        f"novelty decay: {'WARNED' if review.decay_warned else 'ok'} "
        f"({review.decay_early_lift:+.1%} -> {review.decay_late_lift:+.1%}); "
        f"variance: {'WARNED' if review.variance_warned else 'ok'} "
        f"(std ratio {review.variance_std_ratio:.2f})",
    ]
    if review.flags:
        lines += ["", "Flags:"]
        lines += [f"  ! {f}" for f in review.flags]
    return "\n".join(lines)


def render_scorecard(graded: list[GradedScenario]) -> str:
    lines = [
        "=" * 78,
        "EVAL SCORECARD — did the reviewer catch the traps?",
        "-" * 78,
        f"{'scenario':24s} {'verdict':12s} {'expected':12s} {'traps caught':20s} {'':6s}",
        "-" * 78,
    ]
    for g in graded:
        traps = (
            ", ".join(f"{t}={'Y' if c else 'N'}" for t, c in g.traps.items())
            or "— (none injected)"
        )
        mark = "PASS" if g.passed else "FAIL"
        lines.append(
            f"{g.scenario:24s} {g.verdict:12s} {g.expected_verdict:12s} "
            f"{traps:20s} [{mark}]"
        )
    lines.append("-" * 78)
    # per-trap detection matrix
    trap_names = sorted({t for g in graded for t in g.traps})
    for trap in trap_names:
        caught = sum(1 for g in graded if g.traps.get(trap))
        total = sum(1 for g in graded if trap in g.traps)
        lines.append(f"trap detection — {trap:20s}: {caught}/{total}")
    n_pass = sum(1 for g in graded if g.passed)
    lines.append("-" * 78)
    lines.append(f"{n_pass}/{len(graded)} scenarios handled correctly")
    lines.append("=" * 78)
    return "\n".join(lines)
