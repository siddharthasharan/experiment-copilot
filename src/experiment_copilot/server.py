"""Experiment Copilot — an MCP server that reviews A/B tests like a statistician.

Tools:
    analyze_experiment  Run the full review checklist on experiment results.
    design_review       Critique an experiment plan *before* it launches.
    power_analysis      Sample-size calculation for a proportions test.
"""

from __future__ import annotations

import json

import pandas as pd
import fastmcp
from fastmcp import FastMCP

# A local stdio server should never phone home to PyPI on launch: it slows
# startup, leaks nothing useful, and breaks entirely in sandboxed or proxied
# environments (httpx chokes on some NO_PROXY formats). Off, deliberately.
fastmcp.settings.check_for_updates = "off"

from .stats import analyze_experiment as _analyze, review_summary, sample_size_proportions

mcp = FastMCP("experiment-copilot")


@mcp.tool()
def analyze_experiment(
    csv_path: str,
    group_col: str = "group",
    y_col: str = "converted",
    pre_col: str = "pre_metric",
    segment_col: str = "segment",
    expected_allocation: float = 0.5,
    n_interim_looks: int = 0,
) -> str:
    """Review experiment results from a CSV of user-level rows.

    Expected columns: a group column with values "control"/"treatment",
    an outcome column, optionally a pre-period covariate, a segment column,
    and a "day" column for time-based checks.

    Returns the full checklist (SRM, lift, CUPED, segments, peeking/novelty/
    variance flags) and a launch verdict. Pass n_interim_looks if the team
    monitored the experiment with interim decisions — uncorrected sequential
    looks are flagged explicitly.
    """
    df = pd.read_csv(csv_path)
    review = _analyze(
        df,
        group_col=group_col,
        y_col=y_col,
        pre_col=pre_col,
        segment_col=segment_col,
        expected_allocation=expected_allocation,
        scenario_name=csv_path,
        n_interim_looks=n_interim_looks,
    )
    return review_summary(review)


@mcp.tool()
def design_review(plan_json: str) -> str:
    """Critique an experiment design before launch.

    plan_json: JSON object with any of these fields —
        hypothesis, primary_metric, secondary_metrics, allocation,
        duration_days, n_interim_looks, guardrail_metrics,
        pre_registered (bool), expected_effect, segment_plan.

    Returns what's missing or risky, in the order that matters.
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return f"Could not parse plan JSON: {e}"

    issues: list[str] = []

    if not plan.get("hypothesis"):
        issues.append("BLOCKER: no hypothesis stated. Write the one-sentence causal claim first.")
    if not plan.get("primary_metric"):
        issues.append("BLOCKER: no primary metric. Exactly one decision metric must be pre-registered.")
    elif not plan.get("pre_registered"):
        issues.append("RISK: metric not pre-registered. Decide the primary metric *before* launch, or the analysis is exploratory.")
    if not plan.get("guardrail_metrics"):
        issues.append("RISK: no guardrail metrics. Every launch needs metrics that would veto it (e.g., latency, retention, revenue).")
    if plan.get("n_interim_looks", 0) and not plan.get("sequential_method"):
        issues.append(
            f"RISK: {plan['n_interim_looks']} interim looks planned with no sequential method. "
            "Use alpha-spending (e.g., O'Brien-Fleming) or a fixed horizon — otherwise the p-value is invalid."
        )
    alloc = plan.get("allocation", 0.5)
    if alloc != 0.5 and not plan.get("allocation_rationale"):
        issues.append(
            f"RISK: unequal allocation ({alloc:.0%} treatment) with no stated rationale. "
            "Unequal splits need a reason (risk, capacity); they also cost power."
        )
    if not plan.get("duration_days"):
        issues.append("RISK: no duration. Size the run from the MDE and weekly seasonality, not from vibes.")
    if not plan.get("segment_plan"):
        issues.append("NOTE: no segment plan. Decide up front which segments you'll check, so post-hoc slicing doesn't become p-hacking.")
    if plan.get("expected_effect") and plan.get("sample_size"):
        issues.append(
            "NOTE: expected effect and sample size both given — verify the sample size covers the MDE at 80% power via power_analysis."
        )

    if not issues:
        return (
            "Design looks solid: hypothesis, pre-registered primary metric, guardrails, "
            "no uncorrected interim looks, and a segment plan. Ship it."
        )
    return "Design review:\n" + "\n".join(f"- {i}" for i in issues)


@mcp.tool()
def power_analysis(
    base_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> str:
    """Per-group sample size to detect a relative MDE on a conversion rate.

    Example: base_rate=0.10, mde_relative=0.05 → users per group needed to
    detect a +5% relative lift with 80% power at 95% confidence.
    """
    n = sample_size_proportions(base_rate, mde_relative, alpha=alpha, power=power)
    return (
        f"To detect a {mde_relative:+.1%} relative move on a {base_rate:.1%} base rate "
        f"(alpha={alpha}, power={power}): {n:,} users per group, {2*n:,} total."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
