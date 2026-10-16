"""
Synthetic A/B test data generator with trap injection.

Everything here has known ground truth, which is the point: the eval harness
(and you, while debugging) always knows exactly which traps were planted and
where. This makes synthetic data strictly better than real open datasets for
grading an experiment-review agent — you own the answer key.

Traps implemented:
    - "srm":          sample-ratio mismatch (allocation ≠ intended split)
    - "peeking":      sequential interim looks where a naive early call would
                       be wrong; data carries a day-by-day structure
    - "novelty":      inflated treatment effect in the first days that decays
    - "variance_bomb": heavy-tailed outliers in the treatment group

The DataFrame also records the true data-generating parameters so the agent
never has to guess whether its verdict matched reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """A fully specified synthetic experiment scenario."""

    name: str
    n_users: int = 50_000
    base_rate: float = 0.10
    true_lift: float = 0.0            # relative lift of treatment on conversion
    allocation: float = 0.5           # intended fraction assigned to treatment
    traps: tuple[str, ...] = ()       # which traps to inject
    seed: int = 42
    days: int = 14                    # experiment duration in days


@dataclass
class GeneratedExperiment:
    """Data plus its answer key."""

    df: pd.DataFrame
    ground_truth: dict = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _apply_srm(groups: np.ndarray, rng: np.random.Generator, severity: float = 0.55) -> np.ndarray:
    """Re-assign some users so allocation skews to (1-severity, severity)."""
    n = len(groups)
    n_treat_target = int(n * severity)
    treat_idx = rng.choice(n, size=n_treat_target, replace=False)
    new_groups = np.zeros(n, dtype=int)
    new_groups[treat_idx] = 1
    return new_groups


def _apply_novelty(
    outcomes: np.ndarray,
    groups: np.ndarray,
    days: np.ndarray,
    total_days: int,
    rng: np.random.Generator,
    boost: float = 0.05,
) -> np.ndarray:
    """Add a decaying early-days boost to treated users (novelty effect)."""
    treat = groups == 1
    decay = np.exp(-days / (total_days / 5.0))
    lift_prob = boost * decay * treat
    bumped = rng.random(len(outcomes)) < lift_prob
    return np.clip(outcomes + bumped.astype(int), 0, 1)


def _apply_variance_bomb(
    values: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    outlier_frac: float = 0.002,
    outlier_scale: float = 10.0,
) -> np.ndarray:
    """Inject heavy-tailed outliers into the treatment group's metric."""
    out = values.copy()
    treat = groups == 1
    treat_idx = np.flatnonzero(treat)
    n_outliers = max(1, int(len(treat_idx) * outlier_frac))
    chosen = rng.choice(treat_idx, size=n_outliers, replace=False)
    out[chosen] += np.abs(rng.standard_normal(n_outliers)) * outlier_scale * values.std()
    return out


def _daily_structure(df: pd.DataFrame, rng: np.random.Generator, days: int) -> pd.DataFrame:
    """Assign each user an enrollment day and compute cumulative per-day stats.

    This gives the 'peeking' trap its structure: the agent can see what a
    daily peeker would have concluded on each day, which may differ from the
    final-day verdict.
    """
    df = df.copy()
    df["day"] = rng.integers(0, days, size=len(df))
    df.sort_values("day", inplace=True, ignore_index=True)
    return df


def generate_experiment(scenario: Scenario) -> GeneratedExperiment:
    """Generate one synthetic experiment from a scenario.

    Returns tidy user-level rows plus the ground-truth answer key.
    """
    rng = _rng(scenario.seed)
    n = scenario.n_users

    groups = (rng.random(n) < scenario.allocation).astype(int)

    if "srm" in scenario.traps:
        groups = _apply_srm(groups, rng)

    # Pre-period covariate (for CUPED), correlated with outcome.
    pre = rng.normal(loc=50.0, scale=10.0, size=n)

    # Base conversion, heterogeneous by pre-period value.
    logit = math.log(scenario.base_rate / (1 - scenario.base_rate))
    logit += (pre - 50.0) / 10.0 * 0.3
    p = 1.0 / (1.0 + np.exp(-logit))
    converted = (rng.random(n) < p).astype(int)

    # True treatment effect: shift conversion probability relatively.
    treat = groups == 1
    p_treat = np.clip(p * (1 + scenario.true_lift), 0, 1)
    converted[treat] = (rng.random(treat.sum()) < p_treat[treat]).astype(int)

    # Continuous secondary metric (e.g., session minutes), correlated with pre.
    post_metric = pre * 0.5 + rng.normal(loc=30.0, scale=8.0, size=n)
    post_metric[treat] += 2.0 * scenario.true_lift

    # Segment label: new vs returning users.
    segment = np.where(rng.random(n) < 0.3, "new", "returning")

    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "group": np.where(treat, "treatment", "control"),
            "converted": converted,
            "pre_metric": pre,
            "post_metric": post_metric,
            "segment": segment,
        }
    )

    if "novelty" in scenario.traps:
        df = _daily_structure(df, rng, scenario.days)
        df["converted"] = _apply_novelty(
            df["converted"].to_numpy(),
            (df["group"] == "treatment").to_numpy().astype(int),
            df["day"].to_numpy(),
            scenario.days,
            rng,
        )
    else:
        df = _daily_structure(df, rng, scenario.days)

    if "variance_bomb" in scenario.traps:
        df["post_metric"] = _apply_variance_bomb(
            df["post_metric"].to_numpy(),
            (df["group"] == "treatment").to_numpy().astype(int),
            rng,
        )

    # --- answer key -----------------------------------------------------
    actual_allocation = float((df["group"] == "treatment").mean())
    truth: dict = {
        "scenario": scenario.name,
        "n_users": n,
        "intended_allocation": scenario.allocation,
        "actual_allocation": round(actual_allocation, 4),
        "traps": list(scenario.traps),
        "true_lift": scenario.true_lift,
        "seed": scenario.seed,
    }
    # what the *correct* review should conclude — trap branches first, because
    # a broken experiment is never launchable no matter what the lift says
    if "srm" in scenario.traps:
        truth["expected_verdict"] = "invalid"
        truth["expected_flag"] = "SRM detected — do not trust the lift"
    elif "peeking" in scenario.traps:
        # Real peeking is a process fact (dashboard checked daily, decision made
        # on an interim look), not something inferable from the final CSV. So
        # the answer key carries the monitoring history, and the review tool
        # accepts it explicitly (see analyze_experiment(n_interim_looks=...)).
        # The seeded data below is additionally chosen so a naive daily peeker
        # would have falsely shipped it — both paths flag the trap.
        truth["interim_looks"] = 13
        truth["expected_verdict"] = "inconclusive"
        truth["expected_flag"] = "sequential looks without correction — significance not trustworthy"
    elif "novelty" in scenario.traps:
        truth["expected_verdict"] = "inconclusive"
        truth["expected_flag"] = "early effect decays — likely novelty, needs longer run"
    elif "variance_bomb" in scenario.traps:
        truth["expected_verdict"] = "inconclusive"
        truth["expected_flag"] = "variance inflated by outliers — investigate before launching"
    elif scenario.true_lift > 0:
        truth["expected_verdict"] = "launch"
    else:
        truth["expected_verdict"] = "don't launch"

    return GeneratedExperiment(df=df, ground_truth=truth)


def demo_scenarios() -> dict[str, GeneratedExperiment]:
    """The five scenarios that seed the eval harness."""
    specs = [
        Scenario("clean_win", true_lift=0.05, seed=1),
        Scenario("clean_null", true_lift=0.0, seed=2),
        Scenario("srm_trap", true_lift=0.05, traps=("srm",), seed=3),
        Scenario("peeking_trap", true_lift=0.0, traps=("peeking",), seed=71),
        Scenario("novelty_trap", true_lift=0.0, traps=("novelty",), seed=5),
        Scenario("variance_trap", true_lift=0.05, traps=("variance_bomb",), seed=1),
    ]
    return {s.name: generate_experiment(s) for s in specs}
