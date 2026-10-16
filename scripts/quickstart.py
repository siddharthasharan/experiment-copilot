"""Quickstart: generate the trap scenarios, review each one, grade the verdicts.

This is the seed of the eval harness — every run answers: did the agent
catch the trap? Run with the project venv:

    .venv/bin/python scripts/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_copilot import analyze_experiment, demo_scenarios, review_summary  # noqa: E402


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)

    scenarios = demo_scenarios()
    scorecard: list[tuple[str, str, str, bool]] = []

    for name, exp in scenarios.items():
        path = data_dir / f"{name}.csv"
        exp.df.to_csv(path, index=False)

        review = analyze_experiment(
            exp.df,
            scenario_name=name,
            n_interim_looks=exp.ground_truth.get("interim_looks", 0),
        )
        print("=" * 72)
        print(review_summary(review))
        print()

        expected = exp.ground_truth["expected_verdict"]
        # "invalid" and "inconclusive" both mean "did not naively ship it"
        caught = (review.verdict == expected) or (
            expected in {"invalid", "inconclusive"} and review.verdict != "launch"
        )
        scorecard.append((name, review.verdict, expected, caught))

    print("=" * 72)
    print("SCORECARD (agent verdict vs answer key)")
    print("-" * 72)
    for name, got, expected, caught in scorecard:
        mark = "PASS" if caught else "FAIL"
        print(f"[{mark}] {name:16s} verdict={got:12s} expected={expected}")
    print("-" * 72)
    n_pass = sum(1 for s in scorecard if s[3])
    print(f"{n_pass}/{len(scorecard)} scenarios handled correctly")


if __name__ == "__main__":
    main()
