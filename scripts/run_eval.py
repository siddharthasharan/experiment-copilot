"""Run the eval harness: generate the platform trap scenarios (if needed),
review each one with the stats engine, and grade the verdicts against the
answer keys in experiment.json.

Usage:
    .venv/bin/python scripts/run_eval.py [--platform tiktok] [--n-users 20000]
                                         [--days 14] [--seed 42] [--regenerate]

Datasets are cached under eval_data/ so re-runs are cheap; pass --regenerate
to rebuild them from scratch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_copilot.eval import (  # noqa: E402
    grade,
    load_experiment,
    render_scorecard,
    review_platform_experiment,
    review_summary,
)


def load_generator():
    """synthetic_data/generator.py is a script, not a package — import by path."""
    path = ROOT / "synthetic_data" / "generator.py"
    spec = importlib.util.spec_from_file_location("platform_generator", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve types via sys.modules — register before exec
    sys.modules["platform_generator"] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_datasets(gen, platform: str, n_users: int, days: int, seed: int,
                    data_dir: Path, regenerate: bool) -> list[Path]:
    exp_dirs: list[Path] = []
    for scenario in gen.SCENARIOS:
        exp_id = f"{platform}_{scenario}_s{seed}"
        exp_dir = data_dir / exp_id
        needed = [exp_dir / "assignments.csv", exp_dir / "metrics_daily.csv",
                  exp_dir / "experiment.json"]
        if regenerate or not all(p.exists() for p in needed):
            print(f"generating {exp_id} ...", flush=True)
            assignments, metrics, key = gen.generate_experiment(
                platform, scenario, n_users, days, seed
            )
            gen.save_experiment(assignments, metrics, key, exp_dir)
        exp_dirs.append(exp_dir)
    return exp_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval harness for the experiment reviewer.")
    ap.add_argument("--platform", default="tiktok",
                    help="platform profile from the generator (default: tiktok)")
    ap.add_argument("--n-users", type=int, default=20000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild datasets even if cached")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the scorecard, not per-scenario reviews")
    args = ap.parse_args()

    gen = load_generator()
    if args.platform not in gen.PLATFORMS:
        print(f"unknown platform {args.platform!r}; "
              f"choose from {sorted(gen.PLATFORMS)}", file=sys.stderr)
        return 2

    data_dir = ROOT / "eval_data"
    data_dir.mkdir(exist_ok=True)

    exp_dirs = ensure_datasets(gen, args.platform, args.n_users, args.days,
                               args.seed, data_dir, args.regenerate)

    graded = []
    for exp_dir in exp_dirs:
        assignments, metrics, answer_key = load_experiment(exp_dir)
        scenario = answer_key["scenario"]
        review = review_platform_experiment(
            assignments, metrics,
            scenario_name=f"{args.platform}_{scenario}",
        )
        if not args.quiet:
            print("=" * 78)
            print(review_summary(review))
            print()
        graded.append(grade(review, answer_key))

    print(render_scorecard(graded))

    results_path = data_dir / f"{args.platform}_results_s{args.seed}.json"
    results_path.write_text(json.dumps(
        [
            {
                "scenario": g.scenario,
                "verdict": g.verdict,
                "expected_verdict": g.expected_verdict,
                "verdict_ok": g.verdict_ok,
                "traps": g.traps,
                "all_traps_caught": g.all_traps_caught,
                "passed": g.passed,
                "lift": g.lift,
                "lift_p_value": g.lift_p_value,
                "n_flags": g.n_flags,
            }
            for g in graded
        ],
        indent=2,
    ))
    print(f"\nresults written to {results_path.relative_to(ROOT)}")

    return 0 if all(g.passed for g in graded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
