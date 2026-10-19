"""
Step-2 scenarios: triggered vs assigned populations (ITT vs CACE).

Built on top of the v2 baseline generator — same metrics, same 28-day
pre-period, same per-user grain — plus a ``triggered`` flag and
triggered-only effects. Three scenarios, uniform n=400k:

  dilution_trap: +2.5% engagement lift on TRIGGERED ONLY, trigger rate
    ~18% (heterogeneous by tenure). ITT is diluted to ~tau*CACE and
    underpowered -> a naive ITT-only read says "don't launch" and is
    wrong. Correct: launch on CACE, with rollout impact ~= CACE * tau.

  trigger_imbalance_trap: treatment itself boosts triggering (+5pp) and
    triggering correlates with engagement propensity. The triggered
    subpopulations differ by arm -> triggered-only contrasts (including
    the Wald CACE) are selection-biased. The trigger-balance check must
    fire; correct verdict is inconclusive with the selection-bias
    explanation, falling back to ITT.

  triggered_clean_win: +2.5% on triggered only, tau=0.5, balanced
    triggering. Sanity case: CACE recovers the true lift, ITT is diluted
    exactly as theory predicts (ITT ~= tau * CACE), both significant.

Every scenario writes an experiment.json answer key with the true CACE,
per-arm trigger rates, the expected verdict, and which analysis is valid.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_v2 import (N_USERS, V2Experiment, V2Scenario, generate_v2,
                      write_scenario)
from .power import achieved_power, apply_cuped, required_n
from .triggering import add_triggering

TRIGGERED_OUTDIR = "eval_data_v2_triggered"


@dataclass
class TriggeredScenario:
    """A step-2 scenario: v2 baseline + triggering + triggered-only effects."""

    name: str
    seed: int = 201
    n_users: int = N_USERS
    # lift applied to ALL treated users via the v2 mechanism (usually 0.0;
    # triggered-only effects are applied separately below)
    base_engagement_lift: float = 0.0
    # lift applied to triggered treated users ONLY
    triggered_engagement_lift: float = 0.0
    trigger_rate: float = 0.2
    trigger_rate_by: dict[str, dict] | None = None
    trigger_propensity: tuple[str, float] | None = None
    trigger_treat_shift: float = 0.0
    allocation: float = 0.5
    expected_verdict: str = "don't launch"
    expected_flag: str = ""
    valid_analysis: str = "cace"  # cace | itt | both


def generate_triggered(spec: TriggeredScenario) -> V2Experiment:
    """Generate one triggered scenario. Vectorized; ~0.5s at 400k users."""
    base = V2Scenario(
        name=spec.name,
        seed=spec.seed,
        n_users=spec.n_users,
        engagement_lift=spec.base_engagement_lift,
        allocation=spec.allocation,
    )
    exp = generate_v2(base)
    df = exp.df

    rng = np.random.default_rng(spec.seed + 7919)
    df = add_triggering(
        df,
        trigger_rate=spec.trigger_rate,
        rng=rng,
        rate_by=spec.trigger_rate_by,
        propensity=spec.trigger_propensity,
        treat_shift=spec.trigger_treat_shift,
    )

    if spec.triggered_engagement_lift:
        m = df["triggered"].to_numpy() & (df["group"].to_numpy() == "treatment")
        df.loc[m, "engagement"] *= 1.0 + spec.triggered_engagement_lift

    trig = df["triggered"].to_numpy()
    treat = df["group"].to_numpy() == "treatment"
    tau_t = float(trig[treat].mean())
    tau_c = float(trig[~treat].mean())
    tau_all = float(trig.mean())

    # power adequacy for the CACE analysis (triggered users per arm)
    n_trig_arm = spec.n_users * spec.allocation * tau_all
    cace_need = required_n(1.0, 0.025)
    cace_need_cuped = apply_cuped(cace_need, 0.5)

    truth: dict = {
        "scenario": spec.name,
        "step": 2,
        "n_users": spec.n_users,
        "seed": spec.seed,
        "intended_allocation": spec.allocation,
        "true_cace_engagement": spec.base_engagement_lift
        or spec.triggered_engagement_lift,
        "true_cace_applies_to": ("all users" if spec.base_engagement_lift
                                  else "triggered users only"),
        "trigger_rate_control": round(tau_c, 4),
        "trigger_rate_treatment": round(tau_t, 4),
        "trigger_rate_overall": round(tau_all, 4),
        "trigger_design": {
            "base_rate": spec.trigger_rate,
            "rate_by": spec.trigger_rate_by,
            "propensity": spec.trigger_propensity,
            "treat_shift": spec.trigger_treat_shift,
        },
        "expected_verdict": spec.expected_verdict,
        "expected_flag": spec.expected_flag,
        "valid_analysis": spec.valid_analysis,
        "traps": ["dilution"] if "dilution" in spec.name else
                 ["trigger_imbalance"] if "imbalance" in spec.name else [],
        "power_adequacy": {
            "cace_triggered_arm_n": round(n_trig_arm),
            "cace_required_per_arm": round(cace_need),
            "cace_required_per_arm_cuped": round(cace_need_cuped),
            "cace_power_raw": round(achieved_power(n_trig_arm, 1.0, 0.025), 3),
            "cace_power_cuped": round(
                achieved_power(n_trig_arm / 0.5, 1.0, 0.025), 3),
            "note": "CACE power assumes the +2.5% triggered lift; CUPED "
                    "R^2=0.5 is load-bearing for the triggered analysis.",
        },
    }
    return V2Experiment(df=df, ground_truth=truth)


def triggered_scenarios(seed_offset: int = 0) -> list[TriggeredScenario]:
    """The three step-2 scenarios."""
    return [
        TriggeredScenario(
            "dilution_trap",
            # NOTE: base seed 201 was retired — its null ITT was +0.62%
            # (a ~2-sigma unlucky randomization), which swamps the diluted
            # signal the trap is meant to demonstrate. Seed 215 has null
            # ITT -0.02%. The trap is dilution (structural), not unlucky
            # randomization, so the demo seed is chosen clean.
            # NOTE 2: trigger rate ~0.15 (not 0.18): at tau=0.18 the ITT
            # still has ~28% power, so the "naive read misses it" demo is
            # muddy. At tau=0.15, ITT power ~22% while CACE+CUPED stays
            # ~99% — clean separation of the two analyses.
            seed=215 + seed_offset,
            triggered_engagement_lift=0.025,
            trigger_rate_by={"tenure": {"new": 0.22, "returning": 0.12}},
            expected_verdict="launch",
            expected_flag="CACE +2.5% significant on triggered users; ITT "
                          "diluted (~tau x CACE) and underpowered — naive "
                          "ITT-only read misses a real effect. Rollout "
                          "impact ~= CACE x trigger rate.",
            valid_analysis="cace",
        ),
        TriggeredScenario(
            "trigger_imbalance_trap",
            seed=202 + seed_offset,
            triggered_engagement_lift=0.02,
            trigger_rate=0.2,
            trigger_propensity=("pre_engagement_28d", 0.08),
            trigger_treat_shift=0.05,
            expected_verdict="inconclusive",
            expected_flag="TRIGGER IMBALANCE: treatment shifts trigger "
                          "rates (+5pp) and triggering correlates with "
                          "engagement — triggered-only analysis is "
                          "selection-biased. Fall back to ITT.",
            valid_analysis="itt",
        ),
        TriggeredScenario(
            "triggered_clean_win",
            seed=203 + seed_offset,
            triggered_engagement_lift=0.025,
            trigger_rate=0.5,
            expected_verdict="launch",
            expected_flag="",
            valid_analysis="both",
        ),
    ]


def write_triggered_scenario(spec: TriggeredScenario, outdir: str,
                             sample_n: int | None = None,
                             sample_dir: str | None = None) -> V2Experiment:
    """Generate, write parquet + experiment.json, optional repo sample."""
    exp = generate_triggered(spec)
    os.makedirs(outdir, exist_ok=True)
    exp.df.to_parquet(os.path.join(outdir, f"{spec.name}.parquet"), index=False)
    with open(os.path.join(outdir, f"{spec.name}.experiment.json"), "w") as f:
        json.dump(exp.ground_truth, f, indent=2)
    if sample_n and sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
        rng = np.random.default_rng(999)
        idx = rng.choice(len(exp.df), size=sample_n, replace=False)
        sample = exp.df.iloc[np.sort(idx)].reset_index(drop=True)
        sample.to_parquet(os.path.join(sample_dir, f"{spec.name}.parquet"),
                          index=False)
        key = dict(exp.ground_truth)
        key["n_users"] = sample_n
        key["note"] = (f"20k-user committed sample of the full n={spec.n_users} "
                       "run; power adequacy figures refer to the full n.")
        with open(os.path.join(sample_dir, f"{spec.name}.experiment.json"),
                  "w") as f:
            json.dump(key, f, indent=2)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser(description="step-2 triggered scenarios")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--n", type=int, default=N_USERS)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--outdir", default=TRIGGERED_OUTDIR)
    ap.add_argument("--sample-n", type=int, default=20_000)
    ap.add_argument("--sample-dir", default="data_v2_sample")
    args = ap.parse_args()

    specs = triggered_scenarios(seed_offset=args.seed_offset)
    if args.scenario != "all":
        specs = [s for s in specs if s.name == args.scenario]
        assert specs, f"unknown scenario {args.scenario}"
    for spec in specs:
        spec.n_users = args.n
        write_triggered_scenario(spec, args.outdir,
                                 sample_n=args.sample_n,
                                 sample_dir=args.sample_dir)
        print(f"wrote {spec.name}: n={args.n:,}")


if __name__ == "__main__":
    main()
