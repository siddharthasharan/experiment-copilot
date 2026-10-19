"""
Step-3 scenarios: overlapping experiments, segment HTE (mirage and real),
and a network-interference sketch. 400k users/scenario, per-user grain,
parquet — same metric design as the v2 baseline (lognormal engagement,
CV~=1, 28d pre-period for realism; engagement is the working metric here).

Scenarios:
  overlap_trap:        independent concurrent experiment B (50/50). TRUE
                       A×B interaction: A lifts engagement +2% when
                       B=control, -0.5% when B=treatment (~0% per spec;
                       the -0.5pp buys the interaction power the ~100%
                       detection gate needs). Pooled naive read (+0.75%,
                       significant ~2/3 of seeds) misleads — correct verdict
                       flags the overlap and reports stratified effects.
  hte_mirage_trap:     13 segments ("new" + 12 returning cells), TRUE effect
                       homogeneous +2% everywhere. ~half the seeds produce a
                       raw nominal segment deviation by chance (the mirage);
                       BH-FDR + Cochran's Q must not endorse it. Correct
                       verdict: no credible heterogeneity.
  hte_real:            same 13 segments; pre-specified "new" segment has TRUE
                       +4%, others +1%. The new cell must survive BH-FDR
                       ~100% of seeds. Correct verdict: credible
                       heterogeneity, decide per segment.
  interference_sketch: ~2k clusters x 200 users, user-level randomization,
                       outcome += lambda * 100 * (frac_treated_mates - 0.5)
                       with lambda=3.2 (documented exaggeration — see below).
                       The demeaned spillover injects intra-cluster
                       correlation with zero mean shift. Naive i.i.d. SE
                       understates uncertainty; cluster-robust SE is wider
                       and the SUTVA flag fires.

On lambda: the sandwich estimator works per arm, so the effective cluster
size is m/2 (~100 users). Cluster-SE variance inflation is 1 + (m/2-1)*rho
with rho ~= (lambda*100*sd(f))^2 / sigma^2, sd(f) = 0.5/sqrt(m), and
sigma ~= 143 (the v2 engagement SD). lambda=3.2 -> SE ratio ~1.28 by
construction (measured 1.26-1.30 across 10 seeds), deliberately visible.
Real spillovers (lambda << 1) are invisible to this check — which is
exactly why SUTVA violations are dangerous. Sketch, not a network model:
clusters are assumed independent.

Every scenario writes an experiment.json answer key with true effects,
the expected verdict, and which checks must fire.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_v2 import ENG_MEAN, ENG_S, EXP_DAYS, N_USERS, PRE_RHO

N_CLUSTERS = 2_000
INTERFERENCE_LAMBDA = 3.2   # exaggerated for visibility; see module docstring
HTE_COUNTRIES = ["US", "IN", "BR", "other"]
HTE_COUNTRY_P = [0.40, 0.25, 0.15, 0.20]
HTE_PLATFORMS = ["ios", "android", "web"]
HTE_PLATFORM_P = [0.45, 0.45, 0.10]


@dataclass
class InteractionScenario:
    """A fully specified step-3 interaction scenario."""

    name: str
    seed: int = 301
    n_users: int = N_USERS
    expected_verdict: str = "don't launch"
    expected_flag: str = ""


@dataclass
class InteractionExperiment:
    df: pd.DataFrame
    ground_truth: dict = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Shared primitives: assignment, segments, engagement + pre-period."""
    groups = (rng.random(n) < 0.5).astype(np.int8)
    treat = groups == 1
    countries = rng.choice(HTE_COUNTRIES, size=n, p=HTE_COUNTRY_P)
    platforms = rng.choice(HTE_PLATFORMS, size=n, p=HTE_PLATFORM_P)
    tenure = np.where(rng.random(n) < 0.30, "new", "returning")
    segment = np.where(tenure == "new", "new",
                       np.char.add(np.char.add(countries, "|"), platforms))

    eta = rng.standard_normal(n)
    eps = math.sqrt(1.0 - PRE_RHO ** 2)
    pre_eng = np.exp(math.log(ENG_MEAN) + ENG_S * eta)
    exp_eng = np.exp(math.log(ENG_MEAN)
                     + ENG_S * (PRE_RHO * eta + eps * rng.standard_normal(n)))
    return pd.DataFrame({
        "user_id": np.arange(n, dtype=np.int64),
        "group": np.where(treat, "treatment", "control"),
        "country": countries,
        "platform": platforms,
        "tenure": tenure,
        "segment": segment,
        "pre_engagement_28d": pre_eng,
        "engagement": exp_eng,
    })


def _answer_key(spec: InteractionScenario, df: pd.DataFrame,
                true_effects: dict, checks: list[str],
                design_notes: dict) -> dict:
    treat = (df["group"] == "treatment").to_numpy()
    return {
        "scenario": spec.name,
        "n_users": int(len(df)),
        "seed": spec.seed,
        "actual_allocation": round(float(treat.mean()), 4),
        "true_effects": true_effects,
        "expected_verdict": spec.expected_verdict,
        "expected_flag": spec.expected_flag,
        "checks_that_must_fire": checks,
        "design_notes": design_notes,
    }


# --- scenario builders ---------------------------------------------------------

def generate_overlap(spec: InteractionScenario) -> InteractionExperiment:
    """A×B interaction trap. B independent 50/50; A works only when B=0."""
    rng = _rng(spec.seed)
    n = spec.n_users
    df = _base_frame(n, rng)
    treat = (df["group"] == "treatment").to_numpy()

    b = (rng.random(n) < 0.5).astype(np.int8)
    df["exp_b"] = b
    y = df["engagement"].to_numpy(dtype=float).copy()
    y[treat & (b == 0)] *= 1.02    # A lifts +2% when B=control
    y[treat & (b == 1)] *= 0.995   # ~0% (slightly negative) when B=treatment
    df["engagement"] = y

    truth = _answer_key(
        spec, df,
        true_effects={"A_lift_given_B0": 0.02, "A_lift_given_B1": -0.005,
                      "AxB_interaction_pp": -0.025,
                      "pooled_A_lift": 0.0075},
        checks=["overlap interaction detected (A×B p<0.05)",
                "stratified estimates recover +2% / -0.5%"],
        design_notes={"B_assignment": "independent 50/50, column exp_b",
                      "trap": "pooled naive read (+0.75%, significant ~2/3 of "
                              "seeds) hides the interaction"})
    return InteractionExperiment(df=df, ground_truth=truth)


def _hte_frame(spec: InteractionScenario, new_lift: float,
               other_lift: float) -> tuple[pd.DataFrame, np.random.Generator]:
    rng = _rng(spec.seed)
    df = _base_frame(spec.n_users, rng)
    treat = (df["group"] == "treatment").to_numpy()
    is_new = (df["segment"] == "new").to_numpy()
    y = df["engagement"].to_numpy(dtype=float).copy()
    y[treat] *= 1.0 + other_lift
    y[treat & is_new] *= (1.0 + new_lift) / (1.0 + other_lift)
    df["engagement"] = y
    return df, rng


def generate_hte_mirage(spec: InteractionScenario) -> InteractionExperiment:
    """Homogeneous +2% everywhere: any 'heterogeneity' is noise."""
    df, _ = _hte_frame(spec, new_lift=0.02, other_lift=0.02)
    truth = _answer_key(
        spec, df,
        true_effects={"lift_all_segments": 0.02, "heterogeneity": 0.0},
        checks=["Cochran's Q not significant (no real heterogeneity)",
                "BH-FDR leaves zero surviving segments"],
        design_notes={"n_segments": 13,
                      "trap": "~half the seeds show a raw nominal segment "
                              "deviation by chance — the mirage BH must kill"})
    return InteractionExperiment(df=df, ground_truth=truth)


def generate_hte_real(spec: InteractionScenario) -> InteractionExperiment:
    """Real HTE: pre-specified 'new' segment +4%, others +1%."""
    df, _ = _hte_frame(spec, new_lift=0.04, other_lift=0.01)
    truth = _answer_key(
        spec, df,
        true_effects={"lift_new_segment": 0.04, "lift_other_segments": 0.01,
                      "heterogeneity_pp": 0.03},
        checks=["Cochran's Q significant",
                "'new' segment survives BH-FDR"],
        design_notes={"n_segments": 13,
                      "prespecified_segment": "new (~30% of users)"})
    return InteractionExperiment(df=df, ground_truth=truth)


def generate_interference(spec: InteractionScenario,
                          lam: float = INTERFERENCE_LAMBDA,
                          direct_lift: float = 0.02) -> InteractionExperiment:
    """SUTVA sketch: spillover = lam * frac_treated_mates (additive)."""
    rng = _rng(spec.seed)
    n = spec.n_users
    df = _base_frame(n, rng)
    treat = (df["group"] == "treatment").to_numpy().astype(np.int8)

    cluster_id = (np.arange(n) // (n // N_CLUSTERS)).astype(np.int32)
    df["cluster_id"] = cluster_id
    n_cl = N_CLUSTERS
    treated_per_cl = np.bincount(cluster_id, weights=treat, minlength=n_cl)
    size_per_cl = np.bincount(cluster_id, minlength=n_cl)
    frac_mates = (treated_per_cl[cluster_id] - treat) / np.maximum(
        size_per_cl[cluster_id] - 1, 1)

    y = df["engagement"].to_numpy()
    # direct effect (multiplicative, like the rest of v2) + demeaned additive
    # spillover: zero mean shift, pure intra-cluster correlation injection
    y = y * (1.0 + direct_lift * treat) + lam * 100.0 * (frac_mates - 0.5)
    df["engagement"] = y

    truth = _answer_key(
        spec, df,
        true_effects={"direct_lift": direct_lift,
                      "spillover_lambda": lam,
                      "spillover_form": "y += lambda * 100 * "
                                        "(frac_treated_cluster_mates - 0.5)"},
        checks=["cluster-robust SE > naive SE (ratio > 1.15)",
                "SUTVA flag fires (ratio > 1.25)"],
        design_notes={"n_clusters": N_CLUSTERS,
                      "randomization": "user-level (not clustered)",
                      "lambda_note": "3.2 is deliberately exaggerated for "
                                     "visibility and demeaned so it adds no "
                                     "level shift; SE inflation ~= 1 + "
                                     "(m/2-1)*rho with rho ~= "
                                     "(lambda*100*0.5/sqrt(m)/sigma)^2, "
                                     "sigma ~= 143 -> ratio ~1.28; real "
                                     "spillovers are subtler and invisible"})
    return InteractionExperiment(df=df, ground_truth=truth)


_BUILDERS = {
    "overlap_trap": generate_overlap,
    "hte_mirage_trap": generate_hte_mirage,
    "hte_real": generate_hte_real,
    "interference_sketch": generate_interference,
}


def interaction_scenarios(seed_offset: int = 0) -> list[InteractionScenario]:
    return [
        InteractionScenario(
            "overlap_trap", seed=301 + seed_offset,
            expected_verdict="inconclusive — flag overlap, report stratified",
            expected_flag="A×B interaction: +2% when B=control, -0.5% when "
                          "B=treatment; pooled +0.75% is misleading"),
        InteractionScenario(
            "hte_mirage_trap", seed=302 + seed_offset,
            expected_verdict="no credible heterogeneity",
            expected_flag="homogeneous +2%; nominal segment hits are "
                          "multiple-testing noise — BH-FDR clean"),
        InteractionScenario(
            "hte_real", seed=303 + seed_offset,
            expected_verdict="credible heterogeneity — decide per segment",
            expected_flag="'new' segment +4% survives FDR; others +1%"),
        InteractionScenario(
            "interference_sketch", seed=304 + seed_offset,
            expected_verdict="SUTVA risk flagged — treat SEs as understated",
            expected_flag="cluster-robust SE ~1.3x naive; user-level "
                          "randomization with spillover"),
    ]


def generate_interaction(spec: InteractionScenario) -> InteractionExperiment:
    return _BUILDERS[spec.name](spec)


def write_scenario(spec: InteractionScenario, outdir: str,
                   sample_n: int | None = None,
                   sample_dir: str | None = None) -> InteractionExperiment:
    exp = generate_interaction(spec)
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
                       "run; power/adequacy figures refer to the full n.")
        with open(os.path.join(sample_dir, f"{spec.name}.experiment.json"), "w") as f:
            json.dump(key, f, indent=2)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser(description="step-3 interaction generator")
    ap.add_argument("--scenario", default="all", help="scenario name or 'all'")
    ap.add_argument("--n", type=int, default=N_USERS)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--outdir", default="eval_data_v2_interaction")
    ap.add_argument("--sample-n", type=int, default=20_000)
    ap.add_argument("--sample-dir", default="data_v2_sample")
    args = ap.parse_args()

    specs = interaction_scenarios(seed_offset=args.seed_offset)
    if args.scenario != "all":
        specs = [s for s in specs if s.name == args.scenario]
        assert specs, f"unknown scenario {args.scenario}"
    for spec in specs:
        spec.n_users = args.n
        write_scenario(spec, args.outdir,
                       sample_n=args.sample_n, sample_dir=args.sample_dir)
        print(f"wrote {spec.name}: n={args.n:,}")


if __name__ == "__main__":
    main()
