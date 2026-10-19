"""
Triggered analysis: ITT vs CACE for experiments where only a fraction of
randomized users ever reach the treated surface.

The core problem: randomize 400k users into a new checkout flow, but only
~20% ever reach checkout. Three analyses answer three different questions:

  ITT (intent-to-treat): effect over ALL randomized users. Diluted by the
    trigger rate (ITT ~= tau * CACE) but unbiased for the policy question
    "what happens to the overall metric if we ship this to everyone".

  CACE (complier average causal effect): the effect on the triggered. The
    primary estimator is the *direct triggered contrast* — Welch's t on
    triggered users only — which is unbiased when triggering is unaffected
    by treatment (exactly what ``check_trigger_balance`` verifies) and much
    more efficient than IV: the Wald estimator ITT/P(trigger|treatment)
    pays the full-sample noise cost (its SE is SE(ITT)/tau), so at
    tau=0.18 it needs ~3x the lift for the same power. Wald is kept as
    ``cace_wald`` for the unobservable-trigger case (triggering only
    measurable in treatment), where the direct contrast isn't available.

  Trigger-balance check: an SRM-style chi-square on trigger *rates* by arm.
    If treatment changes who triggers, the triggered subpopulations differ
    by arm and any triggered-only contrast is selection-biased. When this
    fires, fall back to ITT (still unbiased — randomization is intact at
    the assignment level).

Power math (see power.py): against a design where everyone triggers, a
triggered (CACE) analysis needs 1/tau randomized users per arm and ITT
needs 1/tau^2. On the *same* experiment, ITT needs 1/tau the users of the
CACE analysis — ~3.3x at a 30% trigger rate — because ITT sees tau x more
users, offsetting one factor of tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .stats import LiftResult, cuped, estimate_lift


# --- triggering assignment -------------------------------------------------

def add_triggering(
    df: pd.DataFrame,
    trigger_rate: float = 0.2,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    rate_by: dict[str, dict] | None = None,
    propensity: tuple[str, float] | None = None,
    treat_shift: float = 0.0,
    group_col: str = "group",
    out_col: str = "triggered",
) -> pd.DataFrame:
    """Assign each user a ``triggered`` flag (reached the treated surface).

    Parameters
    ----------
    trigger_rate: base P(trigger) when no heterogeneity is specified.
    rate_by: {column: {level: rate}} for heterogeneous trigger rates, e.g.
        ``{"tenure": {"new": 0.26, "returning": 0.14}}``. Levels missing
        from the mapping fall back to ``trigger_rate``.
    propensity: (column, tilt) — shifts P(trigger) by ``tilt`` times the
        z-scored log of ``column``. Models "high-engagement users are more
        likely to reach the surface". The log keeps heavy-tailed columns
        from dominating.
    treat_shift: added to the treatment arm's trigger probability. Models
        the trap where the treatment itself changes who triggers (e.g. a
        more prominent entry point). Nonzero values violate the exclusion
        restriction — ``check_trigger_balance`` must catch this.
    """
    rng = rng or np.random.default_rng(seed)
    n = len(df)
    rates = np.full(n, trigger_rate, dtype=float)

    if rate_by:
        for col, mapping in rate_by.items():
            mapped = df[col].map(mapping)
            rates = np.where(mapped.notna().to_numpy(),
                             mapped.to_numpy(dtype=float), rates)

    if propensity is not None:
        col, tilt = propensity
        vals = np.log(df[col].to_numpy(dtype=float) + 1e-9)
        z = (vals - vals.mean()) / (vals.std() + 1e-12)
        rates = rates + tilt * z

    if treat_shift:
        is_treat = (df[group_col].to_numpy() == "treatment")
        rates = np.where(is_treat, rates + treat_shift, rates)

    rates = np.clip(rates, 0.005, 0.995)
    out = df.copy()
    out[out_col] = rng.random(n) < rates
    return out


# --- estimators -------------------------------------------------------------

def itt_effect(df: pd.DataFrame, y_col: str = "engagement",
               group_col: str = "group") -> LiftResult:
    """Intent-to-treat: Welch's t over ALL randomized users.

    Diluted by the trigger rate but unbiased for the overall policy effect.
    """
    c = df.loc[df[group_col] == "control", y_col].to_numpy(dtype=float)
    t = df.loc[df[group_col] == "treatment", y_col].to_numpy(dtype=float)
    return estimate_lift(c, t)


def triggered_contrast(df: pd.DataFrame, y_col: str = "engagement",
                       group_col: str = "group",
                       trigger_col: str = "triggered",
                       alpha: float = 0.05) -> LiftResult:
    """Direct CACE estimator: Welch's t on triggered users only.

    This is the PRIMARY triggered estimator — not the Wald IV. When
    triggering is unaffected by treatment (what ``check_trigger_balance``
    verifies), the triggered subpopulation is distributionally identical
    across arms, so conditioning on it is unbiased for the effect on the
    triggered. It is much more efficient than Wald: Wald's SE is
    SE(ITT)/tau, paying full-sample noise for a diluted signal, while the
    direct contrast only pays the triggered users' noise. Use
    ``cace_wald`` only when triggering is unobservable in control.
    """
    m = df[trigger_col].to_numpy().astype(bool)
    g = df[group_col].to_numpy()
    c = df.loc[(g == "control") & m, y_col].to_numpy(dtype=float)
    t = df.loc[(g == "treatment") & m, y_col].to_numpy(dtype=float)
    return estimate_lift(c, t, alpha=alpha)


def triggered_cuped_lift(df: pd.DataFrame, y_col: str = "engagement",
                         pre_col: str = "pre_engagement_28d",
                         group_col: str = "group",
                         trigger_col: str = "triggered",
                         alpha: float = 0.05) -> LiftResult:
    """``triggered_contrast`` on CUPED-adjusted values.

    Theta is fit on the triggered controls, so the adjustment can't absorb
    the treatment effect. In v2 CUPED is load-bearing — this is the primary
    analysis for powered triggered scenarios.
    """
    m = df[trigger_col].to_numpy().astype(bool)
    sub = df.loc[m].copy()
    adj = cuped_residuals(sub, y_col, pre_col)
    sub = sub.assign(_cuped_adj=adj)
    return triggered_contrast(sub, y_col="_cuped_adj", group_col=group_col,
                              trigger_col=trigger_col, alpha=alpha)


@dataclass
class CACEResult:
    estimate: float        # Wald estimate of the effect on the triggered
    se: float              # delta-method standard error
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    tau_treatment: float   # P(triggered | treatment) — the first stage
    tau_control: float     # P(triggered | control)
    n_triggered_treatment: int
    n_triggered_control: int


def cace_wald(df: pd.DataFrame, y_col: str = "engagement",
              group_col: str = "group", trigger_col: str = "triggered",
              alpha: float = 0.05) -> CACEResult:
    """CACE via the Wald (IV) estimator: ITT / P(triggered | treatment).

    Standard error by the delta method, treating the first stage as
    estimated: Var(CACE) ~= Var(ITT)/tau^2 + ITT^2 * Var(tau_hat)/tau^4.

    Assumptions (documented, not verified — that is what
    ``check_trigger_balance`` is for): exclusion (treatment affects the
    outcome only through triggering), monotonicity, and triggering
    independent of treatment assignment.
    """
    itt = itt_effect(df, y_col=y_col, group_col=group_col)
    se_itt = (itt.ci_high - itt.ci_low) / (2 * 1.959963984540054)

    treat = df[group_col].to_numpy() == "treatment"
    trig = df[trigger_col].to_numpy().astype(bool)
    tau_t = float(trig[treat].mean())
    tau_c = float(trig[~treat].mean())
    n_t = int(treat.sum())

    if tau_t <= 0:
        raise ValueError("no triggered users in treatment — CACE undefined")

    est = itt.lift / tau_t
    var_tau = tau_t * (1 - tau_t) / n_t
    var_cace = se_itt**2 / tau_t**2 + (itt.lift**2) * var_tau / tau_t**4
    se = float(np.sqrt(var_cace))
    z = est / se if se > 0 else 0.0
    p = float(2 * sstats.norm.sf(abs(z)))
    crit = sstats.norm.ppf(1 - alpha / 2)
    return CACEResult(
        estimate=float(est), se=se,
        ci_low=float(est - crit * se), ci_high=float(est + crit * se),
        p_value=p, significant=bool(p < alpha),
        tau_treatment=tau_t, tau_control=tau_c,
        n_triggered_treatment=int(trig[treat].sum()),
        n_triggered_control=int(trig[~treat].sum()),
    )


@dataclass
class TriggerBalanceResult:
    detected: bool
    p_value: float
    chi2: float
    trigger_rate_control: float
    trigger_rate_treatment: float


def check_trigger_balance(df: pd.DataFrame, group_col: str = "group",
                          trigger_col: str = "triggered",
                          alpha: float = 0.001) -> TriggerBalanceResult:
    """SRM-style chi-square on trigger rates by arm (2x2 table).

    Uses the same strict default alpha as ``check_srm`` (0.001): at 400k
    users the test is extremely powerful, and only a real shift in who
    triggers should fire it. A firing check means the triggered
    subpopulations differ by arm — any triggered-only contrast, including
    the Wald CACE, is selection-biased. Fall back to ITT.
    """
    treat = df[group_col].to_numpy() == "treatment"
    trig = df[trigger_col].to_numpy().astype(bool)
    table = np.array([
        [(~treat & trig).sum(), (~treat & ~trig).sum()],
        [(treat & trig).sum(), (treat & ~trig).sum()],
    ], dtype=float)
    chi2, p, _, _ = sstats.chi2_contingency(table)
    return TriggerBalanceResult(
        detected=bool(p < alpha), p_value=float(p), chi2=float(chi2),
        trigger_rate_control=float(trig[~treat].mean()),
        trigger_rate_treatment=float(trig[treat].mean()),
    )


def cuped_residuals(df: pd.DataFrame, y_col: str,
                    pre_col: str) -> np.ndarray:
    """CUPED-adjusted outcome series (theta from control, pooled centering).

    Lets the Wald estimator run on variance-reduced values — in v2 CUPED is
    load-bearing, so CACE-with-CUPED is the primary triggered analysis.
    """
    res = cuped(df, y_col=y_col, pre_col=pre_col)
    y = df[y_col].to_numpy(dtype=float)
    x = df[pre_col].to_numpy(dtype=float)
    return y - res.theta * (x - x.mean())
