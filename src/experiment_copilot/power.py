"""
Power analysis: how much data is "enough for p < 0.05".

The core relationship for a two-arm experiment (equal allocation,
two-sided alpha, target power):

    n_per_arm = 2 * CV^2 * (z_{1-a/2} + z_{1-b})^2 / r^2

where CV is the metric's coefficient of variation (std/mean) and r is the
relative lift (MDE) you want to detect. For alpha=0.05, power=0.8 the
constant is 2*(1.96+0.84)^2 = 15.68.

Two practical consequences drive the v2 data design:
  1. Heavy-tailed metrics explode the requirement (it scales with CV^2 —
     a lognormal revenue metric with sigma=1.6 has CV~3.5, so ~12x the
     sample of a CV=1 engagement metric for the same lift).
  2. The cheapest sample size is variance reduction, not more rows:
     CUPED with a pre-period covariate at R^2 multiplies required n by
     (1 - R^2).
"""

from __future__ import annotations

from scipy import stats as sstats

_Z_975 = sstats.norm.ppf(0.975)   # 1.9599...
_Z_80 = sstats.norm.ppf(0.80)     # 0.8416...


def _z_sum(power: float, alpha: float) -> float:
    return sstats.norm.ppf(1 - alpha / 2) + sstats.norm.ppf(power)


def required_n(cv: float, relative_lift: float,
               power: float = 0.8, alpha: float = 0.05) -> float:
    """Per-arm sample size to detect `relative_lift` on a metric with
    coefficient of variation `cv`, at the given power and alpha."""
    z = _z_sum(power, alpha)
    return 2.0 * cv**2 * z**2 / relative_lift**2


def required_n_binary(base_rate: float, relative_lift: float,
                      power: float = 0.8, alpha: float = 0.05) -> float:
    """Per-arm sample size for a binary (conversion) metric.

    Variance of a Bernoulli(p0) is p0*(1-p0); the absolute MDE is
    r*p0, so n = 2*p0*(1-p0)*z^2 / (r*p0)^2.
    """
    p0 = base_rate
    z = _z_sum(power, alpha)
    return 2.0 * p0 * (1 - p0) * z**2 / (relative_lift * p0) ** 2


def apply_cuped(n: float, r_squared: float) -> float:
    """CUPED-adjusted sample size: a pre-period covariate explaining R^2
    of the outcome variance multiplies required n by (1 - R^2)."""
    return n * (1.0 - r_squared)


def achieved_power(n_per_arm: float, cv: float, relative_lift: float,
                   alpha: float = 0.05) -> float:
    """Power achieved with `n_per_arm` per arm for a continuous metric."""
    z_alpha = sstats.norm.ppf(1 - alpha / 2)
    # non-centrality: delta / SE, SE of the difference = sqrt(2*CV^2/n)
    ncp = relative_lift / ((2.0 * cv**2 / n_per_arm) ** 0.5)
    return float(sstats.norm.cdf(ncp - z_alpha))


def achieved_power_binary(n_per_arm: float, base_rate: float,
                          relative_lift: float, alpha: float = 0.05) -> float:
    """Power achieved with `n_per_arm` per arm for a binary metric."""
    z_alpha = sstats.norm.ppf(1 - alpha / 2)
    p0 = base_rate
    se = (2.0 * p0 * (1 - p0) / n_per_arm) ** 0.5
    ncp = (relative_lift * p0) / se
    return float(sstats.norm.cdf(ncp - z_alpha))


def size_scenario(n_total: int = 400_000, cuped_r2: float = 0.55) -> str:
    """Print the v2 power table: what each baseline metric needs vs the
    chosen uniform n, and whether CUPED makes it adequate."""
    rows = [
        ("engagement", "continuous", dict(cv=1.0, lift=0.02)),
        ("revenue_per_user", "continuous", dict(cv=3.45, lift=0.02)),
        ("converted", "binary", dict(base_rate=0.30, lift=0.02)),
        ("reports", "binary", dict(base_rate=0.001, lift=0.35)),
    ]
    n_arm = n_total / 2
    lines = []
    lines.append(f"Power design: n_total={n_total:,} ({n_arm:,.0f}/arm), "
                 f"alpha=0.05 two-sided, target power=0.80, CUPED R^2={cuped_r2}")
    lines.append("-" * 96)
    lines.append(f"{'metric':<18}{'MDE':>8}{'n/arm raw':>12}{'n/arm CUPED':>12}"
                 f"{'power@400k':>12}{'power@400k+CUPED':>16}")
    lines.append("-" * 96)
    for name, kind, kw in rows:
        lift = kw["lift"]
        if kind == "continuous":
            need = required_n(kw["cv"], lift)
            need_c = apply_cuped(need, cuped_r2)
            pw = achieved_power(n_arm, kw["cv"], lift)
            pw_c = achieved_power(n_arm / (1 - cuped_r2), kw["cv"], lift)
        else:
            need = required_n_binary(kw["base_rate"], lift)
            need_c = need  # CUPED rarely used on rare binary outcomes
            pw = achieved_power_binary(n_arm, kw["base_rate"], lift)
            pw_c = pw
        mde = f"+{lift:.0%}" if lift < 1 else f"+{lift:.0%}"
        lines.append(f"{name:<18}{mde:>8}{need:>12,.0f}{need_c:>12,.0f}"
                     f"{pw:>12.2f}{pw_c:>16.2f}")
    lines.append("-" * 96)
    lines.append("Read: revenue +2% is underpowered raw at 400k (needs CUPED); "
                 "reports +35% is the volume driver among binary metrics.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(size_scenario())
