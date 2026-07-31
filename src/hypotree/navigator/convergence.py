"""Posterior credible-interval and convergence-gate helpers.

The convergence gate decides when a node has "enough" evidence to auto-transition
to VERIFIED or INVALIDATED. For deterministic nodes this is immediate (one sample).
For stochastic nodes the posterior credible-interval width must shrink below a
threshold (or N_max is reached), preventing a single lucky sample from declaring
victory.
"""

from __future__ import annotations

from scipy.stats import beta as beta_dist

# Default credible-interval level (central, two-sided).
CI_LEVEL = 0.95


def credible_interval(alpha: float, beta: float, level: float = CI_LEVEL) -> tuple[float, float]:
    """Central credible interval (lo, hi) for a Beta(alpha, beta) posterior.

    Returns the (1-level)/2 and 1-(1-level)/2 quantiles — a bounded, intuitive
    uncertainty measure that normalizes across nodes regardless of sample count.
    """
    tail = (1 - level) / 2
    lo = float(beta_dist.ppf(tail, alpha, beta))
    hi = float(beta_dist.ppf(1 - tail, alpha, beta))
    return lo, hi


def credible_width(alpha: float, beta: float, level: float = CI_LEVEL) -> float:
    """Width of the central credible interval — the convergence-gate metric."""
    lo, hi = credible_interval(alpha, beta, level)
    return hi - lo


def convergence_gate(
    evidence_regime: str,
    evidence_count: int,
    alpha: float,
    beta: float,
    epsilon_ci: float = 0.1,
    n_max: int = 50,
) -> bool:
    """Decide whether a node has enough evidence to auto-transition.

    Deterministic nodes converge after a single sample.
    Stochastic nodes converge when the credible-interval width drops below
    epsilon_ci, or when n_max samples have been collected (whichever first).
    """
    if evidence_regime == "deterministic":
        return evidence_count >= 1
    # Stochastic: tight enough to decide, or hit the sample ceiling.
    if evidence_count >= n_max:
        return True
    return credible_width(alpha, beta) < epsilon_ci
