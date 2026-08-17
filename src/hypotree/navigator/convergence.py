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


# Why a node's evidence was judged conclusive. The distinction that matters is
# the last two: an interval that tightened is a measurement, a ceiling that
# fired is a budget, and a belief state that cannot tell them apart will report
# a high-variance guess with the same confidence as a settled fact.
CONVERGED_DETERMINISTIC = "deterministic"
CONVERGED_INTERVAL = "credible_interval"
CONVERGED_CEILING = "sample_ceiling"
NOT_CONVERGED = "not_converged"


def convergence_verdict(
    evidence_regime: str,
    evidence_count: int,
    alpha: float,
    beta: float,
    epsilon_ci: float = 0.1,
    n_max: int = 50,
) -> tuple[bool, str]:
    """Whether a node has enough evidence to auto-transition, **and why**.

    The reason is the point. A stochastic node whose credible interval is still
    wide at ``n_max`` is settled by exhaustion of budget rather than by
    evidence, and until that was recorded the two were indistinguishable
    afterwards: the status said VERIFIED and nothing anywhere said the interval
    was never tight. Callers put the reason in the status-change record so the
    audit log can answer "why did this settle?" the same way it answers every
    other question about how the belief state got here.
    """
    if evidence_regime == "deterministic":
        if evidence_count >= 1:
            return True, CONVERGED_DETERMINISTIC
        return False, NOT_CONVERGED
    # Stochastic: tight enough to decide, or hit the sample ceiling. The
    # interval is checked first so a run that reaches both on the same
    # observation is credited to the measurement, not to the budget.
    if credible_width(alpha, beta) < epsilon_ci:
        return True, CONVERGED_INTERVAL
    if evidence_count >= n_max:
        return True, CONVERGED_CEILING
    return False, NOT_CONVERGED


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
    return convergence_verdict(evidence_regime, evidence_count, alpha, beta, epsilon_ci, n_max)[0]
