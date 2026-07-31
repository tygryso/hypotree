"""Navigator package — Thompson Sampling selection, convergence gate, DONE sentinel."""

from hypotree.navigator.convergence import (
    CI_LEVEL,
    convergence_gate,
    credible_interval,
    credible_width,
)
from hypotree.navigator.sampler import (
    DEFAULT_LEASE_TTL_S,
    EPSILON_CI,
    EPSILON_LOW,
    EPSILON_TIE,
    N_MAX_SAMPLES,
    VERIFY_THRESHOLD,
    SelectionResult,
    ThompsonSampler,
)

__all__ = [
    "CI_LEVEL",
    "DEFAULT_LEASE_TTL_S",
    "EPSILON_CI",
    "EPSILON_LOW",
    "EPSILON_TIE",
    "N_MAX_SAMPLES",
    "VERIFY_THRESHOLD",
    "SelectionResult",
    "ThompsonSampler",
    "convergence_gate",
    "credible_interval",
    "credible_width",
]
