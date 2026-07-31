"""Pydantic models: Node, Status — the core hypothesis entity."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


class Status(str, Enum):
    """Lifecycle states for a hypothesis node.

    EXHAUSTED is a *terminal* state distinct from INVALIDATED: the hypothesis
    was tested conclusively and did not clear its verify bar, but it was not
    refuted. Nothing further can be learned by re-dispatching it, so it leaves
    the frontier — but its subtree is NOT pruned, because a mediocre result is
    still a legitimate base to refine from.

    BLOCKED and NEEDS_REVISION are conditionally-deferred:
    they exist in the enum but are treated identically to UNTESTED for
    frontier eligibility in v0.
    """

    UNTESTED = "UNTESTED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    EXHAUSTED = "EXHAUSTED"
    INVALIDATED = "INVALIDATED"
    PRUNED = "PRUNED"
    BLOCKED = "BLOCKED"
    NEEDS_REVISION = "NEEDS_REVISION"


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp for default factories."""
    return datetime.now(timezone.utc)


# Posterior mean convenience — not stored, computed on demand.
def posterior_mean(alpha: float, beta: float) -> float:
    """Mean of a Beta(alpha, beta) distribution = α / (α + β)."""
    return alpha / (alpha + beta)


# Posterior variance.
def posterior_variance(alpha: float, beta: float) -> float:
    """Variance of Beta(α, β) = αβ / ((α+β)²(α+β+1))."""
    s = alpha + beta
    return (alpha * beta) / (s * s * (s + 1))
