"""Unit tests for the navigator: TS sampler determinism, epsilon-tiebreak decay,
DONE sentinel, convergence gate, claim consumption, regime-aware transitions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from hypotree.models.node import Node
from hypotree.models.status import Status
from hypotree.navigator.convergence import (
    convergence_gate,
    credible_interval,
    credible_width,
)
from hypotree.navigator.sampler import ThompsonSampler


def make_node(
    id: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    status: Status = Status.UNTESTED,
    is_goal: bool = False,
    target_metric: float | None = None,
    evidence_regime: str = "deterministic",
    updated_at: datetime | None = None,
    exclusion_group: str | None = None,
) -> Node:
    """Helper to create a Node with sensible defaults for navigator tests."""
    node = Node(
        id=id,
        statement=f"Hypothesis {id}",
        status=status,
        alpha=alpha,
        beta=beta,
        is_goal=is_goal,
        target_metric=target_metric,
        evidence_regime=evidence_regime,
        exclusion_group=exclusion_group,
    )
    if updated_at is not None:
        node.updated_at = updated_at
    return node


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sampler(rng: np.random.Generator) -> ThompsonSampler:
    return ThompsonSampler(rng=rng)


# ---------------------------------------------------------------------------
# Seeded RNG determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_same_seed_same_selection() -> None:
    """Two samplers with the same seed must pick the same node."""
    nodes = [
        make_node("a", alpha=3.0, beta=2.0),
        make_node("b", alpha=1.0, beta=5.0),
        make_node("c", alpha=4.0, beta=1.0),
    ]
    s1 = ThompsonSampler(np.random.default_rng(seed=123))
    s2 = ThompsonSampler(np.random.default_rng(seed=123))
    r1 = s1.select(nodes, nodes)
    r2 = s2.select(nodes, nodes)
    assert r1.node_id == r2.node_id


@pytest.mark.unit
def test_different_seed_may_differ() -> None:
    """Different seeds can produce different selection sequences (probabilistic)."""
    nodes = [
        make_node("a", alpha=2.0, beta=2.0),
        make_node("b", alpha=2.0, beta=2.0),
    ]
    selections: set[str] = set()
    for seed in range(100):
        s = ThompsonSampler(np.random.default_rng(seed=seed))
        result = s.select(nodes, nodes)
        selections.add(result.node_id)  # type: ignore
    # With identical posteriors, both should be selected across 100 seeds.
    assert len(selections) == 2


# ---------------------------------------------------------------------------
# DONE sentinel
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_done_empty_frontier(sampler: ThompsonSampler) -> None:
    result = sampler.select([], [])
    assert result.status == "DONE"
    assert result.reason == "empty_frontier"


@pytest.mark.unit
def test_done_all_goals_met(sampler: ThompsonSampler) -> None:
    """Goal completion is decided by the caller, which can see the graph.

    The sampler only sees posteriors, and a goal's posterior is not evidence its
    objective was reached — it is the record of how many times something was
    filed against it.
    """
    goal = make_node(
        "g1",
        alpha=10.0,
        beta=1.0,
        status=Status.VERIFIED,
        is_goal=True,
        target_metric=0.8,
    )
    assert sampler.select([], [goal]).reason == "empty_frontier"

    result = sampler.select([], [goal], all_goals_met=True)
    assert result.status == "DONE"
    assert result.reason == "all_goals_met"


@pytest.mark.unit
def test_not_done_goal_not_verified(sampler: ThompsonSampler) -> None:
    """Goal not yet verified should not trigger DONE even if frontier is non-empty."""
    goal = make_node(
        "g1",
        alpha=1.0,
        beta=1.0,
        status=Status.UNTESTED,
        is_goal=True,
        target_metric=0.9,
    )
    result = sampler.select([goal], [goal])
    assert result.status == "SELECTED"
    assert result.node_id == "g1"


@pytest.mark.unit
def test_not_done_goal_verified_low_posterior(sampler: ThompsonSampler) -> None:
    """Goal is VERIFIED but posterior dropped — should not be DONE."""
    goal = make_node(
        "g1",
        alpha=1.0,
        beta=10.0,  # low posterior mean
        status=Status.VERIFIED,
        is_goal=True,
        target_metric=0.8,
    )
    result = sampler.select([], [goal])
    # Posterior mean < target_metric, so goal not met → empty_frontier DONE
    assert result.status == "DONE"
    assert result.reason == "empty_frontier"


# ---------------------------------------------------------------------------
# TS sampling sanity — higher posterior wins more often
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_higher_posterior_wins_more_often() -> None:
    """A node with alpha=10, beta=1 should be selected far more often than alpha=1, beta=10."""
    good = make_node("good", alpha=10.0, beta=1.0)
    bad = make_node("bad", alpha=1.0, beta=10.0)
    nodes = [good, bad]
    good_wins = 0
    for seed in range(200):
        s = ThompsonSampler(np.random.default_rng(seed=seed))
        result = s.select(nodes, nodes)
        if result.node_id == "good":
            good_wins += 1
    # The good node should win the vast majority of the time (>90%).
    assert good_wins > 180


# ---------------------------------------------------------------------------
# Epsilon-tiebreak decay
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_epsilon_tiebreak_prefers_stalest() -> None:
    """When theta draws are within epsilon_tie, the stalest node is preferred."""
    now = datetime.now(timezone.utc)
    stale = make_node("stale", alpha=5.0, beta=1.0, updated_at=now - timedelta(hours=10))
    fresh = make_node("fresh", alpha=5.0, beta=1.0, updated_at=now - timedelta(seconds=1))
    # Use a large epsilon_tie so both draws are guaranteed to be within the band.
    sampler = ThompsonSampler(np.random.default_rng(42), epsilon_tie=10.0)
    result = sampler.select([stale, fresh], [stale, fresh], now=now)
    assert result.node_id == "stale"


# ---------------------------------------------------------------------------
# Convergence gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_convergence_gate_deterministic() -> None:
    """Deterministic nodes converge after 1 evidence."""
    assert convergence_gate("deterministic", 1, 2.0, 1.0) is True
    assert convergence_gate("deterministic", 0, 1.0, 1.0) is False


@pytest.mark.unit
def test_convergence_gate_stochastic_narrow_ci() -> None:
    """Stochastic node converges when credible width < epsilon_ci."""
    # Alpha=500, beta=50 → very narrow CI → converged.
    assert convergence_gate("stochastic", 20, 500.0, 50.0) is True


@pytest.mark.unit
def test_convergence_gate_stochastic_wide_ci() -> None:
    """Stochastic node with wide CI does not converge."""
    # Alpha=2, beta=2 → wide CI.
    assert convergence_gate("stochastic", 3, 2.0, 2.0) is False


@pytest.mark.unit
def test_convergence_gate_stochastic_n_max() -> None:
    """Stochastic node hits N_max even if CI is still wide."""
    assert convergence_gate("stochastic", 50, 2.0, 2.0, n_max=50) is True


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_credible_interval_bounds() -> None:
    lo, hi = credible_interval(10.0, 2.0)
    assert 0.0 <= lo < hi <= 1.0
    # Posterior mean is 10/12 ≈ 0.833; CI should straddle it.
    assert lo < 0.833 < hi


@pytest.mark.unit
def test_credible_width_uniform_prior() -> None:
    """Beta(1,1) is uniform → 95% CI width should be ~0.95."""
    w = credible_width(1.0, 1.0)
    assert 0.9 < w < 1.0


@pytest.mark.unit
def test_credible_width_narrows_with_evidence() -> None:
    """More evidence → narrower credible width."""
    w1 = credible_width(2.0, 2.0)
    w2 = credible_width(50.0, 50.0)
    assert w2 < w1


# ---------------------------------------------------------------------------
# Regime-aware transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_should_verify_deterministic_above_threshold(
    sampler: ThompsonSampler,
) -> None:
    node = make_node("n1", alpha=10.0, beta=1.0, evidence_regime="deterministic")
    assert sampler.should_verify(node, evidence_count=1) is True


@pytest.mark.unit
def test_should_verify_deterministic_below_threshold(
    sampler: ThompsonSampler,
) -> None:
    node = make_node("n1", alpha=1.0, beta=3.0, evidence_regime="deterministic")
    # Posterior mean = 0.25 < 0.8 → not verified.
    assert sampler.should_verify(node, evidence_count=1) is False


@pytest.mark.unit
def test_should_verify_goal_node_uses_target_metric(
    sampler: ThompsonSampler,
) -> None:
    """Goal nodes verify at their own target_metric, not the global threshold."""
    node = make_node(
        "g1",
        alpha=5.0,
        beta=2.0,  # mean ≈ 0.714
        is_goal=True,
        target_metric=0.6,
        evidence_regime="deterministic",
    )
    # 0.714 > 0.6 (target) but < 0.8 (global). Should verify because goal bar is lower.
    assert sampler.should_verify(node, evidence_count=1) is True


@pytest.mark.unit
def test_should_verify_goal_without_target_metric_uses_global(
    sampler: ThompsonSampler,
) -> None:
    """A goal that never declared a target must fall back to the global bar, not crash."""
    node = make_node(
        "g1",
        alpha=9.0,
        beta=1.0,  # mean = 0.9 > 0.8 global
        is_goal=True,
        target_metric=None,
        evidence_regime="deterministic",
    )
    # Must not raise (comparing against None) and must clear the global 0.8 bar.
    assert sampler.should_verify(node, evidence_count=1) is True

    below = make_node(
        "g2",
        alpha=3.0,
        beta=2.0,  # mean = 0.6 < 0.8 global
        is_goal=True,
        target_metric=None,
    )
    assert sampler.should_verify(below, evidence_count=1) is False


@pytest.mark.unit
def test_should_invalidate_stochastic_converged_low_mean(
    sampler: ThompsonSampler,
) -> None:
    """A stochastic node that has converged with a confidently-low mean invalidates."""
    node = make_node("n1", alpha=1.0, beta=20.0, evidence_regime="stochastic")
    # 50 samples → n_max convergence; mean ≈ 0.048 < epsilon_low (0.2).
    assert sampler.should_invalidate(node, evidence_count=50) is True


@pytest.mark.unit
def test_should_not_invalidate_stochastic_not_converged(
    sampler: ThompsonSampler,
) -> None:
    """A stochastic node with a wide CI (not converged) never invalidates."""
    node = make_node("n1", alpha=1.0, beta=4.0, evidence_regime="stochastic")
    assert sampler.should_invalidate(node, evidence_count=2) is False


@pytest.mark.unit
def test_should_not_invalidate_stochastic_converged_high_mean(
    sampler: ThompsonSampler,
) -> None:
    """Converged but a healthy mean → not confidently bad → no invalidation."""
    node = make_node("n1", alpha=5.0, beta=5.0, evidence_regime="stochastic")
    # n_max convergence at count=50, mean = 0.5, well above epsilon_low.
    assert sampler.should_invalidate(node, evidence_count=50) is False


@pytest.mark.unit
def test_should_invalidate_deterministic_one_failure(
    sampler: ThompsonSampler,
) -> None:
    """A deterministic node with one success=0.0 invalidates immediately."""
    node = make_node("n1", alpha=1.0, beta=2.0, evidence_regime="deterministic")
    assert sampler.should_invalidate(node, evidence_count=1, last_success=0.0) is True


@pytest.mark.unit
def test_should_not_invalidate_deterministic_nonzero(
    sampler: ThompsonSampler,
) -> None:
    """A deterministic node with success > 0.0 does not invalidate."""
    node = make_node("n1", alpha=3.0, beta=1.0, evidence_regime="deterministic")
    assert sampler.should_invalidate(node, evidence_count=1, last_success=0.5) is False


# ---------------------------------------------------------------------------
# SelectionResult / claim
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_issues_claim_id(sampler: ThompsonSampler) -> None:
    node = make_node("n1", alpha=5.0, beta=1.0)
    result = sampler.select([node], [node])
    assert result.status == "SELECTED"
    assert result.claim_id is not None
    assert len(result.claim_id) > 0
    assert result.credible_interval is not None


@pytest.mark.unit
def test_select_returns_credible_interval(sampler: ThompsonSampler) -> None:
    node = make_node("n1", alpha=10.0, beta=2.0)
    result = sampler.select([node], [node])
    assert result.credible_interval is not None
    lo, hi = result.credible_interval
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# RNG discipline — selection must not draw a second, wasted sample
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_does_not_waste_rng_draws() -> None:
    """select() must consume exactly one Beta draw per frontier node.

    The rationale used to draw a fresh theta after the pick, advancing the RNG
    and misreporting the value. This asserts the RNG has advanced by exactly
    len(frontier) draws, so the reported theta is the one that drove the pick.
    """
    nodes = [
        make_node("a", alpha=2.0, beta=2.0),
        make_node("b", alpha=3.0, beta=1.0),
    ]
    s = ThompsonSampler(np.random.default_rng(7))
    s.select(nodes, nodes)

    reference = np.random.default_rng(7)
    for n in nodes:
        reference.beta(n.alpha, n.beta)
    # If exactly len(nodes) draws were consumed, the next draw matches.
    assert s._rng.beta(1.0, 1.0) == reference.beta(1.0, 1.0)


@pytest.mark.unit
def test_select_rationale_reports_selection_theta() -> None:
    """The rationale theta must equal the theta actually drawn for the winner."""
    node = make_node("solo", alpha=4.0, beta=2.0)
    s = ThompsonSampler(np.random.default_rng(11))

    # Reproduce the single draw the sampler makes for the sole frontier node.
    expected_theta = np.random.default_rng(11).beta(4.0, 2.0)
    result = s.select([node], [node])
    assert result.rationale == f"theta={expected_theta:.4f}"


# ---------------------------------------------------------------------------
# Conclusiveness guard — EXHAUSTED
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_should_exhaust_deterministic_dead_zone(sampler: ThompsonSampler) -> None:
    """A deterministic reading between the two bars settles the node.

    0 < success <= verify_bar is conclusive for a deterministic node: it neither
    refutes nor verifies, and no further probe can change it. Without this the
    node stays IN_PROGRESS forever, sits permanently on the frontier, and is
    re-selected on every dispatch — the pathology that consumed most of the
    budget in the superseded run.
    """
    node = make_node("n1", alpha=1.35, beta=1.65, evidence_regime="deterministic")
    assert sampler.should_exhaust(node, evidence_count=1, last_success=0.35) is True


@pytest.mark.unit
def test_should_not_exhaust_without_a_reading(sampler: ThompsonSampler) -> None:
    """No logical observation → nothing was concluded → stay selectable."""
    node = make_node("n1", evidence_regime="deterministic")
    assert sampler.should_exhaust(node, evidence_count=0, last_success=None) is False


@pytest.mark.unit
def test_should_not_exhaust_stochastic_before_convergence(sampler: ThompsonSampler) -> None:
    """A stochastic node stays selectable while more evidence can still move it."""
    node = make_node("n1", alpha=1.5, beta=2.5, evidence_regime="stochastic")
    assert sampler.should_exhaust(node, evidence_count=2, last_success=0.4) is False


@pytest.mark.unit
def test_should_exhaust_stochastic_after_convergence(sampler: ThompsonSampler) -> None:
    """Once the convergence gate closes, a mid-band stochastic node is settled."""
    node = make_node("n1", alpha=20.0, beta=20.0, evidence_regime="stochastic")
    assert sampler.should_exhaust(node, evidence_count=50, last_success=0.5) is True


# ---------------------------------------------------------------------------
# Exclusion-aware tie-break
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tiebreak_prefers_the_question_already_in_progress(sampler: ThompsonSampler) -> None:
    """Untested nodes share the Beta(1,1) prior, so their draws tie.

    Among ties the sampler should stay on the question it was already working,
    rather than hopping to an unrelated one and leaving a half-answered question
    behind for the agent to re-derive later.
    """
    now = datetime.now(timezone.utc)
    # The other-group node is deliberately staler, so only the group preference
    # can produce the expected pick.
    same = make_node("same", updated_at=now - timedelta(seconds=10), exclusion_group="axis-1")
    other = make_node("other", updated_at=now - timedelta(seconds=999), exclusion_group="axis-2")

    result = sampler.select([other, same], [other, same], now=now, last_group="axis-1")
    assert result.node_id == "same"


@pytest.mark.unit
def test_tiebreak_falls_back_to_staleness_without_a_last_group(
    sampler: ThompsonSampler,
) -> None:
    """With no question in progress the anti-starvation rule is unchanged."""
    now = datetime.now(timezone.utc)
    fresh = make_node("fresh", updated_at=now - timedelta(seconds=10), exclusion_group="axis-1")
    stale = make_node("stale", updated_at=now - timedelta(seconds=999), exclusion_group="axis-2")

    result = sampler.select([fresh, stale], [fresh, stale], now=now, last_group=None)
    assert result.node_id == "stale"


@pytest.mark.unit
def test_tiebreak_never_overrides_a_real_theta_gap(sampler: ThompsonSampler) -> None:
    """Coherence only breaks ties — it must not outrank the posterior itself."""
    now = datetime.now(timezone.utc)
    promising = make_node("promising", alpha=50.0, beta=1.0, exclusion_group="axis-2")
    hopeless = make_node("hopeless", alpha=1.0, beta=50.0, exclusion_group="axis-1")

    result = sampler.select(
        [promising, hopeless], [promising, hopeless], now=now, last_group="axis-1"
    )
    assert result.node_id == "promising"
