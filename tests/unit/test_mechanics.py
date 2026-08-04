"""Tests for the mechanisms that turn a belief state into a search strategy.

Covers the closed-world group posterior, deduction by elimination, batch
dispatch, conflict-driven selection, and the goal-node guard. Each of these
exists because the previous evaluation run showed a specific, measured way the
belief state was failing to earn its keep.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hypotree.engine import (
    MAX_REVIEW_DISPATCHES,
    ClaimError,
    GoalDependencyError,
    GoalEvidenceError,
    HypoTreeEngine,
)
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status
from hypotree.navigator.sampler import (
    ThompsonSampler,
    effective_posterior,
    live_group_counts,
)


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    e = HypoTreeEngine(tmp_path / "mechanics.db", rng_seed=11)
    yield e
    e.close()


def _member(node_id: str, group: str | None, status: Status = Status.UNTESTED) -> Node:
    return Node(id=node_id, statement=node_id, status=status, exclusion_group=group)


# -- closed-world group posterior ---------------------------------------------


@pytest.mark.unit
def test_live_group_counts_ignores_refuted_members() -> None:
    nodes = [
        _member("a1", "axis"),
        _member("a2", "axis", Status.INVALIDATED),
        _member("a3", "axis", Status.PRUNED),
        _member("a4", "axis", Status.EXHAUSTED),
        _member("loose", None),
    ]
    # EXHAUSTED still counts: it was settled by inference or scored below the
    # bar, neither of which rules it out as the group's answer the way a
    # refutation does.
    assert live_group_counts(nodes) == {"axis": 2}


@pytest.mark.unit
def test_group_prior_is_the_categorical_marginal() -> None:
    """One of k candidates is true, so each starts at probability 1/k."""
    counts = {"axis": 4}
    alpha, beta = effective_posterior(_member("a1", "axis"), counts)
    assert alpha / (alpha + beta) == pytest.approx(0.25)

    counts = {"axis": 2}
    alpha, beta = effective_posterior(_member("a1", "axis"), counts)
    assert alpha / (alpha + beta) == pytest.approx(0.5)


@pytest.mark.unit
def test_ungrouped_node_keeps_the_uniform_prior() -> None:
    alpha, beta = effective_posterior(_member("loose", None), {})
    assert (alpha, beta) == (1.0, 1.0)


@pytest.mark.unit
def test_group_prior_carries_observations_through() -> None:
    """Swapping the prior must not discard evidence already folded in."""
    node = _member("a1", "axis")
    node.alpha, node.beta = 3.0, 2.0  # two successes, one failure over Beta(1,1)
    alpha, beta = effective_posterior(node, {"axis": 4})
    assert (alpha, beta) == (3.0, 4.0)  # beta gains the prior's extra k-2


@pytest.mark.unit
def test_narrower_question_outranks_an_untouched_one() -> None:
    """A question with one candidate eliminated is closer to being answered.

    This is what produces one-factor-at-a-time behaviour without any explicit
    rule for it: the group the agent has already invested in has the higher
    probability of resolving on the next probe, so Thompson Sampling returns to
    it rather than interleaving every question at random.
    """
    sampler = ThompsonSampler(rng=np.random.default_rng(0))
    all_nodes = [
        _member("narrow_1", "narrow"),
        _member("narrow_2", "narrow", Status.INVALIDATED),
        _member("narrow_3", "narrow", Status.INVALIDATED),
        *[_member(f"wide_{i}", "wide") for i in range(4)],
    ]
    frontier = [n for n in all_nodes if n.status == Status.UNTESTED]

    picks = [sampler.select(frontier, all_nodes).node_id for _ in range(200)]
    narrow = sum(1 for p in picks if p == "narrow_1")
    # One candidate out of four wide ones would win ~20% of the time on a
    # uniform prior; the narrowed question must win far more often than that.
    assert narrow > 0.45 * len(picks)


# -- deduction by elimination --------------------------------------------------


@pytest.mark.unit
def test_last_surviving_member_is_deduced(engine: HypoTreeEngine) -> None:
    """Refuting every alternative confirms the survivor without a probe."""
    for nid in ("a1", "a2", "a3"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="axis")

    engine.record_evidence("a1", LogicalEvidence(success=0.0))
    assert engine._store.get_node("a3").status == Status.UNTESTED
    engine.record_evidence("a2", LogicalEvidence(success=0.0))

    survivor = engine._store.get_node("a3")
    assert survivor.status == Status.VERIFIED
    # No observation was made, so the posterior must not pretend otherwise.
    assert survivor.evidence_count == 0
    assert (survivor.alpha, survivor.beta) == (1.0, 1.0)


@pytest.mark.unit
def test_deduction_needs_an_actual_refutation(engine: HypoTreeEngine) -> None:
    """A group nobody has touched is never collapsed onto an arbitrary member."""
    engine.create_hypothesis("solo", node_id="solo", exclusion_group="axis")
    assert engine._deduce_last_member("axis", datetime.now(timezone.utc)) is None
    assert engine._store.get_node("solo").status == Status.UNTESTED


@pytest.mark.unit
def test_deduction_does_not_override_contrary_evidence(engine: HypoTreeEngine) -> None:
    """An exhausted survivor contradicts the group's premise — say so, don't paper over it.

    If every alternative is refuted and the last one was itself tested and found
    wanting, the exclusion group was mis-declared. Inventing a confirmation would
    bury exactly the signal that says so.
    """
    engine.create_hypothesis("a1", node_id="a1", exclusion_group="axis")
    engine.create_hypothesis("a2", node_id="a2", exclusion_group="axis")

    engine.record_evidence("a2", LogicalEvidence(success=0.5))  # conclusive, sub-bar
    assert engine._store.get_node("a2").status == Status.EXHAUSTED
    engine.record_evidence("a1", LogicalEvidence(success=0.0))

    assert engine._store.get_node("a2").status == Status.EXHAUSTED


@pytest.mark.unit
def test_deduction_unblocks_a_dependent_combination(engine: HypoTreeEngine) -> None:
    """The point of deducing is that everything downstream becomes reachable."""
    for nid in ("a1", "a2"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="axis")
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a2"], edge_type=EdgeType.DEPENDENCY
    )

    engine.record_evidence("a1", LogicalEvidence(success=0.0))

    assert engine._store.get_node("a2").status == Status.VERIFIED
    assert "combo" in {n.id for n in engine._frontier_nodes()}


@pytest.mark.unit
def test_a_conclusively_sub_par_member_counts_as_ruled_out(engine: HypoTreeEngine) -> None:
    """EXHAUSTED by its own evidence eliminates a member as firmly as refutation.

    The question was put and the answer did not clear the bar. Requiring an
    outright 0.0 left groups one probe short of an entailed answer whenever any
    member happened to land in the dead zone instead of at zero.
    """
    for nid in ("a1", "a2", "a3"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="axis")

    engine.record_evidence("a1", LogicalEvidence(success=0.5))  # conclusive, sub-bar
    assert engine._store.get_node("a1").status == Status.EXHAUSTED
    engine.record_evidence("a2", LogicalEvidence(success=0.0))

    assert engine._store.get_node("a3").status == Status.VERIFIED


@pytest.mark.unit
def test_exhausting_the_second_to_last_member_deduces_the_survivor(
    engine: HypoTreeEngine,
) -> None:
    """Elimination is checked when a member is exhausted, not only when refuted."""
    for nid in ("a1", "a2"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="axis")

    engine.record_evidence("a1", LogicalEvidence(success=0.5))

    assert engine._store.get_node("a2").status == Status.VERIFIED


@pytest.mark.unit
def test_an_excluded_sibling_is_not_treated_as_ruled_out(engine: HypoTreeEngine) -> None:
    """A member set aside by the exclusion inference was never observed.

    Counting it as eliminated would let a single confirmation deduce the rest of
    its own group from itself, manufacturing confirmations nothing supports.
    """
    for nid in ("a1", "a2", "a3"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="axis")

    engine.record_evidence("a1", LogicalEvidence(success=1.0))

    assert engine._store.get_node("a1").status == Status.VERIFIED
    for nid in ("a2", "a3"):
        assert engine._store.get_node(nid).status == Status.EXHAUSTED


# -- goal guard ----------------------------------------------------------------


@pytest.mark.unit
def test_a_goal_refuses_evidence_rather_than_absorbing_it(engine: HypoTreeEngine) -> None:
    """A goal states an objective, not a claim that could turn out false.

    Agents repeatedly recorded a failed composition's score against the goal.
    Silently accepting it was the worst of both worlds: the goal could not be
    refuted (which would cascade-prune the very subtree meant to achieve it), so
    the reading only ever nudged it toward success, while the result that
    mattered — which composition failed, and on what assumptions — was thrown
    away. Refusing the record keeps the failure attributable.
    """
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    engine.create_hypothesis("child", node_id="child")

    with pytest.raises(GoalEvidenceError):
        engine.record_evidence("goal", LogicalEvidence(success=0.0))

    goal = engine._store.get_node("goal")
    assert goal.status != Status.INVALIDATED
    assert goal.evidence_count == 0
    assert engine._store.get_node("child").status != Status.PRUNED


@pytest.mark.unit
def test_a_goal_is_reached_through_what_supports_it(engine: HypoTreeEngine) -> None:
    """The refusal must not make a goal unreachable, only unprobeable."""
    engine.create_hypothesis("combo", node_id="combo")
    engine.create_hypothesis(
        "goal",
        node_id="goal",
        is_goal=True,
        target_metric=0.75,
        parent_ids=["combo"],
        edge_type=EdgeType.DEPENDENCY,
    )

    assert engine.goal_achieved(engine._store.get_node("goal")) is False
    engine.record_evidence("combo", LogicalEvidence(success=0.9))
    assert engine.goal_achieved(engine._store.get_node("goal")) is True


@pytest.mark.unit
def test_a_goal_is_never_handed_out_as_a_target(engine: HypoTreeEngine) -> None:
    """Dispatching a goal is an instruction the caller cannot carry out.

    Nothing settles a goal, so it stayed dispatchable no matter what came back
    and the navigator offered it turn after turn. Each of those turns cost a
    probe whose result was then filed against the goal and lost.
    """
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    engine.create_hypothesis("h1", node_id="h1")

    engine.record_evidence("h1", LogicalEvidence(success=0.0))

    assert engine._frontier_nodes() == []
    assert engine.get_next_targets()[0].status == "DONE"


# -- composition ---------------------------------------------------------------


@pytest.mark.unit
def test_settled_questions_ask_for_a_composition_not_an_ending(
    engine: HypoTreeEngine,
) -> None:
    """Every question answered is the middle of the task, not the end of it.

    Reporting a bare empty frontier here ends the run at precisely the point the
    answers are ready to be put together, which is the one move left worth
    making.
    """
    for axis in ("alpha", "beta"):
        engine.create_hypothesis(f"{axis}=v0", node_id=f"{axis}_v0", exclusion_group=axis)
        engine.create_hypothesis(f"{axis}=v1", node_id=f"{axis}_v1", exclusion_group=axis)
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)

    engine.record_evidence("alpha_v0", LogicalEvidence(success=1.0))
    engine.record_evidence("beta_v0", LogicalEvidence(success=1.0))

    done = engine.get_next_targets()[0]
    assert done.status == "DONE"
    assert done.reason == "awaiting_composition"
    # The rationale must name the parents, or the caller cannot wire them.
    assert "alpha_v0" in done.rationale and "beta_v0" in done.rationale


@pytest.mark.unit
def test_a_composed_answer_is_not_proposed_for_composition_again(
    engine: HypoTreeEngine,
) -> None:
    """Once the confirmed answers have been combined, repeating the advice loops."""
    engine.create_hypothesis("a=v0", node_id="a_v0", exclusion_group="a")
    engine.record_evidence("a_v0", LogicalEvidence(success=1.0))

    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a_v0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal", node_id="goal", is_goal=True, target_metric=0.75, parent_ids=["combo"]
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.5))  # settled, sub-bar

    # A single-parent composition has no competing assumption to reopen, so the
    # sub-par result really is the end of this line of enquiry.
    assert engine.get_next_targets()[0].reason == "empty_frontier"


@pytest.mark.unit
def test_nothing_confirmed_is_an_empty_frontier_not_a_composition(
    engine: HypoTreeEngine,
) -> None:
    """With no confirmed answers there is nothing to compose, and the run is over."""
    engine.create_hypothesis("h1", node_id="h1")
    engine.record_evidence("h1", LogicalEvidence(success=0.0))

    assert engine.get_next_targets()[0].reason == "empty_frontier"


# -- batch dispatch ------------------------------------------------------------


@pytest.mark.unit
def test_batch_dispatch_returns_distinct_claimed_targets(engine: HypoTreeEngine) -> None:
    for i in range(4):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")

    targets = engine.get_next_targets(count=3)

    assert len(targets) == 3
    assert len({t.node_id for t in targets}) == 3
    assert all(t.claim_id for t in targets)
    assert len({t.claim_id for t in targets}) == 3


@pytest.mark.unit
def test_batch_dispatch_stops_at_the_frontier_edge(engine: HypoTreeEngine) -> None:
    """Asking for more than exists returns what exists, not a DONE sentinel."""
    engine.create_hypothesis("only", node_id="only")

    targets = engine.get_next_targets(count=5)

    assert [t.status for t in targets] == ["SELECTED"]


@pytest.mark.unit
def test_batch_dispatch_reports_done_only_when_empty(engine: HypoTreeEngine) -> None:
    targets = engine.get_next_targets(count=3)
    assert [t.status for t in targets] == ["DONE"]
    assert targets[0].reason == "empty_frontier"


@pytest.mark.unit
def test_dry_run_never_claims_more_than_one(engine: HypoTreeEngine) -> None:
    """Without claims every pick would see the same frontier and repeat itself."""
    for i in range(4):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")

    targets = engine.get_next_targets(count=3, dry_run=True)

    assert len(targets) == 1
    assert targets[0].claim_id is None
    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_batch_dispatch_rejects_a_meaningless_count(engine: HypoTreeEngine) -> None:
    with pytest.raises(ValueError):
        engine.get_next_targets(count=0)


@pytest.mark.unit
def test_a_batch_never_asks_one_question_twice(engine: HypoTreeEngine) -> None:
    """Competing answers dispatched together defeat the exclusion inference.

    Confirming one member of an exclusion group retires the rest for free, so a
    batch holding two members of the same group has already committed to
    spending a probe the first result would have saved. The tie-break that keeps
    *sequential* dispatch on one question made this the common case, and a real
    run handed out three members of one axis in a single batch.
    """
    for axis in ("alpha", "beta"):
        for i in range(4):
            engine.create_hypothesis(f"{axis}={i}", node_id=f"{axis}_{i}", exclusion_group=axis)

    targets = engine.get_next_targets(count=4)

    groups = [engine._store.get_node(t.node_id).exclusion_group for t in targets]
    assert len(groups) == len(set(groups)), groups
    # Only two questions exist, so a request for four is answered with two.
    assert sorted(groups) == ["alpha", "beta"]


@pytest.mark.unit
def test_ungrouped_nodes_are_not_treated_as_one_question(engine: HypoTreeEngine) -> None:
    """A missing exclusion group means 'no group', not 'all in the same group'."""
    for i in range(3):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")

    targets = engine.get_next_targets(count=3)

    assert len(targets) == 3


@pytest.mark.unit
def test_the_next_batch_may_revisit_a_question_once_it_is_answered(
    engine: HypoTreeEngine,
) -> None:
    """The constraint is per batch, not a permanent ban on a group."""
    for i in range(3):
        engine.create_hypothesis(f"a={i}", node_id=f"a_{i}", exclusion_group="a")

    first = engine.get_next_targets(count=3)
    assert len(first) == 1
    engine.record_evidence(first[0].node_id, LogicalEvidence(success=0.0), first[0].claim_id)

    second = engine.get_next_targets(count=3)
    assert len(second) == 1
    assert second[0].node_id != first[0].node_id


def _build_conflicted_pair(engine: HypoTreeEngine) -> None:
    """Two answered questions and a combination of them that fails at depth."""
    for group, ids in (("comp", ("c1", "c2")), ("reg", ("r1", "r2"))):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["c1", "r1"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("c1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("r1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))


# -- conflict-driven selection -------------------------------------------------


@pytest.mark.unit
def test_selection_prioritises_a_convicted_questions_alternatives(
    engine: HypoTreeEngine,
) -> None:
    """Once a conflict names a culprit, its rivals outrank any fresh hypothesis.

    The answer to that question is almost certainly among the alternatives the
    convicted value had retired, and every one of them carries an untouched
    prior — so without a priority the navigator picks among them and a pile of
    unrelated work at random.

    The conviction here is an outright refutation. A conviction reached by a
    successful *swap* also names the replacement, so it confirms the answer
    outright and leaves nothing for the navigator to prioritise — three
    alternatives, so refuting one still leaves a real choice rather than a lone
    survivor to be deduced.
    """
    for group, ids in (("comp", ("c1", "c2", "c3")), ("reg", ("r1", "r2"))):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    for i in range(5):
        engine.create_hypothesis(f"unrelated{i}", node_id=f"unrelated{i}")
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["c1", "r1"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("c1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("r1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    engine.record_evidence("c1", LogicalEvidence(success=0.0, depth=2))
    assert engine._store.get_node("c1").status == Status.INVALIDATED

    targets = engine.get_next_targets(count=1)

    assert targets[0].node_id in ("c2", "c3")
    assert targets[0].min_depth == 2


@pytest.mark.unit
def test_selection_is_normal_without_conflicts(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="h1")
    target = engine.get_next_targets()[0]
    assert target.node_id == "h1"
    assert target.min_depth is None


# -- history integrity ---------------------------------------------------------


@pytest.mark.unit
def test_reasserting_a_status_is_not_a_transition(engine: HypoTreeEngine) -> None:
    """Re-confirming an existing belief must not corrupt the history.

    Deepening a shallow confirmation is a normal action, and it lands on a node
    that is already VERIFIED. Recording that as a transition would open a second
    interval at the instant the first one closes.
    """
    engine.create_hypothesis("n1", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=1))
    before = len(engine._store.get_status_history("n1"))

    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=2))

    assert engine._store.get_node("n1").status == Status.VERIFIED
    assert len(engine._store.get_status_history("n1")) == before
    # Exactly one interval may be open at a time.
    open_intervals = [r for r in engine._store.get_status_history("n1") if r["valid_to"] is None]
    assert len(open_intervals) == 1


@pytest.mark.unit
def test_confirmation_depth_only_ratchets_upward(engine: HypoTreeEngine) -> None:
    """A shallower re-test never weakens what a deeper one established."""
    engine.create_hypothesis("n1", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=3))
    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=1))
    assert engine._store.get_node("n1").confirmed_depth == 3


@pytest.mark.unit
def test_refutation_withdraws_the_confirmation_depth(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("n1", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=2))
    engine.record_evidence("n1", LogicalEvidence(success=0.0, depth=2))
    assert engine._store.get_node("n1").confirmed_depth is None


@pytest.mark.unit
def test_stale_tiebreak_still_applies_within_a_group() -> None:
    """The coherence tiebreak must survive the prior change."""
    sampler = ThompsonSampler(rng=np.random.default_rng(3))
    now = datetime.now(timezone.utc)
    fresh = _member("fresh", "axis")
    stale = _member("stale", "axis")
    fresh.updated_at = now - timedelta(seconds=1)
    stale.updated_at = now - timedelta(seconds=9999)

    picks = [sampler.select([fresh, stale], [fresh, stale], now=now).node_id for _ in range(50)]
    assert "stale" in picks


@pytest.mark.unit
def test_available_work_is_never_starved_by_a_conflict(engine: HypoTreeEngine) -> None:
    """A conflict waits its turn: real work outranks a diagnostic instruction.

    The swap is only useful once the questions still open have been answered —
    they may change which combination is worth building — so the advice appears
    exactly when nothing else is dispatchable, and cannot hold the frontier
    hostage in the meantime.
    """
    _build_conflicted_pair(engine)
    engine.create_hypothesis("other", node_id="other")

    target = engine.get_next_targets()[0]

    assert target.status == "SELECTED"
    assert target.node_id == "other"


@pytest.mark.unit
def test_advice_that_is_never_acted_on_stops_being_repeated(engine: HypoTreeEngine) -> None:
    """Advice is a bet that the caller acts on it; a losing bet is dropped.

    A conflict whose diagnostic swap is never run would otherwise repeat the
    same instruction for the rest of the episode.
    """
    _build_conflicted_pair(engine)

    reasons = [engine.get_next_targets()[0].reason for _ in range(MAX_REVIEW_DISPATCHES + 1)]

    assert reasons[0] == "awaiting_substitution"
    assert reasons[-1] != "awaiting_substitution"
    # Having given up on the targeted route, it falls back to the broad one.
    assert engine._store.get_nogoods()[0]["reopened_at"] is not None


@pytest.mark.unit
def test_a_leased_node_is_not_dispatched_again(engine: HypoTreeEngine) -> None:
    """A claim reserves the node. Otherwise it is decoration, not a lease."""
    engine.create_hypothesis("h1", node_id="h1")

    first = engine.get_next_targets()[0]
    second = engine.get_next_targets()[0]

    assert first.node_id == "h1"
    assert second.status == "DONE"
    # Held, not finished — the caller is holding the only node there is.
    assert second.reason == "awaiting_evidence"


@pytest.mark.unit
def test_a_node_never_holds_two_live_claims(engine: HypoTreeEngine) -> None:
    """Two live leases on one node mean two callers can each spend one."""
    engine.create_hypothesis("h1", node_id="h1")

    first = engine.get_next_targets()[0]
    engine.release_claims()
    second = engine.get_next_targets()[0]

    live = engine.get_active_claims()
    assert len(live) == 1
    assert live[0].claim_id == second.claim_id
    # The superseded lease is dead and can no longer be spent.
    with pytest.raises(ClaimError):
        engine.record_evidence("h1", LogicalEvidence(success=1.0), claim_id=first.claim_id)


@pytest.mark.unit
def test_recording_evidence_returns_the_node_to_the_pool(engine: HypoTreeEngine) -> None:
    """Consuming the lease is what makes a node dispatchable again.

    A stochastic node needs repeated sampling, so the release must happen on
    consumption rather than only on expiry.
    """
    engine.create_hypothesis("h1", node_id="h1", evidence_regime="stochastic")

    first = engine.get_next_targets()[0]
    engine.record_evidence("h1", LogicalEvidence(success=0.6), claim_id=first.claim_id)
    second = engine.get_next_targets()[0]

    assert second.node_id == "h1"
    assert second.claim_id != first.claim_id


@pytest.mark.unit
def test_batching_without_reporting_cannot_duplicate_work(engine: HypoTreeEngine) -> None:
    """The exact failure mode observed in run 20260728b, in miniature."""
    for i in range(4):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")

    dispatched = [t.node_id for t in engine.get_next_targets(count=4)]
    dispatched += [t.node_id for t in engine.get_next_targets(count=4) if t.node_id]

    assert sorted(dispatched) == ["h0", "h1", "h2", "h3"]


@pytest.mark.unit
def test_releasing_claims_hands_the_work_back(engine: HypoTreeEngine) -> None:
    """A caller whose context was wiped cannot report; stranding its nodes is worse."""
    for i in range(3):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")
    engine.get_next_targets(count=3)
    assert engine.get_next_targets()[0].status == "DONE"

    released = engine.release_claims()

    assert sorted(released) == ["h0", "h1", "h2"]
    assert engine.get_active_claims() == []
    assert engine.get_next_targets()[0].status == "SELECTED"


@pytest.mark.unit
def test_an_expired_lease_stops_blocking(engine: HypoTreeEngine) -> None:
    """The TTL is the backstop for an agent that simply walks away."""
    engine.create_hypothesis("h1", node_id="h1")
    engine.get_next_targets(lease_ttl_s=0)

    assert engine.get_next_targets()[0].node_id == "h1"


# -- goal dispatch priority ----------------------------------------------------


@pytest.mark.unit
def test_a_goal_yields_the_frontier_to_real_hypotheses(engine: HypoTreeEngine) -> None:
    """A goal without dependencies is eligible from step one and settles never.

    Left at equal priority it consumed a batch slot on every dispatch — nine
    times in one observed episode — for a node no experiment can resolve.
    """
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    for i in range(3):
        engine.create_hypothesis(f"h{i}", node_id=f"h{i}")

    batch = engine.get_next_targets(count=3)

    assert "goal" not in {t.node_id for t in batch}


@pytest.mark.unit
def test_a_lone_goal_is_not_dispatched_either(engine: HypoTreeEngine) -> None:
    """An empty frontier is the honest answer when only a goal remains.

    Offering it as a last resort merely postponed the trap to the moment the
    rest of the frontier emptied — which is exactly when the caller was most
    likely to probe a composition and file the result against the goal.
    """
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)

    done = engine.get_next_targets()[0]
    assert done.status == "DONE"
    assert done.node_id is None


@pytest.mark.unit
def test_holding_every_node_is_not_an_end_state(engine: HypoTreeEngine) -> None:
    """ "You are holding everything" is an instruction to report, not a conclusion.

    Collapsing it into `empty_frontier` told a batching agent its work was
    finished the moment it got ahead of its own bookkeeping — and the harness
    then ended the run.
    """
    engine.create_hypothesis("h1", node_id="h1")
    engine.get_next_targets()

    done = engine.get_next_targets()[0]

    assert done.status == "DONE"
    assert done.reason == "awaiting_evidence"
    assert "record evidence" in done.rationale


@pytest.mark.unit
def test_a_genuinely_settled_frontier_still_reports_empty(engine: HypoTreeEngine) -> None:
    """With nothing outstanding, an empty frontier really is the end."""
    engine.create_hypothesis("h1", node_id="h1")
    target = engine.get_next_targets()[0]
    engine.record_evidence("h1", LogicalEvidence(success=0.0), claim_id=target.claim_id)

    done = engine.get_next_targets()[0]

    assert done.status == "DONE"
    assert done.reason == "empty_frontier"


@pytest.mark.unit
def test_goal_status_reflects_edges_after_a_reload(tmp_path: Path) -> None:
    """Goal achievement is read off the graph, so the graph must be current.

    A freshly constructed engine has not loaded its edges yet. Reporting goal
    status before syncing would answer from an empty structure and declare a
    reached goal unmet on every reload.
    """
    db = tmp_path / "reload.db"
    first = HypoTreeEngine(db, rng_seed=3)
    first.create_hypothesis("combo", node_id="combo")
    first.create_hypothesis(
        "goal",
        node_id="goal",
        is_goal=True,
        target_metric=0.75,
        parent_ids=["combo"],
        edge_type=EdgeType.DEPENDENCY,
    )
    first.record_evidence("combo", LogicalEvidence(success=1.0))
    assert first.get_goal_status().all_met is True
    first.close()

    reloaded = HypoTreeEngine(db, rng_seed=3)
    try:
        assert reloaded.get_goal_status().all_met is True
    finally:
        reloaded.close()


# -- unreachable graphs --------------------------------------------------------


@pytest.mark.unit
def test_a_goal_cannot_be_something_elses_premise(engine: HypoTreeEngine) -> None:
    """The one modelling mistake strong models make reliably, refused where it is made.

    "The goal decomposes into phases" is how everyone thinks about objectives, so
    `parent_ids=[goal]` on the phase reads as "this phase belongs to that goal".
    In hypotree it means the opposite: the phase cannot be tested until the goal
    is VERIFIED. A goal is never dispatched and `verify_upstream` only promotes
    IN_PROGRESS nodes along REFINEMENT edges, so the goal can never reach
    VERIFIED and the child is blocked forever — while the goal still depends on
    nothing and so can never be reached either. Both halves broken, silently.

    Documentation lost this argument against a very capable model. Refusing the
    edge does not, because the message arrives while the caller still holds the
    intent, and it carries the corrected call.
    """
    engine.create_hypothesis("ship it", node_id="goal", is_goal=True, target_metric=0.8)

    with pytest.raises(GoalDependencyError) as exc:
        engine.create_hypothesis(
            "phase0", node_id="phase0", parent_ids=["goal"], edge_type=EdgeType.DEPENDENCY
        )
    assert "parent, not its child" in str(exc.value)
    assert '"parent_ids": ["phase0"]' in str(exc.value)
    # Refused means refused: no half-created node left behind.
    assert engine._store.get_node("phase0") is None

    # The right way round is untouched.
    engine.create_hypothesis("phase0", node_id="phase0")
    engine.create_hypothesis(
        "ship it",
        node_id="goal",
        is_goal=True,
        target_metric=0.8,
        parent_ids=["phase0"],
        if_exists="overwrite",
    )
    assert engine._graph.parents("goal", EdgeType.DEPENDENCY) == ["phase0"]


@pytest.mark.unit
def test_a_goal_may_still_refine_or_alternate(engine: HypoTreeEngine) -> None:
    """Only DEPENDENCY is refused, because only DEPENDENCY gates a child on it.

    The other two edge types carry no "parent must be VERIFIED first" rule, so
    they create no dead branch and there is nothing to protect the caller from.
    """
    engine.create_hypothesis("ship it", node_id="goal", is_goal=True, target_metric=0.8)
    engine.create_hypothesis(
        "a narrower reading", node_id="narrow", parent_ids=["goal"], edge_type=EdgeType.REFINEMENT
    )
    assert engine._store.get_node("narrow") is not None


@pytest.mark.unit
def test_an_unreachable_graph_is_reported_as_blocked_not_finished(
    engine: HypoTreeEngine,
) -> None:
    """ "Nothing is testable" and "nothing is reachable" are opposite situations.

    A premise wired as a *child* of the combination that assumes it can never be
    dispatched, because a DEPENDENCY parent must be confirmed first and the
    combination cannot be confirmed without the premise. A real episode ended at
    step zero this way, with twenty-five untested hypotheses in the store, and
    was scored as a completed search.

    Built from two ordinary nodes rather than from a goal: the goal form of this
    mistake is refused at creation now, so this covers the general case that is
    still representable.
    """
    engine.create_hypothesis("combo", node_id="combo")
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    for i in range(3):
        engine.create_hypothesis(
            f"p{i}", node_id=f"p{i}", parent_ids=["combo"], edge_type=EdgeType.DEPENDENCY
        )

    # EXHAUSTED settles the root without cascading, so its children stay untested
    # and gated on a parent that will never be VERIFIED — nothing is reachable
    # while three hypotheses sit unprobed, which is the situation under test.
    for _ in range(6):
        engine.record_evidence("combo", LogicalEvidence(success=0.2, depth=1))
    assert engine._store.get_node("combo").status == Status.EXHAUSTED

    done = engine.get_next_targets()[0]

    assert done.status == "DONE"
    assert done.reason == "blocked_frontier"
    # The caller is told which nodes and what gates them, not merely that it is over.
    assert "p0" in done.rationale and "combo" in done.rationale


@pytest.mark.unit
def test_a_genuinely_settled_graph_is_not_reported_as_blocked(
    engine: HypoTreeEngine,
) -> None:
    """The distinction only helps if a finished search still reads as finished."""
    engine.create_hypothesis("h1", node_id="h1")
    engine.record_evidence("h1", LogicalEvidence(success=0.0))

    assert engine.get_next_targets()[0].reason == "empty_frontier"


@pytest.mark.unit
def test_blocked_nodes_become_available_once_their_parent_is_confirmed(
    engine: HypoTreeEngine,
) -> None:
    """Blocked is a temporary state, so the report must stop once it is untrue."""
    engine.create_hypothesis("base", node_id="base")
    engine.create_hypothesis(
        "child", node_id="child", parent_ids=["base"], edge_type=EdgeType.DEPENDENCY
    )

    engine.record_evidence("base", LogicalEvidence(success=1.0))

    assert engine._blocked_nodes() == []
    assert engine.get_next_targets()[0].node_id == "child"


@pytest.mark.unit
def test_a_question_with_an_answer_in_flight_is_not_asked_again(
    engine: HypoTreeEngine,
) -> None:
    """A leased answer settles its siblings the moment it comes back.

    Scoping this to a single batch was enough while dispatch and record
    alternated in fixed pairs. Once a record can carry its own dispatch the
    ordinary loop hands out one target per call, and every such waste falls in
    the gap *between* two calls rather than inside one.
    """
    for nid in ("c1", "c2", "c3"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="component")
    engine.create_hypothesis("unrelated", node_id="u1")

    dispatched = [engine.get_next_targets(count=1)[0].node_id for _ in range(2)]

    # Whichever the sampler reached for first, the second must not be another
    # answer to a question already out for one.
    assert len([n for n in dispatched if n.startswith("c")]) <= 1
    assert len(set(dispatched)) == 2


@pytest.mark.unit
def test_a_settled_question_frees_its_group_again(engine: HypoTreeEngine) -> None:
    """The block is on the lease, not on the group: reporting releases it.

    And what it releases is usually nothing to do — the confirmation retires the
    siblings — which is exactly the probe the block was protecting.
    """
    for nid in ("c1", "c2"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="component")

    first = engine.get_next_targets(count=1)[0]
    assert engine.get_next_targets(count=1)[0].reason == "awaiting_evidence"

    engine.record_evidence(first.node_id, LogicalEvidence(success=1.0), claim_id=first.claim_id)
    sibling = next(n for n in ("c1", "c2") if n != first.node_id)
    assert engine._store.get_node(sibling).status == Status.EXHAUSTED
