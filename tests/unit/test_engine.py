"""Unit tests for the HypoTreeEngine — Tools API, claim lifecycle, transitions,
cascading prune, upstream propagation, DONE sentinel, Mermaid render."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from hypotree.engine import (
    ClaimError,
    GoalEvidenceError,
    HypoTreeEngine,
    NodeNotFoundError,
)
from hypotree.graph.dag import CycleError
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import InfraError, LogicalEvidence
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    e = HypoTreeEngine(tmp_path / "test.db", rng_seed=42)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# create_hypothesis
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_hypothesis_basic(engine: HypoTreeEngine) -> None:
    result = engine.create_hypothesis("test hypothesis")
    assert result.created is True
    node = result.node
    assert node.statement == "test hypothesis"
    assert node.status == Status.UNTESTED
    assert node.parent_ids == []


@pytest.mark.unit
def test_create_hypothesis_with_goal(engine: HypoTreeEngine) -> None:
    result = engine.create_hypothesis(
        "the goal", is_goal=True, target_metric=0.9, evidence_regime="stochastic"
    )
    node = result.node
    assert node.is_goal is True
    assert node.target_metric == 0.9
    assert node.evidence_regime == "stochastic"


@pytest.mark.unit
def test_create_hypothesis_with_parent(engine: HypoTreeEngine) -> None:
    parent = engine.create_hypothesis("parent")
    child = engine.create_hypothesis(
        "child", parent_ids=[parent.node.id], edge_type=EdgeType.DEPENDENCY
    )
    assert child.node.parent_ids == [parent.node.id]


@pytest.mark.unit
def test_create_hypothesis_cycle_rejected(engine: HypoTreeEngine) -> None:
    a = engine.create_hypothesis("A")
    b = engine.create_hypothesis("B", parent_ids=[a.node.id])
    # Use overwrite to bypass the collision guard and reach the cycle check.
    with pytest.raises(CycleError):
        engine.create_hypothesis(
            "A2", parent_ids=[b.node.id], node_id=a.node.id, if_exists="overwrite"
        )


# ---------------------------------------------------------------------------
# get_next_targets
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_next_targets_selects_node(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1")
    result = engine.get_next_targets()[0]
    assert result.status == "SELECTED"
    assert result.node_id is not None
    assert result.claim_id is not None
    assert result.credible_interval is not None
    assert result.statement == "h1"


@pytest.mark.unit
def test_get_next_targets_transitions_to_in_progress(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    engine.get_next_targets()[0]
    node = engine._store.get_node("n1")
    assert node is not None
    assert node.status == Status.IN_PROGRESS


@pytest.mark.unit
def test_get_next_targets_done_empty(engine: HypoTreeEngine) -> None:
    result = engine.get_next_targets()[0]
    assert result.status == "DONE"
    assert result.reason == "empty_frontier"


@pytest.mark.unit
def test_get_next_targets_done_all_goals_met(engine: HypoTreeEngine) -> None:
    """A goal is reached when the work supporting it is confirmed."""
    engine.create_hypothesis("combo", node_id="combo")
    engine.create_hypothesis(
        "goal",
        is_goal=True,
        target_metric=0.8,
        node_id="g1",
        parent_ids=["combo"],
        edge_type=EdgeType.DEPENDENCY,
    )

    engine.record_evidence("combo", LogicalEvidence(success=1.0))

    result = engine.get_next_targets()[0]
    assert result.status == "DONE"
    assert result.reason == "all_goals_met"


@pytest.mark.unit
def test_a_goal_nothing_supports_is_not_met(engine: HypoTreeEngine) -> None:
    """A high posterior on a goal is not achievement.

    Nothing has been proposed to reach it, so there is nothing to have worked.
    Reading the goal's own posterior counted attempts as progress.
    """
    engine.create_hypothesis("goal", is_goal=True, target_metric=0.8, node_id="g1")
    node = engine._store.get_node("g1")
    assert node is not None
    node.status = Status.VERIFIED
    node.alpha, node.beta = 10.0, 1.0
    engine._store.save_node(node)
    engine._sync_graph_from_store()

    assert engine.goal_achieved(node) is False
    # And the caller is told *why* nothing can satisfy it, rather than that the
    # search is over: a goal wired to nothing depends on nothing.
    assert engine.get_next_targets()[0].reason == "unreachable_goal"


# ---------------------------------------------------------------------------
# record_evidence — claim lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_evidence_consumes_claim(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    target = engine.get_next_targets()[0]
    assert target.claim_id is not None

    engine.record_evidence("n1", LogicalEvidence(success=0.9), claim_id=target.claim_id)

    # Second use of the same claim should fail
    with pytest.raises(ClaimError):
        engine.record_evidence("n1", LogicalEvidence(success=0.5), claim_id=target.claim_id)


@pytest.mark.unit
def test_record_evidence_without_claim(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    node = engine.record_evidence("n1", LogicalEvidence(success=1.0)).node
    assert node.alpha > 1.0
    assert node.first_evidence_at is not None


@pytest.mark.unit
def test_record_evidence_node_not_found(engine: HypoTreeEngine) -> None:
    with pytest.raises(NodeNotFoundError):
        engine.record_evidence("nonexistent", LogicalEvidence(success=0.5))


@pytest.mark.unit
def test_record_evidence_invalid_claim(engine: HypoTreeEngine) -> None:
    """An unknown id on a node holding a real lease consumes that lease."""
    engine.create_hypothesis("h1", node_id="n1")
    target = engine.get_next_targets()[0]
    assert target.claim_id is not None

    engine.record_evidence("n1", LogicalEvidence(success=0.5), claim_id="bogus")

    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_record_evidence_claim_wrong_node(engine: HypoTreeEngine) -> None:
    """A claim issued for one node cannot be consumed against a different node."""
    engine.create_hypothesis("a", node_id="a")
    engine.create_hypothesis("b", node_id="b")
    target = engine.get_next_targets()[0]
    assert target.claim_id is not None
    other = "b" if target.node_id == "a" else "a"
    with pytest.raises(ClaimError):
        engine.record_evidence(other, LogicalEvidence(success=0.9), claim_id=target.claim_id)
    # The mismatched claim must remain unconsumed and reusable on its own node.
    engine.record_evidence(target.node_id, LogicalEvidence(success=0.9), claim_id=target.claim_id)


# ---------------------------------------------------------------------------
# record_evidence — posterior update + transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_evidence_updates_posterior(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    node = engine.record_evidence("n1", LogicalEvidence(success=0.9)).node
    assert node.alpha == pytest.approx(1.9)
    assert node.beta == pytest.approx(1.1)


@pytest.mark.unit
def test_record_evidence_deterministic_success_verifies(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    # Deterministic: one success=1.0 should verify (alpha=2.0, beta=1.0, mean=0.667 < 0.8)
    # Need multiple successes to clear the bar
    for _ in range(8):
        node = engine.record_evidence("n1", LogicalEvidence(success=1.0)).node
    assert node.status == Status.VERIFIED


@pytest.mark.unit
def test_record_evidence_deterministic_failure_invalidates(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    node = engine.record_evidence("n1", LogicalEvidence(success=0.0)).node
    assert node.status == Status.INVALIDATED


@pytest.mark.unit
def test_record_evidence_deterministic_failure_cascades_prune(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", parent_ids=["p1"], edge_type=EdgeType.DEPENDENCY, node_id="c1"
    )
    engine.record_evidence("p1", LogicalEvidence(success=0.0))

    child_reloaded = engine._store.get_node("c1")
    assert child_reloaded is not None
    assert child_reloaded.status == Status.PRUNED


@pytest.mark.unit
def test_record_evidence_stochastic_single_failure_no_invalidate(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", evidence_regime="stochastic", node_id="n1")
    node = engine.record_evidence("n1", LogicalEvidence(success=0.0)).node
    # One failure on a stochastic node should NOT invalidate
    assert node.status != Status.INVALIDATED


@pytest.mark.unit
def test_record_evidence_infra_error(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    node = engine.record_evidence("n1", InfraError(error_type="OOM", message="killed")).node
    assert node.infra_retry_count == 1
    assert node.status != Status.INVALIDATED


@pytest.mark.unit
def test_record_evidence_infra_error_blocks_after_max(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    from hypotree.engine import MAX_INFRA_RETRIES

    for _ in range(MAX_INFRA_RETRIES):
        node = engine.record_evidence("n1", InfraError(error_type="OOM", message="killed")).node
    assert node.status == Status.BLOCKED


# ---------------------------------------------------------------------------
# record_evidence — deltas + monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_evidence_delta_first(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    rows = engine._store.get_evidence_for_node("n1")
    assert rows[-1]["monotonicity"] == "first"
    assert rows[-1]["delta_success"] == 0.0


@pytest.mark.unit
def test_record_evidence_delta_up(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    engine.record_evidence("n1", LogicalEvidence(success=0.8))
    rows = engine._store.get_evidence_for_node("n1")
    assert rows[-1]["monotonicity"] == "up"
    assert rows[-1]["delta_success"] == pytest.approx(0.3)


@pytest.mark.unit
def test_record_evidence_delta_flat(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", evidence_regime="stochastic", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    rows = engine._store.get_evidence_for_node("n1")
    assert rows[-1]["monotonicity"] == "flat"
    assert rows[-1]["delta_success"] == 0.0


@pytest.mark.unit
def test_record_evidence_delta_down(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", evidence_regime="stochastic", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.8))
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    rows = engine._store.get_evidence_for_node("n1")
    assert rows[-1]["monotonicity"] == "down"
    assert rows[-1]["delta_success"] == pytest.approx(-0.3)


@pytest.mark.unit
def test_record_evidence_delta_ignores_leading_infra(engine: HypoTreeEngine) -> None:
    """An infra error before the first logical observation must not distort the
    first-measurement delta — the baseline is the last *logical* success only."""
    engine.create_hypothesis("h1", node_id="n1")
    engine.record_evidence("n1", InfraError(error_type="OOM", message="killed"))
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    logical = [r for r in engine._store.get_evidence_for_node("n1") if r["kind"] == "logical"]
    assert logical[-1]["monotonicity"] == "first"
    assert logical[-1]["delta_success"] == 0.0


# ---------------------------------------------------------------------------
# upstream propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_upstream_refinement(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    engine.create_hypothesis(
        "child", parent_ids=["r1"], edge_type=EdgeType.REFINEMENT, node_id="c1"
    )
    # Dispatch root and set to IN_PROGRESS
    engine.update_status("r1", Status.IN_PROGRESS)

    # Verify child — should promote root
    affected = engine.verify_upstream("c1")
    assert "r1" in affected
    root = engine._store.get_node("r1")
    assert root is not None
    assert root.status == Status.VERIFIED


@pytest.mark.unit
def test_invalidate_upstream_dependency(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    engine.create_hypothesis(
        "child", parent_ids=["r1"], edge_type=EdgeType.DEPENDENCY, node_id="c1"
    )
    engine.update_status("r1", Status.VERIFIED)
    affected = engine.invalidate_upstream("c1")
    assert "r1" in affected
    root = engine._store.get_node("r1")
    assert root is not None
    assert root.status == Status.NEEDS_REVISION


# ---------------------------------------------------------------------------
# get_goal_status
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_goal_status_empty(engine: HypoTreeEngine) -> None:
    resp = engine.get_goal_status()
    assert resp.goals == []
    assert resp.all_met is False


@pytest.mark.unit
def test_get_goal_status_not_met(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("goal", is_goal=True, target_metric=0.9, node_id="g1")
    resp = engine.get_goal_status()
    assert len(resp.goals) == 1
    assert resp.goals[0].met is False
    assert resp.all_met is False


@pytest.mark.unit
def test_get_goal_status_met(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("combo", node_id="combo")
    engine.create_hypothesis(
        "goal",
        is_goal=True,
        target_metric=0.8,
        node_id="g1",
        parent_ids=["combo"],
        edge_type=EdgeType.DEPENDENCY,
    )
    engine.record_evidence("combo", LogicalEvidence(success=1.0))

    resp = engine.get_goal_status()
    assert resp.goals[0].met is True
    assert resp.all_met is True


# ---------------------------------------------------------------------------
# get_dag_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_dag_context_empty(engine: HypoTreeEngine) -> None:
    resp = engine.get_dag_context()
    assert resp.nodes == []
    assert resp.elisions == []


@pytest.mark.unit
def test_get_dag_context_single_node(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    resp = engine.get_dag_context()
    assert len(resp.nodes) == 1
    assert resp.nodes[0].id == "r1"
    assert resp.nodes[0].credible_interval is not None


@pytest.mark.unit
def test_get_dag_context_depth_bounded(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("r", node_id="r")
    engine.create_hypothesis("c", parent_ids=["r"], node_id="c")
    engine.create_hypothesis("gc", parent_ids=["c"], node_id="gc")
    resp = engine.get_dag_context(node_id="r", max_depth=1)
    ids = {n.id for n in resp.nodes}
    assert "r" in ids
    assert "c" in ids
    assert "gc" not in ids


@pytest.mark.unit
def test_get_dag_context_width_bounded(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="root")
    for i in range(15):
        engine.create_hypothesis(f"c{i}", parent_ids=["root"], node_id=f"c{i}")
    resp = engine.get_dag_context(node_id="root", max_depth=1, max_children=10)
    assert len(resp.elisions) == 1
    assert resp.elisions[0].hidden_count == 5


@pytest.mark.unit
def test_get_dag_context_unknown_node(engine: HypoTreeEngine) -> None:
    with pytest.raises(NodeNotFoundError):
        engine.get_dag_context(node_id="does-not-exist")


@pytest.mark.unit
def test_get_dag_context_diamond_dedup(engine: HypoTreeEngine) -> None:
    """A node reachable by two paths appears exactly once in the subgraph."""
    engine.create_hypothesis("root", node_id="root")
    engine.create_hypothesis("a", parent_ids=["root"], node_id="a")
    engine.create_hypothesis("b", parent_ids=["root"], node_id="b")
    engine.create_hypothesis("g", parent_ids=["a", "b"], node_id="g")
    resp = engine.get_dag_context(node_id="root", max_depth=3)
    ids = [n.id for n in resp.nodes]
    assert ids.count("g") == 1


@pytest.mark.unit
def test_update_status_node_not_found(engine: HypoTreeEngine) -> None:
    with pytest.raises(NodeNotFoundError):
        engine.update_status("nope", Status.BLOCKED)


# ---------------------------------------------------------------------------
# render_dag_map (Mermaid)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_dag_map_basic(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    mermaid = engine.render_dag_map()
    assert "graph TD" in mermaid
    assert "r1" in mermaid
    assert "UNTESTED" in mermaid


@pytest.mark.unit
def test_render_dag_map_with_edge(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    engine.create_hypothesis("child", parent_ids=["r1"], node_id="c1")
    mermaid = engine.render_dag_map()
    assert "r1 --> c1" in mermaid or "r1 -->" in mermaid


@pytest.mark.unit
def test_render_dag_map_goal_marker(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("goal", is_goal=True, target_metric=0.9, node_id="g1")
    mermaid = engine.render_dag_map()
    assert "🎯" in mermaid


@pytest.mark.unit
def test_render_dag_map_elision(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="root")
    for i in range(15):
        engine.create_hypothesis(f"c{i}", parent_ids=["root"], node_id=f"c{i}")
    mermaid = engine.render_dag_map(max_children=10)
    assert "more..." in mermaid


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_persistence_roundtrip(tmp_path: Path) -> None:
    """Data written by one engine instance is visible to a new instance."""
    db = tmp_path / "state.db"
    e1 = HypoTreeEngine(db, rng_seed=42)
    e1.create_hypothesis("persisted", node_id="p1")
    e1.create_hypothesis("child", parent_ids=["p1"], node_id="c1")
    e1.close()

    e2 = HypoTreeEngine(db, rng_seed=99)
    node = e2._store.get_node("p1")
    assert node is not None
    assert node.statement == "persisted"
    child = e2._store.get_node("c1")
    assert child is not None
    assert child.parent_ids == ["p1"]
    e2.close()


# ---------------------------------------------------------------------------
# Stale claim reclaim
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_claim_reclaim(engine: HypoTreeEngine) -> None:

    from hypotree.models.status import utcnow

    engine.create_hypothesis("h1", node_id="n1")
    claimed = utcnow() - timedelta(seconds=600)
    engine._store.create_claim("stale", "n1", claimed, ttl_s=60)

    # get_next_targets should reclaim the stale claim and re-select n1
    result = engine.get_next_targets()[0]
    assert result.status == "SELECTED"
    assert result.node_id == "n1"
    assert result.claim_id != "stale"


# ---------------------------------------------------------------------------
# Dead-zone resolution — EXHAUSTED
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_deterministic_dead_zone_evidence_exhausts_node(engine: HypoTreeEngine) -> None:
    """Evidence between the invalidate and verify bars must settle the node.

    Previously such a reading left the node IN_PROGRESS forever: it neither
    refuted (needs exactly 0.0) nor verified (needs > 0.8). The node then stayed
    on the frontier permanently and the navigator kept re-selecting it, spending
    budget on repeat probes that carry no information.
    """
    engine.create_hypothesis("h1", node_id="n1")
    node = engine.record_evidence("n1", LogicalEvidence(success=0.35)).node
    assert node.status == Status.EXHAUSTED


@pytest.mark.unit
def test_exhausted_node_is_not_reselected(engine: HypoTreeEngine) -> None:
    """The conclusiveness guard: a settled node leaves the selectable frontier."""
    engine.create_hypothesis("h1", node_id="n1")
    engine.create_hypothesis("h2", node_id="n2")
    engine.record_evidence("n1", LogicalEvidence(success=0.35))

    # Only the untested node can be selected now, no matter how often we ask.
    # The lease is released between asks because a node already dispatched is
    # deliberately withheld; without that this would test leasing, not settling.
    for _ in range(5):
        target = engine.get_next_targets()[0]
        assert target.node_id == "n2"
        engine.release_claims()


@pytest.mark.unit
def test_exhausted_node_does_not_prune_its_subtree(engine: HypoTreeEngine) -> None:
    """EXHAUSTED is not a refutation, so descendants must survive.

    Only an exact-zero refutation voids downstream work. Pruning on a merely
    mediocre score would destroy live branches and manufacture bogus revision
    events.
    """
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", parent_ids=["p1"], edge_type=EdgeType.REFINEMENT, node_id="c1"
    )
    engine.record_evidence("p1", LogicalEvidence(success=0.35))

    assert engine._store.get_node("p1").status == Status.EXHAUSTED
    child = engine._store.get_node("c1")
    assert child.status == Status.UNTESTED  # untouched, still refinable


@pytest.mark.unit
def test_zero_evidence_still_invalidates_not_exhausts(engine: HypoTreeEngine) -> None:
    """An exact zero is a refutation and must keep its cascade semantics."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", parent_ids=["p1"], edge_type=EdgeType.DEPENDENCY, node_id="c1"
    )
    node = engine.record_evidence("p1", LogicalEvidence(success=0.0)).node
    assert node.status == Status.INVALIDATED
    assert engine._store.get_node("c1").status == Status.PRUNED


@pytest.mark.unit
def test_stochastic_node_is_not_exhausted_prematurely(engine: HypoTreeEngine) -> None:
    """A stochastic node must stay selectable while evidence can still move it."""
    engine.create_hypothesis("h1", evidence_regime="stochastic", node_id="s1")
    node = engine.record_evidence("s1", LogicalEvidence(success=0.35)).node
    assert node.status != Status.EXHAUSTED


@pytest.mark.unit
def test_a_goal_takes_no_evidence_and_is_never_dispatched(engine: HypoTreeEngine) -> None:
    """A goal states an objective, so there is no result to file against it.

    The engine refuses to invalidate or exhaust a goal, which means evidence
    against one can only ever push it toward success — and, worse, the result
    being filed belongs to whatever was actually probed. A goal that absorbed
    results also never settled, so the navigator handed the same goal back turn
    after turn while every probe recorded against it was lost.
    """
    engine.create_hypothesis("GOAL", node_id="g", is_goal=True, target_metric=0.75)

    with pytest.raises(GoalEvidenceError):
        engine.record_evidence("g", LogicalEvidence(success=0.53))

    assert engine._store.get_node("g").evidence_count == 0
    assert "g" not in {n.id for n in engine._frontier_nodes()}
    assert engine.get_next_targets()[0].status == "DONE"


# ---------------------------------------------------------------------------
# Mutual exclusion — inference and retraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confirming_one_alternative_settles_its_exclusion_group(engine: HypoTreeEngine) -> None:
    """Confirming one member of a mutually-exclusive set settles the others.

    "Which of these four values is correct" is one question, not four. Once it is
    answered the remaining candidates need no experiment at all — resolving them
    by inference is what stops the navigator handing back a question that has
    already been settled.
    """
    for v in range(4):
        engine.create_hypothesis(f"component=v{v}", node_id=f"c{v}", exclusion_group="component")

    engine.record_evidence("c1", LogicalEvidence(success=1.0))

    assert engine._store.get_node("c1").status == Status.VERIFIED
    for other in ("c0", "c2", "c3"):
        assert engine._store.get_node(other).status == Status.EXHAUSTED, other
    # Nothing from that question is offered again.
    assert engine.get_next_targets()[0].status == "DONE"


@pytest.mark.unit
def test_exclusion_settles_but_does_not_prune(engine: HypoTreeEngine) -> None:
    """Excluded alternatives are settled, not refuted — their subtrees survive.

    The inference is conditional on the confirmation holding, so it must not
    destroy work the way a real refutation does.
    """
    engine.create_hypothesis("a", node_id="a", exclusion_group="g")
    engine.create_hypothesis("b", node_id="b", exclusion_group="g")
    engine.create_hypothesis(
        "b_child", parent_ids=["b"], edge_type=EdgeType.REFINEMENT, node_id="bc"
    )

    engine.record_evidence("a", LogicalEvidence(success=1.0))

    assert engine._store.get_node("b").status == Status.EXHAUSTED
    assert engine._store.get_node("bc").status == Status.UNTESTED


@pytest.mark.unit
def test_exclusion_never_overwrites_a_state_reached_by_evidence(engine: HypoTreeEngine) -> None:
    """An alternative already refuted by its own probe keeps that verdict."""
    engine.create_hypothesis("a", node_id="a", exclusion_group="g")
    engine.create_hypothesis("b", node_id="b", exclusion_group="g")

    engine.record_evidence("b", LogicalEvidence(success=0.0))
    assert engine._store.get_node("b").status == Status.INVALIDATED

    engine.record_evidence("a", LogicalEvidence(success=1.0))
    # Still INVALIDATED — a real refutation outranks an inference.
    assert engine._store.get_node("b").status == Status.INVALIDATED


@pytest.mark.unit
def test_retracting_a_confirmation_reopens_its_excluded_alternatives(
    engine: HypoTreeEngine,
) -> None:
    """Withdrawing a confirmation must reopen the alternatives it settled.

    Without this a wrong confirmation would permanently bury the correct
    alternative and the search could never recover. An inference has to be
    exactly as retractable as the belief that produced it.
    """
    for v in range(4):
        engine.create_hypothesis(f"component=v{v}", node_id=f"c{v}", exclusion_group="component")

    engine.record_evidence("c1", LogicalEvidence(success=1.0))
    assert engine._store.get_node("c2").status == Status.EXHAUSTED

    # The confirmation turns out to be wrong.
    engine.record_evidence("c1", LogicalEvidence(success=0.0))

    assert engine._store.get_node("c1").status == Status.INVALIDATED
    for other in ("c0", "c2", "c3"):
        assert engine._store.get_node(other).status == Status.UNTESTED, other


@pytest.mark.unit
def test_integration_failure_drives_a_full_revision_cycle(engine: HypoTreeEngine) -> None:
    """The end-to-end belief-revision loop the gate's criterion 3 measures.

    A premise confirms in isolation, so it is VERIFIED and its competing
    alternatives are settled by inference. The combination built on it then fails
    outright, which must (a) invalidate the combination, (b) flip the premises it
    depended on to NEEDS_REVISION, and (c) reopen the alternatives that
    confirmation had settled — leaving the agent able to find the real answer.
    """
    engine.create_hypothesis("axis=v1", node_id="p_decoy", exclusion_group="axis")
    engine.create_hypothesis("axis=v2", node_id="p_true", exclusion_group="axis")
    engine.create_hypothesis(
        "axis=v1;other=v0", parent_ids=["p_decoy"], edge_type=EdgeType.DEPENDENCY, node_id="combo"
    )

    # The decoy passes in isolation.
    engine.record_evidence("p_decoy", LogicalEvidence(success=1.0))
    assert engine._store.get_node("p_decoy").status == Status.VERIFIED
    assert engine._store.get_node("p_true").status == Status.EXHAUSTED

    # Assembled, it fails outright.
    engine.record_evidence("combo", LogicalEvidence(success=0.0))

    assert engine._store.get_node("combo").status == Status.INVALIDATED
    # The premise it rested on is flagged for revision …
    assert engine._store.get_node("p_decoy").status == Status.NEEDS_REVISION
    # … and the alternative it had excluded is testable again.
    assert engine._store.get_node("p_true").status == Status.UNTESTED


@pytest.mark.unit
def test_manual_verification_also_settles_the_exclusion_group(engine: HypoTreeEngine) -> None:
    """The exclusion rule belongs to the transition, not to one code path.

    VERIFIED reached by a manual override must have the same consequences as
    VERIFIED reached from evidence, otherwise the belief state means different
    things depending on how it got there.
    """
    engine.create_hypothesis("a", node_id="a", exclusion_group="g")
    engine.create_hypothesis("b", node_id="b", exclusion_group="g")

    engine.update_status("a", Status.VERIFIED, reason="manual")
    assert engine._store.get_node("b").status == Status.EXHAUSTED

    # Manually withdrawing the confirmation reopens the alternative.
    engine.update_status("a", Status.NEEDS_REVISION, reason="manual revision")
    assert engine._store.get_node("b").status == Status.UNTESTED


@pytest.mark.unit
def test_upstream_promotion_settles_the_exclusion_group(engine: HypoTreeEngine) -> None:
    """A node promoted via REFINEMENT is as confirmed as one probed directly."""
    engine.create_hypothesis("parent", node_id="p", exclusion_group="g")
    engine.create_hypothesis("rival", node_id="rival", exclusion_group="g")
    engine.create_hypothesis("child", parent_ids=["p"], edge_type=EdgeType.REFINEMENT, node_id="c")
    engine.update_status("p", Status.IN_PROGRESS, reason="in progress")

    engine.update_status("c", Status.VERIFIED, reason="child verified")
    engine.verify_upstream("c")

    assert engine._store.get_node("p").status == Status.VERIFIED
    assert engine._store.get_node("rival").status == Status.EXHAUSTED
