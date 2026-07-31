"""Tests for collision guard, dry-run peek mode, edge-type styling,
hide_statuses filtering, and update_status old_status reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine, NodeNotFoundError
from hypotree.models.edge import EdgeType
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    eng = HypoTreeEngine(tmp_path / "state.db", rng_seed=42, project_path=tmp_path)
    yield eng
    eng.close()


# -- create_hypothesis if_exists ------------------------------------------


@pytest.mark.unit
def test_collision_guard_error_raises(engine: HypoTreeEngine) -> None:
    """Default if_exists='error' raises ValueError on ID collision."""
    engine.create_hypothesis("first", node_id="n1")
    with pytest.raises(ValueError, match="already exists"):
        engine.create_hypothesis("second", node_id="n1")


@pytest.mark.unit
def test_collision_guard_skip_returns_existing(engine: HypoTreeEngine) -> None:
    """if_exists='skip' returns the existing node with created=False."""
    result1 = engine.create_hypothesis("original", node_id="n1")
    assert result1.created is True
    result2 = engine.create_hypothesis("replacement", node_id="n1", if_exists="skip")
    assert result2.created is False
    assert result2.reason == "id_exists"
    assert result2.node.statement == "original"


@pytest.mark.unit
def test_collision_guard_overwrite_replaces(engine: HypoTreeEngine) -> None:
    """if_exists='overwrite' silently replaces the node."""
    engine.create_hypothesis("original", node_id="n1")
    result = engine.create_hypothesis("replacement", node_id="n1", if_exists="overwrite")
    assert result.created is True
    assert result.node.statement == "replacement"


@pytest.mark.unit
def test_overwrite_keeps_history_clean(engine: HypoTreeEngine) -> None:
    """Overwrite discards the old node so history has exactly one open interval."""
    engine.create_hypothesis("orig", node_id="n1")
    engine.create_hypothesis("repl", node_id="n1", if_exists="overwrite")
    conn = engine._store._conn  # noqa: SLF001
    open_intervals = conn.execute(
        "SELECT COUNT(*) c FROM status_history WHERE node_id='n1' AND valid_to IS NULL"
    ).fetchone()["c"]
    deleted_events = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE type='NodeDeleted' AND payload LIKE '%n1%'"
    ).fetchone()["c"]
    # Exactly one live status interval (the stale one from the old node is gone),
    # and the overwrite recorded a deletion in the append-only event log.
    assert open_intervals == 1
    assert deleted_events == 1


@pytest.mark.unit
def test_overwrite_preserves_child_edges(engine: HypoTreeEngine) -> None:
    """Overwriting a parent keeps its children attached (outgoing edges kept)."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis("child", node_id="c1", parent_ids=["p1"])
    engine.create_hypothesis("parent v2", node_id="p1", if_exists="overwrite")
    child = engine._store.get_node("c1")  # noqa: SLF001
    assert child is not None
    assert child.parent_ids == ["p1"]


@pytest.mark.unit
def test_invalid_if_exists_rejected(engine: HypoTreeEngine) -> None:
    """An unknown if_exists policy raises rather than silently defaulting."""
    with pytest.raises(ValueError, match="invalid if_exists"):
        engine.create_hypothesis("x", node_id="n1", if_exists="replace")


@pytest.mark.unit
def test_missing_parent_rejected(engine: HypoTreeEngine) -> None:
    """A parent_id that does not exist raises instead of a dangling edge."""
    with pytest.raises(NodeNotFoundError, match="parent node not found"):
        engine.create_hypothesis("child", node_id="c1", parent_ids=["ghost"])
    # No phantom edge or node should have been persisted.
    assert engine._store.get_node("c1") is None  # noqa: SLF001


# -- get_next_targets dry_run ----------------------------------------------


@pytest.mark.unit
def test_dry_run_issues_no_claim(engine: HypoTreeEngine) -> None:
    """dry_run=True returns the selection but issues no claim_id."""
    engine.create_hypothesis("test", node_id="n1")
    resp = engine.get_next_targets(dry_run=True)[0]
    assert resp.status == "SELECTED"
    assert resp.node_id == "n1"
    assert resp.claim_id is None  # No claim issued


@pytest.mark.unit
def test_dry_run_leaves_frontier_intact(engine: HypoTreeEngine) -> None:
    """After a dry-run, the node is still claimable in a real get_next_targets."""
    engine.create_hypothesis("test", node_id="n1")
    engine.get_next_targets(dry_run=True)[0]
    resp = engine.get_next_targets()[0]
    assert resp.status == "SELECTED"
    assert resp.claim_id is not None  # Real claim issued


@pytest.mark.unit
def test_non_dry_run_issues_claim(engine: HypoTreeEngine) -> None:
    """Normal (non-dry-run) get_next_targets issues a claim."""
    engine.create_hypothesis("test", node_id="n1")
    resp = engine.get_next_targets()[0]
    assert resp.claim_id is not None


# -- render_dag_map edge-type styling -------------------------------------


@pytest.mark.unit
def test_edge_type_dependency_solid(engine: HypoTreeEngine) -> None:
    """DEPENDENCY edges render as solid -->."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", node_id="c1", parent_ids=["p1"], edge_type=EdgeType.DEPENDENCY
    )
    mermaid = engine.render_dag_map()
    assert "-->" in mermaid


@pytest.mark.unit
def test_edge_type_alternative_dashed(engine: HypoTreeEngine) -> None:
    """ALTERNATIVE edges render as dashed -.->."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", node_id="c1", parent_ids=["p1"], edge_type=EdgeType.ALTERNATIVE
    )
    mermaid = engine.render_dag_map()
    assert "-.->" in mermaid


@pytest.mark.unit
def test_edge_type_refinement_thick(engine: HypoTreeEngine) -> None:
    """REFINEMENT edges render as thick ==>."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", node_id="c1", parent_ids=["p1"], edge_type=EdgeType.REFINEMENT
    )
    mermaid = engine.render_dag_map()
    assert "==>" in mermaid


# -- render_dag_map hide_statuses -----------------------------------------


@pytest.mark.unit
def test_hide_statuses_removes_pruned(engine: HypoTreeEngine) -> None:
    """hide_statuses drops matching nodes from the Mermaid output."""
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis("child", node_id="c1", parent_ids=["p1"])
    engine.update_status("c1", Status.PRUNED, reason="test")

    mermaid = engine.render_dag_map(hide_statuses=["PRUNED"])
    assert "p1" in mermaid
    assert "c1" not in mermaid


@pytest.mark.unit
def test_hide_statuses_empty_shows_all(engine: HypoTreeEngine) -> None:
    """No hide_statuses means all nodes visible (parent + child)."""
    engine.create_hypothesis("parent", node_id="n1")
    engine.create_hypothesis("child", node_id="n2", parent_ids=["n1"])
    engine.update_status("n2", Status.PRUNED, reason="test")

    mermaid = engine.render_dag_map()
    assert "n1" in mermaid
    assert "n2" in mermaid


# -- update_status returns old_status -------------------------------------


@pytest.mark.unit
def test_update_status_returns_old_status(engine: HypoTreeEngine) -> None:
    """update_status returns old_status and transition string."""
    engine.create_hypothesis("test", node_id="n1")
    result = engine.update_status("n1", Status.VERIFIED, reason="manual")
    assert result.old_status == Status.UNTESTED
    assert result.transition == "UNTESTED → VERIFIED"
    assert result.node.status == Status.VERIFIED


@pytest.mark.unit
def test_update_status_transition_needs_revision(engine: HypoTreeEngine) -> None:
    """Transition string works for NEEDS_REVISION."""
    engine.create_hypothesis("test", node_id="n1")
    engine.update_status("n1", Status.VERIFIED, reason="step 1")
    result = engine.update_status("n1", Status.NEEDS_REVISION, reason="revert")
    assert result.old_status == Status.VERIFIED
    assert result.transition == "VERIFIED → NEEDS_REVISION"


# -- integration: create_hypothesis + get_next_targets full flow -----------


@pytest.mark.integration
def test_create_and_claim_flow(engine: HypoTreeEngine) -> None:
    """Integration: create → dry_run peek → real claim → evidence.

    Driven by an ordinary hypothesis rather than a goal: a goal is an objective,
    so it is never dispatched and never accepts a result.
    """
    result = engine.create_hypothesis("candidate", node_id="h1")
    assert result.created is True
    assert result.node.id == "h1"

    # Peek
    peek = engine.get_next_targets(dry_run=True)[0]
    assert peek.node_id == "h1"
    assert peek.claim_id is None

    # Real dispatch
    resp = engine.get_next_targets()[0]
    assert resp.node_id == "h1"
    assert resp.claim_id is not None

    # Record evidence using the claim
    from hypotree.models.evidence import LogicalEvidence

    node = engine.record_evidence("h1", LogicalEvidence(success=1.0), claim_id=resp.claim_id).node
    assert node.alpha > 1.0


# -- record → dispatch fusion ---------------------------------------------


@pytest.mark.unit
def test_recording_does_not_dispatch_by_default(engine: HypoTreeEngine) -> None:
    """Recording a result and asking for more work are separate questions.

    An experiment that takes days is reported by someone who is not, in that
    moment, asking to be handed the next one — pushing work at them would either
    strand a lease or force them to release it.
    """
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses(
        [{"statement": "a", "node_id": "n1"}, {"statement": "b", "node_id": "n2"}]
    )
    result = engine.record_evidence("n1", LogicalEvidence(success=1.0))
    assert result.next_targets == []
    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_recording_can_fuse_the_next_dispatch(engine: HypoTreeEngine) -> None:
    """A synchronous caller pays a full round-trip for each of the two calls."""
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses(
        [
            {"statement": "a", "node_id": "n1"},
            {"statement": "b", "node_id": "n2"},
            {"statement": "c", "node_id": "n3"},
        ]
    )
    result = engine.record_evidence("n1", LogicalEvidence(success=1.0), count_next_targets=2)
    assert [t.status for t in result.next_targets] == ["SELECTED", "SELECTED"]
    assert {t.node_id for t in result.next_targets} == {"n2", "n3"}
    # The fused dispatch is a real one: it issues claims like any other.
    assert len(engine.get_active_claims()) == 2


@pytest.mark.unit
def test_a_fused_dispatch_reports_done_like_any_other(engine: HypoTreeEngine) -> None:
    """The DONE sentinel must survive the fusion, or a caller never learns it stopped."""
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses([{"statement": "a", "node_id": "n1"}])
    result = engine.record_evidence("n1", LogicalEvidence(success=1.0), count_next_targets=2)
    assert [t.status for t in result.next_targets] == ["DONE"]


@pytest.mark.unit
def test_a_negative_fusion_count_is_rejected(engine: HypoTreeEngine) -> None:
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses([{"statement": "a", "node_id": "n1"}])
    with pytest.raises(ValueError, match="count_next_targets must be >= 0"):
        engine.record_evidence("n1", LogicalEvidence(success=1.0), count_next_targets=-1)


# -- leases for long-running work -----------------------------------------


@pytest.mark.unit
def test_a_lease_can_be_renewed_while_the_work_is_still_running(engine: HypoTreeEngine) -> None:
    """The alternative is a TTL sized for the longest experiment, which makes
    every genuinely abandoned node unreclaimable for just as long."""
    engine.create_hypothesis("a", node_id="n1")
    target = engine.get_next_targets()[0]

    renewed = engine.renew_claim(target.claim_id, lease_ttl_s=86_400)
    assert renewed.node_id == "n1"
    assert renewed.expires_in_s == 86_400
    # The clock restarted, so what is left is the full TTL bar the microseconds
    # spent asking.
    assert [c.expires_in_s for c in engine.get_active_claims()] == [pytest.approx(86_400, abs=2)]


@pytest.mark.unit
def test_a_consumed_lease_cannot_be_renewed(engine: HypoTreeEngine) -> None:
    """It may already have gone to another caller; re-arming it puts two on one node."""
    from hypotree.engine import ClaimError
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypothesis("a", node_id="n1")
    target = engine.get_next_targets()[0]
    engine.record_evidence("n1", LogicalEvidence(success=1.0), claim_id=target.claim_id)

    with pytest.raises(ClaimError, match="cannot be renewed"):
        engine.renew_claim(target.claim_id)


@pytest.mark.unit
def test_one_lease_can_be_handed_back_without_a_result(engine: HypoTreeEngine) -> None:
    """Declining work is normal when the experiment costs a day of compute.

    The alternatives were to fabricate a result or to strand the node for the
    whole lease.
    """
    engine.create_hypotheses(
        [{"statement": "a", "node_id": "n1"}, {"statement": "b", "node_id": "n2"}]
    )
    targets = engine.get_next_targets(count=2)
    declined = targets[0]

    assert engine.release_claims([declined.claim_id]) == [declined.node_id]
    remaining = engine.get_active_claims()
    assert [c.node_id for c in remaining] == [targets[1].node_id]
    # And it is immediately available again rather than waiting out its TTL.
    assert declined.node_id in {n.id for n in engine._frontier_nodes()}  # noqa: SLF001


@pytest.mark.unit
def test_releasing_everything_is_still_the_default(engine: HypoTreeEngine) -> None:
    """A caller whose context was reset cannot report on dispatches it forgot."""
    engine.create_hypotheses(
        [{"statement": "a", "node_id": "n1"}, {"statement": "b", "node_id": "n2"}]
    )
    engine.get_next_targets(count=2)
    assert sorted(engine.release_claims()) == ["n1", "n2"]
    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_releasing_an_unknown_claim_is_a_no_op(engine: HypoTreeEngine) -> None:
    """Nothing is held under it, so there is nothing to hand back and nothing to report."""
    engine.create_hypothesis("a", node_id="n1")
    engine.get_next_targets()
    assert engine.release_claims(["not-a-claim"]) == []
    assert len(engine.get_active_claims()) == 1


@pytest.mark.unit
def test_the_fused_dispatch_tops_up_rather_than_adds(engine: HypoTreeEngine) -> None:
    """Recording a batch of two must leave the caller holding two, not four.

    Adding to the outstanding set instead hands work out faster than it is
    reported, which is exactly the waste the fusion exists to remove: the
    accelerator would pay for itself in round-trips and then spend the saving on
    stranded leases.
    """
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses([{"statement": f"h{i}", "node_id": f"n{i}"} for i in range(6)])
    batch = engine.get_next_targets(count=2)

    for target in batch:
        engine.record_evidence(
            target.node_id,
            LogicalEvidence(success=1.0),
            claim_id=target.claim_id,
            count_next_targets=2,
        )

    assert len(engine.get_active_claims()) == 2


@pytest.mark.unit
def test_a_caller_already_at_capacity_is_handed_nothing(engine: HypoTreeEngine) -> None:
    """Asking to hold two while holding two is a request that is already satisfied."""
    from hypotree.models.evidence import LogicalEvidence

    engine.create_hypotheses([{"statement": f"h{i}", "node_id": f"n{i}"} for i in range(4)])
    engine.get_next_targets(count=2)

    # Recorded against a node it was never dispatched, so no lease is consumed.
    result = engine.record_evidence("n3", LogicalEvidence(success=1.0), count_next_targets=2)
    assert result.next_targets == []
    assert len(engine.get_active_claims()) == 2
