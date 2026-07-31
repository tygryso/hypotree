"""Integration tests — full closed-loop flows via direct HypoTreeEngine calls.

Every test drives the engine through realistic agent-loop scenarios: create →
select → claim → evidence → auto-transition → prune/upstream propagation →
goal completion → DONE. No HTTP, no subprocess — just method calls.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from hypotree.engine import ClaimError, HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import InfraError, LogicalEvidence
from hypotree.models.status import Status, utcnow


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    e = HypoTreeEngine(tmp_path / "state.db", rng_seed=42)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Full create → select → evidence → prune cycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_create_select_prune_cycle(engine: HypoTreeEngine) -> None:
    """create → get_next_targets → record_evidence(0) → assert pruned + claim rejected."""
    engine.create_hypothesis("root", node_id="r1")
    engine.create_hypothesis(
        "child", parent_ids=["r1"], edge_type=EdgeType.DEPENDENCY, node_id="c1"
    )

    target = engine.get_next_targets()[0]
    assert target.node_id == "r1"
    assert target.claim_id is not None

    engine.record_evidence("r1", LogicalEvidence(success=0.0), claim_id=target.claim_id)

    assert engine._store.get_node("r1").status == Status.INVALIDATED
    assert engine._store.get_node("c1").status == Status.PRUNED

    with pytest.raises(ClaimError):
        engine.record_evidence("r1", LogicalEvidence(success=0.5), claim_id=target.claim_id)


# ---------------------------------------------------------------------------
# InfraError → BLOCKED
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_infra_errors_block_not_invalidate(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    from hypotree.engine import MAX_INFRA_RETRIES

    for _ in range(MAX_INFRA_RETRIES):
        node = engine.record_evidence("n1", InfraError(error_type="OOM", message="killed")).node
    assert node.status == Status.BLOCKED


# ---------------------------------------------------------------------------
# Upstream invalidation with history rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upstream_invalidation_history_correct(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("root", node_id="r1")
    engine.create_hypothesis("leaf", parent_ids=["r1"], edge_type=EdgeType.DEPENDENCY, node_id="l1")
    engine.update_status("r1", Status.VERIFIED)

    engine.record_evidence("l1", LogicalEvidence(success=0.0))

    root_history = engine._store.get_status_history("r1")
    statuses = [h["status"] for h in root_history]
    assert "NEEDS_REVISION" in statuses


# ---------------------------------------------------------------------------
# REFINEMENT verify_upstream
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_refinement_verify_upstream(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("parent", node_id="p1")
    engine.create_hypothesis(
        "child", parent_ids=["p1"], edge_type=EdgeType.REFINEMENT, node_id="c1"
    )
    engine.update_status("p1", Status.IN_PROGRESS)
    engine.update_status("c1", Status.VERIFIED)

    affected = engine.verify_upstream("c1")
    assert "p1" in affected
    assert engine._store.get_node("p1").status == Status.VERIFIED


# ---------------------------------------------------------------------------
# Regime-aware invalidation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_deterministic_invalidates_on_zero(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("det", node_id="d1")
    node = engine.record_evidence("d1", LogicalEvidence(success=0.0)).node
    assert node.status == Status.INVALIDATED


@pytest.mark.integration
def test_stochastic_survives_single_zero(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("sto", evidence_regime="stochastic", node_id="s1")
    node = engine.record_evidence("s1", LogicalEvidence(success=0.0)).node
    assert node.status != Status.INVALIDATED


# ---------------------------------------------------------------------------
# Stale claim reclaim + re-selection
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stale_claim_reclaim(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("h1", node_id="n1")
    target = engine.get_next_targets()[0]
    assert target.claim_id is not None

    # Simulate expiry by setting claimed_at far in the past
    claimed = utcnow() - timedelta(seconds=600)
    engine._store.create_claim("manual-stale", "n1", claimed, ttl_s=60)

    result = engine.get_next_targets()[0]
    assert result.status == "SELECTED"
    assert result.claim_id != "manual-stale"


# ---------------------------------------------------------------------------
# Goal completion → DONE
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_goal_completion_done(engine: HypoTreeEngine) -> None:
    """A goal completes when the hypothesis supporting it is confirmed."""
    engine.create_hypothesis("the answer", node_id="combo")
    engine.create_hypothesis(
        "goal",
        is_goal=True,
        target_metric=0.8,
        node_id="g1",
        parent_ids=["combo"],
        edge_type=EdgeType.DEPENDENCY,
    )

    engine.record_evidence("combo", LogicalEvidence(success=1.0))
    assert engine._store.get_node("combo").status == Status.VERIFIED

    result = engine.get_next_targets()[0]
    assert result.status == "DONE"
    assert result.reason == "all_goals_met"


# ---------------------------------------------------------------------------
# events.jsonl dump fidelity
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_events_jsonl_dump(engine: HypoTreeEngine, tmp_path: Path) -> None:
    import json

    engine.create_hypothesis("a", node_id="a")
    engine.create_hypothesis("b", node_id="b")
    engine.update_status("a", Status.IN_PROGRESS, reason="go")

    dump = tmp_path / "events.jsonl"
    engine._store.dump_events_jsonl(dump)

    lines = [json.loads(line) for line in dump.read_text(encoding="utf-8").strip().splitlines()]
    types_list = [line["type"] for line in lines]
    assert "NodeCreated" in types_list
    assert "StatusChanged" in types_list
    assert [line["seq"] for line in lines] == sorted(line["seq"] for line in lines)
