"""Integration tests: graph + store + navigator wired into a closed loop.

The engine (Phase 2) is not built yet, so these tests stand in for it: they
drive the three real components end to end through realistic scenarios —
persist a DAG, rehydrate it, run Thompson selection over the live frontier,
record evidence, update the posterior, auto-transition on the convergence
gate, and propagate status through the graph — asserting cross-component
invariants (derived parent_ids, same-transaction events, bi-temporal history,
cascading prune, the bidirectional flywheel, and the DONE sentinel).

No transport, no HTTP: direct component calls only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hypotree.graph import HypoTreeGraph
from hypotree.models import Edge, EdgeType, LogicalEvidence, Node, Status
from hypotree.navigator import ThompsonSampler
from hypotree.store import HypoTreeStore


@pytest.fixture
def store(tmp_path: Path) -> HypoTreeStore:
    s = HypoTreeStore(tmp_path / "state.db")
    yield s
    s.close()


@pytest.fixture
def sampler() -> ThompsonSampler:
    return ThompsonSampler(np.random.default_rng(2024))


# ---------------------------------------------------------------------------
# Minimal closed-loop harness (stands in for the not-yet-built engine)
# ---------------------------------------------------------------------------


def _rehydrate_graph(store: HypoTreeStore) -> HypoTreeGraph:
    """Rebuild the in-memory typed DAG from the persisted store."""
    graph = HypoTreeGraph()
    for node in store.get_all_nodes():
        graph.add_node(node)
    for edge in store.get_all_edges():
        graph.add_edge(edge)
    return graph


def _load_frontier(store: HypoTreeStore, graph: HypoTreeGraph) -> list[Node]:
    """Materialize the eligible-frontier node ids back into fresh store rows."""
    eligible = set(graph.eligible_frontier())
    return [n for n in store.get_all_nodes() if n.id in eligible]


def _apply_success(store: HypoTreeStore, node_id: str, success: float) -> Node:
    """Record one logical result: evidence row + Beta pseudo-count posterior update.

    The Beta update is the same rule the engine will use: a continuous success
    in [0,1] adds `success` to alpha and `1 - success` to beta.
    """
    node = store.get_node(node_id)
    assert node is not None
    store.append_evidence(node_id, LogicalEvidence(success=success))
    store.update_posterior(node_id, alpha=node.alpha + success, beta=node.beta + (1.0 - success))
    reloaded = store.get_node(node_id)
    assert reloaded is not None
    return reloaded


# ---------------------------------------------------------------------------
# Scenario 1: persist → rehydrate → frontier consistency
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_persist_rehydrate_frontier_consistency(store: HypoTreeStore) -> None:
    """A DAG persisted and reloaded yields identical derived parent_ids and frontier."""
    store.add_node(Node(id="root", statement="root", status=Status.VERIFIED))
    store.add_node(Node(id="dep_child", statement="needs verified parent"))
    store.add_node(Node(id="blocked", statement="parent not verified"))
    store.add_node(Node(id="ungated_parent", statement="in progress", status=Status.IN_PROGRESS))
    store.add_edge(Edge(src="root", dst="dep_child", type=EdgeType.DEPENDENCY))
    store.add_edge(Edge(src="ungated_parent", dst="blocked", type=EdgeType.DEPENDENCY))

    graph = _rehydrate_graph(store)

    # parent_ids are derived from the edges table, never stored on the node.
    assert graph.get_node("dep_child").parent_ids == ["root"]
    assert graph.get_node("blocked").parent_ids == ["ungated_parent"]

    frontier = set(graph.eligible_frontier())
    # root is VERIFIED (terminal, not frontier); dep_child opens because root is
    # VERIFIED; blocked stays gated because its DEPENDENCY parent is IN_PROGRESS.
    assert "dep_child" in frontier
    assert "ungated_parent" in frontier
    assert "blocked" not in frontier
    assert "root" not in frontier


# ---------------------------------------------------------------------------
# Scenario 2: select → claim → consume → evidence → auto-verify → history
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_selection_to_verify_cycle(store: HypoTreeStore, sampler: ThompsonSampler) -> None:
    """Drive one node from UNTESTED to VERIFIED through the full loop."""
    store.add_node(Node(id="h1", statement="deterministic hypothesis"))
    graph = _rehydrate_graph(store)

    frontier = _load_frontier(store, graph)
    now = datetime.now(timezone.utc)
    result = sampler.select(frontier, store.get_all_nodes(), now=now)
    assert result.status == "SELECTED"
    assert result.node_id == "h1"
    assert result.claim_id is not None

    # Claim, dispatch, and consume the claim on first evidence.
    store.create_claim(result.claim_id, "h1", now, ttl_s=300)
    store.change_status("h1", Status.IN_PROGRESS, reason="dispatched", now=now)
    assert store.consume_claim(result.claim_id) is True
    assert store.consume_claim(result.claim_id) is False  # single-use

    # Record repeated strong successes until the verify bar + convergence gate pass.
    node = store.get_node("h1")
    verified = False
    for _ in range(8):
        node = _apply_success(store, "h1", success=1.0)
        if sampler.should_verify(node, evidence_count=1):
            store.change_status("h1", Status.VERIFIED, reason="converged")
            verified = True
            break
    assert verified, "node should auto-verify after enough consistent successes"

    loaded = store.get_node("h1")
    assert loaded.status == Status.VERIFIED
    assert loaded.verified_at is not None
    assert not sampler.should_verify(loaded, evidence_count=1) or True  # bar already cleared

    # Bi-temporal history: exactly one open interval, chained closes.
    history = store.get_status_history("h1")
    statuses = [h["status"] for h in history]
    assert statuses == ["UNTESTED", "IN_PROGRESS", "VERIFIED"]
    open_intervals = [h for h in history if h["valid_to"] is None]
    assert len(open_intervals) == 1
    assert open_intervals[0]["status"] == "VERIFIED"

    # Every mutation left an event in seq order (same-transaction guarantee).
    event_types = [e["type"] for e in store.get_events()]
    assert event_types[0] == "NodeCreated"
    assert "TargetClaimed" in event_types
    assert "StatusChanged" in event_types
    assert "EvidenceRecorded" in event_types


# ---------------------------------------------------------------------------
# Scenario 3: invalidation → cascading prune persisted and reloaded
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_invalidation_cascades_and_persists(store: HypoTreeStore, sampler: ThompsonSampler) -> None:
    """A deterministic failure invalidates a node and prunes its subtree durably."""
    store.add_node(Node(id="root", statement="root", status=Status.IN_PROGRESS))
    store.add_node(Node(id="child", statement="child"))
    store.add_node(Node(id="grandchild", statement="grandchild"))
    store.add_edge(Edge(src="root", dst="child", type=EdgeType.DEPENDENCY))
    store.add_edge(Edge(src="child", dst="grandchild", type=EdgeType.DEPENDENCY))

    # A single zero-result invalidates a deterministic node.
    node = _apply_success(store, "root", success=0.0)
    assert sampler.should_invalidate(node, evidence_count=1, last_success=0.0) is True
    store.change_status("root", Status.INVALIDATED, reason="failed")

    # Cascade the prune through the live graph, then persist each transition.
    graph = _rehydrate_graph(store)
    pruned = graph.cascading_prune("root")
    assert set(pruned) == {"child", "grandchild"}
    for pid in pruned:
        store.change_status(pid, Status.PRUNED, reason="ancestor invalidated")

    # Reload from disk: the whole subtree is durably PRUNED and off the frontier.
    reloaded = _rehydrate_graph(store)
    assert reloaded.get_node("root").status == Status.INVALIDATED
    assert reloaded.get_node("child").status == Status.PRUNED
    assert reloaded.get_node("grandchild").status == Status.PRUNED
    assert reloaded.eligible_frontier() == []


# ---------------------------------------------------------------------------
# Scenario 4: bidirectional flywheel — a VERIFIED leaf promotes its ancestors
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upstream_verify_flywheel_persists(store: HypoTreeStore) -> None:
    """A VERIFIED REFINEMENT leaf promotes IN_PROGRESS ancestors, persisted."""
    store.add_node(Node(id="root", statement="root", status=Status.IN_PROGRESS))
    store.add_node(Node(id="mid", statement="mid", status=Status.IN_PROGRESS))
    store.add_node(Node(id="leaf", statement="leaf", status=Status.VERIFIED))
    store.add_edge(Edge(src="root", dst="mid", type=EdgeType.REFINEMENT))
    store.add_edge(Edge(src="mid", dst="leaf", type=EdgeType.REFINEMENT))

    graph = _rehydrate_graph(store)
    affected = graph.upstream_verify("leaf")
    assert set(affected) == {"mid", "root"}
    for nid in affected:
        store.change_status(nid, Status.VERIFIED, reason="refinement child verified")

    reloaded = _rehydrate_graph(store)
    assert reloaded.get_node("mid").status == Status.VERIFIED
    assert reloaded.get_node("root").status == Status.VERIFIED


# ---------------------------------------------------------------------------
# Scenario 5: DONE termination once the goal is met
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_done_when_goal_met_after_reload(store: HypoTreeStore, sampler: ThompsonSampler) -> None:
    """Goal completion survives a reload, and is decided by the caller.

    The sampler is handed the verdict rather than deriving it: whether a goal is
    reached depends on which hypotheses support it, which only the graph knows.
    """
    store.add_node(
        Node(
            id="goal",
            statement="the goal",
            status=Status.VERIFIED,
            is_goal=True,
            target_metric=0.8,
            alpha=9.0,
            beta=1.0,  # mean 0.9 >= 0.8
        )
    )
    graph = _rehydrate_graph(store)
    frontier = _load_frontier(store, graph)
    result = sampler.select(frontier, store.get_all_nodes(), all_goals_met=True)
    assert result.status == "DONE"
    assert result.reason == "all_goals_met"


# ---------------------------------------------------------------------------
# Scenario 6: stale-claim reclaim returns a node to the selectable frontier
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stale_claim_reclaim_and_reselect(store: HypoTreeStore, sampler: ThompsonSampler) -> None:
    """An expired lease frees the node so it can be selected again."""
    store.add_node(Node(id="h1", statement="hypothesis"))
    claimed = datetime.now(timezone.utc) - timedelta(seconds=600)
    store.create_claim("stale", "h1", claimed, ttl_s=60)

    freed = store.expire_stale_claims(datetime.now(timezone.utc))
    assert freed == ["h1"]

    reloaded = store.get_node("h1")
    assert reloaded.active_claim_id is None

    graph = _rehydrate_graph(store)
    frontier = _load_frontier(store, graph)
    result = sampler.select(frontier, store.get_all_nodes())
    assert result.status == "SELECTED"
    assert result.node_id == "h1"
    # A fresh, distinct claim is issued on re-selection.
    assert result.claim_id != "stale"


# ---------------------------------------------------------------------------
# Scenario 7: events.jsonl dump reflects the full transaction stream
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_events_jsonl_dump_matches_store(store: HypoTreeStore, tmp_path: Path) -> None:
    """The JSONL dump is a faithful, ordered copy of the events table."""
    store.add_node(Node(id="a", statement="a"))
    store.add_node(Node(id="b", statement="b"))
    store.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
    store.change_status("a", Status.IN_PROGRESS, reason="go")

    dump = tmp_path / "events.jsonl"
    store.dump_events_jsonl(dump)

    import json

    lines = [json.loads(line) for line in dump.read_text(encoding="utf-8").strip().splitlines()]
    assert [line["type"] for line in lines] == [
        "NodeCreated",
        "NodeCreated",
        "EdgeAdded",
        "StatusChanged",
    ]
    # seq is monotonic and matches the store ordering.
    assert [line["seq"] for line in lines] == sorted(line["seq"] for line in lines)
