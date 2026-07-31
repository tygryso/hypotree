"""Property/statistical tests — TS convergence, prune idempotency, replay
determinism, upstream termination on adversarial graphs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.graph.dag import HypoTreeGraph
from hypotree.models.edge import Edge, EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status
from hypotree.navigator.sampler import ThompsonSampler

# ---------------------------------------------------------------------------
# TS convergence over N runs (seeded)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ts_converges_to_best_node() -> None:
    """TS selects the higher-posterior node the majority of the time (seeded)."""
    good = Node(id="good", alpha=10.0, beta=1.0, statement="good")
    bad = Node(id="bad", alpha=1.0, beta=10.0, statement="bad")
    nodes = [good, bad]

    good_wins = 0
    for seed in range(100):
        sampler = ThompsonSampler(np.random.default_rng(seed))
        result = sampler.select(nodes, nodes)
        if result.node_id == "good":
            good_wins += 1

    assert good_wins > 80  # >80% of runs


# ---------------------------------------------------------------------------
# Prune idempotency
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_prune_idempotent() -> None:
    """Cascading prune twice produces the same result (no double-prune)."""
    graph = HypoTreeGraph()
    root = Node(id="root", statement="root", status=Status.INVALIDATED)
    child = Node(id="child", statement="child", status=Status.UNTESTED)
    graph.add_node(root)
    graph.add_node(child)
    graph.add_edge(Edge(src="root", dst="child", type=EdgeType.DEPENDENCY))

    pruned1 = graph.cascading_prune("root")
    pruned2 = graph.cascading_prune("root")

    assert set(pruned1) == {"child"}
    assert pruned2 == []  # already pruned — no-op


# ---------------------------------------------------------------------------
# Events-table replay determinism
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_events_replay_determinism(tmp_path: Path) -> None:
    """Two engines with the same seed produce the same event sequence."""
    db1 = tmp_path / "e1.db"
    db2 = tmp_path / "e2.db"

    e1 = HypoTreeEngine(db1, rng_seed=42)
    e1.create_hypothesis("a", node_id="a")
    e1.create_hypothesis("b", parent_ids=["a"], node_id="b")
    e1.record_evidence("a", LogicalEvidence(success=0.9))
    events1 = [(e["seq"], e["type"]) for e in e1._store.get_events()]
    e1.close()

    e2 = HypoTreeEngine(db2, rng_seed=42)
    e2.create_hypothesis("a", node_id="a")
    e2.create_hypothesis("b", parent_ids=["a"], node_id="b")
    e2.record_evidence("a", LogicalEvidence(success=0.9))
    events2 = [(e["seq"], e["type"]) for e in e2._store.get_events()]
    e2.close()

    assert events1 == events2


# ---------------------------------------------------------------------------
# Upstream propagation termination on adversarial graphs
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upstream_verify_terminates_on_deep_chain() -> None:
    """A deep REFINEMENT chain terminates without infinite loop."""
    graph = HypoTreeGraph()
    for i in range(200):
        status = Status.IN_PROGRESS if i > 0 else Status.VERIFIED
        graph.add_node(Node(id=f"n{i}", statement=f"n{i}", status=status))
        if i > 0:
            graph.add_edge(Edge(src=f"n{i - 1}", dst=f"n{i}", type=EdgeType.REFINEMENT))

    affected = graph.upstream_verify("n199", max_depth=100)
    assert len(affected) == 100  # exactly max_depth ancestors flipped, no hang


@pytest.mark.integration
def test_upstream_invalidate_terminates_on_diamond() -> None:
    """A diamond-shaped DEPENDENCY graph terminates correctly (dedup)."""
    graph = HypoTreeGraph()
    for nid in ["a", "b", "c", "d"]:
        graph.add_node(Node(id=nid, statement=nid, status=Status.VERIFIED))
    # Diamond: a → b → d, a → c → d
    graph.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
    graph.add_edge(Edge(src="a", dst="c", type=EdgeType.DEPENDENCY))
    graph.add_edge(Edge(src="b", dst="d", type=EdgeType.DEPENDENCY))
    graph.add_edge(Edge(src="c", dst="d", type=EdgeType.DEPENDENCY))

    affected = graph.upstream_invalidate("d")
    assert "b" in affected
    assert "c" in affected
    assert "a" in affected
    # No duplicates despite diamond
    assert len(affected) == len(set(affected))
