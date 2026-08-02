"""Unit tests: graph DAG operations.

Covers: cycle detection, topo sort, edge-type-aware frontier eligibility,
cascading prune, upstream invalidation, upstream verification with depth cap
and termination on adversarial graphs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.graph import CycleError, HypoTreeGraph
from hypotree.models import Edge, EdgeType, Node, Status


def _node(nid: str, status: Status = Status.UNTESTED) -> Node:
    return Node(id=nid, statement=f"hypo-{nid}", status=status)


@pytest.mark.unit
class TestCycleDetection:
    def test_simple_dag_ok(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        g.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
        assert g.is_acyclic()

    def test_cycle_rejected(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        g.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
        with pytest.raises(CycleError):
            g.add_edge(Edge(src="b", dst="a", type=EdgeType.DEPENDENCY))


@pytest.mark.unit
class TestTopoSort:
    def test_topological_order(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        g.add_node(_node("c"))
        g.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="b", dst="c", type=EdgeType.DEPENDENCY))
        order = g.topological_order()
        assert order.index("a") < order.index("b") < order.index("c")


@pytest.mark.unit
class TestFrontier:
    def test_root_eligible(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("root"))
        assert "root" in g.eligible_frontier()

    def test_dependency_gate_blocks_child(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("parent"))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="parent", dst="child", type=EdgeType.DEPENDENCY))
        # Parent UNTESTED → child blocked
        assert "child" not in g.eligible_frontier()

    def test_dependency_gate_opens_when_verified(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("parent", Status.VERIFIED))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="parent", dst="child", type=EdgeType.DEPENDENCY))
        assert "child" in g.eligible_frontier()

    def test_alternative_gate_any_verified(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("alt1", Status.INVALIDATED))
        g.add_node(_node("alt2", Status.VERIFIED))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="alt1", dst="child", type=EdgeType.ALTERNATIVE))
        g.add_edge(Edge(src="alt2", dst="child", type=EdgeType.ALTERNATIVE))
        assert "child" in g.eligible_frontier()

    def test_alternative_gate_blocks_when_none_verified(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("alt1", Status.UNTESTED))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="alt1", dst="child", type=EdgeType.ALTERNATIVE))
        assert "child" not in g.eligible_frontier()

    def test_refinement_gate_allows_in_progress_parent(self) -> None:
        """The key REFINEMENT enabler — child eligible while parent is IN_PROGRESS."""
        g = HypoTreeGraph()
        g.add_node(_node("parent", Status.IN_PROGRESS))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="parent", dst="child", type=EdgeType.REFINEMENT))
        assert "child" in g.eligible_frontier()

    def test_refinement_gate_blocks_when_parent_invalidated(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("parent", Status.INVALIDATED))
        g.add_node(_node("child"))
        g.add_edge(Edge(src="parent", dst="child", type=EdgeType.REFINEMENT))
        assert "child" not in g.eligible_frontier()

    def test_pruned_and_invalidated_excluded(self) -> None:
        g = HypoTreeGraph()
        g.add_node(_node("p1", Status.PRUNED))
        g.add_node(_node("p2", Status.INVALIDATED))
        g.add_node(_node("root"))
        frontier = g.eligible_frontier()
        assert "p1" not in frontier
        assert "p2" not in frontier


@pytest.mark.unit
class TestCascadingPrune:
    def test_subtree_pruned(self) -> None:
        """INVALIDATED → descendants PRUNED (the 'memory that forgets' property)."""
        g = HypoTreeGraph()
        g.add_node(_node("root", Status.INVALIDATED))
        g.add_node(_node("child1"))
        g.add_node(_node("child2"))
        g.add_node(_node("grandchild"))
        g.add_edge(Edge(src="root", dst="child1", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="root", dst="child2", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="child1", dst="grandchild", type=EdgeType.DEPENDENCY))

        pruned = g.cascading_prune("root")
        assert set(pruned) == {"child1", "child2", "grandchild"}
        assert g.get_node("child1").status == Status.PRUNED
        assert g.get_node("grandchild").status == Status.PRUNED


@pytest.mark.unit
class TestUpstreamInvalidation:
    def test_verified_ancestor_flipped(self) -> None:
        """Leaf failure → VERIFIED DEPENDENCY ancestors → NEEDS_REVISION."""
        g = HypoTreeGraph()
        g.add_node(_node("root", Status.VERIFIED))
        g.add_node(_node("mid", Status.VERIFIED))
        g.add_node(_node("leaf", Status.INVALIDATED))
        g.add_edge(Edge(src="root", dst="mid", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="mid", dst="leaf", type=EdgeType.DEPENDENCY))

        affected = g.upstream_invalidate("leaf")
        assert "mid" in affected
        assert "root" in affected
        assert g.get_node("mid").status == Status.NEEDS_REVISION
        assert g.get_node("root").status == Status.NEEDS_REVISION

    def test_refinement_failure_does_not_kill_sibling_parent(self) -> None:
        """A failed REFINEMENT child must NOT invalidate its siblings' parent."""
        g = HypoTreeGraph()
        g.add_node(_node("parent", Status.IN_PROGRESS))
        g.add_node(_node("child_a", Status.INVALIDATED))
        g.add_node(_node("child_b"))
        g.add_edge(Edge(src="parent", dst="child_a", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="parent", dst="child_b", type=EdgeType.REFINEMENT))

        affected = g.upstream_invalidate("child_a")
        # upstream_invalidate walks DEPENDENCY only; REFINEMENT is ignored.
        assert affected == []
        assert g.get_node("parent").status == Status.IN_PROGRESS

    def test_diamond_ancestor_invalidated_once(self) -> None:
        """A shared grandparent reached by two DEPENDENCY paths is flipped exactly once."""
        g = HypoTreeGraph()
        g.add_node(_node("grand", Status.VERIFIED))
        g.add_node(_node("left", Status.VERIFIED))
        g.add_node(_node("right", Status.VERIFIED))
        g.add_node(_node("leaf", Status.INVALIDATED))
        g.add_edge(Edge(src="grand", dst="left", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="grand", dst="right", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="left", dst="leaf", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="right", dst="leaf", type=EdgeType.DEPENDENCY))

        affected = g.upstream_invalidate("leaf")
        # grand is discovered via both left and right, but appears once (visited-set).
        assert affected.count("grand") == 1
        assert set(affected) == {"left", "right", "grand"}


@pytest.mark.unit
class TestUpstreamVerification:
    def test_refinement_chain_promotes_in_progress(self) -> None:
        """VERIFIED child → IN_PROGRESS REFINEMENT ancestors → VERIFIED (flywheel up)."""
        g = HypoTreeGraph()
        g.add_node(_node("root", Status.IN_PROGRESS))
        g.add_node(_node("mid", Status.IN_PROGRESS))
        g.add_node(_node("leaf", Status.VERIFIED))
        g.add_edge(Edge(src="root", dst="mid", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="mid", dst="leaf", type=EdgeType.REFINEMENT))

        affected = g.upstream_verify("leaf")
        assert "mid" in affected
        assert "root" in affected
        assert g.get_node("mid").status == Status.VERIFIED
        assert g.get_node("root").status == Status.VERIFIED

    def test_stops_at_dependency_boundary(self) -> None:
        """upstream_verify only follows REFINEMENT, not DEPENDENCY."""
        g = HypoTreeGraph()
        g.add_node(_node("dep_parent", Status.IN_PROGRESS))
        g.add_node(_node("ref_parent", Status.IN_PROGRESS))
        g.add_node(_node("leaf", Status.VERIFIED))
        g.add_edge(Edge(src="dep_parent", dst="leaf", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="ref_parent", dst="leaf", type=EdgeType.REFINEMENT))

        affected = g.upstream_verify("leaf")
        assert "ref_parent" in affected
        assert "dep_parent" not in affected
        assert g.get_node("dep_parent").status == Status.IN_PROGRESS

    def test_depth_cap_prevents_over_propagation(self) -> None:
        """A depth cap of 1 stops propagation beyond one hop."""
        g = HypoTreeGraph()
        g.add_node(_node("n0", Status.IN_PROGRESS))
        g.add_node(_node("n1", Status.IN_PROGRESS))
        g.add_node(_node("n2", Status.IN_PROGRESS))
        g.add_node(_node("leaf", Status.VERIFIED))
        g.add_edge(Edge(src="n0", dst="n1", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="n1", dst="n2", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="n2", dst="leaf", type=EdgeType.REFINEMENT))

        affected = g.upstream_verify("leaf", max_depth=1)
        # Only n2 (direct parent, depth 1) is promoted; n1 and n0 are beyond cap.
        assert affected == ["n2"]
        assert g.get_node("n1").status == Status.IN_PROGRESS

    def test_termination_on_long_chain(self) -> None:
        """Termination proof: a long REFINEMENT chain doesn't loop or hang."""
        g = HypoTreeGraph()
        g.add_node(_node("leaf", Status.VERIFIED))
        prev = "leaf"
        for i in range(200):
            nid = f"chain_{i}"
            g.add_node(_node(nid, Status.IN_PROGRESS))
            g.add_edge(Edge(src=nid, dst=prev, type=EdgeType.REFINEMENT))
            prev = nid
        # Must terminate without error, capped at default max_depth.
        affected = g.upstream_verify("leaf")
        assert len(affected) <= 101  # max_depth=100 + leaf itself's direct parent

    def test_termination_on_cycle_attempt(self) -> None:
        """Even if the graph somehow had a back-edge, the visited-set prevents infinite loop."""
        g = HypoTreeGraph()
        # Build a normal DAG (cycles can't be added via add_edge, but test visited-set robustness)
        g.add_node(_node("a", Status.IN_PROGRESS))
        g.add_node(_node("b", Status.IN_PROGRESS))
        g.add_node(_node("c", Status.VERIFIED))
        g.add_edge(Edge(src="a", dst="b", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="b", dst="c", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="a", dst="c", type=EdgeType.REFINEMENT))  # transitive edge
        affected = g.upstream_verify("c")
        # Both a and b should be promoted, visited-set prevents double-processing.
        assert "a" in affected
        assert "b" in affected


@pytest.mark.unit
class TestGraphAccessors:
    def _diamond(self) -> HypoTreeGraph:
        g = HypoTreeGraph()
        g.add_node(_node("root"))
        g.add_node(_node("left"))
        g.add_node(_node("right"))
        g.add_node(_node("leaf"))
        g.add_edge(Edge(src="root", dst="left", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="root", dst="right", type=EdgeType.REFINEMENT))
        g.add_edge(Edge(src="left", dst="leaf", type=EdgeType.DEPENDENCY))
        g.add_edge(Edge(src="right", dst="leaf", type=EdgeType.ALTERNATIVE))
        return g

    def test_get_node_missing_returns_none(self) -> None:
        g = HypoTreeGraph()
        assert g.get_node("ghost") is None

    def test_all_nodes(self) -> None:
        g = self._diamond()
        assert {n.id for n in g.all_nodes()} == {"root", "left", "right", "leaf"}

    def test_children(self) -> None:
        g = self._diamond()
        assert set(g.children("root")) == {"left", "right"}
        assert g.children("leaf") == []

    def test_parents_filtered_by_edge_type(self) -> None:
        g = self._diamond()
        assert set(g.parents("leaf")) == {"left", "right"}
        assert g.parents("leaf", EdgeType.DEPENDENCY) == ["left"]
        assert g.parents("leaf", EdgeType.ALTERNATIVE) == ["right"]

    def test_ancestors_and_descendants(self) -> None:
        g = self._diamond()
        assert g.ancestors("leaf") == {"root", "left", "right"}
        assert g.descendants("root") == {"left", "right", "leaf"}

    def test_add_edge_syncs_child_parent_ids(self) -> None:
        g = HypoTreeGraph()
        parent = _node("p")
        child = _node("c")
        g.add_node(parent)
        g.add_node(child)
        g.add_edge(Edge(src="p", dst="c", type=EdgeType.DEPENDENCY))
        # Adding the same edge twice must not duplicate the derived parent id.
        g.add_edge(Edge(src="p", dst="c", type=EdgeType.DEPENDENCY))
        assert g.get_node("c").parent_ids == ["p"]

    def test_upstream_invalidate_skips_non_verified_ancestor(self) -> None:
        """A non-VERIFIED DEPENDENCY ancestor is left untouched (branch coverage)."""
        g = HypoTreeGraph()
        g.add_node(_node("root", Status.IN_PROGRESS))
        g.add_node(_node("leaf", Status.INVALIDATED))
        g.add_edge(Edge(src="root", dst="leaf", type=EdgeType.DEPENDENCY))
        assert g.upstream_invalidate("leaf") == []
        assert g.get_node("root").status == Status.IN_PROGRESS

    def test_upstream_verify_skips_non_in_progress_ancestor(self) -> None:
        """An already-VERIFIED REFINEMENT ancestor is not re-flipped (branch coverage)."""
        g = HypoTreeGraph()
        g.add_node(_node("root", Status.VERIFIED))
        g.add_node(_node("leaf", Status.VERIFIED))
        g.add_edge(Edge(src="root", dst="leaf", type=EdgeType.REFINEMENT))
        assert g.upstream_verify("leaf") == []


# ---------------------------------------------------------------------------
# EXHAUSTED — conclusiveness guard + refinement semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exhausted_node_leaves_the_frontier() -> None:
    """A settled node must never be handed out again by the navigator."""
    g = HypoTreeGraph()
    g.add_node(Node(id="a", statement="a", status=Status.EXHAUSTED))
    g.add_node(Node(id="b", statement="b", status=Status.UNTESTED))
    assert g.eligible_frontier() == ["b"]


@pytest.mark.unit
def test_refinement_child_of_exhausted_parent_is_eligible() -> None:
    """EXHAUSTED opens the REFINEMENT gate — a mediocre result is refinable.

    This is what separates EXHAUSTED from INVALIDATED: the parent was not
    refuted, so building an improvement on top of it is exactly the right next
    move and must stay available.
    """
    g = HypoTreeGraph()
    g.add_node(Node(id="p", statement="p", status=Status.EXHAUSTED))
    g.add_node(Node(id="c", statement="c", status=Status.UNTESTED))
    g.add_edge(Edge(src="p", dst="c", type=EdgeType.REFINEMENT))
    assert "c" in g.eligible_frontier()


@pytest.mark.unit
def test_dependency_child_of_exhausted_parent_is_blocked() -> None:
    """Work that strictly depends on a parent clearing its bar must still wait."""
    g = HypoTreeGraph()
    g.add_node(Node(id="p", statement="p", status=Status.EXHAUSTED))
    g.add_node(Node(id="c", statement="c", status=Status.UNTESTED))
    g.add_edge(Edge(src="p", dst="c", type=EdgeType.DEPENDENCY))
    assert "c" not in g.eligible_frontier()


@pytest.mark.unit
def test_bulk_edge_load_checks_acyclicity_once() -> None:
    """Rebuilding edge by edge cost O(E*(V+E)) — the quadratic dispatch term."""
    g = HypoTreeGraph()
    for i in range(6):
        g.add_node(Node(id=f"n{i}", statement=f"h{i}"))
    edges = [Edge(src=f"n{i}", dst=f"n{i + 1}", type=EdgeType.DEPENDENCY) for i in range(5)]
    g.add_edges_bulk(edges)

    assert g.is_acyclic()
    assert g.parents("n3") == ["n2"]
    assert g.parents("n0") == []
    # parent_ids stays derived, exactly as the per-edge path maintains it.
    assert g.get_node("n5").parent_ids == ["n4"]


@pytest.mark.unit
def test_bulk_edge_load_rejects_a_cyclic_set_and_rolls_back() -> None:
    """A rejected bulk load must leave the graph exactly as it found it."""
    g = HypoTreeGraph()
    for i in range(3):
        g.add_node(Node(id=f"n{i}", statement=f"h{i}"))
    g.add_edge(Edge(src="n0", dst="n1", type=EdgeType.DEPENDENCY))

    cyclic = [
        Edge(src="n1", dst="n2", type=EdgeType.DEPENDENCY),
        Edge(src="n2", dst="n0", type=EdgeType.DEPENDENCY),
    ]
    with pytest.raises(CycleError, match="bulk edge load"):
        g.add_edges_bulk(cyclic)

    assert g.is_acyclic()
    assert g.parents("n2") == []
    assert g.parents("n0") == []
    assert g.parents("n1") == ["n0"]


@pytest.mark.unit
def test_parents_are_indexed_not_scanned() -> None:
    """The frontier calls parents() per node, so a full edge scan is quadratic."""
    g = HypoTreeGraph()
    for name in ("a", "b", "c", "child"):
        g.add_node(Node(id=name, statement=name))
    g.add_edge(Edge(src="a", dst="child", type=EdgeType.DEPENDENCY))
    g.add_edge(Edge(src="b", dst="child", type=EdgeType.ALTERNATIVE))
    g.add_edge(Edge(src="c", dst="child", type=EdgeType.DEPENDENCY))

    assert sorted(g.parents("child")) == ["a", "b", "c"]
    assert sorted(g.parents("child", EdgeType.DEPENDENCY)) == ["a", "c"]
    assert g.parents("child", EdgeType.REFINEMENT) == []
    # Re-adding the same edge must not duplicate the index entry.
    g.add_edge(Edge(src="a", dst="child", type=EdgeType.DEPENDENCY))
    assert sorted(g.parents("child")) == ["a", "b", "c"]


@pytest.mark.unit
def test_a_rejected_bulk_load_keeps_edges_that_already_existed() -> None:
    """A bad set must not delete good edges it merely mentions."""
    g = HypoTreeGraph()
    for i in range(3):
        g.add_node(Node(id=f"n{i}", statement=f"h{i}"))
    g.add_edge(Edge(src="n0", dst="n1", type=EdgeType.DEPENDENCY))

    with pytest.raises(CycleError):
        g.add_edges_bulk(
            [
                # Already present — must survive the rollback.
                Edge(src="n0", dst="n1", type=EdgeType.DEPENDENCY),
                Edge(src="n1", dst="n2", type=EdgeType.DEPENDENCY),
                Edge(src="n2", dst="n0", type=EdgeType.DEPENDENCY),
            ]
        )

    assert g.parents("n1") == ["n0"]
    assert g.get_node("n1").parent_ids == ["n0"]
    assert g.parents("n2") == []
    assert g.is_acyclic()


@pytest.mark.unit
def test_a_cyclic_store_degrades_to_per_edge_loading_not_to_no_edges(tmp_path: Path) -> None:
    """Dropping every edge would make every node look like a root.

    A frontier computed from an edgeless graph is not a degraded answer, it is a
    wrong one: every gated hypothesis becomes dispatchable at once.
    """
    engine = HypoTreeEngine(tmp_path / "cyc.db", rng_seed=1)
    try:
        engine.create_hypotheses(
            [
                {"statement": "a", "node_id": "a"},
                {"statement": "b", "node_id": "b", "parent_ids": ["a"]},
            ]
        )
        # Forge a cycle straight into the store, bypassing the engine's guard.
        engine._store.add_edge(Edge(src="b", dst="a", type=EdgeType.DEPENDENCY))
        engine._sync_graph_from_store()

        assert engine._graph.is_acyclic()
        # The good edge survived; only the one closing the cycle was dropped.
        assert engine._graph.parents("b") == ["a"]
    finally:
        engine.close()
