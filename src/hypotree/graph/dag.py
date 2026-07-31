"""networkx-backed DAG operations.

Provides: add_node/edge with cycle detection, topological sort, edge-type-aware
eligible_frontier(), cascading_prune(), upstream_invalidate(), upstream_verify()
(depth-capped, termination-proven).
"""

from __future__ import annotations

from networkx import DiGraph, is_directed_acyclic_graph, topological_sort
from networkx import ancestors as nx_ancestors
from networkx import descendants as nx_descendants

from hypotree.models.edge import Edge, EdgeType
from hypotree.models.node import Node
from hypotree.models.status import Status

# Upstream verification walk depth cap (C8). Bounds the transitive REFINEMENT
# ancestor walk so it provably terminates on any DAG and can't cascade too far.
UPSTREAM_VERIFY_MAX_DEPTH = 100


class CycleError(Exception):
    """Raised when adding an edge would create a cycle."""


class HypoTreeGraph:
    """In-memory typed DAG over Node/Edge models."""

    def __init__(self) -> None:
        # node_id -> Node; edge_key (src, dst, type) -> Edge
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, EdgeType], Edge] = {}
        self._nx: DiGraph = DiGraph()

    # ------------------------------------------------------------------
    # Node / Edge mutation
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Insert a node. Idempotent on id (overwrites the stored object)."""
        self._nodes[node.id] = node
        if node.id not in self._nx:
            self._nx.add_node(node.id)

    def add_edge(self, edge: Edge) -> None:
        """Insert a typed edge, validating acyclicity.

        Raises CycleError if the edge would introduce a cycle.
        """
        # Ensure both endpoints exist in the networkx graph (defensive).
        self._nx.add_edge(edge.src, edge.dst)
        if not is_directed_acyclic_graph(self._nx):
            # Roll back the edge that broke acyclicity.
            self._nx.remove_edge(edge.src, edge.dst)
            raise CycleError(f"edge {edge.src} → {edge.dst} would create a cycle")
        self._edges[(edge.src, edge.dst, edge.type)] = edge
        # Keep the child's derived parent_ids in sync.
        child = self._nodes.get(edge.dst)
        if child is not None and edge.src not in child.parent_ids:
            child.parent_ids.append(edge.src)

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def children(self, node_id: str) -> list[str]:
        """Direct child ids of node_id."""
        return list(self._nx.successors(node_id))

    def parents(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        """Direct parent ids, optionally filtered by edge type."""
        result: list[str] = []
        for src, dst, etype in self._edges:
            if dst == node_id and (edge_type is None or etype == edge_type):
                result.append(src)
        return result

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return node ids in topological order."""
        return list(topological_sort(self._nx))

    def ancestors(self, node_id: str) -> set[str]:
        """All ancestor ids reachable upstream (any edge type)."""
        return nx_ancestors(self._nx, node_id)

    def descendants(self, node_id: str) -> set[str]:
        """All descendant ids reachable downstream (any edge type)."""
        return nx_descendants(self._nx, node_id)

    def is_acyclic(self) -> bool:
        return is_directed_acyclic_graph(self._nx)

    # ------------------------------------------------------------------
    # Edge-type-aware frontier
    # ------------------------------------------------------------------

    def _is_frontier_status(self, status: Status) -> bool:
        """A node is frontier-eligible only in these statuses.

        EXHAUSTED is deliberately absent: a conclusively-tested node that did
        not clear its bar has no more information to give, so re-dispatching it
        only burns budget. This is the conclusiveness guard that stops the
        navigator from re-selecting the same settled nodes forever.
        """
        return status in {
            Status.UNTESTED,
            Status.IN_PROGRESS,
            # Conditionally-deferred: treated identically to UNTESTED in v0.
            Status.BLOCKED,
            Status.NEEDS_REVISION,
        }

    def _parent_gate_satisfied(self, node_id: str) -> bool:
        """Edge-type-aware gating.

        Root (no parents): eligible.
        DEPENDENCY parents: ALL must be VERIFIED.
        ALTERNATIVE parents: ANY must be VERIFIED.
        REFINEMENT parents: parent in {IN_PROGRESS, VERIFIED, EXHAUSTED}.

        EXHAUSTED opens the REFINEMENT gate but not the DEPENDENCY/ALTERNATIVE
        gates: refining a conclusively-mediocre result is exactly the useful
        next move, whereas work that strictly *depends* on it clearing the bar
        must still wait for a real verification.
        """
        deps = self.parents(node_id, EdgeType.DEPENDENCY)
        alts = self.parents(node_id, EdgeType.ALTERNATIVE)
        refs = self.parents(node_id, EdgeType.REFINEMENT)

        # If a node has no parents at all, the root gate is satisfied.
        if not deps and not alts and not refs:
            return True

        # DEPENDENCY: ALL parents must be VERIFIED.
        for pid in deps:
            p = self._nodes.get(pid)
            if p is None or p.status != Status.VERIFIED:
                return False

        # ALTERNATIVE: ANY parent must be VERIFIED (if there are alt parents).
        if alts and not any(
            (p := self._nodes.get(pid)) is not None and p.status == Status.VERIFIED for pid in alts
        ):
            return False

        # REFINEMENT: parent in {IN_PROGRESS, VERIFIED, EXHAUSTED}.
        for pid in refs:
            p = self._nodes.get(pid)
            if p is None or p.status not in {
                Status.IN_PROGRESS,
                Status.VERIFIED,
                Status.EXHAUSTED,
            }:
                return False

        return True

    def eligible_frontier(self) -> list[str]:
        """Compute the eligible frontier.

        Returns node ids whose status is frontier-eligible AND whose edge-type
        parent gate is satisfied.
        """
        return [
            nid
            for nid, node in self._nodes.items()
            if self._is_frontier_status(node.status) and self._parent_gate_satisfied(nid)
        ]

    # ------------------------------------------------------------------
    # Cascading prune
    # ------------------------------------------------------------------

    def cascading_prune(self, root_id: str) -> list[str]:
        """Set all descendants of root_id to PRUNED. Returns the pruned ids.

        The root itself is expected to already be INVALIDATED by the caller.
        Only PRUNED *descendants* are returned; the root is not included.
        """
        pruned: list[str] = []
        for did in self.descendants(root_id):
            node = self._nodes.get(did)
            if node is not None and node.status not in {Status.INVALIDATED, Status.PRUNED}:
                node.status = Status.PRUNED
                pruned.append(did)
        return pruned

    # ------------------------------------------------------------------
    # Upstream invalidation (DEPENDENCY ancestors → NEEDS_REVISION)
    # ------------------------------------------------------------------

    def upstream_invalidate(self, leaf_id: str) -> list[str]:
        """Walk DEPENDENCY ancestors, flip VERIFIED → NEEDS_REVISION.

        Returns affected ancestor ids. A failed REFINEMENT child does NOT
        kill its siblings' parent.
        """
        affected: list[str] = []
        visited: set[str] = set()
        stack = [leaf_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for parent_id in self.parents(nid, EdgeType.DEPENDENCY):
                p = self._nodes.get(parent_id)
                if p is not None and p.status == Status.VERIFIED:
                    p.status = Status.NEEDS_REVISION
                    affected.append(parent_id)
                    stack.append(parent_id)
        return affected

    # ------------------------------------------------------------------
    # Upstream verification (REFINEMENT ancestors → VERIFIED, depth-capped — C8)
    # ------------------------------------------------------------------

    def upstream_verify(
        self, child_id: str, max_depth: int = UPSTREAM_VERIFY_MAX_DEPTH
    ) -> list[str]:
        """Walk REFINEMENT ancestors that are IN_PROGRESS, flip → VERIFIED.

        Depth-capped at max_depth and scoped strictly to REFINEMENT edges:
        stops at DEPENDENCY/ALTERNATIVE boundaries. Returns affected ids.
        """
        affected: list[str] = []
        visited: set[str] = set()
        stack = [(child_id, 0)]
        while stack:
            nid, depth = stack.pop()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            for parent_id in self.parents(nid, EdgeType.REFINEMENT):
                if depth + 1 > max_depth:
                    continue
                p = self._nodes.get(parent_id)
                if p is not None and p.status == Status.IN_PROGRESS:
                    p.status = Status.VERIFIED
                    affected.append(parent_id)
                    stack.append((parent_id, depth + 1))
        return affected
