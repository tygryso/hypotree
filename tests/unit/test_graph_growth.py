"""Tests for `add_edges` — growing a graph forward without a destructive write.

A graph grows two ways and only one of them was expressible. Backward growth
discovers a premise underneath something already pinned to the goal and leaves
the goal alone. Forward growth extends a pipeline and *must* re-pin the goal to
the new last stage, or the goal reports itself achieved as soon as stage one
verifies while the rest sit untested.

Until `add_edges` the only way to re-pin was recreating the goal with
`if_exists="overwrite"` — a full replace that silently dropped any field the
caller left out, including the bar the goal is measured against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import GoalDependencyError, HypoTreeEngine, NodeNotFoundError
from hypotree.graph.dag import CycleError
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence


def _pipeline(db: Path) -> HypoTreeEngine:
    """`P2a -> goal`, with `P2b` following `P2a` but not yet wired to the goal."""
    engine = HypoTreeEngine(db, rng_seed=7)
    engine.create_hypotheses(
        [
            {"statement": "stage a", "node_id": "P2a"},
            {
                "statement": "ship it",
                "node_id": "goal",
                "is_goal": True,
                "target_metric": 0.8,
                "parent_ids": ["P2a"],
            },
            {"statement": "stage b", "node_id": "P2b", "parent_ids": ["P2a"]},
        ]
    )
    return engine


@pytest.mark.unit
def test_forward_growth_no_longer_needs_a_destructive_write(tmp_path: Path) -> None:
    """The goal keeps its identity, its metric and its history.

    Re-creating it with `if_exists="overwrite"` discards the node entirely, so a
    re-pin that forgot `target_metric` left a goal with no bar to miss.
    """
    engine = _pipeline(tmp_path / "forward.db")
    try:
        engine.add_edges([{"src": "P2b", "dst": "goal", "type": "DEPENDENCY"}])

        engine.record_evidence("P2a", LogicalEvidence(success=1.0, depth=1))
        goal = engine._store.get_node("goal")
        assert engine.goal_achieved(goal) is False, "stage b has not run yet"
        assert goal.target_metric == 0.8, "the bar must survive a re-pin"

        engine.record_evidence("P2b", LogicalEvidence(success=1.0, depth=1))
        assert engine.goal_achieved(engine._store.get_node("goal")) is True
    finally:
        engine.close()


@pytest.mark.unit
def test_the_old_edge_does_not_have_to_be_removed(tmp_path: Path) -> None:
    """DEPENDENCY is AND and the later stage already depends on the earlier one.

    So a goal wired to both is satisfied exactly when the later one is — adding
    tightens the condition and can never loosen it. That is why there is no edge
    removal to get wrong.
    """
    engine = _pipeline(tmp_path / "both.db")
    try:
        engine.add_edges([{"src": "P2b", "dst": "goal"}])
        assert sorted(engine._store.get_parent_ids("goal")) == ["P2a", "P2b"]

        engine.record_evidence("P2a", LogicalEvidence(success=1.0, depth=1))
        assert engine.goal_achieved(engine._store.get_node("goal")) is False
    finally:
        engine.close()


@pytest.mark.unit
def test_backward_growth_leaves_the_goal_alone(tmp_path: Path) -> None:
    """Discovering a premise underneath pinned work does not move the objective."""
    engine = _pipeline(tmp_path / "backward.db")
    try:
        engine.create_hypotheses([{"statement": "a fix", "node_id": "Fix_1"}])
        engine.add_edges([{"src": "Fix_1", "dst": "P2a", "type": "DEPENDENCY"}])

        assert engine._store.get_parent_ids("goal") == ["P2a"]
        assert engine._store.get_parent_ids("P2a") == ["Fix_1"]
        # P2a now waits on its new premise rather than being immediately testable.
        assert "P2a" not in {n.id for n in engine._frontier_nodes()}
        assert "Fix_1" in {n.id for n in engine._frontier_nodes()}
    finally:
        engine.close()


@pytest.mark.unit
def test_adding_an_edge_that_already_exists_is_a_no_op(tmp_path: Path) -> None:
    """Re-sending a plan must be safe, so a duplicate reports rather than raises."""
    engine = _pipeline(tmp_path / "idem.db")
    try:
        first = engine.add_edges([{"src": "P2b", "dst": "goal"}])
        second = engine.add_edges([{"src": "P2b", "dst": "goal"}])
        assert [r.created for r in first] == [True]
        assert [r.created for r in second] == [False]
        assert sorted(engine._store.get_parent_ids("goal")) == ["P2a", "P2b"]
    finally:
        engine.close()


@pytest.mark.unit
def test_a_goal_still_cannot_be_given_something_to_depend_on(tmp_path: Path) -> None:
    """The direction rule is the one strong models get wrong, so both paths enforce it.

    `create_hypotheses` refuses it; a second way in that did not would simply
    move the defect rather than fix it.
    """
    engine = _pipeline(tmp_path / "direction.db")
    try:
        with pytest.raises(GoalDependencyError, match="point the wrong way"):
            engine.add_edges([{"src": "goal", "dst": "P2b", "type": "DEPENDENCY"}])
        # Nothing was written.
        assert engine._store.get_parent_ids("P2b") == ["P2a"]
    finally:
        engine.close()


@pytest.mark.unit
def test_a_rejected_batch_writes_nothing(tmp_path: Path) -> None:
    """All-or-nothing, exactly as creation is: fix the bad entry and resend."""
    engine = _pipeline(tmp_path / "atomic.db")
    try:
        with pytest.raises(NodeNotFoundError, match="ghost"):
            engine.add_edges(
                [
                    {"src": "P2b", "dst": "goal"},
                    {"src": "ghost", "dst": "goal"},
                ]
            )
        assert engine._store.get_parent_ids("goal") == ["P2a"], "the good edge must not land"
    finally:
        engine.close()


@pytest.mark.unit
def test_a_cycle_is_refused(tmp_path: Path) -> None:
    engine = _pipeline(tmp_path / "cycle.db")
    try:
        with pytest.raises(CycleError, match="cycle"):
            engine.add_edges([{"src": "P2b", "dst": "P2a"}])
    finally:
        engine.close()


@pytest.mark.unit
def test_malformed_edges_name_the_contract(tmp_path: Path) -> None:
    """An agent told `'src'` learns nothing; one told the shape can fix it."""
    engine = _pipeline(tmp_path / "malformed.db")
    try:
        with pytest.raises(ValueError, match="needs src and dst"):
            engine.add_edges([{"dst": "goal"}])
        with pytest.raises(ValueError, match="unknown field"):
            engine.add_edges([{"src": "P2b", "dst": "goal", "kind": "DEPENDENCY"}])
        with pytest.raises(ValueError, match="type must be one of"):
            engine.add_edges([{"src": "P2b", "dst": "goal", "type": "DEPENDS_ON"}])
    finally:
        engine.close()


@pytest.mark.unit
def test_edge_type_is_honoured(tmp_path: Path) -> None:
    """A REFINEMENT edge opens its child while the parent is merely in progress."""
    engine = _pipeline(tmp_path / "types.db")
    try:
        engine.create_hypotheses([{"statement": "a tweak", "node_id": "tweak"}])
        engine.add_edges([{"src": "P2a", "dst": "tweak", "type": "REFINEMENT"}])
        edges = {(e.src, e.dst): e.type for e in engine._store.get_all_edges()}
        assert edges[("P2a", "tweak")] is EdgeType.REFINEMENT
    finally:
        engine.close()
