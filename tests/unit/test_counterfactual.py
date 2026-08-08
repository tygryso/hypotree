"""Tests for the counterfactual view — what it would take for a conclusion to be wrong.

A belief state that can only show what it currently thinks is a report. One that
can name the experiment which would overturn it is an instrument, and that is the
question a reviewer actually asks.

The property protected hardest here is the **ordering**: fragility is not the
inverse of the posterior. A belief confirmed by elimination carries a confident
posterior and no observation at all, so it must rank above a measured one however
sure the engine is — and it is simultaneously the cheapest thing in the graph to
settle, because one probe touches what no probe has ever touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine, NodeNotFoundError
from hypotree.models.evidence import LogicalEvidence


def _pipeline(db: Path, *, deep_combo: int = 3) -> HypoTreeEngine:
    """A goal resting on one deduced premise and one shallowly-observed premise."""
    engine = HypoTreeEngine(db, rng_seed=7, project_path=db.parent)
    engine.create_hypotheses(
        [
            {"statement": "c=v0", "node_id": "c0", "exclusion_group": "c"},
            {"statement": "c=v1", "node_id": "c1", "exclusion_group": "c"},
            {"statement": "m=v0", "node_id": "m0"},
            {
                "statement": "combo",
                "node_id": "combo",
                "parent_ids": ["c1", "m0"],
                "edge_type": "DEPENDENCY",
            },
            {
                "statement": "goal",
                "node_id": "goal",
                "is_goal": True,
                "target_metric": 0.9,
                "parent_ids": ["combo"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    engine.record_evidence("c0", LogicalEvidence(success=0.0, depth=1))
    engine.record_evidence("m0", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=1.0, depth=deep_combo))
    return engine


@pytest.mark.unit
def test_a_belief_nobody_measured_is_the_first_thing_to_attack(tmp_path: Path) -> None:
    """Confidence with no observation behind it is the weakest link, not the strongest."""
    engine = _pipeline(tmp_path / "cf.db")
    try:
        entries = engine.what_would_change_my_mind("goal")
        assert entries[0].node_id == "c1"
        assert entries[0].fragility == 1.0
        assert "never observed" in entries[0].weakest_link
        assert "probe" in entries[0].experiment
    finally:
        engine.close()


@pytest.mark.unit
def test_a_shallow_confirmation_under_a_deep_result_is_named(tmp_path: Path) -> None:
    """A confirmation supports no claim tested deeper than itself."""
    engine = _pipeline(tmp_path / "cf.db")
    try:
        by_id = {e.node_id: e for e in engine.what_would_change_my_mind("goal")}
        assert "depth 1" in by_id["m0"].weakest_link
        assert "depth 3" in by_id["m0"].experiment
    finally:
        engine.close()


@pytest.mark.unit
def test_nothing_is_invented_when_the_evidence_is_sound(tmp_path: Path) -> None:
    """A panel that always finds something to say is one nobody reads twice."""
    engine = HypoTreeEngine(tmp_path / "sound.db", rng_seed=7, project_path=tmp_path)
    try:
        engine.create_hypotheses(
            [
                {"statement": "p", "node_id": "p"},
                {
                    "statement": "goal",
                    "node_id": "goal",
                    "is_goal": True,
                    "target_metric": 0.9,
                    "parent_ids": ["p"],
                    "edge_type": "DEPENDENCY",
                },
            ]
        )
        # Observed twice at the depth nothing deeper rests on: genuinely solid.
        engine.record_evidence("p", LogicalEvidence(success=1.0, depth=2))
        engine.record_evidence("p", LogicalEvidence(success=1.0, depth=2))
        assert engine.what_would_change_my_mind("goal") == []
    finally:
        engine.close()


@pytest.mark.unit
def test_asking_what_would_change_your_mind_does_not_change_it(tmp_path: Path) -> None:
    """Read-only, and worth pinning: it walks the frontier machinery to price things."""
    engine = _pipeline(tmp_path / "ro.db")
    try:
        before = {n.id: (n.status, n.alpha, n.beta) for n in engine._store.get_all_nodes()}
        claims_before = len(engine.get_active_claims())
        engine.what_would_change_my_mind()
        after = {n.id: (n.status, n.alpha, n.beta) for n in engine._store.get_all_nodes()}
        assert before == after
        assert len(engine.get_active_claims()) == claims_before
    finally:
        engine.close()


@pytest.mark.unit
def test_an_unknown_goal_is_refused_rather_than_silently_empty(tmp_path: Path) -> None:
    """An empty list means 'nothing is fragile'. A typo must not look like that."""
    engine = _pipeline(tmp_path / "unknown.db")
    try:
        with pytest.raises(NodeNotFoundError):
            engine.what_would_change_my_mind("no-such-goal")
    finally:
        engine.close()


@pytest.mark.unit
def test_only_beliefs_actually_holding_the_goal_up_are_considered(tmp_path: Path) -> None:
    """A node that is not VERIFIED is not currently holding anything up."""
    engine = _pipeline(tmp_path / "scope.db")
    try:
        ids = {e.node_id for e in engine.what_would_change_my_mind("goal")}
        assert "c0" not in ids, "a refuted node supports nothing and cannot be overturned"
        assert "goal" not in ids, "a goal is an objective, not a belief resting on evidence"
    finally:
        engine.close()


@pytest.mark.unit
def test_the_ranking_is_by_fragility_and_the_limit_keeps_the_worst(tmp_path: Path) -> None:
    engine = _pipeline(tmp_path / "rank.db")
    try:
        entries = engine.what_would_change_my_mind("goal")
        assert [e.fragility for e in entries] == sorted(
            (e.fragility for e in entries), reverse=True
        )
        assert engine.what_would_change_my_mind("goal", limit=1) == entries[:1]
    finally:
        engine.close()


@pytest.mark.unit
def test_a_declared_cost_prices_the_flip(tmp_path: Path) -> None:
    """The price quoted is the one selection would act on, not a second opinion."""
    engine = HypoTreeEngine(tmp_path / "cost.db", rng_seed=7, project_path=tmp_path)
    try:
        engine.create_hypotheses(
            [
                {
                    "statement": "cheap",
                    "node_id": "a",
                    "exclusion_group": "g",
                    "estimated_cost": 1.0,
                },
                {
                    "statement": "dear",
                    "node_id": "b",
                    "exclusion_group": "g",
                    "estimated_cost": 100.0,
                },
                {
                    "statement": "goal",
                    "node_id": "goal",
                    "is_goal": True,
                    "target_metric": 0.9,
                    "parent_ids": ["b"],
                    "edge_type": "DEPENDENCY",
                },
            ]
        )
        engine.record_evidence("b", LogicalEvidence(success=1.0, depth=1))
        entry = next(e for e in engine.what_would_change_my_mind("goal") if e.node_id == "b")
        assert entry.estimated_cost is not None and entry.estimated_cost > 1.0
    finally:
        engine.close()
