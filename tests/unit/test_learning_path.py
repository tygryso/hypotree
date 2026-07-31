"""Tests for generate_learning_path — the derivation trail, not the snapshot.

The other read tools answer "what does the belief state hold?". This one answers
"how did it come to hold that, and what did it cost?", which is a property of the
transition history and is destroyed by every current-state view. The distinction
these tests pin down is *observed* versus *inferred*: a conclusion an experiment
paid for and one the engine derived for free look identical in a status column
and are worth entirely different things.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    e = HypoTreeEngine(tmp_path / "learning.db", rng_seed=7)
    yield e
    e.close()


def _three_way_question(engine: HypoTreeEngine) -> None:
    """One question with three competing answers, none tested yet."""
    for nid in ("cat_a", "cat_b", "cat_c"):
        engine.create_hypothesis(f"catalyst {nid}", node_id=nid, exclusion_group="catalyst")


@pytest.mark.unit
def test_an_empty_workspace_reports_no_objective(engine: HypoTreeEngine) -> None:
    """A briefing on nothing must still be a briefing, not a crash or a blank."""
    path = engine.generate_learning_path()

    assert path.conclusions == 0
    assert path.probes_spent == 0
    assert "none declared" in path.markdown


@pytest.mark.unit
def test_a_probed_confirmation_is_reported_as_observed(engine: HypoTreeEngine) -> None:
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))

    path = engine.generate_learning_path()
    step = next(s for s in path.steps if s.node_id == "cat_a")

    assert step.status == Status.VERIFIED
    assert step.origin == "observed"
    assert step.cost_a_probe is True


@pytest.mark.unit
def test_the_siblings_a_confirmation_retires_are_reported_as_inferred(
    engine: HypoTreeEngine,
) -> None:
    """The exclusion inference is the biggest efficiency lever and it costs nothing.

    Reporting those retirements beside the probed confirmation, without saying
    which was which, would claim credit for three experiments where one was run.
    """
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))

    path = engine.generate_learning_path()
    retired = [s for s in path.steps if s.node_id in ("cat_b", "cat_c")]

    assert len(retired) == 2
    assert all(s.origin == "inferred" for s in retired)
    assert all(s.cost_a_probe is False for s in retired)
    assert path.conclusions_without_a_probe == 2
    assert path.probes_spent == 1


@pytest.mark.unit
def test_a_deduction_by_elimination_is_reported_as_free(engine: HypoTreeEngine) -> None:
    """The last one standing is confirmed without an experiment — say so."""
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=0.0))
    engine.record_evidence("cat_b", LogicalEvidence(success=0.0))

    path = engine.generate_learning_path()
    survivor = next(s for s in path.steps if s.node_id == "cat_c")

    assert survivor.status == Status.VERIFIED
    assert survivor.origin == "inferred"
    assert survivor.cost_a_probe is False
    assert "no experiment spent" in path.markdown


@pytest.mark.unit
def test_a_withdrawn_belief_is_reported_as_a_reversal(engine: HypoTreeEngine) -> None:
    """A hypothesis confirmed, then put under review, is the whole point of the tool.

    A snapshot of the current status cannot show it at all, and it is the one
    thing a returning agent most needs to know about its own conclusions.
    """
    for group, ids in (("component", ("comp_v1", "comp_v2")), ("regime", ("reg_v1", "reg_v2"))):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["comp_v1", "reg_v1"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("comp_v1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("reg_v1", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    path = engine.generate_learning_path()

    assert any(s.origin == "reversed" for s in path.steps)
    assert "changed our minds" in path.markdown
    assert path.open_conflicts == 1


@pytest.mark.unit
def test_goal_progress_is_reported_without_probing_the_goal(engine: HypoTreeEngine) -> None:
    """Goal achievement is derived from its parents, so it never costs a probe."""
    engine.create_hypothesis("premise", node_id="premise")
    engine.create_hypothesis(
        "ship it",
        node_id="goal",
        parent_ids=["premise"],
        edge_type=EdgeType.DEPENDENCY,
        is_goal=True,
        target_metric=0.75,
    )

    before = engine.generate_learning_path()
    assert (before.goals_met, before.goals_total) == (0, 1)

    engine.record_evidence("premise", LogicalEvidence(success=1.0))
    after = engine.generate_learning_path()

    assert (after.goals_met, after.goals_total) == (1, 1)
    assert "1/1 goal(s) reached" in after.markdown
    assert not any(s.node_id == "goal" and s.cost_a_probe for s in after.steps)


@pytest.mark.unit
def test_a_node_revised_and_reconfirmed_counts_as_one_conclusion(
    engine: HypoTreeEngine,
) -> None:
    """Conclusions are nodes, not transitions.

    Counting transitions would let a single hypothesis that was confirmed,
    withdrawn and re-confirmed report as three discoveries — which is the shape
    a conflict recovery always has.
    """
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))
    engine.update_status("cat_a", Status.NEEDS_REVISION, reason="manual review")
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))

    path = engine.generate_learning_path()

    assert sum(1 for nid in {s.node_id for s in path.steps} if nid == "cat_a") == 1
    assert path.conclusions == len({s.node_id for s in path.steps if s.status != Status.UNTESTED})


@pytest.mark.unit
def test_the_limit_bounds_the_narrative_but_not_the_counters(engine: HypoTreeEngine) -> None:
    """A long-running workspace must still return something readable."""
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))

    full = engine.generate_learning_path()
    clipped = engine.generate_learning_path(limit=1)

    assert len(clipped.steps) == 1
    assert clipped.conclusions == full.conclusions
    assert clipped.conclusions_without_a_probe == full.conclusions_without_a_probe


@pytest.mark.unit
def test_an_exclusion_retraction_is_reported_as_a_reversal(engine: HypoTreeEngine) -> None:
    """A sibling handed back when its retiring confirmation was withdrawn.

    This is the single most valuable line in the report and it was invisible in
    the first version: the reason string had no shared marker, so the engine
    wrote one thing and the reader looked for another — the exact drift the
    other markers exist to prevent.
    """
    _three_way_question(engine)
    engine.record_evidence("cat_a", LogicalEvidence(success=1.0))
    assert engine._store.get_node("cat_b").status == Status.EXHAUSTED

    engine.update_status("cat_a", Status.INVALIDATED, reason="withdrawn on review")

    path = engine.generate_learning_path()
    reopened = [s for s in path.steps if s.node_id == "cat_b" and s.origin == "reversed"]

    assert reopened, "the retraction must appear in the narrative"
    assert "changed our minds" in path.markdown
