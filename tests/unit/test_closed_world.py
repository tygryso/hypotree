"""Tests for the closed-world assumption behind deduction by elimination.

"All but one candidate is ruled out, so the survivor holds" is sound over a
complete list of answers and asserts something false over a partial one. The
engine assumed completeness of every group for months. These tests pin the
declaration that replaced the assumption, and the two mechanisms that now depend
on it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from hypotree.engine import (
    DEDUCTION_REASON_PREFIX,
    DEDUCTION_RETRACT_PREFIX,
    HypoTreeEngine,
)
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[HypoTreeEngine]:
    e = HypoTreeEngine(tmp_path / "closed.db", rng_seed=13)
    yield e
    e.close()


def _question(engine: HypoTreeEngine, group: str, n: int, closed: bool) -> list[str]:
    ids = [f"{group}_v{i}" for i in range(n)]
    engine.create_hypotheses(
        [
            {
                "statement": f"{group} = v{i}",
                "node_id": nid,
                "exclusion_group": group,
                "exclusion_closed": closed,
            }
            for i, nid in enumerate(ids)
        ]
    )
    return ids


def _refute(engine: HypoTreeEngine, node_id: str) -> None:
    engine.record_evidence(node_id, LogicalEvidence(success=0.0, depth=1))


# -- the declaration -----------------------------------------------------------


@pytest.mark.unit
def test_a_group_is_closed_unless_someone_says_otherwise(engine: HypoTreeEngine) -> None:
    """Every group written before the flag existed meant `closed`, so that is the default."""
    _question(engine, "a", 2, closed=True)
    engine.create_hypothesis("b = v0", node_id="b_v0", exclusion_group="b")

    assert engine._group_is_closed("a") is True
    assert engine._group_is_closed("b") is True
    assert engine._store.get_node("b_v0").exclusion_closed is True


@pytest.mark.unit
def test_one_member_declaring_openness_governs_the_group(engine: HypoTreeEngine) -> None:
    """Openness withdraws an inference, so the cautious declaration has to win.

    The other way round, a later `create_hypotheses` that forgot the flag would
    quietly re-enable deduction over a list its author knew was partial.
    """
    engine.create_hypotheses(
        [
            {"statement": "lr=1e-3", "node_id": "lr_a", "exclusion_group": "lr"},
            {
                "statement": "lr=1e-4",
                "node_id": "lr_b",
                "exclusion_group": "lr",
                "exclusion_closed": False,
            },
        ]
    )
    assert engine._group_is_closed("lr") is False


@pytest.mark.unit
def test_the_declaration_survives_a_reload(tmp_path: Path) -> None:
    """It is a column, not a session flag — the next process must read the same claim."""
    db = tmp_path / "persist.db"
    engine = HypoTreeEngine(db, rng_seed=1)
    try:
        engine.create_hypothesis(
            "lr=1e-3", node_id="lr_a", exclusion_group="lr", exclusion_closed=False
        )
    finally:
        engine.close()

    reopened = HypoTreeEngine(db, rng_seed=1)
    try:
        assert reopened._store.get_node("lr_a").exclusion_closed is False
        assert reopened._group_is_closed("lr") is False
    finally:
        reopened.close()


# -- deduction -----------------------------------------------------------------


@pytest.mark.unit
def test_deduction_still_fires_over_a_closed_question(engine: HypoTreeEngine) -> None:
    """The mechanism the moat is built on must be untouched by the new flag."""
    ids = _question(engine, "a", 3, closed=True)
    _refute(engine, ids[0])
    _refute(engine, ids[1])

    survivor = engine._store.get_node(ids[2])
    assert survivor.status == Status.VERIFIED
    assert survivor.evidence_count == 0
    history = engine._store.get_status_history(ids[2])
    assert str(history[-1]["reason"]).startswith(DEDUCTION_REASON_PREFIX)


@pytest.mark.unit
def test_deduction_is_withheld_over_an_open_question(engine: HypoTreeEngine) -> None:
    """ "The other three learning rates failed" says nothing about the fourth."""
    ids = _question(engine, "lr", 3, closed=False)
    _refute(engine, ids[0])
    _refute(engine, ids[1])

    survivor = engine._store.get_node(ids[2])
    assert survivor.status == Status.UNTESTED
    # And it is still offered, because it is still a real question.
    assert ids[2] in {n.id for n in engine._frontier_nodes()}


@pytest.mark.unit
def test_confirming_a_member_still_retires_the_others_when_open(engine: HypoTreeEngine) -> None:
    """Only the last-one-standing deduction is withheld.

    Exclusion itself is not a closed-world inference: one answer being right
    makes the others wrong whether or not the list was exhaustive.
    """
    ids = _question(engine, "lr", 3, closed=False)
    engine.record_evidence(ids[0], LogicalEvidence(success=1.0, depth=1))

    assert all(engine._store.get_node(n).status == Status.EXHAUSTED for n in ids[1:])


# -- a question that runs out --------------------------------------------------


@pytest.mark.unit
def test_an_open_question_that_runs_out_is_reported_but_not_pruned(
    engine: HypoTreeEngine,
) -> None:
    """An untried candidate could still satisfy what depends on the group."""
    ids = _question(engine, "lr", 2, closed=False)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=[ids[0]], edge_type=EdgeType.DEPENDENCY
    )
    for nid in ids:
        _refute(engine, nid)

    assert engine._question_is_dead("lr") is True
    assert engine._propagate_dead_question("lr", engine._store.get_node(ids[0]).updated_at) == []


@pytest.mark.unit
def test_the_navigator_says_add_a_candidate_for_an_open_question(engine: HypoTreeEngine) -> None:
    """ "Add the value you have not tried" and "one of your eliminations is wrong"
    are opposite instructions, and the group says which applies."""
    ids = _question(engine, "lr", 2, closed=False)
    for nid in ids:
        _refute(engine, nid)

    done = engine.get_next_targets()[0]
    assert done.reason == "dead_question"
    assert "declared open" in done.rationale
    assert "Nothing downstream has been pruned" in done.rationale


@pytest.mark.unit
def test_the_navigator_blames_the_list_for_a_closed_question(engine: HypoTreeEngine) -> None:
    ids = _question(engine, "a", 2, closed=True)
    for nid in ids:
        _refute(engine, nid)

    done = engine.get_next_targets()[0]
    assert done.reason == "dead_question"
    assert "exclusion_closed=false" in done.rationale


# -- withdrawing a deduction ---------------------------------------------------


@pytest.mark.unit
def test_a_shortfall_hands_back_the_belief_with_no_measurement_behind_it(
    engine: HypoTreeEngine,
) -> None:
    """The state that stranded a real run at `empty_frontier` with the goal unmet.

    Every question is answered, the assembled answer falls short, and one of the
    confirmations was deduced rather than observed. Its rivals were refuted on
    their own evidence, so there is no retired sibling to hand back — and the
    deduction is the only belief in the assembly nothing was ever measured
    about. It is handed back for a real test rather than asserted false.
    """
    a = _question(engine, "a", 2, closed=True)
    _refute(engine, a[0])
    # a[1] is now confirmed by elimination, with nothing observed about it. Its
    # rival was refuted on its own evidence rather than retired by the exclusion
    # inference, so there is no sibling to hand back — which is exactly the state
    # that left the recovery with nothing to do.
    assert engine._is_deduced(engine._store.get_node(a[1]))
    engine.create_hypothesis("standalone", node_id="c")
    engine.record_evidence("c", LogicalEvidence(success=1.0, depth=1))

    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=[a[1], "c"], edge_type=EdgeType.DEPENDENCY
    )
    # Conclusive but below the bar: it settles without failing, so nothing is
    # blamed and nothing is reopened — the case the engine had no reading for.
    for _ in range(6):
        engine.record_evidence("combo", LogicalEvidence(success=0.2, depth=2))

    target = engine.get_next_targets(count=1)[0]

    # Handed back *and* handed out: the recovery does not merely un-assert the
    # deduction, it turns it into the next experiment. One probe now settles
    # whether the value was wrong or the candidate list was incomplete.
    assert target.status == "SELECTED"
    assert target.node_id == a[1]
    survivor = engine._store.get_node(a[1])
    assert survivor.status != Status.VERIFIED
    reasons = [str(r["reason"]) for r in engine._store.get_status_history(a[1])]
    assert any(r.startswith(DEDUCTION_RETRACT_PREFIX) for r in reasons)
    # Handed back, never refuted: nothing was observed, so nothing is asserted.
    assert survivor.evidence_count == 0
    assert survivor.status != Status.INVALIDATED


@pytest.mark.unit
def test_an_observed_confirmation_is_never_withdrawn(engine: HypoTreeEngine) -> None:
    """The guard. Withdrawing a measured belief would discard evidence."""
    a = _question(engine, "a", 2, closed=True)
    engine.record_evidence(a[0], LogicalEvidence(success=1.0, depth=1))

    assert engine._is_deduced(engine._store.get_node(a[0])) is False
    assert engine._withdraw_deduction(a[0], engine._store.get_node(a[0]).updated_at) is False
    assert engine._store.get_node(a[0]).status == Status.VERIFIED


@pytest.mark.unit
def test_a_deduction_that_passed_through_review_is_still_a_deduction(
    engine: HypoTreeEngine,
) -> None:
    """Conflict review rewrites the reason to "released from review", which
    describes the last thing that happened and erases where the confirmation came
    from. Reading only the newest entry made a node nothing was ever measured
    about look like one that had stood up to scrutiny."""
    a = _question(engine, "a", 2, closed=True)
    _refute(engine, a[0])
    node = engine._store.get_node(a[1])
    assert engine._is_deduced(node)

    engine._store.change_status(
        a[1], Status.NEEDS_REVISION, reason="under conflict review — re-test at depth >= 2"
    )
    engine._store.change_status(
        a[1], Status.VERIFIED, reason="released from conflict review: another member was the cause"
    )

    assert engine._is_deduced(engine._store.get_node(a[1])) is True


# -- what counts as ruled out --------------------------------------------------


@pytest.mark.unit
def test_a_sub_par_substitution_counts_as_an_elimination(engine: HypoTreeEngine) -> None:
    """The probe was spent on the composition that isolated the value, not on the
    value itself — but it was still spent, and the engine already trusts the
    inference enough to write EXHAUSTED. Requiring the node's own evidence made
    the dead-question rule miss exactly the groups diagnosis had finished with.
    """
    from hypotree.engine import SUBSTITUTION_ELIMINATE_PREFIX

    ids = _question(engine, "a", 2, closed=True)
    _refute(engine, ids[0])
    engine._store.change_status(
        ids[1], Status.EXHAUSTED, reason=f"{SUBSTITUTION_ELIMINATE_PREFIX}composition_3"
    )

    assert engine._eliminated_on_its_own_evidence(engine._store.get_node(ids[1])) is True
    assert engine._question_is_dead("a") is True


@pytest.mark.unit
def test_a_member_retired_by_a_sibling_never_counts(engine: HypoTreeEngine) -> None:
    """Nothing was observed about it, so it cannot help declare its question dead."""
    ids = _question(engine, "a", 3, closed=True)
    engine.record_evidence(ids[0], LogicalEvidence(success=1.0, depth=1))

    for nid in ids[1:]:
        node = engine._store.get_node(nid)
        assert node.status == Status.EXHAUSTED
        assert engine._eliminated_on_its_own_evidence(node) is False
    assert engine._question_is_dead("a") is False
