"""Tests for backward pruning over a complete question.

An exclusion group declares the competing answers to one question. When every
one of them is ruled out *on its own evidence*, the question has no answer among
the candidates offered, and everything that assumed one of them is dead. That is
the exact dual of deducing the last survivor, and these tests pin down both the
inference and — more importantly — the three ways it must refuse to fire.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from hypotree.engine import DEAD_QUESTION_PREFIX, HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status
from hypotree.store.store import utcnow


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[HypoTreeEngine]:
    e = HypoTreeEngine(tmp_path / "dead.db", rng_seed=11)
    yield e
    e.close()


def _question(engine: HypoTreeEngine, group: str = "colour", n: int = 3) -> list[str]:
    """A declared question with ``n`` competing answers, plus a composition on the first.

    The composition hangs off member 0 deliberately. Member 0 is the one the
    tests settle as EXHAUSTED, which the refutation cascade spares — so anything
    that happens to ``combo`` is attributable to the dead-question rule and not
    to a cascade that would have pruned it anyway.
    """
    ids = [f"{group}_v{i}" for i in range(n)]
    for nid in ids:
        engine.create_hypothesis(f"{group} is {nid}", node_id=nid, exclusion_group=group)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=[ids[0]], edge_type=EdgeType.DEPENDENCY
    )
    return ids


def _refute(engine: HypoTreeEngine, node_id: str) -> None:
    engine.record_evidence(node_id, LogicalEvidence(success=0.0, depth=1))


def _exhaust(engine: HypoTreeEngine, node_id: str) -> None:
    """Drive a node to EXHAUSTED: conclusively tested, below the bar, never refuted."""
    for _ in range(6):
        engine.record_evidence(node_id, LogicalEvidence(success=0.2, depth=1))


@pytest.mark.unit
def test_ruling_out_every_answer_prunes_what_assumed_one(engine: HypoTreeEngine) -> None:
    """The inference itself: no candidate survives, so nothing built on one can."""
    ids = _question(engine)
    _exhaust(engine, ids[0])
    assert engine._store.get_node("combo").status != Status.PRUNED, (
        "an exhausted premise must not cascade — otherwise this test proves nothing"
    )
    for nid in ids[1:]:
        _refute(engine, nid)

    combo = engine._store.get_node("combo")
    assert combo is not None
    assert combo.status == Status.PRUNED
    history = engine._store.get_status_history("combo")
    assert str(history[-1]["reason"]).startswith(DEAD_QUESTION_PREFIX)
    assert "colour" in str(history[-1]["reason"])


@pytest.mark.unit
def test_it_reaches_what_the_refutation_cascade_leaves_alone(engine: HypoTreeEngine) -> None:
    """The case that makes the mechanism worth having.

    A member settled as EXHAUSTED was never refuted, so the ordinary cascade
    deliberately spares its descendants. Without this rule a composition resting
    on one stays on the frontier being re-attempted after its premise ran out.
    """
    ids = _question(engine)
    # Conclusive but below the bar: EXHAUSTED, no cascade.
    for nid in ids:
        for _ in range(6):
            engine.record_evidence(nid, LogicalEvidence(success=0.2, depth=1))

    assert all(engine._store.get_node(n).status == Status.EXHAUSTED for n in ids)
    combo = engine._store.get_node("combo")
    assert combo is not None and combo.status == Status.PRUNED


@pytest.mark.unit
def test_what_rested_on_a_pruned_dependent_goes_too(engine: HypoTreeEngine) -> None:
    """Stopping at the direct dependents leaves the next layer on the frontier.

    A node whose only support has just been pruned is exactly the waste this
    mechanism removes one level up.
    """
    ids = _question(engine)
    engine.create_hypothesis(
        "downstream", node_id="downstream", parent_ids=["combo"], edge_type=EdgeType.DEPENDENCY
    )
    _exhaust(engine, ids[0])
    for nid in ids[1:]:
        _refute(engine, nid)

    assert engine._store.get_node("combo").status == Status.PRUNED
    assert engine._store.get_node("downstream").status == Status.PRUNED


@pytest.mark.unit
def test_an_unasked_question_is_not_a_dead_one(engine: HypoTreeEngine) -> None:
    """The failure mode that would destroy a live search.

    Two answers refuted and one never probed is a question still being asked.
    """
    ids = _question(engine)
    # Everything except the one `combo` rests on, so no cascade can reach it and
    # the assertion is about the rule rather than about the cascade.
    for nid in ids[1:]:
        _refute(engine, nid)

    assert engine._question_is_dead("colour") is False
    assert engine._store.get_node("combo").status != Status.PRUNED
    assert engine._dead_questions() == []


@pytest.mark.unit
def test_a_member_retired_by_a_confirmed_sibling_does_not_count(engine: HypoTreeEngine) -> None:
    """A single confirmation must not be able to kill its own question."""
    ids = _question(engine)
    engine.record_evidence(ids[0], LogicalEvidence(success=1.0, depth=1))

    # The siblings are EXHAUSTED, but by the exclusion inference and with no
    # observation of their own.
    assert all(engine._store.get_node(n).status == Status.EXHAUSTED for n in ids[1:])
    assert engine._question_is_dead("colour") is False


@pytest.mark.unit
def test_a_node_killed_by_its_ancestry_does_not_count(engine: HypoTreeEngine) -> None:
    """PRUNED says something about an ancestor's subtree, not about this answer."""
    engine.create_hypothesis("root", node_id="root")
    ids = []
    for i in range(2):
        nid = f"leaf_v{i}"
        ids.append(nid)
        engine.create_hypothesis(
            f"leaf {i}", node_id=nid, exclusion_group="leaf", parent_ids=["root"]
        )
    _refute(engine, ids[0])
    _refute(engine, "root")

    assert engine._store.get_node(ids[1]).status == Status.PRUNED
    # Pruned by ancestry with no evidence of its own — the question was never
    # actually put to it.
    assert engine._question_is_dead("leaf") is False


@pytest.mark.unit
def test_one_candidate_is_not_a_question(engine: HypoTreeEngine) -> None:
    """A lone refuted node is the ordinary cascade's job, not this one."""
    engine.create_hypothesis("only", node_id="only", exclusion_group="solo")
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["only"], edge_type=EdgeType.DEPENDENCY
    )
    _refute(engine, "only")

    assert engine._question_is_dead("solo") is False
    history = engine._store.get_status_history("combo")
    # Still pruned — but by the refutation cascade, under its own reason.
    assert engine._store.get_node("combo").status == Status.PRUNED
    assert not str(history[-1]["reason"]).startswith(DEAD_QUESTION_PREFIX)


@pytest.mark.unit
def test_a_group_under_open_conflict_is_mid_argument_not_dead(engine: HypoTreeEngine) -> None:
    """Conflict diagnosis may hand a member straight back, so the group is not out.

    The guard is asserted directly on the predicate: reaching this state through
    the recovery machinery would test the recovery, not the guard, and the guard
    is the thing that stops a live argument from being reported as a dead end.
    """
    ids = _question(engine)
    _exhaust(engine, ids[0])
    for nid in ids[1:]:
        _refute(engine, nid)
    assert engine._question_is_dead("colour") is True

    engine._store.add_nogood("combo", [ids[0], "elsewhere"], utcnow(), conflict_depth=1)
    assert engine._question_is_dead("colour") is False


@pytest.mark.unit
def test_the_navigator_names_the_question_that_ran_out(engine: HypoTreeEngine) -> None:
    """`dead_question` exists because `blocked_frontier` sends the caller to fix
    edges that are already correct."""
    ids = _question(engine)
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.8)
    _exhaust(engine, ids[0])
    for nid in ids[1:]:
        _refute(engine, nid)

    done = engine.get_next_targets()[0]
    assert done.status == "DONE"
    assert done.reason == "dead_question"
    assert "colour" in done.rationale
    assert "exclusion_group" in done.rationale
