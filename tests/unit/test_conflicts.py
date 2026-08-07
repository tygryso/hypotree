"""Tests for conflict sets (nogoods) — deferred, then provable, blame.

A hypothesis resting on several assumptions that fails proves exactly one thing:
those assumptions cannot all hold together. These tests pin down that the engine
records that fact instead of accusing every assumption, and that it converts the
conflict into a single culprit only once the evidence makes that inescapable.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from hypotree.engine import (
    INTERACTION_REOPEN_PREFIX,
    MAX_REVIEW_DISPATCHES,
    UNDERPERFORMANCE_REOPEN_PREFIX,
    ClaimError,
    HypoTreeEngine,
    NodeNotFoundError,
)
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status
from hypotree.store.store import utcnow


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    e = HypoTreeEngine(tmp_path / "conflicts.db", rng_seed=7)
    yield e
    e.close()


def _confirm(engine: HypoTreeEngine, node_id: str, depth: int = 0) -> None:
    """Drive a node to VERIFIED with a decisive success at ``depth``."""
    engine.record_evidence(node_id, LogicalEvidence(success=1.0, depth=depth))


def _exhaust_substitutions(engine: HypoTreeEngine) -> None:
    """Run every diagnostic swap the engine asks for, all of them failing.

    Drives a conflict to the interaction-effect ending: each assumption is
    removed in turn, the composition fails regardless, so none of them is the
    sole cause.
    """
    for i in range(8):
        done = engine.get_next_targets()[0]
        if done.reason != "awaiting_substitution":
            return
        parents = re.findall(r"'([a-z]+_v\d)'", done.rationale)
        keep = re.search(r"parent_ids=\[([^\]]*)\]", done.rationale)
        assert keep is not None, done.rationale
        ids = [s.strip().strip("'") for s in keep.group(1).split(",")]
        engine.create_hypothesis(
            f"swap{i}", node_id=f"swap{i}", parent_ids=ids, edge_type=EdgeType.DEPENDENCY
        )
        engine.record_evidence(f"swap{i}", LogicalEvidence(success=0.0, depth=2))
        assert parents  # the advice always names the member it is testing


def _refute(engine: HypoTreeEngine, node_id: str, depth: int = 0) -> None:
    """Drive a node to INVALIDATED with a decisive failure at ``depth``."""
    engine.record_evidence(node_id, LogicalEvidence(success=0.0, depth=depth))


def _build_two_axis_landscape(engine: HypoTreeEngine) -> None:
    """Two questions of two answers each, plus a combination resting on one of each.

    ``combo1`` depends on ``comp_v1`` and ``reg_v1``; both are confirmed at depth
    1 first, so a later failure of ``combo1`` at depth 2 is exactly the
    indeterminate case: two confirmed assumptions, one failure, and neither
    confirmation deep enough to cover the context that failed.
    """
    for group, ids in (("component", ("comp_v1", "comp_v2")), ("regime", ("reg_v1", "reg_v2"))):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    engine.create_hypothesis(
        "combo1",
        node_id="combo1",
        parent_ids=["comp_v1", "reg_v1"],
        edge_type=EdgeType.DEPENDENCY,
    )
    _confirm(engine, "comp_v1", depth=1)
    _confirm(engine, "reg_v1", depth=1)


# -- store layer ---------------------------------------------------------------


@pytest.mark.unit
def test_store_nogood_round_trip(engine: HypoTreeEngine) -> None:
    now = datetime.now()
    nid = engine._store.add_nogood("combo1", ["a", "b"], now)

    open_before = engine._store.get_nogoods(open_only=True)
    assert len(open_before) == 1
    assert open_before[0]["member_ids"] == ["a", "b"]
    assert open_before[0]["resolved_at"] is None

    engine._store.resolve_nogood(nid, "b", now)
    assert engine._store.get_nogoods(open_only=True) == []
    resolved = engine._store.get_nogoods(open_only=False)
    assert len(resolved) == 1
    assert resolved[0]["resolved_culprit_id"] == "b"


@pytest.mark.unit
def test_store_nogoods_newest_first(engine: HypoTreeEngine) -> None:
    """The suggester reasons about the most recent failure, so order matters."""
    now = datetime.now()
    engine._store.add_nogood("combo1", ["a"], now)
    engine._store.add_nogood("combo2", ["b"], now)
    assert [n["member_ids"] for n in engine._store.get_nogoods()] == [["b"], ["a"]]


# -- blame attribution ---------------------------------------------------------


@pytest.mark.unit
def test_multi_parent_failure_records_conflict_and_spares_assumptions(
    engine: HypoTreeEngine,
) -> None:
    """Neither assumption is refuted; both are put under review at the failure depth."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    # Nothing is refuted — the failure only shows the two cannot hold together.
    assert engine._store.get_node("comp_v1").status == Status.NEEDS_REVISION
    assert engine._store.get_node("reg_v1").status == Status.NEEDS_REVISION
    assert engine._store.get_node("combo1").status == Status.INVALIDATED

    conflicts = engine.get_conflicts()
    assert len(conflicts) == 1
    assert sorted(conflicts[0]["member_ids"]) == ["comp_v1", "reg_v1"]
    assert sorted(conflicts[0]["remaining_suspects"]) == ["comp_v1", "reg_v1"]
    assert conflicts[0]["conflict_depth"] == 2


@pytest.mark.unit
def test_conflict_members_are_not_dispatched_for_isolated_re_testing(
    engine: HypoTreeEngine,
) -> None:
    """A part that already passed its own test is not re-run to explain a whole.

    Every member of a conflict has, by construction, passed the isolated test —
    that is exactly why the conflict is indeterminate. Handing them back as
    individual targets bought nothing and cost a tenth of a real run's entire
    probe budget re-confirming assumptions nobody doubted.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    frontier = {n.id for n in engine._frontier_nodes()}
    assert "comp_v1" not in frontier
    assert "reg_v1" not in frontier


@pytest.mark.unit
def test_suggestion_proposes_a_substitution_not_a_re_test(engine: HypoTreeEngine) -> None:
    """One swap eliminates a whole question; one re-test eliminates nothing.

    Rebuilding the failed combination with a single assumption replaced answers
    "was it this one?" outright. Re-running that assumption alone re-asks a
    question already answered.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    suggestion = engine.suggest_discriminating_experiment()
    assert suggestion["status"] == "SUGGESTED"
    assert suggestion["action"] == "substitute"
    assert suggestion["node_id"] in {"comp_v1", "reg_v1"}
    assert suggestion["replace_with"] in {"comp_v2", "reg_v2"}
    assert suggestion["min_depth"] == 2
    # The caller is handed the exact parent set to build, not left to derive it.
    assert len(suggestion["parent_ids"]) == 2
    assert suggestion["node_id"] not in suggestion["parent_ids"]


@pytest.mark.unit
def test_suggestion_recombines_once_no_suspect_remains(engine: HypoTreeEngine) -> None:
    """With every assumption vindicated the failure is an interaction effect.

    Nothing is left to re-test, so the useful move becomes a different
    combination — and it should differ from the refuted one as little as
    possible, to keep the next result interpretable.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    # Both assumptions survive a re-test at the failure depth.
    _exhaust_substitutions(engine)

    suggestion = engine.suggest_discriminating_experiment()
    assert suggestion["status"] == "SUGGESTED"
    assert suggestion["action"] == "recombine"
    assert suggestion["changed_assumptions"] == 1
    chosen = {a["node_id"] for a in suggestion["assignment"]}
    assert chosen != {"comp_v1", "reg_v1"}
    assert len(chosen) == 2


@pytest.mark.unit
def test_suggestion_never_repeats_a_known_conflict(engine: HypoTreeEngine) -> None:
    """Every distinct pairing is ruled out, so there is nothing left to propose."""
    _build_two_axis_landscape(engine)
    now = datetime.now()
    # Every candidate has survived a test as demanding as the conflicts below, so
    # none of them is a suspect and the suggester must reason about combinations.
    for nid in ("comp_v1", "comp_v2", "reg_v1", "reg_v2"):
        engine._store.set_confirmed_depth(nid, 1)
    for combo in (
        ("comp_v1", "reg_v1"),
        ("comp_v1", "reg_v2"),
        ("comp_v2", "reg_v1"),
        ("comp_v2", "reg_v2"),
    ):
        engine._store.add_nogood("combo1", list(combo), now, conflict_depth=1)

    assert engine.suggest_discriminating_experiment()["status"] == "EXHAUSTED"


@pytest.mark.unit
def test_suggestion_ignores_refuted_alternatives(engine: HypoTreeEngine) -> None:
    """A refuted sibling is not a candidate, so it never appears in a suggestion."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    # Both assumptions are swapped out and the composition fails anyway, so the
    # failure is an interaction effect and the suggester moves to recombination.
    _exhaust_substitutions(engine)
    engine._store.set_confirmed_depth("reg_v2", 2)
    engine._store.change_status(
        "comp_v2", Status.INVALIDATED, reason="probed and refuted", now=datetime.now()
    )
    engine._refresh_node_in_graph("comp_v2")

    suggestion = engine.suggest_discriminating_experiment()
    assert suggestion["action"] == "recombine"
    assert "comp_v2" not in {a["node_id"] for a in suggestion["assignment"]}


@pytest.mark.unit
def test_suggestion_without_exclusion_groups_is_exhausted(engine: HypoTreeEngine) -> None:
    """Nothing can be varied if the assumptions never said what question they answer."""
    engine.create_hypothesis("p1", node_id="p1")
    engine.create_hypothesis("p2", node_id="p2")
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["p1", "p2"], edge_type=EdgeType.DEPENDENCY
    )
    _confirm(engine, "p1", depth=2)
    _confirm(engine, "p2", depth=2)
    # Failure at the same depth the assumptions were confirmed at: no suspect,
    # so the suggester goes straight to recombination and finds nothing to vary.
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    result = engine.suggest_discriminating_experiment()
    assert result["status"] == "EXHAUSTED"
    assert "exclusion groups" in result["reason"]


@pytest.mark.unit
def test_get_conflicts_marks_exonerated_members(engine: HypoTreeEngine) -> None:
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0))

    entries: list[dict[str, Any]] = engine.get_conflicts()
    members = {m["node_id"]: m for m in entries[0]["members"]}
    assert set(members) == {"comp_v1", "reg_v1"}
    assert all(m["statement"] for m in members.values())
    assert not any(m["exonerated"] for m in members.values())


# -- selection cursor ----------------------------------------------------------


@pytest.mark.unit
def test_dispatch_moves_the_group_cursor(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("only", node_id="only", exclusion_group="component")
    engine.get_next_targets()[0]
    assert engine._last_selected_group == "component"


@pytest.mark.unit
def test_dry_run_leaves_the_group_cursor_alone(engine: HypoTreeEngine) -> None:
    """A peek is not a dispatch, so it must not steer the next real selection."""
    engine.create_hypothesis("only", node_id="only", exclusion_group="component")
    engine.get_next_targets(dry_run=True)[0]
    assert engine._last_selected_group is None


@pytest.mark.unit
def test_conviction_reopens_the_culprits_alternatives(engine: HypoTreeEngine) -> None:
    """A convicted assumption loses the authority to keep its rivals retired.

    The correct answer to that question is very likely among them, so leaving
    them EXHAUSTED would bury it.
    """
    for nid in ("comp_v1", "comp_v2", "comp_v3"):
        engine.create_hypothesis(nid, node_id=nid, exclusion_group="component")
    engine.create_hypothesis("reg_v1", node_id="reg_v1", exclusion_group="regime")
    engine.create_hypothesis("reg_v2", node_id="reg_v2", exclusion_group="regime")
    engine.create_hypothesis(
        "combo1", node_id="combo1", parent_ids=["comp_v1", "reg_v1"], edge_type=EdgeType.DEPENDENCY
    )
    _confirm(engine, "comp_v1")
    _confirm(engine, "reg_v1")
    engine.record_evidence("combo1", LogicalEvidence(success=0.0))

    # A combination reusing reg_v1 succeeds, so reg_v1 is cleared and comp_v1 is
    # the sole surviving suspect.
    engine.create_hypothesis(
        "combo2", node_id="combo2", parent_ids=["comp_v2", "reg_v1"], edge_type=EdgeType.DEPENDENCY
    )
    _confirm(engine, "comp_v2")
    _confirm(engine, "combo2")

    assert engine._store.get_node("comp_v1").status == Status.INVALIDATED
    # comp_v2's own confirmation now governs the question; comp_v3 stays settled
    # by that live confirmation rather than by the convicted one.
    assert engine._store.get_node("comp_v2").status == Status.VERIFIED
    assert engine._store.get_node("comp_v3").status == Status.EXHAUSTED


@pytest.mark.unit
def test_overwriting_a_node_drops_its_conflicts(engine: HypoTreeEngine) -> None:
    """A conflict about a redefined node can never narrow, so it is discarded."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0))
    assert len(engine._store.get_nogoods()) == 1

    engine.create_hypothesis(
        "comp_v1 redefined", node_id="comp_v1", exclusion_group="component", if_exists="overwrite"
    )

    assert engine._store.get_nogoods() == []


# -- substitution diagnosis ----------------------------------------------------


def _substitute(engine: HypoTreeEngine, node_id: str, parents: list[str], success: float) -> None:
    """Build and probe the swap the engine just asked for."""
    engine.create_hypothesis(
        node_id, node_id=node_id, parent_ids=parents, edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence(node_id, LogicalEvidence(success=success, depth=2))


@pytest.mark.unit
def test_a_conflict_asks_for_a_swap_not_a_re_test(engine: HypoTreeEngine) -> None:
    """The engine names both halves of the experiment, so building it needs no thought."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    done = engine.get_next_targets()[0]
    assert done.status == "DONE"
    assert done.reason == "awaiting_substitution"
    assert done.min_depth == 2
    # It must name a member to drop and a live alternative to put in its place.
    assert "comp_v1" in done.rationale or "reg_v1" in done.rationale
    assert "comp_v2" in done.rationale or "reg_v2" in done.rationale


@pytest.mark.unit
def test_a_swap_that_still_fails_clears_the_assumption_it_removed(
    engine: HypoTreeEngine,
) -> None:
    """Removing it changed nothing, so it was not what the failure was about.

    The cleared assumption keeps its confirmation and its alternatives stay
    retired: nothing refuted it. One probe has eliminated an entire question.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.0)

    nogood = engine._store.get_nogoods()[0]
    assert nogood["probe_index"] == 1
    assert nogood["cleared_ids"] == ["comp_v1"]
    assert nogood["resolved_at"] is None
    assert engine._store.get_node("comp_v1").status == Status.NEEDS_REVISION
    assert engine._store.get_node("comp_v2").status == Status.EXHAUSTED
    # ...and the engine now asks about the *other* assumption.
    assert "reg_v1" in engine.get_next_targets()[0].rationale


def _strip_substitute(engine: HypoTreeEngine, member_id: str) -> None:
    """Refute every live alternative to ``member_id``, so no swap can be built for it."""
    node = engine._store.get_node(member_id)
    assert node is not None and node.exclusion_group
    for sibling in engine._store.get_nodes_in_exclusion_group(node.exclusion_group, member_id):
        engine.update_status(sibling.id, Status.INVALIDATED, reason="ruled out separately")


@pytest.mark.unit
def test_a_member_with_no_substitute_is_reported_as_skipped_not_cleared(
    engine: HypoTreeEngine,
) -> None:
    """Opposite claims: one says it was tested and exonerated, the other untested.

    The integer cursor this replaced could only say "the first k were dealt
    with", so a member the plan passed over for want of a live alternative was
    left *behind* the cursor the moment a later member was cleared, and reported
    as cleared — crediting the belief state with a conclusion no experiment had
    produced. The first member in the diagnosis order is stripped deliberately,
    because that is the position where the two records disagree.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    first, second = engine._store.get_nogoods()[0]["member_ids"]
    _strip_substitute(engine, first)

    members = {m["node_id"]: m for m in engine.get_conflicts()[0]["members"]}
    assert members[first]["skipped_no_substitute"] is True
    assert members[first]["cleared_by_substitution"] is False

    # Clear the member that *can* be swapped out. A cursor would now sit past
    # both and report both as cleared.
    stand_in = engine._store.get_nodes_in_exclusion_group(
        engine._store.get_node(second).exclusion_group, second
    )[0]
    _substitute(engine, "swap", [first, stand_in.id], success=0.0)

    assert engine._store.get_nogoods()[0]["cleared_ids"] == [second]


@pytest.mark.unit
def test_a_skipped_member_is_revisited_once_a_substitute_frees_up(
    engine: HypoTreeEngine,
) -> None:
    """A cursor could never come back to it; a cleared-set can.

    The member was passed over because its question had no live alternative. The
    moment one appears, the swap that interrogates it exists — and the diagnosis
    stops reporting it as unreachable instead of giving up and reopening every
    question by hand.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    first, second = engine._store.get_nogoods()[0]["member_ids"]
    _strip_substitute(engine, first)

    plan = engine._substitution_plan(engine._store.get_nogoods()[0])
    assert plan["member_id"] == second
    assert plan["skipped"] == [first]

    # A fresh answer to the same question appears.
    group = engine._store.get_node(first).exclusion_group
    engine.create_hypothesis("late_v9", node_id="late_v9", exclusion_group=group)

    plan = engine._substitution_plan(engine._store.get_nogoods()[0])
    assert plan["member_id"] == first
    assert plan["candidate_id"] == "late_v9"
    assert plan["skipped"] == []


@pytest.mark.unit
def test_a_repeated_swap_clears_a_member_only_once(engine: HypoTreeEngine) -> None:
    """A caller re-reporting the same swap has established nothing new."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    nogood_id = engine._store.get_nogoods()[0]["id"]

    engine._store.clear_nogood_member(nogood_id, "comp_v1", datetime.now())
    cleared = engine._store.clear_nogood_member(nogood_id, "comp_v1", datetime.now())

    assert cleared == ["comp_v1"]
    assert engine._store.get_nogoods()[0]["probe_index"] == 1


@pytest.mark.unit
def test_the_deprecated_cursor_api_still_clears_the_right_members(
    engine: HypoTreeEngine,
) -> None:
    """Deprecation precedes removal: an external caller is warned, not broken."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    nogood = engine._store.get_nogoods()[0]

    with pytest.deprecated_call():
        engine._store.advance_nogood_probe(nogood["id"], 1, datetime.now())

    assert engine._store.get_nogoods()[0]["cleared_ids"] == [nogood["member_ids"][0]]


@pytest.mark.unit
def test_a_failing_swap_does_not_open_a_second_conflict(engine: HypoTreeEngine) -> None:
    """Its failure was predicted, so recording it as news would double the work."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.0)

    assert len(engine._store.get_nogoods()) == 1


@pytest.mark.unit
def test_a_swap_that_stops_failing_convicts_the_assumption_it_removed(
    engine: HypoTreeEngine,
) -> None:
    """Removing it was the only change, so it was the cause.

    Convicting reopens exactly that question's alternatives — and only that
    question's. The other assumption is released, its confirmation intact.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.5)

    nogood = engine._store.get_nogoods()[0]
    assert nogood["resolved_culprit_id"] == "comp_v1"
    assert engine._store.get_node("comp_v1").status == Status.INVALIDATED
    assert engine._store.get_node("reg_v1").status == Status.VERIFIED
    # Only the convicted question reopens; the innocent one stays settled.
    assert engine._store.get_node("reg_v2").status == Status.EXHAUSTED


@pytest.mark.unit
def test_a_swap_that_clears_the_bar_confirms_the_value_that_replaced_the_culprit(
    engine: HypoTreeEngine,
) -> None:
    """A composition that *achieves the objective* names its parts.

    The same combination failed with the convicted member in that slot and now
    clears the bar with this one, so the substitute is the answer and not merely
    a different one. Leaving it UNTESTED handed the navigator the very answer
    the diagnosis had just produced.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=1.0)

    substitute = engine._store.get_node("comp_v2")
    assert substitute.status == Status.VERIFIED
    # Confirmed at the depth the composition was actually tested at, so a later,
    # deeper failure is still entitled to reopen it.
    assert substitute.confirmed_depth == 2
    assert "comp_v2" not in {n.id for n in engine._frontier_nodes()}
    assert "comp_v2" not in engine._conflict_suspects()


@pytest.mark.unit
def test_a_swap_that_only_stops_the_failure_does_not_pick_the_answer(
    engine: HypoTreeEngine,
) -> None:
    """The swap is chosen to be *different*, not to be correct.

    A composition that stops failing at a sub-par score has answered exactly one
    question — which member was at fault — and nothing about which value is
    right. Confirming the substitute there retires its siblings, burying the
    correct answer behind an exclusion nobody ever tested. Three real episodes
    ended on an empty frontier with the goal unmet because of it, at 14, 16 and
    23 probes out of a budget of 100.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.5)

    # The verdict still lands: removing comp_v1 was the only change.
    assert engine._store.get_nogoods()[0]["resolved_culprit_id"] == "comp_v1"
    assert engine._store.get_node("comp_v1").status == Status.INVALIDATED
    # The substitute is not the declared answer either — it is not VERIFIED.
    assert engine._store.get_node("comp_v2").status != Status.VERIFIED


@pytest.mark.unit
def test_a_sub_par_swap_rules_the_substitute_out(engine: HypoTreeEngine) -> None:
    """The conviction exonerates every other slot, so the swapped one is refuted.

    The rebuilt composition holds the right answer everywhere except the slot
    that was swapped, and it still fell short of the bar. That rules the
    substitute out for its own question. Leaving it open put it straight back on
    the frontier: nine of fifteen conflict episodes in the last run spent a
    probe re-establishing in isolation what the swap had already shown, and all
    nine came back refuted.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.5)

    assert engine._store.get_node("comp_v2").status == Status.EXHAUSTED
    assert "comp_v2" not in {n.id for n in engine._frontier_nodes()}


@pytest.mark.unit
def test_a_sub_par_swap_leaves_the_substitute_posterior_alone(
    engine: HypoTreeEngine,
) -> None:
    """Nothing was observed of the substitute on its own, so nothing is recorded."""
    _build_two_axis_landscape(engine)
    before = engine._store.get_node("comp_v2")
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.5)

    after = engine._store.get_node("comp_v2")
    assert (after.alpha, after.beta) == (before.alpha, before.beta)
    assert after.evidence_count == 0


@pytest.mark.unit
def test_a_successful_swap_leaves_the_posterior_alone(engine: HypoTreeEngine) -> None:
    """Status records what was established; the posterior records what was seen.

    Nothing was ever observed of the substitute on its own, exactly as for
    deduction by elimination, so inventing a Beta update for it would put a
    fabricated observation into the belief state.
    """
    _build_two_axis_landscape(engine)
    before = engine._store.get_node("comp_v2")
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=1.0)

    after = engine._store.get_node("comp_v2")
    assert (after.alpha, after.beta) == (before.alpha, before.beta)
    assert after.evidence_count == 0


@pytest.mark.unit
def test_a_substitute_settled_by_its_own_evidence_is_not_overwritten(
    engine: HypoTreeEngine,
) -> None:
    """An observation of the node outranks an inference drawn about it.

    The swap says the composition works; it cannot un-refute a value that was
    directly tested and failed, and quietly promoting one would bury the signal.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    engine.update_status("comp_v2", Status.INVALIDATED, reason="probed and failed")

    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=1.0)

    assert engine._store.get_node("comp_v2").status == Status.INVALIDATED


@pytest.mark.unit
def test_a_convicted_questions_alternatives_are_prioritised(engine: HypoTreeEngine) -> None:
    """The answer is almost certainly there, and every candidate looks identical.

    Freshly reopened alternatives all carry an untouched prior, so without a
    priority the navigator picks among them at random and the targeted recovery
    degenerates into the blind sweep it exists to prevent.

    Conviction here comes from an outright refutation rather than from a swap:
    a swap that succeeds also *names* the replacement, so it leaves nothing to
    choose between. Three answers, so refuting one does not leave a single
    survivor to be deduced.
    """
    for group, ids in (
        ("component", ("comp_v1", "comp_v2", "comp_v3")),
        ("regime", ("reg_v1", "reg_v2")),
    ):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    engine.create_hypothesis(
        "combo1", node_id="combo1", parent_ids=["comp_v1", "reg_v1"], edge_type=EdgeType.DEPENDENCY
    )
    _confirm(engine, "comp_v1", depth=1)
    _confirm(engine, "reg_v1", depth=1)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    # A direct refutation of one member explains the conflict on its own.
    _refute(engine, "comp_v1", depth=2)

    assert engine._store.get_nogoods()[0]["resolved_culprit_id"] == "comp_v1"
    assert {"comp_v2", "comp_v3"} <= set(engine._conflict_suspects())


@pytest.mark.unit
def test_clearing_every_assumption_reopens_all_the_alternatives(
    engine: HypoTreeEngine,
) -> None:
    """The other honest ending: nobody is guilty, so look everywhere they closed.

    Every assumption has been swapped out and the composition failed each time,
    so no single one is the cause. At least one confirmation holds in isolation
    and not in composition, and the value that does work must be among the
    alternatives those confirmations retired.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.0)
    _substitute(engine, "swap_reg", ["comp_v1", "reg_v2"], success=0.0)

    nogood = engine._store.get_nogoods()[0]
    assert nogood["reopened_at"] is not None
    assert nogood["resolved_at"] is None  # still open: nobody was blamed
    # Nobody was convicted — the confirmations stand.
    assert engine._store.get_node("comp_v1").status == Status.VERIFIED
    assert engine._store.get_node("reg_v1").status == Status.VERIFIED
    # ...but the questions they closed are open again, and dispatchable.
    frontier = {n.id for n in engine._frontier_nodes()}
    assert {"comp_v2", "reg_v2"} <= frontier


@pytest.mark.unit
def test_the_broad_recovery_happens_once(engine: HypoTreeEngine) -> None:
    """Repeating it would undo every answer found after the conflict."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap_comp", ["comp_v2", "reg_v1"], success=0.0)
    _substitute(engine, "swap_reg", ["comp_v1", "reg_v2"], success=0.0)

    _refute(engine, "comp_v2", depth=2)
    assert engine._store.get_node("comp_v2").status == Status.INVALIDATED

    assert engine._recover_from_interaction(engine._store.get_nogoods()[0], datetime.now()) == []
    assert engine._store.get_node("comp_v2").status == Status.INVALIDATED


@pytest.mark.unit
def test_nothing_is_reopened_while_a_swap_is_still_untried(engine: HypoTreeEngine) -> None:
    """While a targeted swap could still name the culprit, a blind sweep is waste."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    assert engine._store.get_node("comp_v2").status == Status.EXHAUSTED
    assert engine._store.get_node("reg_v2").status == Status.EXHAUSTED
    assert engine._store.get_nogoods()[0]["reopened_at"] is None


@pytest.mark.unit
def test_advice_that_is_never_acted_on_falls_back_to_the_broad_recovery(
    engine: HypoTreeEngine,
) -> None:
    """A caller that will not run the experiment must not be stuck forever."""
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    reasons = [engine.get_next_targets()[0].reason for _ in range(MAX_REVIEW_DISPATCHES + 1)]

    assert reasons[0] == "awaiting_substitution"
    assert reasons[-1] != "awaiting_substitution"
    assert engine._store.get_nogoods()[0]["reopened_at"] is not None


@pytest.mark.unit
def test_clearing_an_assumption_does_not_consume_the_advice_budget(
    engine: HypoTreeEngine,
) -> None:
    """Progress must not count against the give-up bound.

    A flat budget for the whole narrowing abandoned a five-assumption conflict
    two swaps from the answer: each successful elimination spent one of the three
    allowed instructions, so the diagnosis ran out before the members did.
    """
    for group in ("a", "b", "c", "d"):
        for i in range(3):
            engine.create_hypothesis(f"{group}{i}", node_id=f"{group}{i}", exclusion_group=group)
    firsts = [f"{g}0" for g in ("a", "b", "c", "d")]
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=firsts, edge_type=EdgeType.DEPENDENCY
    )
    for nid in firsts:
        _confirm(engine, nid, depth=1)
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    # Four members, each cleared in turn: more instructions than the per-conflict
    # bound, and every one of them must still be offered.
    named = []
    for i in range(4):
        done = engine.get_next_targets()[0]
        assert done.reason == "awaiting_substitution", (i, done.reason)
        ids = re.search(r"parent_ids=\[([^\]]*)\]", done.rationale)
        assert ids is not None
        parents = [s.strip().strip("'") for s in ids.group(1).split(",")]
        named.append(sorted(set(firsts) - set(parents))[0])
        engine.create_hypothesis(
            f"swap{i}", node_id=f"swap{i}", parent_ids=parents, edge_type=EdgeType.DEPENDENCY
        )
        engine.record_evidence(f"swap{i}", LogicalEvidence(success=0.0, depth=2))

    assert sorted(named) == sorted(firsts)
    assert engine._store.get_nogoods()[0]["reopened_at"] is not None


@pytest.mark.unit
def test_giving_up_hands_back_the_work_it_just_reopened(engine: HypoTreeEngine) -> None:
    """A recovery that creates work must not report that there is none.

    The frontier is computed before the DONE reason is chosen, so a fallback that
    reopens alternatives leaves the caller holding a stale "nothing is testable"
    — announcing the end of the search at the exact moment new work appeared.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    # Ignore the advice until the engine gives up on the targeted route.
    for _ in range(MAX_REVIEW_DISPATCHES):
        engine.get_next_targets()

    result = engine.get_next_targets()[0]

    assert result.status == "SELECTED"
    assert result.node_id in {"comp_v2", "reg_v2"}


@pytest.mark.unit
def test_a_peek_never_gives_up_on_a_conflict(engine: HypoTreeEngine) -> None:
    """`dry_run=True` is a peek, and a peek that abandons a narrowing is a write.

    Giving up on the targeted route reopens the alternatives its members had
    retired and marks the conflict recovered. Reached through a dry run, that
    settled the very question the caller was only asking about — and the next
    real call then saw a belief state a peek had moved.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    for _ in range(MAX_REVIEW_DISPATCHES):
        engine.get_next_targets(dry_run=True)

    peek = engine.get_next_targets(dry_run=True)[0]

    assert [n["reopened_at"] for n in engine._store.get_nogoods()] == [None]
    assert engine._store.get_node("comp_v2").status == Status.EXHAUSTED
    # It still says something useful — it just says it without moving anything.
    assert peek.status == "DONE"
    assert peek.reason == "awaiting_substitution"


# -- the search must not declare itself over while the goal is unmet ---------


def _five_axis_landscape(engine: HypoTreeEngine, values: int = 5) -> None:
    """Five questions of ``values`` answers each, wired to a goal — the eval shape."""
    for group in ("comp", "meth", "param", "reg", "enc"):
        for i in range(values):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)


@pytest.mark.unit
def test_a_sub_par_swap_does_not_strand_the_search(engine: HypoTreeEngine) -> None:
    """The exact shape that ended three real episodes early.

    Five axes, one confirmed answer each, a combination that fails, four swaps
    that fail and a fifth that stops the failure at a sub-par score. Confirming
    that fifth substitute re-retired its siblings and emptied the frontier with
    the objective unreached, at 23 probes out of a budget of 100 — reported as a
    finished search.
    """
    _five_axis_landscape(engine)
    winners = ["comp3", "meth3", "param2", "reg2", "enc2"]
    for w in winners:
        _confirm(engine, w, depth=1)
    engine.create_hypothesis(
        "combo1", node_id="combo1", parent_ids=winners, edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))

    for i, parents in enumerate(
        [
            ["comp1", "meth3", "param2", "reg2", "enc2"],
            ["comp3", "meth3", "param2", "reg2", "enc3"],
            ["comp3", "meth1", "param2", "reg2", "enc2"],
            ["comp3", "meth3", "param0", "reg2", "enc2"],
        ]
    ):
        _substitute(engine, f"fail{i}", parents, success=0.0)
    _substitute(engine, "close", ["comp3", "meth3", "param2", "reg0", "enc2"], success=0.73)

    # reg2 is convicted, but reg0 is not thereby the answer — the regime question
    # is open again and the navigator has work.
    assert engine._store.get_node("reg2").status == Status.INVALIDATED
    assert engine._store.get_node("reg0").status != Status.VERIFIED
    target = engine.get_next_targets()[0]
    assert target.status == "SELECTED"
    assert target.node_id.startswith("reg")


@pytest.mark.unit
def test_a_settled_search_that_falls_short_reopens_its_answers(
    engine: HypoTreeEngine,
) -> None:
    """Every question answered, the answers assembled, and still short of the goal.

    A composition that is conclusive but sub-par settles without failing, so
    nothing is blamed and nothing is reopened. With every question already
    answered the frontier then empties with the objective unreached — which the
    caller reads as "the search is over". It is the opposite: landing short with
    every answer in hand is positive evidence that one of those answers is wrong.
    """
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal", node_id="goal", is_goal=True, target_metric=0.75, parent_ids=["combo"]
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.6, depth=2))

    assert engine._store.get_node("combo").status == Status.EXHAUSTED
    targets = engine.get_next_targets(count=2)
    assert [t.status for t in targets] == ["SELECTED", "SELECTED"]
    assert {t.node_id for t in targets} <= {"a1", "a2", "b1", "b2"}


@pytest.mark.unit
def test_recovery_reopens_only_what_was_never_probed(engine: HypoTreeEngine) -> None:
    """A sibling settled on its own evidence stays settled, so the recovery ends.

    That bound is what stops it cycling: every reopened node costs at most one
    probe before it is settled for good.
    """
    for i in range(3):
        engine.create_hypothesis(f"a={i}", node_id=f"a{i}", exclusion_group="a")
    engine.create_hypothesis("b=0", node_id="b0", exclusion_group="b")
    _refute(engine, "a1", depth=1)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.6, depth=2))

    # Recovery is a selection-time last resort, so it fires on the dispatch.
    target = engine.get_next_targets()[0]
    assert target.node_id == "a2"  # retired by the exclusion inference, never probed
    assert engine._store.get_node("a1").status == Status.INVALIDATED  # stays refuted


@pytest.mark.unit
def test_a_peek_never_reopens_anything(engine: HypoTreeEngine) -> None:
    """dry_run is a read. Recovery mutates five questions at once, so it waits."""
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal", node_id="goal", is_goal=True, target_metric=0.75, parent_ids=["combo"]
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.6, depth=2))

    before = {n.id: n.status for n in engine._store.get_all_nodes()}
    engine.get_next_targets(dry_run=True)
    assert {n.id: n.status for n in engine._store.get_all_nodes()} == before


@pytest.mark.unit
def test_a_single_parent_composition_has_nothing_to_reopen(engine: HypoTreeEngine) -> None:
    """One assumption is not an interaction, so a sub-par result really is an end."""
    engine.create_hypothesis("a=0", node_id="a0", exclusion_group="a")
    engine.create_hypothesis("a=1", node_id="a1", exclusion_group="a")
    _confirm(engine, "a0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal", node_id="goal", is_goal=True, target_metric=0.75, parent_ids=["combo"]
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.5, depth=2))

    assert engine.get_next_targets()[0].reason == "empty_frontier"


@pytest.mark.unit
def test_an_interaction_reopen_carries_the_interaction_marker(
    engine: HypoTreeEngine,
) -> None:
    """The counterpart to the shortfall test: the interaction path keeps its own marker.

    Nothing pinned the reason this path *writes*, only the reasons it reads, and
    that gap let a parameter named ``marker`` be silently shadowed by a local of
    the same name — rewriting every reopen reason with the exclusion prefix while
    the whole suite stayed green.
    """
    _build_two_axis_landscape(engine)
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _exhaust_substitutions(engine)

    reasons = [
        str(h["reason"] or "")
        for node in engine._store.get_all_nodes()
        for h in engine._store.get_status_history(node.id)
        if str(h["reason"] or "").startswith(INTERACTION_REOPEN_PREFIX)
    ]
    assert reasons, "the interaction ending should have reopened something"
    assert not any(r.startswith(UNDERPERFORMANCE_REOPEN_PREFIX) for r in reasons)


@pytest.mark.unit
def test_a_shortfall_reopen_is_not_labelled_an_interaction_effect(
    engine: HypoTreeEngine,
) -> None:
    """The two recoveries lead to the same place and must not share a marker.

    Both hand retired alternatives back, but one says "a conflict turned out to
    be an interaction effect" and the other says "every answer was in and the
    assembly still missed". Sharing ``INTERACTION_REOPEN_PREFIX`` made a report
    claim every conflict had been narrowed to a culprit *and* that six
    alternatives had been reopened because a conflict was an interaction
    effect — mutually exclusive endings, in the same sentence.
    """
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.6, depth=2))

    engine.get_next_targets()

    reopened = [
        h
        for node_id in ("a1", "a2", "b1", "b2")
        for h in engine._store.get_status_history(node_id)
        if str(h["reason"] or "").startswith(
            (INTERACTION_REOPEN_PREFIX, UNDERPERFORMANCE_REOPEN_PREFIX)
        )
    ]
    assert reopened, "the shortfall recovery should have reopened something"
    assert all(str(h["reason"]).startswith(UNDERPERFORMANCE_REOPEN_PREFIX) for h in reopened), (
        "a shortfall must carry its own marker, not the interaction one"
    )


@pytest.mark.unit
def test_a_composition_that_cleared_the_bar_stands_the_recovery_down(
    engine: HypoTreeEngine,
) -> None:
    """Once some assembly succeeds, landing short earlier is no longer evidence.

    The recovery rests on a shortfall meaning one of the confirmations is wrong.
    A composition that *did* clear the bar withdraws that reading: the answers
    were right and the earlier assembly was simply the wrong combination of
    them. Without this the recovery fires on a search that already succeeded —
    one real episode reopened six settled questions on the turn it found its
    answer, because its goal node had never been wired to the winning node.
    """
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    # The sub-par assembly that would normally trigger the recovery.
    engine.create_hypothesis(
        "weak", node_id="weak", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("weak", LogicalEvidence(success=0.6, depth=2))
    # ...and a second assembly that cleared it.
    engine.create_hypothesis(
        "strong", node_id="strong", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    for _ in range(3):
        engine.record_evidence("strong", LogicalEvidence(success=0.95, depth=2))
    assert engine._store.get_node("strong").status == Status.VERIFIED

    before = {n.id: n.status for n in engine._store.get_all_nodes()}
    engine.get_next_targets()
    assert {n.id: n.status for n in engine._store.get_all_nodes()} == before, (
        "nothing may be reopened once an assembly has cleared the bar"
    )


@pytest.mark.unit
def test_a_goal_wired_to_nothing_is_named_rather_than_reported_as_finished(
    engine: HypoTreeEngine,
) -> None:
    """Goal achievement is derived, so a goal that depends on nothing is inert.

    The failure is silent: the goal sits at UNTESTED while the search around it
    looks healthy, and the run ends looking complete. Every arm-B episode of one
    evaluation run created the objective first and never wired anything to it.
    """
    engine.create_hypothesis("a=0", node_id="a0", exclusion_group="a")
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    _confirm(engine, "a0", depth=1)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.record_evidence("combo", LogicalEvidence(success=0.5, depth=2))

    done = engine.get_next_targets()[0]
    assert done.reason == "unreachable_goal"
    assert "goal" in done.rationale
    assert "DEPENDENCY" in done.rationale


@pytest.mark.unit
def test_a_blank_claim_id_is_treated_as_no_claim(engine: HypoTreeEngine) -> None:
    """A caller reaching for the field it was told to omit has still done the work.

    Rejecting the empty string threw away a probe that was already paid for,
    over punctuation.
    """
    engine.create_hypothesis("a=0", node_id="a0")
    result = engine.record_evidence("a0", LogicalEvidence(success=1.0), claim_id="  ")
    assert result.node.evidence_count == 1


# -- the objective survives a failed attempt at it --------------------------


@pytest.mark.unit
def test_a_failed_combination_does_not_destroy_the_goal(engine: HypoTreeEngine) -> None:
    """A goal is protected everywhere else; the cascade was the one path to it.

    And it reached the goal through the very edge the caller is *told* to create
    — wiring a candidate combination to the objective so the objective can be
    reached. The first combination to fail then pruned the goal, and every real
    episode of one evaluation run spent LLM turns re-creating a goal the engine
    had thrown away.
    """
    engine.create_hypotheses(
        [
            {"statement": "a=0", "node_id": "a0"},
            {"statement": "b=0", "node_id": "b0"},
            {
                "statement": "combo",
                "node_id": "combo",
                "parent_ids": ["a0", "b0"],
                "edge_type": "DEPENDENCY",
            },
            {
                "statement": "goal",
                "node_id": "goal",
                "is_goal": True,
                "target_metric": 0.75,
                "parent_ids": ["combo"],
            },
        ]
    )
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    assert engine._store.get_node("combo").status == Status.INVALIDATED
    assert engine._store.get_node("goal").status != Status.PRUNED


@pytest.mark.unit
def test_a_conviction_does_not_destroy_the_goal(engine: HypoTreeEngine) -> None:
    """The other cascade path reaches the goal exactly the same way."""
    _build_two_axis_landscape(engine)
    engine.create_hypotheses(
        [
            {
                "statement": "goal",
                "node_id": "goal",
                "is_goal": True,
                "target_metric": 0.75,
                "parent_ids": ["combo1"],
            }
        ]
    )
    engine.record_evidence("combo1", LogicalEvidence(success=0.0, depth=2))
    _substitute(engine, "swap", ["comp_v2", "reg_v1"], success=1.0)

    assert engine._store.get_node("comp_v1").status == Status.INVALIDATED
    assert engine._store.get_node("goal").status != Status.PRUNED


@pytest.mark.unit
def test_what_rested_on_the_refutation_is_still_pruned(engine: HypoTreeEngine) -> None:
    """Protecting the objective must not protect the work that assumed a falsehood."""
    engine.create_hypotheses(
        [
            {"statement": "a=0", "node_id": "a0"},
            {
                "statement": "child",
                "node_id": "child",
                "parent_ids": ["a0"],
                "edge_type": "DEPENDENCY",
            },
            {
                "statement": "grandchild",
                "node_id": "grandchild",
                "parent_ids": ["child"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    _refute(engine, "a0", depth=1)

    assert engine._store.get_node("child").status == Status.PRUNED
    assert engine._store.get_node("grandchild").status == Status.PRUNED


# -- a mistyped claim id must not destroy a paid-for probe ------------------


@pytest.mark.unit
def test_an_unknown_claim_id_falls_back_to_the_nodes_own_lease(engine: HypoTreeEngine) -> None:
    """One wrong character cost a real run a probe and stranded its lease."""
    engine.create_hypothesis("a=0", node_id="a0")
    target = engine.get_next_targets()[0]
    typo = target.claim_id[:-1] + ("0" if target.claim_id[-1] != "0" else "1")

    result = engine.record_evidence("a0", LogicalEvidence(success=1.0), claim_id=typo)

    assert result.node.evidence_count == 1
    # The lease is consumed, so the node is not left looking dispatched-and-lost.
    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_an_unknown_claim_id_on_an_unleased_node_is_treated_as_omitted(
    engine: HypoTreeEngine,
) -> None:
    """An id that names no lease reserves nothing, so refusing it only loses work.

    A fabricated placeholder (`meth_v4_claim`) destroyed two real probes in the
    last run. There is no belief state to protect here: the node holds no lease,
    so the id cannot be answering anyone else's dispatch and there is nothing to
    consume. A claim belonging to a *different* node is a separate case and
    still fails loudly.
    """
    engine.create_hypothesis("a=0", node_id="a0")
    engine.record_evidence("a0", LogicalEvidence(success=1.0), claim_id="invented")

    assert engine._store.get_node("a0").evidence_count == 1


@pytest.mark.unit
def test_a_claim_for_another_node_is_still_refused(engine: HypoTreeEngine) -> None:
    """The check that actually protects the belief state must keep failing loudly."""
    from hypotree.engine import ClaimError

    engine.create_hypotheses(
        [{"statement": "a=0", "node_id": "a0"}, {"statement": "b=0", "node_id": "b0"}]
    )
    targets = engine.get_next_targets(count=2)
    other = next(t for t in targets if t.node_id != "a0")

    with pytest.raises(ClaimError, match="different hypothesis"):
        engine.record_evidence("a0", LogicalEvidence(success=1.0), claim_id=other.claim_id)


@pytest.mark.unit
def test_diagnosis_interrogates_the_least_corroborated_question_first(
    engine: HypoTreeEngine,
) -> None:
    """A member whose rivals were never tested is the better first suspect.

    Beating four refuted competitors is a real search; being the first thing
    tried while the exclusion inference quietly retired the rest is not. The
    second rests on nothing, so diagnosis should doubt it first. Measured on the
    held-out seeds this moved the culprit from position 3.75 to 2.38 of five.
    """
    engine.create_hypotheses(
        [
            {"statement": "a=0", "node_id": "a0", "exclusion_group": "a"},
            {"statement": "a=1", "node_id": "a1", "exclusion_group": "a"},
            {"statement": "b=0", "node_id": "b0", "exclusion_group": "b"},
            {"statement": "b=1", "node_id": "b1", "exclusion_group": "b"},
            {
                "statement": "combination",
                "node_id": "combo",
                "parent_ids": ["a0", "b0"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    # 'a' is interrogated: its rival is refuted on its own evidence.
    # 'b' is not: b1 is only retired by the exclusion inference when b0 confirms.
    engine.record_evidence("a1", LogicalEvidence(success=0.0, depth=1))
    engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("b0", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=1))

    members = engine._store.get_nogoods(open_only=True)[0]["member_ids"]
    assert members[0] == "b0", "the question nobody actually searched must be doubted first"


@pytest.mark.unit
def test_diagnosis_falls_back_to_depth_when_corroboration_ties(
    engine: HypoTreeEngine,
) -> None:
    """With both questions equally searched, the shallower confirmation leads."""
    engine.create_hypotheses(
        [
            {"statement": "deep premise", "node_id": "aaa_deep"},
            {"statement": "shallow premise", "node_id": "zzz_shallow"},
            {
                "statement": "combination",
                "node_id": "combo",
                "parent_ids": ["aaa_deep", "zzz_shallow"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    # Alphabetically first, but the better-established of the two.
    engine.record_evidence("aaa_deep", LogicalEvidence(success=1.0, depth=3))
    engine.record_evidence("zzz_shallow", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=3))

    conflict = engine._store.get_nogoods(open_only=True)[0]
    assert conflict["member_ids"][0] == "zzz_shallow"


@pytest.mark.unit
def test_diagnosis_order_does_not_depend_on_what_the_nodes_are_called(
    engine: HypoTreeEngine,
) -> None:
    """Renaming a hypothesis must not make a workspace converge faster.

    Members used to be stored alphabetically, so with ids conventionally
    prefixed by the question they answer, one question was always interrogated
    first and another always last. That made the cost of diagnosis a function of
    the caller's naming convention, which is not a property any inference
    procedure should have.
    """
    engine.create_hypotheses(
        [
            {"statement": "a", "node_id": "aaa"},
            {"statement": "b", "node_id": "bbb"},
            {"statement": "c", "node_id": "ccc"},
            {"statement": "d", "node_id": "ddd"},
            {
                "statement": "combination",
                "node_id": "combo",
                "parent_ids": ["aaa", "bbb", "ccc", "ddd"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    # Every premise equally well established, so nothing but the tiebreak is
    # left to order them by.
    for pid in ("aaa", "bbb", "ccc", "ddd"):
        engine.record_evidence(pid, LogicalEvidence(success=1.0, depth=2))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    members = engine._store.get_nogoods(open_only=True)[0]["member_ids"]
    assert sorted(members) == ["aaa", "bbb", "ccc", "ddd"]
    assert members != sorted(members)


@pytest.mark.unit
def test_the_stored_diagnosis_order_is_stable_across_reads(
    engine: HypoTreeEngine,
) -> None:
    """`probe_index` counts into the stored order, so that order must be frozen.

    If it were recomputed per read, a member gaining evidence mid-diagnosis
    would reshuffle the list under a live cursor and either re-test something
    already cleared or skip a suspect entirely.
    """
    engine.create_hypotheses(
        [
            {"statement": "a", "node_id": "aaa"},
            {"statement": "b", "node_id": "bbb"},
            {
                "statement": "combination",
                "node_id": "combo",
                "parent_ids": ["aaa", "bbb"],
                "edge_type": "DEPENDENCY",
            },
        ]
    )
    engine.record_evidence("aaa", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("bbb", LogicalEvidence(success=1.0, depth=1))
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=1))

    first = engine._store.get_nogoods(open_only=True)[0]["member_ids"]
    engine.record_evidence("aaa", LogicalEvidence(success=1.0, depth=9))
    second = engine._store.get_nogoods(open_only=True)[0]["member_ids"]

    assert first == second


# --------------------------------------------------------------------------
# Goal scoping. The frontier filter and the read model share one definition of
# "what belongs to this goal", so a scoped search and a scoped view cannot
# disagree about it.
# --------------------------------------------------------------------------


def _two_goal_landscape(engine: HypoTreeEngine) -> None:
    """Two independent objectives, each resting on its own question."""
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    engine.create_hypothesis(
        "combo_a", node_id="combo_a", parent_ids=["a0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "combo_b", node_id="combo_b", parent_ids=["b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal_a", node_id="goal_a", is_goal=True, target_metric=0.75, parent_ids=["combo_a"]
    )
    engine.create_hypothesis(
        "goal_b", node_id="goal_b", is_goal=True, target_metric=0.75, parent_ids=["combo_b"]
    )


@pytest.mark.unit
def test_goal_scope_includes_the_alternatives_that_answer_its_questions(
    engine: HypoTreeEngine,
) -> None:
    """Dependency ancestry alone cannot answer the questions the goal rests on.

    ``a1`` is not a dependency-ancestor of ``goal_a`` — it is a sibling of one.
    Testing it is nevertheless how the engine learns whether ``a0`` holds, so a
    scope built from ancestry alone would hand the navigator a question it is
    forbidden from answering.
    """
    _two_goal_landscape(engine)
    engine._sync_graph_from_store()
    scope = engine._goal_scope("goal_a")

    assert {"goal_a", "combo_a", "a0", "a1", "a2"} <= scope
    assert not scope & {"goal_b", "combo_b", "b0", "b1", "b2"}


@pytest.mark.unit
def test_a_goal_filter_only_hands_out_work_inside_that_goal(engine: HypoTreeEngine) -> None:
    """The whole point: two objectives, one at a time."""
    _two_goal_landscape(engine)
    for _ in range(6):
        targets = engine.get_next_targets(goal_id="goal_a")
        if targets[0].node_id is None:
            break
        assert targets[0].node_id.startswith("a"), targets[0].node_id
        engine.record_evidence(targets[0].node_id, LogicalEvidence(success=0.0, depth=1))


@pytest.mark.unit
def test_no_goal_id_selects_exactly_what_it_always_did(tmp_path: Path) -> None:
    """The default path must not move: same seed, same state, same pick.

    Two engines rather than two calls on one — the sampler draws from a seeded
    RNG that every call advances, so consecutive dry runs differ by design and
    comparing them would test the RNG rather than the filter.
    """
    picks = []
    for pass_explicit_none in (False, True):
        e = HypoTreeEngine(tmp_path / f"scope-{pass_explicit_none}.db", rng_seed=7)
        try:
            _two_goal_landscape(e)
            targets = (
                e.get_next_targets(goal_id=None, dry_run=True)
                if pass_explicit_none
                else e.get_next_targets(dry_run=True)
            )
            picks.append(targets[0].node_id)
        finally:
            e.close()
    assert picks[0] == picks[1]


@pytest.mark.unit
def test_a_goal_filter_that_hides_the_work_says_so(engine: HypoTreeEngine) -> None:
    """The step that stops goal scoping from being a bug.

    Agents create premises before wiring them — ``awaiting_composition`` exists
    because they do — and those unwired nodes are outside every goal's scope by
    construction. Reporting a bare empty frontier here would announce the search
    was over while a dozen untested hypotheses sat one missing edge away.
    """
    engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.75)
    for i in range(3):
        engine.create_hypothesis(f"loose={i}", node_id=f"loose{i}", exclusion_group="loose")

    done = engine.get_next_targets(goal_id="goal")[0]
    assert done.status == "DONE"
    assert done.reason == "goal_scope_empty"
    assert "3 untested" in done.rationale
    assert "DEPENDENCY" in done.rationale


@pytest.mark.unit
def test_scoping_to_something_that_is_not_a_goal_is_refused(engine: HypoTreeEngine) -> None:
    """Silently scoping to an ordinary node yields a plausible, meaningless filter."""
    _two_goal_landscape(engine)
    with pytest.raises(ClaimError, match="is not a goal"):
        engine.get_next_targets(goal_id="a0")
    with pytest.raises(NodeNotFoundError):
        engine.get_next_targets(goal_id="ghost")


@pytest.mark.unit
def test_the_learning_path_and_goal_status_scope_to_the_same_set(
    engine: HypoTreeEngine,
) -> None:
    """One primitive, three callers — they must not drift apart."""
    _two_goal_landscape(engine)
    _confirm(engine, "a0", depth=1)
    _confirm(engine, "b0", depth=1)

    path = engine.generate_learning_path(goal_id="goal_a")
    assert all(not s.node_id.startswith("b") for s in path.steps)

    status = engine.get_goal_status(goal_id="goal_a")
    assert [g.node_id for g in status.goals] == ["goal_a"]
    assert status.total_nodes == len(engine._goal_scope("goal_a"))


@pytest.mark.unit
def test_a_conflict_nobody_can_swap_does_not_withhold_its_members(tmp_path: Path) -> None:
    """Diagnosis by substitution needs an alternative to substitute.

    A member with no exclusion group can never be cleared, so the conflict never
    left the diagnosing set and its members were held off the frontier forever.
    The caller was handed a bare `empty_frontier` — no rationale at all — while
    two untested hypotheses sat in the store, which is the exact failure the DONE
    taxonomy exists to prevent.
    """
    engine = HypoTreeEngine(tmp_path / "deadlock.db", rng_seed=7)
    try:
        engine.create_hypothesis("A", node_id="A")
        engine.create_hypothesis("B", node_id="B")
        engine.create_hypothesis(
            "AB", node_id="AB", parent_ids=["A", "B"], edge_type=EdgeType.DEPENDENCY
        )
        engine.record_evidence("A", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("B", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("AB", LogicalEvidence(success=0.0, depth=2))

        assert engine._store.get_nogoods(open_only=True), "the conflict is still recorded"
        assert {n.id for n in engine._frontier_nodes()} == {"A", "B"}
        assert engine.get_next_targets(count=1)[0].status == "SELECTED"
    finally:
        engine.close()


@pytest.mark.unit
def test_a_pruned_member_is_not_convicted_of_causing_the_conflict(tmp_path: Path) -> None:
    """PRUNED says an ancestor was refuted, not that this assumption is at fault.

    Convicting on it closed the conflict on collateral damage, released every
    other member, and then spent the review budget prioritising the alternatives
    of a question that was never implicated.
    """
    engine = HypoTreeEngine(tmp_path / "pruned.db", rng_seed=7)
    try:
        engine.create_hypothesis("anc", node_id="anc")
        engine.create_hypothesis(
            "A", node_id="A", parent_ids=["anc"], edge_type=EdgeType.REFINEMENT
        )
        for nid in ("B", "C"):
            engine.create_hypothesis(nid, node_id=nid)
        engine.create_hypothesis(
            "combo", node_id="combo", parent_ids=["A", "B", "C"], edge_type=EdgeType.DEPENDENCY
        )
        for nid in ("anc", "A", "B", "C"):
            engine.record_evidence(nid, LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

        engine.record_evidence("anc", LogicalEvidence(success=0.0, depth=1))

        assert engine._store.get_node("A").status is Status.PRUNED
        open_now = engine._store.get_nogoods(open_only=True)
        assert open_now, "collateral damage must not close the conflict"
        assert open_now[0]["resolved_at"] is None
    finally:
        engine.close()


@pytest.mark.unit
def test_recording_without_a_claim_id_consumes_the_nodes_own_lease(tmp_path: Path) -> None:
    """`claim_id` is optional everywhere, so omitting it is the ordinary path.

    A *wrong* id was recovered from the node's own live lease and consumed; an
    absent one was not, which is backwards. The lease stayed live for its whole
    TTL, holding a settled node off the frontier and reporting it as
    dispatched-and-never-reported.
    """
    engine = HypoTreeEngine(tmp_path / "lease.db", rng_seed=7)
    try:
        engine.create_hypothesis("L", node_id="L")
        engine.get_next_targets(count=1)
        engine.record_evidence("L", LogicalEvidence(success=1.0, depth=1))

        assert engine._store.get_active_claims(utcnow()) == []
        assert engine._store.get_node("L").active_claim_id is None
    finally:
        engine.close()


@pytest.mark.unit
def test_a_live_lease_no_longer_masks_the_diagnosis_that_names_the_next_move(
    tmp_path: Path,
) -> None:
    """`awaiting_evidence` was checked before every substantive diagnosis.

    One leaked lease therefore told the caller to record evidence it had already
    recorded, while an unbuilt composition, a conflict, a dead question or a
    wiring error went unreported — and recording could not help, because there
    was nothing left to record.
    """
    engine = HypoTreeEngine(tmp_path / "mask.db", rng_seed=7)
    try:
        engine.create_hypothesis("goal", node_id="goal", is_goal=True, target_metric=0.7)
        for i in range(2):
            engine.create_hypothesis(f"a{i}", node_id=f"a{i}", exclusion_group="a")
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))

        # Strand a lease on a node the exclusion inference then settles.
        engine.create_hypothesis("spare", node_id="spare")
        engine.get_next_targets(count=1)
        engine._store.change_status(
            "spare", Status.EXHAUSTED, reason="settled elsewhere", now=utcnow()
        )

        result = engine.get_next_targets(count=1)[0]
        assert result.reason == "awaiting_composition", result.reason
        assert "a0" in (result.rationale or "")
    finally:
        engine.close()
