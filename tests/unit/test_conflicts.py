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

from hypotree.engine import MAX_REVIEW_DISPATCHES, HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status


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
    assert nogood["resolved_at"] is None
    assert engine._store.get_node("comp_v1").status == Status.NEEDS_REVISION
    assert engine._store.get_node("comp_v2").status == Status.EXHAUSTED
    # ...and the engine now asks about the *other* assumption.
    assert "reg_v1" in engine.get_next_targets()[0].rationale


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
