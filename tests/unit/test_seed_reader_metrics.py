"""Tests for the report reader's derived metrics.

These numbers are how a run gets read, so an artefact here is indistinguishable
from a result. Each test below pins a case where the previous arithmetic said
something the logs did not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.seed_reader import RunLog, _build_run_log, _exclusion_yield_lines


def _log(seed: int = 1201, arm: str = "B") -> RunLog:
    log = RunLog(seed=seed, arm=arm, path=Path("dummy"), events=[])
    log.complete = True
    return log


@pytest.mark.unit
def test_a_group_is_sized_by_its_members_not_by_its_creation_events() -> None:
    """The arm-B prompt tells the agent to re-create nodes with `if_exists="overwrite"`.

    Every re-creation logs `created` again, so counting events inflated k — and k
    sets the baseline the run's ordering is judged against, so an agent that
    rewrote its own nodes made itself look better by doing nothing.
    """
    log = _log()
    log.group_members["colour"].update({"c0", "c1", "c2"})
    # The same three nodes declared a second time.
    log.group_members["colour"].update({"c0", "c1", "c2"})
    log.exclusions_applied = 4
    log.premise_probes = 6

    lines = _exclusion_yield_lines([log])
    assert lines and "mean group of 3.0 answers" in lines[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exclusions", "premise", "expected"),
    [
        (9, 1, "better than chance"),
        (1, 9, "worse than probing"),
        (4, 6, "at the baseline"),
    ],
)
def test_the_yield_verdict_can_report_all_three_outcomes(
    exclusions: int, premise: int, expected: str
) -> None:
    """The old form compared `> baseline + 0.02` against `< baseline + 0.02`.

    That left "at the baseline" reachable only on exact float equality, and
    described a run performing *below* chance in the same words as one performing
    at it. With five answers per question the baseline is 40%.
    """
    log = _log()
    log.group_members["colour"].update({f"c{i}" for i in range(5)})
    log.exclusions_applied = exclusions
    log.premise_probes = premise

    lines = _exclusion_yield_lines([log])
    assert lines and expected in lines[0]


@pytest.mark.unit
def test_a_second_conflict_no_longer_steals_the_first_ones_diagnosis() -> None:
    """Attribution used to live in one slot, so two open conflicts lost one each.

    The first resolution was measured from the *second* conflict's mark, and the
    second resolution was attributed to nothing at all.
    """
    events = [
        {"event_type": "conflict_recorded", "nogood_id": 1, "n_members": 2},
        {"event_type": "experiment", "config": "x", "depth": 1, "success": 0.0},
        {"event_type": "conflict_recorded", "nogood_id": 2, "n_members": 2},
        {"event_type": "experiment", "config": "y", "depth": 1, "success": 0.0},
        {"event_type": "conflict_resolved", "nogood_id": 1, "culprit_id": "a"},
        {"event_type": "conflict_resolved", "nogood_id": 2, "culprit_id": "b"},
    ]
    log = _build_run_log(Path("dummy"), events)

    assert log.conflicts_resolved == 2
    assert sorted(log.diagnosis_swaps) == [1, 2]


@pytest.mark.unit
def test_a_log_written_before_conflict_ids_still_reads() -> None:
    """Runs J and K predate the id, and a reader that drops their diagnosis cost
    silently rewrites history rather than reporting it."""
    events = [
        {"event_type": "conflict_recorded", "n_members": 2},
        {"event_type": "experiment", "config": "x", "depth": 1, "success": 0.0},
        {"event_type": "experiment", "config": "y", "depth": 1, "success": 0.0},
        {"event_type": "conflict_resolved", "nogood_id": 7, "culprit_id": "a"},
    ]
    log = _build_run_log(Path("dummy"), events)

    assert log.diagnosis_swaps == [2]
