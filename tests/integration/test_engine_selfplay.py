"""Tests for the engine self-play pre-flight.

The check exists because a regression in the substitution verdict made the engine
declare real episodes finished with the goal unmet, and every unit test passed
throughout: each covered one mechanism in isolation, and the failure was in how
two of them composed. These tests pin the *check itself*, since a pre-flight that
cannot fail is worse than none — it converts an unrun test into a green light.
"""

from __future__ import annotations

import pytest

from eval.runner.config import TASK_SEEDS
from eval.runner.engine_selfplay import (
    MAX_PROBE_RATIO,
    SelfPlayResult,
    check,
    main,
    solve_seed,
)


@pytest.mark.integration
def test_the_engine_solves_a_seed_without_an_llm() -> None:
    """A perfectly-disciplined caller must reach the goal on the engine alone.

    This is the property the whole harness rests on: if the engine cannot solve
    the landscape when the caller never forgets and never improvises, no model
    can be expected to, and every step count measured against it is noise.
    """
    result = solve_seed(TASK_SEEDS[0])
    assert result.solved, f"{result.end_reason} after {result.probes} probes"
    assert result.probes <= MAX_PROBE_RATIO * result.reference


@pytest.mark.integration
def test_the_engine_does_not_need_more_probes_than_perfect_recall() -> None:
    """The belief state should cost no more than remembering everything.

    Measured on a spread of seeds rather than one, because a single landscape can
    be lucky in either direction.
    """
    results = [solve_seed(seed) for seed in TASK_SEEDS[:5]]
    assert all(r.solved for r in results)
    mean_ratio = sum(r.ratio for r in results) / len(results)
    assert mean_ratio <= MAX_PROBE_RATIO


@pytest.mark.unit
def test_an_unsolved_seed_is_a_failure() -> None:
    """`empty_frontier` with the goal unmet is the defect this check is for."""
    problems = check([SelfPlayResult(1201, 23, 20, False, "empty_frontier")])
    assert len(problems) == 1
    assert "empty_frontier" in problems[0]
    assert "gave up" in problems[0]


@pytest.mark.unit
def test_a_solved_run_that_got_much_slower_is_a_failure() -> None:
    """Not every regression stops the search — some just make it wasteful."""
    slow = [SelfPlayResult(s, 100, 20, True, "solved") for s in (1201, 1202)]
    problems = check(slow)
    assert len(problems) == 1
    assert "regressed" in problems[0]


@pytest.mark.unit
def test_a_healthy_run_reports_nothing() -> None:
    assert check([SelfPlayResult(1201, 18, 20, True, "solved")]) == []


@pytest.mark.integration
def test_the_cli_exits_zero_on_a_healthy_engine(capsys: pytest.CaptureFixture[str]) -> None:
    """eval.sh gates the whole run on this exit code."""
    assert main(["--seeds", str(TASK_SEEDS[0])]) == 0
    assert "Engine self-play OK" in capsys.readouterr().out
