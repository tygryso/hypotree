"""Tests for the Hypotree Capability Index (HCI) computation in seed_reader."""

from __future__ import annotations

import pytest

from eval.seed_reader import RunLog, _section_hci


def _make_log(
    seed: int = 1201,
    arm: str = "B",
    steps: int = 15,
    goals_met: bool = True,
    complete: bool = True,
    experiments: int = 15,
    duplicates: int = 0,
    tool_errors: dict[str, int] | None = None,
    redispatched: int = 0,
    settled_reselects: int = 0,
    pruned_reexecutions: int = 0,
    goal_contamination: int = 0,
    same_question_dispatches: int = 0,
) -> RunLog:
    """Build a minimal RunLog for HCI testing."""
    from pathlib import Path

    log = RunLog(seed=seed, arm=arm, path=Path("dummy"), events=[])
    log.steps = steps
    log.goals_met = goals_met
    log.complete = complete
    log.experiments = experiments
    log.duplicates = duplicates
    log.redispatched = redispatched
    log.settled_reselects = settled_reselects
    log.pruned_reexecutions = pruned_reexecutions
    log.goal_contamination = goal_contamination
    log.same_question_dispatches = same_question_dispatches
    if tool_errors:
        from collections import Counter

        log.tool_errors = Counter(tool_errors)
    return log


def _parse_hci_from_section(section: list[str]) -> float:
    """Extract the HCI score from the rendered markdown table."""
    text = "\n".join(section)
    # Find the data row (skip header + separator). The arm is in col 1, HCI in col 2.
    for line in text.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3 and parts[1].strip() in ("A", "B", "F"):
            return float(parts[2].strip())
    raise ValueError(f"HCI not found in section: {text}")


@pytest.mark.unit
def test_perfect_run_scores_near_one():
    """A perfect run (0 waste, 0 duplicates, 0 errors, 0 violations) should approach 1.0."""
    log = _make_log(seed=1201, steps=15, goals_met=True)
    hci = _parse_hci_from_section(_section_hci([log]))
    assert hci > 0.5


@pytest.mark.unit
def test_zero_goals_met_zeroes_score():
    """If no goals are met, E_base's completion rate is 0, so HCI is 0."""
    log = _make_log(seed=1201, steps=50, goals_met=False)
    hci = _parse_hci_from_section(_section_hci([log]))
    assert hci == pytest.approx(0.0, abs=0.001)


@pytest.mark.unit
def test_duplicates_reduce_p_mem():
    """Duplicates should reduce the P_mem factor below 1.0."""
    clean = _make_log(seed=1201, steps=15, duplicates=0, experiments=15)
    dirty = _make_log(seed=1201, steps=15, duplicates=5, experiments=15)
    clean_hci = _parse_hci_from_section(_section_hci([clean]))
    dirty_hci = _parse_hci_from_section(_section_hci([dirty]))
    assert dirty_hci < clean_hci


@pytest.mark.unit
def test_tool_errors_reduce_p_tools():
    """Tool errors should reduce the P_tools factor below 1.0."""
    clean = _make_log(seed=1201, steps=15, tool_errors={})
    error_log = _make_log(seed=1201, steps=15, tool_errors={"create_hypotheses": 2})
    clean_hci = _parse_hci_from_section(_section_hci([clean]))
    error_hci = _parse_hci_from_section(_section_hci([error_log]))
    assert error_hci < clean_hci


@pytest.mark.unit
def test_health_violations_reduce_p_health():
    """Engine rule violations should reduce the P_health factor (arm B only)."""
    clean = _make_log(seed=1201, steps=15, arm="B", redispatched=0)
    violator = _make_log(seed=1201, steps=15, arm="B", redispatched=2)
    clean_hci = _parse_hci_from_section(_section_hci([clean]))
    violator_hci = _parse_hci_from_section(_section_hci([violator]))
    assert violator_hci < clean_hci


@pytest.mark.unit
def test_baselines_have_p_health_one():
    """Arms A and F should have P_health = 1.0 (no belief state to violate)."""
    log_f = _make_log(seed=1201, arm="F", steps=27)
    section = _section_hci([log_f])
    text = "\n".join(section)
    assert "1.0 (n/a)" in text


@pytest.mark.unit
def test_hci_in_report():
    """The HCI section should appear in a rendered report."""
    import tempfile
    from pathlib import Path

    from eval.seed_reader import render_report

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "test-run"
        runs_dir.mkdir()
        log_path = runs_dir / "seed-1201-arm-B.jsonl"
        log_path.write_text(
            '{"event_type": "run_start", "step": 0, "seed": 1201, "arm": "B", '
            '"run_id": "test-run", "tool_budget": 100, "llm_model": "test"}\n'
            '{"event_type": "experiment", "step": 1, "seed": 1201, "arm": "B", '
            '"config": "component=v0", "depth": 1, "success": 0.0, '
            '"probe_mode": "premise", "duplicate": false, "distinct_configs": 1}\n'
            '{"event_type": "run_end", "step": 1, "seed": 1201, "arm": "B", '
            '"reason": "all_goals_met", "goals_met": true}\n',
            encoding="utf-8",
        )
        report = render_report("test-run", runs_dir)
        assert "## 11. Hypotree Capability Index (HCI)" in report
