"""Tests for run-id isolation and the markdown run reader.

Two runs of the same seed and arm must not be able to see each other's logs or
belief state; a report must never be assembled from more than one run. Both
properties are cheap to break by accident and expensive to notice afterwards, so
they are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.runner.config import (
    ARM_A,
    ARM_B,
    ARM_F,
    DEFAULT_WORKSPACE_ID,
    TASK_SEEDS,
    make_ablation_config,
    make_run_config,
    reset_eval_db,
    resolve_eval_db_path,
    run_dir,
    run_workspace_id,
)
from eval.seed_reader import _sign_test_p, load_run, pair_arms, render_report
from hypotree.engine import (
    DEDUCTION_REASON_PREFIX,
    EXCLUSION_REASON_PREFIX,
    INTERACTION_REOPEN_PREFIX,
    HypoTreeEngine,
)
from hypotree.store.identity import _validate_name

# -- run-id isolation ----------------------------------------------------------


@pytest.mark.unit
def test_workspace_is_namespaced_by_run_id() -> None:
    assert run_workspace_id("2026-07-27a") == f"2026-07-27a@{DEFAULT_WORKSPACE_ID}"


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "  ", "../escape", "a/b", "x" * 129, "with space", "a$b"])
def test_invalid_run_ids_are_rejected(bad: str) -> None:
    """A run id is a path component and a workspace key — never sanitised silently."""
    with pytest.raises(ValueError):
        run_workspace_id(bad)


@pytest.mark.unit
def test_generated_run_ids_survive_the_round_trip() -> None:
    """The auto-generated shape must be a legal run id *and* a legal workspace name.

    Two independent patterns have to accept it — the run-id regex and the
    workspace-name one — and they have drifted apart before, which fails only at
    the moment a real run starts.
    """
    run_id = "v0.3.0_run-iteration~a_llm-model~qwen3.6-27b-q8_0_branch~main"
    assert run_workspace_id(run_id) == f"{run_id}@{DEFAULT_WORKSPACE_ID}"
    assert _validate_name(run_id) == run_id


@pytest.mark.unit
def test_run_id_is_trimmed_not_mangled() -> None:
    assert run_dir(Path("eval"), " tidy ") == Path("eval/runs/tidy")


@pytest.mark.unit
def test_two_runs_never_share_a_log_path(tmp_path: Path) -> None:
    seed = TASK_SEEDS[0]
    first = make_run_config(ARM_B, seed, Path("eval"), "run-one")
    second = make_run_config(ARM_B, seed, Path("eval"), "run-two")
    assert first.log_path != second.log_path
    assert first.log_path.parent.name == "run-one"
    assert second.log_path.parent.name == "run-two"


@pytest.mark.unit
def test_two_runs_never_share_a_belief_state_db(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    seed = TASK_SEEDS[0]
    first = make_run_config(ARM_B, seed, Path("eval"), "run-one")
    second = make_run_config(ARM_B, seed, Path("eval"), "run-two")
    assert first.workspace_id != second.workspace_id
    assert resolve_eval_db_path(first.workspace_id, "state") != resolve_eval_db_path(
        second.workspace_id, "state"
    )


@pytest.mark.unit
def test_landscape_is_shared_across_runs() -> None:
    """The task must be identical between runs, or the runs are incomparable."""
    seed = TASK_SEEDS[0]
    first = make_run_config(ARM_B, seed, Path("eval"), "run-one")
    second = make_run_config(ARM_B, seed, Path("eval"), "run-two")
    assert first.landscape_path == second.landscape_path
    assert first.briefing_path == second.briefing_path


@pytest.mark.unit
def test_ablation_logs_are_run_scoped() -> None:
    config = make_ablation_config(TASK_SEEDS[0], 2001, Path("eval"), "run-one")
    assert config.run_id == "run-one"
    assert config.log_path.parent == Path("eval/runs/run-one")


# -- reader --------------------------------------------------------------------


def _write_log(
    runs_dir: Path,
    seed: int,
    arm: str,
    *,
    steps: int,
    goals_met: bool = True,
    events: list[dict] | None = None,
) -> Path:
    """Write a minimal but structurally valid episode log."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"seed-{seed}-arm-{arm}.jsonl"
    rows: list[dict] = [
        {"event_type": "run_start", "seed": seed, "arm": arm, "step": 0, "tool_budget": 60}
    ]
    rows += events or []
    rows.append(
        {
            "event_type": "run_end",
            "seed": seed,
            "arm": arm,
            "step": steps,
            "goals_met": goals_met,
            "reason": "all_goals_met" if goals_met else "budget_exhausted",
        }
    )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.unit
def test_reader_reads_only_its_own_run(tmp_path: Path) -> None:
    _write_log(tmp_path / "runs" / "mine", 1201, ARM_B, steps=11)
    _write_log(tmp_path / "runs" / "theirs", 1201, ARM_B, steps=99)

    logs, _ = load_run(run_dir(tmp_path, "mine"))
    assert [log.steps for log in logs] == [11]


@pytest.mark.unit
def test_reader_survives_a_truncated_line(tmp_path: Path) -> None:
    """A run killed mid-write must not take the whole report down."""
    runs = tmp_path / "runs" / "r"
    path = _write_log(runs, 1201, ARM_B, steps=11)
    path.write_text(path.read_text(encoding="utf-8") + '{"event_type": "experi')

    logs, _ = load_run(runs)
    assert logs[0].steps == 11


@pytest.mark.unit
def test_reader_counts_belief_state_mechanics(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=3,
        events=[
            {"event_type": "node_created", "created": True, "exclusion_group": "component"},
            {"event_type": "node_created", "created": True, "exclusion_group": None},
            {
                "event_type": "node_created",
                "created": True,
                "exclusion_group": None,
                "composed": True,
            },
            {"event_type": "experiment", "config": "component=v1", "success": 1.0, "depth": 2},
            {
                "event_type": "experiment",
                "config": "component=v1",
                "success": 1.0,
                "duplicate": True,
            },
            {
                "event_type": "status_transition",
                "old_status": "VERIFIED",
                "new_status": "NEEDS_REVISION",
                "propagated": True,
            },
            {
                "event_type": "status_transition",
                "old_status": "UNTESTED",
                "new_status": "EXHAUSTED",
                "propagated": True,
                "reason": f"{EXCLUSION_REASON_PREFIX}comp_v1",
            },
            {
                "event_type": "status_transition",
                "old_status": "EXHAUSTED",
                "new_status": "UNTESTED",
                "reason": f"{INTERACTION_REOPEN_PREFIX}'comp_v1' holds in isolation",
            },
            {
                "event_type": "status_transition",
                "old_status": "UNTESTED",
                "new_status": "VERIFIED",
                "reason": f"{DEDUCTION_REASON_PREFIX}every other member is ruled out",
            },
            {"event_type": "evidence_recorded", "node_id": "goal", "success": 0.0},
            {"event_type": "goal_evidence_refused", "node_id": "goal"},
            {"event_type": "tool_call", "tool": "record_evidence", "ok": False},
            {
                "event_type": "conflict_recorded",
                "source_node_id": "c1",
                "member_ids": ["a", "b"],
                "n_members": 2,
            },
            {"event_type": "conflict_resolved", "nogood_id": 1, "culprit_id": "b"},
            {"event_type": "target_selected", "node_id": "a", "status": "VERIFIED"},
            {"event_type": "llm_call", "duration_s": 2.5, "n_tool_calls": 3, "prompt_tokens": 100},
        ],
    )

    log = load_run(runs)[0][0]
    assert log.nodes_created == 3
    assert log.nodes_with_group == 1
    # The composition cannot declare a group, so it is out of the denominator:
    # 1 of the 2 groupable nodes did, not 1 of 3.
    assert log.composed_nodes == 1
    assert log.group_adoption == pytest.approx(1 / 2)
    assert log.duplicates == 1
    assert log.revision_transitions == 1 and log.revision_fired is True
    assert log.exclusions_applied == 1
    assert log.reopened == 1
    assert log.interaction_reopens == 1
    assert log.deduced == 1
    assert log.goal_contamination == 1
    assert log.tool_errors["record_evidence"] == 1
    assert log.probe_depths[2] == 1
    assert log.conflicts_recorded == 1 and log.conflicts_resolved == 1
    assert log.settled_reselects == 1
    assert log.llm_calls == 1 and log.llm_seconds == pytest.approx(2.5)
    assert log.prompt_tokens == 100


@pytest.mark.unit
def test_pairing_drops_incomplete_episodes(tmp_path: Path) -> None:
    """A crashed episode has no defensible step count, so it cannot be paired."""
    runs = tmp_path / "runs" / "r"
    _write_log(runs, 1201, ARM_B, steps=10)
    _write_log(runs, 1201, ARM_F, steps=18)
    _write_log(runs, 1202, ARM_B, steps=12)
    crashed = runs / f"seed-1202-arm-{ARM_F}.jsonl"
    crashed.write_text(
        json.dumps({"event_type": "run_start", "seed": 1202, "arm": ARM_F}) + "\n", encoding="utf-8"
    )

    logs, _ = load_run(runs)
    pair = pair_arms(logs, ARM_B, ARM_F)
    assert pair.seeds == [1201]
    assert pair.diffs == [8]
    assert pair.wins == 1


@pytest.mark.unit
def test_sign_test_matches_known_values() -> None:
    assert _sign_test_p(0, 0) is None
    assert _sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert _sign_test_p(3, 3) == pytest.approx(1.0)


@pytest.mark.unit
def test_report_renders_every_section(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "r"
    for seed, (b, f, a) in {1201: (10, 18, 30), 1202: (12, 17, 28)}.items():
        _write_log(runs, seed, ARM_B, steps=b)
        _write_log(runs, seed, ARM_F, steps=f)
        _write_log(runs, seed, ARM_A, steps=a)
    (runs / "ablation-seed-1201-rng-2001.jsonl").write_text(
        json.dumps({"event_type": "ablation_result", "strategy": "ts", "cumulative_regret": 4.0})
        + "\n",
        encoding="utf-8",
    )

    report = render_report("r", runs)
    for heading in (
        "## 1. Headline",
        "## 2. Paired comparisons",
        "## 3. Per-episode detail",
        "## 4. Probe economy",
        "## 5. Belief-state mechanics",
        "## 6. Stratified by conflict",
        "## 7. Memory maintenance",
        "## 8. Cost per arm",
        "## 9. Navigator ablation",
        "## 10. Action taxonomy",
        "## 11. Hypotree Capability Index",
        "## 12. Data-quality warnings",
    ):
        assert heading in report
    assert "r@hypotree-eval" in report


@pytest.mark.unit
def test_report_reports_an_empty_run(tmp_path: Path) -> None:
    report = render_report("nothing", tmp_path)
    assert "No `seed-*-arm-*.jsonl` logs found" in report


@pytest.mark.unit
def test_report_flags_a_foreign_log(tmp_path: Path) -> None:
    """A log from another run in this directory invalidates every aggregate."""
    runs = tmp_path / "runs" / "mine"
    runs.mkdir(parents=True)
    path = runs / f"seed-1201-arm-{ARM_B}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"event_type": "run_start", "seed": 1201, "arm": ARM_B, "run_id": "theirs"},
                {
                    "event_type": "run_end",
                    "seed": 1201,
                    "arm": ARM_B,
                    "step": 5,
                    "goals_met": True,
                    "reason": "all_goals_met",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = render_report("mine", runs)
    assert "declare a different run id" in report


@pytest.mark.unit
def test_reader_separates_lease_violations_from_resampling(tmp_path: Path) -> None:
    """Re-dispatching a reported node is normal; re-dispatching a live one is a bug.

    A stochastic node is deliberately sampled many times, so counting every
    repeat dispatch would flag correct behaviour and bury the real signal.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=4,
        events=[
            # Legitimate: dispatched, reported, dispatched again.
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
            {"event_type": "evidence_recorded", "node_id": "a", "success": 0.5},
            {"event_type": "target_selected", "node_id": "a", "status": "IN_PROGRESS"},
            # Violation: dispatched twice with nothing reported in between.
            {"event_type": "target_selected", "node_id": "b", "status": "UNTESTED"},
            {"event_type": "target_selected", "node_id": "b", "status": "IN_PROGRESS"},
        ],
    )

    log = load_run(runs)[0][0]
    assert log.redispatched == 1
    assert log.targets_selected == 4
    assert log.evidence_records == 1


@pytest.mark.unit
def test_reader_counts_released_leases(tmp_path: Path) -> None:
    """A release clears the outstanding set, so what follows is not a violation."""
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=2,
        events=[
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
            {"event_type": "claims_released", "node_ids": ["a"], "count": 1},
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
        ],
    )

    log = load_run(runs)[0][0]
    assert log.claims_released == 1
    assert log.redispatched == 0


@pytest.mark.unit
def test_a_failed_episode_is_scored_at_the_budget(tmp_path: Path) -> None:
    """An episode that gave up early did not solve the task quickly.

    The frozen gate right-censors an unmet goal to the tool budget. This report
    did not, and the two then disagreed about the same run: a crashed episode
    that probed *nothing* was scored here as a 20-step win over the baseline
    while the gate scored it as a 40-step loss. The report was the optimistic
    one, which is the worst direction for the error to run.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(runs, 1201, ARM_B, steps=0, goals_met=False)
    _write_log(runs, 1201, ARM_F, steps=20)

    logs, _ = load_run(run_dir(tmp_path, "r"))
    b = next(log for log in logs if log.arm == ARM_B)

    # Raw steps stay available for diagnosis; scoring uses the censored value.
    assert b.steps == 0
    assert b.scored_steps == 60

    pair = pair_arms(logs, ARM_B, ARM_F)
    assert pair.diffs == [-40]
    assert pair.wins == 0


@pytest.mark.unit
def test_a_completed_episode_is_scored_on_what_it_spent(tmp_path: Path) -> None:
    """Censoring must not touch an episode that actually reached the goal."""
    runs = tmp_path / "runs" / "r"
    _write_log(runs, 1201, ARM_B, steps=12)
    _write_log(runs, 1201, ARM_F, steps=20)

    logs, _ = load_run(run_dir(tmp_path, "r"))
    assert pair_arms(logs, ARM_B, ARM_F).diffs == [8]


@pytest.mark.unit
def test_reset_eval_db_clears_a_previous_attempt(tmp_path: Path) -> None:
    """An arm's database must not survive into a re-run of that arm.

    The file is keyed by (run, seed, arm) and outlives the process, so a crashed
    arm that is run again would resume from its own earlier nodes and score an
    agent that started out already knowing the answer.
    """
    db = tmp_path / "seed-1201-arm-B.db"
    engine = HypoTreeEngine(db, rng_seed=1)
    engine.create_hypothesis("carried over", node_id="stale")
    engine.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{db}{suffix}").touch()

    reset_eval_db(db, run_workspace_id("2026-07-30a"))

    assert not db.exists()
    assert not Path(f"{db}-wal").exists()
    assert not Path(f"{db}-shm").exists()
    engine = HypoTreeEngine(db, rng_seed=1)
    assert engine._store.get_node("stale") is None
    engine.close()


@pytest.mark.unit
def test_reset_eval_db_refuses_a_non_eval_workspace(tmp_path: Path) -> None:
    """The live dogfooding belief state is never collateral of an eval run."""
    db = tmp_path / "state.db"
    db.touch()

    with pytest.raises(ValueError, match="not an eval workspace"):
        reset_eval_db(db, "hypotree-dev")

    assert db.exists()


@pytest.mark.unit
def test_only_claimed_records_answer_a_dispatch(tmp_path: Path) -> None:
    """A composition the agent assembles itself was never handed to it.

    Counting every record against every dispatch made "dispatches never
    reported" go *negative* on a run where the agent did exactly what the engine
    asked and composed the answers itself — a metric that reported honest
    progress as work paid for and lost.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=3,
        events=[
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
            {"event_type": "evidence_recorded", "node_id": "a", "claimed": True},
            # Self-composed: no dispatch behind it.
            {"event_type": "evidence_recorded", "node_id": "combo", "claimed": False},
            {"event_type": "evidence_recorded", "node_id": "combo2", "claimed": False},
        ],
    )

    log = load_run(runs)[0][0]
    assert log.targets_selected == 1
    assert log.evidence_records == 3
    assert log.claimed_records == 1


@pytest.mark.unit
def test_the_report_never_claims_a_negative_shortfall(tmp_path: Path) -> None:
    """More records than dispatches is normal, not a deficit of minus seven percent."""
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=3,
        events=[
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
            {"event_type": "evidence_recorded", "node_id": "a", "claimed": True},
            {"event_type": "evidence_recorded", "node_id": "combo", "claimed": False},
        ],
    )
    _write_log(runs, 1201, ARM_F, steps=9)

    report = render_report("r", runs)
    assert "dispatches never reported" in report
    assert "-1" not in report.split("dispatches never reported")[1].split("\n")[0]
    assert "results the agent initiated itself" in report


@pytest.mark.unit
def test_the_report_shows_the_action_taxonomy(tmp_path: Path) -> None:
    """A p-value says which way criterion 4 went; the table says how.

    The shape of the disagreement between the two columns is the actionable
    part, and it is not recoverable from the gate JSON.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=2,
        events=[
            {
                "event_type": "agent_action",
                "action": "EXECUTE_EXPERIMENT",
                "status_context": "open",
            },
            {"event_type": "agent_action", "action": "REPLAN", "status_context": "revision"},
            {"event_type": "agent_action", "action": "REPLAN", "status_context": "revision"},
        ],
    )

    log = load_run(runs)[0][0]
    assert log.agent_actions[("REPLAN", "revision")] == 2

    report = render_report("r", runs)
    section = report.split("## 10. Action taxonomy")[1]
    assert "EXECUTE_EXPERIMENT" in section
    assert "REPLAN" in section


@pytest.mark.unit
def test_a_batch_is_what_one_tool_call_handed_out(tmp_path: Path) -> None:
    """Two answers to one question are only wasteful when dispatched *together*.

    Keying the batch boundary on `get_next_targets` alone was correct while that
    was the only tool that could dispatch. Once a record could carry its own
    dispatch, nothing closed the batch and the counter became "every repeat of a
    question in the whole episode" — four phantom violations per five-value axis,
    twenty across a run, which then drove the capability index to 0.01.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=3,
        events=[
            # Three separate fused dispatches, each its own batch of one.
            {"event_type": "target_selected", "node_id": "a0", "exclusion_group": "a"},
            {"event_type": "tool_call", "tool": "record_evidence", "ok": True},
            {"event_type": "target_selected", "node_id": "a1", "exclusion_group": "a"},
            {"event_type": "tool_call", "tool": "record_evidence", "ok": True},
            {"event_type": "target_selected", "node_id": "a2", "exclusion_group": "a"},
            {"event_type": "tool_call", "tool": "record_evidence", "ok": True},
        ],
    )
    assert load_run(runs)[0][0].same_question_dispatches == 0


@pytest.mark.unit
def test_two_answers_in_one_call_are_still_counted(tmp_path: Path) -> None:
    """The violation the counter exists for must survive the boundary fix."""
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=1,
        events=[
            {"event_type": "target_selected", "node_id": "a0", "exclusion_group": "a"},
            {"event_type": "target_selected", "node_id": "a1", "exclusion_group": "a"},
            {"event_type": "tool_call", "tool": "get_next_targets", "ok": True},
        ],
    )
    assert load_run(runs)[0][0].same_question_dispatches == 1


@pytest.mark.unit
def test_the_stratified_table_censors_like_the_headline(tmp_path: Path) -> None:
    """One episode cannot be 17.7 steps in one table and 100.0 in another.

    The baseline's mean was reconstructed by adding a right-censored *difference*
    to an uncensored base, which produced a negative number of probes — a
    baseline that ran minus fifty-five experiments.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=23,
        goals_met=False,
        events=[{"event_type": "conflict_recorded", "member_ids": ["a", "b"], "n_members": 2}],
    )
    _write_log(runs, 1201, ARM_F, steps=26)

    report = render_report("r", runs)
    stratified = report.split("## 6. Stratified by conflict")[1].split("##")[0]
    assert "-55" not in stratified
    # B is censored to the 60-step budget, so the baseline reads as its own 26.
    assert "60.00" in stratified
    assert "26.00" in stratified


@pytest.mark.unit
def test_results_carried_across_a_reset_are_reported(tmp_path: Path) -> None:
    """The metric that would have named a two-probe-per-episode loss on sight.

    Before these were carried forward they were destroyed: one config re-probed
    as a duplicate, the other silently retired by the exclusion inference before
    its own result was ever recorded. Neither shows up as a failure anywhere
    else — the run just looks slower.
    """
    runs = tmp_path / "runs" / "r"
    _write_log(
        runs,
        1201,
        ARM_B,
        steps=3,
        events=[
            {"event_type": "target_selected", "node_id": "a", "status": "UNTESTED"},
            {"event_type": "probes_carried", "configs": ["component=v0", "method=v3"]},
        ],
    )
    _write_log(runs, 1201, ARM_F, steps=9)

    log = load_run(runs)[0][0]
    assert log.probes_carried == 2

    report = render_report("r", runs)
    assert "results carried across a context reset" in report
