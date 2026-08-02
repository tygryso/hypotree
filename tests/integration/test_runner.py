"""Tests for the headless agent runner and harness.

Exercises the full pipeline using the mock LLM backend — no network calls,
no real landscape server. The mock agent drives both Arm A and Arm B through
the tool surface so we verify the runner logic, session resets, logging,
and tool execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eval.environment.fake_hypothesis_tree import generate_briefing
from eval.environment.landscape_generator import _generate_dag
from eval.runner.config import ARM_A, ARM_B, TASK_SEEDS, EvalConfig, make_run_config
from eval.runner.runner import (
    MAX_IDLE_TURNS,
    MockAgent,
    RunLogger,
    _execute_tool,
    get_tools_for_arm,
    load_system_prompt,
    run,
)
from hypotree.engine import HypoTreeEngine


def _setup_eval_env(tmp_path: Path, seed: int = 1001) -> Path:
    """Create the landscape + briefing files needed for a test run."""
    eval_dir = tmp_path / "eval"
    landscapes_dir = eval_dir / "environment" / "landscapes"
    briefings_dir = eval_dir / "environment" / "briefings"
    runs_dir = eval_dir / "runs"
    landscapes_dir.mkdir(parents=True)
    briefings_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    landscape = _generate_dag(seed)
    landscape_path = landscapes_dir / f"landscape_seed_{seed}.json"
    landscape_path.write_text(json.dumps(landscape), encoding="utf-8")

    briefing = generate_briefing(landscape_path)
    briefing_path = briefings_dir / f"briefing_seed_{seed}.md"
    briefing_path.write_text(briefing, encoding="utf-8")

    return eval_dir


# -- Tool definition tests ----------------------------------------------------


@pytest.mark.unit
def test_arm_a_tools_include_scratchpad() -> None:
    """Arm A gets evaluate_config + update_scratchpad, nothing else."""
    tools = get_tools_for_arm(ARM_A)
    names = [t["function"]["name"] for t in tools]
    assert "evaluate_config" in names
    assert "update_scratchpad" in names
    assert "create_hypotheses" not in names
    assert "get_next_targets" not in names


@pytest.mark.unit
def test_arm_b_tools_include_hypotree() -> None:
    """Arm B gets evaluate_config + all hypotree tools."""
    tools = get_tools_for_arm(ARM_B)
    names = [t["function"]["name"] for t in tools]
    assert "evaluate_config" in names
    assert "create_hypotheses" in names
    assert "get_next_targets" in names
    assert "record_evidence" in names
    assert "update_scratchpad" not in names


@pytest.mark.unit
def test_arm_b_offers_exactly_one_way_to_create() -> None:
    """A singular tool beside a batch one is a decision the caller gets wrong.

    In a full evaluation run the batch variant failed eight times on payload
    shape while the singular one never did; the two differed only in arity.
    """
    names = {t["function"]["name"] for t in get_tools_for_arm(ARM_B)}
    assert "create_hypotheses" in names
    assert "create_hypothesis" not in names
    assert "bulk_create_hypotheses" not in names


# -- System prompt tests ------------------------------------------------------


@pytest.mark.unit
def test_system_prompts_exist() -> None:
    """Both arm system prompts load and contain key instructions."""
    prompt_a = load_system_prompt(ARM_A)
    prompt_b = load_system_prompt(ARM_B)
    assert "scratchpad" in prompt_a.lower()
    assert "hypotree" in prompt_b.lower()
    assert "evaluate_config" in prompt_a
    assert "evaluate_config" in prompt_b


# -- Logger tests -------------------------------------------------------------


@pytest.mark.unit
def test_logger_writes_jsonl(tmp_path: Path) -> None:
    """RunLogger appends JSONL entries with correct schema."""
    log_path = tmp_path / "test.jsonl"
    logger = RunLogger(log_path, seed=1001, arm="B")
    logger.log_run_start(
        EvalConfig(
            arm="B",
            seed=1001,
            run_id="test-run",
            landscape_path=tmp_path / "l.json",
            briefing_path=tmp_path / "b.md",
            log_path=log_path,
            landscape_url="http://x",
            tool_budget=5,
            session_breakpoints=(3,),
        )
    )
    logger.log_experiment("test_config", 0, 0.75)
    logger.log_run_end("all_goals_met", True)

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    run_start = json.loads(lines[0])
    assert run_start["event_type"] == "run_start"
    assert run_start["seed"] == 1001
    assert run_start["arm"] == "B"

    experiment = json.loads(lines[1])
    assert experiment["event_type"] == "experiment"
    assert experiment["success"] == 0.75
    assert experiment["step"] == 1

    run_end = json.loads(lines[2])
    assert run_end["event_type"] == "run_end"
    assert run_end["goals_met"] is True


# -- Mock agent tests ---------------------------------------------------------


@pytest.mark.unit
def test_mock_agent_arm_b_state_machine(tmp_path: Path) -> None:
    """Mock agent for Arm B follows: create goal → bulk → target → eval → record."""
    eval_dir = _setup_eval_env(tmp_path)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run")
    agent = MockAgent(config)
    tools = get_tools_for_arm(ARM_B)
    tool_names = [t["function"]["name"] for t in tools]

    # Phase 1: create the goal and the hypotheses in one call.
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "create_hypotheses"
    assert call["arguments"]["hypotheses"][0]["is_goal"] is True
    assert len(call["arguments"]["hypotheses"]) > 1

    # Phase 2: get_next_targets.
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "get_next_targets"

    # Simulate a target response.
    agent.observe_result(
        "get_next_targets",
        json.dumps(
            [
                {
                    "status": "SELECTED",
                    "node_id": "H001",
                    "statement": "test statement",
                    "claim_id": "claim123",
                }
            ]
        ),
    )

    # Step: evaluate the selected target config.
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "evaluate_config"
    assert call["arguments"]["config"] == "test statement"

    # Simulate eval result.
    agent.observe_result("evaluate_config", json.dumps({"success": 0.8}))

    # Phase 5: record_evidence.
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "record_evidence"
    assert call["arguments"]["node_id"] == "H001"
    assert call["arguments"]["success"] == 0.8
    assert call["arguments"]["claim_id"] == "claim123"


@pytest.mark.unit
def test_mock_agent_arm_a_state_machine(tmp_path: Path) -> None:
    """Mock agent for Arm A follows: probe → scratchpad → probe ..."""
    eval_dir = _setup_eval_env(tmp_path)
    config = make_run_config(ARM_A, 1001, eval_dir, "test-run")
    agent = MockAgent(config)
    tools = get_tools_for_arm(ARM_A)
    tool_names = [t["function"]["name"] for t in tools]

    # Phase 1: probe (evaluate_config).
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "evaluate_config"

    # Simulate eval result.
    agent.observe_result("evaluate_config", json.dumps({"success": 0.3}))

    # Phase 2: scratchpad update.
    call = agent.get_tool_call(tool_names)
    assert call["name"] == "update_scratchpad"
    assert "0.30" in call["arguments"]["content"]


# -- End-to-end runner tests --------------------------------------------------


@pytest.mark.integration
def test_runner_arm_b_mock_completes(tmp_path: Path) -> None:
    """Full runner pipeline for Arm B with mock agent — verifies logging + completion.

    The mock agent does not need a landscape server in Arm B because the
    evaluate_config tool is patched to return ground truth directly.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="mock")
    # Small budget for fast test.
    config = config.__class__(**{**config.__dict__, "tool_budget": 10})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    # Patch the landscape probe so the mock agent doesn't need a real server.
    def mock_probe(config_str: str, depth: int, url: str = "") -> dict:
        # Return low scores so the run doesn't win early — we're testing
        # session reset logic, not the win condition.
        return {"success": 0.3, "metrics": {}}

    with patch("eval.runner.runner.landscape_probe", side_effect=mock_probe):
        result = run(config)

    assert result["seed"] == 1001
    assert result["arm"] == "B"
    assert result["log_path"] == str(config.log_path)

    # Verify the JSONL log was populated.
    log_lines = config.log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(log_lines) >= 2  # at least run_start + run_end

    events = [json.loads(line) for line in log_lines]
    assert events[0]["event_type"] == "run_start"
    assert events[-1]["event_type"] == "run_end"

    # Verify experiments were logged.
    experiments = [e for e in events if e["event_type"] == "experiment"]
    assert len(experiments) > 0

    # Verify nodes were created.
    node_events = [e for e in events if e["event_type"] == "node_created"]
    assert len(node_events) >= 1  # at least the goal node


@pytest.mark.integration
def test_runner_arm_a_mock_completes(tmp_path: Path) -> None:
    """Full runner pipeline for Arm A with mock agent."""
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_A, 1001, eval_dir, "test-run", llm_backend="mock")
    config = config.__class__(**{**config.__dict__, "tool_budget": 5})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    def mock_probe(config_str: str, depth: int, url: str = "") -> dict:
        # Return low scores so the run doesn't win early — we're testing
        # session reset logic, not the win condition.
        return {"success": 0.3, "metrics": {}}

    with patch("eval.runner.runner.landscape_probe", side_effect=mock_probe):
        result = run(config)

    assert result["arm"] == "A"
    assert result["goals_met"] is False  # Arm A has no goal mechanism
    assert result["reason"] in ("budget_exhausted", "agent_stopped")

    log_lines = config.log_path.read_text(encoding="utf-8").strip().split("\n")
    events = [json.loads(line) for line in log_lines]
    experiments = [e for e in events if e["event_type"] == "experiment"]
    assert len(experiments) > 0


@pytest.mark.integration
def test_runner_llm_path_stops_at_landscape_win(tmp_path: Path) -> None:
    """Regression: a landscape win in the LLM path must break the OUTER budget
    loop, not just the inner tool-call loop.

    The win check lived inside ``for tc in tool_calls`` and its ``break`` only
    exited that inner loop; the outer ``while step < tool_budget`` kept running
    to exhaustion, so every ``steps_to_target`` censored to the full budget and
    criterion 1 could never differentiate the arms. This asserts the run stops
    at the winning step instead.
    """
    from unittest.mock import patch

    from eval.runner.runner import _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    # Big budget, no session resets — isolate the win-break behaviour.
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    # A complete candidate probed at confirm depth — the only shape that can win.
    criteria = _load_win_criteria(config.landscape_path)
    winning_config = criteria["goal_config"]
    win_depth = criteria["min_confirm_depth"]

    # The LLM issues exactly one probe on its first turn; the probe clears the
    # target. If the fix is absent, the outer loop keeps going to budget.
    def fake_api(*_args: object, **_kwargs: object) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "evaluate_config",
                                    "arguments": json.dumps(
                                        {"config": winning_config, "depth": win_depth}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    def mock_probe(config_str: str, depth: int, url: str = "") -> dict:
        return {"success": 0.95, "metrics": {}}

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=fake_api),
        patch("eval.runner.runner.landscape_probe", side_effect=mock_probe),
    ):
        result = run(config)

    assert result["goals_met"] is True
    assert result["reason"] == "all_goals_met"
    # The win fired on the first probe → the run MUST stop at step 1, not drain
    # the full 60-step budget.
    assert result["steps_to_target"] == 1


@pytest.mark.integration
def test_a_turn_without_tool_calls_is_nudged_not_fatal(tmp_path: Path) -> None:
    """One turn of prose must not end an episode that still has budget.

    Ending there censored a *baseline* arm to the full 100-step budget after 25
    productive probes and scored it as a failure, inflating the treatment arm's
    reported advantage. The agent gets told to use a tool and carries on; the
    idle-turn guard still bounds a model that never experiments.
    """
    from unittest.mock import patch

    from eval.runner.runner import _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    criteria = _load_win_criteria(config.landscape_path)
    turns: list[str] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        # First turn: prose only. Second turn: the winning probe.
        turns.append("x")
        if len(turns) == 1:
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Let me think..."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "evaluate_config",
                                    "arguments": json.dumps(
                                        {
                                            "config": criteria["goal_config"],
                                            "depth": criteria["min_confirm_depth"],
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    def mock_probe(config_str: str, depth: int, url: str = "") -> dict:
        return {"success": 0.95, "metrics": {}}

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=fake_api),
        patch("eval.runner.runner.landscape_probe", side_effect=mock_probe),
    ):
        result = run(config)

    assert result["goals_met"] is True
    assert result["steps_to_target"] == 1


@pytest.mark.integration
def test_an_agent_that_never_calls_a_tool_still_terminates(tmp_path: Path) -> None:
    """The nudge must not become an unbounded loop — MAX_IDLE_TURNS bounds it."""
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    calls: list[str] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        calls.append("x")
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "thinking"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    assert result["goals_met"] is False
    assert result["reason"] == "no_progress"
    assert len(calls) == MAX_IDLE_TURNS


@pytest.mark.integration
def test_runner_session_reset_logged(tmp_path: Path) -> None:
    """Session resets are logged when breakpoints are hit."""
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="mock")
    # Override to force a session reset at step 2.
    config = config.__class__(**{**config.__dict__, "tool_budget": 8, "session_breakpoints": (2,)})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    def mock_probe(config_str: str, depth: int, url: str = "") -> dict:
        # Return low scores so the run doesn't win early — we're testing
        # session reset logic, not the win condition.
        return {"success": 0.3, "metrics": {}}

    with patch("eval.runner.runner.landscape_probe", side_effect=mock_probe):
        run(config)

    log_lines = config.log_path.read_text(encoding="utf-8").strip().split("\n")
    events = [json.loads(line) for line in log_lines]
    resets = [e for e in events if e["event_type"] == "session_reset"]
    assert len(resets) >= 1


# -- Status-transition + pruned-reexecution instrumentation -------------------


def _bare_config(tmp_path: Path, log_path: Path) -> EvalConfig:
    """Minimal EvalConfig for exercising _execute_tool directly (record_evidence
    ignores every config field, so paths/URL are placeholders)."""
    return EvalConfig(
        arm=ARM_B,
        seed=1001,
        run_id="test-run",
        landscape_path=tmp_path / "l.json",
        briefing_path=tmp_path / "b.md",
        log_path=log_path,
        landscape_url="http://x",
    )


@pytest.mark.integration
def test_record_evidence_logs_propagated_prune(tmp_path: Path) -> None:
    """Failing evidence on a parent cascades a prune to its child, and the
    runner surfaces that as a propagated status_transition — the raw signal
    criterion 3 (status utility) depends on."""
    log_path = tmp_path / "prune.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    engine.create_hypothesis("parent", node_id="P")
    engine.create_hypothesis("child", parent_ids=["P"], node_id="C")

    # Deterministic node: a single failing observation invalidates P and
    # cascade-prunes its descendant C.
    _execute_tool("record_evidence", {"node_id": "P", "success": 0.0}, engine, [], config, logger)

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
    transitions = [e for e in events if e["event_type"] == "status_transition"]
    propagated = [t for t in transitions if t["propagated"] is True]
    assert any(t["node_id"] == "C" and t["new_status"] == "PRUNED" for t in propagated)
    # The evidence target itself is NOT flagged as propagated.
    assert all(not t["propagated"] for t in transitions if t["node_id"] == "P")
    engine.close()


@pytest.mark.integration
def test_record_evidence_on_pruned_node_logs_reexecution(tmp_path: Path) -> None:
    """Recording evidence on an already-pruned (dead) branch emits a
    pruned_reexecution event — the hard-gate wasted-work signal."""
    log_path = tmp_path / "reexec.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    engine.create_hypothesis("parent", node_id="P")
    engine.create_hypothesis("child", parent_ids=["P"], node_id="C")
    _execute_tool("record_evidence", {"node_id": "P", "success": 0.0}, engine, [], config, logger)

    # C is now PRUNED — re-executing it is redundant work.
    _execute_tool("record_evidence", {"node_id": "C", "success": 0.5}, engine, [], config, logger)

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
    reexec = [e for e in events if e["event_type"] == "pruned_reexecution"]
    assert any(e["node_id"] == "C" for e in reexec)
    engine.close()


# -- Arm-agnostic win condition tests -----------------------------------------


@pytest.mark.unit
def test_check_landscape_win_requires_complete_candidate_at_depth() -> None:
    """A win needs all axes assigned, the target cleared, and confirmation at depth."""
    from eval.runner.runner import _check_landscape_win

    criteria = {"target_metric": 0.75, "axes": ["a", "b"], "min_confirm_depth": 2}
    full = "a=v1;b=v2"
    assert _check_landscape_win(full, depth=2, success=0.8, criteria=criteria)
    assert _check_landscape_win(full, depth=3, success=0.85, criteria=criteria)


@pytest.mark.unit
def test_check_landscape_win_below_target_fails() -> None:
    """A probe below target_metric never wins."""
    from eval.runner.runner import _check_landscape_win

    criteria = {"target_metric": 0.75, "axes": ["a", "b"], "min_confirm_depth": 2}
    assert not _check_landscape_win("a=v1;b=v2", depth=3, success=0.5, criteria=criteria)


@pytest.mark.unit
def test_check_landscape_win_shallow_depth_fails() -> None:
    """A high score at shallow depth never wins, for ANY config.

    Shallow probing cannot separate a genuine answer from the planted decoy, so
    the depth requirement applies uniformly rather than to one special-cased
    trap string — simpler, and impossible to sidestep by probing a different
    config that happens to mirage.
    """
    from eval.runner.runner import _check_landscape_win

    criteria = {"target_metric": 0.75, "axes": ["a", "b"], "min_confirm_depth": 2}
    assert not _check_landscape_win("a=v1;b=v2", depth=0, success=0.9, criteria=criteria)
    assert not _check_landscape_win("a=v1;b=v2", depth=1, success=0.9, criteria=criteria)


@pytest.mark.unit
def test_check_landscape_win_rejects_premise_probe() -> None:
    """Confirming one axis is not achieving the goal.

    A confirmed premise legitimately scores full marks, so without the
    completeness rule a single premise probe would end the run at step one.
    """
    from eval.runner.runner import _check_landscape_win

    criteria = {"target_metric": 0.75, "axes": ["a", "b"], "min_confirm_depth": 2}
    assert not _check_landscape_win("a=v1", depth=2, success=1.0, criteria=criteria)


@pytest.mark.unit
def test_load_win_criteria_extracts_correct_fields(tmp_path: Path) -> None:
    """_load_win_criteria extracts the target, axes, depth and regime."""
    from eval.runner.config import TASK_SEEDS
    from eval.runner.runner import _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=TASK_SEEDS[0])
    config = make_run_config(ARM_B, TASK_SEEDS[0], eval_dir, "test-run")
    criteria = _load_win_criteria(config.landscape_path)
    # The win threshold must be the goal node's DECLARED target_metric (0.75),
    # NOT its hidden true_success. Reading true_success silently raised the bar
    # and rejected legitimate wins just above the declared target.
    assert criteria["target_metric"] == 0.75
    assert criteria["min_confirm_depth"] > 0
    assert criteria["axes"]
    assert criteria["evidence_regime"] == "deterministic"
    assert criteria["goal_config"] != ""


@pytest.mark.unit
def test_load_win_criteria_uses_target_metric_not_true_success(tmp_path: Path) -> None:
    """A config just above the declared target still counts as a win."""
    from eval.runner.config import TASK_SEEDS
    from eval.runner.runner import _check_landscape_win, _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=TASK_SEEDS[0])
    config = make_run_config(ARM_B, TASK_SEEDS[0], eval_dir, "test-run")
    criteria = _load_win_criteria(config.landscape_path)
    win_cfg = criteria["goal_config"]
    assert _check_landscape_win(win_cfg, depth=3, success=0.78, criteria=criteria)
    assert not _check_landscape_win(win_cfg, depth=3, success=0.70, criteria=criteria)


@pytest.mark.unit
def test_compact_summary_surfaces_verified_nodes(tmp_path: Path) -> None:
    """Arm B session-reset summary lists VERIFIED wins with their statements."""
    from eval.runner.runner import _compact_summary
    from hypotree.models.status import Status

    engine = HypoTreeEngine(":memory:", rng_seed=7)
    engine.create_hypothesis(statement="winning_config_alpha", node_id="W")
    engine.update_status("W", Status.VERIFIED, reason="test")

    summary = _compact_summary(engine, [], ARM_B)
    # The reset summary must surface confirmed wins (config strings), not an
    # opaque created_at-ordered dump that is useless after a context reset.
    assert "CONFIRMED" in summary
    assert "winning_config_alpha" in summary
    engine.close()


@pytest.mark.unit
def test_compact_summary_carries_the_evidence_ledger(tmp_path: Path) -> None:
    """The Arm B summary must carry config→score results, not just statuses.

    Statuses alone tell the agent nothing about which candidates were already
    tried or how well they scored, so after a reset it re-probes them — the
    single largest source of wasted budget observed in the superseded run. The
    ledger is the belief state that actually matters for a search task.
    """
    from eval.runner.runner import _compact_summary
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=7)
    engine.create_hypothesis(statement="component=v1;method=v2", node_id="A")
    engine.create_hypothesis(statement="component=v3;method=v0", node_id="B")
    engine.record_evidence("A", LogicalEvidence(success=0.73))
    engine.record_evidence("B", LogicalEvidence(success=0.25))

    summary = _compact_summary(engine, [], ARM_B)
    assert "Evidence ledger" in summary
    assert "component=v1;method=v2" in summary
    assert "0.730" in summary
    assert "0.250" in summary
    # Best-scoring lead must survive ahead of the weaker one.
    assert summary.index("0.730") < summary.index("0.250")
    assert "Do NOT re-probe" in summary
    engine.close()


@pytest.mark.unit
def test_compact_summary_arm_a_reports_missing_notes(tmp_path: Path) -> None:
    """Arm A's summary must state plainly when nothing was saved."""
    from eval.runner.runner import _compact_summary

    engine = HypoTreeEngine(":memory:", rng_seed=7)
    empty = _compact_summary(engine, [], ARM_A)
    assert "no notes" in empty.lower()

    # Multiple appended entries are all carried across the reset, not just the
    # first — the baseline arm's memory must be a genuine cumulative notebook.
    filled = _compact_summary(engine, ["first finding", "second finding"], ARM_A)
    assert "first finding" in filled
    assert "second finding" in filled
    engine.close()


@pytest.mark.unit
def test_briefing_is_reinjected_after_every_session_reset(tmp_path: Path) -> None:
    """The task specification must survive a context reset, for BOTH arms.

    The briefing is the only place the config grammar and the axis menu are
    stated. When it was dropped at a reset the baseline arm stopped emitting
    parseable probes entirely, so criterion 1 measured grammar recall rather than
    retained findings. Re-injecting it is what makes the comparison valid.
    """
    from unittest.mock import patch

    for arm in (ARM_A, ARM_B):
        eval_dir = _setup_eval_env(tmp_path / arm, seed=1001)
        config = make_run_config(arm, 1001, eval_dir, "test-run", llm_backend="openai")
        config = config.__class__(
            **{**config.__dict__, "tool_budget": 6, "session_breakpoints": (2, 4)}
        )
        briefing = config.briefing_path.read_text(encoding="utf-8")
        seen_after_reset: list[list[dict]] = []

        def fake_api(
            _base_url: str,
            _model: str,
            messages: list[dict],
            *_a: object,
            _sink: list = seen_after_reset,
            **_k: object,
        ) -> dict:
            _sink.append(list(messages))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "evaluate_config",
                                        "arguments": json.dumps(
                                            {"config": "component=v0", "depth": 0}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }

        with (
            patch("eval.runner.runner._call_openai_api", side_effect=fake_api),
            patch(
                "eval.runner.runner.landscape_probe",
                side_effect=lambda c, d, url="": {"success": 0.1, "metrics": {}},
            ),
        ):
            run(config)

        # Every turn the model ever sees must contain the full briefing.
        assert seen_after_reset, arm
        for messages in seen_after_reset:
            joined = "\n".join(m.get("content") or "" for m in messages)
            assert briefing in joined, arm


@pytest.mark.unit
def test_experiment_log_flags_duplicate_probes(tmp_path: Path) -> None:
    """Repeat probes are flagged so wasted budget is measurable from the logs."""
    log_path = tmp_path / "dup.jsonl"
    logger = RunLogger(log_path, seed=1101, arm=ARM_B)
    logger.log_experiment("component=v1", 0, 0.5)
    logger.log_experiment("component=v2", 0, 0.0)
    logger.log_experiment("component=v1", 0, 0.5)  # repeat — zero information

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [e["duplicate"] for e in entries] == [False, False, True]
    assert entries[-1]["distinct_configs"] == 2


@pytest.mark.unit
def test_evidence_log_records_regime_and_status_change(tmp_path: Path) -> None:
    """regime and old_status were unlogged, hiding a frozen belief state."""
    log_path = tmp_path / "ev.jsonl"
    logger = RunLogger(log_path, seed=1101, arm=ARM_B)
    logger.log_evidence_recorded(
        "n1",
        0.25,
        "EXHAUSTED",
        old_status="IN_PROGRESS",
        regime="deterministic",
        evidence_count=1,
        posterior_mean=0.4,
    )
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["regime"] == "deterministic"
    assert entry["old_status"] == "IN_PROGRESS"
    assert entry["new_status"] == "EXHAUSTED"
    assert entry["status_changed"] is True
    assert entry["evidence_count"] == 1


# -- Baseline symmetry, telemetry and regime pinning --------------------------


@pytest.mark.unit
def test_arm_f_summary_auto_preserves_every_probe(tmp_path: Path) -> None:
    """The flat baseline keeps all raw facts without the agent doing anything.

    Arm A's memory depends on the agent choosing to write notes; a previous gate
    was decided entirely by that, with the baseline saving nothing on 6 of 10
    seeds. Arm F removes the confound so the remaining comparison is about
    structure rather than diligence.
    """
    from eval.runner.config import ARM_F
    from eval.runner.runner import _compact_summary

    engine = HypoTreeEngine(":memory:", rng_seed=3)
    transcript = [
        {"config": "component=v0", "depth": 1, "success": 0.0},
        {"config": "component=v2", "depth": 1, "success": 1.0},
    ]
    summary = _compact_summary(engine, [], ARM_F, transcript)

    assert "component=v0" in summary
    assert "component=v2" in summary
    # Raw facts only — no interpretation the structured arm would supply.
    assert "EXHAUSTED" not in summary
    assert "frontier" not in summary.lower()
    engine.close()


@pytest.mark.unit
def test_arm_a_summary_has_no_auto_transcript(tmp_path: Path) -> None:
    """Arm A must stay the discipline-dependent baseline, or 1a measures nothing."""
    from eval.runner.runner import _compact_summary

    engine = HypoTreeEngine(":memory:", rng_seed=3)
    transcript = [{"config": "component=v0", "depth": 1, "success": 0.0}]
    summary = _compact_summary(engine, [], ARM_A, transcript)

    assert "component=v0" not in summary
    assert "no notes" in summary.lower()
    engine.close()


@pytest.mark.unit
def test_scratchpad_writes_are_logged(tmp_path: Path) -> None:
    """Note-taking must be visible in the logs, not inferred from a summary size."""
    from eval.runner.config import make_run_config
    from eval.runner.runner import _execute_tool

    eval_dir = _setup_eval_env(tmp_path, seed=TASK_SEEDS[0])
    config = make_run_config(ARM_A, TASK_SEEDS[0], eval_dir, "test-run")
    log_path = tmp_path / "sp.jsonl"
    logger = RunLogger(log_path, seed=TASK_SEEDS[0], arm=ARM_A)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    scratchpad: list[str] = []

    _execute_tool("update_scratchpad", {"content": "first"}, engine, scratchpad, config, logger)
    _execute_tool("update_scratchpad", {"content": "second"}, engine, scratchpad, config, logger)

    writes = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "scratchpad_write"
    ]
    assert len(writes) == 2
    assert writes[-1]["entries"] == 2  # append is the default, nothing overwritten
    assert writes[-1]["total_chars"] == len("first") + len("second")
    engine.close()


@pytest.mark.unit
def test_evidence_regime_is_pinned_to_the_environment(tmp_path: Path) -> None:
    """An agent must not be able to switch off the revision machinery.

    A run in which the agent declared every node stochastic produced zero
    invalidations, verifications and exhaustions from 14 conclusive observations:
    the stochastic path waits on a convergence gate a one-shot oracle never
    closes. The regime describes the environment, so the harness pins it.
    """
    from eval.runner.config import make_run_config
    from eval.runner.runner import _execute_tool

    eval_dir = _setup_eval_env(tmp_path, seed=TASK_SEEDS[0])
    config = make_run_config(ARM_B, TASK_SEEDS[0], eval_dir, "test-run")
    log_path = tmp_path / "regime.jsonl"
    logger = RunLogger(log_path, seed=TASK_SEEDS[0], arm=ARM_B)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    _execute_tool(
        "create_hypotheses",
        {"hypotheses": [{"statement": "x", "node_id": "n1", "evidence_regime": "stochastic"}]},
        engine,
        [],
        config,
        logger,
    )

    assert engine._store.get_node("n1").evidence_regime == "deterministic"
    overrides = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "regime_override"
    ]
    assert len(overrides) == 1
    assert overrides[0]["requested"] == "stochastic"
    engine.close()


# -- Conflict + exclusion-group instrumentation -------------------------------


@pytest.mark.integration
def test_record_evidence_logs_conflict_events(tmp_path: Path) -> None:
    """A refuted combination over two assumptions is logged as a conflict, and its
    later resolution as a culprit — the two signals criterion 3 now counts."""
    log_path = tmp_path / "conflict.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    for group, ids in (("component", ("c1", "c2")), ("regime", ("r1", "r2"))):
        for nid in ids:
            engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
    engine.create_hypothesis("combo", node_id="combo", parent_ids=["c1", "r1"])
    _execute_tool("record_evidence", {"node_id": "c1", "success": 1.0}, engine, [], config, logger)
    _execute_tool("record_evidence", {"node_id": "r1", "success": 1.0}, engine, [], config, logger)
    _execute_tool(
        "record_evidence", {"node_id": "combo", "success": 0.0}, engine, [], config, logger
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
    conflicts = [e for e in events if e["event_type"] == "conflict_recorded"]
    assert len(conflicts) == 1
    assert sorted(conflicts[0]["member_ids"]) == ["c1", "r1"]
    assert conflicts[0]["n_members"] == 2

    # A different combination succeeds, exonerating c1 and convicting r1.
    engine.create_hypothesis("combo2", node_id="combo2", parent_ids=["c1", "r2"])
    _execute_tool("record_evidence", {"node_id": "r2", "success": 1.0}, engine, [], config, logger)
    _execute_tool(
        "record_evidence", {"node_id": "combo2", "success": 1.0}, engine, [], config, logger
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
    resolved = [e for e in events if e["event_type"] == "conflict_resolved"]
    assert [e["culprit_id"] for e in resolved] == ["r1"]
    engine.close()


@pytest.mark.integration
def test_create_hypotheses_logs_exclusion_group(tmp_path: Path) -> None:
    """Group adoption was reverse-engineered from transitions before; log it directly."""
    log_path = tmp_path / "groups.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    _execute_tool(
        "create_hypotheses",
        {"hypotheses": [{"statement": "component=v1", "exclusion_group": "component"}]},
        engine,
        [],
        config,
        logger,
    )
    _execute_tool(
        "create_hypotheses", {"hypotheses": [{"statement": "loose"}]}, engine, [], config, logger
    )

    created = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").strip().split("\n")
        if json.loads(line)["event_type"] == "node_created"
    ]
    assert [e["exclusion_group"] for e in created] == ["component", None]
    engine.close()


@pytest.mark.integration
def test_create_hypotheses_counts_declared_groups(tmp_path: Path) -> None:
    """One event per node, and a composition is marked as unable to declare a group."""
    log_path = tmp_path / "bulk.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    _execute_tool(
        "create_hypotheses",
        {
            "hypotheses": [
                {
                    "statement": "component=v1",
                    "node_id": "c1",
                    "exclusion_group": "component",
                },
                {
                    "statement": "component=v2",
                    "node_id": "c2",
                    "exclusion_group": "component",
                },
                {"statement": "combination", "node_id": "combo", "parent_ids": ["c1"]},
            ]
        },
        engine,
        [],
        config,
        logger,
    )

    created = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").strip().split("\n")
        if json.loads(line)["event_type"] == "node_created"
    ]
    assert len(created) == 3
    assert sum(1 for e in created if e["exclusion_group"]) == 2
    assert [e["composed"] for e in created] == [False, False, True]
    engine.close()


@pytest.mark.unit
def test_arm_b_exposes_the_conflict_tools() -> None:
    """The belief state can only be used through tools the agent can see."""
    names = {t["function"]["name"] for t in get_tools_for_arm(ARM_B)}
    assert {"get_conflicts", "suggest_discriminating_experiment"} <= names


@pytest.mark.unit
def test_baseline_arms_never_see_belief_state_tools() -> None:
    for arm in (ARM_A, "F"):
        names = {t["function"]["name"] for t in get_tools_for_arm(arm)}
        assert "get_conflicts" not in names
        assert "suggest_discriminating_experiment" not in names


@pytest.mark.unit
def test_llm_call_logs_latency_and_tokens(tmp_path: Path) -> None:
    """Turn-level overhead is invisible in step counts, so it is logged directly."""
    log_path = tmp_path / "llm.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    logger.log_llm_call(
        1.25, 2, prompt_tokens=900, completion_tokens=60, finish_reason="tool_calls"
    )

    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "llm_call"
    assert event["duration_s"] == 1.25
    assert event["n_tool_calls"] == 2
    assert event["prompt_tokens"] == 900
    assert event["completion_tokens"] == 60
    assert event["finish_reason"] == "tool_calls"


@pytest.mark.unit
def test_arm_b_prompt_documents_every_tool_it_offers() -> None:
    """The prompt forbids calling undocumented tools, so the two must agree.

    A tool present in the schema but absent from the prompt is unreachable; a
    tool named in the prompt but absent from the schema is an invitation to
    hallucinate a call.
    """
    prompt = load_system_prompt(ARM_B)
    for tool in get_tools_for_arm(ARM_B):
        assert f"`{tool['function']['name']}`" in prompt


@pytest.mark.integration
def test_session_summary_surfaces_unresolved_conflicts(tmp_path: Path) -> None:
    """A reset must not erase the one thing a flat log cannot express.

    Nodes under conflict review sit in NEEDS_REVISION, which appears in none of
    the confirmed / settled / open listings — without a dedicated section they
    vanish from the summary and the agent restarts blind to the contradiction.
    """
    from eval.runner.runner import _compact_summary
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        for group, ids in (("comp", ("c1", "c2")), ("reg", ("r1", "r2"))):
            for nid in ids:
                engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
        engine.create_hypothesis("combo", node_id="combo", parent_ids=["c1", "r1"])
        engine.record_evidence("c1", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("r1", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

        summary = _compact_summary(engine, [], ARM_B)

        assert "UNRESOLVED CONFLICTS" in summary
        assert "depth 2" in summary
        assert "`c1`" in summary and "`r1`" in summary
    finally:
        engine.close()


@pytest.mark.integration
def test_session_summary_omits_the_section_when_nothing_conflicts(tmp_path: Path) -> None:
    from eval.runner.runner import _compact_summary

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypothesis("h1", node_id="h1")
        assert "UNRESOLVED CONFLICTS" not in _compact_summary(engine, [], ARM_B)
    finally:
        engine.close()


# -- reset scheduling + depth inheritance -------------------------------------


@pytest.mark.unit
def test_reset_is_deferred_while_the_agent_holds_an_unrecorded_result() -> None:
    """A reset landing between a probe and its record destroys the probe.

    The dangerous state is holding a *result*, not holding a lease. A lease with
    nothing probed against it costs nothing to reset — the node returns to the
    frontier. Reading live claims instead was correct only while a dispatch was
    the last thing to happen before a probe: once a record could carry its own
    dispatch the agent holds a claim continuously, so the check read "never
    safe", the deferral budget burned down and the reset fired anyway, in
    exactly the window it existed to avoid.
    """
    from eval.runner.runner import _reset_is_safe
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses(
            [{"statement": "h1", "node_id": "h1"}, {"statement": "h2", "node_id": "h2"}]
        )
        transcript: list[dict[str, object]] = []
        assert _reset_is_safe(engine, ARM_B, transcript) is True

        # Holding a lease with nothing probed against it is not a reason to wait.
        target = engine.get_next_targets()[0]
        assert _reset_is_safe(engine, ARM_B, transcript) is True

        # Probing it is: that answer exists nowhere but the agent's context.
        transcript.append({"config": target.statement, "depth": 1, "success": 0.5})
        assert _reset_is_safe(engine, ARM_B, transcript) is False
        # The transcript arms are never mid-flight, so they never defer.
        assert _reset_is_safe(engine, ARM_A, transcript) is True
        assert _reset_is_safe(engine, "F", transcript) is True

        result = engine.record_evidence(
            target.node_id, LogicalEvidence(success=0.5), claim_id=target.claim_id
        )
        # The fused dispatch hands back another lease, and that must not by
        # itself make a reset unsafe again.
        assert result.next_targets == [] or result.next_targets[0].status in ("SELECTED", "DONE")
        assert _reset_is_safe(engine, ARM_B, transcript) is True
    finally:
        engine.close()


@pytest.mark.unit
def test_a_fused_dispatch_alone_never_blocks_a_reset() -> None:
    """The steady state of a fused loop is holding a lease; that is not a warning.

    This is the shape that broke it: every record hands back the next targets, so
    an agent in the ordinary loop never has zero live claims, and a check keyed
    on claims can never report safe.
    """
    from eval.runner.runner import _reset_is_safe
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": f"h{i}", "node_id": f"h{i}"} for i in range(4)])
        target = engine.get_next_targets()[0]
        engine.record_evidence(
            target.node_id,
            LogicalEvidence(success=1.0),
            claim_id=target.claim_id,
            count_next_targets=2,
        )
        assert engine.get_active_claims()  # the loop is holding work, as designed
        assert _reset_is_safe(engine, ARM_B, []) is True
    finally:
        engine.close()


@pytest.mark.unit
def test_the_summary_carries_what_the_reset_would_have_destroyed() -> None:
    """Parity with the transcript arm, which cannot lose a probe by construction.

    Arm F is handed every raw probe across a reset precisely so it never loses a
    fact for want of writing it down. Arm B's belief state holds only what was
    *recorded*, so an in-flight result vanished — and the reset stopped measuring
    which substrate preserves what was learned.
    """
    from eval.runner.runner import _compact_summary

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": "component=v1", "node_id": "c1"}])
        transcript = [{"config": "component=v1", "depth": 2, "success": 0.9}]
        summary = _compact_summary(engine, [], ARM_B, transcript)
        assert "UNRECORDED" in summary
        assert "component=v1" in summary

        from hypotree.models.evidence import LogicalEvidence

        engine.record_evidence("c1", LogicalEvidence(success=0.9, depth=2))
        assert "UNRECORDED" not in _compact_summary(engine, [], ARM_B, transcript)
    finally:
        engine.close()


@pytest.mark.unit
def test_arm_b_does_not_receive_the_whole_transcript() -> None:
    """Only the in-flight results cross the reset, never the flat log itself.

    Handing B the transcript would erase the distinction the gate exists to
    measure: structure against a raw record of everything.
    """
    from eval.runner.runner import _compact_summary
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": "component=v1", "node_id": "c1"}])
        engine.record_evidence("c1", LogicalEvidence(success=0.9, depth=2))
        transcript = [
            {"config": "component=v1", "depth": 2, "success": 0.9},
            {"config": "component=v9", "depth": 1, "success": 0.1},
        ]
        summary = _compact_summary(engine, [], ARM_B, transcript)
        # The recorded probe is in the ledger, not replayed raw; the unmodelled
        # one is in flight and must survive.
        assert "UNRECORDED" in summary
        assert "component=v9" in summary
        assert summary.count("component=v1 | 2 | 0.9") == 0
    finally:
        engine.close()


@pytest.mark.unit
def test_evidence_inherits_the_depth_of_its_probe() -> None:
    """The harness saw the depth; making the agent restate it invites silent error."""
    from eval.runner.runner import _last_probe_depth

    transcript = [
        {"config": "component=v1", "depth": 1, "success": 1.0},
        {"config": "a=1;b=2", "depth": 2, "success": 0.0},
        {"config": "component=v1", "depth": 3, "success": 1.0},
    ]

    # The most recent probe of that configuration wins.
    assert _last_probe_depth(transcript, "component=v1") == 3
    assert _last_probe_depth(transcript, "a=1;b=2") == 2
    # Never probed, and the no-transcript case.
    assert _last_probe_depth(transcript, "never=probed") == 0
    assert _last_probe_depth(None, "component=v1") == 0


@pytest.mark.integration
def test_record_evidence_falls_back_to_the_probe_depth(tmp_path: Path) -> None:
    """An agent that omits depth still gets depth-aware blame, not a silent 0."""
    log_path = tmp_path / "depth.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    transcript: list[dict] = [{"config": "component=v1", "depth": 2, "success": 1.0}]
    try:
        engine.create_hypothesis("component=v1", node_id="c1")
        _execute_tool(
            "record_evidence",
            {"node_id": "c1", "success": 1.0},
            engine,
            [],
            config,
            logger,
            transcript,
        )
        assert engine._store.get_node("c1").confirmed_depth == 2
    finally:
        engine.close()


@pytest.mark.integration
def test_an_explicit_depth_beats_the_inferred_one(tmp_path: Path) -> None:
    log_path = tmp_path / "depth2.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    transcript: list[dict] = [{"config": "component=v1", "depth": 1, "success": 1.0}]
    try:
        engine.create_hypothesis("component=v1", node_id="c1")
        _execute_tool(
            "record_evidence",
            {"node_id": "c1", "success": 1.0, "depth": 3},
            engine,
            [],
            config,
            logger,
            transcript,
        )
        assert engine._store.get_node("c1").confirmed_depth == 3
    finally:
        engine.close()


@pytest.mark.integration
def test_an_exhausted_frontier_is_not_scored_as_a_win(tmp_path: Path) -> None:
    """`empty_frontier` is the opposite of success and must not be scored as one.

    The runner treated any DONE sentinel as "all goals met", so a run that
    stalled — everything settled, or every remaining node held under an
    unreported lease — ended early, was flagged `goals_met=True`, and handed the
    gate its low step count as a fast solve instead of being censored to the
    budget.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    # The agent asks for work before creating any, so the frontier is empty.
    def fake_api(*_args: object, **_kwargs: object) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_next_targets",
                                    "arguments": json.dumps({"count": 1}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    assert result["goals_met"] is False
    assert result["reason"] == "frontier_exhausted"
    # Censored to the budget, because nothing was achieved.
    assert result["steps_to_target"] == config.tool_budget


@pytest.mark.integration
def test_believing_the_goal_is_met_is_not_a_win(tmp_path: Path) -> None:
    """Filing results against the goal can neither win the run nor be accepted.

    A goal's posterior used to clear its own bar after a few successes recorded
    against it, at which point the navigator reported `all_goals_met` — with no
    winning configuration ever probed. That is a route to victory no baseline arm
    has. The engine now refuses the record outright, and the harness scores the
    run on the environment result alone, so neither half of the shortcut works.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    calls: list[str] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        # Create a goal, then repeatedly claim success against it.
        if not calls:
            calls.append("create")
            tool = {
                "name": "create_hypotheses",
                "arguments": json.dumps(
                    {
                        "hypotheses": [
                            {
                                "statement": "goal",
                                "node_id": "goal",
                                "is_goal": True,
                                "target_metric": 0.75,
                            }
                        ]
                    }
                ),
            }
        else:
            calls.append("record")
            tool = {
                "name": "record_evidence",
                "arguments": json.dumps({"node_id": "goal", "success": 1.0}),
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "function": tool}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    assert result["goals_met"] is False
    assert result["steps_to_target"] == config.tool_budget
    # The goal absorbed nothing: every record was refused.
    events = [
        json.loads(line)
        for line in config.log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not [e for e in events if e.get("event_type") == "evidence_recorded"]
    assert [e for e in events if e.get("event_type") == "tool_call" and not e.get("ok")]


@pytest.mark.integration
def test_an_unreachable_graph_does_not_end_the_run(tmp_path: Path) -> None:
    """A wiring mistake is recoverable; ending the episode makes it permanent.

    An agent that wires its premises under the goal produces a graph in which
    nothing is dispatchable. Reported as an empty frontier, that ended a real
    episode at step zero and the gate then scored it censored to the full budget
    — the worst possible outcome for a mistake the agent could have fixed in one
    turn had it been told.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    calls: list[str] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        if not calls:
            calls.append("create")
            tool = {
                "name": "create_hypotheses",
                "arguments": json.dumps(
                    {
                        "hypotheses": [
                            {"statement": "goal", "node_id": "goal", "is_goal": True},
                            {
                                "statement": "a=v0",
                                "node_id": "a_v0",
                                "parent_ids": ["goal"],
                                "edge_type": "DEPENDENCY",
                            },
                        ]
                    }
                ),
            }
        else:
            calls.append("ask")
            tool = {"name": "get_next_targets", "arguments": json.dumps({"count": 2})}
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "function": tool}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    # It kept going and was stopped by the idle guard, not by mistaking an
    # unreachable graph for a completed investigation.
    assert result["reason"] == "no_progress"
    assert calls.count("ask") >= 2
    events = [
        json.loads(line)
        for line in config.log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reasons = {e.get("reason") for e in events if e.get("event_type") == "target_selected"}
    assert "blocked_frontier" in reasons


@pytest.mark.integration
def test_a_malformed_create_says_what_shape_it_wanted(tmp_path: Path) -> None:
    """The native error names nothing the caller can act on.

    A list of bare strings raised "dictionary update sequence element #0 has
    length 1", which cost one run its entire hypothesis graph.
    """
    log_path = tmp_path / "bulk.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        out = _execute_tool(
            "create_hypotheses",
            {"hypotheses": ["component=v0", "component=v1"]},
            engine,
            [],
            config,
            logger,
        )
        assert "object" in out
        assert "statement" in out
    finally:
        engine.close()


@pytest.mark.integration
def test_a_single_object_is_accepted_where_a_list_was_asked_for(tmp_path: Path) -> None:
    """The intent is unambiguous, and bouncing it costs a whole turn.

    Everything else about the payload is validated strictly; this one shape is
    forgiven because rejecting it teaches the caller nothing it did not already
    mean.
    """
    log_path = tmp_path / "single.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        _execute_tool(
            "create_hypotheses",
            {"hypotheses": {"statement": "component=v0", "node_id": "c0"}},
            engine,
            [],
            config,
            logger,
        )
        assert engine._store.get_node("c0") is not None  # noqa: SLF001
    finally:
        engine.close()


@pytest.mark.integration
def test_settled_questions_do_not_end_the_run(tmp_path: Path) -> None:
    """`awaiting_composition` is an instruction, not an ending.

    Once goals stopped being dispatchable, the frontier legitimately empties the
    moment every question is answered — which is the point the agent still has
    to assemble those answers. Treating that DONE sentinel as "nothing testable"
    would abandon every arm-B run at exactly the step before the payoff.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    calls: list[str] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        if not calls:
            calls.append("create")
            tool = {
                "name": "create_hypotheses",
                "arguments": json.dumps(
                    {
                        "hypotheses": [
                            {
                                "statement": "a=v0",
                                "node_id": "a_v0",
                                "exclusion_group": "a",
                            }
                        ]
                    }
                ),
            }
        elif len(calls) == 1:
            calls.append("record")
            tool = {
                "name": "record_evidence",
                "arguments": json.dumps(
                    {"node_id": "a_v0", "success": 1.0, "count_next_targets": 0}
                ),
            }
        else:
            calls.append("ask")
            tool = {"name": "get_next_targets", "arguments": json.dumps({"count": 2})}
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "function": tool}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    # The run continued past the composition prompt and was stopped by the
    # idle-turn guard, not by mistaking a settled frontier for a dead one.
    assert result["reason"] == "no_progress"
    assert calls.count("ask") >= 2


@pytest.mark.integration
def test_a_model_that_never_experiments_cannot_loop_forever(tmp_path: Path) -> None:
    """The tool budget is counted in probes, so a non-probing turn never spends it.

    Unbounded, that is an infinite loop: on an unattended multi-hour gate run one
    stuck seed would block every seed after it.
    """
    from unittest.mock import patch

    from eval.runner.runner import MAX_IDLE_TURNS

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    calls = 0

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        # Always a read-only query: never an experiment, so `step` never moves.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{calls}",
                                "function": {
                                    "name": "get_goal_status",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with patch("eval.runner.runner._call_openai_api", side_effect=fake_api):
        result = run(config)

    assert result["reason"] == "no_progress"
    assert result["goals_met"] is False
    assert calls <= MAX_IDLE_TURNS + 1


# -- criterion-4 action taxonomy ---------------------------------------------


@pytest.mark.unit
def test_every_agent_turn_lands_in_exactly_one_bucket() -> None:
    """The taxonomy is only a measurement if it is total over the tool surface.

    A tool nobody classified would silently drop out of the contingency table and
    bias the very independence test it feeds.
    """
    from eval.runner.runner import ACTION_TAXONOMY, _classify_action

    for tool in get_tools_for_arm(ARM_B) + get_tools_for_arm(ARM_A):
        assert _classify_action(tool["function"]["name"], {}, None) in ACTION_TAXONOMY


@pytest.mark.unit
def test_asking_for_work_while_holding_work_is_abandonment(tmp_path: Path) -> None:
    """Not otherwise observable, and it is exactly the behaviour that loses probes."""
    from eval.runner.runner import ACTION_ABANDON, ACTION_CONTEXT, ACTION_REPLAN, _classify_action

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": "a", "node_id": "n1"}])
        assert _classify_action("get_next_targets", {}, engine) == ACTION_REPLAN

        engine.get_next_targets()
        assert _classify_action("get_next_targets", {}, engine) == ACTION_ABANDON
        # A peek claims nothing, so it abandons nothing.
        assert _classify_action("get_next_targets", {"dry_run": True}, engine) == ACTION_CONTEXT
    finally:
        engine.close()


@pytest.mark.integration
def test_the_status_context_follows_the_belief_state(tmp_path: Path) -> None:
    """The conditioning variable of criterion 4 must actually vary with the statuses."""
    from eval.runner.runner import _status_context
    from hypotree.models.status import Status

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": "a", "node_id": "n1"}])
        assert _status_context(engine) == "open"
        engine.update_status("n1", Status.NEEDS_REVISION, reason="under review")
        assert _status_context(engine) == "revision"
    finally:
        engine.close()


@pytest.mark.integration
def test_each_tool_call_is_classified_before_it_runs(tmp_path: Path) -> None:
    """Classified in the state the agent decided in, not the one its call produced."""
    log_path = tmp_path / "actions.jsonl"
    logger = RunLogger(log_path, seed=1201, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        _execute_tool(
            "create_hypotheses",
            {"hypotheses": [{"statement": "a", "node_id": "n1"}]},
            engine,
            [],
            config,
            logger,
        )
        _execute_tool("get_goal_status", {}, engine, [], config, logger)

        actions = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").strip().split("\n")
            if json.loads(line)["event_type"] == "agent_action"
        ]
        assert [a["action"] for a in actions] == ["REPLAN", "REQUEST_CONTEXT"]
        assert {a["status_context"] for a in actions} == {"open"}
    finally:
        engine.close()


# -- fused dispatch ----------------------------------------------------------


@pytest.mark.unit
def test_a_fused_done_ends_the_run_like_a_standalone_one() -> None:
    """A DONE that only get_next_targets was watched for would be missed entirely,
    and the run would burn its whole budget after the search was already over."""
    from eval.runner.runner import _dispatch_stop_reason

    standalone = json.dumps([{"status": "DONE", "reason": "all_goals_met"}])
    fused = json.dumps(
        {"id": "n1", "next_targets": [{"status": "DONE", "reason": "all_goals_met"}]}
    )
    assert _dispatch_stop_reason(standalone) == "believes_goals_met"
    assert _dispatch_stop_reason(fused) == "believes_goals_met"


@pytest.mark.unit
def test_a_fused_instruction_is_not_an_ending() -> None:
    """`awaiting_composition` is the step before the payoff, in either shape."""
    from eval.runner.runner import _dispatch_stop_reason

    for payload in (
        json.dumps([{"status": "DONE", "reason": "awaiting_composition"}]),
        json.dumps({"next_targets": [{"status": "DONE", "reason": "awaiting_substitution"}]}),
        json.dumps({"id": "n1"}),
        json.dumps([{"status": "SELECTED", "node_id": "n1"}]),
        "not json at all",
    ):
        assert _dispatch_stop_reason(payload) is None


@pytest.mark.integration
def test_recording_hands_back_the_next_targets(tmp_path: Path) -> None:
    """The whole point of the fusion: one round-trip where there were two."""
    log_path = tmp_path / "fused.jsonl"
    logger = RunLogger(log_path, seed=1201, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        _execute_tool(
            "create_hypotheses",
            {
                "hypotheses": [
                    {"statement": "a", "node_id": "n1"},
                    {"statement": "b", "node_id": "n2"},
                ]
            },
            engine,
            [],
            config,
            logger,
        )
        out = _execute_tool(
            "record_evidence",
            {"node_id": "n1", "success": 1.0, "count_next_targets": 1},
            engine,
            [],
            config,
            logger,
        )
        payload = json.loads(out)
        assert payload["next_targets"][0]["node_id"] == "n2"

        events = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")
        ]
        # A fused dispatch is logged exactly like a standalone one, or half the
        # dispatches in a run become invisible to the analysis.
        dispatches = [e for e in events if e["event_type"] == "target_selected"]
        assert [d["node_id"] for d in dispatches] == ["n2"]
        # And the record says whether it answered a dispatch.
        records = [e for e in events if e["event_type"] == "evidence_recorded"]
        assert records[0]["claimed"] is False
    finally:
        engine.close()


# -- a context reset must not confiscate work ---------------------------------


def _scripted_api(turns: list[dict[str, object]]):
    """Return a fake chat-completions callable that replays `turns` in order."""
    calls = {"n": 0}

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        index = min(calls["n"], len(turns) - 1)
        calls["n"] += 1
        tool_calls = [
            {"id": f"call_{i}", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(turns[index].items())
        ]
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    return fake_api


@pytest.mark.integration
def test_a_reset_never_strands_a_probe_the_agent_is_about_to_record(tmp_path: Path) -> None:
    """The failure that cost a real run two probes on every episode that reset.

    The agent probes in a batch and records in the next turn, so there is always
    a window where results exist only in its context. A reset landing there
    destroys them: one gets re-probed as a duplicate, the other is silently
    retired by the exclusion inference before its own result is ever recorded.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1201)
    config = make_run_config(ARM_B, 1201, eval_dir, "test-run", llm_backend="openai")
    # A breakpoint of 1 puts a reset in the window on the very first batch.
    config = config.__class__(**{**config.__dict__, "tool_budget": 12, "session_breakpoints": (1,)})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    turns = [
        {
            "create_hypotheses": {
                "hypotheses": [
                    {"statement": "component=v0", "node_id": "c0", "exclusion_group": "component"},
                    {"statement": "method=v0", "node_id": "m0", "exclusion_group": "method"},
                ]
            }
        },
        {"get_next_targets": {"count": 2}},
        # Probe both in one turn — the batch pattern every real episode used.
        {
            "evaluate_config": {"config": "component=v0", "depth": 1},
        },
        {"record_evidence": {"node_id": "c0", "success": 1.0, "count_next_targets": 0}},
        {"get_goal_status": {}},
    ]

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=_scripted_api(turns)),
        patch(
            "eval.runner.runner.landscape_probe",
            return_value={"success": 1.0, "metrics": {"probe_mode": "premise"}},
        ),
    ):
        run(config)

    events = [
        json.loads(line)
        for line in (config.log_path).read_text(encoding="utf-8").strip().split("\n")
    ]
    # The probe must have actually landed, or the assertion below is vacuous.
    assert [e for e in events if e["event_type"] == "experiment"]
    carried = [e for e in events if e["event_type"] == "probes_carried"]
    assert carried == [], (
        f"a reset landed on an unrecorded result it could have waited out: {carried}"
    )
    # The reset must still have happened — deferring forever would suppress it.
    assert any(e["event_type"] == "session_reset" for e in events)


@pytest.mark.integration
def test_a_reset_that_cannot_be_deferred_carries_the_result_across(tmp_path: Path) -> None:
    """Deferral is bounded, so the summary is the backstop.

    An agent that keeps probing without recording exhausts the deferral budget by
    design — otherwise it could suppress resets indefinitely. What it must not do
    is lose the answers, which arm F keeps by construction.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1201)
    config = make_run_config(ARM_B, 1201, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 12, "session_breakpoints": (1,)})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    # Probe, and never record: the deferral budget must run out and fire.
    turns = [
        {"create_hypotheses": {"hypotheses": [{"statement": "component=v0", "node_id": "c0"}]}},
        {"evaluate_config": {"config": "component=v0", "depth": 1}},
        {"get_goal_status": {}},
    ]

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=_scripted_api(turns)),
        patch(
            "eval.runner.runner.landscape_probe",
            return_value={"success": 1.0, "metrics": {"probe_mode": "premise"}},
        ),
    ):
        run(config)

    events = [
        json.loads(line)
        for line in (config.log_path).read_text(encoding="utf-8").strip().split("\n")
    ]
    resets = [e for e in events if e["event_type"] == "session_reset"]
    assert resets, "a bounded deferral must eventually fire"
    # And what it could not wait out was handed forward rather than destroyed.
    carried = [e for e in events if e["event_type"] == "probes_carried"]
    assert carried and "component=v0" in carried[0]["configs"]


@pytest.mark.unit
def test_the_carried_summary_states_the_result_not_just_the_config(tmp_path: Path) -> None:
    """Parity with arm F, which is handed every raw probe by construction.

    Carrying only "you probed this" would be worse than useless: the agent would
    know it had spent a probe and not what the probe said, so it would re-probe
    anyway. The number is the thing a reset destroys.
    """
    from eval.runner.runner import _compact_summary
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(":memory:", rng_seed=1)
    try:
        engine.create_hypotheses([{"statement": "component=v0", "node_id": "c0"}])
        transcript = [{"config": "component=v0", "depth": 1, "success": 1.0}]

        summary = _compact_summary(engine, [], ARM_B, transcript)
        assert "UNRECORDED" in summary
        assert "component=v0" in summary
        assert "1.0" in summary

        # Once folded in, it is the belief state's job and must not be repeated.
        engine.record_evidence("c0", LogicalEvidence(success=1.0, depth=1))
        assert "UNRECORDED" not in _compact_summary(engine, [], ARM_B, transcript)
    finally:
        engine.close()


@pytest.mark.integration
def test_a_probe_that_never_reached_the_oracle_costs_no_step(tmp_path: Path) -> None:
    """A transport failure is not an experiment.

    Charging one inflates every arm's step count whenever the server hiccups —
    and because the step counter is also the clock the session breakpoints run
    on, it advances that clock while leaving the transcript empty, so a reset
    then fires over a probe that never happened.
    """
    from unittest.mock import patch

    eval_dir = _setup_eval_env(tmp_path, seed=1201)
    config = make_run_config(ARM_B, 1201, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 6, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    turns = [
        {"create_hypotheses": {"hypotheses": [{"statement": "component=v0", "node_id": "c0"}]}},
        {"evaluate_config": {"config": "component=v0", "depth": 1}},
        {"get_goal_status": {}},
    ]

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=_scripted_api(turns)),
        patch("eval.runner.runner.landscape_probe", side_effect=OSError("connection refused")),
    ):
        result = run(config)

    # The unreachable oracle must terminate the run, not spin it to the budget.
    assert result["reason"] == "no_progress"
    events = [
        json.loads(line)
        for line in (config.log_path).read_text(encoding="utf-8").strip().split("\n")
    ]
    assert not [e for e in events if e["event_type"] == "experiment"]
    assert result["steps_to_target"] == config.tool_budget  # censored, never met


@pytest.mark.integration
def test_a_result_obtained_before_the_prune_is_not_a_re_execution() -> None:
    """Recording a measurement you already paid for is not wasted budget.

    The batching protocol makes this routine: probe two configurations, record
    the first, and its cascade can prune the node the second belongs to. The
    honest response is to record the second anyway — discarding a finished
    experiment is strictly worse than filing it. Counting that as a redundant
    re-execution flipped a run's hard gate on a single event.
    """
    from datetime import timedelta

    from eval.runner.runner import _probe_postdates_prune

    pruned_at = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    transcript = [
        {"config": "x=1", "depth": 2, "success": 0.0, "at": pruned_at - timedelta(seconds=5)}
    ]

    assert _probe_postdates_prune(transcript, "x=1", pruned_at) is False


@pytest.mark.integration
def test_probing_a_branch_that_was_already_dead_still_counts() -> None:
    """The behaviour the hard gate exists to catch must keep being caught."""
    from datetime import timedelta

    from eval.runner.runner import _probe_postdates_prune

    pruned_at = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    transcript = [
        {"config": "x=1", "depth": 2, "success": 0.0, "at": pruned_at + timedelta(seconds=5)}
    ]

    assert _probe_postdates_prune(transcript, "x=1", pruned_at) is True


@pytest.mark.integration
def test_a_result_with_no_experiment_behind_it_counts_as_re_execution() -> None:
    """Unknown timings fail open.

    A pruned node with no matching probe in the transcript is the shape a
    fabricated result takes, and that is exactly what the gate is watching for.
    """
    from eval.runner.runner import _probe_postdates_prune

    pruned_at = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

    assert _probe_postdates_prune([], "x=1", pruned_at) is True
    assert _probe_postdates_prune([{"config": "x=1"}], "x=1", pruned_at) is True


@pytest.mark.integration
def test_double_encoded_hypotheses_are_parsed_not_rejected(tmp_path: Path) -> None:
    """A model that quotes its JSON argument should not lose a turn over it.

    Two of thirty episodes in run G opened `create_hypotheses` with the list
    carried as a *string*. The shape is unambiguous, and at ~13.6k prompt tokens
    a bounced turn is the most expensive possible way to say "add brackets".
    """
    log_path = tmp_path / "coerce.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    _execute_tool(
        "create_hypotheses",
        {"hypotheses": '[{"statement": "component=v0", "node_id": "comp_v0"}]'},
        engine,
        [],
        config,
        logger,
    )

    assert engine._store.get_node("comp_v0") is not None
    engine.close()


@pytest.mark.integration
def test_an_unparseable_hypotheses_argument_still_names_the_shape(tmp_path: Path) -> None:
    """Coercion must not swallow genuine garbage into a confusing failure."""
    log_path = tmp_path / "bad.jsonl"
    logger = RunLogger(log_path, seed=1001, arm=ARM_B)
    config = _bare_config(tmp_path, log_path)
    engine = HypoTreeEngine(":memory:", rng_seed=1)

    result = _execute_tool(
        "create_hypotheses", {"hypotheses": "not json at all"}, engine, [], config, logger
    )

    assert "must be a list of objects" in result
    engine.close()


@pytest.mark.integration
def test_a_win_lets_the_agent_file_what_it_is_still_holding(tmp_path: Path) -> None:
    """The environment decides the win mid-protocol; the record has to land.

    The agent probes and reports in separate turns, so at the instant a probe
    clears the target it is holding an unreported result. Ending the episode
    there left the belief state inconsistent at exactly the moment it is
    measured: on the seeds where the swap that names a culprit is also the
    winning combination, the engine resolved the conflict and the scoreboard
    recorded it as unresolved. `step` is frozen, so the wind-down cannot move
    the headline metric.
    """
    from unittest.mock import patch

    from eval.runner.runner import _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    criteria = _load_win_criteria(config.landscape_path)
    winning_config = criteria["goal_config"]
    win_depth = criteria["min_confirm_depth"]

    def _call(name: str, args: dict) -> dict:
        return {
            "id": f"call_{name}",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    # Turn 1 creates a node and claims it; turn 2 probes the winning config
    # (which wins while the result is unreported); turn 3 is the wind-down, in
    # which the agent finally files it.
    turns = [
        [
            _call(
                "create_hypotheses",
                {"hypotheses": [{"statement": winning_config, "node_id": "win"}]},
            ),
            _call("get_next_targets", {"count": 1}),
        ],
        [_call("evaluate_config", {"config": winning_config, "depth": win_depth})],
        [
            _call(
                "record_evidence",
                {"node_id": "win", "success": 0.95, "depth": win_depth},
            )
        ],
    ]
    issued: list[int] = []

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        index = min(len(issued), len(turns) - 1)
        issued.append(index)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "tool_calls": turns[index]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=fake_api),
        patch("eval.runner.runner.landscape_probe", return_value={"success": 0.95, "metrics": {}}),
    ):
        result = run(config)

    assert result["goals_met"] is True
    # The win landed on the first (and only) probe, and the wind-down turn ran no
    # experiment, so the headline metric is untouched.
    assert result["steps_to_target"] == 1

    events = [json.loads(line) for line in config.log_path.read_text().strip().split("\n")]
    recorded = [e for e in events if e["event_type"] == "evidence_recorded"]
    assert [e["node_id"] for e in recorded] == ["win"]


@pytest.mark.integration
def test_the_wind_down_is_bounded(tmp_path: Path) -> None:
    """An agent that will not file its results cannot hold the episode open."""
    from unittest.mock import patch

    from eval.runner.runner import MAX_WINDDOWN_TURNS, _load_win_criteria

    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="openai")
    config = config.__class__(**{**config.__dict__, "tool_budget": 60, "session_breakpoints": ()})

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    criteria = _load_win_criteria(config.landscape_path)
    calls = 0

    def fake_api(*_args: object, **_kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            tool_calls = [
                {
                    "id": "c1",
                    "function": {
                        "name": "create_hypotheses",
                        "arguments": json.dumps(
                            {"hypotheses": [{"statement": "x", "node_id": "n1"}]}
                        ),
                    },
                },
                {
                    "id": "c2",
                    "function": {
                        "name": "get_next_targets",
                        "arguments": json.dumps({"count": 1}),
                    },
                },
            ]
        else:
            # Never records; just keeps probing. Every probe after the win is
            # refused, so `step` stays put and only the wind-down bound ends it.
            tool_calls = [
                {
                    "id": f"c{calls}",
                    "function": {
                        "name": "evaluate_config",
                        "arguments": json.dumps(
                            {
                                "config": criteria["goal_config"],
                                "depth": criteria["min_confirm_depth"],
                            }
                        ),
                    },
                }
            ]
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    with (
        patch("eval.runner.runner._call_openai_api", side_effect=fake_api),
        patch("eval.runner.runner.landscape_probe", return_value={"success": 0.95, "metrics": {}}),
    ):
        result = run(config)

    assert result["goals_met"] is True
    assert result["steps_to_target"] == 1
    # One setup turn, the winning turn, and at most MAX_WINDDOWN_TURNS more.
    assert calls <= 2 + MAX_WINDDOWN_TURNS


@pytest.mark.integration
def test_a_batch_of_results_is_reported_in_one_call(tmp_path: Path) -> None:
    """Reporting k results in one turn is the whole point of the batch shape."""
    eval_dir = _setup_eval_env(tmp_path, seed=1001)
    config = make_run_config(ARM_B, 1001, eval_dir, "test-run", llm_backend="mock")

    import os

    os.environ["XDG_DATA_HOME"] = str(tmp_path / "xdg")

    engine = HypoTreeEngine(tmp_path / "batch.db", rng_seed=1)
    logger = RunLogger(tmp_path / "batch.jsonl", seed=1001, arm=ARM_B)
    try:
        engine.create_hypotheses(
            [
                {"statement": "a=1", "node_id": "n1"},
                {"statement": "b=1", "node_id": "n2"},
            ]
        )
        out = _execute_tool(
            "record_evidence",
            {
                "results": [
                    {"node_id": "n1", "success": 1.0, "depth": 1},
                    {"node_id": "n2", "success": 0.0, "depth": 1},
                ],
                "count_next_targets": 0,
            },
            engine,
            [],
            config,
            logger,
            [],
        )
    finally:
        engine.close()

    payload = json.loads(out)
    assert [p["id"] for p in payload["recorded"]] == ["n1", "n2"]

    lines = (tmp_path / "batch.jsonl").read_text().strip().split("\n")
    events = [json.loads(line) for line in lines]
    recorded = [e for e in events if e["event_type"] == "evidence_recorded"]
    # Both results are instrumented individually, so a batch is analysed exactly
    # as a sequence of single calls would be.
    assert [e["node_id"] for e in recorded] == ["n1", "n2"]
