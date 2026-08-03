"""Headless agent runner — drives an LLM through the evaluation task.

Connects to the hypotree engine directly (in-process) and the landscape server
via HTTP. No GUI, no interactive terminal. The runner IS the agent harness.

Two LLM backends:
- ``mock``: deterministic simulated agent for testing (no network calls).
- ``openai``: calls an OpenAI-compatible chat completions API (Ollama, GLM, etc.).

The runner enforces:
- Tool budget (max experiments per task).
- Session resets at predefined breakpoints (context cleared, summary injected).
- JSONL logging of every event for analyse_gate.py.

Both arms share the landscape probe tool (``evaluate_config``). Arm A gets a
scratchpad tool; Arm B gets the full hypotree engine tool surface.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.environment.agent_eval_client import evaluate as landscape_probe
from eval.runner.config import (
    ALL_ARMS,
    ARM_A,
    ARM_B,
    ARM_F,
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_BASE_S,
    LLM_RETRY_MAX_S,
    LLM_TIMEOUT_S,
    EvalConfig,
    make_run_config,
    reset_eval_db,
    resolve_eval_db_path,
)
from hypotree.engine import (
    ClaimError,
    GoalEvidenceError,
    HypoTreeEngine,
    NodeNotFoundError,
    TargetResponse,
)
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status, posterior_mean

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# How many extra steps a session reset may be postponed while the agent still
# owes results for dispatches it has already made. Bounded so an agent that
# never reports cannot postpone every reset and escape the memory test entirely.
MAX_RESET_DEFERRAL = 3

# Consecutive LLM turns without a single experiment before the run is abandoned.
# The tool budget is denominated in probes, so a turn that dispatches none does
# not advance it — without this bound an agent that reasons, queries and never
# experiments keeps the harness in an unbounded loop, which on an unattended
# multi-hour gate run means one stuck seed blocks every seed after it.
MAX_IDLE_TURNS = 12

# Turns granted after a win so the agent can file results it already holds.
#
# The environment decides the win the moment a probe clears the target, but the
# agent is mid-protocol when that happens: it has run an experiment and not yet
# reported it. Cutting the episode there left the belief state inconsistent at
# the exact moment it is measured — and on the seeds where the swap that
# identifies a culprit *is* the winning combination, the conflict was resolved
# by the engine and scored as unresolved, because the record that proves it was
# never allowed to land. Two turns is enough to report a batch; the step counter
# is frozen and further experiments are refused, so this cannot flatter the
# headline metric.
MAX_WINDDOWN_TURNS = 2

# DONE reasons that are instructions rather than endings. The navigator returns
# a DONE sentinel whenever it has nothing to hand out, but "nothing to hand out"
# covers several states the agent is expected to act on and continue from: it is
# holding every remaining node under an unreported lease, a conflict is waiting
# for a diagnostic swap, every question is answered and the answers have still to
# be composed, the graph is wired so nothing is reachable, or the goal is wired
# to nothing and so can never be satisfied. Ending a run on any of these would
# score ordinary mid-task progress — or a fixable modelling mistake — as a dead
# end.
_CONTINUE_REASONS = frozenset(
    {
        "awaiting_evidence",
        "awaiting_substitution",
        "awaiting_composition",
        "blocked_frontier",
        "unreachable_goal",
    }
)


def _probe_result(result_str: str) -> dict[str, Any] | None:
    """The measurement a probe returned, or None if it never reached the oracle.

    A transport failure is not an experiment. Charging one against the budget
    inflates every arm's step count whenever the server hiccups, and — because
    the step counter is also the clock the session breakpoints run on — it
    advances that clock while leaving the transcript empty, so a reset then
    fires on a probe that never happened.
    """
    try:
        payload = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None
    return payload


def _unrecorded_probes(
    engine: HypoTreeEngine, transcript: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Probe results the agent is holding that the belief state does not have.

    The exact thing a context reset destroys, and the reason it must be measured
    rather than approximated by "does the agent hold a lease". A lease with
    nothing probed against it costs nothing to reset — the node simply returns
    to the frontier. A *result* that has not been folded in is gone for good:
    the environment was paid for it and nobody will ever see the answer again.

    Approximating it by live claims was correct only while a dispatch was the
    last thing that happened before a probe. Once a record could carry its own
    dispatch, the agent holds a claim continuously — the steady state, not a
    warning — so the check read as "never safe", the deferral budget burned down
    in three turns and the reset then fired *regardless*, landing in exactly the
    window it existed to avoid. It cost two probes on every episode that reset
    mid-batch: one re-probed as a duplicate, one silently retired by the
    exclusion inference before its own result was ever recorded.

    A config counts as recorded once the hypothesis stating it carries evidence.
    A probe of something the agent never modelled counts as unrecorded, which is
    right: that result has nowhere to live either.
    """
    if not transcript:
        return []
    recorded = {node.statement for node in engine._store.get_all_nodes() if node.evidence_count > 0}
    seen: set[str] = set()
    pending: list[dict[str, Any]] = []
    for entry in transcript:
        config = str(entry.get("config", ""))
        if config in recorded or config in seen:
            continue
        seen.add(config)
        pending.append(entry)
    return pending


def _reset_is_safe(
    engine: HypoTreeEngine, arm: str, transcript: list[dict[str, Any]] | None
) -> bool:
    """Whether wiping the agent's context right now would destroy unreported work.

    Only the belief-state arm can be mid-flight: it probes and reports in
    separate turns, so a reset landing between the two erases probes that were
    paid for. The transcript arms keep every probe automatically, so a reset
    costs them nothing and is always safe — firing resets on a schedule that
    penalises one protocol and not the other would measure turn alignment rather
    than memory.
    """
    if arm != ARM_B:
        return True
    return not _unrecorded_probes(engine, transcript)


def _status_snapshot(engine: Any) -> dict[str, tuple[Any, str]]:
    """Every node's current status *and* the reason it was last set for.

    Status alone loses whole inferences. Ruling a substitute out after a sub-par
    swap first retracts its exclusion (EXHAUSTED->UNTESTED) and then re-settles
    it (UNTESTED->EXHAUSTED) at the same instant, so a before/after comparison
    of statuses sees no change and the mechanism reports zero forever. The
    reason is what actually changed, so the reason is what has to be watched.

    Two queries regardless of workspace size: the per-node history accessor in a
    loop would be one query per node per recorded result.
    """
    latest: dict[str, str] = {}
    for row in engine._store.get_all_status_history():
        latest[str(row["node_id"])] = str(row["reason"] or "")
    return {n.id: (n.status, latest.get(n.id, "")) for n in engine._store.get_all_nodes()}


def _last_probe_depth(transcript: list[dict[str, Any]] | None, statement: str) -> int:
    """Depth the given configuration was most recently probed at, else 0.

    Read from the run transcript rather than cached in module state so it can
    never bleed between runs sharing a process.
    """
    for entry in reversed(transcript or []):
        if entry.get("config") == statement:
            return int(entry.get("depth") or 0)
    return 0


def _probe_postdates_prune(
    transcript: list[dict[str, Any]] | None,
    statement: str,
    pruned_at: datetime | None,
) -> bool:
    """Whether the experiment behind this result was run after the branch died.

    Only then has budget actually been wasted. A result obtained before the
    prune is a measurement already paid for, and discarding it would be strictly
    worse than recording it — so it must not be scored as redundant work.

    Unknown timings fail *open* (True): a node pruned with no matching probe in
    the transcript is the shape a genuine re-execution takes when the agent
    fabricates a result, and the hard gate exists to catch exactly that.
    """
    if pruned_at is None:
        return True
    for entry in reversed(transcript or []):
        if entry.get("config") == statement:
            probed_at = entry.get("at")
            return not isinstance(probed_at, datetime) or probed_at > pruned_at
    return True


# -- Arm-agnostic landscape win condition -------------------------------------
# The runner must detect a "goal met" for BOTH arms — not just Arm B (which
# has the hypotree goal_status tool). Arm A only has evaluate_config, so the
# win condition must be based on the landscape ground truth: an evaluate_config
# call that returns success >= the goal's target_metric at a depth that passes
# the ambush check constitutes a win. This makes the gate comparison fair.


def _load_win_criteria(landscape_path: Path) -> dict[str, Any]:
    """Load the win rules for this landscape.

    Returns the declared ``target_metric``, the ``axes`` a config must all
    assign for the probe to count as a complete candidate, the
    ``min_confirm_depth`` a result must be confirmed at, and the environment's
    ``evidence_regime``.
    """
    data = json.loads(landscape_path.read_text(encoding="utf-8"))
    goal_id = data.get("goal_node_id", "")
    nodes = data.get("nodes", [])

    goal_node = next((n for n in nodes if n["id"] == goal_id), None)
    # The win threshold is the goal's DECLARED target_metric (what the agent is
    # asked to clear), not the goal node's hidden true_success. A config that
    # meets the stated target is a legitimate win even if it isn't the global
    # maximum; conflating the two silently raised the bar and rejected valid wins.
    target_metric = (goal_node or {}).get("target_metric")
    if target_metric is None:
        target_metric = data.get("node_truth", {}).get(goal_id, {}).get("true_success", 0.75)

    return {
        "goal_id": goal_id,
        "goal_config": data.get("winning_config", ""),
        "target_metric": target_metric,
        "axes": list(data.get("axes", [])),
        "min_confirm_depth": data.get("min_confirm_depth", 0),
        "evidence_regime": data.get("evidence_regime", "deterministic"),
    }


def _check_landscape_win(
    config_str: str,
    depth: int,
    success: float,
    criteria: dict[str, Any],
) -> bool:
    """Check whether an evaluate_config result constitutes a goal-achieving win.

    Three conditions, applied uniformly to every config — there is no
    special-cased trap configuration any more:

    1. **Complete candidate.** The probe must assign a value to every axis. A
       premise probe confirms a single component and legitimately scores full
       marks, but confirming one part of an answer is not achieving the goal;
       without this rule a single premise probe would end the run.
    2. **Clears the declared target.**
    3. **Confirmed at depth.** Shallow probing cannot distinguish the real answer
       from the planted decoy, so a result only counts once it has been confirmed
       at ``min_confirm_depth`` or deeper. This applies to every config equally,
       which is both fairer and far simpler than the previous per-config depth
       exception.
    """
    axes = criteria.get("axes") or []
    if axes:
        assigned = {
            token.split("=", 1)[0].strip() for token in config_str.split(";") if "=" in token
        }
        if not all(axis in assigned for axis in axes):
            return False

    if success < criteria["target_metric"]:
        return False

    return depth >= criteria.get("min_confirm_depth", 0)


def load_system_prompt(arm: str) -> str:
    """Load the system prompt markdown for the given arm."""
    path = _PROMPTS_DIR / f"system_prompt_arm_{arm.lower()}.md"
    return path.read_text(encoding="utf-8")


# -- Tool definitions (OpenAI function-calling format) -----------------------


def _tool_evaluate_config() -> dict[str, Any]:
    """The landscape probe tool — available in both arms."""
    return {
        "type": "function",
        "function": {
            "name": "evaluate_config",
            "description": "Probe the black-box landscape. Returns success in [0,1].",
            "parameters": {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "string",
                        "description": "Configuration string to probe",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Exploration depth (0-4)",
                        "default": 0,
                    },
                },
                "required": ["config"],
            },
        },
    }


def _tool_update_scratchpad() -> dict[str, Any]:
    """The scratchpad tool — Arm A only."""
    return {
        "type": "function",
        "function": {
            "name": "update_scratchpad",
            "description": (
                "Append to (or replace) your working notes — your ONLY memory across "
                "context resets. Anything not written here is lost at the next reset, "
                "including the task's configuration format and everything you have "
                "ruled out. Write after every informative probe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Markdown content for your notes",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "replace"],
                        "description": (
                            "append (default) adds an entry and keeps earlier notes; "
                            "replace overwrites the whole notebook."
                        ),
                    },
                },
                "required": ["content"],
            },
        },
    }


def _tool_create_hypotheses() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "create_hypotheses",
            "description": (
                "Add one or many hypothesis nodes. Pass a list of one to create a "
                "single hypothesis. Parents may be created by this same call in any "
                "order. ALWAYS set exclusion_group on nodes that are competing answers "
                "to the same question. The whole list is checked before anything is "
                "written, so a rejected call creates nothing and you can fix and resend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string"},
                                "exclusion_group": {
                                    "type": "string",
                                    "description": (
                                        "REQUIRED whenever this node competes with others "
                                        "to answer one question (e.g. which value an axis "
                                        "takes) — use the question's name. Confirming any "
                                        "member then settles all the others automatically, "
                                        "so you never spend a probe on a question that is "
                                        "already answered. Omitting it is the single most "
                                        "expensive mistake you can make here."
                                    ),
                                },
                                "node_id": {"type": "string"},
                                "parent_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "edge_type": {
                                    "type": "string",
                                    "enum": ["DEPENDENCY", "ALTERNATIVE", "REFINEMENT"],
                                    "default": "DEPENDENCY",
                                },
                                "is_goal": {"type": "boolean", "default": False},
                                "target_metric": {"type": "number"},
                                "if_exists": {
                                    "type": "string",
                                    "enum": ["error", "overwrite", "skip"],
                                    "default": "error",
                                },
                            },
                            "required": ["statement"],
                        },
                    },
                },
                "required": ["hypotheses"],
            },
        },
    }


def _tool_get_next_targets() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_next_targets",
            "description": (
                "Select the next best hypotheses to test and claim them. Returns a "
                "LIST of targets, each with node_id, claim_id and statement. Ask for "
                "count=2 and probe both before calling again — a claimed node is "
                "reserved for you, so anything you do not report on is work nobody "
                "else can do either. An entry carrying min_depth is under conflict "
                "review and must be probed at that depth or deeper. Returns a single "
                "DONE entry when there is nothing left to hand out. You rarely need "
                "this call on its own: record_evidence(count_next_targets=2) returns "
                "the same list and costs one round-trip instead of two."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "default": 2,
                        "description": "How many targets to claim at once.",
                    },
                    "dry_run": {"type": "boolean", "default": False},
                },
            },
        },
    }


def _tool_record_evidence() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "record_evidence",
            "description": (
                "Record one result, or every result from this turn at once. Updates "
                "posterior + transitions and consumes the claim_id when one is given. "
                "Record against the hypothesis whose statement you actually probed. "
                "Probed several configurations this turn? Pass them together in "
                "`results` — one call instead of one per probe. Leave "
                "count_next_targets at its default so your next targets come back with "
                "the result, which is one round-trip instead of two."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": (
                            "Several results at once, applied in the order given. Use "
                            "this whenever you ran more than one experiment this turn. "
                            "When present, the single-result fields below are ignored."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "success": {"type": "number"},
                                "depth": {"type": "integer", "default": 0},
                                "claim_id": {"type": "string"},
                                "notes": {"type": "string", "default": ""},
                            },
                            "required": ["node_id", "success"],
                        },
                    },
                    "node_id": {"type": "string"},
                    "success": {
                        "type": "number",
                        "description": "Normalized success in [0,1]",
                    },
                    "depth": {
                        "type": "integer",
                        "default": 0,
                        "description": (
                            "The depth you probed at. Pass the SAME depth you gave "
                            "evaluate_config — a confirmation obtained at a shallow "
                            "depth does not support a combination tested deeper."
                        ),
                    },
                    "claim_id": {
                        "type": "string",
                        "description": (
                            "The claim this result answers, exactly as get_next_targets "
                            "returned it. OPTIONAL — omit the field entirely for a probe "
                            "you chose yourself. Never invent or guess a value: an "
                            "unrecognised claim is rejected and the probe is wasted."
                        ),
                    },
                    "count_next_targets": {
                        "type": "integer",
                        "default": 2,
                        "description": (
                            "How many targets you want to be holding when this returns, "
                            "handed back under the `next_targets` key. It is a top-up, not "
                            "an addition: recording two results in a batch leaves you "
                            "holding two, not four. Set 0 only when you do not want more "
                            "work yet."
                        ),
                    },
                    "notes": {"type": "string", "default": ""},
                },
                "required": [],
            },
        },
    }


def _tool_get_goal_status() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_goal_status",
            "description": "Report goal progress: met count, frontier size, status breakdown.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_list_nodes() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_nodes",
            "description": "Filter/search/sort nodes. Use to review what you've tried.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {"type": "array", "items": {"type": "string"}},
                    "query_filter": {"type": "string"},
                    "order_by": {"type": "string", "default": "created_at"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        },
    }


def _tool_get_conflicts() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_conflicts",
            "description": (
                "List recorded conflicts: sets of assumptions that cannot ALL hold "
                "together, discovered when a combination built on them failed. Shows "
                "which members are already exonerated and which remain suspects."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_suggest_experiment() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "suggest_discriminating_experiment",
            "description": (
                "Given every combination ruled out so far, propose the next one worth "
                "testing: the closest alternative to the last failure that is not "
                "already known to be impossible. Call this after a combination fails."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }


def get_tools_for_arm(arm: str) -> list[dict[str, Any]]:
    """Return the tool definitions available to the given arm.

    Both baseline arms get the same two tools; they differ only in what the
    harness preserves for them across a context reset.
    """
    tools = [_tool_evaluate_config()]
    if arm in (ARM_A, ARM_F):
        tools.append(_tool_update_scratchpad())
    else:
        tools.extend(
            [
                _tool_create_hypotheses(),
                _tool_get_next_targets(),
                _tool_record_evidence(),
                _tool_get_goal_status(),
                _tool_list_nodes(),
                _tool_get_conflicts(),
                _tool_suggest_experiment(),
            ]
        )
    return tools


# -- Tool execution -----------------------------------------------------------


def _pin_regime(spec: dict[str, Any], config: EvalConfig, logger: RunLogger) -> dict[str, Any]:
    """Force every created node onto the environment's actual evidence regime.

    The regime is a property of the *environment*, not a modelling preference:
    this landscape is a deterministic oracle, so one probe of a configuration is
    the whole truth about it. Leaving the choice to the agent proved actively
    dangerous — a run in which the agent declared every node ``stochastic`` never
    invalidated, verified or exhausted a single node, because the stochastic path
    waits for a convergence gate that a one-shot oracle never closes. The entire
    revision machinery was silently switched off by one free-text field. Any
    override is logged rather than hidden.
    """
    pinned = dict(spec)
    requested = pinned.get("evidence_regime")
    if requested is not None and requested != config.evidence_regime:
        logger.log_regime_override(str(pinned.get("node_id") or ""), str(requested))
    pinned["evidence_regime"] = config.evidence_regime
    return pinned


def _dispatch_stop_reason(result_str: str) -> str | None:
    """Read a run-ending DONE out of a tool result, whichever tool produced it.

    None of the DONE reasons is a win. A win is only ever an environment result
    (see ``_check_landscape_win``) — the engine reporting "all goals met" is arm
    B's *belief*, and no baseline arm has any comparable way to declare itself
    finished. Scoring it as a win handed the treatment a shortcut the gate is
    supposed to measure.

    "empty_frontier" means nothing is testable and the run is over. The reasons
    in ``_CONTINUE_REASONS`` are instructions, not endings: report the results
    you are holding, or put the answers you have found together. Both are places
    an agent is mid-task, and ending the run there would score ordinary progress
    as a dead end (the idle-turn guard bounds the case where the agent never
    acts on the instruction).
    """
    try:
        payload = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    # A bare list is a dispatch call. A dict is a record: a single result carries
    # the fused dispatch under next_targets, a batch wraps the per-result
    # payloads under `recorded` and the dispatch rides on the last of them,
    # because the top-up runs once after the whole batch has landed.
    if isinstance(payload, list):
        batch = payload
    elif isinstance(payload.get("recorded"), list) and payload["recorded"]:
        batch = payload["recorded"][-1].get("next_targets")
    else:
        batch = payload.get("next_targets")
    if not isinstance(batch, list) or not batch or not isinstance(batch[0], dict):
        return None
    if batch[0].get("status") != "DONE":
        return None
    reason = batch[0].get("reason")
    if reason in _CONTINUE_REASONS:
        return None
    return "believes_goals_met" if reason == "all_goals_met" else "frontier_exhausted"


# The pre-registered action taxonomy for the status-utility test (criterion 4).
# Every agent turn lands in exactly one bucket.
ACTION_EXECUTE = "EXECUTE_EXPERIMENT"
ACTION_REPLAN = "REPLAN"
ACTION_ABANDON = "ABANDON_BRANCH"
ACTION_CONTEXT = "REQUEST_CONTEXT"
ACTION_NOOP = "NO_OP"

ACTION_TAXONOMY = (ACTION_EXECUTE, ACTION_REPLAN, ACTION_ABANDON, ACTION_CONTEXT, ACTION_NOOP)

# Tools whose call is unambiguously one kind of action regardless of state.
_ACTION_BY_TOOL = {
    "evaluate_config": ACTION_EXECUTE,
    "record_evidence": ACTION_EXECUTE,
    "create_hypotheses": ACTION_REPLAN,
    "suggest_discriminating_experiment": ACTION_REPLAN,
    "update_scratchpad": ACTION_REPLAN,
    "get_goal_status": ACTION_CONTEXT,
    "list_nodes": ACTION_CONTEXT,
    "get_conflicts": ACTION_CONTEXT,
}


def _classify_action(tool_name: str, args: dict[str, Any], engine: HypoTreeEngine | None) -> str:
    """Bucket one agent turn into the pre-registered action taxonomy.

    Only ``get_next_targets`` is state-dependent, and deliberately so: asking to
    be handed new work while still holding leases on work already dispatched is
    the agent walking away from that work, which is precisely ABANDON_BRANCH and
    is not otherwise observable. A peek (``dry_run``) claims nothing and is a
    request for context, not a dispatch.
    """
    if tool_name in _ACTION_BY_TOOL:
        return _ACTION_BY_TOOL[tool_name]
    if tool_name == "get_next_targets":
        if args.get("dry_run"):
            return ACTION_CONTEXT
        if engine is not None and engine.get_active_claims():
            return ACTION_ABANDON
        return ACTION_REPLAN
    return ACTION_CONTEXT


def _status_context(engine: HypoTreeEngine | None) -> str:
    """Which belief-state regime the agent is acting in.

    The conditioning variable of criterion 4: whether the richer statuses are
    live at all. "revision" means at least one hypothesis is BLOCKED or
    NEEDS_REVISION — the situations the extra statuses exist to represent —
    and "open" means the belief state is expressible with UNTESTED alone. If the
    action distribution is the same in both, the extra statuses changed no
    behaviour and the pre-registered response is to collapse them.

    Read straight off the status counts rather than through get_goal_status,
    which rebuilds the graph and recomputes the frontier: this runs before every
    single tool call, and none of that work bears on the answer.
    """
    if engine is None:
        return "open"
    counts = engine._store.count_nodes_by_status()
    revising = counts.get(Status.BLOCKED.value, 0) + counts.get(Status.NEEDS_REVISION.value, 0)
    return "revision" if revising else "open"


def _log_targets(engine: HypoTreeEngine, logger: RunLogger, targets: list[TargetResponse]) -> None:
    """Record every dispatch, however it was requested.

    Shared by the standalone dispatch call and by the one fused into
    record_evidence, so fusing the two saves the agent a round-trip without
    making half the dispatches invisible to the analysis.

    The exclusion group travels with the dispatch so the analysis can tell
    whether a batch asked one question several times over — two competing
    answers dispatched together means the first result cannot retire the second,
    which is the exclusion inference being paid for and not used. The DONE reason
    travels for a blunter reason: it is the difference between "finished",
    "report what you hold", "compose the answers" and "your graph is
    unreachable", and it was previously visible only inside a truncated result
    blob.
    """
    for target in targets:
        node = engine._store.get_node(target.node_id) if target.node_id else None
        logger.log_target_selected(
            target.node_id,
            target.claim_id,
            target.status,
            exclusion_group=node.exclusion_group if node else None,
            reason=target.reason or None,
        )


def _record_one_result(
    item: dict[str, Any],
    engine: HypoTreeEngine,
    logger: RunLogger,
    transcript: list[dict[str, Any]] | None,
    *,
    count_next: int,
) -> dict[str, Any]:
    """Record one result and log every side effect it caused.

    Split out of the tool handler so a batch of results is instrumented exactly
    as a sequence of single calls would be: the belief-revision events a result
    triggers are attributed to that result, not to whichever one happened to be
    last in the batch.
    """
    node_id = item["node_id"]
    success = item["success"]
    claim_id = item.get("claim_id")
    notes = item.get("notes", "")
    prior = engine._store.get_node(node_id)
    # Depth of the test that produced this result. The agent may state it;
    # if it does not, inherit the depth its own configuration was last
    # probed at. The harness already saw that number, and making the agent
    # restate it turns bookkeeping into a reasoning step it can silently get
    # wrong — a wrong depth quietly disables the depth-aware blame machinery.
    depth = int(item.get("depth") or 0)
    if not depth and prior is not None:
        depth = _last_probe_depth(transcript, prior.statement)

    # Hard gate signal: spending an experiment on a branch that was already
    # dead. The probe has to post-date the prune for that to be true —
    # recording a result obtained *before* the branch died is not waste, it
    # is the only sensible thing to do with a measurement already paid for,
    # and the batching protocol makes it routine: probe two configs, record
    # the first, and its cascade can prune the node the second belongs to.
    # Counting that flipped a GO to an ITERATE on a single event.
    if (
        prior is not None
        and prior.status == Status.PRUNED
        and _probe_postdates_prune(transcript, prior.statement, prior.pruned_at)
    ):
        logger.log_pruned_reexecution(node_id)
    old_status = prior.status.value if prior is not None else None

    # Snapshot statuses so we can surface the transitions the engine performs
    # internally (cascading prune of descendants + upstream propagation).
    before = _status_snapshot(engine)
    conflicts_before = {c["id"] for c in engine._store.get_nogoods()}
    resolved_before = {c["id"] for c in engine._store.get_nogoods() if c["resolved_at"] is not None}

    evidence = LogicalEvidence(success=success, depth=depth, notes=notes)
    outcome = engine.record_evidence(
        node_id, evidence, claim_id=claim_id, count_next_targets=count_next
    )
    node = outcome.node

    # Conflicts recorded / narrowed by this observation. These are the
    # belief-revision events that a per-node status simply cannot express.
    for conflict in engine._store.get_nogoods():
        if conflict["id"] not in conflicts_before:
            logger.log_conflict_recorded(conflict["source_node_id"], conflict["member_ids"])
        if (
            conflict["resolved_at"] is not None
            and conflict["id"] not in resolved_before
            and conflict["resolved_culprit_id"]
        ):
            logger.log_conflict_resolved(conflict["id"], str(conflict["resolved_culprit_id"]))

    after = _status_snapshot(engine)
    for node_id_after, (status_after, reason_after) in after.items():
        prior_entry = before.get(node_id_after)
        if prior_entry is None or prior_entry == (status_after, reason_after):
            continue
        # A transition on a node other than the evidence target can only
        # come from propagation (cascading prune / upstream revision).
        #
        # The engine's own reason is carried through rather than being
        # re-derived from the (old, new) pair: several distinct
        # mechanisms share a transition — UNTESTED->EXHAUSTED is the
        # exclusion inference, IN_PROGRESS->VERIFIED is either direct
        # evidence or deduction by elimination — so the pair alone
        # cannot say which fired, and the analysis was silently
        # reporting zero for mechanisms that were working.
        logger.log_status_transition(
            node_id_after,
            prior_entry[0].value,
            status_after.value,
            propagated=node_id_after != node_id,
            reason=reason_after,
        )

    logger.log_evidence_recorded(
        node_id,
        success,
        node.status.value,
        old_status=old_status,
        regime=node.evidence_regime,
        evidence_count=node.evidence_count,
        posterior_mean=posterior_mean(node.alpha, node.beta),
        # Whether this result answered a dispatch or a probe the agent chose
        # for itself. Without the distinction, "dispatches never reported"
        # was computed as targets minus *all* records and went negative,
        # because a combination the agent composes has no dispatch behind it.
        claimed=claim_id is not None,
    )
    _log_targets(engine, logger, outcome.next_targets)
    payload: dict[str, Any] = json.loads(node.model_dump_json())
    if outcome.next_targets:
        payload["next_targets"] = [t.model_dump(mode="json") for t in outcome.next_targets]
    return payload


def _execute_tool(
    tool_name: str,
    args: dict[str, Any],
    engine: HypoTreeEngine,
    scratchpad: list[str],
    config: EvalConfig,
    logger: RunLogger,
    transcript: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a tool call and return the result as a JSON string.

    Every call is logged, successful or not. Logging only failures — as this did
    originally — turns the tool histogram into an error histogram that reads like
    a usage census, which is worse than having no census at all: it silently
    reports the tools an agent struggled with as the tools it used.

    Any exception during execution is caught and returned as a JSON error
    object so the LLM can recover and retry — the run never crashes on a
    single bad tool call.
    """
    # Classified before the call, not after: the conditioning variable is the
    # belief state the agent *decided in*, and this very call is about to change
    # it.
    logger.log_agent_action(
        _classify_action(tool_name, args, engine),
        _status_context(engine),
        tool=tool_name,
    )
    try:
        result = _execute_tool_inner(
            tool_name, args, engine, scratchpad, config, logger, transcript
        )
    except GoalEvidenceError as e:
        # Counted separately from ordinary tool errors: it means the agent
        # probed something and had nowhere to put the answer, so a probe was
        # paid for and the belief state learned nothing from it. Silently
        # accepting these is what made the failure invisible before.
        logger.log_goal_evidence_refused(str(args.get("node_id", "")))
        logger.log_tool_call(tool_name, f"ERROR: {e}", ok=False)
        return json.dumps({"error": str(e), "tool": tool_name, "args": args})
    except Exception as e:
        logger.log_tool_call(tool_name, f"ERROR: {e}", ok=False)
        return json.dumps({"error": str(e), "tool": tool_name, "args": args})
    logger.log_tool_call(tool_name, result[:120], ok=True)
    return result


def _execute_tool_inner(
    tool_name: str,
    args: dict[str, Any],
    engine: HypoTreeEngine,
    scratchpad: list[str],
    config: EvalConfig,
    logger: RunLogger,
    transcript: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a tool call and return the result as a JSON string."""
    if tool_name == "evaluate_config":
        # A missing `config` used to surface as a bare KeyError whose whole
        # message was the string 'config' — unactionable for the agent, which
        # then had no idea what to send instead. State the contract.
        raw_config = args.get("config")
        if not isinstance(raw_config, str) or not raw_config.strip():
            raise ValueError(
                "`config` is required and must be a non-empty string, e.g. "
                '"component=v0" for a single-axis premise probe or '
                '"component=v0;method=v1;..." for a full combination. '
                "`depth` is optional and defaults to 0."
            )
        try:
            result = landscape_probe(
                raw_config,
                args.get("depth", 0),
                url=config.landscape_url,
            )
        except Exception as e:
            # Re-raised rather than logged here: the caller logs every failure
            # under the tool's real name. Logging a synthetic
            # "evaluate_config_ERROR" invented a tool row that never existed and
            # left the genuine call uncounted in the usage census.
            raise RuntimeError(f"landscape_probe FAILED: {e} (url={config.landscape_url})") from e
        logger.log_experiment(
            raw_config,
            args.get("depth", 0),
            result["success"],
            probe_mode=result.get("metrics", {}).get("probe_mode"),
        )
        # Every probe is appended to the run transcript regardless of arm. Arms
        # that auto-persist replay it verbatim across a context reset, which is
        # what makes the baseline comparison about *structure* rather than about
        # whether the agent remembered to take notes.
        if transcript is not None:
            transcript.append(
                {
                    "config": raw_config,
                    "depth": args.get("depth", 0),
                    "success": result["success"],
                    # When the oracle was actually consulted. Needed to tell a
                    # probe that was wasted on a dead branch from one that was
                    # merely *reported* after the branch died.
                    "at": datetime.now(timezone.utc),
                }
            )
        return json.dumps(result)

    elif tool_name == "update_scratchpad":
        # Append-or-replace notebook. Append is the default so the agent builds a
        # cumulative record instead of silently destroying earlier findings with
        # a partial rewrite — the baseline arm's memory is only as good as what
        # survives here, and it must be a fair steel-man of "just use notes".
        content = args["content"]
        mode = args.get("mode", "append")
        if mode == "replace":
            scratchpad.clear()
        scratchpad.append(content)
        total = sum(len(s) for s in scratchpad)
        logger.log_scratchpad_write(mode, len(scratchpad), total)
        return json.dumps({"status": "saved", "entries": len(scratchpad), "total_chars": total})

    elif tool_name == "create_hypotheses":
        raw = args.get("hypotheses")
        if isinstance(raw, str):
            # Double-encoded JSON: the model emitted the argument as a *string*
            # containing the list rather than as the list. Unambiguous, common,
            # and the only alternative is bouncing a turn over quoting.
            with contextlib.suppress(json.JSONDecodeError):
                raw = json.loads(raw)
        if isinstance(raw, dict):
            # A single object where a list was asked for is the one malformation
            # worth accepting rather than rejecting: the intent is unambiguous,
            # and bouncing it costs the agent a whole turn to add two brackets.
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError(
                "`hypotheses` must be a list of objects, one per hypothesis, e.g. "
                '[{"statement": "component=v0", "node_id": "comp_v0", '
                '"exclusion_group": "component"}]. Pass a list of one to create a '
                "single hypothesis."
            )
        specs = [_pin_regime(spec, config, logger) for spec in raw if isinstance(spec, dict)]
        if len(specs) != len(raw):
            raise ValueError(
                "every entry in `hypotheses` must be an object with at least a "
                '`statement`, e.g. {"statement": "component=v0"}. A list of plain '
                "strings is not enough."
            )
        results = engine.create_hypotheses(specs)
        for spec, created in zip(specs, results, strict=True):
            logger.log_node_created(
                created.node.id,
                created.node.statement,
                created.created,
                exclusion_group=created.node.exclusion_group,
                composed=bool(spec.get("parent_ids")),
            )
        return json.dumps([r.model_dump() for r in results], default=str)

    elif tool_name == "get_conflicts":
        return json.dumps(engine.get_conflicts(), default=str)

    elif tool_name == "suggest_discriminating_experiment":
        return json.dumps(engine.suggest_discriminating_experiment(), default=str)

    elif tool_name == "get_next_targets":
        targets = engine.get_next_targets(
            count=int(args.get("count", 2)),
            dry_run=args.get("dry_run", False),
        )
        _log_targets(engine, logger, targets)
        return json.dumps([t.model_dump(mode="json") for t in targets], default=str)

    elif tool_name == "record_evidence":
        # One call may carry several results. They are applied one at a time so
        # every instrumented side effect — conflicts opened and narrowed,
        # cascading prunes, upstream revisions — is attributed to the result that
        # actually caused it. Folding them into one before/after snapshot would
        # credit the whole batch with the last result's transitions.
        items = args.get("results") or [args]
        # Fusing the next dispatch into the record is the whole point of the
        # parameter: the two calls are one decision for a synchronous agent and
        # cost it a full model round-trip each. Topped up once, after every
        # result in the batch has landed.
        count_next = int(args.get("count_next_targets", 2) or 0)
        payloads: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        # Per-item isolation, matching `engine.record_results`. Looping the
        # single-result call without this let one bad node id abort the batch —
        # the results already applied stayed in the belief state while the whole
        # call was reported as an error, and every result after the bad one was
        # never attempted. Each of those is an experiment that has already been
        # paid for, which is exactly what the engine's batch contract exists to
        # protect and what the harness was quietly overriding.
        for index, item in enumerate(items):
            last = index == len(items) - 1
            try:
                payloads.append(
                    _record_one_result(
                        item,
                        engine,
                        logger,
                        transcript,
                        count_next=count_next if last else 0,
                    )
                )
            except GoalEvidenceError as exc:
                logger.log_goal_evidence_refused(str(item.get("node_id", "")))
                logger.log_record_rejected(str(item.get("node_id", "")), str(exc))
                failures.append({"node_id": item.get("node_id"), "error": str(exc)})
            except (NodeNotFoundError, ClaimError, KeyError) as exc:
                logger.log_record_rejected(str(item.get("node_id", "")), str(exc))
                failures.append({"node_id": item.get("node_id"), "error": str(exc)})
        if args.get("results"):
            return json.dumps({"recorded": payloads, "failed": failures}, default=str)
        if failures and not payloads:
            # A single result that was refused keeps raising, so the agent sees
            # the error in full rather than an empty success.
            raise ValueError(failures[0]["error"])
        return json.dumps(payloads[0], default=str)

    elif tool_name == "get_goal_status":
        result = engine.get_goal_status()
        return result.model_dump_json()

    elif tool_name == "list_nodes":
        table = engine.list_nodes(
            status_filter=args.get("status_filter"),
            query_filter=args.get("query_filter"),
            order_by=args.get("order_by", "created_at"),
            limit=args.get("limit", 20),
        )
        return table

    else:
        return json.dumps({"error": f"unknown tool: {tool_name}"})


# -- JSONL logging ------------------------------------------------------------


class RunLogger:
    """Append-only JSONL logger for one evaluation run.

    Every dispatched experiment, claim, status transition, and lifecycle event
    is logged here. analyse_gate.py ingests these logs to compute the gate
    metrics.
    """

    def __init__(self, log_path: Path, seed: int, arm: str) -> None:
        self.log_path = log_path
        self.seed = seed
        self.arm = arm
        self.step = 0
        # Configs already probed, so every experiment can be flagged as fresh or
        # repeated. A repeat of a deterministic oracle carries zero information,
        # so the duplicate rate is a direct measure of wasted budget.
        self._seen_configs: set[str] = set()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def _write(self, event_type: str, **kwargs: Any) -> None:
        entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "step": self.step,
            "seed": self.seed,
            "arm": self.arm,
            **kwargs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_run_start(self, config: EvalConfig) -> None:
        self._write(
            "run_start",
            run_id=config.run_id,
            workspace_id=config.workspace_id,
            tool_budget=config.tool_budget,
            session_breakpoints=list(config.session_breakpoints),
            llm_backend=config.llm_backend,
            llm_model=config.llm_model,
        )

    def log_run_end(self, reason: str, goals_met: bool, infra_failed: bool = False) -> None:
        """Close out an episode.

        ``infra_failed`` marks an episode that ended because the harness could
        not reach the inference server, not because the agent ran out of budget.
        Downstream scoring excludes those rather than censoring them at budget:
        counting a dropped HTTP connection as a failure to solve the task would
        charge one arm for the server's bad minute.
        """
        self._write("run_end", reason=reason, goals_met=goals_met, infra_failed=infra_failed)

    def log_llm_retry(self, attempt: int, error: str, delay_s: float) -> None:
        """Record a transient inference-server fault that is about to be retried.

        Logged even though the request usually succeeds on the next attempt, so
        a run whose numbers look strange can be checked against how much trouble
        the transport was having at the time.
        """
        self._write("llm_retry", attempt=attempt, error=error, delay_s=round(delay_s, 2))

    def log_session_reset(
        self,
        session_num: int,
        summary: str,
        *,
        frontier_size: int | None = None,
        status_breakdown: dict[str, int] | None = None,
    ) -> None:
        """Record a context reset plus the belief state carried across it."""
        self._write(
            "session_reset",
            session_num=session_num,
            summary_length=len(summary),
            frontier_size=frontier_size,
            status_breakdown=status_breakdown,
        )

    def log_experiment(
        self,
        config_str: str,
        depth: int,
        success: float,
        probe_mode: str | None = None,
    ) -> None:
        self.step += 1
        duplicate = config_str in self._seen_configs
        self._seen_configs.add(config_str)
        self._write(
            "experiment",
            config=config_str,
            depth=depth,
            success=success,
            probe_mode=probe_mode,
            # Whether this exact config was already probed earlier in the run.
            duplicate=duplicate,
            distinct_configs=len(self._seen_configs),
        )

    def log_node_created(
        self,
        node_id: str,
        statement: str,
        created: bool,
        exclusion_group: str | None = None,
        composed: bool = False,
    ) -> None:
        """Record a node creation, including whether it declared its question.

        ``exclusion_group`` adoption had to be reverse-engineered from status
        transitions last run; logging it directly makes the single biggest
        efficiency lever in the belief state measurable rather than inferred.

        ``composed`` marks a node built on top of other hypotheses. Such a node
        answers no question of its own, so it *cannot* declare an exclusion
        group, and counting it in the adoption denominator made the metric
        measure how many combinations an episode built rather than how
        disciplined it was — conflict episodes read 77% against 89% purely
        because they build more of them.
        """
        self._write(
            "node_created",
            node_id=node_id,
            statement=statement,
            created=created,
            exclusion_group=exclusion_group,
            composed=composed,
        )

    def log_conflict_recorded(self, source_node_id: str, member_ids: list[str]) -> None:
        """A failure whose cause is indeterminate: these cannot all hold."""
        self._write(
            "conflict_recorded",
            source_node_id=source_node_id,
            member_ids=member_ids,
            n_members=len(member_ids),
        )

    def log_conflict_resolved(self, nogood_id: int, culprit_id: str) -> None:
        """A recorded conflict narrowed to its single provable culprit."""
        self._write("conflict_resolved", nogood_id=nogood_id, culprit_id=culprit_id)

    def log_claims_released(self, node_ids: list[str]) -> None:
        """Leases handed back because the context that held them was wiped.

        Each released node is work the agent dispatched and never reported. The
        count is the size of the hole a context reset punched in the belief
        state, which is otherwise invisible.
        """
        self._write("claims_released", node_ids=node_ids, count=len(node_ids))

    def log_probes_carried(self, configs: list[str]) -> None:
        """Probe results a reset landed on top of, handed forward in the summary.

        Distinct from a released lease, which costs nothing: the node simply
        returns to the frontier. This is the *answer* the agent was holding —
        the environment was paid for it and the belief state had not seen it. It
        used to be destroyed, costing one duplicate re-probe and one question
        settled by inference before its own evidence ever arrived. Counted even
        though it is now preserved, because a rising count means the reset
        schedule keeps landing mid-batch and the deferral is doing all the work.
        """
        self._write("probes_carried", configs=configs, count=len(configs))

    def log_goal_evidence_refused(self, node_id: str) -> None:
        """A result the agent tried to file against an objective.

        Distinct from an ordinary tool error because of what it implies: the
        agent probed a configuration and had nowhere to record the answer, so the
        probe was paid for and nothing was learned from it. It also means the
        hypothesis actually under test is still sitting untested.
        """
        self._write("goal_evidence_refused", node_id=node_id)

    def log_record_rejected(self, node_id: str, error: str) -> None:
        """One result of a batch the engine refused, with the rest still applied.

        Per-item isolation is what stops a bad node id destroying results that
        were already paid for, but it also means the containing call succeeds —
        so without this the refusal appears nowhere: not in the tool census,
        which sees an ok call, and not in the evidence count, which never saw the
        result. A rejection that costs a probe has to be visible somewhere.
        """
        self._write("record_rejected", node_id=node_id, error=error[:200])

    def log_llm_call(
        self,
        duration_s: float,
        n_tool_calls: int,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Wall-clock, tool-call volume and token usage per LLM turn.

        The extra round-trips the belief-state arm pays per probe are a real
        cost that step counts do not capture; without this the overhead argument
        can only be asserted, never measured. Tokens matter for the same reason
        in the other direction: a compacted belief state should cost fewer
        prompt tokens than a growing flat transcript.
        """
        self._write(
            "llm_call",
            duration_s=round(duration_s, 3),
            n_tool_calls=n_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def log_target_selected(
        self,
        node_id: str | None,
        claim_id: str | None,
        status: str,
        exclusion_group: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._write(
            "target_selected",
            node_id=node_id,
            claim_id=claim_id,
            status=status,
            exclusion_group=exclusion_group,
            reason=reason,
        )

    def log_evidence_recorded(
        self,
        node_id: str,
        success: float,
        status: str,
        *,
        old_status: str | None = None,
        regime: str | None = None,
        evidence_count: int | None = None,
        posterior_mean: float | None = None,
        claimed: bool = False,
    ) -> None:
        """Record an evidence fold and the belief-state change it produced.

        old_status/regime were previously unlogged, which made it impossible to
        tell whether the belief state was actually updating or the node was
        silently sitting in a dead zone. They are mandatory diagnostics now.

        ``claimed`` says whether this answered a dispatch. Compositions the agent
        assembles itself are recorded without one, so counting all records
        against all dispatches measures nothing and goes negative as soon as the
        agent does the composing the engine asks it to do.
        """
        self._write(
            "evidence_recorded",
            node_id=node_id,
            success=success,
            node_status=status,
            old_status=old_status,
            new_status=status,
            status_changed=old_status is not None and old_status != status,
            regime=regime,
            evidence_count=evidence_count,
            posterior_mean=posterior_mean,
            claimed=claimed,
        )

    def log_agent_action(self, action: str, status_context: str, tool: str | None = None) -> None:
        """Classify one agent turn for the status-utility test (criterion 4).

        The question that criterion answers is whether the richer statuses earn
        their keep: if what the agent *does* is independent of whether anything
        is BLOCKED or NEEDS_REVISION, the extra machinery is decoration and the
        pre-registered response is to collapse it. Answering that needs the
        agent's action and the belief-state context it acted in, on the same row
        — neither is recoverable after the fact from a tool histogram, which is
        why the test defaulted to collapse for want of data rather than for want
        of an effect.
        """
        self._write("agent_action", action=action, status_context=status_context, tool=tool)

    def log_lease_event(self, kind: str, claim_ids: list[str], node_ids: list[str]) -> None:
        """Record a lease renewal or a voluntary release."""
        self._write("lease_event", kind=kind, claim_ids=claim_ids, node_ids=node_ids)

    def log_tool_call(self, tool_name: str, result_summary: str, ok: bool = True) -> None:
        """Record every tool invocation and whether it succeeded.

        ``ok`` separates "which tools did the agent use" from "which tools did
        the agent get wrong" — two questions that look identical in the logs
        unless the flag is there, and that point at opposite fixes.
        """
        self._write("tool_call", tool=tool_name, ok=ok, result_summary=result_summary)

    def log_scratchpad_write(self, mode: str, entries: int, total_chars: int) -> None:
        """Record a note-taking action by a baseline arm.

        Without this the single most important property of the baseline — whether
        it maintains its memory at all — is invisible in the logs. A previous
        gate was decided by exactly this and it took reverse-engineering a summary
        length to notice.
        """
        self._write("scratchpad_write", mode=mode, entries=entries, total_chars=total_chars)

    def log_regime_override(self, node_id: str, requested: str) -> None:
        """Record that an agent-requested evidence regime was overridden."""
        self._write("regime_override", node_id=node_id, requested=requested)

    def log_status_transition(
        self, node_id: str, old: str, new: str, propagated: bool = False, reason: str = ""
    ) -> None:
        self._write(
            "status_transition",
            node_id=node_id,
            old_status=old,
            new_status=new,
            propagated=propagated,
            reason=reason,
        )

    def log_pruned_reexecution(self, node_id: str) -> None:
        self._write("pruned_reexecution", node_id=node_id)


# -- Mock LLM agent -----------------------------------------------------------


class MockAgent:
    """Deterministic simulated agent for pipeline testing.

    The mock agent drives the same in-process tool surface as a real LLM, so
    ``evaluate_config`` still flows through the landscape probe (tests patch it
    to stay hermetic). Only the *decision* of which tool to call next is
    simulated — everything downstream exercises the real engine + logger.

    State machine for Arm B (hypotree):
        INIT → create_hypotheses (goal + nodes, one call) → TARGET →
        get_next_targets → EVAL → evaluate_config → RECORD → record_evidence
        (which hands back the next target in the same call) → EVAL ...

    State machine for Arm A (scratchpad):
        Loop: PROBE → evaluate_config → NOTE → update_scratchpad → PROBE ...
    """

    def __init__(self, config: EvalConfig) -> None:
        import numpy as np

        self._rng = np.random.default_rng(config.mock_seed)
        self._landscape = json.loads(config.landscape_path.read_text(encoding="utf-8"))
        self._arm = config.arm
        self._state = "INIT"

        self._pending_target_id: str | None = None
        self._pending_claim_id: str | None = None
        self._pending_config: str = ""
        self._pending_success: float = 0.0
        self._pending_depth: int = 0

    def get_tool_call(self, available_tools: list[str]) -> dict[str, Any] | None:
        """Return the next tool call, or None when the agent is done."""
        tool_set = set(available_tools)

        # -- Arm B (hypotree tools) state machine ----------------------------
        if "create_hypotheses" in tool_set:
            if self._state == "INIT":
                goal = next(n for n in self._landscape["nodes"] if n.get("is_goal"))
                batch = [
                    {
                        "statement": goal["statement"],
                        "node_id": goal["id"],
                        "is_goal": True,
                        "target_metric": goal.get("target_metric", 0.75),
                        "if_exists": "overwrite",
                    },
                    *[
                        {
                            "statement": n["statement"],
                            "node_id": n["id"],
                            "if_exists": "overwrite",
                        }
                        for n in self._landscape["nodes"]
                        if not n.get("is_goal")
                    ][:5],
                ]
                self._state = "TARGET"
                return {"name": "create_hypotheses", "arguments": {"hypotheses": batch}}

            if self._state == "TARGET":
                self._state = "EVAL"
                return {"name": "get_next_targets", "arguments": {"count": 1}}

            if self._state == "EVAL" and self._pending_target_id:
                self._state = "RECORD"
                return {
                    "name": "evaluate_config",
                    "arguments": {"config": self._pending_config, "depth": self._pending_depth},
                }

            if self._state == "RECORD":
                # Ask for the next target with the result: the fused reply keeps
                # the loop at one round-trip per probe instead of two.
                self._state = "EVAL"
                args: dict[str, Any] = {
                    "node_id": self._pending_target_id,
                    "success": self._pending_success,
                    "depth": self._pending_depth,
                    "count_next_targets": 1,
                }
                if self._pending_claim_id:
                    args["claim_id"] = self._pending_claim_id
                return {"name": "record_evidence", "arguments": args}

            # If the dispatch returned DONE, pending_target stays None and we
            # fall through here — agent stops.
            if self._state == "EVAL" and self._pending_target_id is None:
                return None

        # -- Arm A (scratchpad only) state machine ---------------------------
        if "update_scratchpad" in tool_set:
            if self._state in ("INIT", "NOTE"):
                node = self._rng.choice(self._landscape["nodes"])
                self._pending_config = node["statement"]
                self._state = "PROBE"
                return {
                    "name": "evaluate_config",
                    "arguments": {"config": node["statement"], "depth": 0},
                }
            if self._state == "PROBE":
                self._state = "NOTE"
                content = f"Probed {self._pending_config}: success={self._pending_success:.2f}"
                return {
                    "name": "update_scratchpad",
                    "arguments": {"content": content},
                }

        return None

    def observe_result(self, tool_name: str, result: str) -> None:
        """Cache tool results so the state machine can chain the next call."""
        if tool_name in ("get_next_targets", "record_evidence"):
            # A dispatch arrives either on its own or fused into the record; both
            # must be read, or the fused loop stalls waiting for a target it was
            # already handed.
            self._observe_dispatch(result)
        elif tool_name == "evaluate_config":
            try:
                data = json.loads(result)
                self._pending_success = data.get("success", 0.5)
            except (json.JSONDecodeError, TypeError):
                self._pending_success = 0.5

    def _observe_dispatch(self, result: str) -> None:
        """Read the next target out of a dispatch reply, however it arrived."""
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return
        batch = payload if isinstance(payload, list) else payload.get("next_targets") or []
        data = batch[0] if batch else {"status": "DONE"}
        if data.get("status") == "DONE":
            self._pending_target_id = None
            self._pending_claim_id = None
            return
        self._pending_target_id = data.get("node_id")
        self._pending_claim_id = data.get("claim_id")
        self._pending_config = data.get("statement", "")
        # A target under conflict review must be re-probed at least as deeply as
        # the failure that implicated it.
        self._pending_depth = int(data.get("min_depth") or 0)


# -- OpenAI-compatible LLM agent ---------------------------------------------


class LLMUnavailableError(RuntimeError):
    """The inference server could not be reached after exhausting every retry.

    Raised instead of letting a transport error escape as-is, so the episode can
    end itself cleanly and the sweep can carry on to the next one. This is an
    infrastructure fault, never a result: an episode that ends here is excluded
    from the comparison rather than scored as a failure to solve the task.
    """


def _is_retryable(err: Exception) -> bool:
    """Decide whether a failed request is worth sending again.

    Server-side faults (5xx) and rate limits (429) are transient by nature and
    usually clear on their own, as is anything that never got a reply at all. A
    4xx other than 429 is the request itself being wrong, and repeating it
    verbatim would only waste the backoff.
    """
    if isinstance(err, urllib.error.HTTPError):
        return err.code == 429 or err.code >= 500
    # Everything else reaching here is a request that got no usable answer:
    # connection refused/reset, DNS, a socket that opened and went quiet, or a
    # truncated body. All mean "no answer", not "no".
    return True


def _call_openai_api(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    on_retry: Callable[[int, str, float], None] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint, retrying transient faults.

    Works with Ollama (/v1/chat/completions), GLM, and other compatible APIs.
    Uses stdlib urllib to keep eval/ dependency-free.

    A sweep is 90 sequential episodes against one local inference server, so the
    probability that every single request succeeds is not the thing to design
    for. Transient faults are retried with exponential backoff and jitter;
    everything else is reported immediately, because a malformed request does not
    become well-formed by being sent again.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        # Deliberately only the transport failures. Anything else — a bad URL, a
        # payload that will not serialise — is the harness being wrong, and
        # dressing that up as "the server did not answer" would mark every
        # episode infra-failed and silently shrink the seed set instead of
        # failing where someone would notice.
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last = err
            if not _is_retryable(err) or attempt == LLM_MAX_ATTEMPTS:
                break
            # Exponential backoff, capped, with jitter so repeated failures do
            # not settle into a fixed rhythm against a recovering server.
            delay = min(LLM_RETRY_BASE_S * 2 ** (attempt - 1), LLM_RETRY_MAX_S)
            delay += random.uniform(0, delay / 2)
            if on_retry is not None:
                on_retry(attempt, f"{type(err).__name__}: {err}", delay)
            time.sleep(delay)

    raise LLMUnavailableError(
        f"{model} at {base_url} did not answer after {LLM_MAX_ATTEMPTS} attempt(s): "
        f"{type(last).__name__}: {last}"
    ) from last


def _parse_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI-compatible response."""
    choices = response.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {})
    return message.get("tool_calls", [])


# -- Main runner loop ---------------------------------------------------------


def run(config: EvalConfig) -> dict[str, Any]:
    """Execute one evaluation run and return a summary.

    Creates a fresh engine + log. Drives the agent (mock or real LLM) through
    the task, enforcing tool budget and session resets. Returns a summary dict
    with ``steps_to_target``, ``goals_met``, and log path.
    """
    # Resolve the belief-state DB from the run-scoped workspace identity, so
    # every (run, seed, arm) triple gets its own database and no run can ever
    # inherit another's nodes.
    run_tag = f"seed-{config.seed}-arm-{config.arm}"
    db_path = resolve_eval_db_path(config.workspace_id, run_tag)
    # The file outlives the process, so a re-run of this arm would otherwise
    # resume from the previous attempt's nodes and score an agent that started
    # out already knowing the answer.
    reset_eval_db(db_path, config.workspace_id)
    engine = HypoTreeEngine(db_path, rng_seed=config.seed)
    logger = RunLogger(config.log_path, config.seed, config.arm)
    logger.log_run_start(config)

    import sys

    def _progress(msg: str) -> None:
        """Print a progress line to stderr (stdout is reserved for JSON output)."""
        print(f"[seed={config.seed} arm={config.arm}] {msg}", file=sys.stderr, flush=True)

    _progress(
        f"starting (budget={config.tool_budget}, backend={config.llm_backend}, "
        f"workspace={config.workspace_id}, "
        f"landscape_url={config.landscape_url}, "
        f"max_tokens={config.llm_max_tokens})"
    )

    scratchpad: list[str] = []
    # Every probe made in this run, in order. Arms in AUTO_PERSIST_ARMS replay it
    # across resets; for the others it is recorded but never shown, so the same
    # code path serves all three arms and the logs stay comparable.
    transcript: list[dict[str, Any]] = []
    briefing = (
        config.briefing_path.read_text(encoding="utf-8") if config.briefing_path.exists() else ""
    )
    system_prompt = load_system_prompt(config.arm)
    available_tools = [t["function"]["name"] for t in get_tools_for_arm(config.arm)]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": briefing},
    ]

    agent = MockAgent(config) if config.llm_backend == "mock" else None

    # Arm-agnostic win criteria — both arms call evaluate_config, so both can
    # trigger a win when success >= target_metric at sufficient depth.
    win_criteria = _load_win_criteria(config.landscape_path)

    goals_met = False
    session_num = 0
    step = 0
    # Set when the run stops for a reason other than a win or a spent budget,
    # so the log records *why* rather than defaulting to "agent_stopped".
    stop_reason: str | None = None
    # Set when the episode ends because the inference server became unreachable.
    # Distinct from every other stop reason: it says nothing about the agent.
    infra_failed = False
    # Consecutive turns that dispatched no experiment. See MAX_IDLE_TURNS.
    idle_turns = 0
    # Sorted list of not-yet-fired breakpoints. Draining it as ``step`` crosses
    # each threshold fires every reset exactly once, even if the step counter
    # jumps past a breakpoint (multiple experiments in one LLM turn).
    pending_breakpoints = sorted(b for b in set(config.session_breakpoints) if b > 0)
    # Steps a reset may be postponed by while the agent still owes results for
    # dispatches it has already made. See _reset_is_safe.
    reset_slack = 0
    # Turns spent letting the agent file results it already holds after the win.
    # See MAX_WINDDOWN_TURNS.
    winddown_turns = 0

    while step < config.tool_budget:
        step_at_turn_start = step
        while pending_breakpoints and step >= pending_breakpoints[0] and not goals_met:
            if (
                not _reset_is_safe(engine, config.arm, transcript)
                and reset_slack < MAX_RESET_DEFERRAL
            ):
                # The agent has probed something it has not reported yet. Wiping
                # its context now would destroy that result, and only for the arm
                # whose protocol separates dispatch from reporting — the flat
                # transcript arms lose nothing to a reset because their record is
                # automatic. Deferring to the next coherent boundary keeps the
                # reset measuring memory rather than turn alignment. Bounded, so
                # an agent that simply never reports cannot suppress resets.
                reset_slack += 1
                break
            reset_slack = 0
            pending_breakpoints.pop(0)
            session_num += 1
            summary = _compact_summary(engine, scratchpad, config.arm, transcript)
            frontier_size, breakdown = _belief_state_snapshot(engine, config.arm)
            logger.log_session_reset(
                session_num,
                summary,
                frontier_size=frontier_size,
                status_breakdown=breakdown,
            )
            # Whatever the agent still holds a lease on, it is about to forget.
            # Releasing returns those nodes to the frontier instead of stranding
            # them for the whole lease TTL, which on a 60-step episode is forever.
            released = engine.release_claims() if config.arm == ARM_B else []
            if released:
                logger.log_claims_released(released)
            # Anything still unrecorded when the summary was built is a result
            # the environment was paid for and the belief state never saw. The
            # summary carries it forward now; the count is logged either way,
            # because it is the difference between a reset that measures memory
            # and one that confiscates work, and it was invisible for a run.
            carried = _unrecorded_probes(engine, transcript) if config.arm == ARM_B else []
            if carried:
                logger.log_probes_carried([str(p.get("config", "")) for p in carried])
            _progress(f"session reset #{session_num} at step={step}")
            remaining = config.tool_budget - step
            # The briefing is re-injected verbatim on every reset. It is the
            # immutable task specification (config grammar, axes, goal), not
            # something the agent discovered — an agent that "forgets" it cannot
            # even emit a parseable probe, which would make the arms differ on
            # grammar recall rather than on retained findings. Both arms get it
            # identically, so the only thing still under test is which memory
            # substrate better preserves what was *learned*.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": briefing},
                {
                    "role": "user",
                    "content": (
                        f"--- CONTEXT RESET #{session_num} (step {step}) ---\n"
                        f"Your conversation history was cleared. The task briefing above "
                        f"is unchanged. Everything you had learned is gone except the "
                        f"summary below.\n\n"
                        f"{summary}\n\n"
                        f"You have {remaining} of {config.tool_budget} experiments left. "
                        f"Do NOT re-probe anything already listed as settled above. "
                        f"Continue from where the summary leaves off."
                    ),
                },
            ]

        if agent is not None:
            tool_call = agent.get_tool_call(available_tools)
            if tool_call is None:
                break
            result_str = _execute_tool(
                tool_call["name"],
                tool_call["arguments"],
                engine,
                scratchpad,
                config,
                logger,
                transcript,
            )
            agent.observe_result(tool_call["name"], result_str)

            if tool_call["name"] in ("get_next_targets", "record_evidence"):
                # A DONE sentinel is only ever returned alone, so an
                # otherwise-populated batch still has work in it. The record is
                # watched alongside the dispatch because it carries the fused one.
                stop_reason = _dispatch_stop_reason(result_str) or stop_reason
                if stop_reason:
                    break

            if tool_call["name"] == "evaluate_config":
                # A transport failure is not an experiment, so it buys no step — but
                # it must still fall through to the turn's progress accounting,
                # or a permanently unreachable oracle would spin here forever.
                eval_data = _probe_result(result_str)
                if eval_data is not None and not goals_met:
                    step += 1
                    _progress(
                        f"step {step}/{config.tool_budget}: probed config "
                        f"(depth={tool_call['arguments'].get('depth', 0)})"
                    )
                    # Arm-agnostic win check: any probe clearing the goal target
                    # at sufficient depth counts as a win for BOTH arms.
                    if _check_landscape_win(
                        tool_call["arguments"]["config"],
                        tool_call["arguments"].get("depth", 0),
                        eval_data.get("success", 0.0),
                        win_criteria,
                    ):
                        goals_met = True
                        break
        else:
            tools = get_tools_for_arm(config.arm)
            _llm_started = time.monotonic()
            try:
                response = _call_openai_api(
                    config.llm_base_url,
                    config.llm_model,
                    messages,
                    tools,
                    config.llm_temperature,
                    config.llm_max_tokens,
                    on_retry=logger.log_llm_retry,
                )
            except LLMUnavailableError as err:
                # The server is gone, not the agent. End this episode where it
                # stands and let the sweep move on: an exception escaping here is
                # what cost run I its last 61 episodes.
                infra_failed = True
                stop_reason = "llm_unavailable"
                _progress(f"stopping: {err}")
                break
            _llm_elapsed = time.monotonic() - _llm_started
            tool_calls = _parse_tool_calls(response)

            # Log token usage from the LLM response (prompt + completion + total).
            usage = response.get("usage", {})
            logger.log_llm_call(
                _llm_elapsed,
                len(tool_calls),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
            )
            if usage:
                _progress(
                    f"LLM tokens: "
                    f"prompt={usage.get('prompt_tokens', '?')} "
                    f"completion={usage.get('completion_tokens', '?')} "
                    f"total={usage.get('total_tokens', '?')}"
                )

            if not tool_calls:
                # A turn of prose is not a decision to stop. Ending the episode
                # here censored a *baseline* arm to the full budget after 25
                # productive probes — a 75-probe penalty for one chatty turn,
                # scored as if the arm had failed the task, and biased in favour
                # of the treatment. Nudge instead and let the idle-turn guard
                # below bound it: an agent that genuinely has nothing left to do
                # still terminates, in MAX_IDLE_TURNS rather than in one.
                assistant_msg = response.get("choices", [{}])[0].get("message", {})
                content_preview = str(assistant_msg.get("content", ""))[:200]
                finish_reason = response.get("choices", [{}])[0].get("finish_reason", "?")
                logger.log_agent_action(ACTION_NOOP, _status_context(engine))
                _progress(
                    f"LLM returned no tool_calls "
                    f"(finish_reason={finish_reason}, "
                    f"content_preview={content_preview!r})"
                )
                # Rebuilt rather than echoed: the turn is being continued now, so
                # a response missing `role` (or missing `message` entirely) would
                # be handed straight back to the API on the next call.
                messages.append(
                    {
                        "role": "assistant",
                        "content": str(assistant_msg.get("content", "") or ""),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That turn contained no tool call, so nothing happened. "
                            f"You have {config.tool_budget - step} of "
                            f"{config.tool_budget} experiments left and the goal is "
                            "not met. Reply with a tool call — reasoning alone makes "
                            "no progress."
                        ),
                    }
                )
            else:
                # Append the assistant turn once (carrying every tool_call), then
                # one tool result per call — the structure OpenAI-compatible APIs
                # expect. Skipped above, where the nudge already closed the turn.
                assistant_msg = response.get("choices", [{}])[0].get("message", {})
                messages.append(assistant_msg)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}

                    result_str = _execute_tool(
                        tool_name, args, engine, scratchpad, config, logger, transcript
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result_str}
                    )

                    if tool_name in ("get_next_targets", "record_evidence"):
                        # The dispatch may have arrived fused into the record, and
                        # a DONE that only get_next_targets is watched for would be
                        # missed entirely — the run would then burn its whole budget
                        # after the search was already over.
                        stop_reason = _dispatch_stop_reason(result_str) or stop_reason
                        if stop_reason:
                            break

                    if tool_name == "evaluate_config":
                        eval_data = _probe_result(result_str)
                        if eval_data is None:
                            # Not an experiment; the next tool call in this turn still runs.
                            continue
                        if goals_met:
                            # Winding down. The target is already cleared, so a
                            # further probe buys nothing and must not move the
                            # step counter the headline metric is read from.
                            continue
                        step += 1
                        # Arm-agnostic win check (same as mock path).
                        if _check_landscape_win(
                            args.get("config", ""),
                            args.get("depth", 0),
                            eval_data.get("success", 0.0),
                            win_criteria,
                        ):
                            goals_met = True
                            break
                        if step >= config.tool_budget:
                            break

        # A landscape win detected inside the tool-call loop only breaks the
        # inner for-loop; propagate it to the outer budget loop so the run stops
        # at the winning step instead of draining the full budget (which would
        # censor every steps_to_target to the budget and erase arm differences).
        # A stall propagates the same way: with nothing left to dispatch the
        # agent cannot make progress, and looping until the budget runs out would
        # just burn wall-clock on identical turns.
        if goals_met:
            # ...but not the instant the probe lands. The agent is mid-protocol:
            # it has just run an experiment and not yet reported it, and on the
            # seeds where the swap that names a culprit is also the winning
            # combination, that unfiled record is the one that resolves the
            # conflict. Ending here left the engine correct and the scoreboard
            # unable to see it. Give it a bounded number of turns to file what it
            # is holding — no further experiments are accepted and `step` is
            # frozen, so this cannot move the headline metric.
            if _reset_is_safe(engine, config.arm, transcript):
                _progress(f"GOAL MET at step={step} (landscape target cleared)")
                break
            if winddown_turns >= MAX_WINDDOWN_TURNS:
                _progress(f"GOAL MET at step={step}; results still unreported after wind-down")
                break
            winddown_turns += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The target is cleared — this task is finished and no further "
                        "experiments will be accepted. Record the results you are still "
                        "holding so the belief state is complete, then stop."
                    ),
                }
            )
            continue
        if stop_reason:
            _progress(f"stopping at step={step}: {stop_reason}")
            break

        # The budget is counted in probes, so an agent that never probes never
        # advances it. Left unbounded that is an infinite loop: the harness would
        # keep paying for turns forever on a model that talks without
        # experimenting. Bounding idle turns keeps an unattended run terminating.
        if step == step_at_turn_start:
            idle_turns += 1
            if idle_turns >= MAX_IDLE_TURNS:
                stop_reason = "no_progress"
                _progress(f"stopping: {MAX_IDLE_TURNS} turns without an experiment")
                break
        else:
            idle_turns = 0

        # Deliberately no belief-based completion check here. Whether arm B's
        # engine considers its goal node verified is a property of what the agent
        # recorded, not of what it found: three successes recorded against the
        # goal node clear its posterior bar without any winning configuration
        # ever being probed. Only an environment result decides a win, and it
        # decides it identically for every arm.

    if goals_met:
        reason = "all_goals_met"
    elif step >= config.tool_budget:
        reason = "budget_exhausted"
    elif stop_reason:
        reason = stop_reason
    else:
        reason = "agent_stopped"
    _progress(f"done: {reason} (steps={step}, goals_met={goals_met})")
    logger.log_run_end(reason, goals_met, infra_failed=infra_failed)

    summary = {
        "seed": config.seed,
        "arm": config.arm,
        "steps_to_target": step if goals_met else config.tool_budget,
        "goals_met": goals_met,
        "log_path": str(config.log_path),
        "reason": reason,
    }

    engine.close()
    return summary


def _probe_transcript_table(transcript: list[dict[str, Any]], limit: int = 60) -> str:
    """Chronological config→score table of every probe made so far.

    This is the *unstructured* memory the auto-persisting baseline carries across
    a reset: exactly the same raw facts hypotree records, with none of the
    interpretation. No statuses, no refutation semantics, no frontier, no
    ordering by promise — just what was tried and what came back. Holding the
    facts constant is what makes the remaining comparison a test of structure.
    """
    if not transcript:
        return "_No experiments run yet._"
    rows = ["| # | config | depth | success |", "|---|---|---|---|"]
    shown = transcript[-limit:]
    offset = len(transcript) - len(shown)
    for i, entry in enumerate(shown, start=offset + 1):
        rows.append(f"| {i} | `{entry['config']}` | {entry['depth']} | {entry['success']} |")
    if offset:
        rows.append(f"| … | _{offset} earlier probes truncated_ | | |")
    return "\n".join(rows)


def _belief_state_snapshot(engine: HypoTreeEngine, arm: str) -> tuple[int | None, dict | None]:
    """Frontier size + status breakdown at a reset, for post-hoc diagnosis.

    Only Arm B has a belief state to snapshot; the baselines return None so the
    two are never conflated in the logs.
    """
    if arm != ARM_B:
        return None, None
    status = engine.get_goal_status()
    return status.frontier_size, dict(status.status_breakdown)


def _evidence_ledger(engine: HypoTreeEngine, limit: int = 25) -> str:
    """A compact config→score ledger of everything already tested.

    This is the belief state that actually matters for a search task: which
    candidates were tried, what they scored, and which are settled. Statuses
    alone are not enough — an agent that only sees "12 nodes IN_PROGRESS" will
    re-probe them, which is precisely the failure mode that burns the budget.

    Sorted best-score-first so the strongest leads survive truncation.
    """
    rows: list[tuple[float, str, str]] = []
    for node in engine._store.get_all_nodes():
        if node.evidence_count == 0:
            continue
        history = engine.get_evidence_history(node.id, limit=1)
        score = history[0].success if history and history[0].success is not None else None
        if score is None:
            continue
        rows.append((score, node.statement, node.status.value))

    if not rows:
        return "_No evidence recorded yet._"

    rows.sort(key=lambda r: r[0], reverse=True)
    lines = ["| score | configuration | status |", "|---|---|---|"]
    for score, statement, status in rows[:limit]:
        lines.append(f"| {score:.3f} | `{statement}` | {status} |")
    if len(rows) > limit:
        lines.append(f"| … | _{len(rows) - limit} more already tested_ | |")
    return "\n".join(lines)


def _notes_section(scratchpad: list[str]) -> str:
    if not scratchpad:
        return "## Your notes\n_You saved no notes before this reset._"
    return "## Your notes\n" + "\n\n".join(scratchpad)


def _compact_summary(
    engine: HypoTreeEngine,
    scratchpad: list[str],
    arm: str,
    transcript: list[dict[str, Any]] | None = None,
) -> str:
    """Build the memory an arm carries across a context reset.

    Arm A carries only what it chose to write — the honest "just use a
    scratchpad" baseline, where persistence costs discipline.
    Arm F additionally carries an automatic, flat transcript of every probe, so
    it never loses a fact simply because it forgot to write one down.
    Arm B carries its structured belief state: the evidence ledger, the
    confirmed and settled hypotheses, and the open frontier.
    """
    if arm == ARM_A:
        return _notes_section(scratchpad)

    if arm == ARM_F:
        return (
            f"## Experiment log — automatically preserved, every probe so far\n"
            f"These are raw results only; nothing here has been interpreted for you.\n"
            f"{_probe_transcript_table(transcript or [])}\n\n"
            f"{_notes_section(scratchpad)}"
        )

    goal_status = engine.get_goal_status()
    verified = engine.list_nodes(status_filter=["VERIFIED"], limit=15)
    settled = engine.list_nodes(status_filter=["INVALIDATED", "PRUNED", "EXHAUSTED"], limit=20)
    open_frontier = engine.list_nodes(status_filter=["UNTESTED", "IN_PROGRESS"], limit=15)
    return (
        f"## Goal status\n"
        f"Goals met: {goal_status.goals_met_count}/{goal_status.goals_total_count} · "
        f"frontier: {goal_status.frontier_size} · nodes: {goal_status.total_nodes}\n\n"
        f"{_unrecorded_section(engine, transcript)}"
        f"## Evidence ledger — every configuration already tested\n"
        f"Do NOT re-probe any configuration listed here; the result will not change.\n"
        f"{_evidence_ledger(engine)}\n\n"
        f"## CONFIRMED — hypotheses that cleared the bar\n"
        f"{verified}\n\n"
        f"## SETTLED — refuted, pruned, or conclusively tested; do NOT revisit\n"
        f"{settled}\n\n"
        f"{_conflict_section(engine)}"
        f"## OPEN frontier — untested, work on these next\n"
        f"{open_frontier}"
    )


def _unrecorded_section(engine: HypoTreeEngine, transcript: list[dict[str, Any]] | None) -> str:
    """Results the agent probed but had not recorded when its context was wiped.

    Parity, not charity. Arm F is handed "an automatic, flat transcript of every
    probe so it never loses a fact simply because it forgot to write one down" —
    a probe it has not interpreted survives its reset by construction. Arm B's
    memory is its belief state, which by design holds only what has been
    *recorded*, so the same probe vanishes. Without this the reset stops
    measuring which substrate preserves what was learned and starts measuring
    which protocol happens to align with the reset schedule.

    Only the in-flight results are carried, never the whole transcript: the
    distinction under test is whether structure beats a flat log, and handing B
    the flat log would erase it.
    """
    pending = _unrecorded_probes(engine, transcript)
    if not pending:
        return ""
    rows = "\n".join(
        f"| {p.get('config', '')} | {p.get('depth', 0)} | {p.get('success', 0.0)} |"
        for p in pending
    )
    return (
        "## UNRECORDED — you probed these and never recorded them\n"
        "The environment was already paid for these results. Record each one "
        "against the hypothesis it tested **before probing anything else**, or "
        "the work is lost and you will pay for it twice.\n"
        "| config | depth | success |\n|---|---|---|\n"
        f"{rows}\n\n"
    )


def _conflict_section(engine: HypoTreeEngine) -> str:
    """Unresolved conflicts and the re-tests that would settle them.

    Without this, a context reset erases the one thing the belief state knows and
    a flat log cannot express: that a set of confirmations has been contradicted
    by something built on top of them, and which re-test resolves it. The nodes
    involved sit in NEEDS_REVISION, which appears in neither the confirmed nor
    the settled nor the open-frontier listing, so they would otherwise vanish
    from the summary entirely.
    """
    conflicts = engine.get_conflicts(open_only=True)
    if not conflicts:
        return ""

    lines = [
        "## UNRESOLVED CONFLICTS — these assumptions cannot all hold",
        "Re-probe each suspect at the stated depth. One probe either refutes it "
        "(naming the cause) or clears it and shrinks the list. Do this before "
        "trying new combinations.",
        "",
    ]
    for conflict in conflicts:
        suspects = conflict["remaining_suspects"]
        statements = {m["node_id"]: m["statement"] for m in conflict["members"]}
        lines.append(
            f"- `{conflict['source_node_id']}` failed at depth "
            f"{conflict['conflict_depth']}. Still suspect: "
            + (
                ", ".join(f"`{s}` ({statements.get(s, '')})" for s in suspects)
                if suspects
                else "none — every assumption survived a test that deep, so this "
                "is an interaction effect; try a different combination"
            )
        )
    return "\n".join(lines) + "\n\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.runner.runner",
        description="Run one (seed, arm) evaluation episode.",
    )
    parser.add_argument("eval_dir", type=Path, help="path to the eval/ directory")
    parser.add_argument("seed", type=int, help="task seed")
    parser.add_argument("arm", type=str.upper, choices=list(ALL_ARMS), help="arm label")
    parser.add_argument(
        "--run-id",
        required=True,
        help="identifier isolating this batch's logs, workspace and belief-state DB",
    )
    parser.add_argument("--llm-backend", default="mock", choices=("mock", "openai"))
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-model", default="qwen3.6:27b-q8_0")
    parser.add_argument("--llm-max-tokens", type=int, default=65536)
    parser.add_argument("--landscape-url", default=None)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    config_kwargs: dict[str, Any] = dict(
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
    )
    if args.landscape_url:
        config_kwargs["landscape_url"] = args.landscape_url
    config = make_run_config(args.arm, args.seed, args.eval_dir, args.run_id, **config_kwargs)
    result = run(config)
    print(json.dumps(result, indent=2))
