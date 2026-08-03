"""Frozen configuration for the headless evaluation harness.

All settings that govern the gate are locked here: model, temperature, tool
budget, session-reset policy, seeds, LLM endpoint, and the mock-agent toggle.

Both arms (baseline scratchpad + hypotree treatment) use identical settings
except for the system prompt and which tools are available.

The belief-state DB is workspace-isolated via ``workspace_id`` so eval data
never collides with the live dogfooding DB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A run id becomes both a directory name and a workspace key, so it is held to a
# single safe component: no separators, no traversal, bounded length.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9._~-]{1,128}")

# Held-out task seeds.
#
# Two earlier seed sets are retired. {1001…1010} was invalidated by a harness
# confound (the briefing was dropped at every context reset). {1101…1110} ran on
# a landscape whose criterion-3 mechanism was structurally unmeasurable and whose
# criterion-1 result turned out to be decided by whether the baseline agent chose
# to write notes at all. Both results stay on record and are reported alongside
# any re-run; neither is silently dropped.
#
# n is raised from 10 to 30 because the previous 95% CI on the paired median was
# [1, 32] — far too wide to resolve the effect that actually matters once the
# baseline is given automatic persistence.
TASK_SEEDS: list[int] = list(range(1201, 1231))  # 1201..1230

# Retired seed sets, kept so superseded runs remain reproducible and citable.
RETIRED_TASK_SEEDS: list[int] = [*range(1001, 1011), *range(1101, 1111)]

# Navigator RNG seeds for the TS-quality ablation (pre-registered §4.2), one per
# task seed.
NAVIGATOR_RNG_SEEDS: list[int] = list(range(2001, 2031))  # 2001..2030

# Default tool budget per task (max experiments). The landscape generator also
# sets a per-seed budget; this is the fallback if the landscape omits it.
DEFAULT_TOOL_BUDGET: int = 100

# Default landscape server port for the black-box evaluation endpoint.
DEFAULT_LANDSCAPE_PORT: int = 8080

# LLM request timeout (seconds). Generous because local models can be slow.
LLM_TIMEOUT_S: int = 1200

# How many times a transient LLM transport failure is retried before the episode
# gives up. A local inference server dropping one request mid-sweep is a fact of
# life, not a result: run I died on a single HTTP error at seed 1210 of 30 and
# took the remaining 61 episodes with it. Five attempts spans roughly two minutes
# of backoff, which covers a model reload or a brief server restart.
LLM_MAX_ATTEMPTS: int = 5

# Backoff between attempts, in seconds: doubles each time and is capped, with
# jitter added on top so a retry storm cannot synchronise. 2, 4, 8, 16 ...
LLM_RETRY_BASE_S: float = 2.0
LLM_RETRY_MAX_S: float = 60.0

# Arm labels used in logs and filenames.
#
# Three arms, because "does a belief state help?" is really two questions and the
# previous run could not tell them apart:
#
#   A  manual scratchpad only. Memory exists solely if the agent chooses to write
#      it. Against this arm, hypotree is being credited for making persistence
#      *automatic* — an ergonomic advantage (criterion 1a).
#   F  flat auto-persisted transcript. Every probe and its score is preserved
#      across resets without the agent doing anything, plus its own free-text
#      notes. Same information, no structure: no statuses, no refutation
#      semantics, no frontier, no navigator. Against this arm, hypotree is being
#      credited only for *structure* — the informational advantage (criterion 1b),
#      which is the hypothesis the project was actually built to test.
#   B  the full hypotree belief-state toolkit.
ARM_A = "A"  # baseline: manual markdown scratchpad, no automatic persistence
ARM_F = "F"  # baseline: automatic flat probe transcript + manual notes
ARM_B = "B"  # treatment: hypotree belief-state tools

# Every arm the gate executes, in run order.
ALL_ARMS: tuple[str, ...] = (ARM_B, ARM_F, ARM_A)

# Arms that receive an automatically-persisted probe transcript across resets.
AUTO_PERSIST_ARMS: frozenset[str] = frozenset({ARM_F})

# The workspace ID for eval runs. All belief-state DBs created during the gate
# are isolated under this workspace name so they never collide with the live
# dogfooding DB.
DEFAULT_WORKSPACE_ID: str = "hypotree-eval"


def run_workspace_id(run_id: str, base: str = DEFAULT_WORKSPACE_ID) -> str:
    """Workspace name for one identified run, e.g. ``2026-07-27a@hypotree-eval``.

    Every run gets its own belief-state root. Two runs of the same seed and arm
    would otherwise share a database, so a re-run would silently inherit the
    previous run's nodes and its results would be neither reproducible nor
    comparable.
    """
    return f"{_sanitise_run_id(run_id)}@{base}"


def run_dir(eval_dir: Path, run_id: str) -> Path:
    """Directory holding every artefact of one identified run."""
    return eval_dir / "runs" / _sanitise_run_id(run_id)


def _sanitise_run_id(run_id: str) -> str:
    """Validate a run id as a single safe path/workspace component.

    Rejected rather than silently rewritten: a run id ends up in a filesystem
    path and a workspace key, so quietly mangling it would produce two runs that
    look distinct on the command line but collide on disk.
    """
    cleaned = run_id.strip()
    if not cleaned or not _RUN_ID_RE.fullmatch(cleaned):
        raise ValueError(
            f"invalid run id {run_id!r}: use only letters, digits, "
            f"'.', '_', '-' or '~' (max 128 chars)"
        )
    return cleaned


@dataclass(frozen=True)
class EvalConfig:
    """All parameters for one evaluation run.

    Frozen so it cannot be mutated mid-run. The runner receives one instance
    per (seed, arm) combination.
    """

    # Which arm — A (scratchpad) or B (hypotree).
    arm: str

    # Task seed (see TASK_SEEDS).
    seed: int

    # Identifier of the batch this run belongs to. Every artefact — JSONL log,
    # belief-state DB, workspace — is namespaced by it.
    run_id: str

    # Path to the landscape JSON config for this seed.
    landscape_path: Path

    # Path to the human-readable briefing markdown.
    briefing_path: Path

    # Where the runner writes the JSONL event log.
    log_path: Path

    # Landscape server URL (the black-box eval endpoint).
    landscape_url: str

    # Maximum experiments the agent may dispatch.
    tool_budget: int = DEFAULT_TOOL_BUDGET

    # Session-reset breakpoints (step indices at which context is cleared).
    session_breakpoints: tuple[int, ...] = ()

    # The environment's evidence regime. Pinned onto every node the agent
    # creates, because the regime describes the oracle, not the agent's
    # preference — see runner._pin_regime.
    evidence_regime: str = "deterministic"

    # LLM backend: "mock" for testing, "openai" for a real OpenAI-compatible API.
    llm_backend: str = "mock"

    # LLM settings (only used when llm_backend != "mock").
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama default
    llm_model: str = "qwen3.6:27b-q8_0"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 65536

    # The mock agent's internal RNG seed (for deterministic mock behaviour).
    mock_seed: int = 42

    # Workspace ID for belief-state isolation (set HYPOTREE_WORKSPACE_ID).
    # Defaults to the shared eval workspace; make_run_config always narrows it
    # to the run-scoped ``<run_id>@hypotree-eval``.
    workspace_id: str = DEFAULT_WORKSPACE_ID


@dataclass(frozen=True)
class AblationConfig:
    """Parameters for the TS-quality ablation (criterion 2).

    No LLM and no landscape topology are involved — the ablation runs three
    selection strategies against the same seeded stochastic bandit and measures
    cumulative regret over a fixed horizon.
    """

    seed: int  # task seed (see TASK_SEEDS)
    rng_seed: int  # navigator RNG seed (2001..2030)
    run_id: str
    log_path: Path


def make_run_config(
    arm: str,
    seed: int,
    eval_dir: Path,
    run_id: str,
    *,
    landscape_url: str = f"http://127.0.0.1:{DEFAULT_LANDSCAPE_PORT}/evaluate",
    llm_backend: str = "mock",
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "qwen3.6:27b-q8_0",
    llm_max_tokens: int = 65536,
    briefing_path: Path | None = None,
    workspace_id: str | None = None,
) -> EvalConfig:
    """Build a frozen EvalConfig for one (arm, seed) run of ``run_id``.

    Derives paths from ``eval_dir`` following the convention:
    - landscapes: ``eval_dir/environment/landscapes/landscape_seed_XXXX.json``
    - briefings: ``eval_dir/environment/briefings/briefing_seed_XXXX.md``
    - logs: ``eval_dir/runs/<run_id>/seed-XXXX-arm-X.jsonl``

    Landscapes and briefings are deliberately NOT namespaced by run id: they are
    a pure function of the seed, so every run must see the identical task or the
    runs are not comparable.
    """
    landscape_path = eval_dir / "environment" / "landscapes" / f"landscape_seed_{seed}.json"
    if briefing_path is None:
        briefing_path = eval_dir / "environment" / "briefings" / f"briefing_seed_{seed}.md"
    log_path = run_dir(eval_dir, run_id) / f"seed-{seed}-arm-{arm}.jsonl"

    # Load tool budget + breakpoints from the landscape JSON.
    import json

    data = json.loads(landscape_path.read_text(encoding="utf-8"))
    tool_budget = data.get("tool_budget", DEFAULT_TOOL_BUDGET)
    breakpoints = tuple(data.get("session_breakpoints", []))
    evidence_regime = data.get("evidence_regime", "deterministic")

    return EvalConfig(
        arm=arm,
        seed=seed,
        run_id=_sanitise_run_id(run_id),
        landscape_path=landscape_path,
        briefing_path=briefing_path,
        log_path=log_path,
        landscape_url=landscape_url,
        tool_budget=tool_budget,
        session_breakpoints=breakpoints,
        evidence_regime=evidence_regime,
        llm_backend=llm_backend,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        workspace_id=workspace_id or run_workspace_id(run_id),
    )


def make_ablation_config(
    seed: int,
    rng_seed: int,
    eval_dir: Path,
    run_id: str,
) -> AblationConfig:
    """Build a frozen AblationConfig for one (task_seed, rng_seed) pair."""
    log_path = run_dir(eval_dir, run_id) / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl"
    return AblationConfig(
        seed=seed,
        rng_seed=rng_seed,
        run_id=_sanitise_run_id(run_id),
        log_path=log_path,
    )


def resolve_eval_db_path(workspace_id: str, run_tag: str = "state") -> Path:
    """Resolve the SQLite belief-state DB path for an eval workspace.

    Constructs a per-run DB path under the isolated workspace directory so
    multiple (seed, arm) runs never collide:
    ``<data_home>/mcp_hypotree/<workspace_id>/<run_tag>.db``

    Does NOT mutate the global ``HYPOTREE_WORKSPACE_ID`` env var.
    """
    from hypotree.store.identity import data_home

    root = data_home() / "mcp_hypotree" / workspace_id
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{run_tag}.db"


def reset_eval_db(db_path: Path, workspace_id: str, base: str = DEFAULT_WORKSPACE_ID) -> None:
    """Delete an eval belief-state DB so its run starts from nothing.

    An arm's database is keyed by (run, seed, arm) and survives the process, so
    re-running one arm — after a crash, or to reproduce a result — would resume
    from the previous attempt's nodes and measure an agent that already knew the
    answer. Every run must begin with an empty belief state, which is a property
    of the file, not of the engine object created over it.

    Refuses to touch anything outside an eval workspace: the live dogfooding
    database is the accumulated belief state of this repository and deleting it
    is never part of running an evaluation.
    """
    if not workspace_id.endswith(f"@{base}") and workspace_id != base:
        raise ValueError(
            f"refusing to reset {db_path}: workspace {workspace_id!r} is not an eval workspace"
        )
    # WAL mode keeps committed pages in the sidecars; removing only the main
    # file would leave a torn database that reopens with the old contents.
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
