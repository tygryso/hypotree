"""E2E test — spawn mcp_server.py, speak JSON-RPC over stdio, full agent loop.

Uses the MCP SDK client to connect to the spawned server process and exercise
the complete closed-loop: create → select → evidence → convergence → DONE.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import anyio
import pytest

if sys.version_info < (3, 11):  # BaseExceptionGroup is a builtin from 3.11 on.
    from exceptiongroup import BaseExceptionGroup


@pytest.fixture
def server_env(tmp_path: Path) -> dict[str, str]:
    """Environment for the MCP server subprocess."""
    env = os.environ.copy()
    env["HYPOTREE_WORKSPACE_ID"] = str(tmp_path)
    env["XDG_DATA_HOME"] = str(tmp_path / "xdg")
    return env


def _is_client_teardown_race(exc: BaseException) -> bool:
    """Whether a failure is only the stdio client closing its own streams.

    Leaving an `async with stdio_client(...)` block tears the client's task group
    down while the server subprocess may still be writing, and the reader task
    then hits an already-closed stream. That is a shutdown ordering detail of the
    client transport, not a server fault, and it shows up on Windows because the
    subprocess is slower to wind down there.

    Deliberately narrow: every leaf of the group must be a closed-stream error.
    Anything else — a protocol error, a crashed server, a failed assertion — is
    re-raised, and the caller additionally asserts that every request completed,
    so a server that died halfway can never be mistaken for teardown noise.
    """
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(_is_client_teardown_race(e) for e in exc.exceptions)
    return isinstance(exc, anyio.BrokenResourceError | anyio.ClosedResourceError)


# ---------------------------------------------------------------------------
# The transport tier below spawns the real server process and speaks JSON-RPC
# over stdio via the MCP SDK client, exercising the async list_tools/call_tool
# wiring end to end. The remaining tests drive _dispatch directly — the pure
# routing layer — to cover every tool branch without process overhead.
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stdio_round_trip(server_env: dict[str, str]) -> None:
    """Spawn the server as a subprocess and drive it over real stdio JSON-RPC."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypotree.mcp_server"],
        env=server_env,
    )

    # Every step that completed, so a shutdown race cannot be mistaken for the
    # server having failed halfway through.
    done: list[str] = []

    async def _run() -> None:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            done.append("initialize")

            tools = await session.list_tools()
            assert len(tools.tools) == 18
            done.append("list_tools")

            res = await session.call_tool(
                "create_hypotheses", {"hypotheses": [{"statement": "e2e", "node_id": "n1"}]}
            )
            created = json.loads(res.content[0].text)[0]
            assert created["node"]["id"] == "n1"
            assert created["created"] is True
            done.append("create_hypotheses")

            res = await session.call_tool("get_next_targets", {})
            target = json.loads(res.content[0].text)[0]
            assert target["status"] == "SELECTED"
            assert target["node_id"] == "n1"
            done.append("get_next_targets")

            res = await session.call_tool(
                "record_evidence",
                {"node_id": "n1", "success": 1.0, "claim_id": target["claim_id"]},
            )
            updated = json.loads(res.content[0].text)
            assert updated["node"]["alpha"] > 1.0
            # No dispatch was asked for, so none is fused in.
            assert updated["next_targets"] == []
            done.append("record_evidence")

            res = await session.call_tool("render_dag_map", {})
            rendered = json.loads(res.content[0].text)
            assert "graph TD" in rendered["mermaid"]
            done.append("render_dag_map")

    # Generous because this spawns a Python subprocess that imports numpy, scipy
    # and networkx; on a cold Windows runner that is many times slower than on
    # Linux, and a timeout here reads as a protocol failure that never happened.
    try:
        await asyncio.wait_for(_run(), timeout=1200)
    except BaseException as exc:  # noqa: BLE001 — re-raised unless purely teardown noise
        if not _is_client_teardown_race(exc):
            raise

    assert done == [
        "initialize",
        "list_tools",
        "create_hypotheses",
        "get_next_targets",
        "record_evidence",
        "render_dag_map",
    ]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_handle_call_tool_serializes(tmp_path: Path) -> None:
    """The in-process call-tool handler dispatches and JSON-encodes under lock."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _handle_call_tool

    engine = HypoTreeEngine(tmp_path / "state.db", rng_seed=42)
    write_lock = asyncio.Lock()
    try:
        out = await _handle_call_tool(
            engine,
            write_lock,
            "create_hypotheses",
            {"hypotheses": [{"statement": "x", "node_id": "n1"}]},
        )
        assert len(out) == 1
        assert out[0].type == "text"
        parsed = json.loads(out[0].text)[0]
        assert parsed["node"]["id"] == "n1"
        assert parsed["created"] is True
    finally:
        engine.close()


@pytest.mark.e2e
def test_dispatch_all_tools(tmp_path: Path) -> None:
    """Exercise every tool through the _dispatch routing layer."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    db = tmp_path / "state.db"
    engine = HypoTreeEngine(db, rng_seed=42)
    try:
        # create_hypotheses
        result = _dispatch(
            engine, "create_hypotheses", {"hypotheses": [{"statement": "test", "node_id": "n1"}]}
        )[0]
        assert result["node"]["id"] == "n1"
        assert result["created"] is True

        # get_next_targets
        result = _dispatch(engine, "get_next_targets", {})[0]
        assert result["status"] == "SELECTED"
        assert result["node_id"] == "n1"
        claim_id = result["claim_id"]

        # record_evidence
        result = _dispatch(
            engine,
            "record_evidence",
            {"node_id": "n1", "success": 1.0, "claim_id": claim_id},
        )
        assert result["node"]["alpha"] > 1.0

        # get_goal_status
        result = _dispatch(engine, "get_goal_status", {})
        assert "goals" in result

        # get_dag_context
        result = _dispatch(engine, "get_dag_context", {})
        assert len(result["nodes"]) >= 1

        # render_dag_map
        result = _dispatch(engine, "render_dag_map", {})
        assert "mermaid" in result
        assert "graph TD" in result["mermaid"]

        # update_status
        result = _dispatch(
            engine,
            "update_status",
            {"node_ids": ["n1"], "new_status": "IN_PROGRESS", "reason": "manual"},
        )[0]
        assert result["node"]["status"] == "IN_PROGRESS"
        assert "old_status" in result
        assert "transition" in result

        # invalidate_upstream
        result = _dispatch(engine, "invalidate_upstream", {"leaf_id": "n1"})
        assert "affected_ids" in result

        # verify_upstream
        result = _dispatch(engine, "verify_upstream", {"child_id": "n1"})
        assert "affected_ids" in result
    finally:
        engine.close()


@pytest.mark.e2e
def test_dispatch_infra_evidence(tmp_path: Path) -> None:
    """Infra error evidence routes correctly through _dispatch."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "state.db", rng_seed=42)
    try:
        _dispatch(
            engine, "create_hypotheses", {"hypotheses": [{"statement": "h1", "node_id": "n1"}]}
        )
        result = _dispatch(
            engine,
            "record_evidence",
            {
                "node_id": "n1",
                "evidence_kind": "infra",
                "error_type": "OOM",
                "message": "killed",
            },
        )
        assert result["node"]["infra_retry_count"] == 1
    finally:
        engine.close()


@pytest.mark.e2e
def test_dispatch_unknown_tool_raises(tmp_path: Path) -> None:
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "state.db", rng_seed=42)
    try:
        with pytest.raises(ValueError, match="Unknown tool"):
            _dispatch(engine, "nonexistent_tool", {})
    finally:
        engine.close()


@pytest.mark.e2e
def test_tool_definitions_complete() -> None:
    """Every tool the agent can reach is defined with the expected name."""
    from hypotree.mcp_server import _tool_definitions

    tools = _tool_definitions()
    names = {t.name for t in tools}
    expected = {
        "create_hypotheses",
        "get_next_targets",
        "record_evidence",
        "renew_claim",
        "release_claims",
        "update_status",
        "invalidate_upstream",
        "verify_upstream",
        "get_goal_status",
        "get_conflicts",
        "suggest_discriminating_experiment",
        "get_dag_context",
        "render_dag_map",
        "list_nodes",
        "get_evidence_history",
        "get_active_claims",
        "generate_learning_path",
        "get_workspace_info",
    }
    assert names == expected


@pytest.mark.e2e
def test_there_is_exactly_one_way_to_create_and_one_to_update() -> None:
    """A singular tool beside a batch one is a decision the caller gets wrong.

    Both pairs shared every line of logic and differed only in arity, and in a
    full evaluation run it was always the batch variant — with the richer
    payload — that the caller malformed.
    """
    from hypotree.mcp_server import _tool_definitions

    names = {t.name for t in _tool_definitions()}
    assert not names & {"create_hypothesis", "bulk_create_hypotheses", "bulk_update_status"}


@pytest.mark.e2e
def test_record_evidence_can_fuse_the_next_dispatch(tmp_path: Path) -> None:
    """One round-trip instead of two, on the call an agent makes most often."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "fuse.db", rng_seed=7)
    try:
        _dispatch(
            engine,
            "create_hypotheses",
            {
                "hypotheses": [
                    {"statement": "a", "node_id": "n1"},
                    {"statement": "b", "node_id": "n2"},
                ]
            },
        )
        target = _dispatch(engine, "get_next_targets", {})[0]
        result = _dispatch(
            engine,
            "record_evidence",
            {
                "node_id": target["node_id"],
                "success": 1.0,
                "claim_id": target["claim_id"],
                "count_next_targets": 1,
            },
        )
        assert len(result["next_targets"]) == 1
        assert result["next_targets"][0]["status"] == "SELECTED"
        assert result["next_targets"][0]["node_id"] != target["node_id"]
    finally:
        engine.close()


@pytest.mark.e2e
def test_a_lease_can_be_renewed_and_handed_back(tmp_path: Path) -> None:
    """Neither is reachable any other way, and both are needed off the agent loop.

    An experiment that runs for days outlives any TTL short enough to reclaim
    work from a caller that vanished; and deciding not to run a dispatched
    experiment previously left only two options, fabricating a result or
    stranding the node for the whole lease.
    """
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "leases.db", rng_seed=11)
    try:
        _dispatch(
            engine, "create_hypotheses", {"hypotheses": [{"statement": "a", "node_id": "n1"}]}
        )
        target = _dispatch(engine, "get_next_targets", {})[0]

        renewed = _dispatch(
            engine, "renew_claim", {"claim_id": target["claim_id"], "lease_ttl_s": 86400}
        )
        assert renewed["node_id"] == "n1"
        assert renewed["expires_in_s"] == 86400

        released = _dispatch(engine, "release_claims", {"claim_ids": [target["claim_id"]]})
        assert released["released_node_ids"] == ["n1"]
        assert _dispatch(engine, "get_active_claims", {}) == []
    finally:
        engine.close()


@pytest.mark.e2e
def test_conflict_tools_are_callable_through_the_server(tmp_path: Path) -> None:
    """The conflict machinery is only usable if it is reachable over MCP.

    Exercised end-to-end rather than against the engine directly, because the
    dispatch wiring is exactly what a new tool tends to be missing.
    """
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch
    from hypotree.models.edge import EdgeType
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(tmp_path / "mcp_conflicts.db", rng_seed=3)
    try:
        for group, ids in (("component", ("c1", "c2")), ("regime", ("r1", "r2"))):
            for nid in ids:
                engine.create_hypothesis(nid, node_id=nid, exclusion_group=group)
        engine.create_hypothesis(
            "combo", node_id="combo", parent_ids=["c1", "r1"], edge_type=EdgeType.DEPENDENCY
        )
        engine.record_evidence("c1", LogicalEvidence(success=1.0))
        engine.record_evidence("r1", LogicalEvidence(success=1.0))
        engine.record_evidence("combo", LogicalEvidence(success=0.0))

        conflicts = _dispatch(engine, "get_conflicts", {})["conflicts"]
        assert len(conflicts) == 1
        assert sorted(conflicts[0]["member_ids"]) == ["c1", "r1"]

        suggestion = _dispatch(engine, "suggest_discriminating_experiment", {})
        assert suggestion["status"] == "SUGGESTED"
        # The cheapest next move is a single-assumption swap: one probe
        # eliminates a whole question, where re-testing an assumption on its own
        # re-asks something it has already answered.
        assert suggestion["action"] == "substitute"
        assert suggestion["node_id"] in {"c1", "r1"}
        assert suggestion["replace_with"] in {"c2", "r2"}
    finally:
        engine.close()


@pytest.mark.e2e
def test_generate_learning_path_is_reachable_over_the_tool_surface(tmp_path: Path) -> None:
    """The narrative has to survive JSON serialisation to be worth anything."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch
    from hypotree.models.evidence import LogicalEvidence

    engine = HypoTreeEngine(tmp_path / "mcp_learning.db", rng_seed=3)
    try:
        for nid in ("a", "b"):
            engine.create_hypothesis(nid, node_id=nid, exclusion_group="q")
        engine.record_evidence("a", LogicalEvidence(success=1.0))

        result = _dispatch(engine, "generate_learning_path", {})

        assert json.loads(json.dumps(result, default=str))
        assert result["probes_spent"] == 1
        assert result["conclusions_without_a_probe"] == 1
        assert "What we have learned so far" in result["markdown"]
    finally:
        engine.close()


@pytest.mark.e2e
def test_the_three_slash_commands_are_offered_as_prompts() -> None:
    """A human types `/hypotree-next`; the model must receive the real instruction.

    Left to paraphrase it, an agent asks for targets it does not intend to probe
    and records results against whatever node it happens to be looking at — the
    two mistakes the whole protocol exists to prevent.
    """
    from hypotree.mcp_server import _prompt_definitions, _prompt_text

    names = {p.name for p in _prompt_definitions()}
    assert names == {"hypotree-init", "hypotree-next", "hypotree-status"}

    assert "create_hypotheses" in _prompt_text("hypotree-init", None)
    assert "is_goal=True" in _prompt_text("hypotree-init", None)
    assert "get_next_targets" in _prompt_text("hypotree-next", None)
    assert "record_evidence" in _prompt_text("hypotree-next", None)
    assert "generate_learning_path" in _prompt_text("hypotree-status", None)


@pytest.mark.e2e
def test_the_init_prompt_carries_the_task_when_one_is_given() -> None:
    """The argument is optional, so both shapes have to produce a usable instruction."""
    from hypotree.mcp_server import _prompt_text

    with_task = _prompt_text("hypotree-init", {"task": "cut p99 latency"})
    without = _prompt_text("hypotree-init", {})

    assert "cut p99 latency" in with_task
    assert "The task is" not in without


@pytest.mark.e2e
def test_an_unknown_prompt_is_refused() -> None:
    from hypotree.mcp_server import _prompt_text

    with pytest.raises(ValueError, match="Unknown prompt"):
        _prompt_text("hypotree-nope", None)


@pytest.mark.e2e
def test_the_guide_ships_inside_the_package(tmp_path: Path) -> None:
    """An agent on `uvx hypotree` has no repo, so a repo-root guide reaches nobody."""
    from hypotree.mcp_server import _agent_guide

    guide = _agent_guide()
    assert "exclusion_group" in guide
    assert "get_next_targets" in guide


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_resources_serve_the_guide_and_the_live_state(tmp_path: Path) -> None:
    """Two resources: one static contract, one derived from the workspace."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import (
        GUIDE_RESOURCE_URI,
        STATE_RESOURCE_URI,
        _read_resource,
        _resource_definitions,
    )

    assert {str(r.uri) for r in _resource_definitions()} == {
        GUIDE_RESOURCE_URI,
        STATE_RESOURCE_URI,
    }

    engine = HypoTreeEngine(tmp_path / "mcp_res.db", rng_seed=3)
    lock = asyncio.Lock()
    try:
        assert "exclusion_group" in await _read_resource(engine, lock, GUIDE_RESOURCE_URI)
        assert "What we have learned" in await _read_resource(engine, lock, STATE_RESOURCE_URI)
        with pytest.raises(ValueError, match="Unknown resource"):
            await _read_resource(engine, lock, "hypotree://nope")
    finally:
        engine.close()


@pytest.mark.e2e
def test_the_server_states_its_contract_at_handshake() -> None:
    """MCP hands `instructions` to the model before the first call — use the slot.

    Without it, the operating rules are only ever in context if the agent
    happens to read a resource, which is exactly the failure mode of shipping
    them as a repo-root markdown file.
    """
    from hypotree.mcp_server import SERVER_INSTRUCTIONS

    assert "is_goal=True" in SERVER_INSTRUCTIONS
    assert "exclusion_group" in SERVER_INSTRUCTIONS
    assert "hypotree://guide" in SERVER_INSTRUCTIONS
    # Short enough to survive being prepended to every conversation.
    assert len(SERVER_INSTRUCTIONS) < 2000


@pytest.mark.e2e
def test_the_cli_answers_help_and_version_instead_of_hanging(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An MCP server started by hand looks identical to a hang.

    "Did the install work?" is the first thing anyone types, and blocking on
    stdin is the worst possible answer. Verified against a wheel installed into
    a clean venv, not only in-process.
    """
    from hypotree import __version__
    from hypotree.mcp_server import main

    monkeypatch.setattr(sys, "argv", ["hypotree", "--version"])
    main()
    assert __version__ in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["hypotree", "--help"])
    main()
    help_text = capsys.readouterr().out
    assert "HYPOTREE_WORKSPACE_ID" in help_text
    assert "will appear to hang" in help_text


@pytest.mark.e2e
def test_the_cli_reports_the_resolved_workspace(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--info` is the second thing anyone needs: which belief state is this?"""
    from hypotree.mcp_server import main

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "cli-check")
    monkeypatch.setattr(sys, "argv", ["hypotree", "--info"])

    main()

    info = json.loads(capsys.readouterr().out)
    assert info["workspace_id"] == "cli-check"
    assert info["resolved_from"] == "env"


@pytest.mark.e2e
def test_an_unknown_flag_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently starting a JSON-RPC loop on a typo would look like a hang."""
    from hypotree.mcp_server import main

    monkeypatch.setattr(sys, "argv", ["hypotree", "--nope"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.e2e
def test_get_workspace_info_is_reachable_over_the_tool_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer to "why is my graph empty?" has to be askable by the agent."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "tool-check")

    engine = HypoTreeEngine(tmp_path / "wi.db", project_path=tmp_path, rng_seed=3)
    try:
        info = _dispatch(engine, "get_workspace_info", {})
        assert json.loads(json.dumps(info, default=str))
        assert info["workspace_id"] == "tool-check"
        assert info["resolved_from"] == "env"
    finally:
        engine.close()
