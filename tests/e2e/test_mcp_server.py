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
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="stdio transport deadlocks under the Windows proactor loop in CI",
)
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
            assert len(tools.tools) == 20
            done.append("list_tools")

            res = await session.call_tool(
                "create_hypotheses", {"hypotheses": [{"statement": "e2e", "node_id": "n1"}]}
            )
            created = json.loads(res.content[0].text)[0]
            assert created["node"]["id"] == "n1"
            assert created["created"] is True
            done.append("create_hypotheses")

            # Wiring existing nodes is the forward-growth path, so it has to work
            # over the real transport rather than only in-process.
            await session.call_tool(
                "create_hypotheses", {"hypotheses": [{"statement": "e2e b", "node_id": "n2"}]}
            )
            res = await session.call_tool(
                "add_edges", {"edges": [{"src": "n1", "dst": "n2", "type": "DEPENDENCY"}]}
            )
            wired = json.loads(res.content[0].text)[0]
            assert wired["created"] is True and wired["src"] == "n1"
            done.append("add_edges")

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
        await asyncio.wait_for(_run(), timeout=900)
    except BaseException as exc:  # noqa: BLE001 — re-raised unless purely teardown noise
        if not _is_client_teardown_race(exc):
            raise

    assert done == [
        "initialize",
        "list_tools",
        "create_hypotheses",
        "add_edges",
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
        "add_edges",
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
        "what_would_change_my_mind",
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
def test_record_evidence_reports_a_whole_batch_in_one_call(tmp_path: Path) -> None:
    """The tool surface has to expose the batch shape, not just the engine."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "batch.db", rng_seed=7)
    try:
        _dispatch(
            engine,
            "create_hypotheses",
            {
                "hypotheses": [
                    {"statement": "a", "node_id": "n1"},
                    {"statement": "b", "node_id": "n2"},
                    {"statement": "c", "node_id": "n3"},
                ]
            },
        )
        result = _dispatch(
            engine,
            "record_evidence",
            {
                "results": [
                    {"node_id": "n1", "success": 1.0, "depth": 1},
                    {"node_id": "ghost", "success": 1.0},
                    {"node_id": "n2", "success": 0.0},
                ],
                "count_next_targets": 1,
            },
        )
        assert [r["node"]["id"] for r in result["recorded"]] == ["n1", "n2"]
        # One bad entry does not cost the two results that were paid for.
        assert [f["node_id"] for f in result["failed"]] == ["ghost"]
        # The top-up runs once, after the whole batch.
        assert len(result["next_targets"]) == 1
    finally:
        engine.close()


@pytest.mark.e2e
def test_source_ref_survives_the_tool_boundary(tmp_path: Path) -> None:
    """It was advertised on the tool for a release while the dispatch dropped it.

    An audit trail that says "0.85, from pytest run #4412" is the whole point of
    the field; one that says "0.85" is what shipped.
    """
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _dispatch

    engine = HypoTreeEngine(tmp_path / "ref.db", rng_seed=7)
    try:
        _dispatch(
            engine, "create_hypotheses", {"hypotheses": [{"statement": "a", "node_id": "n1"}]}
        )
        _dispatch(
            engine,
            "record_evidence",
            {"node_id": "n1", "success": 1.0, "source_ref": "pytest run #4412"},
        )
        history = _dispatch(engine, "get_evidence_history", {"node_id": "n1"})
        assert history[0]["source_ref"] == "pytest run #4412"
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
    """Three resources: a static contract, a derived narrative, and where to watch."""
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import (
        DASHBOARD_RESOURCE_URI,
        GUIDE_RESOURCE_URI,
        STATE_RESOURCE_URI,
        _publish_dashboard_url,
        _read_resource,
        _resource_definitions,
    )

    assert {str(r.uri) for r in _resource_definitions()} == {
        GUIDE_RESOURCE_URI,
        STATE_RESOURCE_URI,
        DASHBOARD_RESOURCE_URI,
    }

    engine = HypoTreeEngine(tmp_path / "mcp_res.db", rng_seed=3)
    lock = asyncio.Lock()
    try:
        assert "exclusion_group" in await _read_resource(engine, lock, GUIDE_RESOURCE_URI)
        assert "What we have learned" in await _read_resource(engine, lock, STATE_RESOURCE_URI)
        # Asked before a dashboard is up, the answer says how to get one rather
        # than reporting nothing.
        with_none = await _read_resource(engine, lock, DASHBOARD_RESOURCE_URI)
        assert "--no-mcp" in with_none
        _publish_dashboard_url("http://127.0.0.1:7331/?t=secret")
        try:
            live = await _read_resource(engine, lock, DASHBOARD_RESOURCE_URI)
            assert "http://127.0.0.1:7331/?t=secret" in live
        finally:
            _publish_dashboard_url(None)
        with pytest.raises(ValueError, match="Unknown resource"):
            await _read_resource(engine, lock, "hypotree://nope")
    finally:
        engine.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workspace_info_carries_the_dashboard_link(tmp_path: Path) -> None:
    """A human watching the agent asks for this by name.

    The URL is minted at startup and printed to stderr, which the model never
    sees — so the agent could not answer a question it is asked constantly.
    """
    from hypotree.engine import HypoTreeEngine
    from hypotree.mcp_server import _handle_call_tool, _publish_dashboard_url

    engine = HypoTreeEngine(tmp_path / "mcp_ws.db", rng_seed=3, project_path=tmp_path)
    lock = asyncio.Lock()
    _publish_dashboard_url("http://127.0.0.1:7409/?t=abc")
    try:
        out = await _handle_call_tool(engine, lock, "get_workspace_info", {})
        assert "http://127.0.0.1:7409/?t=abc" in out[0].text
    finally:
        _publish_dashboard_url(None)
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
def test_the_serving_flags_resolve_to_a_port_and_three_switches() -> None:
    """The dashboard is on by default, and turning both halves off is refused.

    Default-on is the point: it was behind a flag while it was unproven and the
    people it was built for never saw it. Cost-awareness is the opposite case —
    off until measured, because it adds a term to an acquisition function a
    frozen gate has already scored.
    """
    from hypotree.dashboard.server import DEFAULT_PORT
    from hypotree.mcp_server import _parse_serve_args

    assert _parse_serve_args([]) == (DEFAULT_PORT, True, True, False)
    assert _parse_serve_args(["--dashboard-port", "9001"]) == (9001, True, True, False)
    assert _parse_serve_args(["--dashboard-port=9001"]) == (9001, True, True, False)
    assert _parse_serve_args(["--no-dashboard"]) == (DEFAULT_PORT, False, True, False)
    assert _parse_serve_args(["--no-mcp"]) == (DEFAULT_PORT, True, False, False)
    assert _parse_serve_args(["--experimental-cost-aware"]) == (DEFAULT_PORT, True, True, True)

    for bad in (
        ["--bogus"],
        ["--dashboard-port"],
        ["--dashboard-port", "x"],
        ["--dashboard-port", "0"],
    ):
        with pytest.raises(ValueError):
            _parse_serve_args(bad)


@pytest.mark.e2e
def test_asking_for_neither_server_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--no-mcp --no-dashboard` leaves nothing to run, and a process that starts
    and does nothing is worse than one that says so."""
    from hypotree.mcp_server import main

    monkeypatch.setattr(sys, "argv", ["hypotree", "--no-mcp", "--no-dashboard"])
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


@pytest.mark.unit
def test_a_result_with_no_success_is_refused_rather_than_invented() -> None:
    """A default `success` writes a measurement nobody took.

    `record_evidence(node_id="X")` is an ordinary LLM truncation, and it used to
    default to 0.5 — which moves the posterior and, in the deterministic regime,
    settles the node outright as EXHAUSTED and retires its competing answers. The
    belief state may assert only what was observed or soundly inferred; a default
    value is the quietest possible way to break that.
    """
    from hypotree.mcp_server import _evidence_report

    with pytest.raises(ValueError, match="needs success"):
        _evidence_report({"node_id": "h1"})
    # And the refusal names the alternative, rather than a Python key.
    with pytest.raises(ValueError, match="evidence_kind='infra'"):
        _evidence_report({"node_id": "h1"})
    with pytest.raises(ValueError, match="needs node_id"):
        _evidence_report({"success": 1.0})
    # An explicit zero is a real measurement and must survive.
    assert _evidence_report({"node_id": "h1", "success": 0.0}).evidence.success == 0.0
    # An infra report legitimately carries no success.
    assert _evidence_report({"node_id": "h1", "evidence_kind": "infra"}).evidence.kind == "infra"


@pytest.mark.unit
def test_the_record_evidence_schema_requires_what_the_dispatch_requires() -> None:
    """Schema and dispatch drifting apart is how `source_ref` was lost for a release."""
    from hypotree.mcp_server import _tool_definitions

    tools = {t.name: t for t in _tool_definitions()}
    item = tools["record_evidence"].inputSchema["properties"]["results"]
    assert item["minItems"] == 1, "an empty batch is a mistake, not a no-op"
    assert set(item["items"]["required"]) == {"node_id", "success"}


@pytest.mark.unit
def test_every_lease_ttl_schema_refuses_a_dead_lease() -> None:
    """A zero-second lease expires the instant it is issued.

    The agent then runs the experiment and cannot file the result. `renew_claim`
    validated this and the two tools that *issue* leases did not.
    """
    from hypotree.mcp_server import _tool_definitions

    tools = {t.name: t for t in _tool_definitions()}
    for name in ("get_next_targets", "record_evidence", "renew_claim"):
        schema = tools[name].inputSchema["properties"]["lease_ttl_s"]
        assert schema.get("minimum") == 1, f"{name} accepts a lease that is dead on arrival"


@pytest.mark.unit
def test_exhausted_can_be_queried_directly() -> None:
    """It is the status the exclusion inference produces most, and the enum omitted it.

    The SDK enforces `inputSchema`, so `list_nodes(status_filter=["EXHAUSTED"])`
    was rejected outright — the `view="settled"` workaround bundles three other
    statuses.
    """
    from hypotree.mcp_server import _tool_definitions

    tools = {t.name: t for t in _tool_definitions()}
    enum = tools["list_nodes"].inputSchema["properties"]["status_filter"]["items"]["enum"]
    assert "EXHAUSTED" in enum
