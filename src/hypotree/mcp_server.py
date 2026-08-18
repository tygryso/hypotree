"""MCP stdio adapter — tools, prompts, resources and the server-level contract.

Wires every HypoTreeEngine method as an MCP tool over stdio. The engine
mutations are serialized behind an asyncio.Lock so interleaved tool calls
never overlap two SQLite transactions.

The operating rules reach the agent through four channels, deliberately, because
each fails differently:

- **Server instructions** (below) are handed to the model at handshake and cost
  nothing per call. The irreducible contract lives here.
- **Tool descriptions** are the only text guaranteed to be in context at the
  moment a tool is chosen, so each carries the one rule that tool is misused
  without.
- **Prompts** are the client's slash commands: a human types `/hypotree-next`
  and the model receives a correctly-phrased instruction instead of an
  approximation of one.
- **Resources** are pulled on demand. The full guide is 23 KB and belongs
  nowhere near a system prompt, but an agent that hits something it does not
  understand can read it.

The tool schemas and the routing table are *not* here. They live in
``hypotree.toolkit``, which imports no transport at all, and this module is one
of its two projections — the other being any Python host that embeds the engine
directly. Keeping them here made them unreachable for callers that were not
speaking JSON-RPC, which is how the evaluation harness ended up hand-writing a
second copy of seven schemas and letting them drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from importlib import resources
from pathlib import Path

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from hypotree import __version__
from hypotree.engine import HypoTreeEngine
from hypotree.store.identity import resolve_project_path, store_root
from hypotree.toolkit.dispatch import dashboard_url, dispatch, publish_dashboard_url
from hypotree.toolkit.specs import TOOL_SPECS

GUIDE_RESOURCE_URI = "hypotree://guide"
STATE_RESOURCE_URI = "hypotree://state"
DASHBOARD_RESOURCE_URI = "hypotree://dashboard"

# Re-exported so `from hypotree.mcp_server import dashboard_url` keeps working;
# the registry itself moved next to the dispatch that answers with it.
_publish_dashboard_url = publish_dashboard_url


def _tool_definitions() -> list[types.Tool]:
    """The shared specs, projected into MCP's tool type."""
    return [
        types.Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema,
        )
        for spec in TOOL_SPECS
    ]


SERVER_INSTRUCTIONS = """\
hypotree is a persistent belief state, not a notebook. It records hypotheses,
what settled them, and what follows from that — across sessions.

The four rules that matter:

1. One hypothesis per node, stated as a claim that could turn out false. Wire a
   node to what it assumes with `parent_ids`, and give competing answers to the
   same question a shared `exclusion_group` — that is what lets confirming one
   retire the rest for free.
2. Mark the objective with `is_goal=True` and put the work meant to achieve it in
   the **goal's** `parent_ids` — edges run from what is assumed to what assumes
   it, so a goal is the last node in the chain, never the first. `parent_ids` on
   the work naming the goal is the inverse and is refused. A goal is never handed
   out as a target and never accepts evidence; it is reached when its DEPENDENCY
   parents are verified.
3. Record every result against the node whose statement you actually tested.
   Filing a composition's failure against a premise corrupts a confirmation that
   is still true on its own.
4. Ask `get_next_targets` for work, and report it. A target is leased to you
   until you record it; anything you hold and never report is work nobody can do.

When the navigator returns no target it says why. `awaiting_evidence`,
`awaiting_composition`, `awaiting_substitution`, `blocked_frontier` and
`dead_question` are instructions, not endings — read the rationale and act on it.

Read `hypotree://guide` for the full contract. Call `generate_learning_path` to
find out what has already been established before you start.
"""


def _prompt_definitions() -> list[types.Prompt]:
    """Slash commands the client exposes to the human.

    Three, not thirty: these are the moments where a human wants to steer and
    the exact wording decides whether the agent uses the belief state properly
    or improvises around it. Anything an agent should do unprompted belongs in
    the tool descriptions instead.
    """
    return [
        types.Prompt(
            name="hypotree-init",
            description="Start a new R&D investigation: create the goal and the first "
            "hypotheses under it.",
            arguments=[
                types.PromptArgument(
                    name="task",
                    description="What you are trying to achieve, in one sentence.",
                    required=False,
                )
            ],
        ),
        types.Prompt(
            name="hypotree-next",
            description="Get the next hypothesis to test and go and test it.",
            arguments=[],
        ),
        types.Prompt(
            name="hypotree-status",
            description="Summarise where the investigation stands and what it has cost.",
            arguments=[],
        ),
    ]


def _prompt_text(name: str, arguments: dict[str, str] | None) -> str:
    """The instruction a slash command pastes into the conversation."""
    args = arguments or {}
    if name == "hypotree-init":
        task = args.get("task", "").strip()
        subject = f" The task is: {task}" if task else ""
        return (
            "Initialise the hypotree belief state for this investigation."
            f"{subject}\n\n"
            "Call `create_hypotheses` once with the whole initial tree:\n"
            "- one node with `is_goal=True` stating the objective and its "
            "`target_metric`;\n"
            "- 3–5 hypotheses that could plausibly achieve it, each a claim that "
            "could turn out false;\n"
            "- where several are competing answers to the same question, give them a "
            "shared `exclusion_group` so confirming one retires the others;\n"
            "- wire the goal to them with `parent_ids` so progress is derivable.\n\n"
            "Then call `generate_learning_path` to confirm the shape is what you "
            "intended, and report the tree back to me before testing anything."
        )
    if name == "hypotree-next":
        return (
            "Call `get_next_targets(count=2)`.\n\n"
            "If it returns targets, test each hypothesis for real — run the code, the "
            "query or the experiment — then report them together in one call: "
            "`record_evidence(results=[{node_id, success, depth, claim_id}, ...])`, "
            "each entry against the node you actually tested, with a `success` score "
            "in [0, 1]. Do not guess a result and do not record against a different "
            "node.\n\n"
            "If it returns DONE, the `reason` tells you what to do: "
            "`awaiting_evidence` means report what you are already holding, "
            "`awaiting_composition` means build the combination its rationale names, "
            "`awaiting_substitution` means run the one swap it describes, "
            "`blocked_frontier` means the edges are wired so nothing is reachable — "
            "fix the graph, and `dead_question` means one question has run out of "
            "candidate answers — add the one you have not thought of. "
            "Only `all_goals_met` and `empty_frontier` mean stop."
        )
    if name == "hypotree-status":
        return (
            "Call `generate_learning_path`, then `get_goal_status`, then "
            "`render_dag_map`.\n\n"
            "Give me a short briefing: what is established and how we know it, what we "
            "have ruled out, anything we changed our minds about, which conflicts are "
            "still open, and what the next experiment should be. Say explicitly how "
            "many conclusions cost an experiment and how many the engine inferred for "
            "free."
        )
    raise ValueError(f"Unknown prompt: {name}")


def _agent_guide() -> str:
    """The full agent contract, read from the installed package.

    Shipped inside the package rather than read from the repo root: an agent
    connecting to a `uvx hypotree` server has no repo, and a guide that is only
    available to people who cloned the source is not documentation.
    """
    return resources.files("hypotree").joinpath("AGENT_GUIDE.md").read_text(encoding="utf-8")


async def _run_main(
    dashboard_port: int | None,
    cost_aware: bool = False,
    db_path_override: Path | None = None,
) -> None:
    """Wire the engine into an MCP stdio server and run it.

    ``dashboard_port`` of ``None`` starts no dashboard. Anything else is the
    *first* port tried, not the one bound: a second workspace on the same
    machine must not fail to start because the first took 7331.

    ``cost_aware`` ranks candidates by value per unit *observed* cost. Off by
    default: with it off every cost ratio is exactly 1.0 and selection is
    identical to a build that has never heard of cost, which is the only honest
    way to add a term to an acquisition function a frozen gate has scored.
    """
    project_path = resolve_project_path()
    db_path = db_path_override or (store_root(project_path) / "state.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = HypoTreeEngine(db_path, project_path=project_path, cost_aware=cost_aware)
    write_lock = asyncio.Lock()
    app = Server("hypotree", instructions=SERVER_INSTRUCTIONS)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_definitions()

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        return await _handle_call_tool(engine, write_lock, name, arguments)

    @app.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        return _prompt_definitions()

    @app.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        return _build_prompt(name, arguments)

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        return _resource_definitions()

    @app.read_resource()
    async def read_resource(uri: types.AnyUrl) -> str:
        return await _read_resource(engine, write_lock, str(uri))

    dashboard = None
    if dashboard_port is not None:
        from hypotree.dashboard import DashboardServer, choose_port

        async def apply_directive(node_id: str, mode: str, reason: str) -> dict[str, object]:
            # The one path from the viewer back into the belief state, and it
            # goes through the same lock every tool call does. A directive is
            # scheduling, never evidence: it never touches alpha/beta.
            async with write_lock:
                if mode == "clear":
                    return {"ok": engine._store.clear_directive(node_id), "node_id": node_id}
                if engine._store.get_node(node_id) is None:
                    return {"ok": False, "error": f"no such node {node_id!r}"}
                engine._store.set_directive(node_id, mode, reason, "human")
                return {"ok": True, "node_id": node_id, "mode": mode}

        # A dashboard that cannot bind must not take the MCP server down with
        # it. It is now on by default, so the failure modes of a viewer have
        # become the failure modes of the server itself unless they are
        # contained here — an occupied port range or a sandbox that forbids
        # listening would otherwise mean no agent tooling at all.
        try:
            dashboard = DashboardServer(
                db_path, port=choose_port(dashboard_port), writer=apply_directive
            )
            await dashboard.start()
            _publish_dashboard_url(dashboard.url)
            # stdout is the JSON-RPC channel. One line written there corrupts
            # the session, so the URL goes to stderr.
            print(f"hypotree dashboard: {dashboard.url}", file=sys.stderr, flush=True)
        except OSError as exc:
            dashboard = None
            print(
                f"hypotree: dashboard not started ({exc}). The MCP server is running "
                f"normally; pass --dashboard-port to pick a free port, or --no-dashboard "
                f"to stop trying.",
                file=sys.stderr,
                flush=True,
            )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )
    finally:
        if dashboard is not None:
            await dashboard.stop()
        _publish_dashboard_url(None)


async def _run_viewer(port: int, db_path_override: Path | None = None) -> None:
    """Serve the dashboard against an existing belief state, with no MCP server.

    The try-before-you-wire path: someone who has never configured an MCP client
    can point this at a workspace and watch it. Read-only throughout, so it is
    safe to run against a database an agent is actively writing.
    """
    from hypotree.dashboard import DashboardServer, choose_port

    db_path = db_path_override or (store_root(resolve_project_path()) / "state.db")
    if not db_path.exists():
        print(
            f"no belief state at {db_path}. Run `hypotree --info` to see which "
            f"workspace resolved, and start an agent against it first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    server = DashboardServer(db_path, port=choose_port(port))
    await server.start()
    print(f"hypotree dashboard: {server.url}", file=sys.stderr, flush=True)
    print("Ctrl-C to stop.", file=sys.stderr, flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


def _build_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    """Expand a slash command into the message the model receives."""
    prompt = next((p for p in _prompt_definitions() if p.name == name), None)
    if prompt is None:
        raise ValueError(f"Unknown prompt: {name}")
    return types.GetPromptResult(
        description=prompt.description,
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=_prompt_text(name, arguments)),
            )
        ],
    )


def _resource_definitions() -> list[types.Resource]:
    """What the agent can pull on demand rather than carry in its context."""
    return [
        types.Resource(
            uri=types.AnyUrl(GUIDE_RESOURCE_URI),
            name="hypotree agent guide",
            description="The full operating contract: every tool, the status lifecycle, "
            "exclusion groups, leases, confirmation depth, conflict sets, and the rules. "
            "Read it when something the engine did is not obvious.",
            mimeType="text/markdown",
        ),
        types.Resource(
            uri=types.AnyUrl(STATE_RESOURCE_URI),
            name="current belief state",
            description="What has been established so far and how — the narrative form of "
            "the workspace, not a snapshot. Cheaper than re-deriving it from the graph.",
            mimeType="text/markdown",
        ),
        types.Resource(
            uri=types.AnyUrl(DASHBOARD_RESOURCE_URI),
            name="live dashboard URL",
            description="Where a human can watch this belief state move, token included. "
            "Hand it over when someone asks to see the graph, the timeline or what an "
            "experiment cost.",
            mimeType="text/plain",
        ),
    ]


async def _read_resource(engine: HypoTreeEngine, write_lock: asyncio.Lock, uri: str) -> str:
    """Serve a resource. The live one goes through the write lock like any read."""
    if uri == GUIDE_RESOURCE_URI:
        return _agent_guide()
    if uri == STATE_RESOURCE_URI:
        async with write_lock:
            return engine.generate_learning_path().markdown
    if uri == DASHBOARD_RESOURCE_URI:
        url = dashboard_url()
        if url is None:
            return (
                "No dashboard is running. It starts by default alongside the MCP server; "
                "this workspace was started with --no-dashboard, or the port range was "
                "busy. `hypotree --no-mcp` serves it against the existing belief state "
                "without touching the running session."
            )
        return (
            f"{url}\n\nLocalhost only. The `t=` parameter is the session token, "
            f"regenerated every start — a link from a previous session will not open."
        )
    raise ValueError(f"Unknown resource: {uri}")


async def _handle_call_tool(
    engine: HypoTreeEngine,
    write_lock: asyncio.Lock,
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    """Serialize a single tool call behind the engine write lock.

    Extracted from the stdio wiring so the dispatch + JSON encoding path is
    exercisable in-process.

    The lock is held for reads as well as writes, and that costs nothing:
    ``dispatch`` is fully synchronous, so it runs to completion without ever
    yielding to the event loop and two calls cannot interleave whether or not a
    lock is held. Letting sensors bypass it would therefore buy no throughput —
    a slow read blocks the loop by being slow, not by being locked — while
    planting a real race for whoever later moves dispatch onto a thread. The
    lock is what makes that move safe, and the store's connection is
    thread-bound, so it has to stay.
    """
    async with write_lock:
        result = dispatch(engine, name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, default=str))]


def main() -> None:
    """Entry point for the `hypotree` console script.

    An MCP server speaks JSON-RPC on stdin, so running it by hand looks
    identical to a hang. `--help` and `--version` are handled before the loop
    starts, because "did the install work?" is the first thing anyone types and
    a silent block is the worst possible answer to it. `--info` prints the
    resolved workspace, which is the second thing they need.

    The dashboard runs by default. It was behind a flag while it was unproven,
    and the result was that the people it was built for never saw it: nobody
    opts into a feature they have not been shown. It binds loopback only, mints
    a fresh token per start, and cannot take the server down if it fails to
    bind — so the cost of it being on is a socket, and the cost of it being off
    was the whole feature.
    """
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(_cli_help())
        return
    if args and args[0] in ("-V", "--version"):
        print(f"hypotree {__version__}")
        return
    if args and args[0] == "--info":
        from hypotree.store.identity import workspace_diagnostics

        print(json.dumps(workspace_diagnostics(resolve_project_path()), indent=2, default=str))
        return

    try:
        port, dashboard, mcp, cost_aware, db_path = _parse_serve_args(args)
    except ValueError as exc:
        print(f"hypotree: {exc}\n", file=sys.stderr)
        print(_cli_help(), file=sys.stderr)
        raise SystemExit(2) from None

    if not mcp:
        if not dashboard:
            print("hypotree: --no-mcp and --no-dashboard leave nothing to run\n", file=sys.stderr)
            raise SystemExit(2)
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_run_viewer(port, db_path))
        return
    asyncio.run(
        _run_main(
            dashboard_port=port if dashboard else None,
            cost_aware=cost_aware,
            db_path_override=db_path,
        )
    )


def _parse_serve_args(args: list[str]) -> tuple[int, bool, bool, bool, Path | None]:
    """Resolve serving flags, including an optional explicit SQLite path.

    Hand-rolled rather than argparse because this entry point must stay
    import-cheap: it is spawned per MCP session, and the flags here do not
    justify the import.
    """
    from hypotree.dashboard.server import DEFAULT_PORT

    port = DEFAULT_PORT
    dashboard = True
    mcp = True
    cost_aware = False
    configured_path = os.environ.get("HYPOTREE_DB_PATH", "").strip()
    db_path = Path(configured_path).expanduser().resolve() if configured_path else None
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg == "--no-dashboard":
            dashboard = False
        elif arg == "--experimental-cost-aware":
            cost_aware = True
        elif arg == "--no-mcp":
            mcp = False
        elif arg == "--dashboard-port":
            if not rest:
                raise ValueError("--dashboard-port needs a port number")
            port = _port(rest.pop(0))
        elif arg.startswith("--dashboard-port="):
            port = _port(arg.split("=", 1)[1])
        elif arg == "--db-path":
            if not rest:
                raise ValueError("--db-path needs a path")
            db_path = Path(rest.pop(0)).expanduser().resolve()
        elif arg.startswith("--db-path="):
            raw_path = arg.split("=", 1)[1]
            if not raw_path:
                raise ValueError("--db-path needs a path")
            db_path = Path(raw_path).expanduser().resolve()
        else:
            raise ValueError(f"unknown option {arg!r}")
    return port, dashboard, mcp, cost_aware, db_path


def _port(raw: str) -> int:
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise ValueError(f"{raw!r} is not a port number")
    return int(raw)


def _default_port() -> int:
    """The dashboard's first-choice port, imported lazily to keep startup cheap."""
    from hypotree.dashboard.server import DEFAULT_PORT

    return DEFAULT_PORT


def _cli_help() -> str:
    return f"""\
hypotree {__version__} — persistent, self-revising hypothesis DAG (MCP server)

Usage:
  hypotree                            Run the MCP server on stdio (what an MCP client does),
                                      with the web dashboard beside it on port {_default_port()},
                                      probing upward if that one is taken.
  hypotree --dashboard-port PORT      Start the dashboard from PORT instead, still probing
                                      upward. Use it when several workspaces are open at once
                                      and you want a predictable address for this one.
  hypotree --no-dashboard             MCP server only, no socket opened.
  hypotree --no-mcp                   Dashboard only, against the existing belief state, with
                                      no MCP server. Read-only, so it is safe to point at a
                                      workspace an agent is actively writing. This is the
                                      try-before-you-wire path.
  hypotree --no-mcp --db-path PATH    Dashboard only against an explicit state.db. The same
                                      override is available as HYPOTREE_DB_PATH.
  hypotree --experimental-cost-aware  EXPERIMENTAL, off by default. Rank candidates by expected
                                      value per unit cost rather than by promise alone, from
                                      the `duration_s` your results report and the
                                      `estimated_cost` you declare. Use it when your
                                      experiments differ in cost by more than they differ in
                                      promise — a fine-tune against a unit test. Measured at
                                      77% less total cost for 1.5% more probes on a
                                      cost-weighted benchmark, but only against a scripted
                                      caller; it is expected to become the default once a
                                      full evaluation with a live model has scored it.
                                      Without it selection is exactly as it is today.
  hypotree --info                     Print the resolved workspace, store path and warnings.
  hypotree --version                  Print the version.
  hypotree --help                     Show this message.

The dashboard binds 127.0.0.1 only and mints a session token at startup; the URL
it prints (to stderr, because stdout is JSON-RPC) carries that token. An agent
can hand it to you from `get_workspace_info` or the `hypotree://dashboard`
resource. If it cannot bind, the MCP server still starts and says so.

Run with no arguments only from an MCP client: the server speaks JSON-RPC on
stdin and will appear to hang if you start it in a terminal.

Configure a client with:
  {{"mcpServers": {{"hypotree": {{"command": "uvx", "args": ["hypotree"]}}}}}}

Workspace resolution, highest priority first:
  1. HYPOTREE_WORKSPACE_ID environment variable
  2. workspace_id: in hypotree.yaml at the project root
  3. hash of the git remote
  4. hash of the project path (weakest — pin it with 1 or 2)

State is stored under XDG_DATA_HOME, %LOCALAPPDATA% on Windows, or
~/.local/share. Run `hypotree --info` to see exactly where."""


if __name__ == "__main__":
    main()
