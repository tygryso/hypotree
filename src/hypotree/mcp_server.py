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
"""

from __future__ import annotations

import asyncio
import json
import sys
from importlib import resources

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from hypotree import __version__
from hypotree.engine import EvidenceReport, HypoTreeEngine
from hypotree.models.evidence import Evidence, InfraError, LogicalEvidence
from hypotree.store.identity import resolve_project_path, store_root

GUIDE_RESOURCE_URI = "hypotree://guide"
STATE_RESOURCE_URI = "hypotree://state"

# Handed to the client in the initialize response, so it is in context before the
# first tool call and does not have to be re-sent with each one. Kept short on
# purpose: an operating contract nobody reads is worth less than four rules
# everybody does. Everything else is a resource the agent can pull.
SERVER_INSTRUCTIONS = """\
hypotree is a persistent belief state, not a notebook. It records hypotheses,
what settled them, and what follows from that — across sessions.

The four rules that matter:

1. One hypothesis per node, stated as a claim that could turn out false. Wire a
   node to what it assumes with `parent_ids`, and give competing answers to the
   same question a shared `exclusion_group` — that is what lets confirming one
   retire the rest for free.
2. Mark the objective with `is_goal=True` and wire it to the work meant to
   achieve it. A goal is never handed out as a target and never accepts
   evidence; it is reached when its DEPENDENCY parents are verified.
3. Record every result against the node whose statement you actually tested.
   Filing a composition's failure against a premise corrupts a confirmation that
   is still true on its own.
4. Ask `get_next_targets` for work, and report it. A target is leased to you
   until you record it; anything you hold and never report is work nobody can do.

When the navigator returns no target it says why. `awaiting_evidence`,
`awaiting_composition`, `awaiting_substitution` and `blocked_frontier` are
instructions, not endings — read the rationale and act on it.

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
            "`awaiting_substitution` means run the one swap it describes, and "
            "`blocked_frontier` means the edges are wired so nothing is reachable — "
            "fix the graph. Only `all_goals_met` and `empty_frontier` mean stop."
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


def _hypothesis_item_schema() -> dict:
    """Per-item schema for create_hypotheses.

    Kept deliberately flat and fully described in one place: the batch tool and
    the singular one used to carry two different versions of this, and the
    richer of the two was the one callers got wrong.
    """
    return {
        "type": "object",
        "properties": {
            "statement": {"type": "string", "description": "The claim being made."},
            "parent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids this hypothesis is wired to. They may be created "
                "by this same call, in any order \u2014 dependencies are sorted out for you.",
            },
            "edge_type": {
                "type": "string",
                "enum": ["DEPENDENCY", "ALTERNATIVE", "REFINEMENT"],
                "default": "DEPENDENCY",
            },
            "is_parametric": {"type": "boolean", "default": False},
            "evidence_regime": {
                "type": "string",
                "enum": ["deterministic", "stochastic"],
                "default": "deterministic",
            },
            "is_goal": {
                "type": "boolean",
                "default": False,
                "description": "Marks an objective rather than a testable claim. A goal "
                "is never handed out as a target and records no evidence; it is "
                "reached when every hypothesis it DEPENDS on is verified, so wire it "
                "to the work meant to achieve it. Set this on nothing else \u2014 a node "
                "marked as a goal can never be tested, refuted or settled.",
            },
            "target_metric": {"type": "number"},
            "exclusion_group": {
                "type": "string",
                "description": "Name of a mutually-exclusive set: competing answers "
                "to the same question, of which exactly one can be true. Confirming "
                "any member automatically settles the others as EXHAUSTED (no need "
                "to test them), and that inference is undone if the confirmation is "
                "later withdrawn.",
            },
            "param_config": {"type": "object"},
            "node_id": {"type": "string"},
            "if_exists": {
                "type": "string",
                "enum": ["error", "overwrite", "skip"],
                "default": "error",
                "description": "Collision policy when node_id already exists. "
                "'error' raises, 'overwrite' replaces, 'skip' returns existing.",
            },
        },
        "required": ["statement"],
    }


def _tool_definitions() -> list[types.Tool]:
    """Static tool schema definitions \u2014 one per engine method."""
    return [
        types.Tool(
            name="create_hypotheses",
            description="Add one or many hypothesis nodes (with optional parent edges). "
            "Pass a list of one to create a single hypothesis. Parents may be created by "
            "the same call in any order. The whole batch is validated before anything is "
            "written, so a rejected call creates nothing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": 1,
                        "items": _hypothesis_item_schema(),
                    }
                },
                "required": ["hypotheses"],
            },
        ),
        types.Tool(
            name="get_next_targets",
            description="Reclaim stale leases and select the next target(s). A claimed "
            "node is reserved for you until you record its result, so ask only for what "
            "you will probe before your next call — anything you hold and do not report "
            "is work nobody can do. A batch never contains two competing answers to the "
            "same question. Returns a list; each entry may carry min_depth when the node "
            "is under conflict review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "How many targets to claim in this call.",
                    },
                    "lease_ttl_s": {
                        "type": "integer",
                        "description": "Override the claim TTL in seconds (default 900). "
                        "Raise it for experiments that run for hours or days, or keep it "
                        "short and call renew_claim while the work is still going.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Peek at the selection without issuing a claim or TTL.",
                    },
                },
            },
        ),
        types.Tool(
            name="record_evidence",
            description="Record one result, or many in one call, and update the belief "
            "state. Auto-captures git context_hash + git_branch when unset. Record "
            "against the hypothesis whose statement you actually tested: evidence "
            "against a goal is refused, and evidence against a premise a composition "
            "assumed corrupts a confirmation that is still true on its own. Ran several "
            "experiments this turn? Report them together with `results` — one call, "
            "applied in order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "Several results at once, applied in the order "
                        "given. Use this whenever you ran more than one experiment: "
                        "reporting k results costs one call instead of k. Each entry "
                        "takes the same fields as a single result. When present, the "
                        "single-result fields below are ignored.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "success": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "depth": {"type": "integer", "default": 0, "minimum": 0},
                                "claim_id": {"type": "string"},
                                "evidence_kind": {
                                    "type": "string",
                                    "enum": ["logical", "infra"],
                                    "default": "logical",
                                },
                                "error_type": {"type": "string"},
                                "message": {"type": "string"},
                                "metrics": {"type": "object"},
                                "source_ref": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            "required": ["node_id"],
                        },
                    },
                    "node_id": {"type": "string"},
                    "success": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "depth": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Rigour/scale of the test that produced this result. "
                        "A confirmation at depth d only supports claims tested no "
                        "deeper than d.",
                    },
                    "claim_id": {
                        "type": "string",
                        "description": "The claim this result answers. Optional: omit it "
                        "entirely for a probe you initiated yourself, which is always "
                        "safe. Pass the one get_next_targets issued for work it handed "
                        "you, so the lease is released.",
                    },
                    "evidence_kind": {
                        "type": "string",
                        "enum": ["logical", "infra"],
                        "default": "logical",
                    },
                    "error_type": {"type": "string"},
                    "message": {"type": "string"},
                    "metrics": {"type": "object"},
                    "source_ref": {
                        "type": "string",
                        "description": (
                            "What was actually run to produce this number — a file path, "
                            "a URL, a CI run id, a commit. Optional, but a trail that says "
                            "'0.85, from pytest run #4412' is worth more later than one "
                            "that says '0.85'."
                        ),
                    },
                    "notes": {"type": "string"},
                    "count_next_targets": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "How many targets you want to be holding when this "
                        "returns \u2014 a top-up, not an addition, so recording a batch of "
                        "results leaves you with this many, not this many per result. "
                        "Saves a separate get_next_targets round-trip. Leave at 0 when you "
                        "are reporting a long-running experiment and are not ready to claim "
                        "more work \u2014 anything claimed and not reported is work nobody "
                        "else can do.",
                    },
                    "lease_ttl_s": {
                        "type": "integer",
                        "description": "TTL for the fused dispatch, if any.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="renew_claim",
            description="Restart a live lease's clock because the experiment is still "
            "running. Use this instead of a very long TTL: the lease exists so work held "
            "by a caller that vanished comes back, and a long TTL makes that recovery as "
            "slow as the longest experiment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "lease_ttl_s": {"type": "integer", "minimum": 1},
                },
                "required": ["claim_id"],
            },
        ),
        types.Tool(
            name="release_claims",
            description="Hand leased nodes back without recording a result \u2014 for work you "
            "have decided not to run, or for resuming after a context reset you cannot "
            "report on. Omit claim_ids to release everything you hold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Release only these; omit to release every live lease.",
                    }
                },
            },
        ),
        types.Tool(
            name="update_status",
            description="Manually override the status of one or many nodes. Every id is "
            "validated before anything changes, so a bad id never leaves a partial update.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "new_status": {
                        "type": "string",
                        "enum": [
                            "UNTESTED",
                            "IN_PROGRESS",
                            "VERIFIED",
                            "EXHAUSTED",
                            "INVALIDATED",
                            "PRUNED",
                            "BLOCKED",
                            "NEEDS_REVISION",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["node_ids", "new_status"],
            },
        ),
        types.Tool(
            name="invalidate_upstream",
            description="Walk DEPENDENCY ancestors, flip VERIFIED → NEEDS_REVISION.",
            inputSchema={
                "type": "object",
                "properties": {"leaf_id": {"type": "string"}},
                "required": ["leaf_id"],
            },
        ),
        types.Tool(
            name="verify_upstream",
            description="Walk REFINEMENT ancestors, flip IN_PROGRESS → VERIFIED (depth-capped).",
            inputSchema={
                "type": "object",
                "properties": {"child_id": {"type": "string"}},
                "required": ["child_id"],
            },
        ),
        types.Tool(
            name="get_goal_status",
            description="Report all goal nodes, target metrics, progress counts, "
            "and whether the global stop holds.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_conflicts",
            description="List recorded conflicts — sets of assumptions that cannot all "
            "hold together, with which members are exonerated and which remain suspects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "open_only": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only conflicts whose culprit is not yet pinned.",
                    }
                },
            },
        ),
        types.Tool(
            name="suggest_discriminating_experiment",
            description="Propose the single most informative next experiment: re-test a "
            "conflict suspect at depth while any remains, otherwise the closest "
            "alternative combination that no recorded conflict rules out.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_dag_context",
            description="Return a depth+width-bounded subgraph with credible intervals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 2},
                    "max_children": {"type": "integer", "default": 10},
                },
            },
        ),
        types.Tool(
            name="render_dag_map",
            description="Render a Mermaid flowchart with depth+width bounding + elision.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 2},
                    "max_children": {"type": "integer", "default": 10},
                    "hide_statuses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Drop nodes matching these statuses (e.g. ['PRUNED']).",
                    },
                },
            },
        ),
        types.Tool(
            name="list_nodes",
            description="Query/filter/sort nodes and return a Markdown table. "
            "Use `view` for the questions actually worth asking — 'frontier' (what "
            "is still open), 'settled', 'verified', 'revision' (what is under "
            "revision), 'stale' — rather than assembling a status filter by hand. "
            "`stale_only=true` keeps only confirmations made against a commit that "
            "is no longer checked out: they are not refuted, but nothing has "
            "re-established them since the code moved.",
            inputSchema={
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["frontier", "settled", "verified", "revision", "stale"],
                        "description": (
                            "Named filter preset; overridden by an explicit status_filter."
                        ),
                    },
                    "stale_only": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Keep only VERIFIED nodes confirmed against a non-HEAD commit."
                        ),
                    },
                    "status_filter": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "UNTESTED",
                                "IN_PROGRESS",
                                "VERIFIED",
                                "INVALIDATED",
                                "PRUNED",
                                "BLOCKED",
                                "NEEDS_REVISION",
                            ],
                        },
                        "description": "Filter to these statuses only",
                    },
                    "query_filter": {
                        "type": "string",
                        "description": "Case-insensitive statement search. "
                        "`*` = multi-char wildcard, `_` = single-char. "
                        "Literal % and _ are escaped.",
                    },
                    "order_by": {
                        "type": "string",
                        "enum": [
                            "created_at",
                            "updated_at",
                            "verified_at",
                            "pruned_at",
                            "invalidated_at",
                            "posterior_mean",
                            "evidence_count",
                            "staleness",
                        ],
                        "default": "created_at",
                    },
                    "ascending": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 20},
                    "offset": {"type": "integer", "default": 0},
                },
            },
        ),
        types.Tool(
            name="get_evidence_history",
            description="Return the evidence trail for a node (newest-first).",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["node_id"],
            },
        ),
        types.Tool(
            name="get_active_claims",
            description="Return live (unconsumed, unexpired) claims for resuming work.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="generate_learning_path",
            description="Narrate what has been settled so far, in order, and how — "
            "separating what an experiment paid for from what the engine inferred for "
            "free, and calling out beliefs that were later withdrawn. Use it to brief "
            "a human, to write a summary, or to re-orient yourself after a context "
            "reset: the other read tools show the current state, this one shows how it "
            "was arrived at.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "description": "Cap on narrated transitions (most recent first). "
                        "Counters always cover the whole history.",
                    }
                },
            },
        ),
        types.Tool(
            name="get_workspace_info",
            description="Which belief state you are connected to, and how it was chosen. "
            "Reports the workspace id, which of the four resolution layers produced it, "
            "where the database lives and whether it exists yet. Call it when the graph "
            "is unexpectedly empty or two clients disagree about what has been "
            "established — that is almost always one project resolving to two workspaces.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def _run_main() -> None:
    """Wire the engine into an MCP stdio server and run it."""
    project_path = resolve_project_path()
    db_path = store_root(project_path) / "state.db"

    engine = HypoTreeEngine(db_path, project_path=project_path)
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

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


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
    ]


async def _read_resource(engine: HypoTreeEngine, write_lock: asyncio.Lock, uri: str) -> str:
    """Serve a resource. The live one goes through the write lock like any read."""
    if uri == GUIDE_RESOURCE_URI:
        return _agent_guide()
    if uri == STATE_RESOURCE_URI:
        async with write_lock:
            return engine.generate_learning_path().markdown
    raise ValueError(f"Unknown resource: {uri}")


async def _handle_call_tool(
    engine: HypoTreeEngine,
    write_lock: asyncio.Lock,
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    """Serialize a single tool call behind the engine write lock.

    Extracted from the stdio wiring so the dispatch + JSON encoding path is
    exercisable in-process; the lock guarantees engine mutations never interleave
    across concurrent tool calls.
    """
    async with write_lock:
        result = _dispatch(engine, name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, default=str))]


def _evidence_report(item: dict) -> EvidenceReport:
    """Build one evidence report from a tool-call payload.

    Shared by the single and batch shapes so the two can never drift — the
    `source_ref` field was advertised on the tool for a release while the
    dispatch that was supposed to read it silently dropped it.
    """
    ev: Evidence
    if item.get("evidence_kind", "logical") == "infra":
        ev = InfraError(
            error_type=item.get("error_type", "unknown"),
            message=item.get("message", ""),
        )
    else:
        ev = LogicalEvidence(
            success=item.get("success", 0.5),
            depth=item.get("depth", 0),
            metrics=item.get("metrics", {}),
            notes=item.get("notes", ""),
            source_ref=item.get("source_ref"),
        )
    return EvidenceReport(
        node_id=item["node_id"],
        evidence=ev,
        claim_id=item.get("claim_id"),
    )


def _dispatch(engine: HypoTreeEngine, name: str, arguments: dict) -> object:
    """Route a tool call to the corresponding engine method."""
    if name == "create_hypotheses":
        results = engine.create_hypotheses(arguments["hypotheses"])
        return [r.model_dump(mode="json") for r in results]

    if name == "get_next_targets":
        count = arguments.pop("count", 1)
        ttl = arguments.pop("lease_ttl_s", None)
        dry_run = arguments.pop("dry_run", False)
        targets = engine.get_next_targets(count=count, lease_ttl_s=ttl, dry_run=dry_run)
        return [t.model_dump(mode="json") for t in targets]

    if name == "record_evidence":
        count_next = arguments.pop("count_next_targets", 0)
        ttl = arguments.pop("lease_ttl_s", None)
        raw_results = arguments.pop("results", None)
        if raw_results:
            reports = [_evidence_report(item) for item in raw_results]
            batch = engine.record_results(reports, count_next_targets=count_next, lease_ttl_s=ttl)
            return batch.model_dump(mode="json")
        report = _evidence_report(arguments)
        result = engine.record_evidence(
            report.node_id,
            report.evidence,
            claim_id=report.claim_id,
            count_next_targets=count_next,
            lease_ttl_s=ttl,
        )
        return result.model_dump(mode="json")

    if name == "renew_claim":
        claim = engine.renew_claim(arguments["claim_id"], arguments.get("lease_ttl_s"))
        return claim.model_dump(mode="json")

    if name == "release_claims":
        return {"released_node_ids": engine.release_claims(arguments.get("claim_ids"))}

    if name == "update_status":
        from hypotree.models.status import Status

        results = engine.bulk_update_status(
            arguments["node_ids"],
            Status(arguments["new_status"]),
            arguments.get("reason", ""),
        )
        return [r.model_dump(mode="json") for r in results]

    if name == "invalidate_upstream":
        return {"affected_ids": engine.invalidate_upstream(arguments["leaf_id"])}

    if name == "verify_upstream":
        return {"affected_ids": engine.verify_upstream(arguments["child_id"])}

    if name == "get_goal_status":
        resp = engine.get_goal_status()
        return resp.model_dump(mode="json")

    if name == "get_conflicts":
        return {"conflicts": engine.get_conflicts(arguments.get("open_only", True))}

    if name == "suggest_discriminating_experiment":
        return engine.suggest_discriminating_experiment()

    if name == "get_dag_context":
        resp = engine.get_dag_context(
            arguments.get("node_id"),
            arguments.get("max_depth", 2),
            arguments.get("max_children", 10),
        )
        return resp.model_dump(mode="json")

    if name == "render_dag_map":
        return {
            "mermaid": engine.render_dag_map(
                arguments.get("node_id"),
                arguments.get("max_depth", 2),
                arguments.get("max_children", 10),
                arguments.get("hide_statuses"),
            )
        }

    if name == "list_nodes":
        return {
            "table": engine.list_nodes(
                status_filter=arguments.get("status_filter"),
                query_filter=arguments.get("query_filter"),
                view=arguments.get("view"),
                stale_only=bool(arguments.get("stale_only", False)),
                order_by=arguments.get("order_by", "created_at"),
                ascending=arguments.get("ascending", False),
                limit=arguments.get("limit", 20),
                offset=arguments.get("offset", 0),
            )
        }

    if name == "get_evidence_history":
        rows = engine.get_evidence_history(
            arguments["node_id"],
            limit=arguments.get("limit", 20),
            offset=arguments.get("offset", 0),
        )
        return [r.model_dump(mode="json") for r in rows]

    if name == "get_active_claims":
        claims = engine.get_active_claims()
        return [c.model_dump(mode="json") for c in claims]

    if name == "generate_learning_path":
        return engine.generate_learning_path(limit=arguments.get("limit", 200)).model_dump(
            mode="json"
        )

    if name == "get_workspace_info":
        from hypotree.store.identity import workspace_diagnostics

        return workspace_diagnostics(engine.project_path)

    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    """Entry point for the `hypotree` console script.

    An MCP server speaks JSON-RPC on stdin, so running it by hand looks
    identical to a hang. `--help` and `--version` are handled before the loop
    starts, because "did the install work?" is the first thing anyone types and
    a silent block is the worst possible answer to it. `--info` prints the
    resolved workspace, which is the second thing they need.
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
    if args:
        print(f"hypotree: unknown option {args[0]!r}\n", file=sys.stderr)
        print(_cli_help(), file=sys.stderr)
        raise SystemExit(2)
    asyncio.run(_run_main())


def _cli_help() -> str:
    return f"""\
hypotree {__version__} — persistent, self-revising hypothesis DAG (MCP server)

Usage:
  hypotree             Run the MCP server on stdio (what an MCP client does).
  hypotree --info      Print the resolved workspace, store path and warnings.
  hypotree --version   Print the version.
  hypotree --help      Show this message.

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
