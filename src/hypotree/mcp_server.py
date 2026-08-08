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
import contextlib
import json
import sys
from datetime import datetime
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
DASHBOARD_RESOURCE_URI = "hypotree://dashboard"

# The live dashboard URL, token included, or None when no dashboard is running.
# Module-level because the tool handler and the resource reader both need it and
# neither is on the call path that starts the server. An agent is routinely asked
# "where can I watch this?" and had no way to answer.
_DASHBOARD_URL: str | None = None


def _publish_dashboard_url(url: str | None) -> None:
    """Record where the dashboard is listening, for the tool and the resource."""
    global _DASHBOARD_URL
    _DASHBOARD_URL = url


def dashboard_url() -> str | None:
    """The running dashboard's URL with its session token, or None."""
    return _DASHBOARD_URL


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
                "description": "What this hypothesis RESTS ON \u2014 its premises, not its "
                "sub-tasks. Edges run from the thing assumed to the thing assuming it, "
                "so a premise is the PARENT of the combination that uses it, and a goal "
                "is the LAST node in that chain: put the work in the goal's parent_ids, "
                "never the goal in the work's. They may be created by this same call, "
                "in any order \u2014 dependencies are sorted out for you.",
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
            "exclusion_closed": {
                "type": "boolean",
                "default": True,
                "description": "Whether these are ALL the candidate answers. True (the "
                "default) licenses the engine to confirm the last one standing for free "
                "once every rival is ruled out \u2014 sound over a complete list, and an "
                "assertion of something false over a partial one. Pass false when the "
                "next candidate always exists: 'which learning rate', 'which prompt "
                "wording'. Confirming one member still retires the others either way; "
                "only the last-one-standing deduction is withheld.",
            },
            "param_config": {"type": "object"},
            "estimated_cost": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Roughly what testing this will cost, in seconds. A hint for "
                "ordering, never a claim about the hypothesis: it changes what gets tried "
                "next and never what the belief state asserts, and the first real "
                "`duration_s` supersedes it. Worth giving when the competing answers to one "
                "question differ in cost \u2014 a 30-second unit test against an overnight "
                "fine-tune \u2014 because the last answer standing is deduced rather than "
                "probed, so putting the expensive one last means never paying for it. Omit "
                "it when they all cost about the same; ordering is then free of it anyway.",
            },
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
            name="add_edges",
            description="Wire hypotheses that already exist, without recreating either. "
            "Use it to grow a graph forward: when a pipeline gains a stage, the goal must "
            "depend on the NEW last stage, or it reports itself achieved as soon as the "
            "first stage verifies while the rest sit untested. You do not need to remove "
            "the old edge — DEPENDENCY is AND and the later stage already depends on the "
            "earlier one, so adding only tightens the condition. Validated like creation: "
            "unknown nodes, a goal used as a DEPENDENCY parent, and cycles are refused "
            "before anything is written, and an edge that already exists is a no-op.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Edges to add. Each runs FROM the hypothesis being "
                        "assumed TO the one assuming it, so src is the parent.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {
                                    "type": "string",
                                    "description": "The hypothesis being assumed (parent).",
                                },
                                "dst": {
                                    "type": "string",
                                    "description": "The hypothesis that assumes it (child).",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["DEPENDENCY", "ALTERNATIVE", "REFINEMENT"],
                                    "default": "DEPENDENCY",
                                },
                            },
                            "required": ["src", "dst"],
                        },
                    },
                },
                "required": ["edges"],
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
                        "minimum": 1,
                        "description": "Override the claim TTL in seconds (default 900). "
                        "Raise it for experiments that run for hours or days, or keep it "
                        "short and call renew_claim while the work is still going.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Peek at the selection without issuing a claim or TTL.",
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Work on one objective only: that goal, everything it "
                        "depends on, and the competing answers to those questions. Omit to "
                        "draw from the whole workspace. If the filter leaves nothing testable "
                        "while untested work sits outside it, the reason is goal_scope_empty "
                        "and the fix is usually a missing DEPENDENCY edge, not a finished "
                        "search.",
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
                        "minItems": 1,
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
                                "duration_s": {"type": "number", "minimum": 0},
                                "notes": {"type": "string"},
                            },
                            "required": ["node_id", "success"],
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
                    "duration_s": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "How long the experiment took, in seconds. Optional, and worth "
                            "sending whenever your probes differ in cost: it is what lets the "
                            "navigator rank by value per unit cost rather than treating a "
                            "three-day run and a one-second check as interchangeable."
                        ),
                    },
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
                        "minimum": 1,
                        "description": "TTL for the fused dispatch, if any.",
                    },
                },
                # Enforced on the single-result shape only when `results` is absent;
                # the dispatch checks both, because a made-up success is a measurement
                # nobody took and there is no safe default for one.
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
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Report on one objective and count only the nodes "
                        "forming its case. Omit for every goal in the workspace.",
                    }
                },
            },
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
            name="what_would_change_my_mind",
            description="Name the cheapest experiments that would OVERTURN what a goal "
            "currently concludes, ranked by how little evidence holds each belief up. "
            "Answers the question a reviewer actually asks — not what do you believe, but "
            "what would it take to be wrong. A belief confirmed by elimination ranks first "
            "however confident the engine is: nothing ever measured it, which makes it both "
            "the weakest link and the cheapest thing in the graph to settle. Read-only — it "
            "issues no lease and changes nothing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Restrict to one objective. Omit for every goal.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "description": "How many beliefs to return, most fragile first.",
                    },
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
                                "EXHAUSTED",
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
            "was arrived at. Pass `since` to get a **diff** instead — what changed "
            "between then and now, which is the answer a standup or a PR description "
            "wants.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "description": "Cap on narrated transitions (most recent first). "
                        "Counters always cover the whole history.",
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Narrate one objective only. A workspace pursuing "
                        "several otherwise interleaves their dead ends into one story.",
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO-8601 instant. Report only what settled or was "
                        "withdrawn since then — 'what changed this week' rather than 'how "
                        "we got here'. Combine with as_of for a closed window.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "ISO-8601 instant. Reconstruct the report as it "
                        "stood then, so it can be read beside a rewound graph.",
                    },
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


async def _run_main(dashboard_port: int | None, cost_aware: bool = False) -> None:
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
    db_path = store_root(project_path) / "state.db"

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


async def _run_viewer(port: int) -> None:
    """Serve the dashboard against an existing belief state, with no MCP server.

    The try-before-you-wire path: someone who has never configured an MCP client
    can point this at a workspace and watch it. Read-only throughout, so it is
    safe to run against a database an agent is actively writing.
    """
    from hypotree.dashboard import DashboardServer, choose_port

    db_path = store_root(resolve_project_path()) / "state.db"
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
    exercisable in-process; the lock guarantees engine mutations never interleave
    across concurrent tool calls.
    """
    async with write_lock:
        result = _dispatch(engine, name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, default=str))]


def _parse_instant(raw: object, field: str) -> datetime | None:
    """Parse an ISO-8601 argument, naming the field when it will not parse.

    A `Z` suffix is what a copied JSON timestamp carries and `fromisoformat`
    rejects it before 3.11, so it is normalised rather than refused.
    """
    if raw is None or raw == "":
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"{field} must be an ISO-8601 instant such as '2026-08-07T09:00:00Z', got {raw!r}"
        ) from None


def _evidence_report(item: dict) -> EvidenceReport:
    """Build one evidence report from a tool-call payload.

    Shared by the single and batch shapes so the two can never drift — the
    `source_ref` field was advertised on the tool for a release while the
    dispatch that was supposed to read it silently dropped it.

    Both required fields are checked here rather than left to the schema. A
    missing ``success`` used to default to 0.5, which wrote a **real observation
    nobody made**: the posterior moved, and in the deterministic regime any
    reading settles a node, so `record_evidence(node_id="X")` — an ordinary LLM
    truncation — permanently EXHAUSTED a hypothesis and retired its competing
    answers. Asserting on no evidence is the one thing the belief state may not
    do, and a default value is the quietest way to do it.
    """
    node_id = item.get("node_id")
    if not node_id:
        raise ValueError(
            "each result needs node_id: the hypothesis this measurement is about. "
            "A result is {node_id, success, depth, claim_id?} — claim_id is optional."
        )

    ev: Evidence
    if item.get("evidence_kind", "logical") == "infra":
        ev = InfraError(
            error_type=item.get("error_type", "unknown"),
            message=item.get("message", ""),
        )
    else:
        if item.get("success") is None:
            raise ValueError(
                f"'{node_id}' needs success: the normalized result in [0,1]. There is no "
                f"default, because a made-up number is indistinguishable from a measurement "
                f"once it is recorded. If the experiment could not be run, send "
                f"evidence_kind='infra' instead — that never touches the belief."
            )
        ev = LogicalEvidence(
            success=item["success"],
            depth=item.get("depth", 0),
            metrics=item.get("metrics", {}),
            notes=item.get("notes", ""),
            source_ref=item.get("source_ref"),
            duration_s=item.get("duration_s"),
        )
    return EvidenceReport(
        node_id=str(node_id),
        evidence=ev,
        claim_id=item.get("claim_id"),
    )


def _dispatch(engine: HypoTreeEngine, name: str, arguments: dict) -> object:
    """Route a tool call to the corresponding engine method."""
    if name == "create_hypotheses":
        results = engine.create_hypotheses(arguments["hypotheses"])
        return [r.model_dump(mode="json") for r in results]

    if name == "add_edges":
        added = engine.add_edges(arguments["edges"])
        return [r.model_dump(mode="json") for r in added]

    if name == "get_next_targets":
        count = arguments.pop("count", 1)
        ttl = arguments.pop("lease_ttl_s", None)
        dry_run = arguments.pop("dry_run", False)
        targets = engine.get_next_targets(
            count=count,
            lease_ttl_s=ttl,
            dry_run=dry_run,
            goal_id=arguments.get("goal_id"),
        )
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
        resp = engine.get_goal_status(goal_id=arguments.get("goal_id"))
        return resp.model_dump(mode="json")

    if name == "get_conflicts":
        return {"conflicts": engine.get_conflicts(arguments.get("open_only", True))}

    if name == "suggest_discriminating_experiment":
        return engine.suggest_discriminating_experiment()

    if name == "what_would_change_my_mind":
        entries = engine.what_would_change_my_mind(
            goal_id=arguments.get("goal_id"),
            limit=int(arguments.get("limit", 5)),
        )
        return {
            "beliefs": [e.model_dump() for e in entries],
            # Said explicitly, because an empty list here is a finding rather
            # than a failure and reads as a bug without it.
            "note": (
                "nothing is holding this conclusion up on thin evidence"
                if not entries
                else "ordered by fragility: the first is the belief with least behind it"
            ),
        }

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
        return engine.generate_learning_path(
            limit=arguments.get("limit", 200),
            goal_id=arguments.get("goal_id"),
            since=_parse_instant(arguments.get("since"), "since"),
            as_of=_parse_instant(arguments.get("as_of"), "as_of"),
        ).model_dump(mode="json")

    if name == "get_workspace_info":
        from hypotree.store.identity import workspace_diagnostics

        info = workspace_diagnostics(engine.project_path)
        # Someone watching the agent work asks for this by name, and the agent
        # had no way to answer: the URL is minted at startup and printed to
        # stderr, which the model never sees. It carries the session token, so
        # it is the whole credential — reported here and nowhere it would be
        # logged as ordinary output.
        info["dashboard_url"] = dashboard_url()
        return info

    raise ValueError(f"Unknown tool: {name}")


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
        port, dashboard, mcp, cost_aware = _parse_serve_args(args)
    except ValueError as exc:
        print(f"hypotree: {exc}\n", file=sys.stderr)
        print(_cli_help(), file=sys.stderr)
        raise SystemExit(2) from None

    if not mcp:
        if not dashboard:
            print("hypotree: --no-mcp and --no-dashboard leave nothing to run\n", file=sys.stderr)
            raise SystemExit(2)
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_run_viewer(port))
        return
    asyncio.run(_run_main(dashboard_port=port if dashboard else None, cost_aware=cost_aware))


def _parse_serve_args(args: list[str]) -> tuple[int, bool, bool, bool]:
    """Resolve the serving flags into (first port to try, dashboard?, mcp?, cost-aware?).

    Hand-rolled rather than argparse because this entry point must stay
    import-cheap: it is spawned per MCP session, and the flags here do not
    justify the import.
    """
    from hypotree.dashboard.server import DEFAULT_PORT

    port = DEFAULT_PORT
    dashboard = True
    mcp = True
    cost_aware = False
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
        else:
            raise ValueError(f"unknown option {arg!r}")
    return port, dashboard, mcp, cost_aware


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
  hypotree             Run the MCP server on stdio (what an MCP client does),
                       with the web dashboard beside it on port {_default_port()},
                       probing upward if that one is taken.
  hypotree --dashboard-port PORT
                       Start the dashboard from PORT instead, still probing
                       upward. Use it when several workspaces are open at once
                       and you want a predictable address for this one.
  hypotree --no-dashboard
                       MCP server only, no socket opened.
  hypotree --no-mcp    Dashboard only, against the existing belief state, with
                       no MCP server. Read-only, so it is safe to point at a
                       workspace an agent is actively writing. This is the
                       try-before-you-wire path.
  hypotree --experimental-cost-aware
                       EXPERIMENTAL, off by default. Rank candidates by expected
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
  hypotree --info      Print the resolved workspace, store path and warnings.
  hypotree --version   Print the version.
  hypotree --help      Show this message.

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
