"""Transport-neutral description of every tool hypotree exposes.

The single source of truth for the tool surface, deliberately **free of any
transport**: this module imports neither ``mcp`` nor anything that talks to a
socket, so a Python host can embed the belief state without acquiring an MCP
client to do it.

That independence is the whole point. The schemas used to live inside the MCP
server, which made them unreachable for any caller that was not speaking
JSON-RPC — so the evaluation harness hand-wrote its own copies of seven of them,
and those copies drifted: by v0.6.0 the harness's ``create_hypotheses`` was
missing ``exclusion_closed``, the one field that stops the engine deducing a
false answer over an incomplete list of candidates. Two hand-maintained
descriptions of one contract diverge; the only durable fix is for there to be
one description and for every transport to be a projection of it.

Each spec also carries what a *host* needs and no schema can express:

- ``mutates`` — whether calling it changes the belief state. An agent with its
  own gating policy (a thinking gate, a review gate, an approval prompt) has to
  know which calls are consequential, and inferring it from the tool's name is
  how a host eventually lets a write through by accident.
- ``essential`` — whether it belongs to the smallest surface that can still run
  the loop. Twenty tool schemas are re-sent on every turn by most clients, and a
  host that already carries forty of its own cannot afford all of them; a
  curated subset is what makes embedding viable at all rather than a nicety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Tools that change the belief state. `get_next_targets` is in the list and is
# the one people get wrong: it reads like a query, and it issues leases and
# expires stale ones, so it writes.
_MUTATING = frozenset(
    {
        "create_hypotheses",
        "add_edges",
        "get_next_targets",
        "record_evidence",
        "renew_claim",
        "release_claims",
        "update_status",
        "invalidate_upstream",
        "verify_upstream",
    }
)

# The smallest surface that can still run the loop: state the hypotheses, be
# handed one, report what happened, know where you stand, and re-orient after a
# context reset. Everything outside it is diagnosis, presentation or repair —
# valuable to a human reading the graph, and not on the path an agent must walk
# to make progress.
#
# `generate_learning_path` earns its place for a reason worth stating: across
# three full evaluation runs the agent called it after a context reset exactly
# zero times, and every redundant probe in those runs followed a reset. A host
# that ships only the loop should ship the tool that rebuilds the loop's memory.
_ESSENTIAL = frozenset(
    {
        "create_hypotheses",
        "add_edges",
        "get_next_targets",
        "record_evidence",
        "get_goal_status",
        "generate_learning_path",
    }
)


@dataclass(frozen=True)
class ToolSpec:
    """One tool, described once and projected into whatever transport wants it."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def mutates(self) -> bool:
        """Whether calling this changes the belief state.

        Hosts with their own gating policy key off this rather than off the
        tool's name, so a tool added later is classified by the package that
        owns it instead of by a pattern match in the host that embeds it.
        """
        return self.name in _MUTATING

    @property
    def essential(self) -> bool:
        """Whether this belongs to the minimal loop (see ``_ESSENTIAL``)."""
        return self.name in _ESSENTIAL

    def as_openai_tool(self) -> dict[str, Any]:
        """This spec in OpenAI function-calling form.

        The shape every chat-completions host wants, including the
        OpenAI-compatible endpoints (Ollama, vLLM, GLM, z.ai) that agents
        outside the MCP ecosystem are actually built on.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def hypothesis_item_schema() -> dict[str, Any]:
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


def _build_specs() -> tuple[ToolSpec, ...]:
    """Every tool hypotree exposes, in a transport-neutral form."""
    return (
        ToolSpec(
            name="create_hypotheses",
            description="Add one or many hypothesis nodes (with optional parent edges). "
            "Pass a list of one to create a single hypothesis. Parents may be created by "
            "the same call in any order. The whole batch is validated before anything is "
            "written, so a rejected call creates nothing.",
            input_schema={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": 1,
                        "items": hypothesis_item_schema(),
                    }
                },
                "required": ["hypotheses"],
            },
        ),
        ToolSpec(
            name="add_edges",
            description="Wire hypotheses that already exist, without recreating either. "
            "Use it to grow a graph forward: when a pipeline gains a stage, the goal must "
            "depend on the NEW last stage, or it reports itself achieved as soon as the "
            "first stage verifies while the rest sit untested. You do not need to remove "
            "the old edge — DEPENDENCY is AND and the later stage already depends on the "
            "earlier one, so adding only tightens the condition. Validated like creation: "
            "unknown nodes, a goal used as a DEPENDENCY parent, and cycles are refused "
            "before anything is written, and an edge that already exists is a no-op.",
            input_schema={
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
        ToolSpec(
            name="get_next_targets",
            description="Reclaim stale leases and select the next target(s). A claimed "
            "node is reserved for you until you record its result, so ask only for what "
            "you will probe before your next call — anything you hold and do not report "
            "is work nobody can do. A batch never contains two competing answers to the "
            "same question. Returns a list; each entry may carry min_depth when the node "
            "is under conflict review.",
            input_schema={
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
        ToolSpec(
            name="record_evidence",
            description="Record one result, or many in one call, and update the belief "
            "state. Auto-captures git context_hash + git_branch when unset. Record "
            "against the hypothesis whose statement you actually tested: evidence "
            "against a goal is refused, and evidence against a premise a composition "
            "assumed corrupts a confirmation that is still true on its own. Ran several "
            "experiments this turn? Report them together with `results` — one call, "
            "applied in order.",
            input_schema={
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
                                "attestation_id": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            # Logical reports still require success in dispatch. Keeping
                            # it out of JSON Schema lets evidence_kind="infra" truthfully
                            # report that no logical measurement was made.
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
                    "attestation_id": {
                        "type": "string",
                        "description": "Runner-minted attestation id. Provenance fields "
                        "cannot be supplied here; unknown ids degrade to self-reported.",
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
        ToolSpec(
            name="renew_claim",
            description="Restart a live lease's clock because the experiment is still "
            "running. Use this instead of a very long TTL: the lease exists so work held "
            "by a caller that vanished comes back, and a long TTL makes that recovery as "
            "slow as the longest experiment.",
            input_schema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "lease_ttl_s": {"type": "integer", "minimum": 1},
                },
                "required": ["claim_id"],
            },
        ),
        ToolSpec(
            name="release_claims",
            description="Hand leased nodes back without recording a result \u2014 for work you "
            "have decided not to run, or for resuming after a context reset you cannot "
            "report on. Omit claim_ids to release everything you hold.",
            input_schema={
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
        ToolSpec(
            name="update_status",
            description="Manually override the status of one or many nodes. Every id is "
            "validated before anything changes, so a bad id never leaves a partial update.",
            input_schema={
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
        ToolSpec(
            name="invalidate_upstream",
            description="Walk DEPENDENCY ancestors, flip VERIFIED → NEEDS_REVISION.",
            input_schema={
                "type": "object",
                "properties": {"leaf_id": {"type": "string"}},
                "required": ["leaf_id"],
            },
        ),
        ToolSpec(
            name="verify_upstream",
            description="Walk REFINEMENT ancestors, flip IN_PROGRESS → VERIFIED (depth-capped).",
            input_schema={
                "type": "object",
                "properties": {"child_id": {"type": "string"}},
                "required": ["child_id"],
            },
        ),
        ToolSpec(
            name="get_goal_status",
            description="Report all goal nodes, target metrics, progress counts, "
            "and whether the global stop holds.",
            input_schema={
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
        ToolSpec(
            name="get_conflicts",
            description="List recorded conflicts — sets of assumptions that cannot all "
            "hold together, with which members are exonerated and which remain suspects.",
            input_schema={
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
        ToolSpec(
            name="what_would_change_my_mind",
            description="Name the cheapest experiments that would OVERTURN what a goal "
            "currently concludes, ranked by how little evidence holds each belief up. "
            "Answers the question a reviewer actually asks — not what do you believe, but "
            "what would it take to be wrong. A belief confirmed by elimination ranks first "
            "however confident the engine is: nothing ever measured it, which makes it both "
            "the weakest link and the cheapest thing in the graph to settle. Read-only — it "
            "issues no lease and changes nothing.",
            input_schema={
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
        ToolSpec(
            name="suggest_discriminating_experiment",
            description="Propose the single most informative next experiment: re-test a "
            "conflict suspect at depth while any remains, otherwise the closest "
            "alternative combination that no recorded conflict rules out.",
            input_schema={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="get_dag_context",
            description="Return a depth+width-bounded subgraph with credible intervals.",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 2},
                    "max_children": {"type": "integer", "default": 10},
                },
            },
        ),
        ToolSpec(
            name="render_dag_map",
            description="Render a Mermaid flowchart with depth+width bounding + elision.",
            input_schema={
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
        ToolSpec(
            name="list_nodes",
            description="Query/filter/sort nodes and return a Markdown table. "
            "Use `view` for the questions actually worth asking — 'frontier' (what "
            "is still open), 'settled', 'verified', 'revision' (what is under "
            "revision), 'stale' — rather than assembling a status filter by hand. "
            "`stale_only=true` keeps only confirmations made against a commit that "
            "is no longer checked out: they are not refuted, but nothing has "
            "re-established them since the code moved.",
            input_schema={
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
        ToolSpec(
            name="get_evidence_history",
            description="Return the evidence trail for a node (newest-first).",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["node_id"],
            },
        ),
        ToolSpec(
            name="get_active_claims",
            description="Return live (unconsumed, unexpired) claims for resuming work.",
            input_schema={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="generate_learning_path",
            description="Narrate what has been settled so far, in order, and how — "
            "separating what an experiment paid for from what the engine inferred for "
            "free, and calling out beliefs that were later withdrawn. Use it to brief "
            "a human, to write a summary, or to re-orient yourself after a context "
            "reset: the other read tools show the current state, this one shows how it "
            "was arrived at. Pass `since` to get a **diff** instead — what changed "
            "between then and now, which is the answer a standup or a PR description "
            "wants.",
            input_schema={
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
        ToolSpec(
            name="get_workspace_info",
            description="Which belief state you are connected to, and how it was chosen. "
            "Reports the workspace id, which of the four resolution layers produced it, "
            "where the database lives and whether it exists yet. Call it when the graph "
            "is unexpectedly empty or two clients disagree about what has been "
            "established — that is almost always one project resolving to two workspaces.",
            input_schema={"type": "object", "properties": {}},
        ),
    )


# Built once at import: the specs are static data and every host asks for them
# on a hot path (most clients re-send the whole tool list on every single turn).
TOOL_SPECS: tuple[ToolSpec, ...] = _build_specs()

TOOL_NAMES: frozenset[str] = frozenset(spec.name for spec in TOOL_SPECS)

_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def get_spec(name: str) -> ToolSpec:
    """The spec for one tool, or a KeyError naming what does exist."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; hypotree exposes {sorted(TOOL_NAMES)}") from None


def select_specs(
    *,
    preset: str = "full",
    include: list[str] | tuple[str, ...] | None = None,
    exclude: list[str] | tuple[str, ...] | None = None,
    read_only: bool = False,
) -> tuple[ToolSpec, ...]:
    """Choose which tools to expose, in a stable order.

    ``preset`` is ``"full"`` (all twenty) or ``"essential"`` (the six that run
    the loop). ``include`` overrides the preset outright; ``exclude`` and
    ``read_only`` then narrow whatever was chosen, so a host can start from a
    preset and remove rather than having to enumerate.

    ``read_only=True`` drops every mutating tool, which is what a reviewer, a
    dashboard or an untrusted sub-agent should be handed: it can read the whole
    belief state and cannot write a word of it.

    Order is deterministic — the declaration order of ``TOOL_SPECS`` — because a
    tool list that reshuffles between calls defeats provider-side prompt caching
    for no benefit.
    """
    if include is not None:
        unknown = sorted(set(include) - TOOL_NAMES)
        if unknown:
            raise KeyError(f"unknown tool(s) {unknown}; hypotree exposes {sorted(TOOL_NAMES)}")
        chosen = set(include)
    elif preset == "full":
        chosen = set(TOOL_NAMES)
    elif preset == "essential":
        chosen = {spec.name for spec in TOOL_SPECS if spec.essential}
    else:
        raise ValueError(f"preset must be 'full' or 'essential', got {preset!r}")

    if exclude:
        chosen -= set(exclude)
    if read_only:
        chosen -= {spec.name for spec in TOOL_SPECS if spec.mutates}
    return tuple(spec for spec in TOOL_SPECS if spec.name in chosen)


def openai_tools(
    *,
    preset: str = "full",
    include: list[str] | tuple[str, ...] | None = None,
    exclude: list[str] | tuple[str, ...] | None = None,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """The selected tools in OpenAI function-calling form. See ``select_specs``."""
    return [
        spec.as_openai_tool()
        for spec in select_specs(
            preset=preset, include=include, exclude=exclude, read_only=read_only
        )
    ]
