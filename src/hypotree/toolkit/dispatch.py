"""Route a tool call to the engine, with no transport in the way.

Lifted out of the MCP server so that embedding hypotree in a Python host does
not require an MCP client. The routing table was always transport-neutral — it
takes a tool name and a dict of arguments and returns plain data — but it lived
behind a module whose first three imports are the MCP SDK, so a host that wanted
the belief state had to acquire a JSON-RPC stack to reach it, or reimplement the
routing and let it drift.

Everything here is deliberately synchronous and free of I/O beyond the engine
itself. Serialisation, locking and the transport envelope belong to the caller:
the MCP server wraps this in an ``asyncio.Lock`` and JSON, and an in-process
host wraps it in whatever it already has.
"""

from __future__ import annotations

from datetime import datetime

from hypotree.engine import EvidenceReport, HypoTreeEngine
from hypotree.models.evidence import Evidence, InfraError, LogicalEvidence
from hypotree.models.status import Status

# Where a human can watch this belief state move, or None when nothing is
# serving it. Owned here rather than by the MCP server because `dispatch` is
# what answers `get_workspace_info`, and a routing table that has to import its
# own server to answer a question is a cycle waiting to be discovered.
_DASHBOARD_URL: str | None = None


def publish_dashboard_url(url: str | None) -> None:
    """Record where the dashboard is listening, for `get_workspace_info`."""
    global _DASHBOARD_URL
    _DASHBOARD_URL = url


def dashboard_url() -> str | None:
    """The running dashboard's URL with its session token, or None."""
    return _DASHBOARD_URL


def parse_instant(raw: object, field: str) -> datetime | None:
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


def evidence_report(item: dict) -> EvidenceReport:
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
            attestation_id=item.get("attestation_id"),
        )
    return EvidenceReport(
        node_id=str(node_id),
        evidence=ev,
        claim_id=item.get("claim_id"),
    )


def dispatch(engine: HypoTreeEngine, name: str, arguments: dict) -> object:
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
            reports = [evidence_report(item) for item in raw_results]
            batch = engine.record_results(reports, count_next_targets=count_next, lease_ttl_s=ttl)
            return batch.model_dump(mode="json")
        report = evidence_report(arguments)
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
        updates = engine.bulk_update_status(
            arguments["node_ids"],
            Status(arguments["new_status"]),
            arguments.get("reason", ""),
        )
        return [r.model_dump(mode="json") for r in updates]

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
        context = engine.get_dag_context(
            arguments.get("node_id"),
            arguments.get("max_depth", 2),
            arguments.get("max_children", 10),
        )
        return context.model_dump(mode="json")

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
            since=parse_instant(arguments.get("since"), "since"),
            as_of=parse_instant(arguments.get("as_of"), "as_of"),
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
