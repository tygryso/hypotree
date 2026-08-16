"""`HypoTreeToolset` — the object a Python host holds instead of an MCP client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypotree import HypoTreeEngine, HypoTreeToolset


def _seed(ht: HypoTreeToolset) -> None:
    """A goal with two competing answers under it."""
    ht.call(
        "create_hypotheses",
        {
            "hypotheses": [
                {"node_id": "goal", "statement": "ship it", "is_goal": True},
                {"node_id": "a", "statement": "cache is stale", "exclusion_group": "cause"},
                {"node_id": "b", "statement": "clock is skewed", "exclusion_group": "cause"},
            ]
        },
    )
    ht.call("add_edges", {"edges": [{"src": "a", "dst": "goal"}, {"src": "b", "dst": "goal"}]})


def _statuses(ht: HypoTreeToolset) -> dict[str, str]:
    """Node id to status, read off the table the agent itself is shown."""
    table = json.loads(ht.call("list_nodes", {}))["table"]
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1]:
            rows[cells[1]] = cells[3]
    return rows


@pytest.mark.integration
def test_a_host_can_run_the_whole_loop_in_process(tmp_path: Path) -> None:
    """Create, be handed a target, report, and see the goal move — no transport.

    This is the integration contract in one test: if a host can do this, it can
    embed hypotree. It runs against a temporary SQLite file in milliseconds,
    which is the point — validating the integration costs no inference.
    """
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path, rng_seed=1) as ht:
        _seed(ht)

        targets = json.loads(ht.call("get_next_targets", {"count": 1}))
        assert targets and targets[0]["node_id"] in {"a", "b"}
        chosen = targets[0]["node_id"]

        recorded = json.loads(
            ht.call(
                "record_evidence",
                {"node_id": chosen, "success": 1.0, "summary": "reproduced", "duration_s": 2.5},
            )
        )
        assert recorded["node"]["status"] == "VERIFIED"

        status = json.loads(ht.call("get_goal_status", {}))
        assert status["goals_total_count"] == 1


@pytest.mark.integration
def test_confirming_one_answer_retires_its_rival_through_the_toolset(tmp_path: Path) -> None:
    """The deduction is the product; it has to survive the new boundary.

    Confirming one member of an exclusion group settles the others without
    testing them. If that only worked over MCP, the embedded path would be a
    strictly worse product wearing the same name.
    """
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path, rng_seed=1) as ht:
        _seed(ht)
        ht.call("record_evidence", {"node_id": "a", "success": 1.0, "summary": "confirmed"})

        statuses = _statuses(ht)
        assert statuses["a"] == "VERIFIED"
        assert statuses["b"] == "EXHAUSTED", "the rival should be retired without a probe"


@pytest.mark.integration
def test_call_returns_errors_as_data_and_never_raises(tmp_path: Path) -> None:
    """A malformed call must not end the session.

    Every failure here is recoverable by the model that caused it — a misspelled
    node id, a missing `success`, an argument that is not an object. Raising
    would turn a correctable mistake into a lost run, and every agent eventually
    sends a malformed argument dict.
    """
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path) as ht:
        for name, args in [
            ("get_dag_context", {"node_id": "nope"}),
            ("record_evidence", {"node_id": "nope"}),
            ("create_hypotheses", {}),
            ("no_such_tool", {}),
        ]:
            payload = json.loads(ht.call(name, args))
            assert "error" in payload, f"{name} should report, not raise"
            assert payload["tool"] == name


@pytest.mark.integration
def test_arguments_that_are_not_an_object_are_refused_as_data(tmp_path: Path) -> None:
    """Models send a JSON string where an object belongs often enough to matter."""
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path) as ht:
        payload = json.loads(ht.call("list_nodes", "count=1"))  # type: ignore[arg-type]
        assert "must be a JSON object" in payload["error"]


@pytest.mark.integration
def test_a_narrowed_surface_is_a_boundary_not_a_suggestion(tmp_path: Path) -> None:
    """Hiding a tool must also refuse it.

    A host that trims the schema list but still executes anything the model asks
    for has not restricted anything: models routinely call tools they were never
    given, from memory of another session.
    """
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path, preset="essential") as ht:
        assert "render_dag_map" not in ht.tool_names
        payload = json.loads(ht.call("render_dag_map", {}))
        assert "not exposed by this toolset" in payload["error"]


@pytest.mark.integration
def test_a_read_only_toolset_exposes_and_executes_no_writes(tmp_path: Path) -> None:
    """What a reviewer or an untrusted sub-agent gets handed."""
    HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path).close()

    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path, read_only=True) as ht:
        assert ht.mutating_tool_names == frozenset()
        assert "list_nodes" in ht.tool_names
        payload = json.loads(ht.call("create_hypotheses", {"hypotheses": []}))
        assert "error" in payload


@pytest.mark.integration
def test_from_engine_wraps_without_taking_ownership(tmp_path: Path) -> None:
    """Two objects that both believe they own one connection close it twice.

    A host that already holds an engine — to read typed results rather than JSON
    — must be able to add the tool surface without handing over its lifecycle.
    """
    engine = HypoTreeEngine(tmp_path / "b.db", project_path=tmp_path)
    try:
        toolset = HypoTreeToolset.from_engine(engine, preset="essential")
        toolset.close()  # must be a no-op: the caller still owns the engine
        assert toolset.engine is engine
        # Still usable, because nothing was closed underneath it.
        assert json.loads(toolset.call("get_goal_status", {}))["goals_total_count"] == 0
    finally:
        engine.close()


@pytest.mark.integration
def test_the_belief_state_survives_the_host_process(tmp_path: Path) -> None:
    """Cross-session memory is the entire reason a host embeds this.

    A toolset that forgets on close is a scratchpad with extra steps.
    """
    db = tmp_path / "b.db"
    with HypoTreeToolset(db, project_path=tmp_path, rng_seed=1) as ht:
        _seed(ht)
        ht.call("record_evidence", {"node_id": "a", "success": 0.0, "summary": "ruled out"})
        before = _statuses(ht)

    with HypoTreeToolset(db, project_path=tmp_path, rng_seed=1) as ht:
        assert _statuses(ht) == before, "a new session must not re-open a settled question"
        assert before["a"] != "UNTESTED", "the refutation should have moved something"


@pytest.mark.integration
def test_mutating_tool_names_is_the_gate_a_host_keys_off(tmp_path: Path) -> None:
    """Hosts with a thinking gate need the write set from the package that owns it."""
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path) as ht:
        assert "record_evidence" in ht.mutating_tool_names
        assert "get_next_targets" in ht.mutating_tool_names, "issues leases, so it writes"
        assert "list_nodes" not in ht.mutating_tool_names
        assert ht.is_mutation("create_hypotheses") is True
        assert ht.is_mutation("get_conflicts") is False


@pytest.mark.integration
def test_tools_are_offered_in_the_shape_a_host_will_send(tmp_path: Path) -> None:
    """The schemas have to be usable verbatim, not after host-side reshaping."""
    with HypoTreeToolset(tmp_path / "b.db", project_path=tmp_path, preset="essential") as ht:
        tools = ht.tools()
        assert len(tools) == len(ht.tool_names)
        assert {t["function"]["name"] for t in tools} == set(ht.tool_names)
        assert json.loads(json.dumps(tools)) == tools, "must be JSON-serialisable as-is"
