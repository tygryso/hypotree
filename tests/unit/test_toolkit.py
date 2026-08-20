"""The transport-neutral tool surface: selection, projection and metadata."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hypotree.toolkit.specs import (
    TOOL_NAMES,
    TOOL_SPECS,
    ToolSpec,
    get_spec,
    hypothesis_item_schema,
    openai_tools,
    select_specs,
)


@pytest.mark.unit
def test_create_hypotheses_schema_exposes_optional_title() -> None:
    spec = next(spec for spec in TOOL_SPECS if spec.name == "create_hypotheses")
    title = spec.input_schema["properties"]["hypotheses"]["items"]["properties"]["title"]
    assert title["maxLength"] == 128


@pytest.mark.unit
def test_importing_hypotree_does_not_require_the_mcp_sdk() -> None:
    """Embedding the belief state must not drag in a JSON-RPC stack.

    The point of the toolkit is that a Python host can hold a belief state
    without acquiring an MCP client to talk to it. If `import hypotree` ever
    pulls in `mcp`, that promise is broken silently — the package still works
    here, where the SDK happens to be installed, and fails for the host that
    installed hypotree without it.

    Run in a subprocess because `sys.modules` in this one is already polluted by
    every other test that imported the server.
    """
    code = (
        "import sys, hypotree\n"
        "from hypotree import HypoTreeToolset, openai_tools\n"
        "assert not any(m == 'mcp' or m.startswith('mcp.') for m in sys.modules), "
        "sorted(m for m in sys.modules if m.startswith('mcp'))\n"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.unit
def test_the_public_api_is_importable_from_the_package_root() -> None:
    """A library nobody can import from its own name is not a library.

    Until the toolkit landed, `from hypotree import HypoTreeEngine` raised, and
    every consumer — including this project's own evaluation harness — reached
    through `hypotree.engine` into what was nominally an internal module.
    """
    import hypotree

    for name in ("HypoTreeEngine", "HypoTreeToolset", "Status", "Evidence", "openai_tools"):
        assert name in hypotree.__all__, f"{name} missing from __all__"
        assert hasattr(hypotree, name), f"{name} not importable from the package root"


@pytest.mark.unit
def test_every_spec_projects_into_openai_function_calling_form() -> None:
    """The shape a chat-completions host will actually send."""
    for tool in openai_tools():
        assert tool["type"] == "function"
        fn = tool["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["name"] in TOOL_NAMES
        assert fn["description"].strip(), f"{fn['name']} has an empty description"
        assert fn["parameters"]["type"] == "object"


@pytest.mark.unit
def test_tool_names_are_unique() -> None:
    """Two specs sharing a name means one of them is unreachable by dispatch."""
    names = [spec.name for spec in TOOL_SPECS]
    assert len(names) == len(set(names)), "duplicate tool name in TOOL_SPECS"
    assert len(TOOL_NAMES) == len(TOOL_SPECS)


@pytest.mark.unit
def test_the_essential_preset_is_a_strict_subset_that_can_still_run_the_loop() -> None:
    """A curated surface is what makes embedding affordable, not a nicety.

    Most clients re-send every tool schema on every turn, so a host that already
    carries its own forty tools cannot also carry twenty of ours. The subset has
    to be small *and* sufficient: state hypotheses, be handed one, report what
    happened, know where you stand, and re-orient after a context reset.
    """
    essential = {spec.name for spec in select_specs(preset="essential")}
    assert essential < TOOL_NAMES, "essential must be a strict subset of the full surface"
    # Without these five the loop cannot close: nothing to test, no way to be
    # handed one, no way to report, no way to know it is done, no way to recover
    # context after a reset.
    assert {
        "create_hypotheses",
        "get_next_targets",
        "record_evidence",
        "get_goal_status",
        "generate_learning_path",
    } <= essential


@pytest.mark.unit
def test_read_only_selection_exposes_no_tool_that_can_write() -> None:
    """What a reviewer or an untrusted sub-agent should be handed.

    A read-only store cannot serve a mutating tool, so advertising one is an
    invitation to an error the host cannot recover from — and a read-only
    *boundary* that still lists writes is not a boundary.
    """
    specs = select_specs(read_only=True)
    assert specs, "read-only selection should still expose the sensors"
    assert not [spec.name for spec in specs if spec.mutates]


@pytest.mark.unit
def test_get_next_targets_is_classified_as_a_mutation() -> None:
    """The classification everyone gets wrong, which is why it is not inferred.

    It reads like a query and it writes: it issues leases and expires stale
    ones. A host that gates writes by guessing from the tool's name lets this
    one straight through.
    """
    assert get_spec("get_next_targets").mutates is True
    assert get_spec("get_goal_status").mutates is False


@pytest.mark.unit
def test_selection_order_is_stable_across_calls() -> None:
    """A tool list that reshuffles defeats provider-side prompt caching."""
    assert [s.name for s in select_specs()] == [s.name for s in select_specs()]
    assert [s.name for s in select_specs()] == [s.name for s in TOOL_SPECS]


@pytest.mark.unit
def test_include_overrides_the_preset_and_exclude_narrows_it() -> None:
    """Start from a preset and remove, rather than enumerating from scratch."""
    only = select_specs(include=["get_goal_status", "list_nodes"])
    assert [s.name for s in only] == ["get_goal_status", "list_nodes"]

    narrowed = {s.name for s in select_specs(preset="essential", exclude=["add_edges"])}
    assert "add_edges" not in narrowed
    assert "create_hypotheses" in narrowed


@pytest.mark.unit
def test_an_unknown_tool_is_refused_with_the_list_of_real_ones() -> None:
    """The error has to be actionable: a typo is the common case."""
    with pytest.raises(KeyError, match="get_goal_status"):
        get_spec("get_goal_statuses")
    with pytest.raises(KeyError, match="unknown tool"):
        select_specs(include=["not_a_tool"])
    with pytest.raises(ValueError, match="preset must be"):
        select_specs(preset="minimal")


@pytest.mark.unit
def test_the_hypothesis_item_schema_keeps_the_soundness_fields() -> None:
    """`exclusion_closed` is what stops the engine deducing over a partial list.

    Dropping it from a schema does not raise anything — it just removes the
    agent's only way to say "these are not all the candidates", after which the
    last-one-standing deduction is made unconditionally and can assert something
    false. It has gone missing from a hand-written copy of this schema once.
    """
    props = hypothesis_item_schema()["properties"]
    assert props["exclusion_closed"]["default"] is True
    for field in ("statement", "exclusion_group", "is_goal", "estimated_cost"):
        assert field in props
    assert hypothesis_item_schema()["required"] == ["statement"]


@pytest.mark.unit
def test_a_spec_is_immutable() -> None:
    """Specs are shared, module-level and handed to every caller."""
    spec = get_spec("list_nodes")
    with pytest.raises(AttributeError):
        spec.name = "something_else"  # type: ignore[misc]
    assert isinstance(spec, ToolSpec)
