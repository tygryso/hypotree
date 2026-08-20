"""The evaluation harness's tool schemas, pinned against the canonical ones.

The harness hands the arm-B agent its own hand-written OpenAI schemas rather
than hypotree's. That was not a design decision so much as an accident of
layout: the real schemas lived inside the MCP server, behind three MCP imports,
so a harness that drives the engine in-process had no way to reach them and
wrote its own.

Two copies of one contract drift, and these did. Both directions matter, and
they fail very differently:

- **Advertising a field dispatch ignores** is the dangerous one. The agent is
  told a parameter exists, supplies it, gets no error, and the value is silently
  dropped — the belief state then differs from what the agent believes it wrote,
  and every number computed from it is measuring something nobody described.
  This has shipped once already, with ``source_ref``. It is a hard failure here.
- **Omitting a field** narrows what the agent can express. That is legitimate —
  a harness may deliberately restrict its agent — but only when it is deliberate.
  So the omissions are enumerated below with the reason each is acceptable, and
  a new one has to be added to the list consciously.

The waiver list is the point. Without it the harness's restrictions are
indistinguishable from its oversights.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.runner import runner
from hypotree.toolkit.specs import TOOL_NAMES, get_spec

# Harness factory -> the canonical tool it stands in for. `evaluate_config` and
# `update_scratchpad` are the environment's own tools and have no counterpart.
_MIRRORED = {
    "_tool_create_hypotheses": "create_hypotheses",
    "_tool_get_next_targets": "get_next_targets",
    "_tool_record_evidence": "record_evidence",
    "_tool_get_goal_status": "get_goal_status",
    "_tool_list_nodes": "list_nodes",
    "_tool_get_conflicts": "get_conflicts",
    "_tool_suggest_experiment": "suggest_discriminating_experiment",
}

# Canonical fields the harness deliberately does not offer its agent, and why.
# Each entry is a restriction on what arm B can express, so each is also a
# statement about what the run can and cannot measure.
_WAIVED: dict[str, set[str]] = {
    # The eval task fixes a closed candidate set by construction, and the
    # environment measures cost itself rather than trusting a self-report, so
    # the title, parametric and cost fields would be noise the agent has to fill in.
    # `exclusion_closed` is the consequential one: with it withheld, arm B can
    # never declare an open candidate list, so the last-one-standing deduction
    # is always licensed and the closed-world guard is never exercised.
    "create_hypotheses": {
        "hypotheses[].estimated_cost",
        "hypotheses[].evidence_regime",
        "hypotheses[].exclusion_closed",
        "hypotheses[].is_parametric",
        "hypotheses[].param_config",
        "hypotheses[].title",
    },
    # Leases and multi-goal routing are not exercised: the harness is
    # single-agent and single-goal, so both would be dead parameters.
    "get_next_targets": {"goal_id", "lease_ttl_s"},
    # The harness times probes itself and writes the duration in, rather than
    # asking the agent how long its own work took. That is the right call for a
    # measurement, and it means arm B's belief state carries no agent-supplied
    # cost signal — cost-aware ordering is therefore not under test here.
    "record_evidence": {
        "attestation_id",
        "duration_s",
        "error_type",
        "evidence_kind",
        "lease_ttl_s",
        "message",
        "metrics",
        "source_ref",
        "results[].duration_s",
        "results[].attestation_id",
        "results[].error_type",
        "results[].evidence_kind",
        "results[].message",
        "results[].metrics",
        "results[].source_ref",
    },
    "get_goal_status": {"goal_id"},
    # Presentation and paging controls; the harness reads the default view.
    "list_nodes": {"ascending", "offset", "stale_only", "view"},
    "get_conflicts": {"open_only"},
}


def _flatten(schema: dict[str, Any]) -> dict[str, str | None]:
    """Every property in a JSON schema, dotted, with its declared type.

    Nested objects and array items are walked because the field that went
    missing was one level down, inside `hypotheses[]`, where a top-level
    comparison would never have looked.
    """
    found: dict[str, str | None] = {}

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in (node.get("properties") or {}).items():
            if not isinstance(value, dict):
                continue
            found[f"{prefix}{key}"] = value.get("type")
            if value.get("type") == "object":
                walk(value, f"{prefix}{key}.")
            elif value.get("type") == "array":
                walk(value.get("items"), f"{prefix}{key}[].")

    walk(schema, "")
    return found


@pytest.mark.unit
@pytest.mark.parametrize(("factory", "canonical"), sorted(_MIRRORED.items()))
def test_the_harness_advertises_no_field_the_engine_would_ignore(
    factory: str, canonical: str
) -> None:
    """The dangerous direction: a parameter the agent supplies and dispatch drops.

    There is no error for this at runtime. The agent believes it recorded
    something the belief state never received, and the run's numbers describe a
    belief state nobody wrote.
    """
    harness = _flatten(getattr(runner, factory)()["function"]["parameters"])
    engine = _flatten(get_spec(canonical).input_schema)

    invented = sorted(set(harness) - set(engine))
    assert not invented, (
        f"{factory} advertises {invented} which {canonical} does not accept; "
        "the agent would supply them and dispatch would silently drop them"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("factory", "canonical"), sorted(_MIRRORED.items()))
def test_shared_fields_agree_on_type(factory: str, canonical: str) -> None:
    """A field that exists in both but disagrees on type is a coercion bug waiting."""
    harness = _flatten(getattr(runner, factory)()["function"]["parameters"])
    engine = _flatten(get_spec(canonical).input_schema)

    mismatched = {
        field: (harness[field], engine[field])
        for field in set(harness) & set(engine)
        if harness[field] != engine[field]
    }
    assert not mismatched, f"{factory} disagrees with {canonical} on {mismatched}"


@pytest.mark.unit
@pytest.mark.parametrize(("factory", "canonical"), sorted(_MIRRORED.items()))
def test_every_omission_is_a_recorded_decision(factory: str, canonical: str) -> None:
    """A restriction the harness never chose is an oversight wearing its clothes.

    When a field is added to a canonical schema, this fails until someone has
    either offered it to the agent or written down why they did not — which is
    the review step that did not happen when `exclusion_closed` was added.
    """
    harness = _flatten(getattr(runner, factory)()["function"]["parameters"])
    engine = _flatten(get_spec(canonical).input_schema)

    omitted = set(engine) - set(harness)
    unreviewed = sorted(omitted - _WAIVED.get(canonical, set()))
    assert not unreviewed, (
        f"{canonical} offers {unreviewed} and {factory} does not; either expose them "
        "to the agent or add them to the waiver list with a reason"
    )

    stale = sorted(_WAIVED.get(canonical, set()) - omitted)
    assert not stale, (
        f"the waiver for {canonical} still lists {stale}, which the harness now "
        "offers; drop them from the waiver so it keeps describing reality"
    )


@pytest.mark.unit
def test_every_mirrored_tool_still_exists_under_that_name() -> None:
    """A renamed tool would leave the harness driving a name nothing answers to."""
    for factory, canonical in _MIRRORED.items():
        assert canonical in TOOL_NAMES, f"{factory} mirrors {canonical}, which no longer exists"
        assert getattr(runner, factory)()["function"]["name"] == canonical


@pytest.mark.unit
def test_the_harness_only_adds_tools_the_environment_owns() -> None:
    """Anything the harness offers beyond its mirrors must be an environment tool.

    A hypotree tool served by a hand-written schema outside `_MIRRORED` would sit
    outside every check in this file.
    """
    environment_only = {"evaluate_config", "update_scratchpad"}
    for arm in ("A", "B"):
        offered = {t["function"]["name"] for t in runner.get_tools_for_arm(arm)}
        unaccounted = offered - set(_MIRRORED.values()) - environment_only
        assert not unaccounted, f"arm {arm} offers unaccounted tools: {sorted(unaccounted)}"
