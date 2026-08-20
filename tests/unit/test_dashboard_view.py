"""Tests for the fields the interface reads off the belief state.

Each of these was a defect the browser showed and the API contract did not: a
timestamp that was really a reason string, a goal filter that looked broken while
being arithmetically right, and a binary asset the manifest silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from hypotree.dashboard.readmodel import ReadModel, _layer_positions
from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import Edge, EdgeType
from hypotree.models.evidence import LogicalEvidence, RunAttestation
from hypotree.store.store import utcnow


@pytest.mark.unit
def test_dashboard_payloads_expose_title(tmp_path: Path) -> None:
    db = tmp_path / "title.db"
    engine = HypoTreeEngine(db)
    engine.create_hypothesis("Long claim", node_id="n1", title="Short label")
    engine.close()
    read = ReadModel(db)
    try:
        graph_node = next(node for node in read.graph().nodes if node["id"] == "n1")
        detail = read.node_detail("n1")
        assert graph_node["title"] == "Short label"
        assert detail is not None
        assert detail["node"]["title"] == "Short label"
    finally:
        read.close()


@pytest.mark.unit
def test_node_detail_exposes_paginated_provenance(tmp_path: Path) -> None:
    db = tmp_path / "evidence.db"
    engine = HypoTreeEngine(db)
    engine.create_hypothesis("claim", node_id="n1")
    attestation = RunAttestation(
        id="run-1",
        runner="pytest",
        argv=["pytest"],
        exit_code=0,
        duration_s=0.25,
        created_at=utcnow(),
    )
    engine.add_attestation(attestation)
    engine.record_evidence(
        "n1",
        LogicalEvidence(
            success=1.0,
            metrics={"passed": 4.0},
            delta_success=0.5,
            monotonicity="up",
            notes="new result",
            attestation_id="run-1",
        ),
    )
    engine.close()
    read = ReadModel(db)
    try:
        detail = read.node_detail("n1", evidence_limit=1, evidence_query="new")
        assert detail is not None
        assert detail["revision"] >= 1
        assert detail["evidence_total"] == 1
        row = detail["evidence"][0]
        assert row["metrics"] == {"passed": 4.0}
        assert row["delta_success"] == 0.0
        assert row["monotonicity"] == "first"
        assert row["verified_by"] == "attested"
        assert row["attestation"]["runner"] == "pytest"
    finally:
        read.close()


@pytest.fixture
def wired(tmp_path: Path) -> Iterator[Path]:
    """One wired goal and one that depends on nothing."""
    db = tmp_path / "view.db"
    engine = HypoTreeEngine(db, rng_seed=5)
    try:
        for i in range(3):
            engine.create_hypothesis(f"a={i}", node_id=f"a{i}", exclusion_group="a")
        engine.create_hypothesis(
            "combo", node_id="combo", parent_ids=["a0"], edge_type=EdgeType.DEPENDENCY
        )
        engine.create_hypothesis(
            "goal", node_id="goal", is_goal=True, target_metric=0.7, parent_ids=["combo"]
        )
        engine.create_hypothesis("orphan", node_id="orphan_goal", is_goal=True, target_metric=0.9)
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
    finally:
        engine.close()
    yield db


@pytest.mark.unit
def test_a_goal_that_depends_on_nothing_is_named_rather_than_shown_as_a_dot(
    wired: Path,
) -> None:
    """The scope is right and the picture is unreadable — so the picture explains itself.

    Filtering to an unwired goal collapses to one node, which is exactly what its
    case contains. Rendered without a word it reads as a broken filter, and the
    engine already knows the diagnosis.
    """
    read = ReadModel(wired)
    try:
        assert read.graph(goal_id="goal").unwired_goal is None
        assert len(read.graph(goal_id="goal").nodes) > 1

        assert [n["id"] for n in read.graph(goal_id="orphan_goal").nodes] == ["orphan_goal"]
        assert read.graph().unwired_goal is None
    finally:
        read.close()


@pytest.mark.unit
def test_a_backwards_wired_goal_still_draws_its_whole_tree(tmp_path: Path) -> None:
    """Agents wired `goal -> phase -> work`, reading the goal as a container.

    The engine refuses that edge now, but databases written before it does still
    hold the shape, so the edges are inserted below the API here — that is
    exactly the state the viewer has to cope with. The nodes are still the user's
    work: scoping the *view* to the selection scope showed one dot beside a
    lecture about wiring while a whole tree hung off the node in question, and a
    viewer that hides your work to make a point about how you built it is not
    making the point.
    """
    db = tmp_path / "backwards.db"
    engine = HypoTreeEngine(db, rng_seed=5)
    try:
        engine.create_hypothesis("ship it", node_id="goal_ship", is_goal=True, target_metric=0.8)
        for nid in ("phase0", "phase1", "work"):
            engine.create_hypothesis(nid, node_id=nid)
        engine.create_hypothesis("unrelated", node_id="unrelated")
        for src, dst in (("goal_ship", "phase0"), ("phase0", "phase1"), ("phase1", "work")):
            engine._store.add_edge(Edge(src=src, dst=dst, type=EdgeType.DEPENDENCY))
    finally:
        engine.close()

    read = ReadModel(db)
    try:
        snap = read.graph(goal_id="goal_ship")
        assert sorted(n["id"] for n in snap.nodes) == ["goal_ship", "phase0", "phase1", "work"]
        assert len(snap.edges) == 3
        # Still unreachable, and still reported — beside the drawing, not instead.
        assert snap.unwired_goal == "goal_ship"
        # Named as the inversion it is: the fix differs from a goal with nothing
        # attached at all.
        assert snap.goal_wiring == "inverted"
        # The navigator's own scope is untouched: it must not select downstream
        # of a goal just because the viewer draws it.
        assert read._engine._goal_scope("goal_ship") == {"goal_ship"}
        # The scrubber moves the picture, so its ticks have to cover the picture.
        drawn = {n["id"] for n in snap.nodes}
        ticks = read.timeline(goal_id="goal_ship")["ticks"]
        assert {t["node_id"] for t in ticks} == drawn
    finally:
        read.close()


@pytest.mark.unit
def test_a_goal_with_nothing_attached_is_named_apart_from_an_inverted_one(
    wired: Path,
) -> None:
    """Two different mistakes with two different fixes.

    "Add the work as a parent" and "turn your edges around" are not the same
    instruction, and one flag left the reader to work out which they were
    looking at.
    """
    read = ReadModel(wired)
    try:
        orphan = read.graph(goal_id="orphan_goal")
        assert orphan.unwired_goal == "orphan_goal"
        assert orphan.goal_wiring == "unwired"
        assert read.graph(goal_id="goal").goal_wiring is None
    finally:
        read.close()


@pytest.mark.unit
def test_every_node_carries_when_it_was_made_and_when_it_settled(wired: Path) -> None:
    """A card shows both, because a status column cannot tell "confirmed" from
    "confirmed last week and untouched since"."""
    read = ReadModel(wired)
    try:
        node = next(n for n in read.graph().nodes if n["id"] == "a0")
        assert node["created_at"] and node["settled_at"]
        assert node["created_at"] <= node["settled_at"]

        detail = read.node_detail("a0")
        assert detail is not None
        assert detail["node"]["created_at"] == node["created_at"]
        assert detail["node"]["settled_at"] == node["settled_at"]
        assert detail["node"]["reason"]
    finally:
        read.close()


@pytest.mark.unit
def test_the_settled_stamp_is_a_timestamp_when_the_view_is_rewound(wired: Path) -> None:
    """Rewound, it is the start of the interval covering that instant.

    The first version read field 1 of a `(status, reason)` pair and put the
    engine's reason string in a date column, which the API contract could not see
    and the browser showed immediately.
    """
    read = ReadModel(wired)
    try:
        ticks = read.timeline()["ticks"]
        assert ticks
        past = ticks[len(ticks) // 2]["t"]
        for node in read.graph(at=past).nodes:
            stamp = node["settled_at"]
            assert stamp is None or stamp[:4].isdigit(), stamp
    finally:
        read.close()


@pytest.mark.unit
def test_a_wide_layer_wraps_instead_of_drawing_one_long_line() -> None:
    """Twenty-five competing answers on one line fit the viewport at a scale
    where nothing is legible."""
    ids = [f"n{i}" for i in range(25)]
    pos = _layer_positions(ids, [])

    rows = {y for _, y in pos.values()}
    assert len(rows) > 1, "a 25-wide layer must wrap"
    span = max(x for x, _ in pos.values()) - min(x for x, _ in pos.values())
    assert span < 24, "wrapping must actually narrow the drawing"
    # Still a pure function of the topology: same input, same picture.
    assert _layer_positions(ids, []) == pos


@pytest.mark.unit
def test_a_narrow_layer_is_left_on_one_line() -> None:
    """Wrapping is for the case that needs it; three nodes are not that case."""
    pos = _layer_positions(["a", "b", "c"], [])
    assert len({y for _, y in pos.values()}) == 1


@pytest.mark.unit
def test_layering_still_puts_premises_above_what_rests_on_them() -> None:
    """The wrap must not disturb the one thing the layout is for."""
    pos = _layer_positions(["p", "q", "combo"], [("p", "combo"), ("q", "combo")])
    assert pos["p"][1] == pos["q"][1] < pos["combo"][1]


@pytest.mark.unit
def test_a_stale_timestamp_does_not_break_the_stamp(wired: Path) -> None:
    """An `at` before anything existed yields a graph with no nodes, not a crash."""
    read = ReadModel(wired)
    try:
        before = (utcnow() - timedelta(days=365)).isoformat()
        assert read.graph(at=before).nodes == []
    finally:
        read.close()
