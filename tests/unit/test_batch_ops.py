"""Tests for batch operations: create_hypotheses and bulk_update_status."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import CycleError, HypoTreeEngine, NodeNotFoundError
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    eng = HypoTreeEngine(tmp_path / "state.db", rng_seed=42, project_path=tmp_path)
    yield eng
    eng.close()


# -- create_hypotheses ------------------------------------------------------


@pytest.mark.unit
def test_create_many_in_one_call(engine: HypoTreeEngine) -> None:
    """Create multiple nodes in one call."""
    results = engine.create_hypotheses(
        [
            {"statement": "alpha", "node_id": "n1"},
            {"statement": "beta", "node_id": "n2"},
            {"statement": "gamma", "node_id": "n3"},
        ]
    )
    assert len(results) == 3
    assert all(r.created for r in results)
    assert [r.node.id for r in results] == ["n1", "n2", "n3"]


@pytest.mark.unit
def test_create_one_is_the_same_call(engine: HypoTreeEngine) -> None:
    """A single hypothesis is a list of one — there is no second tool to reach for."""
    results = engine.create_hypotheses([{"statement": "solo", "node_id": "n1"}])
    assert len(results) == 1
    assert results[0].created is True
    assert results[0].node.statement == "solo"


@pytest.mark.unit
def test_intra_batch_parents(engine: HypoTreeEngine) -> None:
    """A child in the batch can reference a parent created in the same batch."""
    results = engine.create_hypotheses(
        [
            {"statement": "parent", "node_id": "p1"},
            {"statement": "child", "node_id": "c1", "parent_ids": ["p1"]},
        ]
    )
    assert results[0].created is True
    assert results[1].node.parent_ids == ["p1"]


@pytest.mark.unit
def test_child_before_parent_is_sorted_out(engine: HypoTreeEngine) -> None:
    """Declaration order is not the caller's problem — the edges already state it."""
    results = engine.create_hypotheses(
        [
            {"statement": "child", "node_id": "c1", "parent_ids": ["p1"]},
            {"statement": "parent", "node_id": "p1"},
        ]
    )
    # Results come back in input order regardless of the order they were applied.
    assert [r.node.id for r in results] == ["c1", "p1"]
    assert results[0].node.parent_ids == ["p1"]


@pytest.mark.unit
def test_deep_chain_declared_backwards(engine: HypoTreeEngine) -> None:
    """The ordering is a real topological sort, not a one-step lookahead."""
    engine.create_hypotheses(
        [
            {"statement": "d", "node_id": "d1", "parent_ids": ["c1"]},
            {"statement": "c", "node_id": "c1", "parent_ids": ["b1"]},
            {"statement": "b", "node_id": "b1", "parent_ids": ["a1"]},
            {"statement": "a", "node_id": "a1"},
        ]
    )
    assert engine._store.get_node("d1").parent_ids == ["c1"]  # noqa: SLF001
    assert engine._store.get_node("b1").parent_ids == ["a1"]  # noqa: SLF001


@pytest.mark.unit
def test_cycle_inside_the_batch_is_named(engine: HypoTreeEngine) -> None:
    """Two hypotheses that each depend on the other cannot be ordered at all."""
    with pytest.raises(CycleError, match="cycle"):
        engine.create_hypotheses(
            [
                {"statement": "a", "node_id": "a1", "parent_ids": ["b1"]},
                {"statement": "b", "node_id": "b1", "parent_ids": ["a1"]},
            ]
        )


@pytest.mark.unit
def test_collision_policy_skip(engine: HypoTreeEngine) -> None:
    """Per-item if_exists='skip' returns the existing node for that item."""
    engine.create_hypotheses([{"statement": "original", "node_id": "n1"}])
    results = engine.create_hypotheses(
        [
            {"statement": "replacement", "node_id": "n1", "if_exists": "skip"},
            {"statement": "new", "node_id": "n2"},
        ]
    )
    assert results[0].created is False
    assert results[0].reason == "id_exists"
    assert results[0].node.statement == "original"
    assert results[1].created is True


@pytest.mark.unit
def test_collision_under_error_policy_writes_nothing(engine: HypoTreeEngine) -> None:
    """A rejected batch leaves the graph exactly as it was.

    The failure is found before anything is applied, so the caller can fix the
    one bad entry and resend the whole list — rather than having to work out
    which prefix of it already exists.
    """
    engine.create_hypotheses([{"statement": "existing", "node_id": "n1"}])
    with pytest.raises(ValueError, match="already exists"):
        engine.create_hypotheses(
            [
                {"statement": "ok", "node_id": "n2"},
                {"statement": "collision", "node_id": "n1"},
            ]
        )
    assert engine._store.get_node("n2") is None  # noqa: SLF001


@pytest.mark.unit
def test_missing_parent_writes_nothing(engine: HypoTreeEngine) -> None:
    """A parent that is neither in the store nor in the batch is a dangling edge."""
    with pytest.raises(ValueError, match="neither exists nor"):
        engine.create_hypotheses(
            [
                {"statement": "ok", "node_id": "n1"},
                {"statement": "orphan", "node_id": "n2", "parent_ids": ["ghost"]},
            ]
        )
    assert engine._store.get_node("n1") is None  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "non-empty list"),
        ("component=v0", "non-empty list"),
        (["component=v0"], "not an object"),
        ([{"node_id": "n1"}], "non-empty `statement`"),
        ([{"statement": "  "}], "non-empty `statement`"),
        ([{"statement": "a", "exclusion": "x"}], "unknown field"),
        ([{"statement": "a", "node_id": "n1"}, {"statement": "b", "node_id": "n1"}], "reuses"),
        ([{"statement": "a", "if_exists": "replace"}], "invalid if_exists"),
    ],
)
def test_malformed_payloads_say_what_shape_was_wanted(
    engine: HypoTreeEngine, payload: object, match: str
) -> None:
    """Every rejection names the offending entry and what it should look like.

    The batch creation tool was the single most-failed call in a full evaluation
    run, always on payload shape. An error the caller cannot act on costs a whole
    turn and sometimes the entire hypothesis graph.
    """
    with pytest.raises(ValueError, match=match):
        engine.create_hypotheses(payload)  # type: ignore[arg-type]


@pytest.mark.unit
def test_unknown_field_lists_the_accepted_ones(engine: HypoTreeEngine) -> None:
    """Naming the alternatives is what makes the error recoverable in one turn."""
    with pytest.raises(ValueError) as excinfo:
        engine.create_hypotheses([{"statement": "a", "group": "component"}])
    message = str(excinfo.value)
    assert "'group'" in message
    assert "exclusion_group" in message


# -- bulk_update_status ----------------------------------------------------


@pytest.mark.unit
def test_bulk_update_status_basic(engine: HypoTreeEngine) -> None:
    """Batch update multiple nodes to the same status."""
    engine.create_hypothesis("a", node_id="n1")
    engine.create_hypothesis("b", node_id="n2")
    results = engine.bulk_update_status(["n1", "n2"], Status.VERIFIED, reason="batch")
    assert len(results) == 2
    assert all(r.node.status == Status.VERIFIED for r in results)
    assert all(r.old_status == Status.UNTESTED for r in results)
    assert all(r.transition == "UNTESTED → VERIFIED" for r in results)


@pytest.mark.unit
def test_bulk_update_status_mixed_priors(engine: HypoTreeEngine) -> None:
    """Each node reports its own old_status independently."""
    engine.create_hypothesis("a", node_id="n1")
    engine.create_hypothesis("b", node_id="n2")
    engine.update_status("n1", Status.IN_PROGRESS, reason="started")
    results = engine.bulk_update_status(["n1", "n2"], Status.VERIFIED, reason="batch")
    assert results[0].old_status == Status.IN_PROGRESS
    assert results[1].old_status == Status.UNTESTED


@pytest.mark.unit
def test_bulk_update_status_missing_id_no_partial(engine: HypoTreeEngine) -> None:
    """A bad id is caught up front so no node is partially updated."""
    engine.create_hypothesis("a", node_id="n1")
    with pytest.raises(NodeNotFoundError, match="nodes not found"):
        engine.bulk_update_status(["n1", "ghost"], Status.VERIFIED, reason="batch")
    # n1 must remain untouched — the batch validated before mutating.
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.status == Status.UNTESTED


# -- integration: create + bulk update + verify ---------------------------


@pytest.mark.integration
def test_create_then_bulk_verify(engine: HypoTreeEngine) -> None:
    """Integration: create a tree in one call, then batch-verify all nodes."""
    engine.create_hypotheses(
        [
            {"statement": "goal", "node_id": "g1", "is_goal": True, "target_metric": 0.8},
            {"statement": "sub1", "node_id": "s1", "parent_ids": ["g1"]},
            {"statement": "sub2", "node_id": "s2", "parent_ids": ["g1"]},
        ]
    )

    results = engine.bulk_update_status(["g1", "s1", "s2"], Status.VERIFIED, reason="all done")
    assert len(results) == 3
    assert all(r.node.status == Status.VERIFIED for r in results)

    table = engine.list_nodes(status_filter=["VERIFIED"])
    assert "g1" in table
    assert "s1" in table
    assert "s2" in table
