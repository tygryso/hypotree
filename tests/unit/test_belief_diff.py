"""Tests for the belief diff — `generate_learning_path(since=...)`.

A belief state's *changes* are more interesting than its state. "Three
hypotheses were confirmed and one was withdrawn" is the sentence a standup, a PR
description or a review wants, and reconstructing it by diffing two full
narratives by eye is what people did instead.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.models.evidence import LogicalEvidence
from hypotree.store.store import utcnow


@pytest.fixture
def engine(tmp_path: Path):
    e = HypoTreeEngine(tmp_path / "diff.db", rng_seed=7)
    try:
        yield e
    finally:
        e.close()


@pytest.mark.unit
def test_a_diff_reports_only_what_changed_in_the_window(engine: HypoTreeEngine) -> None:
    engine.create_hypotheses(
        [{"statement": f"c={i}", "node_id": f"c{i}", "exclusion_group": "c"} for i in range(4)]
    )
    engine.record_evidence("c0", LogicalEvidence(success=0.0, depth=1))
    midpoint = utcnow()
    time.sleep(0.02)
    engine.record_evidence("c1", LogicalEvidence(success=0.0, depth=1))

    full = engine.generate_learning_path()
    diff = engine.generate_learning_path(since=midpoint)

    assert len(diff.steps) < len(full.steps)
    assert {s.node_id for s in diff.steps} == {"c1"}
    assert diff.settled_in_window == 1
    assert diff.probes_in_window == 1
    # The lifetime counters keep describing the whole history: "what did this
    # cost in total" does not become a different question this week.
    assert diff.probes_spent == full.probes_spent == 2


@pytest.mark.unit
def test_an_unchanged_window_diffs_to_nothing(engine: HypoTreeEngine) -> None:
    """A diff that reports activity when nothing was concluded is worse than none."""
    engine.create_hypotheses([{"statement": "a", "node_id": "a"}])
    engine.record_evidence("a", LogicalEvidence(success=1.0, depth=1))
    time.sleep(0.02)

    quiet = engine.generate_learning_path(since=utcnow())
    assert quiet.steps == []
    assert quiet.settled_in_window == 0
    assert quiet.withdrawn_in_window == 0
    assert "Nothing was settled or withdrawn" in quiet.markdown


@pytest.mark.unit
def test_a_withdrawal_is_the_most_interesting_thing_in_a_window(
    engine: HypoTreeEngine,
) -> None:
    """A belief *retracted* in the window is the easiest thing to omit and the
    one a reader most needs — a snapshot of the current state cannot show it."""
    engine.create_hypotheses(
        [
            {"statement": "premise", "node_id": "p"},
            {"statement": "combo", "node_id": "combo", "parent_ids": ["p"]},
        ]
    )
    engine.record_evidence("p", LogicalEvidence(success=1.0, depth=1))
    midpoint = utcnow()
    time.sleep(0.02)
    engine.record_evidence("combo", LogicalEvidence(success=0.0, depth=2))

    diff = engine.generate_learning_path(since=midpoint)
    assert diff.withdrawn_in_window >= 1
    assert any(s.origin == "reversed" for s in diff.steps)


@pytest.mark.unit
def test_the_diff_heading_says_it_is_a_diff(engine: HypoTreeEngine) -> None:
    """Rendered as a narrative, a windowed report reads as a suspiciously short
    history rather than as an answer to 'what changed'."""
    engine.create_hypotheses([{"statement": "a", "node_id": "a"}])
    start = utcnow() - timedelta(minutes=5)
    engine.record_evidence("a", LogicalEvidence(success=1.0, depth=1))

    diff = engine.generate_learning_path(since=start)
    assert diff.markdown.startswith("# What changed")
    assert "settled" in diff.markdown
    # Without the bound it is still the full narrative.
    assert engine.generate_learning_path().markdown.startswith("# What we have learned so far")


@pytest.mark.unit
def test_since_and_as_of_bound_a_closed_window(engine: HypoTreeEngine) -> None:
    engine.create_hypotheses([{"statement": f"n{i}", "node_id": f"n{i}"} for i in range(3)])
    engine.record_evidence("n0", LogicalEvidence(success=1.0, depth=1))
    start = utcnow()
    time.sleep(0.02)
    engine.record_evidence("n1", LogicalEvidence(success=1.0, depth=1))
    time.sleep(0.02)
    end = utcnow()
    time.sleep(0.02)
    engine.record_evidence("n2", LogicalEvidence(success=1.0, depth=1))

    windowed = engine.generate_learning_path(since=start, as_of=end)
    assert {s.node_id for s in windowed.steps} == {"n1"}


@pytest.mark.unit
def test_a_backwards_window_is_refused(engine: HypoTreeEngine) -> None:
    """Silently returning nothing would read as 'nothing changed'."""
    now = utcnow()
    with pytest.raises(ValueError, match="running backwards"):
        engine.generate_learning_path(since=now, as_of=now - timedelta(hours=1))


@pytest.mark.unit
def test_a_naive_instant_is_read_as_utc(engine: HypoTreeEngine) -> None:
    """Stored instants are aware, so a naive bound would raise on first compare."""
    engine.create_hypotheses([{"statement": "a", "node_id": "a"}])
    engine.record_evidence("a", LogicalEvidence(success=1.0, depth=1))
    naive = (utcnow() - timedelta(minutes=1)).replace(tzinfo=None)
    assert engine.generate_learning_path(since=naive).settled_in_window == 1
