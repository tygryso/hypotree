"""Unit tests: Pydantic models."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.models import (
    Edge,
    EdgeType,
    ElisionNode,
    Evidence,
    InfraError,
    LogicalEvidence,
    Node,
    Status,
    posterior_mean,
    posterior_variance,
)


@pytest.mark.unit
class TestStatus:
    def test_eight_statuses(self) -> None:
        assert len(Status) == 8

    def test_string_values(self) -> None:
        assert Status.UNTESTED == "UNTESTED"
        assert Status.NEEDS_REVISION == "NEEDS_REVISION"
        assert Status.EXHAUSTED == "EXHAUSTED"


@pytest.mark.unit
class TestNode:
    def test_defaults(self) -> None:
        n = Node(id="n1", statement="test hypothesis")
        assert n.status == Status.UNTESTED
        assert n.alpha == 1.0
        assert n.beta == 1.0
        assert n.evidence_regime == "deterministic"
        assert n.is_goal is False
        assert n.target_metric is None
        assert n.infra_retry_count == 0
        assert n.parent_ids == []
        assert n.created_at is not None
        assert n.updated_at == n.created_at

    def test_posterior_fields(self) -> None:
        n = Node(id="n1", statement="x", alpha=3.0, beta=2.0)
        assert posterior_mean(n.alpha, n.beta) == pytest.approx(0.6)
        assert posterior_variance(n.alpha, n.beta) > 0

    def test_title_is_optional_and_bounded(self) -> None:
        assert Node(id="n", title="Parser", statement="works").title == "Parser"
        with pytest.raises(ValueError):
            Node(id="n", title="x" * 129, statement="works")

    def test_timestamps_auto_filled(self) -> None:
        n = Node(id="n1", statement="x")
        assert n.created_at is not None
        assert n.updated_at is not None

    def test_updated_at_synced_on_fresh_creation(self) -> None:
        """A freshly-created node has updated_at == created_at (same instant)."""
        n = Node(id="n1", statement="x")
        assert n.updated_at == n.created_at

    def test_updated_at_preserved_when_explicit(self) -> None:
        """An explicitly-supplied updated_at (e.g. rebuilt from the store) is not clobbered."""
        from datetime import datetime, timezone

        created = datetime(2020, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2020, 6, 1, tzinfo=timezone.utc)
        n = Node(id="n1", statement="x", created_at=created, updated_at=updated)
        assert n.created_at == created
        assert n.updated_at == updated


@pytest.mark.unit
class TestEdge:
    def test_three_types(self) -> None:
        assert len(EdgeType) == 3

    def test_edge_model(self) -> None:
        e = Edge(src="p", dst="c", type=EdgeType.REFINEMENT)
        assert e.src == "p"
        assert e.dst == "c"
        assert e.type == EdgeType.REFINEMENT


@pytest.mark.unit
class TestEvidence:
    def test_logical_success_bounds(self) -> None:
        LogicalEvidence(success=0.0)
        LogicalEvidence(success=1.0)

    def test_logical_success_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            LogicalEvidence(success=-0.1)
        with pytest.raises(ValueError):
            LogicalEvidence(success=1.1)

    def test_infra_error(self) -> None:
        err = InfraError(error_type="TimeoutError", message="timed out")
        assert err.retriable is True
        assert err.kind == "infra"

    def test_discriminator_logical(self) -> None:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Evidence)
        obj = adapter.validate_python({"kind": "logical", "success": 0.7})
        assert isinstance(obj, LogicalEvidence)
        assert obj.success == pytest.approx(0.7)

    def test_discriminator_infra(self) -> None:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Evidence)
        obj = adapter.validate_python({"kind": "infra", "error_type": "OOM", "message": "oom"})
        assert isinstance(obj, InfraError)

    def test_logical_has_provenance(self) -> None:
        ev = LogicalEvidence(success=0.5, artifacts=[Path("/tmp/log.txt")])
        assert ev.artifacts == [Path("/tmp/log.txt")]


@pytest.mark.unit
class TestElisionNode:
    def test_elision(self) -> None:
        e = ElisionNode(parent_id="p1", hidden_count=45)
        assert e.kind == "elision"
        assert e.hidden_count == 45
        assert e.drill_id is None
