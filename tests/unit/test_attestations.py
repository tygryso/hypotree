"""Runner attestation persistence and evidence provenance contracts."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hypotree import HypoTreeEngine, LogicalEvidence, RunAttestation
from hypotree.toolkit.specs import get_spec


def _attestation(attestation_id: str = "run-1") -> RunAttestation:
    return RunAttestation(
        id=attestation_id,
        runner="hyporun/1",
        workspace_id="workspace-1",
        base_commit="abc123",
        argv=["uv", "run", "pytest", "-q"],
        exit_code=0,
        duration_s=1.25,
        stdout_digest="sha256:stdout",
        stderr_digest="sha256:stderr",
        created_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )


@pytest.mark.unit
def test_attestation_round_trip_and_immutable_id(tmp_path: Path) -> None:
    engine = HypoTreeEngine(tmp_path / "belief.db")
    try:
        attestation = _attestation()
        engine.add_attestation(attestation)
        assert engine.get_attestation(attestation.id) == attestation
        with pytest.raises(sqlite3.IntegrityError):
            engine.add_attestation(attestation)
    finally:
        engine.close()


@pytest.mark.unit
def test_attested_evidence_is_posterior_neutral_and_derives_provenance(
    tmp_path: Path,
) -> None:
    plain = HypoTreeEngine(tmp_path / "plain.db")
    attested = HypoTreeEngine(tmp_path / "attested.db")
    try:
        for engine in (plain, attested):
            engine.create_hypothesis("works", node_id="n1")
        attested.add_attestation(_attestation())

        plain_result = plain.record_evidence("n1", LogicalEvidence(success=1.0, depth=3))
        attested_result = attested.record_evidence(
            "n1", LogicalEvidence(success=1.0, depth=3, attestation_id="run-1")
        )

        assert attested_result.node.status == plain_result.node.status
        assert attested_result.node.alpha == plain_result.node.alpha
        assert attested_result.node.beta == plain_result.node.beta
        history = attested.get_evidence_history("n1")[0]
        assert history.verified_by == "attested"
        assert history.context_hash == "abc123"
        assert history.source_ref == "attestation:run-1"
        assert history.attestation_id == "run-1"
    finally:
        plain.close()
        attested.close()


@pytest.mark.unit
def test_unknown_attestation_degrades_and_context_disagreement_is_flagged(
    tmp_path: Path,
) -> None:
    engine = HypoTreeEngine(tmp_path / "belief.db")
    try:
        engine.create_hypothesis("unknown", node_id="unknown")
        engine.record_evidence("unknown", LogicalEvidence(success=1.0, attestation_id="missing"))
        unknown = engine.get_evidence_history("unknown")[0]
        assert unknown.verified_by == "self_reported"
        assert unknown.attestation_id is None

        engine.create_hypothesis("mismatch", node_id="mismatch")
        engine.add_attestation(_attestation())
        engine.record_evidence(
            "mismatch",
            LogicalEvidence(
                success=1.0,
                context_hash="different",
                attestation_id="run-1",
            ),
        )
        mismatch = engine.get_evidence_history("mismatch")[0]
        assert mismatch.context_hash == "different"
        assert mismatch.attestation_context_mismatch is True
    finally:
        engine.close()


@pytest.mark.unit
def test_tool_schema_accepts_reference_but_cannot_mint_attestation() -> None:
    schema = get_spec("record_evidence").input_schema
    properties = schema["properties"]
    result_properties = properties["results"]["items"]["properties"]
    assert "attestation_id" in properties
    assert "attestation_id" in result_properties
    forbidden = {"runner", "workspace_id", "base_commit", "argv", "exit_code"}
    assert forbidden.isdisjoint(properties)
    assert forbidden.isdisjoint(result_properties)
