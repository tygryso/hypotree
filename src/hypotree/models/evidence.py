"""Pydantic models: Evidence — discriminated union with error-type isolation.

LogicalEvidence carries a continuous success ∈ [0,1] and updates the Beta posterior.
InfraError never touches the posterior and never triggers INVALIDATED.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RunAttestation(BaseModel):
    """Runner-minted immutable metadata describing one observed process."""

    id: str
    runner: str
    workspace_id: str | None = None
    base_commit: str | None = None
    argv: list[str]
    exit_code: int | None = None
    duration_s: float | None = Field(default=None, ge=0.0)
    patch_digest: str | None = None
    stdout_digest: str | None = None
    stderr_digest: str | None = None
    created_at: datetime


class LogicalEvidence(BaseModel):
    """Experiment result on the continuous success scale."""

    kind: Literal["logical"] = "logical"
    success: float = Field(ge=0.0, le=1.0)
    # Rigour / scale / context at which the observation was made. Higher means a
    # more demanding test. A confirmation at depth d supports claims at depth
    # <= d only: "it worked in the unit test" is not "it works in production".
    # Compositions record the depth they were tested at, so the engine can tell
    # which of their assumptions were confirmed too shallowly to support them.
    depth: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[Path] = Field(default_factory=list)
    context_hash: str | None = None
    git_branch: str | None = None
    # What was actually run to produce this number — a path, a URL, a CI run id,
    # a commit. Optional and unvalidated: an audit trail that says "0.85" and an
    # audit trail that says "0.85, from pytest run #4412" are different
    # artifacts, and only the caller knows which artifact exists.
    source_ref: str | None = None
    # Wall-clock seconds the experiment took. Optional and never inferred: it is
    # what makes cost *observed* rather than declared, and a caller asked to
    # estimate cost guesses once and never revises. None means unknown, which is
    # not the same as free.
    duration_s: float | None = Field(default=None, ge=0.0)
    # Reference only. Tool callers cannot create attestation rows; a trusted
    # runner must have minted this id before evidence cites it.
    attestation_id: str | None = None
    attestation_context_mismatch: bool = False
    claim_id: str | None = None
    notes: str = ""
    delta_success: float | None = None
    delta_metrics: dict[str, float] = Field(default_factory=dict)
    monotonicity: Literal["up", "down", "flat", "first"] = "first"


class InfraError(BaseModel):
    """Infrastructure/environmental error — retriable, never invalidates."""

    kind: Literal["infra"] = "infra"
    error_type: str
    message: str
    retriable: bool = True
    claim_id: str | None = None


Evidence = Annotated[LogicalEvidence | InfraError, Field(discriminator="kind")]
