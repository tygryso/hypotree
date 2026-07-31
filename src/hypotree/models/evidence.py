"""Pydantic models: Evidence — discriminated union with error-type isolation.

LogicalEvidence carries a continuous success ∈ [0,1] and updates the Beta posterior.
InfraError never touches the posterior and never triggers INVALIDATED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


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
