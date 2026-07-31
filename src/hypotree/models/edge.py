"""Pydantic models: Edge — typed parent→child relation.

Three edge types govern frontier eligibility:
- DEPENDENCY: child eligible only when ALL parents are VERIFIED (strict gate).
- ALTERNATIVE: child eligible when ANY parent is VERIFIED (fallback gate).
- REFINEMENT: child eligible when parent is IN_PROGRESS or VERIFIED (loose gate).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class EdgeType(str, Enum):
    """Type of parent→child edge in the hypothesis DAG."""

    DEPENDENCY = "DEPENDENCY"
    ALTERNATIVE = "ALTERNATIVE"
    REFINEMENT = "REFINEMENT"


class Edge(BaseModel):
    """A directed, typed edge from src (parent) to dst (child)."""

    src: str
    dst: str
    type: EdgeType
