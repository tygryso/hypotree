"""Pydantic models: ElisionNode — collapsed siblings in bounded context.

When a node has more children than max_children, the overflow collapses into a
single ElisionNode so the agent gets a deterministic expand affordance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ElisionNode(BaseModel):
    """Placeholder for max_children overflow at one level of the DAG."""

    kind: Literal["elision"] = "elision"
    parent_id: str
    hidden_count: int
    drill_id: str | None = None
