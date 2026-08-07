"""Pydantic models: Events — audit/replay stream.

Events live inside the SQLite events table (written same-transaction). They are
NOT a live projection source — the engine reads materialized tables directly.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from hypotree.models.edge import Edge
from hypotree.models.evidence import Evidence
from hypotree.models.node import Node
from hypotree.models.status import Status


class Event(BaseModel):
    """Base for all events — carries monotonic sequence and transaction grouping."""

    seq: int | None = None
    txn_id: str | None = None
    written_at: datetime | None = None


class NodeCreated(Event):
    node: Node


class EdgeAdded(Event):
    edge: Edge


class TargetClaimed(Event):
    node_id: str
    claim_id: str
    claimed_at: datetime


class EvidenceRecorded(Event):
    node_id: str
    evidence: Evidence


class StatusChanged(Event):
    node_id: str
    old: Status
    new: Status
    reason: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class StatusReattributed(Event):
    """The status held, the justification for it changed.

    Distinct from ``StatusChanged`` because no transition happened: a sibling
    retired by one confirmation is re-attributed to another when the first is
    withdrawn. Recording it as a transition would open a second history interval
    at the instant the first closed; recording it as nothing left the marker
    naming a node that no longer confirms anything.
    """

    node_id: str
    reason: str = ""


class SubtreePruned(Event):
    root_id: str
    pruned_ids: list[str]


class UpstreamInvalidated(Event):
    leaf_id: str
    affected_ids: list[str]


class UpstreamVerified(Event):
    child_id: str
    affected_ids: list[str]
    depth: int
