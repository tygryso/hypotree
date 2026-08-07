"""Pydantic models: Node — the hypothesis entity.

parent_ids is DERIVED from the edges table on load, not stored on the node,
because the edge TYPE would be lost if duplicated as a plain list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from hypotree.models.status import Status, utcnow


class Node(BaseModel):
    """A single hypothesis node in the R&D DAG."""

    id: str
    statement: str
    status: Status = Status.UNTESTED
    # Derived from the edges table on load — not stored redundantly on the node
    # .
    parent_ids: list[str] = Field(default_factory=list)
    evidence_regime: Literal["deterministic", "stochastic"] = "deterministic"
    is_parametric: bool = False
    param_config: dict | None = None

    # Goal / termination (A7, 2.10). target_metric doubles as the node's verify
    # bar for goals. Global stop = ALL goal nodes VERIFIED (convergence-gated).
    is_goal: bool = False
    target_metric: float | None = None

    # Mutual exclusion. Nodes sharing a non-null exclusion_group are competing
    # answers to the same question, of which exactly one can be true ("which
    # catalyst", "which architecture"). Confirming one lets the engine *infer*
    # that the others are settled without ever testing them — belief revision
    # driven by a logical constraint rather than by observation. The inference
    # is retracted automatically if the confirmation is later withdrawn.
    exclusion_group: str | None = None

    # Whether the group's candidates are believed to be *all* of them. This is
    # the closed-world assumption, and until it was declared the engine simply
    # assumed it: "all but one eliminated" confirmed the survivor for free, which
    # is sound over a complete list of answers and false over a partial one.
    # "Which catalyst, of these three" is closed; "which learning rate" is not,
    # because the next candidate always exists. Defaults to True because that is
    # what every group written before this field meant.
    exclusion_closed: bool = True

    # Roughly what testing this costs, in seconds, before anything has been
    # timed. A *prior*, superseded by the first real observation — the doctrine
    # is that cost is estimated from what was actually measured, and a caller
    # asked to guess guesses once and never revises.
    #
    # It exists because the observed model is blind exactly where the saving is.
    # A question is settled once, so the answers competing to settle it have no
    # history at the moment the navigator must choose between them, and the
    # sibling-median fallback hands every one of them the identical number.
    # Ordering *across* questions saves nothing (every question must be settled
    # anyway); ordering *within* one saves the expensive answer entirely, because
    # the last survivor of a closed question is deduced rather than probed.
    #
    # Safe to consume unmeasured in a way an accuracy prior is not: cost changes
    # only what is *tried next* and never what the belief state *asserts*, so a
    # wrong estimate costs a worse order and is corrected by the first timing.
    estimated_cost: float | None = Field(default=None, gt=0.0)

    # Depth (rigour / scale / context) of the observation that confirmed this
    # node, or None if it was never confirmed. Recorded because a confirmation
    # is only as strong as the test that produced it: a composition that fails
    # at depth D is evidence against exactly those of its assumptions that were
    # never confirmed at depth D or deeper.
    confirmed_depth: int | None = None

    # Thompson Sampling posterior (pseudo-count update).
    # Prior = uniform Beta(1, 1).
    alpha: float = 1.0
    beta: float = 1.0
    # Number of logical observations folded into the posterior (infra errors
    # excluded, consistent with the convergence gate).
    evidence_count: int = 0

    # Claim/lease (A2, 2.6) — consumed on first evidence.
    active_claim_id: str | None = None
    claimed_at: datetime | None = None

    # Infra retry accounting → BLOCKED after max_retries.
    infra_retry_count: int = 0

    # Telemetry (A4).
    created_at: datetime = Field(default_factory=utcnow)
    first_dispatched_at: datetime | None = None
    first_evidence_at: datetime | None = None
    verified_at: datetime | None = None
    # Set on → INVALIDATED and → PRUNED transitions (symmetry with verified_at).
    invalidated_at: datetime | None = None
    pruned_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)

    def model_post_init(self, __context: object) -> None:
        """Sync updated_at to created_at on first creation (same instant).

        Only fires when updated_at was NOT supplied explicitly. Nodes rebuilt
        from the store pass a distinct updated_at (advanced by later status
        changes / posterior updates); clobbering it here would reset every
        loaded node's staleness clock and break the sampler's tiebreak.
        """
        if "updated_at" not in self.model_fields_set:
            self.updated_at = self.created_at

    # Derived (not stored): lead_time = verified_at - created_at;
    #                       cycle_time = verified_at - first_dispatched_at
