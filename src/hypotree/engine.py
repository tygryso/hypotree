"""HypoTreeEngine — orchestrates graph + navigator + store into a closed loop.

The engine is the single entry point for every Tools API operation: it holds
the in-memory graph, the Thompson-Sampling navigator, and the SQLite store in
sync. Every mutation flows through the engine, which persists to the store,
writes the same-transaction event row, and updates the in-memory graph.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import uuid
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from hypotree.graph.dag import CycleError, HypoTreeGraph
from hypotree.models.edge import Edge, EdgeType
from hypotree.models.elision import ElisionNode
from hypotree.models.evidence import Evidence, InfraError, LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status, posterior_mean, utcnow
from hypotree.navigator.convergence import credible_interval
from hypotree.navigator.sampler import (
    DEFAULT_LEASE_TTL_S,
    ThompsonSampler,
)
from hypotree.store.identity import capture_git_context
from hypotree.store.store import HypoTreeStore

# Maximum infra retries before a node auto-transitions to BLOCKED.
MAX_INFRA_RETRIES = 3

# Statuses that still count as an open question, i.e. safe for the exclusion
# inference to settle. A node that already reached a terminal state by its own
# evidence is never overwritten by an inference about a sibling.
_OPEN_STATUSES = frozenset(
    {Status.UNTESTED, Status.IN_PROGRESS, Status.BLOCKED, Status.NEEDS_REVISION}
)

# Marks a status change caused by the mutual-exclusion inference rather than by
# evidence. The justifying node's id is appended, so the inference can be undone
# precisely (and only) when that specific confirmation is withdrawn.
EXCLUSION_REASON_PREFIX = "excluded by confirmed sibling: "

# Marks a confirmation placed under review because something built on top of it
# failed at a greater depth than it was ever tested at. The required depth is
# appended, so the navigator can tell the agent how deep the re-test must go.
REVIEW_REASON_PREFIX = "under conflict review — re-test at depth >= "

# Marks a sibling handed back to the frontier because the confirmation that had
# retired it was itself withdrawn. A constant rather than an inline string for
# the same reason as the others: anything that reads the history matches on the
# marker, and a literal written at the mutation site drifts away from its reader
# without either of them changing.
EXCLUSION_RETRACT_PREFIX = "exclusion retracted: "

# Marks a member confirmed without an observation, because every competing answer
# in its exclusion group has been ruled out.
DEDUCTION_REASON_PREFIX = "deduced by elimination: "

# Marks an alternative handed back to the frontier because the confirmation that
# had retired it takes part in a composition that fails while holding in
# isolation — the answer may be the alternative rather than the confirmed value.
INTERACTION_REOPEN_PREFIX = "reopened: "

# Marks the answer a diagnostic swap confirmed. The substitute was never probed
# on its own: the composition built around it succeeded where the same
# composition with the convicted member failed, and that success is a stronger
# observation about it than any isolated test would have been.
SUBSTITUTION_CONFIRM_PREFIX = "confirmed by successful substitution in: "

# Marks an answer ruled out by a diagnostic swap that stopped the failure without
# clearing the bar. Every other member of that composition has just been
# exonerated, so the composition was the right answer everywhere except this one
# slot and still fell short — which rules this value out for its own question.
SUBSTITUTION_ELIMINATE_PREFIX = "ruled out by sub-par substitution in: "

# Upper bound on the assignment space suggest_discriminating_experiment will
# enumerate exhaustively. Beyond this the search stops being instantaneous and
# the answer stops being worth its latency; the tool says so instead of guessing.
_MAX_SUGGESTION_SPACE = 100_000

# How many times the navigator will force a conflict suspect to the front before
# giving up on it. Prioritisation assumes the caller acts on the required depth;
# a caller that keeps re-testing too shallowly would otherwise be handed the same
# node forever while the rest of the frontier starved.
MAX_REVIEW_DISPATCHES = 3


class ClaimError(RuntimeError):
    """Raised when a claim is already consumed or not found."""


class GoalEvidenceError(ValueError):
    """Raised when evidence is recorded against a goal node.

    Its own error type because the caller's recovery is specific and worth
    naming: the result belongs on the hypothesis that was probed, not on the
    objective that hypothesis was meant to serve.
    """


class NodeNotFoundError(KeyError):
    """Raised when a referenced node does not exist."""


class TargetResponse(BaseModel):
    """Return value of get_next_targets — either a selection or a DONE sentinel."""

    status: str  # "SELECTED" | "DONE"
    node_id: str | None = None
    statement: str | None = None
    rationale: str = ""
    credible_interval: tuple[float, float] | None = None
    claim_id: str | None = None
    reason: str = ""
    # Minimum depth at which this target must be re-tested to be informative.
    # Set only for nodes implicated in an unresolved conflict: their existing
    # confirmation was obtained too shallowly to support what failed on top of
    # it, so repeating that shallow test would settle nothing.
    min_depth: int | None = None


class GoalStatusEntry(BaseModel):
    """One row in the get_goal_status response."""

    node_id: str
    statement: str
    target_metric: float | None
    posterior_mean: float
    status: Status
    met: bool


class GoalStatusResponse(BaseModel):
    """Return value of get_goal_status."""

    goals: list[GoalStatusEntry]
    all_met: bool
    goals_met_count: int = 0
    goals_total_count: int = 0
    frontier_size: int = 0
    total_nodes: int = 0
    status_breakdown: dict[str, int] = {}


class DagContextNode(BaseModel):
    """One node in a bounded subgraph view."""

    id: str
    statement: str
    status: Status
    posterior_mean: float
    credible_interval: tuple[float, float]
    parent_ids: list[str]
    is_goal: bool
    target_metric: float | None = None


class DagContextResponse(BaseModel):
    """Return value of get_dag_context — bounded subgraph with uncertainty."""

    nodes: list[DagContextNode]
    elisions: list[ElisionNode]
    max_depth: int
    max_children: int


class NodeSummary(BaseModel):
    """Compact projection of a Node for list/query results."""

    id: str
    statement: str
    status: Status
    posterior_mean: float
    is_goal: bool
    evidence_count: int
    created_at: datetime
    verified_at: datetime | None = None
    updated_at: datetime


class EvidenceSummary(BaseModel):
    """Compact projection of an evidence row for history queries."""

    id: int
    kind: str
    success: float | None
    metrics: dict[str, float]
    delta_success: float | None
    monotonicity: str
    context_hash: str | None
    git_branch: str | None
    notes: str
    recorded_at: datetime


class ActiveClaimSummary(BaseModel):
    """A live (unconsumed, unexpired) claim."""

    node_id: str
    claim_id: str
    claimed_at: datetime
    expires_in_s: int


class CreateHypothesisResult(BaseModel):
    """Return value of create_hypotheses — carries the node + collision info."""

    node: Node
    created: bool = True
    reason: str = ""


class RecordEvidenceResult(BaseModel):
    """Return value of record_evidence — the updated node + any fused dispatch.

    ``next_targets`` is empty unless the caller asked for it. Recording a result
    and asking what to do next are separate questions, and fusing them
    unconditionally would be wrong for the case this engine is actually for: an
    experiment that takes hours or days, run by a human who records its outcome
    and is not asking to be handed the next one in the same breath. When the
    caller *is* a tight synchronous loop, the two calls cost a full model
    round-trip each and the fusion halves that — so it is an opt-in accelerator
    rather than the shape of the API.
    """

    node: Node
    next_targets: list[TargetResponse] = []


class UpdateStatusResult(BaseModel):
    """Return value of update_status — carries node + prior-status audit trail."""

    node: Node
    old_status: Status
    transition: str


class LearningStep(BaseModel):
    """One conclusion the belief state reached, and how it reached it."""

    node_id: str
    statement: str
    status: Status
    at: datetime
    # "observed" when the conclusion rests on evidence recorded against this very
    # node, "inferred" when the engine derived it from other results, "reversed"
    # when a belief was withdrawn or handed back. The split is the whole point of
    # the report: it separates what cost an experiment from what did not.
    origin: Literal["observed", "inferred", "reversed"]
    how: str
    cost_a_probe: bool


class LearningPathResponse(BaseModel):
    """Return value of generate_learning_path.

    Both a rendered narrative and the structured steps behind it. The markdown is
    what an agent pastes into a summary or a human reads; the steps are what
    another tool can aggregate without re-parsing prose.
    """

    markdown: str
    steps: list[LearningStep]
    probes_spent: int
    conclusions: int
    conclusions_without_a_probe: int
    open_questions: list[str]
    open_conflicts: int
    goals_met: int
    goals_total: int


# How a settled node got there, keyed by the marker the engine wrote into the
# status-history reason. Imported nowhere else: the markers are the contract
# between the mutation that made the decision and the report that explains it,
# so they are matched here rather than re-described in prose that can drift.
_ORIGIN_BY_MARKER: tuple[tuple[str, Literal["observed", "inferred", "reversed"], str], ...] = (
    (DEDUCTION_REASON_PREFIX, "inferred", "deduced by elimination — every rival was ruled out"),
    (
        SUBSTITUTION_CONFIRM_PREFIX,
        "inferred",
        "confirmed by a diagnostic swap that made the composition work",
    ),
    (
        SUBSTITUTION_ELIMINATE_PREFIX,
        "inferred",
        "ruled out by a diagnostic swap that fell short with everything else correct",
    ),
    (EXCLUSION_REASON_PREFIX, "inferred", "retired because a competing answer was confirmed"),
    (
        EXCLUSION_RETRACT_PREFIX,
        "reversed",
        "reopened — the confirmation that had retired it was withdrawn",
    ),
    (
        INTERACTION_REOPEN_PREFIX,
        "reversed",
        "handed back — the confirmation that retired it failed in composition",
    ),
    (
        REVIEW_REASON_PREFIX,
        "reversed",
        "put under review — something built on it failed deeper than it was tested",
    ),
)


def _settlement_origin(reason: str) -> tuple[Literal["observed", "inferred", "reversed"], str]:
    """Read a status-history reason as an origin and a plain-language cause."""
    for marker, origin, phrase in _ORIGIN_BY_MARKER:
        if reason.startswith(marker):
            return origin, phrase
    return "observed", reason or "recorded evidence"


def _translate_like_wildcards(query: str) -> str:
    """Translate user-facing search wildcards into SQL LIKE syntax.

    `*` becomes `%` (multi-char); `_` stays `_` (single-char). Literal `%` and
    `_` in the input are escaped so they match themselves, not wildcard meta.
    """
    # Escape SQL LIKE metacharacters first so literal % and _ in the query
    # match themselves.
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    # Then translate the user-friendly * to the SQL % wildcard.
    return escaped.replace("*", "%")


class HypoTreeEngine:
    """The Tools API orchestrator.

    Holds an in-memory HypoTreeGraph synced to the SQLite store. Every mutation
    persists to the store (with same-transaction events) and updates the graph.
    """

    def __init__(
        self,
        db_path: Path | str,
        rng_seed: int | None = None,
        lease_ttl_s: int = DEFAULT_LEASE_TTL_S,
        project_path: Path | str | None = None,
    ) -> None:
        self._store = HypoTreeStore(db_path)
        self._graph = HypoTreeGraph()
        self._sampler = ThompsonSampler(
            np.random.default_rng(rng_seed),
            lease_ttl_s=lease_ttl_s,
        )
        self._lease_ttl_s = lease_ttl_s
        self._project_path = Path(project_path) if project_path else Path.cwd()
        # Exclusion group of the most recent dispatch, used to break selection
        # ties in favour of finishing the question already in progress.
        self._last_selected_group: str | None = None
        # How often each node has been forced to the front as a conflict suspect.
        # Session-scoped rather than persisted: it bounds one agent's behaviour
        # within a run and carries no belief content worth surviving a restart.
        self._review_dispatches: Counter[str] = Counter()
        self._sync_graph_from_store()

    def close(self) -> None:
        self._store.close()

    @property
    def project_path(self) -> Path:
        """The project this belief state belongs to."""
        return self._project_path

    def _sync_graph_from_store(self) -> None:
        """Rebuild the in-memory graph from the persisted store."""
        self._graph = HypoTreeGraph()
        for node in self._store.get_all_nodes():
            self._graph.add_node(node)
        for edge in self._store.get_all_edges():
            with contextlib.suppress(CycleError):
                self._graph.add_edge(edge)

    def _refresh_node_in_graph(self, node_id: str) -> None:
        """Reload a single node from the store into the in-memory graph."""
        node = self._store.get_node(node_id)
        if node is not None:
            self._graph.add_node(node)

    def _frontier_nodes(self) -> list[Node]:
        """Nodes that are eligible AND actually dispatchable right now.

        Two refinements on top of the graph's edge-type gate, both of which the
        rest of the system already assumed but nothing enforced:

        **Nodes under a live lease** (``active_claim_id`` set) are excluded. A
        claim exists to say "this dispatch owns this node"; handing the same node
        out again while its claim is unconsumed defeats the entire point, and did
        — an agent that batches its dispatches and reports results a turn later
        was served the same nodes repeatedly, re-probing configurations whose
        answers it was still holding. The lease is only meaningful if it removes
        the node from the pool. Callers reclaim expired leases first, so a lease
        that has run out never blocks anything.

        **Goal nodes are never dispatched.** A goal states an objective, not a
        claim to be tested: there is no experiment that settles it, so it can
        never leave the frontier under its own steam. Offering it anyway created
        a genuine trap — a goal cannot be invalidated or exhausted, so every
        result recorded against it left it exactly as dispatchable as before, and
        the navigator handed the same goal out turn after turn while the caller
        dutifully recorded probe after probe against it. The results of those
        probes were destroyed: they belonged to the hypotheses actually being
        tested, and a goal absorbs evidence without ever drawing a conclusion
        from it. Deprioritising goals rather than excluding them only delayed
        this to the moment the rest of the frontier emptied, which is precisely
        when the caller most needs to be told to compose an answer instead.
        Goal achievement is derived from what supports the goal — see
        ``goal_achieved`` — so nothing is lost by never dispatching it.

        **Members of a conflict still being narrowed** are excluded. Each has
        already passed the isolated test, so re-running it settles nothing; the
        experiment that discriminates a conflict is a composition. Leaving them
        dispatchable spent a tenth of a whole run's probe budget re-confirming
        assumptions that were never individually in question.
        """
        eligible = set(self._graph.eligible_frontier())
        under_diagnosis = {m for n in self._diagnosing_nogoods() for m in n["member_ids"]}
        return [
            n
            for n in self._store.get_all_nodes()
            if n.id in eligible
            and n.active_claim_id is None
            and not n.is_goal
            and n.id not in under_diagnosis
        ]

    def _blocked_nodes(self) -> list[tuple[str, list[str]]]:
        """Open questions that no experiment can currently reach, and what gates them.

        An empty frontier has two completely different meanings and only one of
        them is an ending. Everything may genuinely be settled — or the graph may
        be wired so that nothing is *reachable*, which is not a finished search
        but a modelling error waiting to be corrected. Reporting the second as
        the first ended a real run at step zero with twenty-five untested
        hypotheses sitting in the store, and scored it as a completed episode.

        Returns each unreachable open node with the unsatisfied DEPENDENCY
        parents responsible, so the caller is told what to fix rather than that
        there is nothing left to do.
        """
        eligible = set(self._graph.eligible_frontier())
        blocked: list[tuple[str, list[str]]] = []
        for node in self._store.get_all_nodes():
            if node.is_goal or node.id in eligible or node.status not in _OPEN_STATUSES:
                continue
            if node.active_claim_id is not None:
                continue
            unmet = [
                pid
                for pid in self._graph.parents(node.id, EdgeType.DEPENDENCY)
                if (p := self._store.get_node(pid)) is None or p.status != Status.VERIFIED
            ]
            if unmet:
                blocked.append((node.id, unmet))
        return blocked

    def _unreachable_goals(self) -> list[str]:
        """Goals that nothing can ever satisfy, because nothing supports them.

        Goal achievement is *derived*: a goal is reached when every hypothesis it
        DEPENDS on is verified. A goal wired to nothing therefore depends on
        nothing and can never be reached, no matter how much work is done — and
        the failure is silent, because the goal simply sits at UNTESTED while the
        search around it looks healthy. A caller that creates the objective first
        and forgets to wire its combinations to it (the natural order to write
        them in) gets exactly this, and every real episode of one evaluation run
        did.
        """
        return sorted(
            n.id
            for n in self._store.get_all_nodes()
            if n.is_goal and not self._graph.parents(n.id, EdgeType.DEPENDENCY)
        )

    def goal_achieved(self, node: Node) -> bool:
        """Whether a goal has been reached, derived from what supports it.

        A goal is achieved when the work it depends on is confirmed: every
        DEPENDENCY parent VERIFIED, and at least one such parent, because a goal
        nothing has been proposed for has plainly not been reached.

        Deliberately *not* read off the goal's own posterior. Evidence recorded
        against a goal can only ever verify it — the engine refuses to invalidate
        or exhaust a goal, since "not there yet" is not a refutation — so a
        posterior test is a one-way ratchet that mistakes accumulated attempts
        for success. Deriving the answer from the supporting hypotheses is both
        sound and the reason a goal never needs to be probed at all.
        """
        if not node.is_goal:
            return False
        parents = self._graph.parents(node.id, EdgeType.DEPENDENCY)
        if not parents:
            return False
        return all(
            (p := self._store.get_node(pid)) is not None and p.status == Status.VERIFIED
            for pid in parents
        )

    def _confirmed_for_composition(self) -> list[str]:
        """The confirmed hypotheses an untried composition should be built from.

        One confirmed answer per exclusion group, plus any confirmed hypothesis
        that answers no particular question. Only building blocks are offered:
        a node that already rests on DEPENDENCY parents is itself a composition,
        not an ingredient of one.

        Where a question has ended up with **two** confirmed answers — which is
        exactly what an interaction effect leaves behind, since the alternatives
        it reopens can confirm in isolation just as the original did — the one
        that has not yet been built on is preferred. That is the whole point of
        having reopened it.

        Returns nothing when the resulting set has already been composed, so the
        advice cannot become a loop that keeps proposing an assembly the caller
        has tried.
        """
        composed: set[frozenset[str]] = {
            frozenset(self._graph.parents(n.id, EdgeType.DEPENDENCY))
            for n in self._store.get_all_nodes()
            if not n.is_goal
        }
        used = {pid for parents in composed for pid in parents}

        by_group: dict[str, list[str]] = {}
        loose: list[str] = []
        for node in self._store.get_all_nodes():
            if node.status != Status.VERIFIED or node.is_goal:
                continue
            if self._graph.parents(node.id, EdgeType.DEPENDENCY):
                continue
            if node.exclusion_group:
                by_group.setdefault(node.exclusion_group, []).append(node.id)
            else:
                loose.append(node.id)

        chosen = [
            # Untried answers first, then by id so the advice is deterministic.
            sorted(by_group[group], key=lambda nid: (nid in used, nid))[0]
            for group in sorted(by_group)
        ]
        chosen += sorted(loose)
        if not chosen or frozenset(chosen) in composed:
            return []
        return chosen

    def _next_substitution(self, dry_run: bool) -> TargetResponse | None:
        """The next diagnostic swap to run, as an instruction the caller can act on.

        Issued instead of dispatching a node, because the experiment that
        narrows a conflict is a *composition* the caller must assemble, and there
        is no node for it until they do. The instruction names both halves — what
        to keep and what to swap in — so building it requires no reasoning about
        which assumptions were involved.

        A caller that keeps asking without running the experiment is not helped
        by being told a fourth time, so the advice is bounded and the conflict
        then falls through to the broad recovery. The bound is per *member*, not
        per conflict: clearing one assumption is progress, and a flat budget for
        the whole narrowing abandoned a five-assumption conflict two swaps from
        the answer.
        """
        for nogood in self._diagnosing_nogoods():
            plan = self._substitution_plan(nogood)
            if plan is None:
                continue
            nogood_id = int(plan["nogood_id"])
            key = f"nogood:{nogood_id}:{plan['member_id']}"
            if self._review_dispatches[key] >= MAX_REVIEW_DISPATCHES:
                self._recover_from_interaction(self._refetch_nogood(nogood_id) or nogood, utcnow())
                continue
            if not dry_run:
                self._review_dispatches[key] += 1
            cleared = plan["cleared"]
            already = f" Already cleared: {cleared}." if cleared else ""
            parents = sorted([*plan["keep_ids"], plan["candidate_id"]])
            return TargetResponse(
                status="DONE",
                reason="awaiting_substitution",
                rationale=(
                    f"'{plan['member_id']}' and {plan['keep_ids']} cannot all hold "
                    f"together, yet each holds on its own — so one of them only fails "
                    f"in combination. Build the same combination with "
                    f"'{plan['member_id']}' replaced by '{plan['candidate_id']}' "
                    f"(parent_ids={parents}, DEPENDENCY) and probe it at depth >= "
                    f"{plan['min_depth']}. If it still fails, '{plan['member_id']}' is "
                    f"cleared and I will name the next one; if it stops failing, "
                    f"'{plan['member_id']}' was the cause and its alternatives reopen."
                    f"{already} Do not re-test the assumptions on their own — each "
                    f"already passed that test, which is why this is unresolved."
                ),
                min_depth=plan["min_depth"],
            )
        return None

    def release_claims(self, claim_ids: list[str] | None = None) -> list[str]:
        """Hand leased nodes back to the frontier.

        With no argument, every lease: for a caller whose working context has
        been reset, which can no longer report on dispatches it does not
        remember making, so holding those leases only strands the nodes until
        their TTL runs out.

        With ``claim_ids``, exactly those: for a caller that has looked at a
        dispatched experiment and decided not to run it. Declining work is a
        normal outcome when the experiment costs a day of compute, and the only
        alternatives were to fabricate a result or to strand the node for the
        whole lease.
        """
        return self._store.release_claims(utcnow(), claim_ids)

    def renew_claim(self, claim_id: str, lease_ttl_s: int | None = None) -> ActiveClaimSummary:
        """Restart a live lease's clock, for work that is still in progress.

        A lease is a liveness signal, not a deadline: it exists so a node held by
        a caller that crashed or was reset comes back to the frontier. An
        experiment that runs for days will outlast any TTL short enough to serve
        that purpose, and sizing the TTL for the longest experiment instead makes
        every genuinely abandoned node unreclaimable for just as long. Renewal
        keeps both properties: short TTLs stay short, and a caller that is still
        working says so.

        Raises ``ClaimError`` for a lease that is consumed, expired or unknown —
        it may already have been handed to someone else, and silently re-arming
        it would put two callers on the same node.
        """
        now = utcnow()
        ttl = lease_ttl_s if lease_ttl_s is not None else self._lease_ttl_s
        if ttl <= 0:
            raise ValueError(f"lease_ttl_s must be > 0 to renew a lease, got {ttl}")
        if not self._store.renew_claim(claim_id, now, ttl):
            raise ClaimError(
                f"Claim {claim_id} cannot be renewed \u2014 it was already consumed, expired, "
                f"or superseded by a later dispatch. Call get_next_targets to be handed "
                f"work again; the node may since have gone to another caller."
            )
        claim = self._store.get_claim(claim_id)
        assert claim is not None
        return ActiveClaimSummary(
            node_id=claim["node_id"],
            claim_id=claim_id,
            claimed_at=now,
            expires_in_s=ttl,
        )

    def create_hypothesis(
        self,
        statement: str,
        parent_ids: list[str] | None = None,
        edge_type: EdgeType = EdgeType.DEPENDENCY,
        *,
        is_parametric: bool = False,
        evidence_regime: str = "deterministic",
        is_goal: bool = False,
        target_metric: float | None = None,
        param_config: dict[str, Any] | None = None,
        exclusion_group: str | None = None,
        node_id: str | None = None,
        if_exists: str = "error",
    ) -> CreateHypothesisResult:
        """Add a node (and edges to its parents). Validates acyclicity.

        ``exclusion_group`` marks the node as one of several competing answers to
        the same question, of which exactly one can be true. Confirming any
        member settles the rest by inference — see ``_apply_exclusion``.

        Collision policy via ``if_exists``:
        - ``"error"`` (default): raise ``ValueError`` if the ID already exists.
        - ``"overwrite"``: silently replace the existing node.
        - ``"skip"``: return the existing node with ``created=False``.
        """
        node_id = node_id or uuid.uuid4().hex[:12]

        if if_exists not in ("error", "overwrite", "skip"):
            raise ValueError(f"invalid if_exists policy: {if_exists!r}")

        # Collision guard — prevents silent belief-state overwrite.
        existing = self._store.get_node(node_id)
        if existing is not None:
            if if_exists == "error":
                raise ValueError(f"Node '{node_id}' already exists")
            if if_exists == "skip":
                return CreateHypothesisResult(node=existing, created=False, reason="id_exists")
            # if_exists == "overwrite": discard the old node entirely (edges,
            # evidence, history) so the recreated node has clean, single-open
            # history intervals and no duplicate creation event.
            self._store.delete_node(node_id)
            self._sync_graph_from_store()

        # Validate parent existence up front — a dangling edge to a missing
        # parent silently makes this node un-selectable forever (its parent gate
        # can never be satisfied) and leaves a phantom node in the graph.
        for pid in parent_ids or []:
            if self._store.get_node(pid) is None:
                raise NodeNotFoundError(f"parent node not found: {pid}")

        node = Node(
            id=node_id,
            statement=statement,
            is_parametric=is_parametric,
            evidence_regime=evidence_regime,  # type: ignore[arg-type]
            is_goal=is_goal,
            target_metric=target_metric,
            param_config=param_config,
            exclusion_group=exclusion_group,
        )
        self._store.add_node(node)

        for pid in parent_ids or []:
            edge = Edge(src=pid, dst=node_id, type=edge_type)
            try:
                self._graph.add_edge(edge)
                self._store.add_edge(edge)
            except CycleError:
                raise CycleError(f"edge {pid} → {node_id} would create a cycle") from None

        self._refresh_node_in_graph(node_id)
        return CreateHypothesisResult(
            node=self._store.get_node(node_id),  # type: ignore[arg-type]
            created=True,
        )

    def create_hypotheses(self, hypotheses: list[dict[str, Any]]) -> list[CreateHypothesisResult]:
        """Create one or many hypotheses (and their edges) in a single call.

        The only creation entry point, deliberately. A separate singular tool
        alongside a batch one shares every line of logic and differs only in
        arity, so the caller pays an extra decision — *which* of two
        interchangeable tools to reach for — on the operation it performs most
        often, and gets it wrong: in a full evaluation run the batch variant
        failed eight times on payload shape while the singular one, with the
        simpler schema, failed never. One tool that takes a list of one is
        strictly less to get wrong.

        The whole batch is validated **before** anything is written, and the
        items are then applied in dependency order rather than list order. Both
        exist for the same reason: a half-applied batch is the worst possible
        outcome, because the caller cannot tell from the exception which of its
        hypotheses now exist, and a graph that is missing an arbitrary suffix of
        itself is harder to repair than one that was never created.

        Results come back in **input order**, whatever order they were applied
        in, so the caller can match them to what it asked for positionally.
        """
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError(
                "`hypotheses` must be a non-empty list of objects, one per hypothesis, e.g. "
                '[{"statement": "component=v0", "node_id": "comp_v0", '
                '"exclusion_group": "component"}]. Pass a list of one to create a single '
                "hypothesis."
            )

        accepted = set(inspect.signature(self.create_hypothesis).parameters)
        seen_ids: set[str] = set()
        for index, spec in enumerate(hypotheses):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"hypotheses[{index}] is {type(spec).__name__}, not an object. Each entry "
                    f'needs at least a statement, e.g. {{"statement": "component=v0"}}.'
                )
            unknown = sorted(set(spec) - accepted)
            if unknown:
                raise ValueError(
                    f"hypotheses[{index}] has unknown field(s) {unknown}. "
                    f"Accepted fields: {sorted(accepted)}."
                )
            statement = spec.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError(
                    f"hypotheses[{index}] needs a non-empty `statement` — it is the claim "
                    f"being made, and nothing else identifies what was tested."
                )
            nid = spec.get("node_id")
            if nid is not None:
                if nid in seen_ids:
                    raise ValueError(
                        f"hypotheses[{index}] reuses node_id {nid!r}, already claimed earlier "
                        f"in this batch. Two hypotheses cannot share an id."
                    )
                seen_ids.add(nid)

        # Parents may live in the store already or be created by this same batch;
        # anything else is a dangling edge, and a node behind one can never be
        # selected because its parent gate is unsatisfiable forever.
        for index, spec in enumerate(hypotheses):
            for pid in spec.get("parent_ids") or []:
                if pid not in seen_ids and self._store.get_node(pid) is None:
                    raise ValueError(
                        f"hypotheses[{index}] names parent {pid!r}, which neither exists nor "
                        f"is created by this batch. Add it as an earlier entry, or drop it "
                        f"from parent_ids."
                    )

        # Collisions are checked up front for the same reason: under the default
        # policy one is fatal, and discovering it halfway through is what turns a
        # rejected batch into a partially-created graph.
        for index, spec in enumerate(hypotheses):
            policy = spec.get("if_exists", "error")
            if policy not in ("error", "overwrite", "skip"):
                raise ValueError(
                    f"hypotheses[{index}] has invalid if_exists={policy!r}; "
                    f"expected 'error', 'overwrite' or 'skip'."
                )
            nid = spec.get("node_id")
            if policy == "error" and nid is not None and self._store.get_node(nid) is not None:
                raise ValueError(
                    f"hypotheses[{index}]: node {nid!r} already exists. Use "
                    f"if_exists='skip' to keep the existing one or 'overwrite' to replace it."
                )

        results: dict[int, CreateHypothesisResult] = {}
        for position in self._creation_order(hypotheses, seen_ids):
            results[position] = self.create_hypothesis(**hypotheses[position])
        return [results[position] for position in range(len(hypotheses))]

    @staticmethod
    def _creation_order(hypotheses: list[dict[str, Any]], batch_ids: set[str]) -> list[int]:
        """Order batch positions so every parent is created before its children.

        List order is the obvious rule and the wrong one: it makes "declare the
        premise before the thing that assumes it" a property the caller has to
        remember, enforced by an exception after part of the batch is already
        written. The dependency is already stated in ``parent_ids``, so the order
        can simply be derived from it.

        Ties keep the caller's original order, so a batch that was already
        correctly sorted is applied exactly as written and the result is
        deterministic.
        """
        owner = {
            spec["node_id"]: position
            for position, spec in enumerate(hypotheses)
            if spec.get("node_id")
        }
        pending = {
            position: {
                owner[pid]
                for pid in (spec.get("parent_ids") or [])
                if pid in batch_ids and owner.get(pid) != position
            }
            for position, spec in enumerate(hypotheses)
        }

        ordered: list[int] = []
        while pending:
            ready = [position for position, deps in pending.items() if not deps]
            if not ready:
                raise CycleError(
                    f"parent references inside this batch form a cycle among positions "
                    f"{sorted(pending)}; no hypothesis can be created before the others."
                )
            for position in ready:
                del pending[position]
            ordered.extend(ready)
            for deps in pending.values():
                deps.difference_update(ready)
        return ordered

    def get_next_targets(
        self,
        count: int = 1,
        lease_ttl_s: int | None = None,
        dry_run: bool = False,
    ) -> list[TargetResponse]:
        """Reclaim stale leases, select up to ``count`` targets, issue claims.

        Batch-native with ``count=1`` as the ordinary case, rather than a
        single-target method plus a parallel batch one. The two would share every
        line of logic and differ only in arity, and the agent pays a full LLM
        round-trip per call — so dispatching one at a time is a pure tax on
        exactly the loop that runs most often.

        Targets are selected one at a time, so each pick sees the lease the
        previous one took and a node can never appear twice in a response. That
        is the same mechanism that keeps it out of the *next* call: a separate
        within-batch dedup alongside it would mask a regression in the lease
        exactly as it did before, when batches looked clean while consecutive
        batches handed out the same work over and over.

        A batch also never contains two competing answers to the same question.
        Members of an exclusion group are mutually exclusive, so confirming one
        settles the rest for free — dispatching several of them together spends
        probes on alternatives the very first result would have retired, and the
        tie-break that deliberately keeps *sequential* dispatch on one question
        made that the common case. Holding the question open until its answer
        comes back is what turns the exclusion inference from a bookkeeping
        detail into a saved probe.

        Selection stops early when the frontier is exhausted, so the caller may
        get fewer than ``count`` targets — including a single DONE sentinel when
        nothing at all is selectable.

        When ``dry_run=True``, runs selection but issues **no** claim and starts
        no TTL — a peek. A dry run returns at most one target because without a
        lease every pick would see an unchanged frontier and repeat itself.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if dry_run:
            count = 1

        now = utcnow()
        self._store.expire_stale_claims(now)

        responses: list[TargetResponse] = []
        claimed_groups: set[str] = set()
        for _ in range(count):
            response = self._select_one(now, lease_ttl_s, dry_run, claimed_groups)
            if response.status == "DONE":
                # Nothing left to hand out. Report it only if the batch is
                # otherwise empty — a partial batch is a success, not an end.
                if not responses:
                    responses.append(response)
                break
            responses.append(response)
            if response.node_id is not None:
                node = self._store.get_node(response.node_id)
                if node is not None and node.exclusion_group:
                    claimed_groups.add(node.exclusion_group)
        return responses

    def _select_one(
        self,
        now: datetime,
        lease_ttl_s: int | None,
        dry_run: bool,
        claimed_groups: set[str] | None = None,
        _retry: bool = True,
    ) -> TargetResponse:
        """Select and claim a single target. See get_next_targets."""
        self._sync_graph_from_store()

        frontier = self._frontier_nodes()
        # Never offer a second answer to a question that already has one in
        # flight — whether it was claimed a moment ago in this same batch or in
        # an earlier call whose result has not come back yet. The two are the
        # same mistake: confirming any member of an exclusion group settles the
        # rest for free, so a probe spent on a sibling while the first answer is
        # outstanding is a probe the first result was about to make unnecessary.
        # Scoping this to the batch alone was enough while dispatch and record
        # alternated in fixed pairs; once a record can carry its own dispatch,
        # the ordinary loop hands out one target per call and every such waste
        # falls in the gap between two calls rather than inside one.
        # Emptying the frontier here just ends the batch early: the caller
        # already holds work, and the resulting DONE says awaiting_evidence.
        blocked_groups = set(claimed_groups or ())
        for row in self._store.get_active_claims(now):
            leased = self._store.get_node(row["node_id"])
            if leased is not None and leased.exclusion_group:
                blocked_groups.add(leased.exclusion_group)
        if blocked_groups:
            frontier = [n for n in frontier if n.exclusion_group not in blocked_groups]
        all_nodes = self._store.get_all_nodes()
        goals = [n for n in all_nodes if n.is_goal]
        suspects = self._conflict_suspects()
        result = self._sampler.select(
            frontier,
            all_nodes,
            now=now,
            last_group=self._last_selected_group,
            priority_ids=set(suspects),
            all_goals_met=bool(goals) and all(self.goal_achieved(g) for g in goals),
        )

        if result.status == "DONE":
            # An empty frontier has several very different causes, and the
            # caller's next move differs completely between them. Collapsing any
            # pair of them has cost a real run: telling a batching agent its work
            # was finished the moment it got ahead of its own bookkeeping,
            # declaring a search over at exactly the point every question had
            # been answered and the answers had still to be put together, or
            # reporting a graph nothing could reach as a finished investigation.
            if result.reason == "empty_frontier":
                if self._store.get_active_claims(now):
                    return TargetResponse(
                        status="DONE",
                        reason="awaiting_evidence",
                        rationale=(
                            "every remaining hypothesis is already dispatched to you; "
                            "record evidence for what you have probed to get more"
                        ),
                    )
                substitution = self._next_substitution(dry_run)
                if substitution is not None:
                    return substitution
                confirmed = self._confirmed_for_composition()
                if confirmed:
                    goals = [n.id for n in self._store.get_all_nodes() if n.is_goal]
                    reach = (
                        f" Give the goal {goals[0]!r} that combination as a DEPENDENCY parent "
                        f"too, or it can never be reached: a goal is met when everything it "
                        f"depends on is verified."
                        if goals
                        else ""
                    )
                    return TargetResponse(
                        status="DONE",
                        reason="awaiting_composition",
                        rationale=(
                            "every open question is answered and nothing has been built "
                            "on the answers yet — create the hypothesis that combines "
                            "them, with parent_ids="
                            f"{confirmed} (DEPENDENCY), then test that. Without those "
                            f"parents a failure cannot be traced back to an assumption.{reach}"
                        ),
                    )
                blocked = self._blocked_nodes()
                if blocked:
                    shown = blocked[:3]
                    detail = "; ".join(f"{nid} needs {parents}" for nid, parents in shown)
                    more = f" (and {len(blocked) - len(shown)} more)" if len(blocked) > 3 else ""
                    return TargetResponse(
                        status="DONE",
                        reason="blocked_frontier",
                        rationale=(
                            f"{len(blocked)} hypothes(es) are untested but unreachable: "
                            f"{detail}{more}. A DEPENDENCY parent must be VERIFIED before "
                            "its child can be tested, so this usually means the edges run "
                            "the wrong way — a premise should be the parent of the "
                            "combination that assumes it, not its child. Nothing is "
                            "settled here; fix the wiring and the work becomes available."
                        ),
                    )
                # A composition that is conclusive but sub-par settles without
                # failing, so nothing is blamed and nothing is reopened. If every
                # question is answered, the frontier then empties with the
                # objective unreached — which is a finding, not an ending.
                # Gated on ``_retry`` so it is attempted at most once per
                # selection: reopening always puts premises (which have no
                # parents, so are always eligible) back on the frontier, and one
                # pass is therefore enough by construction.
                if _retry and not dry_run and self._recover_from_underperformance(now):
                    return self._select_one(now, lease_ttl_s, dry_run, claimed_groups, _retry=False)
                # Reported only once there is genuinely nothing else to do:
                # unwired goals are worth fixing, but handing the caller real
                # work is worth more, and a caller that cannot fix the wiring
                # would otherwise be told about it until its budget ran out.
                orphan_goals = self._unreachable_goals()
                if orphan_goals:
                    return TargetResponse(
                        status="DONE",
                        reason="unreachable_goal",
                        rationale=(
                            f"goal(s) {orphan_goals} have no DEPENDENCY parents, so nothing "
                            "can ever satisfy them: a goal is reached when every hypothesis "
                            "it depends on is verified, and one that depends on nothing "
                            "depends on nothing. Wire the combination you are trying to "
                            "reach it with as a parent of the goal (parent_ids on the goal, "
                            "or re-create it with them). Nothing is settled here."
                        ),
                    )
                # Giving up on a targeted narrowing reopens the alternatives its
                # members had retired, which happens *after* this pass computed
                # its frontier. Reporting the stale answer would announce the end
                # of the search at the exact moment new work appeared — and the
                # caller, quite reasonably, would stop.
                if _retry and self._frontier_nodes():
                    return self._select_one(now, lease_ttl_s, dry_run, claimed_groups, _retry=False)
            return TargetResponse(status="DONE", reason=result.reason)

        assert result.node_id is not None
        assert result.claim_id is not None

        node = self._store.get_node(result.node_id)
        assert node is not None

        # A node under conflict review must be re-tested at least as deeply as
        # the failure that implicated it; a repeat of the same shallow test would
        # return the same answer and settle nothing.
        min_depth = suspects.get(result.node_id)
        rationale = result.rationale
        if min_depth is not None:
            rationale += f"; implicated in an unresolved conflict — re-test at depth >= {min_depth}"
            if not dry_run:
                self._review_dispatches[result.node_id] += 1

        if dry_run:
            return TargetResponse(
                status="SELECTED",
                node_id=result.node_id,
                statement=node.statement,
                rationale=rationale,
                credible_interval=result.credible_interval,
                claim_id=None,
                min_depth=min_depth,
            )

        ttl = lease_ttl_s if lease_ttl_s is not None else self._lease_ttl_s
        self._store.create_claim(result.claim_id, result.node_id, now, ttl)
        if node.status == Status.UNTESTED:
            self._store.change_status(
                result.node_id, Status.IN_PROGRESS, reason="dispatched", now=now
            )

        # Remember which question this answered, so the next tie-break can keep
        # working the same one instead of hopping to an unrelated node.
        # A dry run must not move this cursor: it is a peek, not a dispatch.
        self._last_selected_group = node.exclusion_group
        self._refresh_node_in_graph(result.node_id)

        return TargetResponse(
            status="SELECTED",
            node_id=result.node_id,
            statement=node.statement,
            rationale=rationale,
            credible_interval=result.credible_interval,
            claim_id=result.claim_id,
            min_depth=min_depth,
        )

    def record_evidence(
        self,
        node_id: str,
        evidence: Evidence,
        claim_id: str | None = None,
        *,
        count_next_targets: int = 0,
        lease_ttl_s: int | None = None,
    ) -> RecordEvidenceResult:
        """Validate + consume claim, update posterior, apply transitions.

        Auto-captures git context_hash + git_branch from the working tree when
        the evidence does not carry them (best-effort; None outside a git repo).

        Evidence against a goal is refused — see ``GoalEvidenceError``.

        ``count_next_targets`` fuses the next dispatch into this call, and
        defaults to **0** — no dispatch. Recording what happened and asking what
        to do next are genuinely separate questions: an experiment that runs for
        hours or days is reported by someone who is not, in that moment, asking
        to be handed the next one, and pushing work at them would either strand a
        lease or force them to release it. A caller that *is* a synchronous loop
        pays a full model round-trip for each of the two calls, so asking for
        both at once roughly halves its turn count — which is why this is an
        accelerator the caller opts into rather than the shape of the API.

        It is a **top-up**, not an addition: the number is how many targets the
        caller wants to be holding when this returns, so a caller that records
        two results in a batch ends the batch holding two, not four. Adding to
        the outstanding set instead would hand work out faster than it is
        reported, which is precisely the waste the fusion is meant to remove —
        the accelerator would have paid for itself in round-trips and then spent
        the saving on stranded leases.
        """
        if count_next_targets < 0:
            raise ValueError(f"count_next_targets must be >= 0, got {count_next_targets}")
        # A blank claim id is a caller reaching for the field it was told to omit,
        # not a claim. Rejecting it threw away a probe that had already been paid
        # for, over punctuation.
        if claim_id is not None and not claim_id.strip():
            claim_id = None
        node = self._store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node not found: {node_id}")

        # A goal is an objective, not a claim, and the engine already refuses to
        # invalidate or exhaust one — so the only thing evidence against a goal
        # can do is push it toward "achieved". Accepting it buys a one-way
        # ratchet, and worse, silently destroys the result: what was actually
        # probed was some hypothesis, and filing its outcome here leaves that
        # hypothesis untested while the failure it revealed explains nothing.
        # Whether a goal is reached is derived from what supports it instead.
        if node.is_goal:
            raise GoalEvidenceError(
                f"'{node_id}' is a goal: it states an objective, so there is no result "
                f"to record against it and it is never handed out as a target. Record "
                f"this against the hypothesis you actually probed — create one for that "
                f"configuration if it does not exist, with parent_ids naming the "
                f"confirmed hypotheses it assumes — and the goal follows once those are "
                f"verified."
            )

        # Best-effort git context capture for the staleness flag. Only fills
        # when the evidence doesn't already carry explicit values.
        if isinstance(evidence, LogicalEvidence) and (
            evidence.context_hash is None or evidence.git_branch is None
        ):
            ctx_hash, branch = capture_git_context(self._project_path)
            if evidence.context_hash is None:
                evidence.context_hash = ctx_hash
            if evidence.git_branch is None:
                evidence.git_branch = branch

        now = utcnow()
        prior_rows = self._store.get_evidence_for_node(node_id)
        # Only logical observations count toward the convergence gate — infra
        # errors are operational noise, not samples of the hypothesis.
        logical_count = sum(1 for r in prior_rows if r["kind"] == "logical")

        if claim_id is not None:
            claim = self._store.get_claim(claim_id)
            if claim is None:
                # An id nobody ever issued is a caller mistyping or inventing
                # one, not a caller reporting on someone else's work. If the
                # node it names is under a live lease, that lease is
                # unambiguously the one being answered, so the result is kept
                # and the lease consumed. Rejecting instead destroyed a probe
                # the environment had already been paid for, over one wrong
                # character — and left the lease live, so the node was then
                # counted as dispatched-and-never-reported on top.
                #
                # With no lease at all the id names nothing and reserves
                # nothing: there is no dispatch it could be answering and
                # nothing to consume, so it is simply dropped and the result is
                # recorded as the self-initiated probe it is. Refusing there
                # destroyed two more probes over a fabricated placeholder.
                claim = self._recoverable_claim(node_id, now)
                claim_id = str(claim["claim_id"]) if claim is not None else None
            elif claim["node_id"] != node_id:
                raise ClaimError(
                    f"Claim {claim_id} belongs to a different hypothesis than {node_id}. "
                    f"Record against the node the claim was issued for, or omit claim_id "
                    f"if this result came from a probe you chose yourself."
                )
            if claim_id is not None and not self._store.consume_claim(claim_id, consumed_at=now):
                raise ClaimError(
                    f"Claim {claim_id} is no longer valid — it was already "
                    f"consumed, expired, or superseded by a later dispatch"
                )

        evidence = self._compute_deltas(evidence, prior_rows)
        self._store.append_evidence(node_id, evidence, recorded_at=now)

        last_success: float | None = None
        if isinstance(evidence, LogicalEvidence):
            new_alpha = node.alpha + evidence.success
            new_beta = node.beta + (1.0 - evidence.success)
            self._store.update_posterior(node_id, new_alpha, new_beta)
            self._store.increment_evidence_count(node_id)
            last_success = evidence.success

        node = self._store.get_node(node_id)
        assert node is not None

        if isinstance(evidence, InfraError):
            self._handle_infra_error(node, now)
        elif last_success is not None:
            depth = evidence.depth if isinstance(evidence, LogicalEvidence) else 0
            new_logical_count = logical_count + 1
            if self._sampler.should_invalidate(node, new_logical_count, last_success):
                self._invalidate_node(node.id, now, depth)
            elif self._sampler.should_verify(node, new_logical_count, last_success):
                self._verify_node(node.id, now, depth)
            elif self._sampler.should_exhaust(node, new_logical_count, last_success):
                # Conclusive but sub-bar: settle the node so it leaves the
                # frontier instead of being re-dispatched forever.
                self._exhaust_node(node.id, now, last_success, depth)

        self._refresh_node_in_graph(node_id)
        updated = self._store.get_node(node_id)
        assert updated is not None
        # Top up to the requested number in flight, counting what the caller
        # already holds.
        shortfall = count_next_targets - len(self._store.get_active_claims(utcnow()))
        next_targets = (
            self.get_next_targets(count=shortfall, lease_ttl_s=lease_ttl_s) if shortfall > 0 else []
        )
        return RecordEvidenceResult(node=updated, next_targets=next_targets)

    def _recoverable_claim(self, node_id: str, now: datetime) -> Any | None:
        """The lease an unknown claim id was almost certainly meant to name.

        Only ever the node's *own* live lease — so this recovers a typo without
        ever letting a caller report against work it does not hold. A lease on a
        different node still fails loudly, which is the check that actually
        protects the belief state. ``create_claim`` supersedes any lease the node
        already held, so there is never more than one to choose between.
        """
        live = [c for c in self._store.get_active_claims(now) if c["node_id"] == node_id]
        return live[0] if live else None

    def update_status(
        self, node_id: str, new_status: Status, reason: str = ""
    ) -> UpdateStatusResult:
        """Manual status override. Returns prior status + transition string."""
        node = self._store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node not found: {node_id}")
        old_status = node.status
        self._store.change_status(node_id, new_status, reason=reason)
        self._refresh_node_in_graph(node_id)
        self._sync_exclusion(node_id, old_status, new_status, utcnow())
        updated = self._store.get_node(node_id)
        assert updated is not None
        return UpdateStatusResult(
            node=updated,
            old_status=old_status,
            transition=f"{old_status.value} → {new_status.value}",
        )

    def invalidate_upstream(self, leaf_id: str) -> list[str]:
        """Walk DEPENDENCY ancestors, flip VERIFIED → NEEDS_REVISION.

        An ancestor losing VERIFIED also loses the authority behind any exclusion
        inference it justified, so those siblings are reopened in the same pass.
        Otherwise a retracted confirmation would leave its competing alternatives
        permanently settled and the correct one could never be reconsidered.
        """
        affected = self._graph.upstream_invalidate(leaf_id)
        now = utcnow()
        for nid in affected:
            self._store.change_status(
                nid, Status.NEEDS_REVISION, reason="upstream invalidation", now=now
            )
            self._refresh_node_in_graph(nid)
            self._sync_exclusion(nid, Status.VERIFIED, Status.NEEDS_REVISION, now)
        return affected

    def verify_upstream(self, child_id: str) -> list[str]:
        """Walk REFINEMENT ancestors, flip IN_PROGRESS → VERIFIED (depth-capped)."""
        affected = self._graph.upstream_verify(child_id)
        now = utcnow()
        for nid in affected:
            self._store.change_status(
                nid, Status.VERIFIED, reason="refinement child verified", now=now
            )
            self._refresh_node_in_graph(nid)
            # A node promoted from below is as confirmed as one probed directly,
            # so it settles its competing alternatives the same way.
            self._sync_exclusion(nid, Status.IN_PROGRESS, Status.VERIFIED, now)
        return affected

    def get_conflicts(self, open_only: bool = True) -> list[dict[str, Any]]:
        """Recorded conflict sets: groups of assumptions that cannot all hold.

        Each entry carries its member statements, the depth at which the
        composition failed, which members have been cleared, and the swap that
        would clear the next one. This is the belief state's record of *what has
        been ruled out as a combination*, which is strictly more information than
        any per-node status can express.

        ``resolve_by`` deliberately never suggests re-testing a member on its
        own. Each has already passed that test — that is what makes the conflict
        indeterminate — so the only experiment that moves it is another
        composition with one assumption swapped out.
        """
        exonerated = self._exonerated_nodes()
        out: list[dict[str, Any]] = []
        for nogood in self._store.get_nogoods(open_only=open_only):
            members = nogood["member_ids"]
            suspects = self._remaining_suspects(nogood, exonerated)
            cleared = set(members[: int(nogood["probe_index"] or 0)])
            plan = self._substitution_plan(nogood) if not nogood["resolved_at"] else None
            out.append(
                {
                    **nogood,
                    "members": [
                        {
                            "node_id": mid,
                            "statement": (n.statement if (n := self._store.get_node(mid)) else ""),
                            "confirmed_depth": (n.confirmed_depth if n else None),
                            "exonerated": mid not in suspects,
                            "cleared_by_substitution": mid in cleared,
                        }
                        for mid in members
                    ],
                    "remaining_suspects": suspects,
                    "resolve_by": (
                        f"rebuild this combination with '{plan['member_id']}' replaced by "
                        f"'{plan['candidate_id']}' and probe at depth >= {plan['min_depth']}"
                        if plan
                        else "every assumption has been swapped out and it still failed — "
                        "this is an interaction effect and the alternatives are reopened"
                    ),
                }
            )
        return out

    def suggest_discriminating_experiment(self) -> dict[str, Any]:
        """Propose the single most informative next experiment.

        Answers the question a bare status list cannot: *"given that these
        combinations are ruled out, what should I do next?"* Two kinds of answer,
        in strict order of information gain:

        1. **Substitute one assumption.** While a conflict is still being
           narrowed, rebuild the failing combination with exactly one member
           swapped for a competing answer. One probe eliminates an entire
           question: if the composition still fails, the swapped assumption was
           not the cause; if it stops failing, it was. Sweeping that question's
           alternatives one at a time costs a probe per alternative and re-testing
           the assumption alone costs a probe for no information at all, because
           it already passed exactly that test.
        2. **Recombine.** Once every assumption has been swapped out and the
           composition failed every time, the failure is a genuine interaction:
           pick one live member per involved question, skip anything a recorded
           conflict already rules out, and choose the assignment closest to the
           last failure so the next result stays interpretable.

        Returns ``{"status": "NO_CONFLICTS" | "EXHAUSTED" | "SUGGESTED", ...}``.
        """
        conflicts = self._store.get_nogoods(open_only=True)
        if not conflicts:
            return {"status": "NO_CONFLICTS", "reason": "no unresolved conflict sets"}

        latest = conflicts[0]
        nogood_sets = [set(c["member_ids"]) for c in conflicts]

        # 1. A single-assumption swap beats a fresh combination while one is left
        #    to try: it eliminates a whole question rather than a single point.
        for nogood in self._diagnosing_nogoods():
            plan = self._substitution_plan(nogood)
            if plan is None:
                continue
            member = self._store.get_node(plan["member_id"])
            candidate = self._store.get_node(plan["candidate_id"])
            return {
                "status": "SUGGESTED",
                "action": "substitute",
                "node_id": plan["member_id"],
                "statement": member.statement if member else "",
                "replace_with": plan["candidate_id"],
                "replacement_statement": candidate.statement if candidate else "",
                "parent_ids": sorted([*plan["keep_ids"], plan["candidate_id"]]),
                "min_depth": plan["min_depth"],
                "remaining_suspects": plan["remaining"],
                "rationale": (
                    f"rebuild the failed combination with '{plan['member_id']}' replaced "
                    f"by '{plan['candidate_id']}' and probe at depth >= "
                    f"{plan['min_depth']}. Still failing clears '{plan['member_id']}'; "
                    f"no longer failing convicts it. {plan['remaining']} assumption(s) "
                    f"left to eliminate. Do not re-test it on its own — it already "
                    f"passed that test, which is why this conflict is unresolved."
                ),
            }

        # 2. No suspect left: vary the combination itself.
        groups: dict[str, list[str]] = {}
        failed: dict[str, str] = {}
        for mid in latest["member_ids"]:
            node = self._store.get_node(mid)
            if node is None or not node.exclusion_group:
                continue
            failed[node.exclusion_group] = mid
            live = [
                sibling.id
                for sibling in self._store.get_nodes_in_exclusion_group(node.exclusion_group)
                if sibling.status not in (Status.INVALIDATED, Status.PRUNED)
            ]
            groups[node.exclusion_group] = sorted(live)

        if not groups:
            return {
                "status": "EXHAUSTED",
                "reason": "conflict members carry no exclusion groups to vary",
            }

        keys = sorted(groups)
        space = 1
        for k in keys:
            space *= len(groups[k])
        if space > _MAX_SUGGESTION_SPACE:
            # Enumerating every assignment is only sane while the space is small.
            # Reporting the refusal beats silently truncating the search and
            # returning a "closest" candidate that was never actually the closest.
            return {
                "status": "EXHAUSTED",
                "reason": (
                    f"combination space is too large to enumerate "
                    f"({space} assignments over {len(keys)} questions)"
                ),
            }

        # Members of a conflict that are not being varied here stay implicitly
        # asserted in every candidate, so a conflict rules out a candidate as soon
        # as its *varied* members all appear. Without this projection, any conflict
        # containing an ungrouped assumption could never rule anything out.
        universe = {nid for k in keys for nid in groups[k]}
        projected = [known & universe for known in nogood_sets]
        projected = [known for known in projected if known]

        best: tuple[int, list[str]] | None = None
        for combo in product(*(groups[k] for k in keys)):
            candidate = set(combo)
            # Never re-propose a combination already known to be impossible.
            if any(known <= candidate for known in projected):
                continue
            distance = sum(
                1 for k, choice in zip(keys, combo, strict=True) if failed.get(k) != choice
            )
            if best is None or distance < best[0]:
                best = (distance, list(combo))
                if distance <= 1:
                    break  # cannot do better than a single-axis swap

        if best is None:
            return {
                "status": "EXHAUSTED",
                "reason": "every remaining combination is already ruled out",
            }

        distance, chosen = best
        return {
            "status": "SUGGESTED",
            "action": "recombine",
            "changed_assumptions": distance,
            "assignment": [
                {
                    "node_id": nid,
                    "statement": (n.statement if (n := self._store.get_node(nid)) else ""),
                }
                for nid in chosen
            ],
            "rationale": (
                f"differs from the last refuted combination in {distance} assumption(s) "
                f"and is not covered by any of the {len(nogood_sets)} recorded conflict(s)"
            ),
        }

    def get_goal_status(self) -> GoalStatusResponse:
        """Report all goal nodes, their bars, and whether the global stop holds.

        Enriched with counts (goals_met / total), frontier_size, total_nodes,
        and a status_breakdown for at-a-glance progress.
        """
        # Sync first: goal achievement is read off the DEPENDENCY edges, so a
        # stale graph would report on a structure the store has already moved on
        # from — including, on a freshly reloaded engine, no edges at all.
        self._sync_graph_from_store()

        goals = [n for n in self._store.get_all_nodes() if n.is_goal]
        entries: list[GoalStatusEntry] = []
        for g in goals:
            mean = posterior_mean(g.alpha, g.beta)
            met = self.goal_achieved(g)
            entries.append(
                GoalStatusEntry(
                    node_id=g.id,
                    statement=g.statement,
                    target_metric=g.target_metric,
                    posterior_mean=mean,
                    status=g.status,
                    met=met,
                )
            )
        all_met = bool(entries) and all(e.met for e in entries)

        # Enrich: counts, frontier size, total nodes, status breakdown
        frontier_size = len(self._frontier_nodes())
        total_nodes = self._store.count_all_nodes()
        status_breakdown = self._store.count_nodes_by_status()

        return GoalStatusResponse(
            goals=entries,
            all_met=all_met,
            goals_met_count=sum(1 for e in entries if e.met),
            goals_total_count=len(entries),
            frontier_size=frontier_size,
            total_nodes=total_nodes,
            status_breakdown=status_breakdown,
        )

    def get_dag_context(
        self,
        node_id: str | None = None,
        max_depth: int = 2,
        max_children: int = 10,
    ) -> DagContextResponse:
        """Return a depth+width-bounded subgraph with credible intervals."""
        if node_id is None:
            roots = [n for n in self._store.get_all_nodes() if not n.parent_ids]
            if not roots:
                return DagContextResponse(
                    nodes=[], elisions=[], max_depth=max_depth, max_children=max_children
                )
            node_id = roots[0].id

        node = self._store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node not found: {node_id}")

        nodes_out: list[DagContextNode] = []
        elisions: list[ElisionNode] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(node_id, 0)]

        while queue:
            nid, depth = queue.pop(0)
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)

            n = self._store.get_node(nid)
            if n is None:
                continue

            ci = credible_interval(n.alpha, n.beta)
            nodes_out.append(
                DagContextNode(
                    id=n.id,
                    statement=n.statement,
                    status=n.status,
                    posterior_mean=posterior_mean(n.alpha, n.beta),
                    credible_interval=ci,
                    parent_ids=n.parent_ids,
                    is_goal=n.is_goal,
                    target_metric=n.target_metric,
                )
            )

            if depth < max_depth:
                child_ids = self._graph.children(nid)
                child_nodes = {cid: self._store.get_node(cid) for cid in child_ids}
                children = sorted(
                    child_ids,
                    key=lambda cid: (
                        posterior_mean(child_nodes[cid].alpha, child_nodes[cid].beta)
                        if child_nodes[cid] is not None
                        else 0.0
                    ),
                    reverse=True,
                )
                for cid in children[:max_children]:
                    queue.append((cid, depth + 1))
                if len(children) > max_children:
                    elisions.append(
                        ElisionNode(
                            parent_id=nid,
                            hidden_count=len(children) - max_children,
                        )
                    )

        return DagContextResponse(
            nodes=nodes_out,
            elisions=elisions,
            max_depth=max_depth,
            max_children=max_children,
        )

    # Mermaid link styles per edge type — makes AND vs OR vs loose-coupling
    # legible at a glance.
    _EDGE_MERMAID: dict[str, str] = {
        "DEPENDENCY": "-->",
        "ALTERNATIVE": "-.->",
        "REFINEMENT": "==>",
    }

    def render_dag_map(
        self,
        node_id: str | None = None,
        max_depth: int = 2,
        max_children: int = 10,
        hide_statuses: list[str] | None = None,
    ) -> str:
        """Render a Mermaid flowchart with depth+width bounding + elision.

        Edge-type styling: DEPENDENCY solid ``-->``, ALTERNATIVE dashed
        ``-.->``, REFINEMENT thick ``==>``. ``hide_statuses`` drops matching
        nodes from the output (e.g. ``["PRUNED"]``).
        """
        ctx = self.get_dag_context(node_id, max_depth, max_children)
        lines = ["graph TD"]

        # Build a set of visible node IDs, optionally filtering by status.
        hide = set(hide_statuses) if hide_statuses else set()
        visible_ids = {n.id for n in ctx.nodes if n.status.value not in hide}

        for node in ctx.nodes:
            if node.id not in visible_ids:
                continue
            safe_id = node.id.replace("-", "_")
            label = f"{node.id} [{node.status.value}] μ={node.posterior_mean:.2f}"
            if node.is_goal:
                label += " 🎯"
            lines.append(f'    {safe_id}["{label}"]')

        # Build a lookup of all edges to determine edge types for rendering.
        all_edges: dict[str, list[Edge]] = {}
        for e in self._store.get_all_edges():
            all_edges.setdefault(e.src, []).append(e)

        for node in ctx.nodes:
            if node.id not in visible_ids:
                continue
            safe_id = node.id.replace("-", "_")
            for pid in node.parent_ids:
                if pid in visible_ids:
                    safe_pid = pid.replace("-", "_")
                    # Look up the edge type to pick the right Mermaid link style.
                    link = "-->"  # default fallback
                    for e in all_edges.get(pid, []):
                        if e.dst == node.id:
                            link = self._EDGE_MERMAID.get(e.type.value, "-->")
                            break
                    lines.append(f"    {safe_pid} {link} {safe_id}")

        for el in ctx.elisions:
            safe_pid = el.parent_id.replace("-", "_")
            elision_id = f"{safe_pid}_elision"
            lines.append(f'    {elision_id}(+"{el.hidden_count} more...")')
            lines.append(f"    {safe_pid} --> {elision_id}")

        return "\n".join(lines)

    def list_nodes(
        self,
        status_filter: list[str] | None = None,
        query_filter: str | None = None,
        order_by: str = "created_at",
        ascending: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> str:
        """Query nodes with filter/sort/pagination and return a Markdown table.

        query_filter: case-insensitive statement match. `*` → `%`, `_` stays
        `_` (SQL LIKE). Literal `%`/`_` are escaped.
        """
        translated_query: str | None = None
        if query_filter:
            # Respect explicit `*` wildcards verbatim; otherwise treat the query
            # as a substring match (wrap in %…%). Escaping + the ESCAPE clause in
            # the store make literal `%`/`_` match themselves.
            has_wildcard = "*" in query_filter
            translated_query = _translate_like_wildcards(query_filter)
            if not has_wildcard:
                translated_query = f"%{translated_query}%"
        rows = self._store.list_nodes(
            status_filter=status_filter,
            query_filter=translated_query,
            order_by=order_by,
            ascending=ascending,
            limit=limit,
            offset=offset,
        )

        summaries: list[NodeSummary] = []
        for r in rows:
            alpha: float = r["alpha"]
            beta: float = r["beta"]
            summaries.append(
                NodeSummary(
                    id=r["id"],
                    statement=r["statement"],
                    status=Status(r["status"]),
                    posterior_mean=posterior_mean(alpha, beta),
                    is_goal=bool(r["is_goal"]),
                    evidence_count=r["evidence_count"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    verified_at=datetime.fromisoformat(r["verified_at"])
                    if r["verified_at"]
                    else None,
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
            )

        # Render as a Markdown table for compact, human-readable output.
        header = "| ID | Statement | Status | μ | Goal | Ev# | Created | Verified |"
        separator = "|----|-----------|--------|---|------|-----|---------|----------|"
        lines = [header, separator]
        for s in summaries:
            stmt = s.statement[:60].replace("|", "\\|")
            if len(s.statement) > 60:
                stmt += "…"
            created = s.created_at.strftime("%Y-%m-%d %H:%M")
            verified = s.verified_at.strftime("%Y-%m-%d %H:%M") if s.verified_at else "—"
            lines.append(
                f"| {s.id} | {stmt} | {s.status.value} | {s.posterior_mean:.2f} | "
                f"{'🎯' if s.is_goal else ''} | {s.evidence_count} | {created} | {verified} |"
            )
        return "\n".join(lines)

    def get_evidence_history(
        self,
        node_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EvidenceSummary]:
        """Return newest-first evidence rows for a node."""
        node = self._store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node not found: {node_id}")

        rows = self._store.get_evidence_paginated(node_id, limit=limit, offset=offset)

        return [
            EvidenceSummary(
                id=r["id"],
                kind=r["kind"],
                success=r["success"],
                metrics=json.loads(r["metrics"]),
                delta_success=r["delta_success"],
                monotonicity=r["monotonicity"],
                context_hash=r["context_hash"],
                git_branch=r["git_branch"],
                notes=r["notes"],
                recorded_at=datetime.fromisoformat(r["recorded_at"]),
            )
            for r in rows
        ]

    def get_active_claims(self) -> list[ActiveClaimSummary]:
        """Return live (unconsumed, unexpired) claims for resuming work."""
        now = utcnow()
        self._store.expire_stale_claims(now)
        rows = self._store.get_active_claims(now)
        summaries: list[ActiveClaimSummary] = []
        for r in rows:
            claimed_at = datetime.fromisoformat(r["claimed_at"])
            elapsed = (now - claimed_at).total_seconds()
            ttl: int = r["lease_ttl_s"]
            remaining = max(0, int(ttl - elapsed))
            summaries.append(
                ActiveClaimSummary(
                    node_id=r["node_id"],
                    claim_id=r["claim_id"],
                    claimed_at=claimed_at,
                    expires_in_s=remaining,
                )
            )
        return summaries

    def generate_learning_path(self, limit: int = 200) -> LearningPathResponse:
        """Narrate what has been settled so far, in order, and how each was settled.

        The other read tools answer *what the belief state currently holds*:
        `get_dag_context` returns a subgraph, `render_dag_map` draws it,
        `list_nodes` filters it. None of them answers the question a human or a
        returning agent actually asks first — **what did we learn, and what did
        it cost?** That is a property of the transition history, not of the
        current state, and it is destroyed by every snapshot view.

        The distinction that carries the report is *observed* versus *inferred*.
        A conclusion the caller paid an experiment for and a conclusion the
        engine derived from other results look identical in a status column and
        are worth entirely different things: the second is free. Reporting them
        together, with the count of conclusions that cost no probe, is the only
        honest way to state what externalising the belief state bought — and the
        only way an agent resuming after a context reset can tell which of its
        beliefs still rest on something it can point at.

        Reversals are called out separately rather than folded into the current
        status. A hypothesis that was confirmed, withdrawn under conflict and
        then re-confirmed is the single most useful thing in the history and the
        one a snapshot cannot show at all.

        ``limit`` bounds the narrative to the most recent transitions so a
        long-running workspace still returns something an agent can read; the
        counters are computed over the whole history regardless.
        """
        self._sync_graph_from_store()

        nodes = {n.id: n for n in self._store.get_all_nodes()}
        # Evidence counts are the authoritative "did this cost an experiment"
        # signal: a node can be VERIFIED with zero observations of its own, which
        # is exactly the case worth reporting.
        probed = {nid for nid, n in nodes.items() if n.evidence_count > 0}

        steps: list[LearningStep] = []
        for row in self._store.get_all_status_history():
            node = nodes.get(row["node_id"])
            if node is None:
                continue
            status = Status(row["status"])
            # UNTESTED is the state every node starts in, not a conclusion. It is
            # only interesting when something *returned* a node to it, which the
            # two reopen markers identify — and those are the most valuable
            # entries in the report, because no snapshot of the current state can
            # show that a belief was ever held and then withdrawn.
            reason = str(row["reason"] or "")
            reopened = reason.startswith(INTERACTION_REOPEN_PREFIX) or reason.startswith(
                EXCLUSION_RETRACT_PREFIX
            )
            if status in (Status.UNTESTED, Status.IN_PROGRESS) and not reopened:
                continue
            origin, how = _settlement_origin(reason)
            if status == Status.NEEDS_REVISION:
                origin = "reversed"
            steps.append(
                LearningStep(
                    node_id=node.id,
                    statement=node.statement,
                    status=status,
                    at=datetime.fromisoformat(row["valid_from"]),
                    origin=origin,
                    how=how,
                    cost_a_probe=node.id in probed,
                )
            )

        goal_status = self.get_goal_status()
        conflicts = self.get_conflicts(open_only=True)
        open_questions = [
            n.statement for n in nodes.values() if n.status in _OPEN_STATUSES and not n.is_goal
        ]
        # A conclusion is a node that ended up settled, counted once — not once
        # per transition, or a node that was revised and re-confirmed would look
        # like two discoveries.
        settled = {s.node_id for s in steps if s.status not in _OPEN_STATUSES}
        free = {nid for nid in settled if nid not in probed}
        probes = sum(n.evidence_count for n in nodes.values())

        return LearningPathResponse(
            markdown=self._render_learning_path(
                steps[-limit:], goal_status, conflicts, open_questions, probes, settled, free
            ),
            steps=steps[-limit:],
            probes_spent=probes,
            conclusions=len(settled),
            conclusions_without_a_probe=len(free),
            open_questions=open_questions,
            open_conflicts=len(conflicts),
            goals_met=goal_status.goals_met_count,
            goals_total=goal_status.goals_total_count,
        )

    def _render_learning_path(
        self,
        steps: list[LearningStep],
        goal_status: GoalStatusResponse,
        conflicts: list[dict[str, Any]],
        open_questions: list[str],
        probes: int,
        settled: set[str],
        free: set[str],
    ) -> str:
        """Render the learning path as a self-contained markdown briefing."""
        out: list[str] = ["# What we have learned so far", ""]

        if goal_status.goals_total_count:
            out.append(
                f"**Objective:** {goal_status.goals_met_count}/{goal_status.goals_total_count} "
                f"goal(s) reached."
            )
            for goal in goal_status.goals:
                mark = "reached" if goal.met else "not yet reached"
                out.append(f"- {goal.statement} — *{mark}*")
        else:
            out.append("**Objective:** none declared. Create a goal node so progress is derivable.")
        out.append("")

        established = [s for s in steps if s.status == Status.VERIFIED]
        ruled_out = [
            s for s in steps if s.status in (Status.INVALIDATED, Status.EXHAUSTED, Status.PRUNED)
        ]
        reversals = [s for s in steps if s.origin == "reversed"]

        if established:
            out += ["## Established", ""]
            out += [
                f"- **{s.statement}** — {s.how}"
                + ("" if s.cost_a_probe else " *(no experiment spent)*")
                for s in established
            ]
            out.append("")

        if ruled_out:
            out += ["## Ruled out", ""]
            out += [
                f"- {s.statement} — {s.how}"
                + ("" if s.cost_a_probe else " *(no experiment spent)*")
                for s in ruled_out
            ]
            out.append("")

        if reversals:
            out += [
                "## Where we changed our minds",
                "",
                "These are beliefs the engine withdrew or handed back after something "
                "built on them failed. A flat log of results cannot produce this section.",
                "",
            ]
            out += [f"- {s.statement} — {s.how}" for s in reversals]
            out.append("")

        if conflicts:
            out += [
                "## Open conflicts",
                "",
                "A set of assumptions that cannot all hold, with none individually "
                "refuted. Narrow one by rebuilding the failing combination with a "
                "single member swapped — `suggest_discriminating_experiment` names which.",
                "",
            ]
            out += [
                f"- failure at `{c['source_node_id']}` over "
                f"{', '.join(f'`{m}`' for m in c['member_ids'])}"
                for c in conflicts
            ]
            out.append("")

        if open_questions:
            out += ["## Still open", ""]
            out += [f"- {q}" for q in open_questions[:20]]
            if len(open_questions) > 20:
                out.append(f"- …and {len(open_questions) - 20} more")
            out.append("")

        out += [
            "## What it cost",
            "",
            f"- Experiments recorded: **{probes}**",
            f"- Questions settled: **{len(settled)}**",
            f"- Settled without running an experiment: **{len(free)}** "
            f"— deduced by elimination, retired by a confirmed rival, or ruled out by a swap",
            "",
            "Call `get_next_targets` for the next hypothesis to test.",
        ]
        return "\n".join(out)

    def bulk_update_status(
        self, node_ids: list[str], new_status: Status, reason: str = ""
    ) -> list[UpdateStatusResult]:
        """Update many nodes' status in one call. Returns per-node results.

        All ids are validated to exist up front so a single bad id never leaves a
        partial update; transitions are then applied in list order.
        """
        missing = [nid for nid in node_ids if self._store.get_node(nid) is None]
        if missing:
            raise NodeNotFoundError(f"nodes not found: {missing}")
        results: list[UpdateStatusResult] = []
        for nid in node_ids:
            result = self.update_status(nid, new_status, reason=reason)
            results.append(result)
        return results

    def _compute_deltas(self, evidence: Evidence, prior_rows: list[Any]) -> Evidence:
        """Compute delta_success + monotonicity relative to the previous evidence.

        The baseline is the most recent *logical* observation; infra errors carry
        no success value and are skipped so a mid-run infra failure never masks a
        node's true first measurement.
        """
        if not isinstance(evidence, LogicalEvidence):
            return evidence

        prior_logical = [
            r for r in prior_rows if r["kind"] == "logical" and r["success"] is not None
        ]
        if not prior_logical:
            evidence.delta_success = 0.0
            evidence.monotonicity = "first"
            return evidence

        prev_success = prior_logical[-1]["success"]
        evidence.delta_success = evidence.success - prev_success
        if evidence.delta_success > 0:
            evidence.monotonicity = "up"
        elif evidence.delta_success < 0:
            evidence.monotonicity = "down"
        else:
            evidence.monotonicity = "flat"

        return evidence

    def _handle_infra_error(self, node: Node, now: datetime) -> None:
        """Increment infra retry count; auto-BLOCKED after MAX_INFRA_RETRIES."""
        node.infra_retry_count += 1
        node.updated_at = now
        self._store.save_node(node)
        if node.infra_retry_count >= MAX_INFRA_RETRIES:
            self._store.change_status(
                node.id, Status.BLOCKED, reason="max infra retries exceeded", now=now
            )

    def _invalidate_node(self, node_id: str, now: datetime, depth: int = 0) -> None:
        """Transition to INVALIDATED + cascade prune + upstream blame + deduction."""
        prior = self._store.get_node(node_id)
        group = prior.exclusion_group if prior is not None else None

        self._store.change_status(
            node_id,
            Status.INVALIDATED,
            reason="evidence triggered invalidation",
            now=now,
        )
        # A refutation withdraws the confirmation entirely, depth included.
        self._store.set_confirmed_depth(node_id, None)
        self._refresh_node_in_graph(node_id)

        self._cascade_prune(node_id, now)

        # Withdraw any exclusion this node justified. Keyed on the siblings' own
        # history marker rather than on this node's prior status, because a node
        # placed under conflict review still holds the authority it acquired when
        # it was confirmed — gating on "was VERIFIED a moment ago" would leave its
        # alternatives buried exactly when they became the way forward.
        self._retract_exclusion(node_id, now)

        # A composition that fails while standing in for a conflict member is the
        # diagnostic swap: it clears that member rather than accusing it, and the
        # ordinary blame path is skipped so the predicted failure does not open a
        # second conflict over assumptions already being narrowed.
        if not self._diagnose_substitution(node_id, refuted=True, now=now):
            self._blame_failure(node_id, now, depth)
        # Refuting a candidate can leave exactly one answer standing, which is
        # then entailed rather than merely likely — no probe needed.
        if group:
            self._deduce_last_member(group, now)
        # A refutation can also be the very fact that explains an open conflict.
        self._narrow_conflicts(now)

    def _blame_failure(self, node_id: str, now: datetime, depth: int = 0) -> None:
        """Decide what a failure says about the assumptions underneath it.

        A hypothesis that rests on several assumptions and fails has established
        exactly one thing: **those assumptions cannot all hold together** at the
        depth it was tested. It has NOT established that any particular one is
        wrong. Blaming all of them — the original behaviour — discards every
        correct confirmation among them and forces the agent to rediscover it,
        which measurably cost more than the failure itself.

        So blame is only assigned when it is determinate:

        - **One assumption** → unambiguous. Propagate upstream exactly as before.
        - **Several** → record a conflict set and leave the assumptions standing.
          They are marked NEEDS_REVISION, which is the honest state — each is now
          in doubt — but that mark is *not* an instruction to re-test them one by
          one. See ``_substitution_plan`` for what is dispatched instead.

        Note what is deliberately NOT done here: the competing alternatives of
        the assumptions are *not* reopened. Reopening every axis before knowing
        which one is at fault hands the navigator a pile of untested siblings and
        turns a targeted question into a blind sweep. They are reopened exactly
        when an assumption is actually convicted.
        """
        dep_parents = self._graph.parents(node_id, EdgeType.DEPENDENCY)
        if len(dep_parents) <= 1:
            self.invalidate_upstream(node_id)
            return

        # A re-probe of the same failing combination proves nothing new. Recording
        # it again would inflate every conflict metric and hand the agent a
        # duplicate to reason about, so identical open conflicts are collapsed.
        members = set(dep_parents)
        if any(set(n["member_ids"]) == members for n in self._store.get_nogoods(open_only=True)):
            return

        self._store.add_nogood(node_id, dep_parents, now, conflict_depth=depth)

        for pid in dep_parents:
            parent = self._store.get_node(pid)
            if parent is None or parent.status != Status.VERIFIED:
                continue
            self._store.change_status(
                pid,
                Status.NEEDS_REVISION,
                reason=(
                    f"{REVIEW_REASON_PREFIX}{depth} "
                    f"(confirmed at depth {parent.confirmed_depth}, "
                    f"but {node_id} failed at depth {depth})"
                ),
                now=now,
            )
            self._refresh_node_in_graph(pid)

    def _diagnosing_nogoods(self) -> list[dict[str, Any]]:
        """Conflicts whose cause is still being narrowed by substitution.

        A conflict leaves this set when a culprit is convicted, when every member
        has been cleared (a genuine multi-assumption interaction), or when the
        broad recovery has already fired for it.
        """
        return [
            n
            for n in self._store.get_nogoods(open_only=True)
            if not n["reopened_at"] and int(n["probe_index"] or 0) < len(n["member_ids"])
        ]

    def _substitution_plan(self, nogood: dict[str, Any]) -> dict[str, Any] | None:
        """The next single-assumption swap that would narrow this conflict.

        The discriminating experiment for "these cannot all hold together" is
        another *composition*, never an isolated re-test. Each member has already
        passed the isolated test — that is precisely why the conflict is
        indeterminate — so re-running it interrogates a proposition nobody
        doubts. In an environment where a component can only fail in integration,
        the isolated re-test is not merely uninformative by accident but by
        construction, and a run of this engine spent a tenth of its entire probe
        budget on exactly that.

        Swapping one member for a competing answer and re-testing the composition
        *is* decisive, and it is decisive on a single bit:

        - the composition still fails → the swapped member was not the cause,
          because removing it changed nothing;
        - the composition stops failing → it was, because removing it was the
          only change.

        One probe therefore eliminates a whole question, where sweeping that
        question's alternatives individually costs one probe per alternative.
        Deliberately expressed as refuted / not-refuted rather than as a score
        threshold: the rule is then a property of the belief state rather than of
        any particular scoring function, and cannot be tuned into a heuristic
        that rewards the appearance of progress.

        Members with no competing answer left to swap in are skipped — there is
        no experiment to run for them — as are swaps whose result is already
        recorded, because an experiment whose answer is known buys nothing.

        Pure: the cursor is advanced only by ``_diagnose_substitution``, when a
        member has actually been cleared. Skipping is recomputed each call so a
        read of the conflict never quietly rewrites it.
        """
        members: list[str] = list(nogood["member_ids"])
        known = {frozenset(n["member_ids"]) for n in self._store.get_nogoods()}
        for index in range(int(nogood["probe_index"] or 0), len(members)):
            member = members[index]
            keep = [m for m in members if m != member]
            candidate = self._substitution_candidate(member, keep, known)
            if candidate is None:
                continue
            return {
                "nogood_id": nogood["id"],
                "member_id": member,
                "candidate_id": candidate,
                "keep_ids": keep,
                "min_depth": int(nogood["conflict_depth"] or 0),
                "cleared": members[: int(nogood["probe_index"] or 0)],
                "remaining": len(members) - index,
            }
        return None

    def _substitution_candidate(
        self, member_id: str, keep: list[str], known: set[frozenset[str]]
    ) -> str | None:
        """A competing answer that could stand in for ``member_id``.

        Any sibling that has not been refuted will do: the swap is a diagnostic
        about *which question* is at fault, so the substitute does not have to be
        the right answer — it only has to be different. Preferring an untested
        sibling keeps the diagnostic from also being a re-probe, and a swap whose
        combination is already a recorded conflict is skipped outright: its
        answer is on file, so running it would spend a probe to learn nothing.
        """
        node = self._store.get_node(member_id)
        if node is None or not node.exclusion_group:
            return None
        siblings = self._store.get_nodes_in_exclusion_group(node.exclusion_group, member_id)
        usable = [s for s in siblings if not self._is_eliminated(s)]
        usable.sort(key=lambda s: (s.evidence_count > 0, s.id))
        for sibling in usable:
            if frozenset([*keep, sibling.id]) not in known:
                return sibling.id
        return None

    def _substitution_of(self, node_id: str) -> tuple[dict[str, Any], str, str] | None:
        """Match a node against the conflicts it could be a substitution for.

        A composition is a substitution when its assumptions are a conflict's
        members with exactly one replaced. Recognising it here rather than asking
        the caller to declare it means the diagnosis works from what was actually
        built, and cannot be thrown off by a caller that forgets to say what it
        was doing.

        Returns the conflict, the member that was swapped **out**, and the
        candidate swapped **in** — the outcome is a verdict on both of them.
        """
        parents = set(self._graph.parents(node_id, EdgeType.DEPENDENCY))
        if not parents:
            return None
        for nogood in self._diagnosing_nogoods():
            members = set(nogood["member_ids"])
            if len(parents) != len(members):
                continue
            swapped_out = members - parents
            swapped_in = parents - members
            if len(swapped_out) == 1 and len(swapped_in) == 1:
                return nogood, swapped_out.pop(), swapped_in.pop()
        return None

    def _diagnose_substitution(
        self, node_id: str, refuted: bool, now: datetime, depth: int = 0, achieved: bool = False
    ) -> bool:
        """Read a substitution's outcome as a verdict on the member it replaced.

        Returns whether the node *was* a substitution, so the caller can skip the
        ordinary blame path: this failure was predicted and has already been
        interpreted, and recording it as a fresh indeterminate conflict would
        start a second narrowing over the same assumptions and put every one of
        them back under review.

        A cleared member keeps its confirmation and its alternatives stay
        retired: nothing about it was refuted, it simply is not what the failure
        was about.

        The swap answers **two** questions, and only the first is answered by it
        stopping the failure:

        - *which assumption was at fault* — the one that was removed, since
          removing it was the only change. True whether the rebuilt composition
          merely stops failing or clears the bar outright.
        - *what the right answer to that question is* — answered only when the
          composition actually **clears the bar** (``achieved``). A swap is
          chosen to be *different*, not to be correct (see
          ``_substitution_candidate``), so a composition that stops failing at a
          sub-par score says the substitute is not the culprit and nothing more.
          Confirming it there retires its siblings and buries the correct answer
          behind an exclusion nobody ever tested — which ended three real
          episodes at an empty frontier with the goal unmet. It does, though,
          rule the substitute *out*: with every other member exonerated by the
          same verdict, a composition that is right everywhere else and still
          falls short cannot have the answer in the one slot that was swapped.
        """
        match = self._substitution_of(node_id)
        if match is None:
            return False
        nogood, member, candidate = match
        members: list[str] = list(nogood["member_ids"])

        if refuted:
            self._store.advance_nogood_probe(nogood["id"], members.index(member) + 1, now)
            refreshed = self._refetch_nogood(nogood["id"]) or nogood
            if self._substitution_plan(refreshed) is None:
                # Every member has been swapped out and the composition failed
                # every time: no single assumption is the cause, so the answer is
                # somewhere among all the alternatives they retired.
                self._recover_from_interaction(self._refetch_nogood(nogood["id"]) or refreshed, now)
            return True

        self._store.resolve_nogood(nogood["id"], member, now)
        self._convict(member, nogood["id"], now)
        self._release_from_review([m for m in members if m != member], now)
        # After the conviction, never before: convicting the member retracts the
        # exclusion that had retired this very candidate, and settling the
        # question first would only see it undone.
        if achieved:
            self._confirm_substitute(candidate, node_id, now, depth)
        else:
            self._eliminate_substitute(candidate, node_id, now)
        return True

    def _confirm_substitute(
        self, candidate: str, composition_id: str, now: datetime, depth: int
    ) -> bool:
        """Settle the question a successful diagnostic swap has just answered.

        Only reached when the rebuilt composition **cleared the bar**, which is
        what makes this sound: the same combination failed with the convicted
        member in that slot and now achieves the objective with this one, so the
        substitute is the answer and not merely a different one. Leaving it
        UNTESTED cost a probe per resolved conflict — the navigator would hand
        out the very answer the diagnosis had just produced.

        The posterior is left untouched, exactly as for deduction by
        elimination: the status records what was established, the belief records
        what was directly observed of this node, and no isolated observation of
        it was ever made. The confirmation depth *is* recorded, because the
        composition really was tested at that depth and a later, deeper failure
        is entitled to reopen this.

        Declines when the candidate is no longer an open question — something
        that settled it on its own evidence outranks an inference drawn about it.
        """
        node = self._store.get_node(candidate)
        if node is None or node.status not in _OPEN_STATUSES:
            return False
        self._store.change_status(
            candidate,
            Status.VERIFIED,
            reason=f"{SUBSTITUTION_CONFIRM_PREFIX}{composition_id}",
            now=now,
        )
        self._store.set_confirmed_depth(candidate, max(depth, node.confirmed_depth or 0))
        self._refresh_node_in_graph(candidate)
        self._sync_exclusion(candidate, node.status, Status.VERIFIED, now)
        self.verify_upstream(candidate)
        return True

    def _eliminate_substitute(self, candidate: str, composition_id: str, now: datetime) -> bool:
        """Rule out the substitute a sub-par diagnostic swap has just tested.

        The mirror of ``_confirm_substitute`` and sound for the same reason. The
        conviction that just fired asserts every *other* member of the rebuilt
        composition is the right answer to its own question; the composition was
        nevertheless conclusive and short of the bar. So the one slot that was
        not exonerated cannot hold the answer either — this value is out, and its
        question stays open for the siblings that have not been tried.

        Left undone, the navigator hands the value straight back and the agent
        spends a probe re-establishing in isolation what the swap already showed:
        nine of fifteen conflict episodes in the last run did exactly that, and
        every one of them came back refuted.

        EXHAUSTED rather than INVALIDATED because nothing was refuted outright —
        the value was tested in the only context that can discriminate it and did
        not clear the bar, which is what EXHAUSTED means. The distinct reason
        marker matters: an elimination drawn from a real test counts toward
        deduction by elimination, whereas one drawn from the exclusion inference
        would let a single confirmation deduce its own group.

        Declines when the candidate is no longer an open question, so evidence
        the caller gathered itself always outranks an inference about it.
        """
        node = self._store.get_node(candidate)
        if node is None or node.status not in _OPEN_STATUSES:
            return False
        self._store.change_status(
            candidate,
            Status.EXHAUSTED,
            reason=f"{SUBSTITUTION_ELIMINATE_PREFIX}{composition_id}",
            now=now,
        )
        self._refresh_node_in_graph(candidate)
        if node.exclusion_group:
            self._deduce_last_member(node.exclusion_group, now)
        return True

    def _refetch_nogood(self, nogood_id: int) -> dict[str, Any] | None:
        """Re-read a conflict after its cursor moved, so callers see it current."""
        return next((n for n in self._store.get_nogoods() if n["id"] == nogood_id), None)

    def _under_confirmed(self, node_ids: list[str], depth: int) -> list[str]:
        """Which of these assumptions a failure at ``depth`` can implicate.

        An assumption that already passed a test at least as demanding as the one
        that failed is not implicated: the failure happened in a context its own
        evidence already covers, so it says nothing new about it.

        ``depth <= 0`` means no depth information was supplied at all, which is
        the ordinary case for a caller that does not model rigour. Rigour-based
        exoneration is then switched off entirely and blame narrows purely on
        participation in confirmed results — exactly the behaviour a
        depth-unaware caller had before depths existed. Treating "unspecified"
        as "shallowest" instead would silently declare every conflict an
        interaction effect and disable narrowing for everyone not using depths.

        An empty result is meaningful rather than degenerate: every assumption
        survived a test as demanding as the one that failed, so the failure is a
        genuine interaction effect and no single assumption is at fault.
        """
        if depth <= 0:
            return list(node_ids)
        return [
            nid
            for nid in node_ids
            if (node := self._store.get_node(nid)) is not None
            and (node.confirmed_depth is None or node.confirmed_depth < depth)
        ]

    def _conflict_suspects(self) -> dict[str, int]:
        """Nodes the conflict machinery wants tested next → the depth they need.

        These are the competing answers of a **convicted** member: the conflict
        has been narrowed to one question, and the answer to that question is
        almost certainly among the alternatives the convicted value had retired.
        Without this the navigator draws from an unprioritised frontier in which
        every freshly reopened alternative carries the same untouched prior, so
        it picks among them at random and the targeted recovery degenerates into
        the blind sweep it exists to prevent.

        Members of an *unresolved* conflict are deliberately absent. They are not
        dispatched individually at all — the experiment that discriminates a
        conflict is a composition, not a re-test of a part (see
        ``_substitution_plan``).

        A suspect handed out ``MAX_REVIEW_DISPATCHES`` times without the conflict
        resolving stops being prioritised, so a caller that ignores the advice
        cannot starve the rest of the frontier.
        """
        suspects: dict[str, int] = {}
        for nogood in self._store.get_nogoods(open_only=False):
            culprit_id = nogood["resolved_culprit_id"]
            if not culprit_id:
                continue
            culprit = self._store.get_node(str(culprit_id))
            if culprit is None or not culprit.exclusion_group:
                continue
            depth = int(nogood["conflict_depth"] or 0)
            for sibling in self._store.get_nodes_in_exclusion_group(
                culprit.exclusion_group, str(culprit_id)
            ):
                if sibling.status not in _OPEN_STATUSES:
                    continue
                if self._review_dispatches[sibling.id] >= MAX_REVIEW_DISPATCHES:
                    continue
                suspects[sibling.id] = max(suspects.get(sibling.id, depth), depth)
        return suspects

    def _remaining_suspects(self, nogood: dict[str, Any], exonerated: set[str]) -> list[str]:
        """Members of a conflict that could still be its cause.

        A member drops out on either of two independent grounds: it has since
        helped produce a confirmed result (exonerated by success), or its own
        confirmation already reaches the depth at which the conflict arose
        (exonerated by rigour). Both are reasons the failure cannot be about it.
        """
        depth = int(nogood["conflict_depth"] or 0)
        deep_enough = set(nogood["member_ids"]) - set(
            self._under_confirmed(list(nogood["member_ids"]), depth)
        )
        return [m for m in nogood["member_ids"] if m not in exonerated and m not in deep_enough]

    def _exonerated_nodes(self) -> set[str]:
        """Assumptions proven compatible by participating in a verified result.

        Only used to narrow blame: a node that helped produce a confirmed
        success cannot be the reason a different combination failed.
        """
        exonerated: set[str] = set()
        for node in self._store.get_all_nodes():
            if node.status != Status.VERIFIED:
                continue
            exonerated.update(self._graph.parents(node.id, EdgeType.DEPENDENCY))
        return exonerated

    def _narrow_conflicts(self, now: datetime) -> list[str]:
        """Pin every conflict whose cause has become determinate.

        Blame lands here, and it lands precisely. Two things make it provable:

        - a member has been **refuted outright**, which explains the failure on
          its own; or
        - every other member has been **exonerated by success** — each has since
          taken part in a combination that actually worked — leaving exactly one
          candidate.

        Exoneration by *rigour* alone is deliberately not enough to convict.
        Assumptions that each pass their own test at the failing depth can still
        fail together: that is an interaction effect, and the last member to be
        re-tested is not its cause, it is merely the last one checked. Convicting
        on that basis would turn an ordering accident into a refutation. Such a
        conflict stays open with its remaining suspect visible, and one more
        re-test settles which of the two situations it is.

        When *every* member has been cleared and the conflict still stands, that
        question is answered: it is an interaction effect, and
        ``_recover_from_interaction`` acts on it — but only once the substitution
        diagnosis has run out of members to clear. Reopening every question while
        a targeted swap could still name the culprit is the expensive mistake
        that recovery is meant to avoid, so the broad path stays a last resort.

        Members that survive the narrowing are released from review: their
        confirmation stood up.
        """
        exonerated = self._exonerated_nodes()
        culprits: list[str] = []
        for nogood in self._store.get_nogoods(open_only=True):
            members = list(nogood["member_ids"])
            suspects = self._remaining_suspects(nogood, exonerated)

            refuted = [
                m
                for m in members
                if (n := self._store.get_node(m)) is not None
                and n.status in (Status.INVALIDATED, Status.PRUNED)
            ]
            if refuted:
                culprit = refuted[0]
            elif len(suspects) == 1 and all(m in exonerated for m in members if m != suspects[0]):
                culprit = suspects[0]
            else:
                if not suspects and self._substitution_plan(nogood) is None:
                    self._recover_from_interaction(nogood, now)
                continue

            self._store.resolve_nogood(nogood["id"], culprit, now)
            culprits.append(culprit)
            self._convict(culprit, nogood["id"], now)
            self._release_from_review([m for m in members if m != culprit], now)
        return culprits

    def _recover_from_interaction(self, nogood: dict[str, Any], now: datetime) -> list[str]:
        """Act on a conflict that has been proven to be an interaction effect.

        Every member has now individually survived a test at least as demanding
        as the one that failed, yet together they still fail. That is no longer
        an open question about *which* assumption is wrong — it is a positive
        finding: at least one of these confirmations does not carry into
        composition, so the correct answer to at least one of these questions is
        something other than the value currently confirmed.

        Those other answers are precisely the alternatives each member retired
        when it was confirmed, and leaving them retired is what strands the
        search. The engine already applies exactly this reasoning to a member it
        *convicts* — a confirmation that loses its authority cannot keep its
        alternatives buried — and the same holds here with more force, because
        the failure has been shown to be about the values themselves rather than
        about which one to doubt. Without this the agent is left to rediscover
        those alternatives by hand, which is the single most expensive thing the
        conflict machinery was built to avoid.

        The members keep their own confirmations: each is still true in
        isolation, and nothing has refuted any of them. They are only released
        from review, since re-testing them has already been shown to settle
        nothing. Recovery is marked so it happens once — repeating it on every
        later observation would undo answers found in the meantime.
        """
        if nogood.get("reopened_at"):
            return []

        members = list(nogood["member_ids"])
        self._store.mark_nogood_reopened(nogood["id"], now)
        self._release_from_review(members, now)

        reopened: list[str] = []
        for member in members:
            reopened.extend(self._reopen_alternatives(member, now))
        return reopened

    def _reopen_alternatives(self, node_id: str, now: datetime, why: str = "") -> list[str]:
        """Return a confirmed node's competing alternatives to the frontier.

        Unlike ``_retract_exclusion`` this does not require the confirmation to
        have been withdrawn, and it deliberately does not re-attribute the
        siblings to the still-standing member — that member's authority is
        exactly what is in doubt.
        """
        node = self._store.get_node(node_id)
        if node is None or not node.exclusion_group:
            return []

        why = why or (
            f"'{node_id}' holds in isolation but takes part in a composition that "
            f"fails, so the answer may be here instead"
        )
        marker = f"{EXCLUSION_REASON_PREFIX}{node_id}"
        reopened: list[str] = []
        for sibling in self._store.get_nodes_in_exclusion_group(node.exclusion_group, node_id):
            if sibling.status != Status.EXHAUSTED:
                continue
            history = self._store.get_status_history(sibling.id)
            if not history or history[-1]["reason"] != marker:
                continue
            self._store.change_status(
                sibling.id,
                Status.UNTESTED,
                reason=f"{INTERACTION_REOPEN_PREFIX}{why}",
                now=now,
            )
            self._refresh_node_in_graph(sibling.id)
            reopened.append(sibling.id)
        return reopened

    def _recover_from_underperformance(self, now: datetime) -> list[str]:
        """Reopen the search when every question is answered and the answer is not good enough.

        The last thing standing between a settled belief state and a premature
        end. A composition that is *conclusive but below the bar* is the one
        outcome the engine had no reading for: it did not fail, so no conflict is
        recorded and no blame is assigned; it did not succeed, so no goal is met.
        It simply settles, and if every question already has a confirmed answer
        the frontier is then empty with the objective unreached — which the
        caller reads as "the search is over". Three real episodes ended exactly
        there, at 14, 16 and 23 probes out of a budget of 100, each one a handful
        of probes from the answer.

        The reading it was missing: assembling every confirmed answer and landing
        short is *positive evidence that one of those confirmations is wrong*.
        Each was established on its own, none has been refuted, and together they
        do not reach the objective — which is the interaction case exactly, and
        the response is the same one: hand back the alternatives those
        confirmations retired.

        Only alternatives retired by the **exclusion inference** come back. A
        sibling that was tested and settled on its own evidence stays settled, so
        every reopened node costs at most one probe before it is settled for
        good and the recovery cannot cycle. Deliberately a last resort, invoked
        only once the frontier is otherwise empty: reopening a question that
        still has cheaper work in front of it would trade a targeted search for a
        sweep.
        """
        why = (
            "every question is answered and the answers assembled still fall short of the "
            "objective, so at least one of them is not the right answer"
        )
        reopened: list[str] = []
        for node in self._store.get_all_nodes():
            if node.status != Status.EXHAUSTED or node.is_goal:
                continue
            parents = self._graph.parents(node.id, EdgeType.DEPENDENCY)
            if len(parents) < 2:
                continue
            for parent in parents:
                reopened.extend(self._reopen_alternatives(parent, now, why))
        return reopened

    def _convict(self, culprit: str, nogood_id: int, now: datetime) -> None:
        """Invalidate the identified cause of a conflict, with the full cascade."""
        node = self._store.get_node(culprit)
        if node is None or node.status in (Status.INVALIDATED, Status.PRUNED):
            return
        self._store.change_status(
            culprit,
            Status.INVALIDATED,
            reason=f"identified as the sole remaining cause of conflict {nogood_id}",
            now=now,
        )
        self._store.set_confirmed_depth(culprit, None)
        self._refresh_node_in_graph(culprit)
        # A convicted assumption loses the authority to keep its alternatives
        # retired — the correct answer to that question is very likely among them.
        self._retract_exclusion(culprit, now)
        self._cascade_prune(culprit, now)
        self.invalidate_upstream(culprit)

    def _cascade_prune(self, root_id: str, now: datetime) -> list[str]:
        """Prune what rested on a refutation — but never the objective itself.

        A goal is protected everywhere else: it is never dispatched, never
        accepts evidence, never invalidated, never exhausted. The cascade was
        the one path that reached it, and it reached it through the very edge
        the caller is *told* to create — wiring a candidate combination to the
        goal so the goal can be reached. The first combination to fail then
        destroyed the objective, and every real episode of one evaluation run
        spent LLM turns re-creating a goal the engine had thrown away.

        The reasoning is the same one that keeps a goal from being invalidated:
        an attempt failing says nothing about whether the objective still
        stands. What rested on the refutation is pruned; the thing it was
        *aiming at* is not.
        """
        pruned: list[str] = []
        for pid in self._graph.cascading_prune(root_id):
            node = self._store.get_node(pid)
            if node is not None and node.is_goal:
                continue
            self._store.change_status(pid, Status.PRUNED, reason="ancestor invalidated", now=now)
            self._refresh_node_in_graph(pid)
            pruned.append(pid)
        return pruned

    def _release_from_review(self, node_ids: list[str], now: datetime) -> list[str]:
        """Restore assumptions that were under review but turned out innocent.

        Leaving them in NEEDS_REVISION would keep the navigator re-testing
        confirmations that have already been vindicated, and would block every
        combination that depends on them.
        """
        released: list[str] = []
        for nid in node_ids:
            node = self._store.get_node(nid)
            if node is None or node.status != Status.NEEDS_REVISION:
                continue
            history = self._store.get_status_history(nid)
            if not history or not str(history[-1]["reason"]).startswith(REVIEW_REASON_PREFIX):
                continue
            self._store.change_status(
                nid,
                Status.VERIFIED,
                reason="released from conflict review: another member was the cause",
                now=now,
            )
            self._refresh_node_in_graph(nid)
            released.append(nid)
        return released

    def _deduce_last_member(self, group: str, now: datetime) -> str | None:
        """Confirm the sole surviving member of an exclusion group by elimination.

        An exclusion group asserts that exactly one of its members is true. Once
        every other member has been refuted, the survivor follows by deduction —
        probing it can only confirm what is already entailed, and that probe is
        pure waste. This is the closed-world assumption being *used* rather than
        merely declared.

        Deliberately conservative:

        - at least one member must have been eliminated, so an untouched group is
          never collapsed onto an arbitrary member;
        - the survivor must still be an open question — one already EXHAUSTED by
          its own sub-par evidence is a contradiction with the group's premise,
          and inventing a confirmation on top of it would bury that signal;
        - the posterior is left untouched, because no observation was made. The
          status records what was deduced; the belief records what was seen.
        """
        members = self._store.get_nodes_in_exclusion_group(group)
        eliminated = [m for m in members if self._is_eliminated(m)]
        survivors = [m for m in members if not self._is_eliminated(m)]
        if not eliminated or len(survivors) != 1:
            return None

        survivor = survivors[0]
        if survivor.status not in _OPEN_STATUSES:
            return None

        self._store.change_status(
            survivor.id,
            Status.VERIFIED,
            reason=(f"{DEDUCTION_REASON_PREFIX}every other member of '{group}' is ruled out"),
            now=now,
        )
        self._refresh_node_in_graph(survivor.id)
        self._sync_exclusion(survivor.id, survivor.status, Status.VERIFIED, now)
        self.verify_upstream(survivor.id)
        return survivor.id

    def _is_eliminated(self, node: Node) -> bool:
        """Whether a node has been ruled out as an answer to its own question.

        Refutation is the obvious case. Conclusive-but-sub-par evidence is the
        other: EXHAUSTED means the question was actually put and the answer did
        not clear the bar, which inside an exclusion group rules the member out
        just as firmly.

        A member EXHAUSTED by the *exclusion inference* is explicitly not
        eliminated. Nothing was observed about it — it was set aside because a
        sibling was confirmed — so counting it as ruled out would let a single
        confirmation deduce the rest of the group from itself.
        """
        if node.status in (Status.INVALIDATED, Status.PRUNED):
            return True
        if node.status != Status.EXHAUSTED:
            return False
        history = self._store.get_status_history(node.id)
        reason = str(history[-1]["reason"]) if history else ""
        return not reason.startswith(EXCLUSION_REASON_PREFIX)

    def _verify_node(self, node_id: str, now: datetime, depth: int = 0) -> None:
        """Transition to VERIFIED + exclusion inference + upstream verify.

        ``depth`` is the rigour of the observation that produced the
        confirmation. It is stored because a confirmation only supports claims
        tested no deeper than itself: everything that later rests on this node
        and fails at a greater depth is entitled to question it.
        """
        node = self._store.get_node(node_id)
        old_status = node.status if node is not None else Status.UNTESTED
        prior_depth = node.confirmed_depth if node is not None else None
        self._store.change_status(
            node_id,
            Status.VERIFIED,
            reason="posterior converged above verify bar",
            now=now,
        )
        # A re-confirmation never weakens what was already established, so the
        # depth only ever ratchets upward.
        self._store.set_confirmed_depth(node_id, max(depth, prior_depth or 0))
        self._refresh_node_in_graph(node_id)
        self._sync_exclusion(node_id, old_status, Status.VERIFIED, now)
        self.verify_upstream(node_id)
        # A composition that stands in for a conflict member and *succeeds*
        # names that member: removing it was the only change. Clearing the bar
        # also names the substitute as the answer.
        self._diagnose_substitution(node_id, refuted=False, now=now, depth=depth, achieved=True)
        # A success may have just cleared the last innocent member of an open
        # conflict, which is exactly when blame becomes provable.
        self._narrow_conflicts(now)

    def _sync_exclusion(
        self, node_id: str, old_status: Status, new_status: Status, now: datetime
    ) -> None:
        """Keep the exclusion inference in step with a status transition.

        The rule belongs to the *transition*, not to any one call site: entering
        VERIFIED settles the competing alternatives, and leaving VERIFIED gives
        that authority up again. Routing every path through here is what stops
        the same status reached two different ways (direct evidence, upstream
        REFINEMENT promotion, a manual override) from having different
        consequences.

        Conflict review is deliberately routed *around* this method rather than
        through it. When blame is determinate the doubted node really has lost
        its authority and its alternatives are the way forward; when several
        assumptions are jointly implicated, none of them has been doubted
        individually yet, and reopening every one of their questions replaces a
        single precise experiment with a blind sweep. See ``_blame_failure``.
        """
        if new_status == old_status:
            return
        if new_status == Status.VERIFIED:
            self._apply_exclusion(node_id, now)
        elif old_status == Status.VERIFIED:
            self._retract_exclusion(node_id, now)

    def _apply_exclusion(self, node_id: str, now: datetime) -> list[str]:
        """Settle the competing alternatives of a just-confirmed node.

        Members of an exclusion group are mutually exclusive answers to one
        question, so confirming one means the others need not be tested. They are
        marked EXHAUSTED rather than INVALIDATED on purpose: this is an
        *inference* conditional on the confirmation holding, not an observation
        that the alternatives are false. EXHAUSTED settles them (they leave the
        frontier, which is the whole point — no budget is spent re-testing a
        question already answered) without cascading a prune through their
        subtrees, so withdrawing the confirmation later is a cheap, local undo.
        """
        node = self._store.get_node(node_id)
        if node is None or not node.exclusion_group:
            return []

        settled: list[str] = []
        for sibling in self._store.get_nodes_in_exclusion_group(node.exclusion_group, node_id):
            # Only settle still-open alternatives; never overwrite a terminal
            # state that was reached by real evidence.
            if sibling.status not in _OPEN_STATUSES:
                continue
            self._store.change_status(
                sibling.id,
                Status.EXHAUSTED,
                reason=f"{EXCLUSION_REASON_PREFIX}{node_id}",
                now=now,
            )
            self._refresh_node_in_graph(sibling.id)
            settled.append(sibling.id)
        return settled

    def _retract_exclusion(self, node_id: str, now: datetime) -> list[str]:
        """Undo the exclusion inference when its justification is withdrawn.

        A sibling settled purely *because* this node was confirmed becomes an
        open question again the moment that confirmation is retracted. Without
        this, a wrong confirmation would permanently bury the correct
        alternative and the search could never recover — the inference must be
        exactly as retractable as the belief that produced it. Only siblings
        settled by THIS node are restored; ones settled by their own evidence
        are left untouched.

        If some *other* member of the group is confirmed by the time this runs,
        the question already has a live answer, so the sibling is re-attributed
        to that confirmation rather than reopened. Reopening it would put a
        settled question back on the frontier and buy a probe that cannot change
        anything.
        """
        node = self._store.get_node(node_id)
        if node is None or not node.exclusion_group:
            return []

        siblings = self._store.get_nodes_in_exclusion_group(node.exclusion_group, node_id)
        standing = next((s for s in siblings if s.status == Status.VERIFIED), None)

        marker = f"{EXCLUSION_REASON_PREFIX}{node_id}"
        restored: list[str] = []
        for sibling in siblings:
            if sibling.status != Status.EXHAUSTED:
                continue
            history = self._store.get_status_history(sibling.id)
            if not history or history[-1]["reason"] != marker:
                continue
            if standing is not None and standing.id != sibling.id:
                self._store.change_status(
                    sibling.id,
                    Status.EXHAUSTED,
                    reason=f"{EXCLUSION_REASON_PREFIX}{standing.id}",
                    now=now,
                )
                self._refresh_node_in_graph(sibling.id)
                continue
            self._store.change_status(
                sibling.id,
                Status.UNTESTED,
                reason=f"{EXCLUSION_RETRACT_PREFIX}{node_id} no longer confirmed",
                now=now,
            )
            self._refresh_node_in_graph(sibling.id)
            restored.append(sibling.id)
        return restored

    def _exhaust_node(
        self, node_id: str, now: datetime, last_success: float, depth: int = 0
    ) -> None:
        """Transition to EXHAUSTED — conclusively tested, did not clear the bar.

        Deliberately does NOT cascade or propagate: the hypothesis was not
        refuted, so its descendants stay valid and a REFINEMENT child can still
        build on it. The only effect is that the node stops being re-dispatched.

        It does, however, rule the node out as an answer to its own exclusion
        group — the question was put and this answer did not clear the bar — so
        it can leave a single survivor standing exactly as a refutation does.

        And it is the ordinary landing place for a diagnostic substitution: a
        composition that no longer *fails* but does not clear the bar either has
        still answered the only question the swap asked — which member was at
        fault — so the verdict is read here as well as on the verified path. It
        has emphatically **not** established that the substitute is the right
        answer, so ``achieved`` stays false; the substitute is ruled out for its
        own question while its remaining siblings keep their place in the search.
        """
        node = self._store.get_node(node_id)
        self._store.change_status(
            node_id,
            Status.EXHAUSTED,
            reason=f"conclusive evidence below verify bar (success={last_success:.4f})",
            now=now,
        )
        self._refresh_node_in_graph(node_id)
        self._diagnose_substitution(node_id, refuted=False, now=now, depth=depth, achieved=False)
        if node is not None and node.exclusion_group:
            self._deduce_last_member(node.exclusion_group, now)
