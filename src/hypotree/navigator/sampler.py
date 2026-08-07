"""Thompson Sampling navigator with seeded RNG, epsilon-tiebreak decay, and
the "DONE" sentinel for terminal states.

The navigator is the selection heart of the closed loop: it takes a list of
frontier-eligible nodes, draws theta from each node's Beta posterior via an
injected seeded Generator, and returns the selected node — or a DONE sentinel
when the frontier is empty or all goals are met.

Selection is a single coherent acquisition function: TS draw + lexicographic
epsilon-tiebreak on staleness (no additive/multiplicative hacks). Anti-starvation
is intentionally weak in v0; the discounted-Beta fix lands in a later phase.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hypotree.models.node import Node
from hypotree.models.status import Status, posterior_mean, utcnow
from hypotree.navigator.convergence import convergence_gate, credible_interval

# Default claim/lease TTL in seconds.
DEFAULT_LEASE_TTL_S = 900

# Nodes whose TS draws are within this band are treated as tied; among ties,
# the stalest (largest now - updated_at) is preferred as an anti-starvation measure.
EPSILON_TIE = 0.01

# Convergence-gate defaults.
EPSILON_CI = 0.1
N_MAX_SAMPLES = 50

# Global default verification threshold for non-goal nodes.
VERIFY_THRESHOLD = 0.8

# Below this posterior mean upper-bound, a stochastic node is "confidently bad."
EPSILON_LOW = 0.2

# Statuses that remove a node from the running for its exclusion group's answer
# on the status alone. EXHAUSTED is deliberately absent: it is reached two ways
# that mean opposite things, and only the reason can tell them apart — see
# `live_group_counts`.
_REFUTED_STATUSES = frozenset({Status.INVALIDATED, Status.PRUNED})


def live_group_counts(
    all_nodes: list[Node], eliminated_ids: set[str] | None = None
) -> dict[str, int]:
    """Count the still-viable members of every exclusion group.

    A group is a closed-world question: its members are competing answers, of
    which one is true. The count of members not yet ruled out is therefore the
    size of the remaining answer space, and it is what makes the group's
    posterior informative — three eliminations out of four leave the survivor
    nearly certain, and no per-node Beta can express that.

    ``eliminated_ids`` carries the engine's own definition of *ruled out*, which
    is the one the guide states and the deduction rule uses: refuted, **or**
    EXHAUSTED by its own evidence. Counting only refutations made this inert on
    the common path, because the deterministic regime refutes only on an exact
    0.0 and sends everything else to EXHAUSTED — so a five-way question with four
    candidates measured and rejected still reported k=5, and the survivor's prior
    mean stayed at 0.2, *below* an untouched ungrouped node. The mechanism
    deprioritised the one candidate nearly certain to be the answer.

    Omitting it falls back to status alone, which is what a caller with no
    history to hand (the read-only dashboard) can offer.
    """
    eliminated_ids = eliminated_ids or set()
    counts: dict[str, int] = {}
    for node in all_nodes:
        if not node.exclusion_group:
            continue
        if node.status in _REFUTED_STATUSES or node.id in eliminated_ids:
            continue
        counts[node.exclusion_group] = counts.get(node.exclusion_group, 0) + 1
    return counts


def effective_posterior(node: Node, live_counts: dict[str, int]) -> tuple[float, float]:
    """The node's Beta parameters under the closed-world group assumption.

    For an ungrouped node this is just its own posterior over a uniform
    Beta(1,1) prior. For a member of an exclusion group with ``k`` live
    candidates, the prior is instead the marginal of a symmetric categorical
    over those candidates — ``Beta(1, k-1)``, mean ``1/k`` — because exactly one
    of them is the answer. Observations accumulated on the node are carried over
    unchanged; only the prior is swapped.

    The consequence that matters in practice: a question with two candidates
    left outranks an untouched question with four, so the navigator finishes the
    question it has already invested in instead of interleaving all of them at
    random. That payoff was inert until the count learned to shrink on an
    EXHAUSTED member — see ``live_group_counts``.

    It deliberately does **not** make a lone survivor certain. Over a closed
    group there is nothing to select: the engine deduces the last member
    outright, without a probe. Over an open one the remaining candidate has no
    stronger claim than any untried node — "the other four learning rates
    failed" says nothing about the fifth, which is the whole content of
    ``exclusion_closed=False``. So k=1 yields Beta(1,1), and that is the answer,
    not a clamp working around one.
    """
    prior_beta = 1.0
    if node.exclusion_group:
        k = live_counts.get(node.exclusion_group, 1)
        prior_beta = float(max(k - 1, 1))
    # node.alpha/beta start at the uniform Beta(1,1) and accumulate evidence, so
    # replacing the prior means adding the difference to beta only.
    return node.alpha, node.beta + (prior_beta - 1.0)


# How far a single node's cost may pull it from the workspace median. Four orders
# of magnitude is the real spread — a unit test against a fine-tune — and beyond
# that the ratio stops carrying information and starts expressing an outlier.
_COST_CLAMP = 1e4

# How long a node may be deferred for being expensive before cost stops counting
# against it at all. One hour: long enough that cost genuinely orders a working
# session, short enough that a gating premise cannot be starved past the point
# anyone would notice. Set to 0 to disable the guard and rank on raw cost.
COST_PATIENCE_S = 3600.0


def relative_costs(
    node_ids: list[str],
    observed: dict[str, float],
    group_of: dict[str, str] | None = None,
    declared: dict[str, float] | None = None,
) -> dict[str, float]:
    """What each probe costs, relative to the workspace median.

    Nothing in the engine knew that experiments cost different amounts. Thompson
    Sampling ranks a three-GPU-day question exactly as it ranks a one-second one,
    because the posterior is the only thing it reads — and every metric in every
    gate to date is counted in *probes*, which is defensible only because the
    eval oracle answers in milliseconds. In real R&D the spread is four orders of
    magnitude inside one project, and a navigator indifferent to that will
    confidently spend the week's compute answering the cheapest question last.

    **Precedence: this node's own timings, then the caller's estimate, then the
    median of its exclusion-group siblings, then the workspace median.** A node
    with nothing to go on costs **1.0**, the median, so a workspace that reports
    neither a duration nor an estimate ranks exactly as it did before this
    existed. That equivalence is the property worth protecting.

    The estimate sits above the sibling median rather than below it because the
    sibling fallback is blind precisely where the saving is. Competing answers to
    one question are settled *once*, so at the moment the navigator chooses
    between them none has been timed and every one inherits the identical
    number — no order, no saving. And within a question is the only place cost
    can be saved at all: ordering across questions changes nothing, since every
    question must be settled either way, whereas the last survivor of a closed
    question is *deduced* rather than probed, so whichever answer is left
    unprobed is never paid for. Ordering cheap-first puts the expensive answer in
    that free slot.

    Consuming an unmeasured estimate is safe here in a way an accuracy prior is
    not: cost changes only what is tried next, never what the belief state
    asserts, so the worst a bad guess can do is produce a worse order — and the
    first real timing overrides it.
    """
    declared = declared or {}
    if not observed and not declared:
        return {nid: 1.0 for nid in node_ids}
    # The median is the scale everything is expressed against, so it is taken
    # over both sources: a workspace that has only ever declared costs still
    # needs a denominator, and one that has timed everything is unaffected
    # because observations override declarations node by node below.
    scale = {**declared, **observed}
    median = statistics.median(scale.values())
    if median <= 0:
        return {nid: 1.0 for nid in node_ids}

    group_of = group_of or {}
    by_group: dict[str, list[float]] = {}
    for nid, cost in observed.items():
        group = group_of.get(nid)
        if group:
            by_group.setdefault(group, []).append(cost)

    costs: dict[str, float] = {}
    for nid in node_ids:
        seconds = observed.get(nid)
        if seconds is None:
            seconds = declared.get(nid)
        if seconds is None:
            siblings = by_group.get(group_of.get(nid, ""), [])
            seconds = statistics.median(siblings) if siblings else median
        ratio = seconds / median if seconds > 0 else 1.0
        costs[nid] = min(max(ratio, 1.0 / _COST_CLAMP), _COST_CLAMP)
    return costs


@dataclass
class SelectionResult:
    """The return value of select(): either a node selection or a DONE sentinel."""

    status: str  # "SELECTED" | "DONE"
    node_id: str | None = None
    claim_id: str | None = None
    credible_interval: tuple[float, float] | None = None
    rationale: str = ""
    reason: str = ""  # only set when status == "DONE"


class ThompsonSampler:
    """Thompson Sampling selection with injected seeded RNG.

    The Generator is constructor-injected so unit tests can reproduce exact
    selection sequences by fixing the seed.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        epsilon_tie: float = EPSILON_TIE,
        epsilon_ci: float = EPSILON_CI,
        n_max: int = N_MAX_SAMPLES,
        verify_threshold: float = VERIFY_THRESHOLD,
        epsilon_low: float = EPSILON_LOW,
        lease_ttl_s: int = DEFAULT_LEASE_TTL_S,
        cost_patience_s: float = COST_PATIENCE_S,
    ) -> None:
        self._rng = rng
        self._epsilon_tie = epsilon_tie
        self._epsilon_ci = epsilon_ci
        self._n_max = n_max
        self._verify_threshold = verify_threshold
        self._epsilon_low = epsilon_low
        self._lease_ttl_s = lease_ttl_s
        self._cost_patience_s = cost_patience_s

    def select(
        self,
        frontier_nodes: list[Node],
        all_nodes: list[Node],
        now: datetime | None = None,
        last_group: str | None = None,
        priority_ids: set[str] | None = None,
        all_goals_met: bool = False,
        eliminated_ids: set[str] | None = None,
        costs: dict[str, float] | None = None,
    ) -> SelectionResult:
        """Run the full selection procedure on the current frontier.

        1. Check all-goals-met / empty-frontier → DONE sentinel.
        2. Restrict to conflict suspects when any of them is selectable.
        3. Draw theta from each node's closed-world posterior.
        4. Lexicographic epsilon-tiebreak: same exclusion group first, then stalest.
        5. Issue a claim_id and return.

        ``priority_ids`` are nodes implicated in an unresolved conflict. When any
        of them is selectable the frontier narrows to exactly those: an open
        conflict means the belief state holds a known contradiction, and every
        hypothesis resting on the disputed assumptions is blocked until it is
        settled — so no other probe can be worth more. This is the same principle
        as conflict-driven branching in a SAT solver: decide on the variables the
        last conflict implicated rather than on whatever happens to look fresh.

        ``last_group`` is the exclusion group of the previous selection. Among
        genuinely tied draws it keeps the dispatch sequence on the question
        already in progress instead of hopping to an unrelated one.

        ``all_goals_met`` is supplied by the caller rather than computed here.
        Whether a goal has been reached depends on which hypotheses support it,
        which is a property of the graph; the sampler only sees posteriors, and a
        goal's own posterior is not evidence that its objective was met.
        """
        if now is None:
            now = utcnow()

        # Step 1: DONE sentinels — all goals met, or frontier is empty.
        if all_goals_met:
            return SelectionResult(status="DONE", reason="all_goals_met")
        if not frontier_nodes:
            return SelectionResult(status="DONE", reason="empty_frontier")

        # Step 2: conflict-driven narrowing.
        if priority_ids:
            suspects = [n for n in frontier_nodes if n.id in priority_ids]
            if suspects:
                frontier_nodes = suspects

        # Step 3: draw theta for each frontier node under the closed-world prior.
        live_counts = live_group_counts(all_nodes, eliminated_ids)
        thetas = {node.id: self._draw_theta(node, live_counts) for node in frontier_nodes}

        # Rank on expected value **per unit cost**. With no costs supplied every
        # ratio is exactly 1.0, so the ordering, the RNG consumption and the
        # epsilon tiebreak are identical to what they were before cost existed —
        # which is the property that makes this safe to turn on by default only
        # once it has been measured.
        costs = costs or {}
        scored = [
            (thetas[node.id] / self._effective_cost(costs.get(node.id, 1.0), node, now), node)
            for node in frontier_nodes
        ]

        # Step 4: lexicographic sort — primary: score desc, tiebreak: same
        # exclusion group as the last pick, then stalest.
        # Report the theta that actually drove the pick — do NOT draw again,
        # which would waste RNG state and misreport the selection rationale.
        _, best = self._pick_best(scored, now, last_group)
        best_theta = thetas[best.id]
        ci = credible_interval(best.alpha, best.beta)

        cost = costs.get(best.id, 1.0)
        rationale = f"theta={best_theta:.4f}"
        if cost != 1.0:
            effective = self._effective_cost(cost, best, now)
            rationale += f", cost={cost:.3g}x median, value/cost={best_theta / effective:.4f}"
            if abs(effective - cost) > 1e-9:
                rationale += f" (cost weight decayed to {effective:.3g}x after waiting)"

        return SelectionResult(
            status="SELECTED",
            node_id=best.id,
            claim_id=uuid.uuid4().hex,
            credible_interval=ci,
            rationale=rationale,
        )

    def _draw_theta(self, node: Node, live_counts: dict[str, int] | None = None) -> float:
        """Draw one sample from the node's closed-world posterior."""
        alpha, beta = effective_posterior(node, live_counts or {})
        return float(self._rng.beta(alpha, beta))

    def _effective_cost(self, cost: float, node: Node, now: datetime) -> float:
        """The cost divisor, with its weight decayed by how long the node has waited.

        Dividing by cost defers an expensive node whenever a cheaper one keeps
        looking marginally better — and a premise that gates the whole graph
        still has to be run. Nothing else fixes this: the cheap candidates are
        genuinely better value each time they are compared, so the expensive one
        loses every comparison it is ever in, forever, and the search stalls one
        probe short of its goal with the navigator confidently doing arithmetic.

        The weight therefore decays with waiting time: at zero wait the full
        ratio applies, at ``cost_patience_s`` it is exactly 1.0 and the node is
        ranked on posterior alone as if cost had never existed. Interpolating the
        *exponent* rather than the ratio keeps it monotone and scale-free — a
        100x node and a 2x node relax at the same rate relative to their own
        penalty, instead of the expensive one taking fifty times as long to be
        forgiven for being expensive.

        Waiting time is the right signal and needs no new state: ``updated_at``
        is the last time anything happened to this node, so for one that has sat
        untouched it is exactly how long it has been passed over. This is the
        same quantity the epsilon tiebreak already uses for anti-starvation, put
        to the same purpose one level up.
        """
        if cost == 1.0 or self._cost_patience_s <= 0:
            return cost
        waited = max(0.0, (now - node.updated_at).total_seconds())
        damping = min(1.0, waited / self._cost_patience_s)
        return float(cost ** (1.0 - damping))

    def _pick_best(
        self,
        scored: list[tuple[float, Node]],
        now: datetime,
        last_group: str | None = None,
    ) -> tuple[float, Node]:
        """Sort by theta desc; break epsilon-ties coherently.

        Among nodes whose draws are within ``epsilon_tie``, prefer one answering
        the same question as the previous selection (same exclusion group), then
        the stalest. Returns the (theta, node) pair actually chosen so the caller
        can report the driving theta without drawing a second sample.
        """
        # Sort by theta descending.
        scored.sort(key=lambda t: t[0], reverse=True)
        top_theta = scored[0][0]

        # Collect all nodes within epsilon_tie of the top theta.
        tied = [(theta, node) for theta, node in scored if top_theta - theta <= self._epsilon_tie]
        return max(
            tied,
            key=lambda t: (
                bool(last_group) and t[1].exclusion_group == last_group,
                (now - t[1].updated_at).total_seconds(),
            ),
        )

    # -- transition helpers (used by the engine on record_evidence) ------------

    def should_verify(
        self,
        node: Node,
        evidence_count: int,
        last_success: float | None = None,
    ) -> bool:
        """Check whether a node should auto-transition to VERIFIED.

        Deterministic: verify when the observed success exceeds the bar. The
        posterior mean (dragged down by the Beta(1,1) prior) is misleading for
        one-shot deterministic evidence — comparing the raw success value is
        the correct test. Falls back to posterior_mean when last_success is
        unavailable (e.g. direct API calls without evidence).

        Stochastic: verify when posterior mean > verify_bar AND the convergence
        gate passes (enough evidence that the credible interval is tight).
        """
        verify_bar = self._verify_bar(node)

        if node.evidence_regime == "deterministic" and last_success is not None:
            if last_success <= verify_bar:
                return False
            return convergence_gate(
                node.evidence_regime,
                evidence_count,
                node.alpha,
                node.beta,
                self._epsilon_ci,
                self._n_max,
            )

        mean = posterior_mean(node.alpha, node.beta)
        if mean <= verify_bar:
            return False
        return convergence_gate(
            node.evidence_regime,
            evidence_count,
            node.alpha,
            node.beta,
            self._epsilon_ci,
            self._n_max,
        )

    def should_invalidate(
        self, node: Node, evidence_count: int, last_success: float | None = None
    ) -> bool:
        """Check whether a node should auto-transition to INVALIDATED.

        Deterministic: a single success=0.0 invalidates immediately. The engine
        passes the last evidence's success value via last_success so we can check
        the exact reading rather than inferring from the posterior mean (which
        after one failure from Beta(1,1) is 0.333 — not near zero).

        Stochastic: only when confidently bad — convergence gate passes AND
        posterior mean is below epsilon_low.

        Goal nodes are never invalidated by evidence, for the same reason they
        are never exhausted: a goal states an objective to reach, not a claim
        that could turn out false. A sub-target reading means "not there yet".
        Refuting it would cascade-prune the very subtree meant to achieve it and
        leave the run with no goal to satisfy — which is exactly what happened
        when agents recorded a failed composition's result against the goal node.
        """
        if node.is_goal:
            return False

        if node.evidence_regime == "deterministic":
            # A deterministic node invalidates on a single zero-result.
            return last_success is not None and last_success == 0.0

        # Stochastic: need convergence AND confident-bad.
        converged = convergence_gate(
            node.evidence_regime,
            evidence_count,
            node.alpha,
            node.beta,
            self._epsilon_ci,
            self._n_max,
        )
        if not converged:
            return False
        mean = posterior_mean(node.alpha, node.beta)
        return mean < self._epsilon_low

    def should_exhaust(
        self, node: Node, evidence_count: int, last_success: float | None = None
    ) -> bool:
        """Check whether a node should auto-transition to EXHAUSTED.

        A node is exhausted when its evidence is *conclusive* but clears neither
        the verify bar nor the invalidation test — the "dead zone" that would
        otherwise leave the node IN_PROGRESS forever, permanently on the
        frontier, and re-selected by the navigator on every dispatch even though
        no further probe can change what is known about it.

        Deterministic: one logical observation IS the whole truth, so any
        reading in ``0 < success <= verify_bar`` settles the node. Callers apply
        this only after should_invalidate and should_verify have both declined,
        so the exact-zero and above-bar cases never reach here.

        Stochastic: only once the convergence gate has closed (the credible
        interval is tight, or n_max samples are in). Before that, more evidence
        genuinely can move the posterior, so the node must stay selectable.

        Goal nodes are never exhausted. A goal states an objective to be reached,
        not a claim to be settled: a sub-target reading means "not there yet",
        not "nothing more to learn". Retiring it would make the objective
        permanently unreachable and strand the run with no goal to satisfy.
        """
        if last_success is None or node.is_goal:
            return False
        if node.evidence_regime == "deterministic":
            return True
        return convergence_gate(
            node.evidence_regime,
            evidence_count,
            node.alpha,
            node.beta,
            self._epsilon_ci,
            self._n_max,
        )

    # -- internal: verify bar -------------------------------------------------

    def _verify_bar(self, node: Node) -> float:
        """The posterior-mean bar a node must clear to verify.

        Goal nodes use their own target_metric; every other node (and a goal
        that never declared a target) falls back to the global threshold. A
        goal without a target_metric would otherwise compare against None.
        """
        if node.is_goal and node.target_metric is not None:
            return node.target_metric
        return self._verify_threshold
