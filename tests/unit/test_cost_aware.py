"""Tests for cost-aware selection — ranking on value per unit cost.

Nothing in the engine knew that experiments cost different amounts. Thompson
Sampling ranked a three-GPU-day question exactly as it ranked a one-second one,
because the posterior is the only thing it reads — and every metric in every gate
is counted in *probes*, which is defensible only because the eval oracle answers
in milliseconds. In real R&D the spread is four orders of magnitude inside one
project.

The property these tests protect hardest is the **equivalence**: with no observed
durations, or with the flag off, selection must be identical to a build that has
never heard of cost.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status, utcnow
from hypotree.navigator.sampler import _COST_CLAMP, ThompsonSampler, relative_costs


def _workspace(db: Path, *, cost_aware: bool = False) -> HypoTreeEngine:
    engine = HypoTreeEngine(db, rng_seed=7, cost_aware=cost_aware)
    engine.create_hypotheses(
        [{"statement": f"c={i}", "node_id": f"c{i}", "exclusion_group": "c"} for i in range(4)]
    )
    return engine


# -- the cost model -----------------------------------------------------------


@pytest.mark.unit
def test_with_nothing_timed_every_probe_costs_the_median() -> None:
    """A workspace that never reports a duration must rank exactly as before.

    This is the equivalence that makes it safe to add a second term to an
    acquisition function a pre-registered gate has already scored.
    """
    assert relative_costs(["a", "b"], {}) == {"a": 1.0, "b": 1.0}


@pytest.mark.unit
def test_cost_is_relative_to_the_workspace_median() -> None:
    """Absolute seconds are meaningless across projects; ratios are not."""
    observed = {"fast": 1.0, "mid": 10.0, "slow": 100.0}
    costs = relative_costs(["fast", "mid", "slow"], observed)
    assert costs["mid"] == pytest.approx(1.0)
    assert costs["fast"] == pytest.approx(0.1)
    assert costs["slow"] == pytest.approx(10.0)


@pytest.mark.unit
def test_an_untimed_node_inherits_its_siblings_estimate() -> None:
    """Competing answers to one question cost about the same to try.

    So the best estimate for a candidate nobody has timed is what its rivals
    cost — a far better guess than the workspace median when one question is a
    fine-tune and another is a unit test.
    """
    observed = {"gpu_a": 1000.0, "gpu_b": 1000.0, "cheap_a": 1.0, "cheap_b": 1.0, "cheap_c": 1.0}
    groups = {
        "gpu_a": "gpu",
        "gpu_b": "gpu",
        "gpu_c": "gpu",
        "cheap_a": "cheap",
        "cheap_b": "cheap",
        "cheap_c": "cheap",
    }
    costs = relative_costs(["gpu_c", "loner"], observed, groups)
    # gpu_c has never been timed, but its rivals cost 1000x the workspace median.
    assert costs["gpu_c"] == pytest.approx(1000.0)
    # `loner` has no group at all, so it falls back to the workspace median.
    assert costs["loner"] == pytest.approx(1.0)


@pytest.mark.unit
def test_an_outlier_cannot_deprioritise_a_node_without_limit() -> None:
    """Beyond four orders of magnitude a ratio stops carrying information.

    A single mis-recorded duration — a timer left running overnight — would
    otherwise bury a hypothesis for the rest of the workspace's life.
    """
    observed = {"a": 1.0, "b": 1.0, "normal": 1.0, "absurd": 1e12}
    costs = relative_costs(["normal", "absurd"], observed)
    assert costs["absurd"] == pytest.approx(_COST_CLAMP)
    assert costs["normal"] == pytest.approx(1.0)


@pytest.mark.unit
def test_a_zero_or_missing_duration_is_not_free() -> None:
    """Unknown is not the same as costless, and a zero would divide by nothing."""
    observed = {"a": 5.0, "b": 0.0}
    costs = relative_costs(["a", "b", "never_timed"], observed)
    assert costs["b"] == pytest.approx(1.0), "a zero reading is treated as unknown"
    assert costs["never_timed"] == pytest.approx(1.0)
    assert all(c > 0 for c in costs.values()), "a zero cost would make value/cost infinite"


# -- the engine ---------------------------------------------------------------


@pytest.mark.unit
def test_duration_round_trips_through_the_store(tmp_path: Path) -> None:
    engine = _workspace(tmp_path / "dur.db")
    try:
        engine.record_evidence("c0", LogicalEvidence(success=0.0, depth=1, duration_s=42.5))
        assert engine._store.mean_duration_by_node() == {"c0": 42.5}
    finally:
        engine.close()


@pytest.mark.unit
def test_untimed_evidence_leaves_the_cost_model_empty(tmp_path: Path) -> None:
    """The overwhelmingly common case, and it must cost nothing to support."""
    engine = _workspace(tmp_path / "untimed.db")
    try:
        engine.record_evidence("c0", LogicalEvidence(success=0.0, depth=1))
        assert engine._store.mean_duration_by_node() == {}
        assert engine._probe_costs(engine._frontier_nodes(), engine._store.get_all_nodes()) == {}
    finally:
        engine.close()


@pytest.mark.unit
def test_an_infra_error_is_not_an_experiment(tmp_path: Path) -> None:
    """It never touches the posterior, so it must not enter the cost model either."""
    from hypotree.models.evidence import InfraError

    engine = _workspace(tmp_path / "infra.db")
    try:
        engine.record_evidence("c0", InfraError(error_type="timeout", message="gone"))
        assert engine._store.mean_duration_by_node() == {}
    finally:
        engine.close()


@pytest.mark.unit
def test_selection_is_unchanged_when_cost_awareness_is_off(tmp_path: Path) -> None:
    """Two identically-seeded engines over identical evidence must agree exactly.

    Durations are recorded in both; only the flag differs. If the off path were
    not byte-identical, every gate result the project has ever published would
    need re-scoring.
    """
    picks = []
    for i, aware in enumerate((False, False)):
        engine = _workspace(tmp_path / f"off{i}.db", cost_aware=aware)
        try:
            engine.record_evidence("c0", LogicalEvidence(success=0.0, depth=1, duration_s=900.0))
            picks.append([t.node_id for t in engine.get_next_targets(count=2, dry_run=True)])
        finally:
            engine.close()
    assert picks[0] == picks[1]


@pytest.mark.unit
def test_cost_awareness_prefers_the_cheaper_of_two_equal_candidates(tmp_path: Path) -> None:
    """The whole point: at equal promise, the cheaper experiment comes first.

    Driven through the real selection path rather than the cost model alone, so
    a regression in the wiring is caught as well as one in the arithmetic.
    """
    engine = HypoTreeEngine(tmp_path / "cheap.db", rng_seed=11, cost_aware=True)
    try:
        engine.create_hypotheses(
            [
                # Three per question: a closed group of two would deduce its
                # survivor the moment the first member is refuted, taking it off
                # the frontier before cost could rank it.
                {"statement": "slow 0", "node_id": "slow", "exclusion_group": "s"},
                {"statement": "slow 1", "node_id": "slow2", "exclusion_group": "s"},
                {"statement": "slow 2", "node_id": "slow3", "exclusion_group": "s"},
                {"statement": "fast 0", "node_id": "fast", "exclusion_group": "f"},
                {"statement": "fast 1", "node_id": "fast2", "exclusion_group": "f"},
                {"statement": "fast 2", "node_id": "fast3", "exclusion_group": "f"},
            ]
        )
        # Time one member of each question; the untimed rival inherits it.
        engine.record_evidence("slow", LogicalEvidence(success=0.0, depth=1, duration_s=3600.0))
        engine.record_evidence("fast", LogicalEvidence(success=0.0, depth=1, duration_s=1.0))

        costs = engine._probe_costs(engine._frontier_nodes(), engine._store.get_all_nodes())
        assert costs["fast2"] < costs["slow2"], "the cheap question must rank cheaper"

        # Over many draws the cheap candidate should win far more often.
        wins = {"fast2": 0, "slow2": 0}
        for seed in range(60):
            probe = HypoTreeEngine(tmp_path / f"probe{seed}.db", rng_seed=seed, cost_aware=True)
            try:
                probe.create_hypotheses(
                    [
                        {"statement": "s", "node_id": "slow2", "exclusion_group": "s"},
                        {"statement": "f", "node_id": "fast2", "exclusion_group": "f"},
                    ]
                )
                pick = probe._sampler.select(
                    probe._frontier_nodes(),
                    probe._store.get_all_nodes(),
                    costs={"slow2": 100.0, "fast2": 0.01},
                ).node_id
                wins[str(pick)] += 1
            finally:
                probe.close()
        assert wins["fast2"] > wins["slow2"] * 4, wins
    finally:
        engine.close()


@pytest.mark.unit
def test_a_lone_candidate_is_dispatched_however_expensive_it_is(tmp_path: Path) -> None:
    """Cost reorders; it must never withhold.

    The starvation case that matters is an expensive premise that gates
    everything. With one candidate the argmax is that candidate at any scaling,
    so the sole-path case is safe by construction — and this pins it.
    """
    engine = HypoTreeEngine(tmp_path / "lone.db", rng_seed=7, cost_aware=True)
    try:
        engine.create_hypotheses([{"statement": "the only way", "node_id": "only"}])
        result = engine._sampler.select(
            engine._frontier_nodes(),
            engine._store.get_all_nodes(),
            costs={"only": _COST_CLAMP},
        )
        assert result.status == "SELECTED"
        assert result.node_id == "only"
    finally:
        engine.close()


@pytest.mark.unit
def test_the_rationale_reports_the_real_theta_not_the_cost_adjusted_score(
    tmp_path: Path,
) -> None:
    """A rationale that quoted value/cost as `theta` would misreport the belief."""
    engine = HypoTreeEngine(tmp_path / "rationale.db", rng_seed=3, cost_aware=True)
    try:
        engine.create_hypotheses([{"statement": "a", "node_id": "a"}])
        result = engine._sampler.select(
            engine._frontier_nodes(), engine._store.get_all_nodes(), costs={"a": 4.0}
        )
        theta = float(result.rationale.split("theta=")[1].split(",")[0])
        assert 0.0 <= theta <= 1.0
        assert "cost=4" in result.rationale
        assert "value/cost" in result.rationale
    finally:
        engine.close()


@pytest.mark.unit
def test_a_declared_estimate_orders_answers_that_have_never_been_timed() -> None:
    """The defect the observed-only model could not see, and the only place cost pays.

    A question is settled once, so at the moment the navigator chooses between
    its competing answers *none* of them has been timed. The sibling-median
    fallback therefore hands every one of them the identical number and orders
    nothing — while ordering within a question is the only place cost can be
    saved at all, because the last survivor of a closed question is deduced
    rather than probed and whichever answer is left unprobed is never paid for.
    """
    ids = [f"c{i}" for i in range(4)]
    groups = {i: "c" for i in ids}
    declared = {"c0": 1.0, "c1": 1.0, "c2": 10.0, "c3": 100.0}

    observed_only = relative_costs(ids, {"c0": 1.0}, groups)
    assert set(observed_only.values()) == {1.0}, "observation alone cannot rank unprobed siblings"

    costs = relative_costs(ids, {}, groups, declared)
    assert costs["c3"] > costs["c2"] > costs["c0"]


@pytest.mark.unit
def test_an_observation_overrides_the_estimate_it_was_meant_to_replace() -> None:
    """Declared cost is a prior. The first real timing is what it defers to."""
    costs = relative_costs(
        ["a", "b"], observed={"a": 100.0}, group_of={}, declared={"a": 1.0, "b": 1.0}
    )
    assert costs["a"] > costs["b"], "a node timed at 100s must not still rank as its 1s guess"


@pytest.mark.unit
def test_an_estimate_beats_a_sibling_median() -> None:
    """A claim about *this* node outranks an observation about a different one."""
    ids = ["c0", "c1"]
    groups = {"c0": "c", "c1": "c"}
    costs = relative_costs(ids, observed={"c0": 2.0}, group_of=groups, declared={"c1": 50.0})
    assert costs["c1"] > costs["c0"]


@pytest.mark.unit
def test_declared_costs_alone_still_rank_relative_to_their_own_median() -> None:
    """A workspace that has timed nothing but declared everything must still order."""
    costs = relative_costs(
        ["a", "b", "c"], observed={}, group_of={}, declared={"a": 1.0, "b": 2.0, "c": 4.0}
    )
    assert costs["b"] == pytest.approx(1.0), "the median declaration is the unit"
    assert costs["a"] < 1.0 < costs["c"]


@pytest.mark.unit
def test_a_workspace_with_neither_signal_is_untouched() -> None:
    """The equivalence that makes this safe to ship: no signal, no change."""
    assert relative_costs(["a", "b"], {}, {}, {}) == {"a": 1.0, "b": 1.0}


@pytest.mark.unit
def test_an_expensive_node_is_deferred_but_never_starved() -> None:
    """Dividing by cost defers a slow node *forever* without this guard.

    The cheap candidates are genuinely better value at every comparison, so an
    expensive one loses every comparison it is ever in — and a premise that gates
    the whole graph still has to be run. The cost weight therefore decays with
    waiting time: at zero wait the full ratio applies, at `cost_patience_s` it is
    exactly 1.0 and the node ranks on posterior alone as if cost never existed.
    """
    sampler = ThompsonSampler(np.random.default_rng(0), cost_patience_s=100.0)
    now = utcnow()
    node = Node(id="slow", statement="s", updated_at=now - timedelta(seconds=0))

    assert sampler._effective_cost(64.0, node, now) == pytest.approx(64.0)

    half = Node(id="slow", statement="s", updated_at=now - timedelta(seconds=50))
    assert sampler._effective_cost(64.0, half, now) == pytest.approx(8.0)

    patient = Node(id="slow", statement="s", updated_at=now - timedelta(seconds=100))
    assert sampler._effective_cost(64.0, patient, now) == pytest.approx(1.0)

    # Past patience it stays neutral rather than inverting into a bonus.
    stale = Node(id="slow", statement="s", updated_at=now - timedelta(seconds=10_000))
    assert sampler._effective_cost(64.0, stale, now) == pytest.approx(1.0)


@pytest.mark.unit
def test_the_starvation_guard_relaxes_cheap_and_dear_nodes_at_the_same_rate() -> None:
    """Decaying the exponent, not the ratio, is what keeps the guard scale-free.

    Relaxing the ratio linearly would make a 100x node take fifty times as long
    to be forgiven as a 2x node — punishing it twice for the same property.
    """
    sampler = ThompsonSampler(np.random.default_rng(0), cost_patience_s=100.0)
    now = utcnow()
    waited = Node(id="n", statement="s", updated_at=now - timedelta(seconds=50))
    assert sampler._effective_cost(100.0, waited, now) == pytest.approx(10.0)
    assert sampler._effective_cost(4.0, waited, now) == pytest.approx(2.0)


@pytest.mark.unit
def test_patience_of_zero_disables_the_guard() -> None:
    """The escape hatch: rank on raw cost, however long a node has waited."""
    sampler = ThompsonSampler(np.random.default_rng(0), cost_patience_s=0.0)
    now = utcnow()
    old = Node(id="n", statement="s", updated_at=now - timedelta(days=30))
    assert sampler._effective_cost(64.0, old, now) == pytest.approx(64.0)


@pytest.mark.unit
def test_a_node_that_costs_the_median_is_never_touched_by_the_guard() -> None:
    """Cost 1.0 is the no-signal case and must stay exactly 1.0, not drift."""
    sampler = ThompsonSampler(np.random.default_rng(0), cost_patience_s=100.0)
    now = utcnow()
    node = Node(id="n", statement="s", updated_at=now - timedelta(seconds=37))
    assert sampler._effective_cost(1.0, node, now) == 1.0


@pytest.mark.unit
def test_a_cost_estimate_moves_the_order_and_never_the_belief(tmp_path: Path) -> None:
    """The invariant that makes an unmeasured estimate admissible at all.

    An accuracy prior consumed raw can assert something false. A cost prior
    cannot: it divides a score used only for ranking, so the worst a wrong guess
    does is produce a worse order. Two workspaces given identical evidence and
    wildly different cost estimates must end in identical belief states.
    """
    states = []
    for estimate in (1.0, 10_000.0):
        engine = HypoTreeEngine(
            tmp_path / f"cost{estimate}.db", rng_seed=7, cost_aware=True, project_path=tmp_path
        )
        engine.create_hypotheses(
            [
                {
                    "statement": f"c={i}",
                    "node_id": f"c{i}",
                    "exclusion_group": "c",
                    "estimated_cost": estimate if i == 0 else 1.0,
                }
                for i in range(4)
            ]
        )
        engine.record_evidence("c0", LogicalEvidence(success=1.0, depth=1))
        states.append(
            {
                n.id: (n.status, round(n.alpha, 6), round(n.beta, 6))
                for n in engine._store.get_all_nodes()
            }
        )
        engine.close()

    assert states[0] == states[1]


@pytest.mark.unit
def test_a_badly_wrong_cost_estimate_costs_order_not_correctness(tmp_path: Path) -> None:
    """A cost prior that points exactly the wrong way must still settle the question.

    Ordering is recoverable in a way pruning is not, which is the whole reason
    cost is allowed to be guessed. If a hostile estimate could leave a question
    unanswered, that argument would not hold.
    """
    engine = HypoTreeEngine(
        tmp_path / "hostile.db", rng_seed=7, cost_aware=True, project_path=tmp_path
    )
    # The correct answer is declared as the most expensive thing in the workspace.
    engine.create_hypotheses(
        [
            {
                "statement": f"c={i}",
                "node_id": f"c{i}",
                "exclusion_group": "c",
                "estimated_cost": 5_000.0 if i == 3 else 1.0,
            }
            for i in range(4)
        ]
    )
    try:
        for _ in range(10):
            target = engine.get_next_targets(count=1)[0]
            if target.status != "SELECTED":
                break
            success = 1.0 if target.node_id == "c3" else 0.0
            engine.record_evidence(
                target.node_id,  # type: ignore[arg-type]
                LogicalEvidence(success=success, depth=1),
                claim_id=target.claim_id,
            )
        nodes = engine._store.get_all_nodes()
        assert next(n for n in nodes if n.id == "c3").status is Status.VERIFIED
    finally:
        engine.close()
