"""Tests for the TS-quality ablation navigator.

Verifies the cumulative-regret bandit pipeline: bandit construction, the three
selection strategies, determinism, and logging. It does NOT re-assert the gate
decision (that is analyse_gate.py's job) — but it does lock in the bandit
calibration that makes TS win the typical case vs random and the worst case vs
greedy, so a regression in the arm rates is caught here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from eval.runner.ablation_navigator import (
    ABLATION_HORIZON,
    DECOY_RATE,
    N_DECOYS,
    WINNER_RATE,
    AblationResult,
    _empirical_mean,
    _populate_bandit,
    _run_strategy,
    log_ablation_result,
    run_ablation,
)
from eval.runner.config import make_ablation_config
from hypotree.engine import HypoTreeEngine
from hypotree.models.evidence import LogicalEvidence
from hypotree.navigator.sampler import ThompsonSampler


def _make_engine(rng_seed: int = 2001) -> HypoTreeEngine:
    """Fresh in-memory engine with a non-converging sampler.

    The tight epsilon_ci + above-horizon n_max match run_ablation, so no arm
    ever verifies and the frontier stays fixed for the whole horizon.
    """
    engine = HypoTreeEngine(":memory:", rng_seed=rng_seed)
    engine._sampler = ThompsonSampler(
        np.random.default_rng(rng_seed), epsilon_ci=0.001, n_max=ABLATION_HORIZON + 1
    )
    return engine


# -- Bandit construction ------------------------------------------------------


@pytest.mark.unit
def test_populate_bandit_builds_winner_and_decoys() -> None:
    """The bandit is one winner + N_DECOYS decoys, all reachable from a verified
    root gateway."""
    engine = _make_engine()
    arm_rates = _populate_bandit(engine)

    assert arm_rates["arm_winner"] == WINNER_RATE
    decoys = [k for k in arm_rates if k.startswith("arm_decoy_")]
    assert len(decoys) == N_DECOYS
    assert all(arm_rates[d] == DECOY_RATE for d in decoys)
    # The winner is the unique optimum.
    assert max(arm_rates.values()) == WINNER_RATE
    assert WINNER_RATE > DECOY_RATE

    # Root is verified (not itself an arm) → every arm is in the frontier.
    frontier = engine._frontier_nodes()
    assert len(frontier) == N_DECOYS + 1
    assert "root" not in {n.id for n in frontier}
    engine.close()


# -- Empirical mean helper ----------------------------------------------------


@pytest.mark.unit
def test_empirical_mean_none_before_any_pull() -> None:
    """An unpulled arm has no defined empirical mean (round-robin greedy signal)."""
    engine = _make_engine()
    _populate_bandit(engine)
    assert all(_empirical_mean(n) is None for n in engine._frontier_nodes())
    engine.close()


@pytest.mark.unit
def test_empirical_mean_tracks_observations() -> None:
    """After k successes and m failures the empirical mean is k / (k + m)."""
    engine = _make_engine()
    _populate_bandit(engine)
    for i, outcome in enumerate((1.0, 1.0, 1.0, 0.0)):
        claim_id = f"c-{i}"
        engine._store.create_claim(claim_id, "arm_winner", datetime.now(), 900)
        engine.record_evidence("arm_winner", LogicalEvidence(success=outcome), claim_id=claim_id)
        engine._sync_graph_from_store()

    node = next(n for n in engine._frontier_nodes() if n.id == "arm_winner")
    assert _empirical_mean(node) == pytest.approx(0.75)
    engine.close()


# -- Strategy execution -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["ts", "random", "greedy"])
def test_run_strategy_produces_wellformed_regret(strategy: str) -> None:
    """Every strategy spends the full horizon (no arm verifies) and returns a
    regret bounded by horizon x the winner-decoy gap."""
    engine = _make_engine()
    arm_rates = _populate_bandit(engine)
    horizon = 30
    result = _run_strategy(
        engine,
        arm_rates,
        strategy,
        seed=1001,
        rng_seed=2001,
        horizon=horizon,
        noise_rng=np.random.default_rng(2001 + 100_003),
        select_rng=np.random.default_rng(2001 + 200_003),
    )

    assert result.strategy == strategy
    assert result.seed == 1001
    assert result.rng_seed == 2001
    assert result.total_pulls == horizon
    assert 0.0 <= result.cumulative_regret <= horizon * (WINNER_RATE - DECOY_RATE) + 1e-9
    engine.close()


@pytest.mark.unit
def test_unknown_strategy_raises() -> None:
    """An unknown strategy name raises ValueError."""
    engine = _make_engine()
    arm_rates = _populate_bandit(engine)
    with pytest.raises(ValueError, match="unknown strategy"):
        _run_strategy(
            engine,
            arm_rates,
            "invalid",
            seed=1001,
            rng_seed=2001,
            horizon=5,
            noise_rng=np.random.default_rng(1),
            select_rng=np.random.default_rng(2),
        )
    engine.close()


# -- Full ablation pipeline ---------------------------------------------------


@pytest.mark.integration
def test_run_ablation_produces_three_results(tmp_path: Path) -> None:
    """run_ablation returns one full-horizon result per strategy."""
    results = run_ablation(make_ablation_config(1001, 2001, tmp_path, "test-run"))

    assert len(results) == 3
    assert {r.strategy for r in results} == {"ts", "random", "greedy"}
    for r in results:
        assert r.seed == 1001
        assert r.rng_seed == 2001
        assert r.total_pulls == ABLATION_HORIZON
        assert r.cumulative_regret >= 0.0


@pytest.mark.integration
def test_ablation_results_are_deterministic(tmp_path: Path) -> None:
    """Same seed + rng_seed produce identical regret across runs."""
    results1 = run_ablation(make_ablation_config(1001, 2001, tmp_path, "test-run"))
    results2 = run_ablation(make_ablation_config(1001, 2001, tmp_path, "test-run"))

    for r1, r2 in zip(results1, results2, strict=True):
        assert r1.strategy == r2.strategy
        assert r1.cumulative_regret == r2.cumulative_regret


@pytest.mark.integration
def test_bandit_calibration_favours_ts(tmp_path: Path) -> None:
    """Regression guard on the arm rates: across the pre-registered seeds, TS
    must decisively beat random on the median AND have a worst case materially
    below greedy's — the two properties criterion 2 gates on."""
    from eval.runner.config import NAVIGATOR_RNG_SEEDS, TASK_SEEDS

    ts, rand, greedy = [], [], []
    for seed, rng_seed in zip(TASK_SEEDS, NAVIGATOR_RNG_SEEDS, strict=True):
        by_strategy = {
            r.strategy: r.cumulative_regret
            for r in run_ablation(make_ablation_config(seed, rng_seed, tmp_path, "test-run"))
        }
        ts.append(by_strategy["ts"])
        rand.append(by_strategy["random"])
        greedy.append(by_strategy["greedy"])
    ts_a, rand_a, greedy_a = np.array(ts), np.array(rand), np.array(greedy)

    # Typical-case superiority over random.
    assert (np.median(rand_a) - np.median(ts_a)) / np.median(rand_a) >= 0.20
    assert int((ts_a < rand_a).sum()) >= 6

    # Worst-case superiority over greedy (no catastrophic lock-in).
    assert (greedy_a.max() - ts_a.max()) / greedy_a.max() >= 0.20
    # Greedy DOES catastrophically lock in on at least one seed (its worst case
    # is far above TS's) — the phenomenon the guardrail exists to catch.
    assert greedy_a.max() > 2 * float(np.median(ts_a))


# -- Logging ------------------------------------------------------------------


@pytest.mark.unit
def test_log_ablation_result_writes_jsonl(tmp_path: Path) -> None:
    """log_ablation_result appends a well-formed JSONL entry."""
    log_path = tmp_path / "ablation_log.jsonl"
    result = AblationResult(
        strategy="ts",
        seed=1001,
        rng_seed=2001,
        cumulative_regret=42.5,
        total_pulls=300,
    )
    log_ablation_result(result, log_path)

    entry = json.loads(log_path.read_text().strip())
    assert entry["event_type"] == "ablation_result"
    assert entry["strategy"] == "ts"
    assert entry["seed"] == 1001
    assert entry["rng_seed"] == 2001
    assert entry["cumulative_regret"] == 42.5
    assert entry["total_pulls"] == 300
