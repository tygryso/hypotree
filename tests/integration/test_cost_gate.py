"""Tests for the cost-weighted landscape and the `P8d-COST` falsifier that scores it.

`P8d-COST` shipped with a falsifier nobody could run: every gate counts *probes*,
which is defensible only because the oracle answers in uniform milliseconds — so
`theta/cost` and `theta` induce the same order and the claimed saving was not
merely unmet but unobservable. These tests protect the instrument that fixed
that, and they protect it from the two ways it could flatter the mechanism:
a tariff correlated with correctness, and a scorer that passes on one criterion.
"""

from __future__ import annotations

import pytest

from eval.cost_gate import MAX_PROBE_INCREASE, MIN_COST_REDUCTION, score
from eval.environment.landscape_scoring import (
    AXES,
    COMBINATION_COST,
    VALUES_PER_AXIS,
    axis_value_costs,
    optimal_strategy_cost,
    probe_cost,
    reference_strategy_cost,
    winning_config,
    winning_values,
)
from eval.runner.config import TASK_SEEDS
from eval.runner.engine_selfplay import SelfPlayResult


def _result(seed: int, probes: int, cost: float, solved: bool = True) -> SelfPlayResult:
    return SelfPlayResult(
        seed=seed,
        probes=probes,
        reference=18,
        solved=solved,
        end_reason="solved" if solved else "empty_frontier",
        cost=cost,
        reference_cost=100.0,
        optimal_cost=20.0,
    )


@pytest.mark.integration
def test_the_tariff_is_deterministic_and_covers_every_answer() -> None:
    """A landscape that changed between the two arms would measure nothing."""
    for seed in (1201, 1215, 1230):
        first, second = axis_value_costs(seed), axis_value_costs(seed)
        assert first == second
        assert set(first) == set(AXES)
        for axis in AXES:
            assert len(first[axis]) == VALUES_PER_AXIS
            assert all(c > 0 for c in first[axis].values())


@pytest.mark.integration
def test_every_axis_gets_the_same_spread_of_costs() -> None:
    """The tiers are permuted, never resampled, so no axis is cheap by luck."""
    tariff = axis_value_costs(1207)
    spreads = {tuple(sorted(values.values())) for values in tariff.values()}
    assert len(spreads) == 1, "each axis must carry the same multiset of costs"


@pytest.mark.integration
def test_being_expensive_says_nothing_about_being_right() -> None:
    """The property that makes the measurement honest rather than circular.

    If the cheap answer were usually the winner, the cost-aware arm would look
    good for finding the answer sooner rather than for deferring the expensive
    probe into the free deduction slot — a different claim, and not the one
    `P8d-COST` makes.
    """
    dearest = 0
    total = 0
    for seed in TASK_SEEDS:
        tariff = axis_value_costs(seed)
        for axis, value in winning_values(seed).items():
            total += 1
            if tariff[axis][value] == max(tariff[axis].values()):
                dearest += 1
    chance = 1 / VALUES_PER_AXIS
    assert abs(dearest / total - chance) < 0.10, (
        f"the winner is the dearest answer {dearest / total:.0%} of the time "
        f"against a {chance:.0%} chance baseline — the tariff leaks the answer"
    )


@pytest.mark.integration
def test_a_premise_probe_costs_its_own_answer_and_a_combination_costs_a_flat_rate() -> None:
    tariff = axis_value_costs(1204)
    assert probe_cost("component=v2", 1204) == tariff["component"]["v2"]
    assert probe_cost(winning_config(1204), 1204) == COMBINATION_COST


@pytest.mark.integration
def test_probing_cheapest_first_is_never_dearer_than_the_declared_order() -> None:
    """The headroom the falsifier is scored against has to actually exist."""
    for seed in TASK_SEEDS:
        assert optimal_strategy_cost(seed) <= reference_strategy_cost(seed)
    blind = sum(reference_strategy_cost(s) for s in TASK_SEEDS)
    cheap = sum(optimal_strategy_cost(s) for s in TASK_SEEDS)
    assert (blind - cheap) / blind > MIN_COST_REDUCTION, (
        "the landscape offers less headroom than the falsifier demands, so the "
        "threshold could not be met however good the navigator was"
    )


@pytest.mark.integration
def test_the_gate_confirms_a_real_saving() -> None:
    blind = [_result(s, 16, 100.0) for s in (1, 2, 3)]
    aware = [_result(s, 16, 50.0) for s in (1, 2, 3)]
    report = score(blind, aware)
    assert report["decision"] == "CONFIRMED"
    assert report["criteria"]["cost_reduction"]["value"] == pytest.approx(0.5)


@pytest.mark.integration
def test_the_gate_refuses_a_saving_bought_with_probes() -> None:
    """Halving the bill by spending far more cheap probes has moved a number, not helped."""
    blind = [_result(s, 10, 100.0) for s in (1, 2, 3)]
    aware = [_result(s, 30, 40.0) for s in (1, 2, 3)]
    report = score(blind, aware)
    assert report["decision"] == "FALSIFIED"
    assert report["criteria"]["cost_reduction"]["passed"]
    assert not report["criteria"]["probe_discipline"]["passed"]
    assert report["criteria"]["probe_discipline"]["value"] > MAX_PROBE_INCREASE


@pytest.mark.integration
def test_the_gate_refuses_a_cheaper_search_that_stops_solving() -> None:
    blind = [_result(s, 16, 100.0) for s in (1, 2, 3)]
    aware = [_result(s, 16, 10.0, solved=s != 2) for s in (1, 2, 3)]
    report = score(blind, aware)
    assert report["decision"] == "FALSIFIED"
    assert not report["criteria"]["still_solves"]["passed"]


@pytest.mark.integration
def test_the_gate_pairs_by_seed_and_ignores_unmatched_episodes() -> None:
    """Comparing arm means over unmatched sets would report the luck of the draw."""
    blind = [_result(s, 16, 100.0) for s in (1, 2, 3)]
    aware = [_result(s, 16, 50.0) for s in (2, 3, 99)]
    report = score(blind, aware)
    assert report["episodes"] == 2, "seed 99 has no pair and must not be scored"


@pytest.mark.integration
def test_the_gate_reports_how_much_of_the_achievable_saving_was_captured() -> None:
    """Scoring against the floor is what separates 'it works' from 'it is optimal'."""
    blind = [_result(s, 16, 100.0) for s in (1, 2)]
    aware = [_result(s, 16, 60.0) for s in (1, 2)]
    report = score(blind, aware)
    # 40 of the 80 available (100 -> 20) was taken.
    assert report["paired"]["headroom_captured"] == pytest.approx(0.5)
