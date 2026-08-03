"""Tests for the frozen gate analysis script.

Uses synthetic JSONL fixtures with known answers to verify the scoring logic
before any real runs execute (pre-registration §2: "blinding of analysis").

Fixtures:
- GO scenario: B beats A on 8/10 seeds with 30% median reduction + δ=0.6.
- STOP scenario: B barely beats A on 4/10 seeds (fails consistency).
- ITERATE scenario: C1+C2 pass but C3 fails (pruned_reexecutions > 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.analyse_gate import (
    _criterion2_ts_quality,
    _extract_steps_to_target,
    analyse,
    bootstrap_ci_median,
    cliffs_delta,
    paired_wilcoxon,
)
from eval.runner.config import TASK_SEEDS


def _write_runner_log(
    path: Path,
    seed: int,
    arm: str,
    steps: int,
    goals_met: bool,
    pruned_reexecutions: int = 0,
    upstream_events: int = 0,
) -> None:
    """Write a synthetic runner JSONL log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"event_type": "run_start", "step": 0, "seed": seed, "arm": arm, "tool_budget": 60},
    ]
    for i in range(steps):
        events.append(
            {
                "event_type": "experiment",
                "step": i + 1,
                "seed": seed,
                "arm": arm,
                "config": f"config_{i}",
                "depth": 0,
                "success": 0.5,
            }
        )
    events.append(
        {
            "event_type": "run_end",
            "step": steps,
            "seed": seed,
            "arm": arm,
            "goals_met": goals_met,
            "reason": "all_goals_met" if goals_met else "budget_exhausted",
        }
    )
    for _ in range(pruned_reexecutions):
        events.append(
            {
                "event_type": "pruned_reexecution",
                "step": 0,
                "seed": seed,
                "arm": arm,
                "node_id": "H005",
            }
        )
    for _ in range(upstream_events):
        events.append(
            {
                "event_type": "status_transition",
                "step": 0,
                "seed": seed,
                "arm": arm,
                "node_id": "H003",
                "old_status": "VERIFIED",
                "new_status": "NEEDS_REVISION",
                "propagated": True,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _write_ablation_log(
    path: Path,
    seed: int,
    rng_seed: int,
    ts_regret: float,
    rand_regret: float,
    greedy_regret: float,
) -> None:
    """Write a synthetic ablation result JSONL entry (cumulative regret)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for strategy, regret in [
        ("ts", ts_regret),
        ("random", rand_regret),
        ("greedy", greedy_regret),
    ]:
        entries.append(
            {
                "event_type": "ablation_result",
                "strategy": strategy,
                "seed": seed,
                "rng_seed": rng_seed,
                "cumulative_regret": regret,
                "total_pulls": 300,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# -- Statistics tests ---------------------------------------------------------


@pytest.mark.unit
def test_steps_to_target_censors_failed_run() -> None:
    """A run that never met its goal must be censored to the tool budget, so an
    early give-up cannot masquerade as the fastest solve."""
    events = [
        {"event_type": "run_start", "tool_budget": 60},
        {"event_type": "experiment", "step": 1},
        {"event_type": "experiment", "step": 2},
        {"event_type": "run_end", "step": 2, "goals_met": False},
    ]
    assert _extract_steps_to_target(events) == 60


@pytest.mark.unit
def test_steps_to_target_uses_actual_steps_on_success() -> None:
    """A successful run reports its actual final step count, not the budget."""
    events = [
        {"event_type": "run_start", "tool_budget": 60},
        {"event_type": "run_end", "step": 12, "goals_met": True},
    ]
    assert _extract_steps_to_target(events) == 12


@pytest.mark.unit
def test_cliffs_delta_perfect_separation() -> None:
    """Cliff's delta = 1.0 when treatment is always lower than control."""
    delta = cliffs_delta([1, 2, 3], [4, 5, 6])
    assert delta == 1.0


@pytest.mark.unit
def test_cliffs_delta_no_effect() -> None:
    """Cliff's delta = 0.0 when treatment equals control."""
    delta = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert delta == 0.0


@pytest.mark.unit
def test_cliffs_delta_partial() -> None:
    """Cliff's delta in (0, 1) for partial dominance."""
    delta = cliffs_delta([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert 0.0 < delta < 1.0


@pytest.mark.unit
def test_paired_wilcoxon_significant() -> None:
    """Wilcoxon detects a clear treatment effect."""
    result = paired_wilcoxon([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
    assert result["p_value"] < 0.05


@pytest.mark.unit
def test_paired_wilcoxon_no_effect() -> None:
    """Wilcoxon p-value is high when there's no effect."""
    result = paired_wilcoxon([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert result["p_value"] >= 0.05


def _runner_events(seed: int, arm: str, *, steps: int, goals_met: bool) -> list[dict]:
    """Build an in-memory runner event list (no file IO) for criterion-1 tests."""
    events: list[dict] = [
        {"event_type": "run_start", "step": 0, "seed": seed, "arm": arm, "tool_budget": 60}
    ]
    for i in range(steps):
        events.append(
            {
                "event_type": "experiment",
                "step": i + 1,
                "seed": seed,
                "arm": arm,
                "config": f"config_{i}",
                "depth": 2,
                "success": 0.5,
            }
        )
    events.append(
        {
            "event_type": "run_end",
            "step": steps,
            "seed": seed,
            "arm": arm,
            "reason": "all_goals_met" if goals_met else "budget_exhausted",
            "goals_met": goals_met,
        }
    )
    return events


# -- Criterion 2 (TS quality) unit tests --------------------------------------


def _ablation_entries(ts: list[float], rand: list[float], greedy: list[float]) -> list[dict]:
    """Build ablation_result entries from per-strategy regrets (one per seed)."""
    entries = []
    for i, (t, r, g) in enumerate(zip(ts, rand, greedy, strict=True)):
        seed = TASK_SEEDS[i]
        for strategy, regret in (("ts", t), ("random", r), ("greedy", g)):
            entries.append(
                {
                    "event_type": "ablation_result",
                    "strategy": strategy,
                    "seed": seed,
                    "rng_seed": seed + 1000,
                    "cumulative_regret": regret,
                    "total_pulls": 300,
                }
            )
    return entries


@pytest.mark.unit
def test_criterion2_passes_on_random_and_worst_case_greedy() -> None:
    """TS beats random on the median AND has a bounded worst case vs greedy."""
    n = len(TASK_SEEDS)
    n_catastrophic = max(1, n // 3)
    ts = [5.0] * n
    rand = [15.0] * n
    # greedy catastrophically locks a decoy on a minority of seeds
    greedy = [2.0] * (n - n_catastrophic) + [30.0] * n_catastrophic
    result = _criterion2_ts_quality(_ablation_entries(ts, rand, greedy))

    assert result.passed is True
    assert result.details["random_ok"] is True
    assert result.details["greedy_ok"] is True
    assert result.metrics["ts_vs_random"]["ts_wins"] == n
    assert result.metrics["ts_vs_greedy"]["worst_case_reduction"] >= 0.20
    # Greedy still wins the typical case — reported honestly, not gated.
    assert result.metrics["ts_vs_greedy"]["median_reduction_typical"] < 0


@pytest.mark.unit
def test_criterion2_fails_when_random_not_beaten() -> None:
    """If TS does not beat random on the median, the criterion fails."""
    n = len(TASK_SEEDS)
    n_catastrophic = max(1, n // 3)
    ts = [14.0] * n
    rand = [15.0] * n  # only ~7% reduction < 20% floor
    greedy = [2.0] * (n - n_catastrophic) + [30.0] * n_catastrophic
    result = _criterion2_ts_quality(_ablation_entries(ts, rand, greedy))

    assert result.passed is False
    assert result.details["random_ok"] is False


@pytest.mark.unit
def test_criterion2_fails_when_greedy_worst_case_not_beaten() -> None:
    """If greedy never catastrophically fails, TS has no worst-case edge → fail."""
    n = len(TASK_SEEDS)
    ts = [5.0] * n
    rand = [15.0] * n
    greedy = [4.0] * n  # greedy always locks the winner; worst case ~ TS's
    result = _criterion2_ts_quality(_ablation_entries(ts, rand, greedy))

    assert result.passed is False
    assert result.details["random_ok"] is True
    assert result.details["greedy_ok"] is False


@pytest.mark.unit
def test_criterion2_fails_on_insufficient_seeds() -> None:
    """Fewer than N_SEEDS common seeds → criterion fails with an error."""
    result = _criterion2_ts_quality(_ablation_entries([5.0] * 5, [15.0] * 5, [30.0] * 5))
    assert result.passed is False
    assert "error" in result.details


@pytest.mark.unit
def test_bootstrap_ci_contains_median() -> None:
    """Bootstrap CI brackets the sample median."""
    data = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 7.0, 14.0, 13.0, 6.0]
    ci_low, ci_high = bootstrap_ci_median(data, n_resamples=1000)
    median = 10.5
    assert ci_low <= median <= ci_high


# -- GO scenario test ---------------------------------------------------------


@pytest.mark.integration
def test_go_decision(tmp_path: Path) -> None:
    """Synthetic fixture where B beats A on 8/10 seeds with 30% reduction → GO."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # B beats A on 8/10 seeds with ~30% fewer steps.
    for i, seed in enumerate(TASK_SEEDS):
        steps_a = 50
        steps_b = 30 if i < int(0.8 * len(TASK_SEEDS)) else 55  # B wins 80% of seeds

        for base_arm in ("A", "F"):
            _write_runner_log(
                runs_dir / f"seed-{seed}-arm-{base_arm}.jsonl",
                seed,
                base_arm,
                steps_a,
                goals_met=True,
            )
        _write_runner_log(
            runs_dir / f"seed-{seed}-arm-B.jsonl",
            seed,
            "B",
            steps_b,
            goals_met=True,
            upstream_events=1,
        )

    # TS wins the typical case vs random on every seed; greedy catastrophically
    # locks in on 3 seeds, so its worst-case regret dwarfs TS's bounded regret.
    for i, seed in enumerate(TASK_SEEDS):
        rng_seed = seed + 1000
        greedy_regret = 30.0 if i >= int(0.7 * len(TASK_SEEDS)) else 2.0
        _write_ablation_log(
            runs_dir / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl",
            seed,
            rng_seed,
            5.0,
            15.0,
            greedy_regret,
        )

    result = analyse(runs_dir)

    assert result.decision == "GO"
    assert result.criteria["criterion_1_moat"]["passed"] is True
    assert result.criteria["criterion_2_ts_quality"]["passed"] is True
    assert result.criteria["criterion_3_revision"]["passed"] is True
    assert result.criteria["criterion_4_status_utility"]["passed"] is True
    # b_wins now lives on the two split comparisons, not the combined gate entry.
    assert result.criteria["criterion_1b_informational_moat"]["b_wins"] == int(
        0.8 * len(TASK_SEEDS)
    )
    assert result.criteria["criterion_1a_ergonomic_moat"]["passed"] is True


# -- STOP scenario test -------------------------------------------------------


@pytest.mark.integration
def test_stop_decision(tmp_path: Path) -> None:
    """Synthetic fixture where B barely beats A on 4/10 → STOP (c1 fails)."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    for i, seed in enumerate(TASK_SEEDS):
        steps_a = 50
        steps_b = 30 if i < int(0.4 * len(TASK_SEEDS)) else 55  # B wins only 40%

        for base_arm in ("A", "F"):
            _write_runner_log(
                runs_dir / f"seed-{seed}-arm-{base_arm}.jsonl",
                seed,
                base_arm,
                steps_a,
                goals_met=True,
            )
        _write_runner_log(
            runs_dir / f"seed-{seed}-arm-B.jsonl",
            seed,
            "B",
            steps_b,
            goals_met=True,
            upstream_events=1,
        )

    for seed in TASK_SEEDS:
        rng_seed = seed + 1000
        _write_ablation_log(
            runs_dir / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl",
            seed,
            rng_seed,
            5.0,
            15.0,
            30.0,
        )

    result = analyse(runs_dir)

    assert result.decision == "STOP"
    assert result.criteria["criterion_1_moat"]["passed"] is False


# -- ITERATE scenario test ----------------------------------------------------


@pytest.mark.integration
def test_iterate_decision(tmp_path: Path) -> None:
    """Synthetic fixture where c1+c2 pass but c3 fails → ITERATE."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    for seed in TASK_SEEDS:
        for base_arm in ("A", "F"):
            _write_runner_log(
                runs_dir / f"seed-{seed}-arm-{base_arm}.jsonl", seed, base_arm, 50, goals_met=True
            )
        # C3 fails: pruned_reexecutions > 0
        _write_runner_log(
            runs_dir / f"seed-{seed}-arm-B.jsonl",
            seed,
            "B",
            30,
            goals_met=True,
            pruned_reexecutions=1,
            upstream_events=1,
        )

    for seed in TASK_SEEDS:
        rng_seed = seed + 1000
        _write_ablation_log(
            runs_dir / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl",
            seed,
            rng_seed,
            5.0,
            15.0,
            30.0,
        )

    result = analyse(runs_dir)

    assert result.decision == "ITERATE"
    assert result.criteria["criterion_1_moat"]["passed"] is True
    assert result.criteria["criterion_2_ts_quality"]["passed"] is True
    assert result.criteria["criterion_3_revision"]["passed"] is False
    assert result.criteria["criterion_3_revision"]["total_pruned_reexecutions"] > 0


# -- Missing data tests -------------------------------------------------------


@pytest.mark.integration
def test_missing_seeds_fails(tmp_path: Path) -> None:
    """An incomplete seed set must fail criterion 1 rather than score a subset.

    Guards against a partially-completed run silently producing a decision from
    whichever seeds happened to finish.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Deliberately write only half the pre-registered seeds.
    for seed in TASK_SEEDS[: len(TASK_SEEDS) // 2]:
        for base_arm in ("A", "F"):
            _write_runner_log(
                runs_dir / f"seed-{seed}-arm-{base_arm}.jsonl", seed, base_arm, 50, goals_met=True
            )
        _write_runner_log(
            runs_dir / f"seed-{seed}-arm-B.jsonl",
            seed,
            "B",
            30,
            goals_met=True,
        )

    for seed in TASK_SEEDS:
        rng_seed = seed + 1000
        _write_ablation_log(
            runs_dir / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl",
            seed,
            rng_seed,
            5.0,
            15.0,
            30.0,
        )

    result = analyse(runs_dir)

    assert result.decision == "STOP"
    assert result.criteria["criterion_1_moat"]["passed"] is False


@pytest.mark.integration
def test_json_output(tmp_path: Path) -> None:
    """analyse_to_json produces valid JSON with expected structure."""
    from eval.analyse_gate import analyse_to_json

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    for seed in TASK_SEEDS:
        for base_arm in ("A", "F"):
            _write_runner_log(
                runs_dir / f"seed-{seed}-arm-{base_arm}.jsonl", seed, base_arm, 50, goals_met=True
            )
        _write_runner_log(
            runs_dir / f"seed-{seed}-arm-B.jsonl",
            seed,
            "B",
            30,
            goals_met=True,
            upstream_events=1,
        )

    for seed in TASK_SEEDS:
        rng_seed = seed + 1000
        _write_ablation_log(
            runs_dir / f"ablation-seed-{seed}-rng-{rng_seed}.jsonl",
            seed,
            rng_seed,
            5.0,
            15.0,
            30.0,
        )

    output = analyse_to_json(runs_dir)
    data = json.loads(output)
    assert "decision" in data
    assert "criteria" in data
    assert data["decision"] in ("GO", "STOP", "ITERATE")


@pytest.mark.unit
def test_load_runner_logs_ignores_retired_seeds(tmp_path: Path) -> None:
    """Logs from seeds outside the pre-registration must never enter the gate.

    The runs directory accumulates across runs. A bare glob would pool a
    superseded run (different landscape design) in with the current one and
    compute the paired statistics over a mixture of the two — silently, with no
    error. The gate is defined by the pre-registered seed set, not by whatever
    files happen to be on disk.
    """
    from eval.analyse_gate import _load_runner_logs
    from eval.runner.config import RETIRED_TASK_SEEDS, TASK_SEEDS

    current, retired = TASK_SEEDS[0], RETIRED_TASK_SEEDS[0]
    for seed in (current, retired):
        for arm in ("A", "B"):
            (tmp_path / f"seed-{seed}-arm-{arm}.jsonl").write_text(
                json.dumps({"event_type": "run_end", "reason": "x", "goals_met": False}) + "\n",
                encoding="utf-8",
            )

    logs = _load_runner_logs(tmp_path)
    assert set(logs["A"]) == {current}
    assert set(logs["B"]) == {current}
    assert retired not in logs["A"]


# -- Criterion 1 split (ergonomic vs informational) ---------------------------


@pytest.mark.integration
def test_criterion1_is_gated_on_the_informational_comparison(tmp_path: Path) -> None:
    """Beating only the discipline-dependent baseline must NOT pass criterion 1.

    This is the exact shape of the superseded result: hypotree crushed a baseline
    that never wrote a note, while being no better than one that kept its facts.
    That establishes an ergonomic advantage, not the informational one the gate
    claims to test, so the combined criterion must follow 1b.
    """
    from eval.analyse_gate import _criterion1_moat

    arm_a, arm_f, arm_b = {}, {}, {}
    for seed in TASK_SEEDS:
        arm_a[seed] = _runner_events(seed, "A", steps=50, goals_met=True)
        arm_f[seed] = _runner_events(seed, "F", steps=20, goals_met=True)
        arm_b[seed] = _runner_events(seed, "B", steps=20, goals_met=True)

    combined, c1a, c1b = _criterion1_moat(arm_a, arm_b, arm_f)

    assert c1a.passed is True, "B should beat the un-maintained scratchpad"
    assert c1b.passed is False, "B is level with the auto-preserved log"
    assert combined.passed is False
    assert combined.metrics["gated_on"] == "criterion_1b_informational_moat"


@pytest.mark.integration
def test_criterion1_passes_when_structure_itself_wins(tmp_path: Path) -> None:
    """Beating the auto-preserved baseline is what passing criterion 1 means."""
    from eval.analyse_gate import _criterion1_moat

    arm_a, arm_f, arm_b = {}, {}, {}
    for seed in TASK_SEEDS:
        arm_a[seed] = _runner_events(seed, "A", steps=50, goals_met=True)
        arm_f[seed] = _runner_events(seed, "F", steps=40, goals_met=True)
        arm_b[seed] = _runner_events(seed, "B", steps=20, goals_met=True)

    combined, c1a, c1b = _criterion1_moat(arm_a, arm_b, arm_f)

    assert c1a.passed is True
    assert c1b.passed is True
    assert combined.passed is True


# -- Criterion 3 (belief revision) --------------------------------------------


@pytest.mark.unit
def test_criterion3_counts_conflicts_as_revision() -> None:
    """Recording a conflict IS a revision: the admissible combinations shrank.

    Counting only upstream NEEDS_REVISION propagations made the criterion an
    artefact of DAG shape — a combination resting on several assumptions could
    never register unless the system blamed all of them, which destroys correct
    knowledge to satisfy a metric.
    """
    from eval.analyse_gate import _criterion3_revision

    events = _runner_events(TASK_SEEDS[0], "B", steps=5, goals_met=True)
    events.append({"event_type": "conflict_recorded", "member_ids": ["a", "b"], "n_members": 2})

    result = _criterion3_revision({TASK_SEEDS[0]: events})

    assert result.passed is True
    assert result.metrics["conflicts_recorded"] == 1
    assert result.metrics["revision_events"] == 1
    assert result.metrics["total_upstream_propagations"] == 0


@pytest.mark.unit
def test_criterion3_counts_identified_culprits() -> None:
    from eval.analyse_gate import _criterion3_revision

    events = _runner_events(TASK_SEEDS[0], "B", steps=5, goals_met=True)
    events += [
        {"event_type": "conflict_recorded", "member_ids": ["a", "b"], "n_members": 2},
        {"event_type": "conflict_resolved", "nogood_id": 1, "culprit_id": "b"},
    ]

    result = _criterion3_revision({TASK_SEEDS[0]: events})

    assert result.metrics["culprits_identified"] == 1
    assert result.metrics["revision_events"] == 2


@pytest.mark.unit
def test_criterion3_fails_without_any_revision() -> None:
    from eval.analyse_gate import _criterion3_revision

    events = _runner_events(TASK_SEEDS[0], "B", steps=5, goals_met=True)
    result = _criterion3_revision({TASK_SEEDS[0]: events})

    assert result.passed is False
    assert result.metrics["revision_events"] == 0


@pytest.mark.unit
def test_criterion3_hard_fails_on_pruned_reexecution() -> None:
    """Re-running settled work is a hard failure no amount of revision offsets."""
    from eval.analyse_gate import _criterion3_revision

    events = _runner_events(TASK_SEEDS[0], "B", steps=5, goals_met=True)
    events += [
        {"event_type": "conflict_recorded", "member_ids": ["a", "b"], "n_members": 2},
        {"event_type": "pruned_reexecution", "node_id": "a"},
    ]

    result = _criterion3_revision({TASK_SEEDS[0]: events})

    assert result.passed is False
    assert result.metrics["total_pruned_reexecutions"] == 1


# -- criterion 4: the status-utility χ² -----------------------------------


def _action_log(counts: dict[tuple[str, str], int]) -> dict[str, list[dict]]:
    """One arm-B seed whose agent_action events match the given cell counts."""
    events = [
        {"event_type": "agent_action", "action": action, "status_context": context}
        for (action, context), n in counts.items()
        for _ in range(n)
    ]
    return {"1201": events}


@pytest.mark.unit
def test_criterion4_collapses_when_one_status_regime_never_occurs() -> None:
    """With nothing under revision there is no contrast, so there is no test."""
    from eval.analyse_gate import _criterion4_status_utility

    result = _criterion4_status_utility(
        _action_log({("EXECUTE_EXPERIMENT", "open"): 80, ("REPLAN", "open"): 40})
    )
    assert result.passed is True
    assert result.metrics["decision"] == "collapse"
    assert "underpowered" in result.metrics["reason"]
    assert "chi2" not in result.metrics


@pytest.mark.unit
def test_criterion4_collapses_on_a_sample_too_small_to_mean_anything() -> None:
    """The pre-committed default is a collapse, not a re-run until significant."""
    from eval.analyse_gate import _criterion4_status_utility

    result = _criterion4_status_utility(
        _action_log(
            {
                ("EXECUTE_EXPERIMENT", "open"): 5,
                ("EXECUTE_EXPERIMENT", "revision"): 1,
                ("REPLAN", "open"): 1,
                ("REPLAN", "revision"): 5,
            }
        )
    )
    assert result.metrics["decision"] == "collapse"
    assert "underpowered" in result.metrics["reason"]


@pytest.mark.unit
def test_criterion4_collapses_when_the_statuses_change_no_behaviour() -> None:
    """Independence is a real answer, and it is the one the collapse is for.

    Distinguishable from the underpowered collapse by the p-value being present:
    this one was measured.
    """
    from eval.analyse_gate import _criterion4_status_utility

    result = _criterion4_status_utility(
        _action_log(
            {
                ("EXECUTE_EXPERIMENT", "open"): 60,
                ("EXECUTE_EXPERIMENT", "revision"): 60,
                ("REPLAN", "open"): 30,
                ("REPLAN", "revision"): 30,
                ("REQUEST_CONTEXT", "open"): 20,
                ("REQUEST_CONTEXT", "revision"): 20,
            }
        )
    )
    assert result.metrics["decision"] == "collapse"
    assert result.metrics["p_value"] > 0.05
    assert result.metrics["cramers_v"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_criterion4_keeps_the_statuses_when_they_change_what_the_agent_does() -> None:
    """The outcome the old default could never report, because it never measured."""
    from eval.analyse_gate import _criterion4_status_utility

    result = _criterion4_status_utility(
        _action_log(
            {
                ("EXECUTE_EXPERIMENT", "open"): 90,
                ("EXECUTE_EXPERIMENT", "revision"): 10,
                ("REPLAN", "open"): 10,
                ("REPLAN", "revision"): 90,
                ("REQUEST_CONTEXT", "open"): 20,
                ("REQUEST_CONTEXT", "revision"): 40,
            }
        )
    )
    assert result.passed is True
    assert result.metrics["decision"] == "keep"
    assert result.metrics["p_value"] < 0.05
    assert result.metrics["cramers_v"] > 0.3
    assert result.details["columns"] == ["open", "revision"]


@pytest.mark.unit
def test_criterion4_never_flips_the_gate_on_its_own() -> None:
    """Collapse is a legitimate design answer, not a failure to gate on."""
    from eval.analyse_gate import _criterion4_status_utility

    for log in ({}, _action_log({("NO_OP", "open"): 3})):
        assert _criterion4_status_utility(log).passed is True


@pytest.mark.integration
def test_permutation_p_value_finds_no_signal_in_an_independent_table() -> None:
    """Identical row profiles mean the context explains nothing.

    Every row here splits the same way, so the observed arrangement is exactly
    what independence predicts and no shuffle can look less extreme. The
    p-value must land near 1, not near 0.
    """
    from eval.analyse_gate import _permutation_p_value

    assert _permutation_p_value([[20, 20], [10, 10], [3, 3]]) > 0.5


@pytest.mark.integration
def test_permutation_p_value_detects_a_real_dependence() -> None:
    """A perfectly separated table is not something shuffling reproduces."""
    from eval.analyse_gate import _permutation_p_value

    assert _permutation_p_value([[40, 0], [0, 40], [2, 2]]) < 0.01


@pytest.mark.integration
def test_permutation_p_value_never_reports_impossible_certainty() -> None:
    """A finite resample cannot justify p = 0, so the estimator must not emit it."""
    from eval.analyse_gate import PERMUTATION_RESAMPLES, _permutation_p_value

    # Perfect separation: no shuffle reproduces it, so this is the floor case.
    p = _permutation_p_value([[400, 0], [0, 400]])

    assert p == pytest.approx(1 / (PERMUTATION_RESAMPLES + 1))


@pytest.mark.integration
def test_permutation_p_value_is_reproducible() -> None:
    """A gate decision is a property of the run, not of when it was read."""
    from eval.analyse_gate import _permutation_p_value

    table = [[18, 9], [7, 14], [2, 3]]

    assert _permutation_p_value(table) == _permutation_p_value(table)


@pytest.mark.unit
def test_an_infra_failed_episode_is_excluded_not_censored(tmp_path: Path) -> None:
    """An episode the inference server killed is not a result and must not score.

    Censoring assumes the agent had its full budget and did not solve the task.
    An episode that died on a dropped HTTP connection had neither, so scoring it
    at budget charges that arm for the server's bad minute. Dropping it costs one
    paired seed; keeping it corrupts the comparison.
    """
    from eval.analyse_gate import _load_runner_logs
    from eval.runner.config import TASK_SEEDS

    healthy, killed = TASK_SEEDS[0], TASK_SEEDS[1]
    for seed in (healthy, killed):
        for arm in ("A", "B", "F"):
            infra = seed == killed and arm == "B"
            (tmp_path / f"seed-{seed}-arm-{arm}.jsonl").write_text(
                json.dumps(
                    {
                        "event_type": "run_end",
                        "step": 3,
                        "reason": "llm_unavailable" if infra else "all_goals_met",
                        "goals_met": not infra,
                        "infra_failed": infra,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    logs = _load_runner_logs(tmp_path)
    assert set(logs["B"]) == {healthy}, "the killed episode must not be scored"
    # The other arms on that seed are untouched — only the dead episode drops out.
    assert set(logs["A"]) == {healthy, killed}
    assert set(logs["F"]) == {healthy, killed}
