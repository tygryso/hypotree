"""Tests for the black-box evaluation environment.

Covers landscape generation, the HTTP server, the eval client, and the
briefing generator. Uses tmp_path for isolation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.unit
def test_landscape_generator_produces_one_config_per_seed(tmp_path: Path) -> None:
    """Generator produces exactly one landscape JSON per pre-registered seed."""
    from eval.environment.landscape_generator import generate_all
    from eval.runner.config import TASK_SEEDS

    paths = generate_all(tmp_path)
    assert len(paths) == len(TASK_SEEDS)
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["seed"] in TASK_SEEDS
        assert data["total_nodes"] >= 40
        assert len(data["session_breakpoints"]) >= 3
        assert data["decoy_node_id"] in data["node_truth"]
        assert any(n.get("is_goal") for n in data["nodes"])


@pytest.mark.unit
def test_first_breakpoint_is_below_the_reference_difficulty_floor(tmp_path: Path) -> None:
    """No seed may be solvable before the first forced context reset.

    This is the analytic difficulty calibration: the canonical strategy the
    briefing describes needs at least ``min_reference_probes()`` probes, and the
    first breakpoint sits strictly below that. Difficulty is therefore proven by
    arithmetic rather than discovered empirically after a run has been spent —
    the failure mode of the superseded landscape, where several seeds turned out
    to be solvable inside the first session and diluted the whole comparison.
    """
    from eval.environment.landscape_generator import generate_all
    from eval.environment.landscape_scoring import min_reference_probes, reference_strategy_probes

    floor = min_reference_probes()
    for p in generate_all(tmp_path):
        data = json.loads(p.read_text(encoding="utf-8"))
        first_reset = data["session_breakpoints"][0]
        assert first_reset < floor, (data["seed"], first_reset, floor)
        # The per-seed cost must never dip below the global floor.
        assert reference_strategy_probes(data["seed"]) >= floor
        assert data["reference_strategy_probes"] > first_reset


@pytest.mark.unit
def test_landscape_plants_a_decoy_that_only_integration_reveals() -> None:
    """Each landscape carries one value that passes alone and fails assembled.

    This is the planted contradiction the pre-registration requires: an early
    hypothesis the agent will accept, later refuted by downstream evidence. It
    is what makes belief revision measurable at all — without it no confirmed
    hypothesis is ever contradicted and upstream propagation can never fire.
    """
    from eval.environment.landscape_generator import _generate_dag
    from eval.environment.landscape_scoring import (
        MIN_CONFIRM_DEPTH,
        PREMISE_CONFIRMED,
        TARGET_METRIC,
        decoy_config,
        score_config,
    )
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        d = _generate_dag(seed)
        axis, value = d["decoy_axis"], d["decoy_value"]
        # The decoy is a genuine second confirmation on its axis.
        assert value != d["winning_values"][axis], seed
        assert score_config(f"{axis}={value}", seed, 0) == PREMISE_CONFIRMED, seed
        # Deep premise probing must NOT expose it — only assembly can.
        assert score_config(f"{axis}={value}", seed, 4) == PREMISE_CONFIRMED, seed
        # Assembled, it mirages above the target shallow and hard-fails at depth.
        cfg = decoy_config(seed)
        assert score_config(cfg, seed, 0) > TARGET_METRIC, seed
        assert score_config(cfg, seed, MIN_CONFIRM_DEPTH) == 0.0, seed


@pytest.mark.unit
def test_landscape_has_goal_node(tmp_path: Path) -> None:
    """Each landscape has exactly one is_goal node with target_metric."""
    from eval.environment.landscape_generator import _generate_dag
    from eval.runner.config import TASK_SEEDS

    config = _generate_dag(TASK_SEEDS[2])
    goal_nodes = [n for n in config["nodes"] if n.get("is_goal")]
    assert len(goal_nodes) == 1
    assert goal_nodes[0]["target_metric"] is not None


@pytest.mark.unit
def test_winning_combination_is_premise_wired_and_verifiable() -> None:
    """The winning combination must depend on exactly its four correct premises.

    This is the DAG's semantic content: a combination is wired by DEPENDENCY to
    every premise it assumes, so refuting a premise legitimately prunes the
    combinations built on it. Without that wiring the graph is inert scaffolding
    and cascading prune has nothing meaningful to act on.
    """
    from eval.environment.landscape_generator import _generate_dag
    from eval.environment.landscape_scoring import AXES, winning_values
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        d = _generate_dag(seed)
        nt = d["node_truth"]
        by_id = {n["id"]: n for n in d["nodes"]}
        win_id = d["winning_node_id"]
        wv = winning_values(seed)

        # The winning combination clears the strict 0.8 verify bar.
        assert nt[win_id]["true_success"] > 0.8, (seed, win_id)

        # Its DEPENDENCY parents are exactly the four correct premise nodes.
        dep_parents = {
            e["src"] for e in d["edges"] if e["dst"] == win_id and e["type"] == "DEPENDENCY"
        }
        assert len(dep_parents) == len(AXES), (seed, dep_parents)
        for pid in dep_parents:
            truth = nt[pid]
            assert truth.get("premise") is True, (seed, pid)
            assert truth["premise_confirms"] is True, (seed, pid)
            assert wv[truth["premise_axis"]] == truth["premise_value"]

        # The goal hangs off the winning combination by DEPENDENCY and verifies.
        goal_id = d["goal_node_id"]
        assert {(e["src"], e["type"]) for e in d["edges"] if e["dst"] == goal_id} == {
            (win_id, "DEPENDENCY")
        }, seed
        assert nt[goal_id]["true_success"] > 0.8, seed

        # The decoy trap must not be the winning combination.
        assert d["decoy_node_id"] != win_id, seed
        assert by_id[win_id]["statement"] == d["winning_config"]


@pytest.mark.unit
def test_refuted_premise_prunes_only_dependent_combinations() -> None:
    """Every wrong premise value has dependents, and the winner is never among them.

    Guards the property that makes cascading prune *correct* here: pruning the
    subtree of a refuted premise removes combinations that genuinely cannot win,
    and never removes the actual answer.
    """
    from eval.environment.landscape_generator import _generate_dag
    from eval.environment.landscape_scoring import PREMISE_REFUTED
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        d = _generate_dag(seed)
        nt = d["node_truth"]
        win_id = d["winning_node_id"]
        dependents: dict[str, set[str]] = {}
        for e in d["edges"]:
            if e["type"] == "DEPENDENCY":
                dependents.setdefault(e["src"], set()).add(e["dst"])

        for nid, truth in nt.items():
            if not truth.get("premise") or truth["premise_confirms"]:
                continue
            # A non-confirming premise scores an exact zero — conclusive refutation.
            assert truth["true_success"] == PREMISE_REFUTED, (seed, nid)
            # Refuting it can never take out the winning combination.
            assert win_id not in dependents.get(nid, set()), (seed, nid)


@pytest.mark.unit
def test_score_is_deterministic() -> None:
    """Same config + seed always produces the same score."""
    from eval.environment.landscape_scoring import score_config, winning_config
    from eval.runner.config import TASK_SEEDS

    seed = TASK_SEEDS[0]
    cfg = winning_config(seed)
    assert score_config(cfg, seed) == score_config(cfg, seed)
    assert 0.0 <= score_config(cfg, seed) <= 1.0


@pytest.mark.unit
def test_only_the_full_combination_clears_the_target() -> None:
    """Dropping any single axis must push the score below the goal target.

    If a 3-of-4 combination could clear the target the task would be winnable
    without ever resolving every axis, which is what keeps the search long enough
    to span a context reset. Evaluated at confirm depth, since that is the only
    depth at which a result counts.
    """
    from eval.environment.landscape_scoring import (
        AXES,
        MIN_CONFIRM_DEPTH,
        TARGET_METRIC,
        VALUES_PER_AXIS,
        confirming_values,
        score_config,
        winning_config,
        winning_values,
    )
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        wv = winning_values(seed)
        confirming = confirming_values(seed)
        assert score_config(winning_config(seed), seed, MIN_CONFIRM_DEPTH) > TARGET_METRIC, seed
        for dropped in AXES:
            # Corrupt exactly one axis to a value that does NOT confirm on it,
            # keeping the rest correct. Swapping in the decoy is covered by its
            # own test — that case collapses to a hard zero, not a near miss.
            broken = dict(wv)
            for offset in range(1, VALUES_PER_AXIS):
                candidate = f"v{(int(wv[dropped][1:]) + offset) % VALUES_PER_AXIS}"
                if candidate not in confirming[dropped]:
                    broken[dropped] = candidate
                    break
            cfg = ";".join(f"{a}={broken[a]}" for a in AXES)
            assert score_config(cfg, seed, MIN_CONFIRM_DEPTH) < TARGET_METRIC, (seed, dropped)


@pytest.mark.unit
def test_synergy_pair_has_no_individual_signal() -> None:
    """The landscape is non-separable: a lone synergy axis earns nothing.

    This is what defeats the naive one-factor-at-a-time sweep that trivially
    solved the superseded additive landscape.
    """
    from eval.environment.landscape_scoring import (
        AXES,
        SEPARABLE_AXES,
        SYNERGY_PAIR,
        VALUES_PER_AXIS,
        score_config,
        winning_values,
    )
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        wv = winning_values(seed)
        wrong = {a: f"v{(int(wv[a][1:]) + 1) % VALUES_PER_AXIS}" for a in AXES}

        def _cfg(overrides: dict[str, str], _base: dict[str, str] = wrong) -> str:
            combo = dict(_base)
            combo.update(overrides)
            return ";".join(f"{a}={combo[a]}" for a in AXES)

        none_right = score_config(_cfg({}), seed)
        first_only = score_config(_cfg({SYNERGY_PAIR[0]: wv[SYNERGY_PAIR[0]]}), seed)
        second_only = score_config(_cfg({SYNERGY_PAIR[1]: wv[SYNERGY_PAIR[1]]}), seed)
        both = score_config(_cfg({a: wv[a] for a in SYNERGY_PAIR}), seed)
        one_separable = score_config(_cfg({SEPARABLE_AXES[0]: wv[SEPARABLE_AXES[0]]}), seed)

        # Half the pair is worth exactly as much as none of it (modulo jitter).
        assert abs(first_only - none_right) < 0.02, seed
        assert abs(second_only - none_right) < 0.02, seed
        # The pair pays more than any single separable axis, so a sweep cannot
        # reach the answer by banking separable credit and ignoring the
        # interaction. Stated against the other rewards rather than an absolute
        # number, so the property survives a recalibration of the weights.
        assert both > one_separable, seed


@pytest.mark.unit
def test_premise_probe_resolves_every_axis_value() -> None:
    """A single-axis probe confirms or refutes with exactly 1.0 or 0.0.

    The exact zero makes a deterministic node INVALIDATE, driving the cascading
    prune; the 1.0 clears the verify bar so a confirmed premise becomes VERIFIED
    and its dependent combinations become reachable. Anything in between would
    strand every premise in the dead zone and leave the combination layer of the
    DAG permanently unreachable — which is exactly what happened when confirmed
    premises returned a mid-band 0.5.
    """
    from eval.environment.landscape_scoring import (
        AXES,
        PREMISE_CONFIRMED,
        PREMISE_REFUTED,
        axis_values,
        confirming_values,
        score_config,
    )
    from eval.runner.config import TASK_SEEDS

    av = axis_values()
    for seed in TASK_SEEDS:
        confirming = confirming_values(seed)
        for axis in AXES:
            for value in av[axis]:
                score = score_config(f"{axis}={value}", seed)
                expected = PREMISE_CONFIRMED if value in confirming[axis] else PREMISE_REFUTED
                assert score == expected, (seed, axis, value)
        # Exactly one axis carries two confirming values — the decoy ambiguity.
        ambiguous = [a for a in AXES if len(confirming[a]) > 1]
        assert len(ambiguous) == 1, (seed, ambiguous)


@pytest.mark.unit
def test_briefing_generation(tmp_path: Path) -> None:
    """Briefing document is generated and contains key elements."""
    from eval.environment.fake_hypothesis_tree import generate_briefing
    from eval.environment.landscape_generator import generate_all
    from eval.runner.config import TASK_SEEDS

    generate_all(tmp_path)
    landscape_path = tmp_path / f"landscape_seed_{TASK_SEEDS[0]}.json"
    briefing = generate_briefing(landscape_path)
    assert "R&D Task" in briefing
    assert "constraints" in briefing.lower()
    assert "agent_eval_client.py" in briefing
    # The briefing must teach the two probe modes and the interaction, since the
    # task is unsolvable by full-combination sweeping alone.
    assert "Premise probe" in briefing
    assert "interact" in briefing.lower()


@pytest.mark.unit
def test_briefing_never_leaks_the_winning_combination(tmp_path: Path) -> None:
    """The briefing describes the search space, never the answer.

    The superseded briefing listed every DAG node's statement, which included the
    winning configuration verbatim — both arms could simply read the answer, so
    no cross-session search was ever required and the moat went untested. This
    guards that regression for every seed.
    """
    import json as _json

    from eval.environment.fake_hypothesis_tree import generate_briefing
    from eval.environment.landscape_generator import generate_all
    from eval.environment.landscape_scoring import AXES

    for path in generate_all(tmp_path):
        data = _json.loads(path.read_text(encoding="utf-8"))
        briefing = generate_briefing(path)
        assert data["winning_config"] not in briefing, data["seed"]
        assert data["decoy_config"] not in briefing, data["seed"]
        # The decoy value must not be identifiable from the briefing either.
        assert f"{data['decoy_axis']}={data['decoy_value']}" not in briefing, data["seed"]
        # Not even a single correct axis assignment may appear as a pair.
        for axis, value in data["winning_values"].items():
            assert briefing.count(f"{axis}={value}") == 0, (data["seed"], axis)
        # The axis names and the candidate menu must still be present.
        for axis in AXES:
            assert axis in briefing


@pytest.mark.integration
def test_landscape_server_evaluate(tmp_path: Path) -> None:
    """Integration test: start server, probe /evaluate, verify response shape.

    Starts the landscape server as a subprocess, probes it with a known
    config, and checks the response contains success ∈ [0,1].
    """
    from eval.environment.landscape_generator import _generate_dag

    # Write a landscape config
    config = _generate_dag(1001)
    config_path = tmp_path / "landscape.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Pick a known node config to probe
    first_node = config["nodes"][1]
    probe_config = first_node["statement"]

    # Start the server
    port = 8099  # Use a non-default port to avoid conflicts
    proc = subprocess.Popen(
        [sys.executable, "eval/environment/landscape_server.py", str(config_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for server to start
        time.sleep(1.0)

        # Probe the health endpoint
        from eval.environment.agent_eval_client import evaluate

        result = evaluate(probe_config, 0, url=f"http://127.0.0.1:{port}/evaluate")
        assert "success" in result
        assert 0.0 <= result["success"] <= 1.0
        assert "metrics" in result
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.integration
def test_landscape_server_arbitrary_config(tmp_path: Path) -> None:
    """Arbitrary config strings get meaningful similarity-based scores."""
    from eval.environment.landscape_generator import _generate_dag

    config = _generate_dag(1001)
    config_path = tmp_path / "landscape.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    port = 8097
    proc = subprocess.Popen(
        [sys.executable, "eval/environment/landscape_server.py", str(config_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.0)
        from eval.environment.agent_eval_client import evaluate

        # Probe the winning combination at confirm depth — should clear the target.
        peak = config["winning_config"]
        result_peak = evaluate(peak, 2, url=f"http://127.0.0.1:{port}/evaluate")
        assert result_peak["success"] > 0.75

        # Probe a garbage config — should be low
        result_garbage = evaluate(
            "totally_unrelated_garbage_xyz", 0, url=f"http://127.0.0.1:{port}/evaluate"
        )
        assert result_garbage["success"] < result_peak["success"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.integration
def test_landscape_server_decoy_reveal(tmp_path: Path) -> None:
    """The decoy scores full marks shallow and hard-fails at depth, over HTTP.

    End-to-end proof that the planted contradiction survives the transport layer:
    this is the probe pair that drives the whole belief-revision cycle.
    """
    from eval.environment.landscape_generator import _generate_dag

    config = _generate_dag(1001)
    config_path = tmp_path / "landscape.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # The decoy combination is the trap.
    ambush_id = config["decoy_node_id"]
    ambush_node = None
    for n in config["nodes"]:
        if n["id"] == ambush_id:
            ambush_node = n
            break
    assert ambush_node is not None
    decoy_cfg = ambush_node["statement"]

    port = 8098
    proc = subprocess.Popen(
        [sys.executable, "eval/environment/landscape_server.py", str(config_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.0)
        from eval.environment.agent_eval_client import evaluate

        # Shallow: indistinguishable from a genuine winner.
        result_shallow = evaluate(decoy_cfg, 0, url=f"http://127.0.0.1:{port}/evaluate")
        assert result_shallow["success"] > 0.75

        # At confirm depth: exposed as a hard integration failure.
        result_deep = evaluate(decoy_cfg, 2, url=f"http://127.0.0.1:{port}/evaluate")
        assert result_deep["success"] < 0.75
        assert result_deep["success"] < result_shallow["success"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.unit
def test_naive_one_factor_sweep_cannot_solve_any_seed() -> None:
    """The strategy that trivially beat the superseded landscape must now fail.

    The previous purely-additive landscape was solved by a fixed ~10-probe sweep
    over full configurations, so the task never required a search long enough to
    span a context reset and the two arms could not be told apart. Here the
    synergy pair yields *identical* scores for every value until its partner is
    also correct, so a sweep has nothing to climb.
    """
    from eval.environment.landscape_scoring import (
        AXES,
        TARGET_METRIC,
        axis_values,
        score_config,
    )
    from eval.runner.config import TASK_SEEDS

    av = axis_values()
    for seed in TASK_SEEDS:
        current = {a: av[a][0] for a in AXES}
        solved = False
        for axis in AXES:
            best_value, best_score = current[axis], -1.0
            for value in av[axis]:
                trial = dict(current)
                trial[axis] = value
                score = score_config(";".join(f"{a}={trial[a]}" for a in AXES), seed)
                if score > TARGET_METRIC:
                    solved = True
                if score > best_score:
                    best_score, best_value = score, value
            current[axis] = best_value
        final = score_config(";".join(f"{a}={current[a]}" for a in AXES), seed)
        assert not solved and final < TARGET_METRIC, seed


@pytest.mark.unit
def test_reference_strategy_solves_every_seed_at_the_predicted_cost() -> None:
    """The briefed strategy must work, and cost exactly what calibration claims.

    Difficulty is only meaningful if the canonical strategy actually succeeds —
    otherwise the task is merely impossible rather than hard — and the predicted
    probe count is what the session breakpoints are calibrated against. This is
    an independent replay: if it and ``reference_strategy_probes`` ever disagree,
    the number the breakpoints rest on is fiction.
    """
    from eval.environment.landscape_scoring import (
        AXES,
        MIN_CONFIRM_DEPTH,
        TARGET_METRIC,
        VALUES_PER_AXIS,
        axis_values,
        decoy_axis,
        reference_strategy_probes,
        score_config,
    )
    from eval.runner.config import TASK_SEEDS

    av = axis_values()
    for seed in TASK_SEEDS:
        probes = 0
        chosen: dict[str, str] = {}
        remaining: dict[str, list[str]] = {}

        # 1. Premise-probe each axis until something confirms.
        for axis in AXES:
            candidates = av[axis]
            for i, value in enumerate(candidates):
                if i == VALUES_PER_AXIS - 1:
                    # Last survivor is implied by elimination — no probe needed.
                    chosen[axis] = value
                    remaining[axis] = []
                    break
                probes += 1
                if score_config(f"{axis}={value}", seed) > 0:
                    chosen[axis] = value
                    remaining[axis] = candidates[i + 1 :]
                    break

        # 2. Assemble and confirm at depth.
        assembled = ";".join(f"{a}={chosen[a]}" for a in AXES)
        probes += 1
        score = score_config(assembled, seed, MIN_CONFIRM_DEPTH)

        # 3. A hard zero means a confirmed premise failed in integration.
        if score == 0.0:
            axis = decoy_axis(seed)
            for i, value in enumerate(remaining[axis]):
                if i == len(remaining[axis]) - 1:
                    chosen[axis] = value
                    break
                probes += 1
                if score_config(f"{axis}={value}", seed) > 0:
                    chosen[axis] = value
                    break
            assembled = ";".join(f"{a}={chosen[a]}" for a in AXES)
            probes += 1
            score = score_config(assembled, seed, MIN_CONFIRM_DEPTH)

        assert score > TARGET_METRIC, (seed, assembled)
        assert probes == reference_strategy_probes(seed), seed


@pytest.mark.integration
def test_breakpoints_scale_with_the_seed_reference_cost() -> None:
    """Reset pressure must track the task, not a fixed step index.

    With absolute breakpoints a faster arm crossed fewer resets than a slower
    one, so improving the treatment arm quietly reduced its own exposure to the
    cross-session persistence the gate exists to measure. Scaling to each seed's
    reference cost keeps every arm under comparable pressure.
    """
    from eval.environment.landscape_generator import (
        SESSION_BREAKPOINT_FRACTIONS,
        session_breakpoints,
    )
    from eval.environment.landscape_scoring import reference_strategy_probes
    from eval.runner.config import TASK_SEEDS

    for seed in TASK_SEEDS:
        reference = reference_strategy_probes(seed)
        points = session_breakpoints(seed)
        assert points == sorted(points), "breakpoints must be increasing"
        assert len(points) == len(set(points)), "breakpoints must be distinct"
        assert points[0] < reference, "the first reset must land before the task can be solved"
        assert points[-1] >= reference, "the last reset must bite a slower-than-perfect agent"
        assert points[-1] == round(max(SESSION_BREAKPOINT_FRACTIONS) * reference)


@pytest.mark.integration
def test_every_seed_still_needs_a_reset_before_it_can_be_solved() -> None:
    """The analytic difficulty guarantee, restated against the scaled breakpoints."""
    from eval.environment.landscape_generator import session_breakpoints
    from eval.environment.landscape_scoring import min_reference_probes
    from eval.runner.config import TASK_SEEDS

    floor = min_reference_probes()
    for seed in TASK_SEEDS:
        assert session_breakpoints(seed)[0] < floor
