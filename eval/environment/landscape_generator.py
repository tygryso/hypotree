"""Generate the held-out landscape configurations for the evaluation gate.

Each landscape is a premise-gated, epistatic, decoy-bearing environment (see
:mod:`eval.environment.landscape_scoring` for the scoring model). The DAG mirrors
the task's logical structure so the engine's revision machinery has something
real to act on:

- **Premise nodes**, one per (axis, value), hanging off the root. The values of
  an axis share an **exclusion group**: they are competing answers to one
  question, of which exactly one holds. Confirming one lets the engine settle the
  rest by inference instead of spending a probe on each.
- **Combination nodes**, each a DEPENDENCY child of the premises it assumes.
  Refuting a premise therefore prunes precisely the combinations it voids, and a
  combination only becomes reachable once its premises are confirmed.
- **A decoy premise** that confirms in isolation but poisons every combination it
  appears in. Assembling it and confirming at depth returns a hard zero, which
  invalidates the combination and propagates back up to the premise that was
  wrongly confirmed — the belief-revision cycle the gate is meant to measure.
- Session breakpoints chosen **analytically**: the first sits strictly below
  ``min_reference_probes()``, proving no seed is solvable before the first
  context reset.

The landscape file holds the hidden ground truth. The agent only ever sees the
briefing (axes, candidate values, the rules of the game — never the answer) and
the live scores returned by the landscape server.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.environment.landscape_scoring import (
    AXES,
    EVIDENCE_REGIME,
    MIN_CONFIRM_DEPTH,
    SEPARABLE_AXES,
    SYNERGY_PAIR,
    TARGET_METRIC,
    axis_values,
    confirming_values,
    decoy_axis,
    decoy_config,
    decoy_value,
    min_reference_probes,
    reference_strategy_probes,
    score_config,
    winning_config,
    winning_values,
)

# Domain templates for the briefing documents. Pure flavour — the abstract axes
# are identical across domains and the domain never leaks which values are right.
_DOMAINS = [
    ("composite_material", "thermal conductivity optimization"),
    ("drug_discovery", "binding affinity maximization"),
    ("algorithm_tuning", "throughput latency minimization"),
    ("sensor_calibration", "signal-to-noise ratio improvement"),
    ("protein_engineering", "enzyme catalytic efficiency"),
    ("rocket_propellant", "specific impulse optimization"),
    ("solar_cell", "photovoltaic conversion efficiency"),
    ("battery_chemistry", "energy density maximization"),
    ("antenna_design", "radiation pattern optimization"),
    ("catalyst_screening", "reaction yield maximization"),
]

# Experiment budget per run, identical across arms.
#
# Generous relative to the ~18-probe reference cost, because the budget is a
# censoring mechanism: an arm that hits it has its steps_to_target truncated to
# the budget, which compresses exactly the differences the gate measures. The
# weakest arm needs roughly three times the reference cost on the wider
# landscape, so the ceiling is set well above that rather than close to it.
TOOL_BUDGET = 100

# Session-reset breakpoints, expressed as fractions of the seed's own reference
# probe count rather than as absolute step indices.
#
# Absolute breakpoints silently decouple the memory test from the task: an arm
# that finishes in 13 steps crossed one reset while an arm that took 18 crossed
# the same one, so the gate measured performance inside roughly a single session
# and the cross-session persistence it exists to test barely applied. Worse, the
# coupling ran the wrong way — every improvement to the treatment arm reduced its
# own exposure to the very mechanism under test.
#
# Fractions fix both: the first reset lands well before any seed can be solved
# (proved by the pre-flight assert against min_reference_probes), and a faster
# arm still crosses the same resets a slower one does.
#
# Three points rather than four: at these reference costs a fourth reset fires
# roughly every third probe, and the cost of re-orienting starts to dominate
# what the reset is meant to measure. Three still gives every arm triple the
# exposure the old absolute breakpoints did.
SESSION_BREAKPOINT_FRACTIONS = (0.5, 0.9, 1.3)


def session_breakpoints(seed: int) -> list[int]:
    """Reset points for one seed, scaled to its reference strategy cost.

    The first point is additionally clamped below ``min_reference_probes()``,
    which is what makes "no seed is solvable before the first context reset" a
    property of the code rather than an accident of the current fractions. The
    fraction alone does not guarantee it: it scales with each seed's *own* cost,
    so the most expensive seed can push its first reset past the cheapest seed's
    solution point, and widening the landscape is exactly what makes that spread
    large enough to matter.
    """
    reference = reference_strategy_probes(seed)
    points = sorted({max(1, round(f * reference)) for f in SESSION_BREAKPOINT_FRACTIONS})
    points[0] = min(points[0], min_reference_probes() - 1)
    return points


# Minimum DAG size required by the pre-registration (the tree must not fit in
# the agent's context in one piece).
MIN_NODES = 40


def _generate_dag(seed: int) -> dict:
    """Generate a full landscape configuration for one seed."""
    domain_key, domain_desc = _DOMAINS[seed % len(_DOMAINS)]

    av = axis_values()
    wv = winning_values(seed)
    win_cfg = winning_config(seed)
    dec_cfg = decoy_config(seed)
    dec_axis, dec_value = decoy_axis(seed), decoy_value(seed)
    confirming = confirming_values(seed)

    nodes: list[dict] = []
    edges: list[dict] = []
    node_truth: dict[str, dict] = {}
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        nid = f"H{counter:03d}"
        counter += 1
        return nid

    # Root — a verified gateway, not a testable hypothesis.
    root_id = _next_id()
    nodes.append(
        {
            "id": root_id,
            "statement": f"Root: explore the {domain_desc} configuration space",
            "layer": 0,
        }
    )
    node_truth[root_id] = {"true_success": 1.0}

    # Premise layer: one node per (axis, value), grouped by axis for mutual
    # exclusion. These are the atomic claims a premise probe tests.
    premise_id: dict[tuple[str, str], str] = {}
    for axis in AXES:
        for value in av[axis]:
            nid = _next_id()
            statement = f"{axis}={value}"
            premise_id[(axis, value)] = nid
            nodes.append(
                {
                    "id": nid,
                    "statement": statement,
                    "layer": 1,
                    "premise_axis": axis,
                    "exclusion_group": axis,
                }
            )
            edges.append({"src": root_id, "dst": nid, "type": "ALTERNATIVE"})
            node_truth[nid] = {
                "true_success": score_config(statement, seed),
                "premise": True,
                "premise_axis": axis,
                "premise_value": value,
                "premise_confirms": value in confirming[axis],
                "is_decoy": axis == dec_axis and value == dec_value,
            }

    def _add_combination(combo: dict[str, str], layer: int) -> str:
        """Add a candidate combination wired to the premises it assumes.

        DEPENDENCY edges from every premise the combination rests on: that is the
        logical content of the graph. If any of those premises is refuted this
        combination is void, and if the combination itself fails the refutation
        travels back up to the premises that were trusted.
        """
        statement = ";".join(f"{a}={combo[a]}" for a in AXES)
        nid = _next_id()
        nodes.append({"id": nid, "statement": statement, "layer": layer})
        for axis in AXES:
            edges.append({"src": premise_id[(axis, combo[axis])], "dst": nid, "type": "DEPENDENCY"})
        node_truth[nid] = {
            "true_success": score_config(statement, seed, MIN_CONFIRM_DEPTH),
            "shallow_success": score_config(statement, seed, 0),
        }
        return nid

    winning_combo_id = _add_combination(dict(wv), layer=2)

    # The decoy combination: full marks at shallow depth, hard zero at depth.
    decoy_combo = dict(wv)
    decoy_combo[dec_axis] = dec_value
    decoy_combo_id = _add_combination(decoy_combo, layer=2)
    node_truth[decoy_combo_id]["decoy_trap"] = True

    # Distractor combinations, deterministically varied, never colliding with the
    # winner or the decoy. These pad the DAG past the pre-registered minimum.
    guard = 0
    while len(nodes) < MIN_NODES:
        combo = {}
        for i, axis in enumerate(AXES):
            idx = (seed * 3 + guard * 7 + i * 5) % len(av[axis])
            combo[axis] = av[axis][idx]
        statement = ";".join(f"{a}={combo[a]}" for a in AXES)
        if statement not in (win_cfg, dec_cfg):
            _add_combination(combo, layer=2)
        guard += 1

    # Goal node — a DEPENDENCY child of the winning combination.
    goal_id = _next_id()
    nodes.append(
        {
            "id": goal_id,
            "statement": f"GOAL: achieve target {domain_desc}",
            "layer": 3,
            "is_goal": True,
            "target_metric": TARGET_METRIC,
        }
    )
    edges.append({"src": winning_combo_id, "dst": goal_id, "type": "DEPENDENCY"})
    node_truth[goal_id] = {"true_success": score_config(win_cfg, seed, MIN_CONFIRM_DEPTH)}

    return {
        "seed": seed,
        "domain": domain_key,
        "domain_description": domain_desc,
        "axes": list(AXES),
        "axis_values": av,
        "synergy_pair": list(SYNERGY_PAIR),
        "separable_axes": list(SEPARABLE_AXES),
        "winning_values": wv,
        "winning_config": win_cfg,
        "winning_node_id": winning_combo_id,
        "decoy_axis": dec_axis,
        "decoy_value": dec_value,
        "decoy_config": dec_cfg,
        "decoy_node_id": decoy_combo_id,
        "min_confirm_depth": MIN_CONFIRM_DEPTH,
        "evidence_regime": EVIDENCE_REGIME,
        "nodes": nodes,
        "edges": edges,
        "goal_node_id": goal_id,
        "goal_parent_id": winning_combo_id,
        "node_truth": node_truth,
        "session_breakpoints": session_breakpoints(seed),
        "total_nodes": len(nodes),
        "target_metric": TARGET_METRIC,
        "tool_budget": TOOL_BUDGET,
        # Analytic difficulty record — asserted by the contract tests and quoted
        # in the pre-registration so calibration is auditable after the fact.
        "reference_strategy_probes": reference_strategy_probes(seed),
        "min_reference_probes": min_reference_probes(),
    }


def generate_all(output_dir: Path, seeds: list[int] | None = None) -> list[Path]:
    """Generate one landscape config per seed and write them to output_dir."""
    from eval.runner.config import TASK_SEEDS

    if seeds is None:
        seeds = list(TASK_SEEDS)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        config = _generate_dag(seed)
        path = output_dir / f"landscape_seed_{seed}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    out = Path(__file__).parent / "landscapes"
    generated = generate_all(out)
    print(f"Generated {len(generated)} landscape configs in {out}")
    for p in generated:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(
            f"  {p.name}: {data['total_nodes']} nodes, domain={data['domain']}, "
            f"reference={data['reference_strategy_probes']} probes "
            f"(first reset at {data['session_breakpoints'][0]}), "
            f"decoy={data['decoy_axis']}={data['decoy_value']}"
        )
