"""Generate human-readable briefing documents for the evaluation agent.

Each briefing describes the R&D configuration space in prose: the independent
axes and the candidate values available for each. It gives the agent enough
structure to plan a search, but it does NOT reveal the winning combination —
only the search space. The agent must discover which value is correct on each
axis by probing.

The briefing is the ONLY information the agent receives about the task.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path


def generate_briefing(landscape_path: Path) -> str:
    """Generate a human-readable briefing from a landscape JSON config.

    Returns a markdown string the agent reads as its task description. Lists the
    configuration axes and their candidate values (the search space) but never
    the winning combination.
    """
    data = json.loads(landscape_path.read_text())
    domain = data["domain"]
    description = data["domain_description"]
    budget = data["tool_budget"]
    session_points = data["session_breakpoints"]
    axes = data["axes"]
    axis_values = data["axis_values"]
    target = data.get("target_metric", 0.75)

    n_combos = prod(len(axis_values[axis]) for axis in axes)
    synergy = data["synergy_pair"]
    separable = data["separable_axes"]
    min_depth = data.get("min_confirm_depth", 2)

    axis_lines = []
    for axis in axes:
        values = ", ".join(axis_values[axis])
        role = "interacting" if axis in synergy else "independent"
        axis_lines.append(f"  - **{axis}** ({role}): one of [{values}]")

    # Illustrative probes use an explicit placeholder rather than concrete
    # values. A concrete example risks either naming a correct value by accident
    # (a partial answer leak) or, if correct values were deliberately avoided,
    # handing over free eliminations to anyone who reasons about how the example
    # was built. A placeholder shows the grammar and leaks exactly nothing.
    example_premise = f"{axes[0]}=<value>"
    example_full = ";".join(f"{axis}=<value>" for axis in axes)

    briefing = f"""# R&D Task: {description}

## Problem Domain
You are optimizing **{domain}** — {description}. A candidate configuration sets
one value on each of several axes; exactly one value per axis is optimal. The
axes are not all independent (see *Interactions* below). Your job is to discover
the combination that maximizes the success metric.

## Configuration Axes
A configuration is written as `axis=value` pairs joined by `;`:

{chr(10).join(axis_lines)}

There are {n_combos} possible combinations — far more than your budget allows,
so brute force is not an option.

## Two Kinds of Probe

In both forms, replace `<value>` with one of the candidate values listed above.

**1. Premise probe — name exactly ONE axis.**
```bash
python agent_eval_client.py "{example_premise}" 1
```
This tests that single axis-value in isolation and is the fastest way to learn
the search space:
- **`success = 0.0` means the value is REFUTED.** It is definitively wrong, and
  every combination using it is therefore dead. Rule it out permanently and never
  probe it again.
- **A positive score means the value is CONFIRMED** in isolation. It is not a
  solution on its own — it is one axis settled.

Because a refutation eliminates a value outright, you never need to test the last
remaining value on an axis: if the other candidates are all refuted, the survivor
must be correct.

**2. Combination probe — name ALL axes.**
```bash
python agent_eval_client.py "{example_full}" {min_depth}
```
This scores a complete candidate configuration.

## Integration Failures — A Component Can Pass Alone And Still Break
Passing in isolation does **not** guarantee a value works in the finished
configuration. One of the candidate values on one of the axes is known to
integrate badly: it confirms perfectly as a premise probe, and a combination
containing it even looks excellent at shallow depth — but probed at depth
**{min_depth}** or deeper, that combination collapses to **0.0**.

A combination scoring `0.0` at depth {min_depth} is therefore telling you
something specific: one of the values you trusted is not actually viable, even
though it passed on its own. Go back, re-open that axis, and try its other
candidates.

**This means more than one value on an axis can confirm as a premise.** Isolated
confirmation narrows the field; only deep assembly settles it.

## Confirming a Result
A result counts **only if confirmed at depth ≥ {min_depth}**. Shallow probes
cannot tell a genuine answer from an integration failure, so a shallow high score
is never sufficient — always re-probe your best candidate at depth {min_depth}.

## Interactions — Why Sweeping One Axis At A Time Fails
The axes are **not** independent. `{synergy[0]}` and `{synergy[1]}` **interact**:
a combination earns credit for them only when **both** are correct at once.
Getting one right and the other wrong scores exactly the same as getting both
wrong, so varying one of them inside a full combination shows **no signal at all**
until its partner is already correct.

`{separable[0]}` and `{separable[1]}` do contribute independently, so they show a
visible gradient in combination probes.

**Practical consequence:** resolve `{synergy[0]}` and `{synergy[1]}` with premise
probes. Trying to tune them by sweeping full combinations will look like a flat,
signal-free landscape and will waste your entire budget.

## Goal
Find and verify the combination whose success metric is ≥ **{target}**, confirmed
at depth ≥ **{min_depth}**. A combination clears the target only when **every**
axis is set correctly — a partially-correct combination always scores below it.

## Constraints
- Budget: **{budget} experiments** (`evaluate_config` calls).
- Each experiment probes one configuration at a given exploration depth (0–4).
- Your context will be **reset at steps: {session_points}**. Resolving every axis
  takes more probes than fit before the first reset, so you *will* have to carry
  what you have learned across at least one reset. Record findings as you go.

## Recommended Approach
1. For each axis, premise-probe its candidate values until one is confirmed
   (refuted values are eliminated permanently — record them).
2. Assemble the confirmed values into one complete combination and probe it at
   depth {min_depth}.
3. If that combination returns `0.0`, one of your confirmed values fails in
   integration. Re-open that axis, try its remaining candidates, and reassemble.

Returns: `{{"success": 0.0-1.0, "metrics": {{...}}}}`
"""
    return briefing


def generate_all_briefings(landscapes_dir: Path, output_dir: Path) -> list[Path]:
    """Generate briefing documents for all landscape configs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for landscape_path in sorted(landscapes_dir.glob("landscape_seed_*.json")):
        seed = landscape_path.stem.split("_")[-1]
        briefing = generate_briefing(landscape_path)
        out_path = output_dir / f"briefing_seed_{seed}.md"
        out_path.write_text(briefing)
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    landscapes = Path(__file__).parent / "landscapes"
    out = Path(__file__).parent / "briefings"
    generated = generate_all_briefings(landscapes, out)
    print(f"Generated {len(generated)} briefings in {out}")
