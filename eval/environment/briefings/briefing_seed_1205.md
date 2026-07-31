# R&D Task: specific impulse optimization

## Problem Domain
You are optimizing **rocket_propellant** — specific impulse optimization. A candidate configuration sets
one value on each of several axes; exactly one value per axis is optimal. The
axes are not all independent (see *Interactions* below). Your job is to discover
the combination that maximizes the success metric.

## Configuration Axes
A configuration is written as `axis=value` pairs joined by `;`:

  - **component** (independent): one of [v0, v1, v2, v3, v4]
  - **method** (interacting): one of [v0, v1, v2, v3, v4]
  - **parameter** (interacting): one of [v0, v1, v2, v3, v4]
  - **regime** (independent): one of [v0, v1, v2, v3, v4]
  - **encoding** (independent): one of [v0, v1, v2, v3, v4]

There are 3125 possible combinations — far more than your budget allows,
so brute force is not an option.

## Two Kinds of Probe

In both forms, replace `<value>` with one of the candidate values listed above.

**1. Premise probe — name exactly ONE axis.**
```bash
python agent_eval_client.py "component=<value>" 1
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
python agent_eval_client.py "component=<value>;method=<value>;parameter=<value>;regime=<value>;encoding=<value>" 2
```
This scores a complete candidate configuration.

## Integration Failures — A Component Can Pass Alone And Still Break
Passing in isolation does **not** guarantee a value works in the finished
configuration. One of the candidate values on one of the axes is known to
integrate badly: it confirms perfectly as a premise probe, and a combination
containing it even looks excellent at shallow depth — but probed at depth
**2** or deeper, that combination collapses to **0.0**.

A combination scoring `0.0` at depth 2 is therefore telling you
something specific: one of the values you trusted is not actually viable, even
though it passed on its own. Go back, re-open that axis, and try its other
candidates.

**This means more than one value on an axis can confirm as a premise.** Isolated
confirmation narrows the field; only deep assembly settles it.

## Confirming a Result
A result counts **only if confirmed at depth ≥ 2**. Shallow probes
cannot tell a genuine answer from an integration failure, so a shallow high score
is never sufficient — always re-probe your best candidate at depth 2.

## Interactions — Why Sweeping One Axis At A Time Fails
The axes are **not** independent. `method` and `parameter` **interact**:
a combination earns credit for them only when **both** are correct at once.
Getting one right and the other wrong scores exactly the same as getting both
wrong, so varying one of them inside a full combination shows **no signal at all**
until its partner is already correct.

`component` and `regime` do contribute independently, so they show a
visible gradient in combination probes.

**Practical consequence:** resolve `method` and `parameter` with premise
probes. Trying to tune them by sweeping full combinations will look like a flat,
signal-free landscape and will waste your entire budget.

## Goal
Find and verify the combination whose success metric is ≥ **0.75**, confirmed
at depth ≥ **2**. A combination clears the target only when **every**
axis is set correctly — a partially-correct combination always scores below it.

## Constraints
- Budget: **100 experiments** (`evaluate_config` calls).
- Each experiment probes one configuration at a given exploration depth (0–4).
- Your context will be **reset at steps: [8, 14, 21]**. Resolving every axis
  takes more probes than fit before the first reset, so you *will* have to carry
  what you have learned across at least one reset. Record findings as you go.

## Recommended Approach
1. For each axis, premise-probe its candidate values until one is confirmed
   (refuted values are eliminated permanently — record them).
2. Assemble the confirmed values into one complete combination and probe it at
   depth 2.
3. If that combination returns `0.0`, one of your confirmed values fails in
   integration. Re-open that axis, try its remaining candidates, and reassemble.

Returns: `{"success": 0.0-1.0, "metrics": {...}}`
