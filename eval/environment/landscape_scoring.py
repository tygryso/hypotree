"""Scoring logic for the premise-gated, epistatic, decoy-bearing landscape.

The evaluation task is a combinatorial search built around three properties that
together make it a fair test of a persistent, self-revising belief state.

**1. Non-separability (epistasis).** The hidden answer sets one correct value on
each axis, but the axes do not contribute independently: two of them form a
*synergy pair* that pays nothing unless BOTH are correct, and equal-tier
configurations score identically. A one-factor-at-a-time sweep over full
configurations therefore has nothing to climb and cannot solve the task.

**2. Premise probes with hard refutation.** A config naming exactly ONE axis is a
*premise probe*, testing that axis-value in isolation. A wrong value returns
exactly ``0.0`` — a conclusive refutation — so every candidate combination built
on it is void. This gives the hypothesis DAG real semantic force: refuting a
premise legitimately prunes its whole subtree. A right value returns ``1.0``,
which clears the engine's verify bar, so a confirmed premise becomes VERIFIED and
its dependent combinations become reachable.

**3. A planted decoy — the contradiction that forces belief revision.** On one
axis, a second value *also* confirms as a premise probe but poisons every
combination it appears in: assembled and probed at depth, such a combination
returns exactly ``0.0``. Component-level validation passes; integration fails.
The decoy is invisible to any amount of isolated probing, so it can only be
caught by assembling and confirming at depth — and catching it *retracts* an
earlier confirmation, which is precisely the belief-revision cycle the project
exists to test.

Because the decoy mirages at shallow depth, a win requires confirmation at
``MIN_CONFIRM_DEPTH``. That single rule replaces the previous bespoke
ambush-config special case.

This module is pure stdlib, side-effect free, and is imported by both the
landscape generator (ground truth) and the HTTP server (live scoring) so the two
can never drift.
"""

from __future__ import annotations

import hashlib

# The configuration axes. `method` and `parameter` form the synergy pair: neither
# pays anything on its own, only both-correct together. The remaining axes
# contribute independently and give the agent a foothold / navigable gradient.
#
# Five axes of five values, not four of four. The gate asks for a 25% reduction
# in *integer* probe counts, so the reference cost sets the resolution of the
# answer: at the previous ~12-probe reference a 25% win had to be found in about
# three whole steps, less headroom than the task contained once its fixed costs
# were paid. Five-by-five raises the reference to ~18 (range 12–22) and the
# observed baseline with it, so the same percentage is worth roughly twice as
# many steps and is expressible at all. It also strengthens the mechanism under
# test: a confirmed value now retires four alternatives instead of three, and one
# more axis must be carried across every context reset.
AXES: tuple[str, ...] = ("component", "method", "parameter", "regime", "encoding")
SYNERGY_PAIR: tuple[str, str] = ("method", "parameter")
SEPARABLE_AXES: tuple[str, ...] = ("component", "regime", "encoding")

VALUES_PER_AXIS = 5

# Scoring weights. Chosen so that ONLY an all-axes-correct config clears the
# target, with a comfortable margin on every boundary:
#   all correct               0.05 + 0.60 + 0.28 = 0.93  (win; also > 0.8 verify bar)
#   pair + two separable      0.05 + 0.40 + 0.28 = 0.73  (below target by 0.02)
#   all separable, no pair    0.05 + 0.60 + 0.00 = 0.65
#   pair + one separable      0.05 + 0.20 + 0.28 = 0.53
#   pair only                 0.05 + 0.00 + 0.28 = 0.33
#   one separable             0.05 + 0.20        = 0.25
#   nothing correct           0.05
#
# The separable credit is unchanged and the synergy bonus absorbs the third
# separable axis, which keeps both the winning score and the 0.02 margin on the
# nearest miss exactly where they were on the previous landscape — the boundary
# behaviour the difficulty calibration depends on is therefore untouched.
_FLOOR = 0.05
_SEPARABLE_CREDIT = 0.20
_SYNERGY_BONUS = 0.28

# Scoring is exact — deliberately no jitter. A cosmetic tie-break would leak a
# usable signal on precisely the axes that must show none: a sweep over a synergy
# axis would rank its four values by noise, and an agent taking the argmax would
# pick the correct one by luck often enough to bypass the interaction entirely.
# Equal-tier configurations therefore score *identically*, which is the honest
# rendering of "this axis has no effect on its own".

# The declared goal threshold an agent must clear to win.
TARGET_METRIC = 0.75

# Premise-probe outcomes. A refuted premise returns exactly 0.0 so a
# deterministic node INVALIDATES (and its dependent combinations prune). A
# confirmed premise returns 1.0 so the node VERIFIES — which is what makes its
# DEPENDENCY children reachable and lets a later contradiction propagate back up
# to it. Anything in between would strand every premise in the dead zone and
# leave the whole combination layer of the DAG permanently unreachable.
PREMISE_REFUTED = 0.0
PREMISE_CONFIRMED = 1.0

# A result must be confirmed at least this deep to count as a win. Shallow probes
# cannot distinguish the real answer from the decoy.
MIN_CONFIRM_DEPTH = 2

# Any combination containing the decoy value scores exactly this at
# MIN_CONFIRM_DEPTH or deeper: a hard integration failure.
DECOY_REFUTED = 0.0

# The environment is a deterministic oracle: one probe of a configuration is the
# whole truth about it. Declared here so the harness can pin every node's
# evidence regime to the environment instead of letting an agent pick one — a
# node wrongly marked stochastic never invalidates, verifies or exhausts, which
# silently disables the entire revision machinery.
EVIDENCE_REGIME = "deterministic"

# Winning value indices are drawn from this pool. Index 0 is excluded so that
# resolving any axis by premise probing costs at least two probes — the basis of
# the analytic difficulty guarantee in reference_strategy_probes().
_WINNING_INDEX_POOL: tuple[int, ...] = tuple(range(1, VALUES_PER_AXIS))

# What each probe costs, in seconds. Every gate to date counts *probes*, which is
# only defensible because this oracle answers in uniform milliseconds — so a
# probe is the unit of cost by construction, and a navigator that ranks by value
# per unit cost induces exactly the same order as one that ignores cost. That
# makes cost-aware selection unmeasurable here, not neutral.
#
# These tiers are assigned to the five values of each axis by a seeded shuffle.
# Heavy-tailed on purpose: real R&D spreads four orders of magnitude inside one
# project (a unit test, a fine-tune, a wet-lab run), and a mechanism that only
# pays on a uniform spread is not the mechanism anyone needs.
#
# The saving this exposes is specific and is the whole point. Probe *count* is
# invariant under reordering — the winner's position within a question is
# uniform, so any order settles it in the same expected number of probes, which
# is why three consecutive audits found exclusion yield pinned at chance. Probe
# *cost* is not invariant, because the last survivor of a closed question is
# deduced for free: whichever answer is left unprobed is never paid for. So
# ordering cheap-first pushes the expensive answer into the free slot, and the
# expected cost falls while the expected probe count does not move at all.
_COST_TIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 8.0, 40.0)

# The cost of assembling and probing a full combination. Flat, and deliberately
# mid-range: the combination is not a lever — every episode must build one — so
# giving it a cost that varied would add noise to the measurement without adding
# a decision. Charging nothing would flatter the cost-aware arm by making the one
# probe it cannot avoid free.
COMBINATION_COST = 4.0


def stable_hash(text: str) -> float:
    """Deterministic float in [0,1) from a string via SHA-256."""
    h = hashlib.sha256(text.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def winning_values(seed: int) -> dict[str, str]:
    """The correct value for each axis, deterministic per seed.

    Uses an independent SHA-256 draw per (seed, axis) so winning combinations are
    well-distributed across the seed range rather than aliasing on a shared
    modulus. Index 0 is never drawn (see ``_WINNING_INDEX_POOL``).
    """
    pool = _WINNING_INDEX_POOL
    return {axis: f"v{pool[int(stable_hash(f'{seed}:{axis}') * len(pool))]}" for axis in AXES}


def axis_values() -> dict[str, list[str]]:
    """The full menu of candidate values for every axis (the search space)."""
    return {axis: [f"v{v}" for v in range(VALUES_PER_AXIS)] for axis in AXES}


def axis_value_costs(seed: int) -> dict[str, dict[str, float]]:
    """Seconds each premise probe costs, per axis and value.

    The tiers are permuted independently per (seed, axis), so which answer is
    expensive is uncorrelated with which answer is *right*. That independence is
    what makes the measurement honest: if the cheap answer were usually the
    winner, a cost-aware navigator would look good for finding the answer sooner
    rather than for deferring the expensive probe into the free deduction slot,
    and those are different claims.
    """
    out: dict[str, dict[str, float]] = {}
    for axis in AXES:
        values = [f"v{v}" for v in range(VALUES_PER_AXIS)]
        # Fisher-Yates driven by the seeded hash: a deterministic permutation
        # that does not depend on Python's own RNG or its iteration order.
        tiers = list(_COST_TIERS)
        for i in range(len(tiers) - 1, 0, -1):
            j = int(stable_hash(f"{seed}:{axis}:cost:{i}") * (i + 1))
            tiers[i], tiers[j] = tiers[j], tiers[i]
        out[axis] = dict(zip(values, tiers, strict=True))
    return out


def probe_cost(config: str, seed: int) -> float:
    """What probing this configuration costs, in seconds.

    A premise probe costs whatever its axis-value costs; anything naming more
    than one axis is a combination and costs a flat ``COMBINATION_COST``.
    """
    parsed = parse_config(config)
    known = {a: v for a, v in parsed.items() if a in AXES}
    if len(known) == 1:
        axis, value = next(iter(known.items()))
        return axis_value_costs(seed)[axis].get(value, COMBINATION_COST)
    return COMBINATION_COST


def decoy_axis(seed: int) -> str:
    """The axis that carries the planted decoy.

    Always a separable axis: a decoy on a synergy axis would be masked by the
    interaction and could not be isolated as a premise, which is the whole point.
    """
    return SEPARABLE_AXES[seed % len(SEPARABLE_AXES)]


def decoy_value(seed: int) -> str:
    """The value that confirms in isolation but poisons every combination.

    Deterministically chosen to differ from the winning value on that axis.
    """
    axis = decoy_axis(seed)
    true_idx = int(winning_values(seed)[axis][1:])
    offset = 1 + int(stable_hash(f"{seed}:decoy") * (VALUES_PER_AXIS - 1))
    return f"v{(true_idx + offset) % VALUES_PER_AXIS}"


def confirming_values(seed: int) -> dict[str, list[str]]:
    """Values that a premise probe reports as confirmed, per axis.

    Exactly one per axis, except the decoy axis which has two — the honest source
    of the ambiguity that only assembly can resolve.
    """
    wv = winning_values(seed)
    out = {axis: [wv[axis]] for axis in AXES}
    out[decoy_axis(seed)].append(decoy_value(seed))
    return out


def winning_config(seed: int) -> str:
    """The canonical winning combination string for a seed."""
    wv = winning_values(seed)
    return ";".join(f"{axis}={wv[axis]}" for axis in AXES)


def decoy_config(seed: int) -> str:
    """The trap: correct on every axis except the decoy one, which is the decoy.

    Mirages as a full-marks result at shallow depth and hard-fails at depth.
    """
    combo = dict(winning_values(seed))
    combo[decoy_axis(seed)] = decoy_value(seed)
    return ";".join(f"{axis}={combo[axis]}" for axis in AXES)


def parse_config(config: str) -> dict[str, str]:
    """Parse an ``axis=value;axis=value`` string into a dict.

    Tolerant of whitespace, unknown tokens, and missing axes — junk is ignored so
    a verbose config is never penalised for anything but its axis values.
    """
    parsed: dict[str, str] = {}
    for token in config.split(";"):
        token = token.strip()
        if "=" not in token:
            continue
        axis, _, value = token.partition("=")
        parsed[axis.strip()] = value.strip()
    return parsed


def known_axes(config: str) -> dict[str, str]:
    """The parsed axis→value pairs restricted to real axes (junk dropped)."""
    parsed = parse_config(config)
    return {a: v for a, v in parsed.items() if a in AXES}


def is_premise_probe(config: str) -> bool:
    """True when the config names exactly one real axis (a premise probe)."""
    return len(known_axes(config)) == 1


def is_full_combination(config: str) -> bool:
    """True when the config assigns a value to every axis."""
    return len(known_axes(config)) == len(AXES)


def contains_decoy(config: str, seed: int) -> bool:
    """True when the config uses the decoy value on the decoy axis."""
    return known_axes(config).get(decoy_axis(seed)) == decoy_value(seed)


def n_correct_axes(config: str, seed: int) -> int:
    """How many axes the config sets to their correct value."""
    wv = winning_values(seed)
    parsed = parse_config(config)
    return sum(1 for axis in AXES if parsed.get(axis) == wv[axis])


def score_config(config: str, seed: int, depth: int = 0) -> float:
    """Score a config string at a given exploration depth.

    Premise probe (exactly one axis named): confirmed values return
    ``PREMISE_CONFIRMED`` and everything else ``PREMISE_REFUTED``. The decoy
    confirms here at *any* depth — an isolated component test cannot reveal an
    integration failure, which is what makes the trap honest rather than merely
    hidden.

    Anything else is scored as a combination: independent credit per correct
    separable axis, plus the synergy bonus only when both synergy-pair axes are
    correct. Unspecified axes count as incorrect, so a partial combination is
    scored consistently with a full one. A combination carrying the decoy scores
    as though the decoy were correct at shallow depth (the mirage) and collapses
    to ``DECOY_REFUTED`` at ``MIN_CONFIRM_DEPTH`` or deeper (the truth).
    """
    wv = winning_values(seed)
    parsed = parse_config(config)

    if is_premise_probe(config):
        axis, value = next(iter(known_axes(config).items()))
        return PREMISE_CONFIRMED if value in confirming_values(seed)[axis] else PREMISE_REFUTED

    carries_decoy = contains_decoy(config, seed)
    if carries_decoy and depth >= MIN_CONFIRM_DEPTH:
        return DECOY_REFUTED

    effective = dict(parsed)
    if carries_decoy:
        # Shallow probing cannot tell the decoy from the real thing.
        effective[decoy_axis(seed)] = wv[decoy_axis(seed)]

    credit = _FLOOR
    for axis in SEPARABLE_AXES:
        if effective.get(axis) == wv[axis]:
            credit += _SEPARABLE_CREDIT
    if all(effective.get(axis) == wv[axis] for axis in SYNERGY_PAIR):
        credit += _SYNERGY_BONUS

    return max(0.0, min(1.0, credit))


def reference_strategy_probes(seed: int) -> int:
    """Probes the canonical strategy needs to solve this seed — the difficulty floor.

    The canonical strategy is the one the briefing describes:

    1. Premise-probe each axis's candidate values in order, stopping at the first
       confirmation. A refutation eliminates a value outright, so the last
       survivor never needs probing — three refutations imply the fourth.
    2. Assemble the confirmed values and probe the combination at
       ``MIN_CONFIRM_DEPTH``.
    3. If that returns 0.0 the chosen value was the decoy: resume probing the
       remaining candidates on that axis, then reassemble and confirm again.

    Simulated rather than derived in closed form, because the decoy makes the
    cost path-dependent. The contract tests assert this matches an independent
    replay of the same strategy, so the number the breakpoints are calibrated
    against is the number a real agent actually pays.
    """
    av = axis_values()
    confirming = confirming_values(seed)
    probes = 0
    chosen: dict[str, str] = {}
    remaining: dict[str, list[str]] = {}

    for axis in AXES:
        candidates = list(av[axis])
        for i, value in enumerate(candidates):
            if i == VALUES_PER_AXIS - 1:
                # Implied by elimination — no probe needed.
                chosen[axis] = value
                remaining[axis] = []
                break
            probes += 1
            if value in confirming[axis]:
                chosen[axis] = value
                remaining[axis] = candidates[i + 1 :]
                break

    probes += 1  # assemble and confirm at depth
    if contains_decoy(";".join(f"{a}={chosen[a]}" for a in AXES), seed):
        # The trap fired: keep probing that axis until the real value appears,
        # then reassemble and confirm once more.
        axis = decoy_axis(seed)
        for i, value in enumerate(remaining[axis]):
            if i == len(remaining[axis]) - 1:
                break  # last survivor implied
            probes += 1
            if value in confirming[axis]:
                break
        probes += 1

    return probes


def min_reference_probes() -> int:
    """Best case of reference_strategy_probes() over every possible seed.

    Every axis costs at least 2 probes (index 0 is excluded from the winning
    pool), plus one probe to assemble and confirm the combination. The decoy can
    only ever add probes, never remove them, so this remains a true lower bound.
    """
    return 2 * len(AXES) + 1


def _sweep_cost(seed: int, order: str) -> float:
    """Cost of settling every axis by sweeping its values in a given order.

    ``order`` is ``"declared"`` (v0, v1, …, the order the briefing lists them) or
    ``"cheap"`` (ascending cost). Both stop at the first confirmation and both
    leave the last survivor unprobed, because a closed question deduces it — so
    the two differ *only* in which answers end up in that free slot, which is
    exactly the quantity cost-aware selection is claimed to exploit.

    Probe count is identical between the two in expectation, so a difference here
    is a difference in cost and in nothing else.
    """
    costs = axis_value_costs(seed)
    confirming = confirming_values(seed)
    total = 0.0
    for axis in AXES:
        values = [f"v{v}" for v in range(VALUES_PER_AXIS)]
        if order == "cheap":
            values.sort(key=lambda v: (costs[axis][v], v))
        for i, value in enumerate(values):
            if i == VALUES_PER_AXIS - 1:
                break  # deduced by elimination: never probed, never paid for
            total += costs[axis][value]
            if value in confirming[axis]:
                break
    return total + COMBINATION_COST


def reference_strategy_cost(seed: int) -> float:
    """Cost of the canonical strategy — the cost-blind baseline.

    The canonical strategy sweeps each axis in declared order, which is what any
    navigator that cannot see cost does on average, because the declared order
    carries no cost information.
    """
    return _sweep_cost(seed, "declared")


def optimal_strategy_cost(seed: int) -> float:
    """Cost of sweeping every axis cheapest-answer-first — the achievable floor.

    Not the *theoretical* floor: an oracle knowing which answer is right would
    probe only that one. This is the floor for a navigator that knows what each
    probe costs and nothing else about which is correct, which is precisely what
    cost-aware selection gives it. Reporting the measured arm against this rather than
    against zero is what separates "the mechanism works" from "the mechanism is
    perfect", and only the first is being claimed.
    """
    return _sweep_cost(seed, "cheap")
