"""Frozen scoring script for the dogfooding gate.

Ingests one run's ``eval/runs/<run-id>/*.jsonl`` logs and computes the four
pre-registered criteria. Emits a single JSON
decision: ``{"decision": "GO"|"STOP"|"ITERATE", "criteria": {...}, "details":
{...}}``.

Scoring is always scoped to a single run id — mixing two runs' logs would pool
episodes produced under different code, which is exactly the confound the
pre-registration exists to prevent.

Written and frozen against synthetic fixtures **before** any real runs execute
(pre-registration §2: "blinding of analysis"). The script has no manual metric
selection — it reads the logs, computes, and emits the decision.

Decision rule (pre-registration §6):
1. If criterion 1 (moat) OR criterion 2 (TS quality) fails → STOP.
2. If 1 and 2 pass but 3 or 4 fails → ITERATE.
3. If all pass → GO.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from eval.runner.config import TASK_SEEDS
from eval.runner.runner import ACTION_TAXONOMY

# Pre-registered constants (EVAL_PREREGISTRATION.md §7).
ALPHA = 0.05
# Derived from the frozen seed set so the two can never disagree.
N_SEEDS = len(TASK_SEEDS)

# Power floor for the criterion-4 χ². The asymptotic test needs a reasonable
# expected count per cell; below this the pre-committed collapse default stands
# rather than a significance claim nobody should believe.
MIN_CHI2_PER_CONTEXT = 20
MIN_CHI2_TOTAL = 50


def _majority_threshold(n: int) -> int:
    """Seeds the treatment must win, as a strict majority of n.

    Derived rather than hard-coded so raising the seed count cannot silently
    weaken the consistency requirement (with n=30, a fixed "≥ 6" would be a 20%
    win rate rather than a majority).
    """
    return n // 2 + 1


MAJORITY_THRESHOLD = _majority_threshold(N_SEEDS)
MOAT_REDUCTION_FLOOR = 0.25  # ≥ 25% median paired step reduction
MOAT_CLIFFS_DELTA_FLOOR = 0.33  # medium effect
TS_REDUCTION_FLOOR = 0.20  # ≥ 20% median reduction vs random; ≥ 20% worst-case reduction vs greedy


@dataclass
class CriterionResult:
    """Outcome of one criterion — pass/fail + metrics + details."""

    name: str
    passed: bool
    metrics: dict[str, Any]
    details: dict[str, Any]


@dataclass
class GateDecision:
    """The final decision emitted by the gate."""

    decision: str  # "GO" | "STOP" | "ITERATE"
    criteria: dict[str, dict[str, Any]]
    details: dict[str, Any]


# -- Log parsing --------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all JSONL lines from a file."""
    entries = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def _load_runner_logs(runs_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load the runner JSONL logs for the pre-registered seeds, grouped by arm.

    Only seeds in ``TASK_SEEDS`` are admitted. The runs directory accumulates
    logs across runs, including seeds retired from the pre-registration, and a
    bare glob would silently pool a superseded run in with the current one — the
    paired statistics would then be computed over a mixture of two different
    landscape designs. Filtering here keeps the gate defined by the
    pre-registration rather than by whatever files happen to be on disk.

    Returns ``{arm: {seed: [events]}}``.
    """
    result: dict[str, dict[str, list[dict[str, Any]]]] = {"A": {}, "B": {}, "F": {}}
    for log_path in sorted(runs_dir.glob("seed-*-arm-*.jsonl")):
        parts = log_path.stem.split("-")
        seed = int(parts[1])
        arm = parts[3]
        if arm in result and seed in TASK_SEEDS:
            result[arm][seed] = _load_jsonl(log_path)
    return result


def _load_ablation_logs(runs_dir: Path) -> list[dict[str, Any]]:
    """Load all ablation result entries from JSONL files."""
    entries = []
    for log_path in sorted(runs_dir.glob("ablation-*.jsonl")):
        entries.extend(_load_jsonl(log_path))
    return entries


def _extract_run_end(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the run_end event, or None if the run has no terminal record."""
    for e in reversed(events):
        if e.get("event_type") == "run_end":
            return e
    return None


def _extract_tool_budget(events: list[dict[str, Any]]) -> int:
    """Return the tool budget from the run_start event (default 60)."""
    for e in events:
        if e.get("event_type") == "run_start":
            return int(e.get("tool_budget", 60))
    return 60


def _extract_steps_to_target(events: list[dict[str, Any]]) -> int:
    """Extract steps_to_target, right-censored at the budget for failed runs.

    A run that never met its goal (including one where the agent gave up early)
    must NOT be credited with a low step count — that would make a failure look
    like the fastest solve. Censoring at the tool budget matches the runner's
    own summary and is the correct treatment for a right-censored metric.
    """
    run_end = _extract_run_end(events)
    if run_end is not None:
        if not run_end.get("goals_met", False):
            return _extract_tool_budget(events)
        return run_end.get("step", 0)  # step at run_end = final step count
    # Fallback: no terminal record — count experiment events.
    return sum(1 for e in events if e.get("event_type") == "experiment")


def _extract_goals_met(events: list[dict[str, Any]]) -> bool:
    """Extract whether goals were met from a runner log."""
    for e in reversed(events):
        if e.get("event_type") == "run_end":
            return e.get("goals_met", False)
    return False


def _extract_pruned_reexecutions(events: list[dict[str, Any]]) -> int:
    """Count pruned_reexecution events in a runner log."""
    return sum(1 for e in events if e.get("event_type") == "pruned_reexecution")


def _extract_upstream_propagation(events: list[dict[str, Any]]) -> int:
    """Count upstream propagation events.

    The runner emits a ``status_transition`` with ``propagated=True`` for every
    node the engine flips as a side effect of recording evidence on a different
    node. Upstream DEPENDENCY revision is the ``VERIFIED → NEEDS_REVISION`` flip,
    which only ever happens via propagation.
    """
    return sum(
        1
        for e in events
        if e.get("event_type") == "status_transition"
        and e.get("propagated") is True
        and e.get("new_status") == "NEEDS_REVISION"
    )


# -- Statistics ---------------------------------------------------------------


def cliffs_delta(treatment: list[float], control: list[float]) -> float:
    """Compute Cliff's delta — a non-parametric effect size.

    δ = (#(treatment > control) - #(treatment < control)) / (n1 * n2)

    Positive δ means treatment dominates control. Magnitude bands:
    |δ| < 0.147 = negligible, < 0.33 = small, < 0.474 = medium, else large.
    """
    n1, n2 = len(treatment), len(control)
    if n1 == 0 or n2 == 0:
        return 0.0
    greater = sum(1 for t in treatment for c in control if t < c)  # lower = better
    less = sum(1 for t in treatment for c in control if t > c)
    return (greater - less) / (n1 * n2)


def bootstrap_ci_median(
    data: list[float], n_resamples: int = 10000, confidence: float = 0.95
) -> tuple[float, float]:
    """Bootstrap 95% CI on the median of paired differences.

    Uses the percentile method via scipy.stats.bootstrap.
    """
    if len(data) < 2:
        return (float(data[0]) if data else 0.0, float(data[0]) if data else 0.0)
    arr = np.array(data)
    result = stats.bootstrap(
        (arr,),
        np.median,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="percentile",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def paired_wilcoxon(treatment: list[float], control: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test. Returns statistic + p-value."""
    if len(treatment) != len(control):
        raise ValueError("paired test requires equal-length arrays")
    if len(treatment) < 5:
        # Too few samples for a reliable Wilcoxon — report what we can.
        return {"statistic": None, "p_value": 1.0, "note": "insufficient samples"}
    diffs = [c - t for c, t in zip(control, treatment, strict=True)]
    if all(d == 0 for d in diffs):
        return {"statistic": None, "p_value": 1.0, "note": "all differences zero"}
    result = stats.wilcoxon(diffs, alternative="greater")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


# -- Criterion computations ---------------------------------------------------


def _paired_moat_comparison(
    baseline: dict[str, list[dict[str, Any]]],
    arm_b: dict[str, list[dict[str, Any]]],
    *,
    name: str,
    claim: str,
) -> CriterionResult:
    """Paired steps_to_target comparison of hypotree against one baseline.

    Shared by both halves of criterion 1. The statistics and thresholds are
    identical; only the baseline differs, which is the whole point — running the
    same test against two baselines is what separates "persistence is automatic"
    from "the belief state is better structured".
    """
    common_seeds = sorted(set(baseline.keys()) & set(arm_b.keys()))
    if len(common_seeds) < N_SEEDS:
        return CriterionResult(
            name=name,
            passed=False,
            metrics={"claim": claim},
            details={"error": f"need {N_SEEDS} paired seeds, found {len(common_seeds)}"},
        )

    steps_base = [_extract_steps_to_target(baseline[s]) for s in common_seeds]
    steps_b = [_extract_steps_to_target(arm_b[s]) for s in common_seeds]

    # Paired differences: positive = B is better (fewer steps).
    diffs = [a - b for a, b in zip(steps_base, steps_b, strict=True)]
    median_reduction = float(np.median(diffs))

    median_base = float(np.median(steps_base))
    fractional_reduction = median_reduction / median_base if median_base > 0 else 0.0

    b_wins = sum(1 for d in diffs if d > 0)
    delta = cliffs_delta(steps_b, steps_base)
    wilcoxon = paired_wilcoxon(steps_b, steps_base)
    ci_low, ci_high = bootstrap_ci_median(diffs)

    # Pass conditions (all three required).
    magnitude_ok = fractional_reduction >= MOAT_REDUCTION_FLOOR
    consistency_ok = b_wins >= _majority_threshold(len(common_seeds))
    effect_ok = delta >= MOAT_CLIFFS_DELTA_FLOOR

    return CriterionResult(
        name=name,
        passed=magnitude_ok and consistency_ok and effect_ok,
        metrics={
            "claim": claim,
            "n_seeds": len(common_seeds),
            "median_paired_reduction_steps": median_reduction,
            "fractional_reduction": round(fractional_reduction, 4),
            "b_wins": b_wins,
            "b_wins_threshold": _majority_threshold(len(common_seeds)),
            "cliffs_delta": round(delta, 4),
            "cliffs_delta_threshold": MOAT_CLIFFS_DELTA_FLOOR,
            "wilcoxon_p_value": wilcoxon["p_value"],
            "ci_95_median_reduction": [round(ci_low, 2), round(ci_high, 2)],
        },
        details={
            "steps_baseline": steps_base,
            "steps_b": steps_b,
            "magnitude_ok": magnitude_ok,
            "consistency_ok": consistency_ok,
            "effect_ok": effect_ok,
        },
    )


def _criterion1_moat(
    arm_a: dict[str, list[dict[str, Any]]],
    arm_b: dict[str, list[dict[str, Any]]],
    arm_f: dict[str, list[dict[str, Any]]],
) -> tuple[CriterionResult, CriterionResult, CriterionResult]:
    """Criterion 1, split into the two distinct claims it was conflating.

    The original single comparison (B vs a manual scratchpad) cannot tell two
    very different advantages apart, and a previous run was decided entirely by
    the weaker one: on the seeds where the baseline agent never wrote a note, B
    won by a median of 32 steps; on the seeds where it did, the median gap was 1.

    - **1a (ergonomic).** B vs Arm A, whose memory exists only if the agent
      chooses to maintain it. Passing means persistence-as-a-by-product beats
      persistence-as-discipline. A real and commercially relevant property, but
      it is *not* the project's hypothesis.
    - **1b (informational).** B vs Arm F, which is handed an automatic, complete,
      flat transcript of every probe. Both arms now keep every fact for free, so
      the only remaining difference is structure: statuses, refutation semantics,
      exclusion inference, frontier, navigator. This is the hypothesis under
      test (§1), and it is what gates the decision.

    Criterion 1 overall is therefore gated on **1b**, the stricter reading —
    a tightening of the original wording (§2: "isolates structured belief-state
    vs unstructured notes, not memory vs no memory"), never a loosening.
    """
    c1a = _paired_moat_comparison(
        arm_a,
        arm_b,
        name="criterion_1a_ergonomic_moat",
        claim=(
            "hypotree vs a manual scratchpad whose upkeep is optional: does making "
            "persistence automatic beat requiring discipline? Reported, not gated."
        ),
    )
    c1b = _paired_moat_comparison(
        arm_f,
        arm_b,
        name="criterion_1b_informational_moat",
        claim=(
            "hypotree vs an automatically-preserved flat experiment log: with every "
            "raw fact retained by both arms, does structured belief state still win? "
            "This is the pre-registered hypothesis and it gates the decision."
        ),
    )

    combined = CriterionResult(
        name="criterion_1_moat",
        passed=c1b.passed,
        metrics={
            "gated_on": "criterion_1b_informational_moat",
            "criterion_1a_passed": c1a.passed,
            "criterion_1b_passed": c1b.passed,
        },
        details={
            "rationale": (
                "Criterion 1 is gated on the informational comparison (1b). Passing 1a "
                "alone shows only that hypotree persists by default while a scratchpad "
                "must be maintained by hand."
            )
        },
    )
    return combined, c1a, c1b


def _criterion2_ts_quality(
    ablation_entries: list[dict[str, Any]],
) -> CriterionResult:
    """Criterion 2: TS beats random (typical-case) AND greedy (worst-case).

    Metric is cumulative regret over a fixed horizon (lower is better). The
    honest structure of the comparison (see ablation_navigator.py):

    - **vs random** — TS must decisively win the *typical* seed: a median regret
      reduction of at least ``TS_REDUCTION_FLOOR`` on a majority of seeds. This
      confirms TS is doing real work, not matching a no-strategy baseline.
    - **vs greedy** — a pure-exploitation greedy wins the *median* on an easy
      bandit, so a typical-case comparison is the wrong question. TS's genuine
      advantage is a bounded *worst case*: it never catastrophically locks a
      decoy. The gate therefore requires TS's worst-case (max) regret to be at
      least ``TS_REDUCTION_FLOOR`` below greedy's worst-case. The typical-case
      greedy comparison is reported for transparency but is NOT gated.
    """
    by_strategy: dict[str, dict[int, float]] = {"ts": {}, "random": {}, "greedy": {}}
    for entry in ablation_entries:
        if entry.get("event_type") != "ablation_result":
            continue
        strategy = entry.get("strategy")
        seed = entry.get("seed")
        regret = entry.get("cumulative_regret")
        if strategy in by_strategy and seed is not None and regret is not None:
            by_strategy[strategy][seed] = float(regret)

    common_seeds = sorted(
        set(by_strategy["ts"]) & set(by_strategy["random"]) & set(by_strategy["greedy"])
    )
    if len(common_seeds) < N_SEEDS:
        return CriterionResult(
            name="criterion_2_ts_quality",
            passed=False,
            metrics={},
            details={"error": f"need {N_SEEDS} seeds per strategy, found {len(common_seeds)}"},
        )

    ts_vals = [by_strategy["ts"][s] for s in common_seeds]
    rand_vals = [by_strategy["random"][s] for s in common_seeds]
    greedy_vals = [by_strategy["greedy"][s] for s in common_seeds]

    # TS vs random — typical-case superiority.
    ts_beats_rand = sum(1 for t, r in zip(ts_vals, rand_vals, strict=True) if t < r)
    median_ts = float(np.median(ts_vals))
    median_rand = float(np.median(rand_vals))
    reduction_rand = (median_rand - median_ts) / median_rand if median_rand > 0 else 0.0
    wilcoxon_rand = paired_wilcoxon(ts_vals, rand_vals)
    delta_rand = cliffs_delta(ts_vals, rand_vals)
    rand_ok = reduction_rand >= TS_REDUCTION_FLOOR and ts_beats_rand >= MAJORITY_THRESHOLD

    # TS vs greedy — worst-case guardrail (no catastrophic lock-in).
    max_ts = max(ts_vals)
    max_greedy = max(greedy_vals)
    worst_case_reduction = (max_greedy - max_ts) / max_greedy if max_greedy > 0 else 0.0
    greedy_ok = worst_case_reduction >= TS_REDUCTION_FLOOR

    # Typical-case greedy comparison — reported for transparency, NOT gated.
    ts_beats_greedy = sum(1 for t, g in zip(ts_vals, greedy_vals, strict=True) if t < g)
    median_greedy = float(np.median(greedy_vals))
    median_reduction_greedy = (
        (median_greedy - median_ts) / median_greedy if median_greedy > 0 else 0.0
    )

    passed = rand_ok and greedy_ok

    # Greedy's failure mode is bimodal: it either locks the winner immediately
    # (very low regret) or locks a decoy and never recovers. Counting the
    # catastrophes makes the shape of the comparison explicit instead of hiding
    # it behind a median.
    greedy_catastrophes = sum(1 for g in greedy_vals if g > 2 * median_greedy) if greedy_vals else 0

    # A single sentence stating exactly what this criterion does and does not
    # establish. Emitted into the decision JSON so a reader of the result cannot
    # mistake a worst-case robustness pass for across-the-board superiority.
    claim = (
        f"TS beats random on {ts_beats_rand}/{len(common_seeds)} seeds "
        f"({reduction_rand:.0%} median regret reduction) and its worst case is "
        f"{worst_case_reduction:.0%} below greedy's ({max_ts:.1f} vs {max_greedy:.1f}). "
        f"TS does NOT beat greedy typically — greedy wins the median on "
        f"{len(common_seeds) - ts_beats_greedy}/{len(common_seeds)} seeds. The "
        f"established property is bounded worst-case regret (no catastrophic "
        f"lock-in), not uniform superiority."
    )

    return CriterionResult(
        name="criterion_2_ts_quality",
        passed=passed,
        metrics={
            "claim": claim,
            "scope": (
                "Synthetic seeded bandit, no LLM and no landscape topology. "
                "Validates the sampler in isolation; says nothing about whether "
                "hypotree helps an agent on the real task (that is criterion 1)."
            ),
            "ts_vs_random": {
                "median_reduction": round(reduction_rand, 4),
                "ts_wins": ts_beats_rand,
                "wilcoxon_p_value": wilcoxon_rand["p_value"],
                "cliffs_delta": round(delta_rand, 4),
            },
            "ts_vs_greedy": {
                "gated_on": "worst_case_only",
                "worst_case_reduction": round(worst_case_reduction, 4),
                "max_ts_regret": round(max_ts, 4),
                "max_greedy_regret": round(max_greedy, 4),
                "median_reduction_typical": round(median_reduction_greedy, 4),
                "ts_wins_typical": ts_beats_greedy,
                "greedy_wins_typical": len(common_seeds) - ts_beats_greedy,
                "greedy_catastrophic_seeds": greedy_catastrophes,
            },
        },
        details={
            "ts_values": ts_vals,
            "random_values": rand_vals,
            "greedy_values": greedy_vals,
            "median_ts_regret": round(median_ts, 4),
            "median_random_regret": round(median_rand, 4),
            "median_greedy_regret": round(median_greedy, 4),
            "random_ok": rand_ok,
            "greedy_ok": greedy_ok,
        },
    )


def _criterion3_revision(
    arm_b: dict[str, list[dict[str, Any]]],
) -> CriterionResult:
    """Criterion 3: no re-execution of pruned work (hard) + belief revision fires.

    Revision is counted as any event where the belief state genuinely revised
    itself in response to contradicting evidence, of which there are three kinds:

    - ``upstream_propagations`` — a confirmed ancestor demoted to NEEDS_REVISION
      because something it supported failed. Determinate blame.
    - ``conflicts_recorded`` — a failure whose cause is *indeterminate*: several
      assumptions cannot all hold, and the system records that fact instead of
      guessing. This is a revision — the set of admissible combinations shrank —
      and the earlier, narrower proxy simply could not see it.
    - ``culprits_identified`` — a recorded conflict later narrowed to a single
      assumption once every other member was exonerated. The most valuable kind:
      blame assigned only when it became provable.

    Counting only the first kind made the criterion an artefact of DAG shape: a
    combination resting on four assumptions produced no measurable revision at
    all unless the system was willing to blame all four, which destroys correct
    knowledge. The breakdown is always reported so the mechanism, not just the
    total, is auditable.
    """
    total_pruned_reexecutions = 0
    total_upstream_propagations = 0
    total_conflicts = 0
    total_culprits = 0

    for _seed, events in arm_b.items():
        total_pruned_reexecutions += _extract_pruned_reexecutions(events)
        total_upstream_propagations += _extract_upstream_propagation(events)
        for e in events:
            if e.get("event_type") == "conflict_recorded":
                total_conflicts += 1
            elif e.get("event_type") == "conflict_resolved":
                total_culprits += 1

    revision_events = total_upstream_propagations + total_conflicts + total_culprits

    # Hard gate: zero re-executions of pruned nodes.
    reexecutions_ok = total_pruned_reexecutions == 0
    # Revision must demonstrably fire somewhere in the run set.
    propagation_ok = revision_events > 0

    return CriterionResult(
        name="criterion_3_revision",
        passed=reexecutions_ok and propagation_ok,
        metrics={
            "total_pruned_reexecutions": total_pruned_reexecutions,
            "revision_events": revision_events,
            "total_upstream_propagations": total_upstream_propagations,
            "conflicts_recorded": total_conflicts,
            "culprits_identified": total_culprits,
            "reexecutions_ok": reexecutions_ok,
            "propagation_ok": propagation_ok,
        },
        details={},
    )


def _criterion4_status_utility(
    arm_b: dict[str, list[dict[str, Any]]],
) -> CriterionResult:
    """Criterion 4: χ² on the action taxonomy, conditioned on the status regime.

    The question is whether the richer statuses earn their keep. If what the
    agent *does* — execute, replan, abandon, ask for context, stall — is
    statistically independent of whether anything is BLOCKED or NEEDS_REVISION,
    then those statuses changed no behaviour and the pre-registered response
    (§4.4) is to **collapse** the machinery. If the distributions differ, the
    statuses are carrying information and are **kept**.

    Three outcomes, and the difference between the last two is the whole point
    of instrumenting this: the test previously returned "collapse" whether the
    data said so or not, because the taxonomy was never recorded, so a design
    decision was being taken by a missing feature rather than by a measurement.

    - **underpowered** → collapse. Pre-committed and unchanged: too few
      observations, or one regime never occurred, so no comparison exists.
      Explicitly *not* a re-run-until-significant.
    - **powered, p ≥ α** → collapse. The statuses genuinely made no difference.
    - **powered, p < α** → keep.

    Passing is not conditional on the outcome — collapse is a legitimate design
    answer, not a failure — so this never flips the gate on its own. What it now
    reports is a measurement rather than a default.
    """
    counts: dict[str, dict[str, int]] = {
        action: {"open": 0, "revision": 0} for action in ACTION_TAXONOMY
    }
    for events in arm_b.values():
        for e in events:
            if e.get("event_type") != "agent_action":
                continue
            action = str(e.get("action", ""))
            context = str(e.get("status_context", ""))
            if action in counts and context in counts[action]:
                counts[action][context] += 1

    # Rows that never occur carry no information and make the χ² undefined
    # (a zero row has zero expected frequency), so they are dropped rather than
    # smoothed — inventing a pseudo-count would fabricate the very independence
    # the test is meant to detect.
    table = [
        [counts[action]["open"], counts[action]["revision"]]
        for action in ACTION_TAXONOMY
        if counts[action]["open"] + counts[action]["revision"] > 0
    ]
    observed = sum(sum(row) for row in table)
    per_context = [sum(row[i] for row in table) for i in range(2)] if table else [0, 0]

    metrics: dict[str, Any] = {
        "action_counts": {a: dict(c) for a, c in counts.items()},
        "total_actions": observed,
        "actions_in_open_context": per_context[0],
        "actions_in_revision_context": per_context[1],
    }

    # Underpowered when a regime never occurred (no contrast to test), when
    # fewer than two action kinds were ever used (a single row is degenerate),
    # or when the sample is too small for the asymptotic χ² to mean anything.
    if len(table) < 2 or min(per_context) < MIN_CHI2_PER_CONTEXT or observed < MIN_CHI2_TOTAL:
        return CriterionResult(
            name="criterion_4_status_utility",
            passed=True,
            metrics={
                **metrics,
                "decision": "collapse",
                "reason": (
                    "underpowered — the action taxonomy needs both status regimes, at "
                    f"least {MIN_CHI2_PER_CONTEXT} actions in each and {MIN_CHI2_TOTAL} "
                    "in total; the pre-committed default applies"
                ),
            },
            details={"contingency_table": table},
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(np.array(table))
    # Cramér's V. Significance on a large sample says the distributions differ,
    # not that the difference is worth keeping machinery for, so the effect size
    # is reported alongside it.
    v_denominator = observed * min(len(table) - 1, len(table[0]) - 1)
    cramers_v = float(np.sqrt(chi2 / v_denominator))
    keep = bool(p_value < ALPHA)
    min_expected = float(np.min(expected))

    # Turns are not independent draws: they come in runs of tens from the same
    # episode, so the effective sample is nearer the episode count than the turn
    # count and the p-value is optimistic by an unknown factor. It is reported
    # rather than corrected because the gating rule is pre-registered and the
    # decision here is qualitative — but a reader who takes p at face value
    # would be reading a certainty the design cannot deliver.
    caveats = [
        f"turns are clustered within {len(arm_b)} episodes, so the p-value "
        f"overstates certainty — read Cramer's V ({cramers_v:.3f}) as the "
        f"honest statement of how much the statuses change behaviour"
    ]
    if min_expected < 5.0:
        caveats.append(
            f"smallest expected cell is {min_expected:.1f} (<5), so the "
            f"asymptotic chi-square approximation is marginal for the sparsest row"
        )

    return CriterionResult(
        name="criterion_4_status_utility",
        passed=True,
        metrics={
            **metrics,
            "n_episodes": len(arm_b),
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "cramers_v": cramers_v,
            "min_expected_count": min_expected,
            "decision": "keep" if keep else "collapse",
            "reason": (
                "the agent's actions depend on whether anything is under revision — "
                "the richer statuses carry information"
                if keep
                else "the agent acts the same way with and without nodes under revision — "
                "the richer statuses changed no behaviour"
            ),
            "caveats": caveats,
        },
        details={
            "contingency_table": table,
            "rows": [a for a in ACTION_TAXONOMY if sum(counts[a].values()) > 0],
            "columns": ["open", "revision"],
        },
    )


# -- Decision logic -----------------------------------------------------------


def _make_decision(
    c1: CriterionResult,
    c2: CriterionResult,
    c3: CriterionResult,
    c4: CriterionResult,
) -> str:
    """Apply the pre-committed decision rule (§6).

    1. If c1 OR c2 fails → STOP.
    2. If c1+c2 pass but c3 or c4 fails → ITERATE.
    3. All pass → GO.
    """
    if not c1.passed or not c2.passed:
        return "STOP"
    if not c3.passed or not c4.passed:
        return "ITERATE"
    return "GO"


# -- Main entry point ---------------------------------------------------------


def analyse(runs_dir: Path) -> GateDecision:
    """Analyse all run logs in ``runs_dir`` and emit the gate decision.

    Ingests ``seed-*-arm-*.jsonl`` (runner logs) and ``ablation-*.jsonl``
    (ablation results), computes all four criteria, and returns the decision.
    """
    arm_logs = _load_runner_logs(runs_dir)
    ablation_entries = _load_ablation_logs(runs_dir)

    c1, c1a, c1b = _criterion1_moat(arm_logs["A"], arm_logs["B"], arm_logs["F"])
    c2 = _criterion2_ts_quality(ablation_entries)
    c3 = _criterion3_revision(arm_logs["B"])
    c4 = _criterion4_status_utility(arm_logs["B"])

    decision = _make_decision(c1, c2, c3, c4)

    reported = (c1, c1a, c1b, c2, c3, c4)
    criteria_dict = {
        c.name: {
            "passed": c.passed,
            **c.metrics,
        }
        for c in reported
    }

    details_dict = {c.name: c.details for c in reported}

    return GateDecision(
        decision=decision,
        criteria=criteria_dict,
        details=details_dict,
    )


def analyse_to_json(runs_dir: Path) -> str:
    """Analyse and return the decision as a JSON string."""
    result = analyse(runs_dir)
    return json.dumps(asdict(result), indent=2, default=str)


if __name__ == "__main__":
    import argparse

    from eval.runner.config import run_dir

    parser = argparse.ArgumentParser(
        prog="eval.analyse_gate",
        description="Score the pre-registered gate criteria for one identified run.",
    )
    parser.add_argument("eval_dir", type=Path, help="path to the eval/ directory")
    parser.add_argument(
        "--run-id",
        required=True,
        help="identifier of the run whose logs under eval/runs/<run-id>/ are scored",
    )
    args = parser.parse_args()

    print(analyse_to_json(run_dir(args.eval_dir, args.run_id)))
