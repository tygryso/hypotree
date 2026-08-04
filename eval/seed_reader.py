"""Render one evaluation run's JSONL logs as a self-contained markdown report.

    uv run python -m eval.seed_reader --run-id 2026-07-27a
    uv run python -m eval.seed_reader --run-id 2026-07-27a --output eval/runs/2026-07-27a/REPORT.md

Reads every ``eval/runs/<run-id>/seed-*-arm-*.jsonl`` (plus any ablation logs)
and emits a markdown document that answers, without the reader ever opening a
raw log:

* how each arm performed, and whether the paired differences are real;
* where the budget went — duplicates, wasted probes, probe composition;
* what the belief state actually did — exclusion groups, cascades, conflicts;
* what it cost — LLM turns, wall-clock, tokens;
* which numbers are untrustworthy, stated explicitly rather than left implicit.

Every derived metric names the raw events it came from, so any claim in the
report can be traced back to the log lines that produced it. Sections are
ordered from conclusion to evidence: an LLM (or a human) that reads only the
first two sections should still get the correct headline.

The ``--run-id`` is mandatory. Reports are only comparable within a run, and a
report that silently mixed two runs' logs would be worse than no report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.environment.landscape_scoring import (
    AXES,
    TARGET_METRIC,
    contains_decoy,
    is_full_combination,
    is_premise_probe,
    n_correct_axes,
    reference_strategy_probes,
)
from eval.runner.config import ALL_ARMS, ARM_B, run_dir, run_workspace_id
from hypotree.engine import (
    DEAD_QUESTION_PREFIX,
    DEDUCTION_REASON_PREFIX,
    DEDUCTION_RETRACT_PREFIX,
    EXCLUSION_REASON_PREFIX,
    INTERACTION_REOPEN_PREFIX,
    SUBSTITUTION_ELIMINATE_PREFIX,
    UNDERPERFORMANCE_REOPEN_PREFIX,
)

# Arm labels, expanded once here so the report explains itself to a reader who
# has never seen the pre-registration.
ARM_LEGEND: dict[str, str] = {
    "B": "hypotree belief-state tools (treatment)",
    "F": "auto-persisted flat probe transcript + manual notes (informational baseline)",
    "A": "manual markdown scratchpad only (ergonomic baseline)",
}

# The status transition that marks an exclusion being retracted: a sibling that
# had been ruled out by a confirmed alternative is handed back to the frontier.
_REOPEN_TRANSITION = ("EXHAUSTED", "UNTESTED")

# Markers the engine writes into a status-history reason. Mechanisms are
# identified by these rather than by the (old, new) status pair, because the pair
# is ambiguous: UNTESTED->EXHAUSTED and IN_PROGRESS->EXHAUSTED are both the
# exclusion inference, and IN_PROGRESS->VERIFIED is either direct evidence or
# deduction by elimination. Matching on the pair reported zero for mechanisms
# that were demonstrably firing.
#
# Imported from the engine rather than restated here: the previous version of
# this analysis matched a literal that the engine never wrote, and a counter
# that can only ever report zero is worse than no counter at all.
_EXCLUDE_REASON = EXCLUSION_REASON_PREFIX
_DEDUCE_REASON = DEDUCTION_REASON_PREFIX
_SUBSTITUTE_OUT_REASON = SUBSTITUTION_ELIMINATE_PREFIX
_INTERACTION_REASON = INTERACTION_REOPEN_PREFIX
_SHORTFALL_REASON = UNDERPERFORMANCE_REOPEN_PREFIX
_DEAD_QUESTION_REASON = DEAD_QUESTION_PREFIX
_DEDUCTION_RETRACT_REASON = DEDUCTION_RETRACT_PREFIX

# Statuses a node lands in only because something else went wrong. These, and
# only these, are the destructive propagation the belief state is supposed to
# keep rare.
_REVISION_STATUSES = frozenset({"NEEDS_REVISION", "PRUNED"})

# Statuses that mean "this node is settled — the navigator should not hand it
# back". Re-selecting one is the failure mode the sampler is meant to avoid.
# NEEDS_REVISION is absent on purpose: a node under conflict review is supposed
# to come back, that is the whole mechanism.
_SETTLED_STATUSES = frozenset({"VERIFIED", "INVALIDATED", "PRUNED", "EXHAUSTED"})


# -- Log loading ---------------------------------------------------------------


@dataclass
class RunLog:
    """Every event of one (seed, arm) episode, plus the derived per-run metrics.

    Metrics are computed once at load time because almost every section of the
    report needs several of them; recomputing per section would make the
    aggregate and per-seed tables silently drift apart.
    """

    seed: int
    arm: str
    path: Path
    events: list[dict[str, Any]]

    # Outcome
    steps: int = 0
    goals_met: bool = False
    end_reason: str = "no_run_end"
    complete: bool = False
    # The episode ended because the inference server stopped answering. Kept
    # separate from ``complete`` so the report can say *why* it was excluded.
    infra_failed: bool = False
    tool_budget: int | None = None
    declared_run_id: str | None = None
    llm_model: str | None = None

    # Probe economy
    experiments: int = 0
    duplicates: int = 0
    distinct_configs: int = 0
    premise_probes: int = 0
    combination_probes: int = 0
    decoy_probes: int = 0
    best_success: float = 0.0
    best_correct_axes: int = 0
    first_hit_step: int | None = None

    # Belief-state mechanics (Arm B)
    nodes_created: int = 0
    nodes_with_group: int = 0
    # Nodes built on top of other hypotheses. They answer no question of their
    # own and so cannot belong to an exclusion group; counting them in the
    # adoption denominator turned that metric into a count of how many
    # combinations an episode happened to build.
    composed_nodes: int = 0
    exclusion_declared: bool = False
    targets_selected: int = 0
    settled_reselects: int = 0
    redispatched: int = 0
    claims_released: int = 0
    # Probe results the agent held when its context was wiped. Before these were
    # carried across the reset they were destroyed outright, costing two probes
    # on every episode that reset mid-batch — one re-probed as a duplicate, one
    # silently retired by the exclusion inference before its result was recorded.
    probes_carried: int = 0
    evidence_records: int = 0
    claimed_records: int = 0
    # (action, status_context) -> count. The contingency table criterion 4 tests.
    agent_actions: Counter[tuple[str, str]] = field(default_factory=Counter)
    review_dispatches: int = 0
    transitions: Counter[str] = field(default_factory=Counter)
    revision_transitions: int = 0
    exclusions_applied: int = 0
    # Values ruled out by a diagnostic swap that stopped the failure without
    # clearing the bar. Free eliminations: the composition was right in every
    # other slot and still fell short, so the swapped-in value cannot be the
    # answer either. Without it the navigator handed the value straight back.
    substitutes_ruled_out: int = 0
    reopened: int = 0
    interaction_reopens: int = 0
    # Reopens triggered by every answer being in and the assembly still falling
    # short. A different mechanism from the interaction case and reported apart:
    # counting them together described run I's six shortfall reopens as conflicts
    # proven to be interaction effects, in the same breath as saying every
    # conflict had been narrowed to a culprit.
    shortfall_reopens: int = 0
    same_question_dispatches: int = 0
    # The distinct members declared per exclusion group. Needed to say what the
    # exclusion yield *should* be: probing a group of k in ignorance costs
    # (k+1)/2 probes and retires the rest, so a run beating that is ordering
    # better than chance and a run matching it is not ordering at all. Node ids
    # rather than a count, because the arm-B prompt tells the agent to re-create
    # nodes with `if_exists="overwrite"` and every re-creation logs `created`
    # again — counting events would inflate k, and k inflates the baseline the
    # run is being judged against.
    group_members: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    done_reasons: Counter[str] = field(default_factory=Counter)
    deduced: int = 0
    # Nodes pruned because every candidate answer to a question they depend on
    # was ruled out on its own evidence. Counted apart from the refutation
    # cascade because it reaches what that cascade deliberately spares — a
    # premise settled as EXHAUSTED was never refuted, so its dependents survive
    # until this fires.
    dead_question_prunes: int = 0
    # Deductions handed back to the frontier because a composition resting on
    # them fell short. Each one is the engine catching its own closed-world
    # assumption being wrong — a free confirmation that turned out not to be
    # free — and each costs exactly one probe to settle.
    deductions_withdrawn: int = 0
    pruned_reexecutions: int = 0
    conflicts_recorded: int = 0
    conflict_members: int = 0
    conflicts_resolved: int = 0
    # Composition probes spent between recording a conflict and naming its
    # culprit. This is what the diagnosis ordering actually buys, and it was
    # invisible: the report said how many conflicts resolved but never what they
    # cost, so an ordering that quietly doubled the swaps read as a clean run.
    diagnosis_swaps: list[int] = field(default_factory=list)
    # Results the engine refused inside a batch while the rest still applied.
    records_rejected: int = 0
    revision_fired: bool = False

    # Memory maintenance
    session_resets: int = 0
    summary_lengths: list[int] = field(default_factory=list)
    scratchpad_writes: int = 0
    scratchpad_chars: int = 0
    regime_overrides: int = 0

    # Cost
    llm_calls: int = 0
    llm_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls_issued: int = 0
    tool_histogram: Counter[str] = field(default_factory=Counter)
    tool_errors: Counter[str] = field(default_factory=Counter)
    finish_reasons: Counter[str] = field(default_factory=Counter)
    probe_depths: Counter[int] = field(default_factory=Counter)
    goal_contamination: int = 0

    @property
    def has_conflict(self) -> bool:
        """Whether the episode ever hit an indeterminate multi-assumption failure.

        The single most explanatory split available: on the previous run the
        arms were level overall, while episodes without a conflict beat the
        baseline by 25% and episodes with one lost to it by 12%.
        """
        return self.conflicts_recorded > 0

    @property
    def scored_steps(self) -> int:
        """The step count the gate scores this episode on.

        An episode that never reached the goal has no defensible step count: it
        did not solve the task, so the probes it happened to spend before giving
        up are not a measure of how quickly it *would* have. The frozen gate
        right-censors those to the tool budget, and this report must do the same
        or the two disagree about the same run.

        They did. A crashed episode that probed **nothing** was scored here as a
        27-step win over the baseline while the gate scored it as a 73-step loss,
        and the report was the optimistic one — it showed a significant result
        (p=0.033) where the gate showed none (p=0.067). Raw ``steps`` stays
        available for diagnosis; every comparison uses this.
        """
        if self.goals_met:
            return self.steps
        return self.tool_budget if self.tool_budget else self.steps

    @property
    def wasted_probes(self) -> int:
        """Probes beyond what a perfect-recall reference strategy would need.

        Negative values are clamped to 0: finishing under the reference count
        means the agent got lucky on an early probe, not that it produced
        negative waste.
        """
        return max(0, self.steps - reference_strategy_probes(self.seed))

    @property
    def duplicate_rate(self) -> float:
        return self.duplicates / self.experiments if self.experiments else 0.0

    @property
    def groupable_nodes(self) -> int:
        """Nodes that could have declared which question they answer."""
        return self.nodes_created - self.composed_nodes

    @property
    def group_adoption(self) -> float:
        """Fraction of *groupable* nodes that declared which question they answer."""
        total = self.groupable_nodes
        return self.nodes_with_group / total if total else 0.0


def _parse_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL log, skipping unparseable lines.

    A truncated final line is normal for a run that was killed mid-write; it
    must not take the whole report down.
    """
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _build_run_log(path: Path, events: list[dict[str, Any]]) -> RunLog:
    """Fold one episode's event stream into a RunLog."""
    seed = next((e["seed"] for e in events if "seed" in e), 0)
    arm = next((e["arm"] for e in events if "arm" in e), "?")
    log = RunLog(seed=seed, arm=arm, path=path, events=events)
    # Nodes dispatched and not yet reported on. A node dispatched while already
    # in this set was handed out under a live lease, which is a violation; a node
    # dispatched after being reported is ordinary re-sampling and must not be
    # counted, or a stochastic node would look like a bug on every extra sample.
    outstanding: set[str] = set()
    # Exclusion groups dispatched in the batch currently being assembled. A
    # batch is delimited by the get_next_targets tool_call that follows its
    # target_selected events. Two members of one group in the same batch means
    # the first answer cannot retire the second, so the second probe is spent on
    # a question the batch had already asked.
    batch_groups: list[str] = []
    # Probe count when the currently-open conflict was recorded, so the swaps it
    # took to name a culprit can be attributed to it.
    open_conflict_at: dict[str, int] = {}

    for ev in events:
        kind = ev.get("event_type")

        if kind == "run_start":
            log.tool_budget = ev.get("tool_budget")
            log.declared_run_id = ev.get("run_id")
            log.llm_model = ev.get("llm_model")

        elif kind == "run_end":
            # Last run_end wins: a resumed episode appends a second one.
            log.steps = ev.get("step", 0)
            log.goals_met = bool(ev.get("goals_met"))
            log.end_reason = str(ev.get("reason", "?"))
            # An episode the inference server killed is not a result. It reached
            # run_end, so it is structurally complete, but treating it as
            # complete would score a dropped connection as a failure to solve the
            # task — so it is excluded from paired comparison exactly like an
            # episode that never finished at all.
            log.infra_failed = bool(ev.get("infra_failed"))
            log.complete = not log.infra_failed

        elif kind == "experiment":
            log.experiments += 1
            if ev.get("duplicate"):
                log.duplicates += 1
            log.distinct_configs = ev.get("distinct_configs", log.distinct_configs)
            config = str(ev.get("config", ""))
            if is_premise_probe(config):
                log.premise_probes += 1
            if is_full_combination(config):
                log.combination_probes += 1
            if contains_decoy(config, seed):
                log.decoy_probes += 1
            success = float(ev.get("success", 0.0) or 0.0)
            log.best_success = max(log.best_success, success)
            log.best_correct_axes = max(log.best_correct_axes, n_correct_axes(config, seed))
            log.probe_depths[int(ev.get("depth") or 0)] += 1
            if success >= TARGET_METRIC and log.first_hit_step is None:
                log.first_hit_step = ev.get("step")

        elif kind == "node_created":
            if ev.get("created"):
                log.nodes_created += 1
                if ev.get("composed"):
                    log.composed_nodes += 1
                if ev.get("exclusion_group"):
                    log.nodes_with_group += 1
                    log.exclusion_declared = True
                    log.group_members[str(ev["exclusion_group"])].add(str(ev.get("node_id", "")))

        elif kind == "target_selected":
            if not ev.get("node_id"):
                # A DONE sentinel. Its reason is the navigator's instruction —
                # report what you hold, run this swap, compose the answers, or
                # fix an unreachable graph — and counting them is the only way to
                # see whether the recovery paths fired at all.
                if ev.get("reason"):
                    log.done_reasons[str(ev["reason"])] += 1
            else:
                log.targets_selected += 1
                node_id = str(ev["node_id"])
                if node_id in outstanding:
                    log.redispatched += 1
                outstanding.add(node_id)
                status = str(ev.get("status", "")).upper()
                if status in _SETTLED_STATUSES:
                    log.settled_reselects += 1
                group = ev.get("exclusion_group")
                if group:
                    if group in batch_groups:
                        log.same_question_dispatches += 1
                    batch_groups.append(str(group))

        elif kind == "claims_released":
            log.claims_released += int(ev.get("count") or 0)
            outstanding.difference_update(ev.get("node_ids") or [])

        elif kind == "probes_carried":
            log.probes_carried += len(ev.get("configs") or [])

        elif kind == "agent_action":
            log.agent_actions[(str(ev.get("action", "")), str(ev.get("status_context", "")))] += 1

        elif kind == "evidence_recorded":
            log.evidence_records += 1
            if ev.get("claimed"):
                # Answered a dispatch. Records without a claim are compositions
                # the agent assembled itself, which the navigator never handed
                # out — counting them against dispatches measures nothing.
                log.claimed_records += 1
            outstanding.discard(str(ev.get("node_id", "")))

        elif kind == "goal_evidence_refused":
            # The agent probed something and tried to file the answer against
            # the objective. The engine refuses it, so nothing is corrupted — but
            # the probe was still paid for, the hypothesis it tested is still
            # untested, and a failure that should have implicated assumptions
            # explained nothing. Agents did this in a third of episodes despite
            # an explicit prohibition, so it is counted rather than assumed
            # absent.
            log.goal_contamination += 1

        elif kind == "status_transition":
            old, new = str(ev.get("old_status")), str(ev.get("new_status"))
            reason = str(ev.get("reason", ""))
            log.transitions[f"{old}->{new}"] += 1
            # Only genuinely destructive propagation counts as revision. The
            # exclusion inference is also flagged `propagated`, and folding the
            # two together made the feature working look like the feature
            # failing — it put every episode in the "cascade fired" cell.
            if new in _REVISION_STATUSES:
                log.revision_transitions += 1
                log.revision_fired = True
            # The snapshot only emits when (status, reason) actually changed, so
            # every event here is a real one. An exclusion recorded at unchanged
            # status is a *re-*exclusion: `_apply_exclusion` skips any sibling
            # that is not open, so the only way to arrive back at EXHAUSTED is to
            # have passed through UNTESTED when the first confirmation was
            # retracted. Requiring old != new dropped those and undercounted the
            # mechanism.
            if new == "EXHAUSTED" and reason.startswith(_EXCLUDE_REASON):
                log.exclusions_applied += 1
            if new == "EXHAUSTED" and reason.startswith(_SUBSTITUTE_OUT_REASON):
                log.substitutes_ruled_out += 1
            if (old, new) == _REOPEN_TRANSITION:
                log.reopened += 1
                if reason.startswith(_INTERACTION_REASON):
                    log.interaction_reopens += 1
                elif reason.startswith(_SHORTFALL_REASON):
                    log.shortfall_reopens += 1
            if new == "VERIFIED" and reason.startswith(_DEDUCE_REASON):
                log.deduced += 1
            if new == "PRUNED" and reason.startswith(_DEAD_QUESTION_REASON):
                log.dead_question_prunes += 1
            if new == "UNTESTED" and reason.startswith(_DEDUCTION_RETRACT_REASON):
                log.deductions_withdrawn += 1

        elif kind == "pruned_reexecution":
            log.pruned_reexecutions += 1

        elif kind == "conflict_recorded":
            log.conflicts_recorded += 1
            log.conflict_members += ev.get("n_members", 0)
            # Keyed by conflict, not held in one slot: a second conflict opening
            # before the first resolves used to overwrite the first's mark, and
            # the resolution after that got no attribution at all. Logs written
            # before the id was carried key on their arrival order instead, which
            # is exactly the single-slot behaviour they were read with.
            key = ev.get("nogood_id")
            open_conflict_at[str(key) if key is not None else f"#{len(open_conflict_at)}"] = (
                log.experiments
            )

        elif kind == "conflict_resolved":
            log.conflicts_resolved += 1
            opened = open_conflict_at.pop(str(ev.get("nogood_id")), None)
            if opened is None and open_conflict_at:
                # A legacy log: nothing to match on, so the oldest conflict still
                # open is the only defensible attribution.
                opened = open_conflict_at.pop(next(iter(open_conflict_at)))
            if opened is not None:
                log.diagnosis_swaps.append(log.experiments - opened)

        elif kind == "record_rejected":
            log.records_rejected += 1

        elif kind == "session_reset":
            log.session_resets += 1
            log.summary_lengths.append(ev.get("summary_length", 0))

        elif kind == "scratchpad_write":
            log.scratchpad_writes += 1
            log.scratchpad_chars = max(log.scratchpad_chars, ev.get("total_chars", 0))

        elif kind == "regime_override":
            log.regime_overrides += 1

        elif kind == "tool_call":
            tool = str(ev.get("tool", "?"))
            log.tool_histogram[tool] += 1
            # ``ok`` is absent in logs written before the census was fixed; those
            # only recorded failures, so treating a missing flag as a failure
            # keeps old runs readable instead of silently recoding them as fine.
            if not ev.get("ok", False):
                log.tool_errors[tool] += 1
            # A batch is whatever one tool call handed out, and every dispatch is
            # logged before the call that produced it — so any tool call closes
            # the batch. Keying this on `get_next_targets` alone was correct only
            # while that was the only tool that could dispatch: once a record
            # could carry its own dispatch, nothing ever closed the batch and the
            # count became "every repeat of a question in the whole episode",
            # which for a five-value axis is four false violations per axis.
            batch_groups.clear()

        elif kind == "llm_call":
            log.llm_calls += 1
            log.llm_seconds += float(ev.get("duration_s", 0.0) or 0.0)
            log.prompt_tokens += int(ev.get("prompt_tokens") or 0)
            log.completion_tokens += int(ev.get("completion_tokens") or 0)
            log.tool_calls_issued += int(ev.get("n_tool_calls") or 0)
            if ev.get("finish_reason"):
                log.finish_reasons[str(ev["finish_reason"])] += 1

    # An episode killed before run_end still has a meaningful step count.
    if not log.complete and events:
        log.steps = max((e.get("step", 0) for e in events), default=0)

    return log


def load_run(runs_dir: Path) -> tuple[list[RunLog], list[dict[str, Any]]]:
    """Load every episode log and every ablation record under ``runs_dir``."""
    logs = [
        _build_run_log(p, _parse_events(p)) for p in sorted(runs_dir.glob("seed-*-arm-*.jsonl"))
    ]
    ablation: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("ablation-*.jsonl")):
        ablation.extend(e for e in _parse_events(path) if e.get("event_type") == "ablation_result")
    return logs, ablation


# -- Statistics ----------------------------------------------------------------


def _sign_test_p(wins: int, losses: int) -> float | None:
    """Two-sided exact sign-test p-value for ``wins`` vs ``losses``.

    Ties are excluded, which is the standard treatment: a tie carries no
    directional information. Returns None when every pair tied, since there is
    then no hypothesis left to test. Deliberately non-parametric — step counts
    are bounded, skewed and censored by the tool budget, so a t-test on them
    would be assuming a normality the data does not have.
    """
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass
class Paired:
    """One paired arm-vs-arm comparison over the seeds both arms completed."""

    treatment: str
    baseline: str
    seeds: list[int]
    diffs: list[int]  # baseline steps - treatment steps; positive = treatment faster

    @property
    def n(self) -> int:
        return len(self.diffs)

    @property
    def wins(self) -> int:
        return sum(1 for d in self.diffs if d > 0)

    @property
    def losses(self) -> int:
        return sum(1 for d in self.diffs if d < 0)

    @property
    def ties(self) -> int:
        return sum(1 for d in self.diffs if d == 0)

    @property
    def mean(self) -> float:
        return statistics.mean(self.diffs) if self.diffs else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.diffs) if self.diffs else 0.0

    @property
    def p_value(self) -> float | None:
        return _sign_test_p(self.wins, self.losses)

    @property
    def wilcoxon_p(self) -> float | None:
        """Two-sided Wilcoxon signed-rank p-value, or None when it is undefined.

        Reported next to the sign test because the two answer different
        questions and can disagree sharply: the sign test uses only the
        direction of each pair, Wilcoxon also uses the size of the gap. A run
        where the treatment wins narrowly but often looks weak to one and strong
        to the other, and quoting either alone invites the wrong conclusion.
        """
        if not any(self.diffs):
            return None
        try:
            from scipy import stats

            return float(stats.wilcoxon(self.diffs).pvalue)
        except (ImportError, ValueError):
            return None


def pair_arms(logs: list[RunLog], treatment: str, baseline: str) -> Paired:
    """Pair two arms seed-by-seed, using only seeds where both episodes finished.

    Incomplete episodes are dropped rather than imputed: a crashed run has no
    defensible step count, and substituting the budget would manufacture a
    result in whichever direction the crash happened to fall.
    """
    t = {log.seed: log for log in logs if log.arm == treatment and log.complete}
    b = {log.seed: log for log in logs if log.arm == baseline and log.complete}
    seeds = sorted(set(t) & set(b))
    return Paired(
        treatment=treatment,
        baseline=baseline,
        seeds=seeds,
        diffs=[b[s].scored_steps - t[s].scored_steps for s in seeds],
    )


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


# -- Markdown rendering --------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(no data)_", ""]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    out.append("")
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _by_arm(logs: list[RunLog]) -> dict[str, list[RunLog]]:
    grouped: dict[str, list[RunLog]] = defaultdict(list)
    for log in logs:
        grouped[log.arm].append(log)
    return grouped


def _arm_order(grouped: dict[str, list[RunLog]]) -> list[str]:
    """Arms in pre-registered order, with any unexpected label appended."""
    known = [a for a in ALL_ARMS if a in grouped]
    return known + sorted(set(grouped) - set(known))


def _section_header(run_id: str, runs_dir: Path, logs: list[RunLog]) -> list[str]:
    seeds = sorted({log.seed for log in logs})
    complete = sum(1 for log in logs if log.complete)
    models = sorted({log.llm_model for log in logs if log.llm_model})
    return [
        f"# Evaluation report — run `{run_id}`",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Logs: `{runs_dir}`",
        f"- Belief-state workspace: `{run_workspace_id(run_id)}`",
        f"- Model: {', '.join(models) if models else 'unrecorded'}",
        f"- Episodes: {len(logs)} found, {complete} completed",
        f"- Seeds: {len(seeds)}"
        + (f" ({seeds[0]}..{seeds[-1]})" if seeds else "")
        + f", arms: {', '.join(_arm_order(_by_arm(logs)))}",
        "",
        "Arms:",
        "",
        *[f"- **{a}** — {ARM_LEGEND.get(a, 'unknown arm')}" for a in _arm_order(_by_arm(logs))],
        "",
    ]


def _section_headline(logs: list[RunLog]) -> list[str]:
    """Outcome per arm — the only table a reader in a hurry needs."""
    grouped = _by_arm(logs)
    rows = []
    for arm in _arm_order(grouped):
        done = [log for log in grouped[arm] if log.complete]
        stats = _describe([float(log.scored_steps) for log in done])
        met = sum(1 for log in done if log.goals_met)
        rows.append(
            [
                arm,
                str(int(stats["n"])),
                f"{stats['mean']:.2f}",
                f"{stats['median']:.1f}",
                f"{stats['stdev']:.2f}",
                f"{stats['min']:.0f}–{stats['max']:.0f}",
                f"{met}/{len(done)}" if done else "0/0",
                f"{statistics.mean([log.wasted_probes for log in done]):.1f}" if done else "–",
            ]
        )
    return [
        "## 1. Headline — steps to target",
        "",
        "`steps` counts dispatched experiments (one `experiment` event each), so it is the "
        "budget the agent actually spent — right-censored to the tool budget for an episode "
        "that never reached the goal, exactly as the frozen gate scores it. An episode that "
        "gave up early did not solve the task quickly. `waste` is steps above the "
        "perfect-recall reference strategy for that seed.",
        "",
        *_table(
            ["arm", "n", "mean", "median", "stdev", "range", "goals met", "waste"],
            rows,
        ),
    ]


def _section_paired(logs: list[RunLog]) -> list[str]:
    """Paired comparisons — the pre-registered criterion-1 statistics."""
    out = [
        "## 2. Paired comparisons",
        "",
        "Diffs are `baseline steps − treatment steps` on seeds both arms completed: "
        "**positive means the treatment was faster**. Two non-parametric tests are "
        "reported because they can disagree: the sign test uses only the direction of "
        "each pair, Wilcoxon also uses how large the gap was. Step counts are skewed "
        "and budget-censored, so neither assumes normality.",
        "",
    ]
    rows = []
    for treatment, baseline in (("B", "F"), ("B", "A"), ("F", "A")):
        pair = pair_arms(logs, treatment, baseline)
        if not pair.n:
            continue
        p = pair.p_value
        wp = pair.wilcoxon_p
        rows.append(
            [
                f"{treatment} vs {baseline}",
                str(pair.n),
                f"{pair.wins}/{pair.losses}/{pair.ties}",
                f"{pair.mean:+.2f}",
                f"{pair.median:+.1f}",
                "–" if p is None else f"{p:.4f}",
                "–" if wp is None else f"{wp:.4f}",
            ]
        )
    out += _table(
        [
            "comparison",
            "n",
            "win/loss/tie",
            "mean diff",
            "median diff",
            "sign p",
            "wilcoxon p",
        ],
        rows,
    )

    bf = pair_arms(logs, "B", "F")
    if bf.n:
        out += [
            "Per-seed `F − B` (the informational-moat comparison — both arms keep every raw "
            "fact for free, so any gap is attributable to structure alone):",
            "",
            "```",
            ", ".join(f"{s}:{d:+d}" for s, d in zip(bf.seeds, bf.diffs, strict=True)),
            "```",
            "",
        ]
    return out


def _section_per_seed(logs: list[RunLog]) -> list[str]:
    """One row per episode — the table every deeper question starts from."""
    rows = []
    for log in sorted(logs, key=lambda x: (x.seed, x.arm)):
        rows.append(
            [
                str(log.seed),
                log.arm,
                str(log.steps),
                "yes" if log.goals_met else "no",
                log.end_reason,
                str(log.duplicates),
                str(log.wasted_probes),
                str(log.revision_transitions),
                str(log.exclusions_applied),
                str(log.reopened),
                str(log.deduced),
                str(log.conflicts_recorded),
                _pct(log.group_adoption) if log.groupable_nodes else "–",
                f"{log.best_success:.2f}",
            ]
        )
    return [
        "## 3. Per-episode detail",
        "",
        "`revision` counts only genuinely destructive propagation (NEEDS_REVISION and "
        "PRUNED). `excluded` is the exclusion inference retiring a settled question — "
        "the efficiency mechanism working, not a failure — and `deduced` is a member "
        "confirmed by elimination without spending a probe.",
        "",
        *_table(
            [
                "seed",
                "arm",
                "steps",
                "goal",
                "end reason",
                "dup",
                "waste",
                "revision",
                "excluded",
                "reopened",
                "deduced",
                "conflicts",
                "group adoption",
                "best score",
            ],
            rows,
        ),
    ]


def _section_probe_economy(logs: list[RunLog]) -> list[str]:
    """Where the budget went, independent of whether the goal was reached."""
    grouped = _by_arm(logs)
    rows = []
    for arm in _arm_order(grouped):
        done = [log for log in grouped[arm] if log.complete]
        if not done:
            continue
        experiments = sum(log.experiments for log in done)
        duplicates = sum(log.duplicates for log in done)
        premise = sum(log.premise_probes for log in done)
        combo = sum(log.combination_probes for log in done)
        decoy = sum(log.decoy_probes for log in done)
        rows.append(
            [
                arm,
                str(experiments),
                f"{duplicates} ({_pct(duplicates / experiments) if experiments else '–'})",
                f"{premise} ({_pct(premise / experiments) if experiments else '–'})",
                f"{combo} ({_pct(combo / experiments) if experiments else '–'})",
                str(decoy),
                f"{statistics.mean([log.best_correct_axes for log in done]):.2f}",
                f"{statistics.mean([log.wasted_probes for log in done]):.1f}",
            ]
        )
    return [
        "## 4. Probe economy",
        "",
        "A duplicate is a re-probe of a config already probed in the same episode. The "
        "oracle is deterministic, so a duplicate returns a known answer and buys nothing — "
        "it is a direct measure of memory failure. Decoy probes are configs containing the "
        "seed's decoy value, which confirms at shallow depth and refutes at depth. "
        f"`best axes` is how many of the {len(AXES)} axes the closest probe got right "
        f"(max {len(AXES)}) — it "
        "separates 'never found the answer' from 'found it one axis short'.",
        "",
        *_table(
            [
                "arm",
                "probes",
                "duplicates",
                "premise",
                "combination",
                "decoy",
                "best axes",
                "mean waste",
            ],
            rows,
        ),
        "",
        *_exclusion_yield_lines(logs),
    ]


def _exclusion_yield_lines(logs: list[RunLog]) -> list[str]:
    """How much of the premise search the exclusion inference paid for.

    Every question is settled exactly once, either by a probe or by the
    exclusion inference retiring it for free, so premise probes and exclusions
    are two halves of one fixed total. That makes the split the single lever on
    premise cost, and it has a known baseline: probing a group of k answers in
    ignorance costs (k+1)/2 probes and retires the remaining (k-1)/2, a yield of
    (k-1)/2k. Matching that baseline means the ordering carries no signal;
    beating it means something is steering. Reported because two runs differing
    by a probe an episode differed here and nowhere else, and without the
    baseline there was no way to tell an ordering win from a lucky draw.
    """
    b_logs = [log for log in logs if log.arm == "B" and log.complete]
    if not b_logs:
        return []
    exclusions = sum(log.exclusions_applied for log in b_logs)
    premise = sum(log.premise_probes for log in b_logs)
    settled = exclusions + premise
    if not settled:
        return []
    sizes = [len(ids) for log in b_logs for ids in log.group_members.values() if len(ids) > 1]
    if not sizes:
        return []
    k = statistics.mean(sizes)
    baseline = (k - 1) / (2 * k)
    yield_ = exclusions / settled
    # A band around the baseline, not a threshold above it. The previous form
    # tested `> baseline + 0.02` against `< baseline + 0.02`, which left "at the
    # baseline" reachable only on exact float equality and reported a run
    # performing *below* chance in the same words as one performing at it.
    margin = 0.02
    if yield_ > baseline + margin:
        verdict = "ordering better than chance"
    elif yield_ < baseline - margin:
        verdict = "ordering worse than probing the answers in a random order"
    else:
        verdict = "at the baseline — the ordering is carrying no signal"
    return [
        f"**Exclusion yield (arm B): {_pct(yield_)}** of premise questions were settled "
        f"without a probe ({exclusions} retired free, {premise} probed). With a mean group "
        f"of {k:.1f} answers, blind ordering yields {_pct(baseline)} — so this run is "
        f"**{verdict}**. This is the only lever on premise cost: every question is settled "
        f"exactly once, so a probe saved here is a probe saved outright.",
    ]


def _section_belief_state(logs: list[RunLog]) -> list[str]:
    """Arm-B mechanics: what the belief state actually did with the evidence."""
    b_logs = [log for log in logs if log.arm == "B" and log.complete]
    if not b_logs:
        return ["## 5. Belief-state mechanics (arm B)", "", "_(no completed arm-B episodes)_", ""]

    nodes = sum(log.nodes_created for log in b_logs)
    groupable = sum(log.groupable_nodes for log in b_logs)
    grouped_nodes = sum(log.nodes_with_group for log in b_logs)
    targets = sum(log.targets_selected for log in b_logs)
    reselects = sum(log.settled_reselects for log in b_logs)
    conflicts = sum(log.conflicts_recorded for log in b_logs)
    members = sum(log.conflict_members for log in b_logs)
    resolved = sum(log.conflicts_resolved for log in b_logs)

    reported = sum(log.evidence_records for log in b_logs)
    claimed = sum(log.claimed_records for log in b_logs)
    composed = reported - claimed
    unreported = max(0, targets - claimed)
    summary_rows = [
        ["nodes created", str(nodes), "`node_created` events that actually created a node"],
        [
            "declared an exclusion group",
            f"{grouped_nodes} ({_pct(grouped_nodes / groupable) if groupable else '–'})",
            "share of the nodes that *could* declare one — compositions rest on other "
            "hypotheses and answer no question of their own, so they are excluded",
        ],
        [
            "episodes using exclusion groups",
            f"{sum(1 for log in b_logs if log.exclusion_declared)}/{len(b_logs)}",
            "per-episode adoption, not per-node",
        ],
        ["navigator targets handed out", str(targets), "`target_selected` with a node"],
        [
            "results reported against a dispatch",
            f"{claimed} ({_pct(claimed / targets) if targets else '–'})",
            "records carrying the claim the navigator issued",
        ],
        [
            "results the agent initiated itself",
            str(composed),
            "compositions it assembled and probed without being asked — never dispatched, "
            "so they are progress, not waste",
        ],
        [
            "dispatches never reported",
            f"{unreported} ({_pct(unreported / targets) if targets else '–'})",
            "probes the agent was given and never recorded — work paid for and lost",
        ],
        [
            "nodes handed out under a live lease",
            str(sum(log.redispatched for log in b_logs)),
            "should be 0 — dispatched again before the previous result was reported",
        ],
        [
            "leases released at a context reset",
            str(sum(log.claims_released for log in b_logs)),
            "nodes reclaimed because the agent could no longer report on them",
        ],
        [
            "results carried across a context reset",
            str(sum(log.probes_carried for log in b_logs)),
            "probed but not yet recorded when the reset landed — handed forward rather "
            "than destroyed, which is what the flat-transcript arm gets for free",
        ],
        [
            "targets already settled",
            f"{reselects} ({_pct(reselects / targets) if targets else '–'})",
            "should be 0 — a settled node handed back is wasted budget",
        ],
        [
            "competing answers in one batch",
            str(sum(log.same_question_dispatches for log in b_logs)),
            "should be 0 — two members of one exclusion group dispatched together, "
            "so the first result cannot retire the second",
        ],
        [
            "exclusions applied",
            str(sum(log.exclusions_applied for log in b_logs)),
            "questions retired by a confirmed answer — the mechanism working",
        ],
        [
            "members deduced by elimination",
            str(sum(log.deduced for log in b_logs)),
            "last candidate standing, confirmed without spending a probe",
        ],
        [
            "deductions withdrawn for testing",
            str(sum(log.deductions_withdrawn for log in b_logs)),
            "a free confirmation whose composition then fell short — either the value is "
            "wrong or the question was missing a candidate, and the engine cannot tell "
            "which because it never observed the node. Handed back for one probe instead "
            "of being defended or convicted",
        ],
        [
            "pruned by a dead question",
            str(sum(log.dead_question_prunes for log in b_logs)),
            "every candidate answer to a question was ruled out on its own evidence, so "
            "what assumed one of them can never be satisfied — the dual of deducing the "
            "last survivor, and it reaches what a refutation cascade spares",
        ],
        [
            "substitutes ruled out by a sub-par swap",
            str(sum(log.substitutes_ruled_out for log in b_logs)),
            "the diagnostic composition was right in every other slot and still fell "
            "short, so the swapped-in value is out — an elimination that costs no probe",
        ],
        [
            "alternatives reopened by an interaction effect",
            str(sum(log.interaction_reopens for log in b_logs)),
            "a conflict whose members all held individually — the answer must be "
            "among the alternatives they retired",
        ],
        [
            "alternatives reopened after a shortfall",
            str(sum(log.shortfall_reopens for log in b_logs)),
            "every question was answered and the answers assembled still missed the "
            "target — so one of those answers is wrong, and the search resumes "
            "instead of reporting itself finished",
        ],
        [
            "destructive revisions (NEEDS_REVISION / PRUNED)",
            str(sum(log.revision_transitions for log in b_logs)),
            "belief withdrawn because something built on it failed",
        ],
        [
            "exclusions retracted (EXHAUSTED→UNTESTED)",
            str(sum(log.reopened for log in b_logs)),
            "siblings handed back after their justification was withdrawn",
        ],
        [
            "conflicts recorded",
            f"{conflicts} (mean {members / conflicts:.1f} members)" if conflicts else "0",
            "a failure over ≥2 assumptions: they cannot all hold, but none is blamed yet",
        ],
        [
            "conflicts resolved to a culprit",
            f"{resolved} ({_pct(resolved / conflicts) if conflicts else '–'})",
            "later evidence exonerated every member but one",
        ],
        [
            "pruned re-executions",
            str(sum(log.pruned_reexecutions for log in b_logs)),
            "should be 0 — re-running a settled branch",
        ],
        [
            "results filed against a goal (refused)",
            str(sum(log.goal_contamination for log in b_logs)),
            "should be 0 — a probe whose answer had nowhere to go, so the hypothesis "
            "it tested is still untested",
        ],
        [
            "results rejected inside a batch",
            str(sum(log.records_rejected for log in b_logs)),
            "the engine refused one result while the rest of the batch still applied — "
            "isolation working, but each one is still a probe whose answer was lost",
        ],
        [
            "evidence-regime overrides",
            str(sum(log.regime_overrides for log in b_logs)),
            "agent asked for a regime the environment does not have",
        ],
    ]

    transitions: Counter[str] = Counter()
    for log in b_logs:
        transitions.update(log.transitions)

    instructions: Counter[str] = Counter()
    for log in b_logs:
        instructions.update(log.done_reasons)

    return [
        "## 5. Belief-state mechanics (arm B)",
        "",
        *_table(["metric", "value", "meaning"], summary_rows),
        "### Navigator instructions",
        "",
        "When the navigator has nothing to hand out it says *why*, and only two of the "
        "reasons are endings. `awaiting_evidence` means report what you are holding; "
        "`awaiting_substitution` names the one swap that would narrow an open conflict; "
        "`awaiting_composition` means every question is answered and the answers must be "
        "assembled; `blocked_frontier` means the graph is wired so nothing is reachable "
        "and `unreachable_goal` means the goal depends on nothing and so can never be "
        "satisfied — both are modelling errors, not results. A run with conflicts but no "
        "`awaiting_substitution` means the targeted recovery never engaged.",
        "",
        *_table(
            ["reason", "count"],
            [[k, str(v)] for k, v in instructions.most_common()] or [],
        ),
        "### Status transitions",
        "",
        *_table(
            ["transition", "count"],
            [[k, str(v)] for k, v in transitions.most_common()],
        ),
    ]


def _section_stratified(logs: list[RunLog]) -> list[str]:
    """Split the paired result by whether the episode hit an indeterminate failure.

    This is the single most explanatory view of the whole run and it is invisible
    from the headline. A combination resting on several assumptions that fails
    tells you only that they cannot all hold; how well the belief state handles
    that moment, rather than how well it eliminates candidates, is what decides
    the aggregate. Reporting only the pooled number averages a strategy that
    works with a recovery path that may not, and reads as "no effect".
    """
    b_done = {log.seed: log for log in logs if log.arm == "B" and log.complete}
    if not b_done:
        return ["## 6. Stratified by conflict (arm B)", "", "_(no completed arm-B episodes)_", ""]

    rows = []
    for label, want in (("no conflict", False), ("conflict fired", True)):
        seeds = [s for s, log in b_done.items() if log.has_conflict is want]
        if not seeds:
            rows.append([label, "0", "–", "–", "–", "–"])
            continue
        # Right-censored throughout, exactly as the headline and the gate score
        # them. Mixing raw steps here reported the same episodes as 17.7 mean
        # steps in this table and 100.0 in the headline — and reconstructing the
        # baseline's mean by adding a censored *difference* to an uncensored
        # base produced a negative number of probes.
        b_steps = [float(b_done[s].scored_steps) for s in seeds]
        paired = pair_arms([log for log in logs if log.seed in set(seeds)], "B", "F")
        f_mean = (
            statistics.mean(
                [
                    b_done[s].scored_steps + d
                    for s, d in zip(paired.seeds, paired.diffs, strict=True)
                ]
            )
            if paired.n
            else 0.0
        )
        reduction = (statistics.mean(paired.diffs) / f_mean) if paired.n and f_mean else 0.0
        rows.append(
            [
                label,
                str(len(seeds)),
                f"{statistics.mean(b_steps):.2f}",
                f"{f_mean:.2f}" if paired.n else "–",
                f"{100 * reduction:+.1f}%" if paired.n else "–",
                f"{paired.wins}/{paired.n}" if paired.n else "–",
            ]
        )

    out = [
        "## 6. Stratified by conflict (arm B)",
        "",
        "A conflict is an indeterminate failure: a combination resting on several "
        "assumptions failed, so they cannot all hold but none is individually refuted. "
        "Eliminating candidates and recovering from an indeterminate failure are two "
        "different competences, and pooling them hides which one is limiting.",
        "",
        *_table(
            ["stratum", "n", "B mean steps", "F mean steps", "reduction", "B wins"],
            rows,
        ),
    ]

    conflict_logs = [log for log in b_done.values() if log.has_conflict]
    if conflict_logs:
        resolved = sum(log.conflicts_resolved for log in conflict_logs)
        recorded = sum(log.conflicts_recorded for log in conflict_logs)
        interaction = sum(log.interaction_reopens for log in conflict_logs)
        shortfall = sum(log.shortfall_reopens for log in conflict_logs)
        swaps = [n for log in conflict_logs for n in log.diagnosis_swaps]
        swap_line = (
            f"{statistics.mean(swaps):.2f} probes to name the culprit "
            f"(min {min(swaps)}, max {max(swaps)})"
            if swaps
            else "no culprit was named, so the diagnosis cost nothing and bought nothing"
        )
        out += [
            f"Diagnosis cost: {swap_line}. This is what the member ordering buys and it is "
            "the number to watch — one swap per position, so a culprit ranked second costs "
            "exactly one probe more than a culprit ranked first. A run can resolve every "
            "conflict and still have got slower.",
            "",
            f"Recovery health: {resolved}/{recorded} conflicts narrowed to a culprit, "
            f"{interaction} alternative(s) reopened after a conflict was shown to be an "
            f"interaction effect, "
            f"{shortfall} alternative(s) reopened after every answer was in and the "
            f"assembly still fell short, "
            f"{statistics.mean([log.reopened for log in conflict_logs]):.1f} alternatives "
            f"reopened per conflict episode, "
            f"{statistics.mean([log.duplicates for log in conflict_logs]):.1f} duplicate "
            f"probes per conflict episode.",
            "",
            "A conflict has two honest endings, and only one of them names a culprit. If "
            "every member survives a re-test at the failing depth, no member is at fault: "
            "the failure is an interaction, and the answer lies among the alternatives "
            "those members retired. Both endings must occur — a run where neither does "
            "means the recovery never completed and the agent searched blind instead.",
            "",
            "A conflict left unresolved on a seed that *won* is worth a look before it is "
            "read as a failure. The swap that names the culprit is often the winning "
            "combination itself, so the episode used to end on the environment's verdict "
            "before the agent could file the record that resolves it — the engine was "
            "right and this counter could not see it. The wind-down turns added in 0.4.0 "
            "close that window; an unresolved conflict on a winning seed now means the "
            "agent did not report within them.",
            "",
        ]
    return out


def _section_memory(logs: list[RunLog]) -> list[str]:
    """How each arm carried belief across the enforced context resets."""
    grouped = _by_arm(logs)
    rows = []
    for arm in _arm_order(grouped):
        done = [log for log in grouped[arm] if log.complete]
        if not done:
            continue
        summaries = [n for log in done for n in log.summary_lengths]
        writes = sum(log.scratchpad_writes for log in done)
        rows.append(
            [
                arm,
                f"{statistics.mean([log.session_resets for log in done]):.1f}",
                f"{statistics.mean(summaries):.0f}" if summaries else "–",
                f"{writes} ({writes / len(done):.1f}/episode)",
                f"{statistics.mean([log.scratchpad_chars for log in done]):.0f}",
                f"{sum(1 for log in done if log.scratchpad_writes == 0)}/{len(done)}",
            ]
        )
    return [
        "## 7. Memory maintenance across context resets",
        "",
        "`never wrote notes` is the number of episodes in which the agent maintained no "
        "manual memory at all. For a baseline arm this decides what the comparison "
        "measures, so it is reported rather than assumed.",
        "",
        *_table(
            [
                "arm",
                "resets/episode",
                "mean summary chars",
                "note writes",
                "mean note chars",
                "never wrote notes",
            ],
            rows,
        ),
    ]


def _section_cost(logs: list[RunLog]) -> list[str]:
    """LLM turns, latency and tokens — the overhead step counts hide."""
    grouped = _by_arm(logs)
    rows = []
    for arm in _arm_order(grouped):
        done = [log for log in grouped[arm] if log.complete]
        calls = sum(log.llm_calls for log in done)
        if not calls:
            continue
        seconds = sum(log.llm_seconds for log in done)
        prompt = sum(log.prompt_tokens for log in done)
        completion = sum(log.completion_tokens for log in done)
        steps = sum(log.steps for log in done) or 1
        # Results reported per record call. 1.0 means every result cost its own
        # turn, which is the floor the batch shape exists to break; the ceiling
        # is the dispatch batch size.
        record_calls = sum(log.tool_histogram.get("record_evidence", 0) for log in done)
        results = sum(log.evidence_records for log in done)
        rows.append(
            [
                arm,
                f"{calls / len(done):.1f}",
                f"{calls / steps:.2f}",
                f"{results / record_calls:.2f}" if record_calls else "–",
                f"{seconds / calls:.1f}s",
                f"{seconds / len(done) / 60:.1f}m",
                f"{prompt / calls:.0f}" if prompt else "–",
                f"{completion / calls:.0f}" if completion else "–",
            ]
        )
    out = [
        "## 8. Cost per arm",
        "",
        "`turns/step` is LLM round-trips per dispatched experiment: the belief-state arm "
        "pays extra turns for its bookkeeping, and that overhead is invisible in a step "
        "count. Empty when the run used the mock backend.",
        "",
        "This is the number the moat has to be worth. The belief-state arm buys fewer "
        "experiments with more tokens, so it is net-positive exactly when an experiment "
        "costs materially more than the bookkeeping around it — which the eval's "
        "millisecond oracle is the pessimal case for. Watch it move when the "
        "record→dispatch fusion is on: a fused loop should approach the baseline's "
        "turns/step without giving up any of the step reduction.",
        "",
        "`results/record call` is what batch recording buys. At 1.00 every result cost "
        "its own turn, which is the two-turns-per-experiment floor the fusion alone "
        "cannot break; the ceiling is the dispatch batch size. If turns/step has not "
        "fallen, check this first — the agent may simply not be using the batch shape.",
        "",
        *_table(
            [
                "arm",
                "turns/episode",
                "turns/step",
                "results/record call",
                "mean latency",
                "wall/episode",
                "prompt tok/turn",
                "completion tok/turn",
            ],
            rows,
        ),
    ]

    tools: dict[str, Counter[str]] = defaultdict(Counter)
    errors: dict[str, Counter[str]] = defaultdict(Counter)
    for log in logs:
        tools[log.arm].update(log.tool_histogram)
        errors[log.arm].update(log.tool_errors)
    for arm in _arm_order(grouped):
        if not tools[arm]:
            continue
        out += [
            f"### Tool usage — arm {arm}",
            "",
            *_table(
                ["tool", "calls", "failed"],
                [[k, str(v), str(errors[arm].get(k, 0))] for k, v in tools[arm].most_common()],
            ),
        ]

    depths: dict[str, Counter[int]] = defaultdict(Counter)
    for log in logs:
        depths[log.arm].update(log.probe_depths)
    depth_rows = [
        [arm, ", ".join(f"d{d}×{n}" for d, n in sorted(depths[arm].items()))]
        for arm in _arm_order(grouped)
        if depths[arm]
    ]
    out += [
        "### Probe depth",
        "",
        "Depth is the rigour a config was tested at. An arm that never probes deeper "
        "than 1 cannot distinguish a shallow confirmation from a real one, which is "
        "exactly what the decoy exploits.",
        "",
        *_table(["arm", "distribution"], depth_rows),
    ]
    return out


def _section_ablation(ablation: list[dict[str, Any]]) -> list[str]:
    """Criterion 2 — is Thompson Sampling actually better than the alternatives?"""
    if not ablation:
        return [
            "## 9. Navigator ablation (criterion 2)",
            "",
            "_(no ablation logs in this run)_",
            "",
        ]
    by_strategy: dict[str, list[float]] = defaultdict(list)
    for entry in ablation:
        by_strategy[str(entry.get("strategy", "?"))].append(
            float(entry.get("cumulative_regret", 0.0))
        )
    rows = []
    for strategy in sorted(by_strategy):
        stats = _describe(by_strategy[strategy])
        rows.append(
            [
                strategy,
                str(int(stats["n"])),
                f"{stats['mean']:.1f}",
                f"{stats['median']:.1f}",
                f"{stats['max']:.1f}",
            ]
        )
    return [
        "## 9. Navigator ablation (criterion 2)",
        "",
        "Cumulative regret against the same seeded bandit, lower is better. Max regret is "
        "the statistic where Thompson Sampling should beat greedy: greedy's failure mode is "
        "occasional total commitment to a bad arm, not a worse average.",
        "",
        *_table(["strategy", "n", "mean regret", "median", "max"], rows),
    ]


def _section_hci(logs: list[RunLog]) -> list[str]:
    """The Hypotree Capability Index (HCI) — a single 0.0-1.0 score per arm.

    HCI measures how efficiently, precisely, and error-free an agent navigates
    the logical space. It is a product of independent quality factors, so any
    hard failure (0 goals met) zeroes the score, while soft failures apply a
    graduated penalty.

    Formula: HCI = E_base × P_mem × P_tools × P_health

    1. **E_base** — completion rate × efficiency against the oracle. Efficiency
       is measured over *every* completed episode using the same right-censored
       step count the gate scores, not only over the episodes that succeeded:
       averaging the ratio over winners alone measures an arm on the subset of
       seeds it happened to find easy, and then multiplies by a completion rate
       that penalises the rest a second time. Clamped at 1.0, because an arm that
       beats the reference strategy on one seed has not exceeded capability — the
       reference is a yardstick, not a bound, and an unclamped ratio let the
       "0.0-1.0 index" report 1.4.
    2. **P_mem** — 1 − duplicate probes / probes. The oracle is deterministic, so
       a duplicate is a direct measurement of memory failure.
    3. **P_tools** — errors per *tool call*, not per episode. An agent making two
       hundred calls with three errors is more proficient than one making ten
       calls with three, and per-episode normalisation cannot tell them apart.
    4. **P_health** — engine-rule violations per episode. Also normalised, and
       for the same reason in reverse: an absolute exponent means a long run is
       scored more harshly than a short one for the same *rate* of violation, and
       one miscounted metric annihilates the index (a reporting bug that counted
       twenty phantom violations drove this factor to 0.01).

    P_health is only defined for the belief-state arm — the baselines have no
    engine rules to break — so the arms are comparable on the first three
    factors and B carries one extra, stricter, factor. That asymmetry is
    deliberate and stated rather than hidden: it is the price of having rules.
    """
    grouped = _by_arm(logs)
    rows = []
    for arm in _arm_order(grouped):
        done = [log for log in grouped[arm] if log.complete]
        if not done:
            continue

        n_eps = len(done)
        goals_met_count = sum(1 for log in done if log.goals_met)

        # 1. E_base: completion rate × oracle efficiency, over every episode.
        completion_rate = goals_met_count / n_eps
        ratios = [
            min(1.0, reference_strategy_probes(log.seed) / log.scored_steps)
            for log in done
            if log.scored_steps > 0
        ]
        efficiency_ratio = statistics.mean(ratios) if ratios else 0.0
        e_base = completion_rate * efficiency_ratio

        # 2. P_mem: memory reliability.
        total_probes = sum(log.experiments for log in done)
        duplicates = sum(log.duplicates for log in done)
        p_mem = max(0.0, 1.0 - duplicates / total_probes) if total_probes else 1.0

        # 3. P_tools: tool proficiency, per call. Every error *is* a call, so the
        # histogram can never be smaller than the error count — falling back to
        # it keeps a log missing its census from scoring a flawless run.
        total_calls = sum(sum(log.tool_histogram.values()) for log in done)
        total_errors = sum(sum(log.tool_errors.values()) for log in done)
        denominator = max(total_calls, total_errors)
        p_tools = 1.0 - (total_errors / denominator if denominator else 0.0)

        # 4. P_health: engine-rule adherence, per episode.
        health_violations = 0
        if arm == ARM_B:
            health_violations = sum(
                log.redispatched
                + log.settled_reselects
                + log.pruned_reexecutions
                + log.goal_contamination
                + log.same_question_dispatches
                for log in done
            )
        p_health = 0.8 ** (health_violations / n_eps)

        hci = e_base * p_mem * p_tools * p_health

        rows.append(
            [
                arm,
                f"{hci:.4f}",
                f"{e_base:.4f}",
                f"{p_mem:.4f}",
                f"{p_tools:.4f}",
                f"{p_health:.4f}" if arm == ARM_B else "1.0 (n/a)",
            ]
        )

    if not rows:
        return ["## 11. Hypotree Capability Index (HCI)", "", "_(no completed episodes)_", ""]

    return [
        "## 11. Hypotree Capability Index (HCI)",
        "",
        "A single 0.0-1.0 score per arm measuring efficiency, memory reliability, tool "
        "proficiency, and engine-rule adherence. **HCI = E_base × P_mem × P_tools × P_health**. "
        "Any hard failure (0 goals met) zeroes the score; soft failures apply graduated "
        "penalties. Efficiency uses the gate's right-censored step count over every episode, "
        "so an arm is not measured only on the seeds it found easy. `P_health` applies to the "
        "belief-state arm alone — the baselines have no engine rules to break — so read the "
        "first three factors when comparing arms and the fourth as a discipline check on B.",
        "",
        *_table(
            [
                "arm",
                "HCI",
                "E_base (efficiency)",
                "P_mem (memory)",
                "P_tools (tools)",
                "P_health (rules)",
            ],
            rows,
        ),
    ]


def _section_warnings(logs: list[RunLog], runs_dir: Path, run_id: str) -> list[str]:
    """Everything that makes a number above less trustworthy than it looks."""
    warnings: list[str] = []

    # A log whose own run_start disagrees with the directory it sits in means two
    # runs' results have been mixed, which invalidates every aggregate above.
    foreign = [log for log in logs if log.declared_run_id and log.declared_run_id != run_id.strip()]
    if foreign:
        warnings.append(
            f"{len(foreign)} log(s) declare a different run id "
            f"({sorted({str(log.declared_run_id) for log in foreign})}) — this directory "
            f"contains results from more than one run and the aggregates are invalid"
        )

    models = {log.llm_model for log in logs if log.llm_model}
    if len(models) > 1:
        warnings.append(
            f"episodes ran against different models ({sorted(models)}) — not comparable"
        )

    # Two different reasons an episode does not count, reported apart because
    # they call for different responses: a missing terminal record means the
    # harness died and the episode must be re-run, while an infrastructure
    # failure is already accounted for and just needs the server looked at.
    infra = [log for log in logs if log.infra_failed]
    if infra:
        warnings.append(
            f"{len(infra)} episode(s) ended because the inference server stopped answering "
            f"and were excluded from paired comparisons (an infrastructure fault is not a "
            f"result): " + ", ".join(f"seed {x.seed} arm {x.arm}" for x in infra)
        )

    incomplete = [log for log in logs if not log.complete and not log.infra_failed]
    if incomplete:
        warnings.append(
            f"{len(incomplete)} episode(s) have no `run_end` and were excluded from paired "
            f"comparisons: " + ", ".join(f"seed {x.seed} arm {x.arm}" for x in incomplete)
        )

    grouped = _by_arm(logs)
    seeds_per_arm = {arm: {log.seed for log in runs} for arm, runs in grouped.items()}
    all_seeds: set[int] = set().union(*seeds_per_arm.values()) if seeds_per_arm else set()
    for arm, seeds in sorted(seeds_per_arm.items()):
        missing = sorted(all_seeds - seeds)
        if missing:
            warnings.append(f"arm {arm} is missing seeds {missing} — comparisons are unbalanced")

    budget_bound = [log for log in logs if log.complete and not log.goals_met]
    if budget_bound:
        detail = ", ".join(
            f"seed {x.seed} arm {x.arm} ({x.end_reason}, {x.steps} probes)" for x in budget_bound
        )
        warnings.append(
            f"{len(budget_bound)} episode(s) ended without meeting the goal ({detail}); "
            f"each is right-censored to the tool budget here and in the gate, so the true "
            f"gap is unknown rather than understated — investigate the end reason before "
            f"reading anything else, since an episode that stopped at zero probes is a "
            f"defect, not a result"
        )

    # An arm-B episode that runs out of *frontier* long before it runs out of
    # *budget*, with the goal unmet, is the belief state declaring the search
    # finished while the search space is plainly not exhausted. It is the single
    # most damaging failure mode this harness has ever produced — three episodes
    # once ended at 14, 16 and 23 probes against a budget of 100, each scored as
    # a maximum-cost loss — and it is invisible in the headline, which sees only
    # the censored budget. Named separately from the generic unmet-goal warning
    # because the fix is in the engine, not the agent.
    premature = [
        log
        for log in logs
        if log.arm == "B"
        and log.complete
        and not log.goals_met
        and log.end_reason == "frontier_exhausted"
        and log.tool_budget
        and log.steps < 0.5 * log.tool_budget
    ]
    if premature:
        detail = ", ".join(f"seed {x.seed} ({x.steps}/{x.tool_budget} probes)" for x in premature)
        warnings.append(
            f"{len(premature)} arm-B episode(s) ran out of frontier at under half the probe "
            f"budget with the goal unmet ({detail}) — the belief state declared the search "
            f"over while the space was plainly not exhausted, and each is then scored at the "
            f"full budget. Read this before any comparison: it is an engine defect, not a "
            f"measurement of search quality"
        )

    reselects = sum(log.settled_reselects for log in logs)
    if reselects:
        warnings.append(
            f"{reselects} navigator target(s) were already settled — the sampler handed back "
            f"a node that needed no further work"
        )

    redispatched = sum(log.redispatched for log in logs)
    if redispatched:
        warnings.append(
            f"{redispatched} node(s) were handed out while already under a live lease — the "
            f"claim failed to reserve them, so the agent was sent to re-probe work it was "
            f"still holding results for"
        )

    b_logs = [log for log in logs if log.arm == "B" and log.complete]
    dispatched = sum(log.targets_selected for log in b_logs)
    # Only records that carried a claim answer a dispatch. Compositions the
    # agent assembles itself are recorded without one, and counting them here
    # made the shortfall look smaller than it was — or negative.
    answered = sum(log.claimed_records for log in b_logs)
    if dispatched and answered < 0.8 * dispatched:
        warnings.append(
            f"only {answered}/{dispatched} arm-B dispatches were ever reported — the belief "
            f"state is missing most of what the agent actually measured, so its step count "
            f"reflects lost work rather than search quality"
        )

    pruned = sum(log.pruned_reexecutions for log in logs)
    if pruned:
        warnings.append(f"{pruned} pruned branch(es) were re-executed — settled work was redone")

    same_question = sum(log.same_question_dispatches for log in b_logs)
    if same_question:
        warnings.append(
            f"{same_question} dispatch(es) offered a second answer to a question the same "
            f"batch had already asked — the first result cannot retire the second, so the "
            f"exclusion inference was declared and then paid for twice"
        )

    believed = sum(
        1 for log in logs if log.end_reason == "believes_goals_met" and not log.goals_met
    )
    if believed:
        warnings.append(
            f"{believed} arm-B episode(s) ended believing the goal was met without any probe "
            f"clearing the target at depth — the belief state was satisfied by evidence "
            f"recorded against the goal node rather than by a solution"
        )

    goal_hits = sum(log.goal_contamination for log in logs)
    if goal_hits:
        warnings.append(
            f"{goal_hits} result(s) were filed against a goal and refused — each is a probe "
            f"the agent paid for and could not use, and the hypothesis it actually tested "
            f"was left untested, so no conflict could be raised over its assumptions"
        )

    tool_failures = sum(sum(log.tool_errors.values()) for log in logs)
    if tool_failures:
        warnings.append(
            f"{tool_failures} tool call(s) failed — see the per-arm tool table for which"
        )

    if not any(log.llm_calls for log in logs):
        warnings.append(
            "no `llm_call` events: this run used the mock backend, so the cost section is empty "
            "and step counts reflect a scripted agent, not a model"
        )

    if not list(runs_dir.glob("ablation-*.jsonl")):
        warnings.append("no ablation logs — criterion 2 cannot be scored from this run")

    return [
        "## 12. Data-quality warnings",
        "",
        *([f"- {w}" for w in warnings] if warnings else ["_(none)_"]),
        "",
    ]


def _section_action_taxonomy(logs: list[RunLog]) -> list[str]:
    """What the agent did, split by whether anything was under revision.

    The evidence behind criterion 4. If the two columns look the same, the
    richer statuses changed no behaviour and the pre-registered response is to
    collapse them; if they differ, they are carrying information. Rendered here
    because a p-value in the gate JSON says which of the two happened but never
    says *how*, and the shape of the disagreement is the actionable part.
    """
    b_logs = [log for log in logs if log.arm == "B"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for log in b_logs:
        for (action, context), n in log.agent_actions.items():
            counts[action][context] += n
    if not counts:
        return ["## 10. Action taxonomy (arm B, criterion 4)", "", "_(not instrumented)_", ""]

    totals = Counter()
    for contexts in counts.values():
        totals.update(contexts)

    rows = [
        [
            action,
            str(counts[action]["open"]),
            _pct(counts[action]["open"] / totals["open"]) if totals["open"] else "–",
            str(counts[action]["revision"]),
            _pct(counts[action]["revision"] / totals["revision"]) if totals["revision"] else "–",
        ]
        for action in sorted(counts)
    ]
    rows.append(["**total**", str(totals["open"]), "", str(totals["revision"]), ""])

    return [
        "## 10. Action taxonomy (arm B, criterion 4)",
        "",
        "Every agent turn, split by the belief state it was taken in. `revision` means at "
        "least one hypothesis was BLOCKED or NEEDS_REVISION at that moment; `open` means "
        "the state was expressible with UNTESTED alone. Two columns with the same shape "
        "mean the richer statuses changed no behaviour and should be collapsed.",
        "",
        *_table(
            ["action", "open", "share", "revision", "share"],
            rows,
        ),
    ]


def render_report(run_id: str, runs_dir: Path) -> str:
    """Build the full markdown report for one run."""
    logs, ablation = load_run(runs_dir)
    if not logs:
        return (
            f"# Evaluation report — run `{run_id}`\n\n"
            f"No `seed-*-arm-*.jsonl` logs found under `{runs_dir}`.\n"
        )

    lines: list[str] = []
    lines += _section_header(run_id, runs_dir, logs)
    lines += _section_headline(logs)
    lines += _section_paired(logs)
    lines += _section_per_seed(logs)
    lines += _section_probe_economy(logs)
    lines += _section_belief_state(logs)
    lines += _section_stratified(logs)
    lines += _section_memory(logs)
    lines += _section_cost(logs)
    lines += _section_ablation(ablation)
    lines += _section_action_taxonomy(logs)
    lines += _section_hci(logs)
    lines += _section_warnings(logs, runs_dir, run_id)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eval.seed_reader",
        description="Render one evaluation run's JSONL logs as a markdown report.",
    )
    parser.add_argument(
        "--run-id", required=True, help="identifier of the run under eval/runs/<run-id>/"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="path to the eval/ directory (default: this script's own directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the report here instead of stdout",
    )
    args = parser.parse_args(argv)

    runs_dir = run_dir(args.eval_dir, args.run_id)
    report = render_report(args.run_id, runs_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
