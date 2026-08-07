"""Engine self-play: can the belief state solve the landscape *without* an LLM?

A pre-flight, and the cheapest useful test this harness has. It drives the engine
with a scripted, perfectly-disciplined caller — one that always probes exactly
what the navigator dispatches, always builds exactly the composition the
navigator names, and never forgets anything — and asks whether the goal is
reached at all.

That isolates the one question an LLM run cannot answer: **is the engine's own
search complete?** A model can paper over an engine that gives up early by
guessing combinations by hand, and an engine that gives up early looks in the
logs exactly like a model that ran out of ideas. Separating them after the fact
costs a day of reading JSONL; separating them here costs two seconds.

It exists because it was missing. A regression in the substitution verdict made
the engine declare three real episodes finished at 14, 16 and 23 probes out of a
budget of 100, each with the goal unmet and each scored as a maximum-cost loss —
after several GPU-hours of a real evaluation run. Every unit test passed
throughout: they each checked one mechanism in isolation, and the failure was in
how two of them composed. This check is the one that fails.

Deliberately *not* a substitute for the gate. It measures the engine's search
with the agent held perfect, which is the upper bound on what any model could do
with it — a floor on correctness, not an estimate of performance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eval.environment.landscape_scoring import (
    axis_value_costs,
    optimal_strategy_cost,
    probe_cost,
    reference_strategy_cost,
    score_config,
)
from eval.runner.config import TASK_SEEDS
from hypotree.engine import HypoTreeEngine
from hypotree.models.evidence import LogicalEvidence

LANDSCAPE_DIR = Path("eval/environment/landscapes")

# Turns before the driver gives up. Far above any legitimate episode; it exists
# only so a cycle in the engine fails the check instead of hanging it.
MAX_TURNS = 400

# The perfect caller must not need materially more probes than the perfect-recall
# reference strategy. Generous, because this is a regression tripwire rather than
# a performance target: it should fire when the search breaks, not when it gets
# slightly less lucky.
MAX_PROBE_RATIO = 1.5

_PARENTS_RE = re.compile(r"parent_ids=\[([^\]]*)\]")


@dataclass(frozen=True)
class SelfPlayResult:
    """One scripted episode."""

    seed: int
    probes: int
    reference: int
    solved: bool
    end_reason: str
    cost: float = 0.0
    reference_cost: float = 0.0
    optimal_cost: float = 0.0

    @property
    def ratio(self) -> float:
        return self.probes / self.reference if self.reference else 0.0

    @property
    def cost_ratio(self) -> float:
        """Total cost as a fraction of the cost-blind reference sweep."""
        return self.cost / self.reference_cost if self.reference_cost else 0.0


def _load(seed: int) -> dict:
    return json.loads((LANDSCAPE_DIR / f"landscape_seed_{seed}.json").read_text(encoding="utf-8"))


def solve_seed(
    seed: int,
    rng_seed: int = 7,
    omit_winner_on: str | None = None,
    cost_aware: bool = False,
) -> SelfPlayResult:
    """Drive one episode with a caller that does exactly what it is told.

    The caller has no strategy of its own on purpose. Every decision — what to
    probe, at what depth, what to compose — comes from the navigator, so the
    result measures the engine and nothing else. That is exactly what makes it
    the right instrument for the cost question: an LLM picks its own order, so a
    run with one in the loop would measure the agent's thrift rather than the
    navigator's.

    ``omit_winner_on`` leaves the winning value off one axis's declared
    candidates, which is the only way this landscape can produce a question that
    genuinely has no answer: every axis carries its winner by construction, so a
    fully-declared question always confirms. Under-declaring is not a contrived
    case — it is an agent thinking of three values where there were five — and it
    is the situation the dead-question rule exists for.

    ``cost_aware`` ranks candidates by value per unit cost. Both arms pay the
    same tariff and record the same durations; they differ only in whether the
    navigator is allowed to look at them.
    """
    land = _load(seed)
    with tempfile.TemporaryDirectory() as tmp:
        engine = HypoTreeEngine(
            Path(tmp) / "state.db",
            rng_seed=rng_seed,
            project_path=Path(tmp),
            cost_aware=cost_aware,
        )
        try:
            return _drive(engine, land, seed, omit_winner_on)
        finally:
            engine.close()


def _drive(
    engine: HypoTreeEngine, land: dict, seed: int, omit_winner_on: str | None = None
) -> SelfPlayResult:
    withheld = land["winning_values"][omit_winner_on] if omit_winner_on else None
    tariff = axis_value_costs(seed)
    engine.create_hypotheses(
        [
            {
                "statement": f"{axis}={value}",
                "node_id": f"{axis}_{value}",
                "exclusion_group": axis,
                # What the caller expects this probe to take. Declared because a
                # question is settled once: none of these has been timed at the
                # moment the navigator has to choose between them, so without an
                # estimate every one of them looks identical and cost cannot
                # order anything. The estimate is superseded by the first real
                # timing, and it moves only what is tried next.
                "estimated_cost": tariff[axis][value],
            }
            for axis in land["axes"]
            for value in land["axis_values"][axis]
            if not (axis == omit_winner_on and value == withheld)
        ]
    )
    reference = int(land["reference_strategy_probes"])
    target = float(land["target_metric"])
    probes = 0
    composed = 0
    spent = 0.0

    def finish(solved: bool, reason: str) -> SelfPlayResult:
        return SelfPlayResult(
            seed,
            probes,
            reference,
            solved,
            reason,
            cost=spent,
            reference_cost=reference_strategy_cost(seed),
            optimal_cost=optimal_strategy_cost(seed),
        )

    for _ in range(MAX_TURNS):
        response = engine.get_next_targets(count=1)[0]

        if response.status == "SELECTED":
            depth = max(1, response.min_depth or 1)
            probes += 1
            duration = probe_cost(response.statement, seed)
            spent += duration
            engine.record_evidence(
                response.node_id,  # type: ignore[arg-type]
                LogicalEvidence(
                    success=score_config(response.statement, seed, depth),
                    depth=depth,
                    duration_s=duration,
                ),
                claim_id=response.claim_id,
            )
            continue

        if response.reason in ("awaiting_composition", "awaiting_substitution"):
            match = _PARENTS_RE.search(response.rationale)
            if match is None:
                return finish(False, "advice_without_parents")
            parents = [p.strip().strip("'") for p in match.group(1).split(",")]
            composed += 1
            node_id = f"composition_{composed}"
            statement = ";".join(
                engine._store.get_node(p).statement  # type: ignore[union-attr]  # noqa: SLF001
                for p in parents
            )
            engine.create_hypotheses(
                [
                    {
                        "statement": statement,
                        "node_id": node_id,
                        "parent_ids": parents,
                        "edge_type": "DEPENDENCY",
                    }
                ]
            )
            depth = max(2, response.min_depth or 2)
            probes += 1
            duration = probe_cost(statement, seed)
            spent += duration
            success = score_config(statement, seed, depth)
            engine.record_evidence(
                node_id, LogicalEvidence(success=success, depth=depth, duration_s=duration)
            )
            if success >= target:
                return finish(True, "solved")
            continue

        # Any other DONE reason ends the episode. `empty_frontier` is the one
        # that matters: it means the engine believes the search is over, and
        # arriving here without having solved anything is the defect.
        return finish(False, response.reason)

    return finish(False, "turn_cap")


def run_selfplay(
    seeds: list[int] | tuple[int, ...] = TASK_SEEDS, cost_aware: bool = False
) -> list[SelfPlayResult]:
    """Play every seed and return the results in seed order."""
    return [solve_seed(seed, cost_aware=cost_aware) for seed in seeds]


def check(results: list[SelfPlayResult]) -> list[str]:
    """Failures a run must not start with. Empty means the engine is sound."""
    problems = [
        f"seed {r.seed}: {r.end_reason} after {r.probes} probes "
        f"(reference {r.reference}) — the engine gave up with the goal unmet"
        for r in results
        if not r.solved
    ]
    solved = [r for r in results if r.solved]
    if solved:
        mean_ratio = sum(r.ratio for r in solved) / len(solved)
        if mean_ratio > MAX_PROBE_RATIO:
            problems.append(
                f"a perfectly-disciplined caller needs {mean_ratio:.2f}x the reference "
                f"strategy's probes (limit {MAX_PROBE_RATIO}x) — the search has regressed"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.runner.engine_selfplay",
        description="Pre-flight: can the engine solve every landscape without an LLM?",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(TASK_SEEDS),
        help="seeds to play (default: the pre-registered set)",
    )
    args = parser.parse_args(argv)

    results = run_selfplay(args.seeds)
    for r in results:
        flag = "ok" if r.solved else "FAIL"
        print(
            f"  seed {r.seed}: {r.probes} probes vs reference {r.reference} "
            f"({r.ratio:.2f}x) -> {flag} [{r.end_reason}]"
        )
    solved = [r for r in results if r.solved]
    print(
        f"  solved {len(solved)}/{len(results)}"
        + (
            f", mean {sum(r.probes for r in solved) / len(solved):.1f} probes vs "
            f"{sum(r.reference for r in solved) / len(solved):.1f} reference"
            if solved
            else ""
        )
    )

    problems = check(results)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    if problems:
        print("Engine self-play FAILED — do not start a run against this build.", file=sys.stderr)
        return 1
    print("Engine self-play OK: the belief state reaches the goal on every seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
