"""The falsifier for `P8d-COST`, scored: does knowing what a probe costs pay?

`P8d-COST` shipped with a falsifier nobody could run. Every gate this project has
ever scored counts *probes*, which is defensible only because the landscape
oracle answers in uniform milliseconds — so a probe **is** the unit of cost by
construction, `theta/cost` and `theta` induce exactly the same order, and the
claimed 20% cost reduction was not merely unmet but unobservable. A falsifier
that cannot fire is not a falsifier.

This module supplies the missing instrument. It plays every seed twice against a
landscape whose probes carry unequal costs — once with the navigator allowed to
see them, once not — and scores the pre-registered thresholds.

**Why the saving exists at all, and why it is not circular.** Probe *count* is
invariant under reordering: the winner's position within a question is uniform,
so every order settles it in the same expected number of probes. That is not a
weakness of the navigator, it is a property of the landscape, and it is why three
consecutive audits found exclusion yield pinned at chance. Probe *cost* is not
invariant, because the last survivor of a closed question is **deduced rather
than probed** — whichever answer is left unprobed is never paid for. So ordering
cheap-first pushes the expensive answer into that free slot. The cost falls; the
probe count does not move. Both halves of the falsifier come from one mechanism,
which is what makes it a single claim rather than two coincidences.

The cost tariff is assigned to each answer by a seeded shuffle that is
independent of which answer is correct — verified at 19% against a 20% chance
baseline. Without that independence the arm would look good for finding the
answer sooner rather than for deferring the expensive probe, and those are
different claims.

**Self-play, not an LLM run, and deliberately.** An agent picks its own probe
order, so a run with a model in the loop would measure the model's thrift. This
drives the engine with a perfectly-disciplined caller that does exactly what the
navigator says, so the difference between the arms is the navigator and nothing
else.

Run it:

    uv run python -m eval.cost_gate
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass

from eval.runner.config import TASK_SEEDS
from eval.runner.engine_selfplay import SelfPlayResult, solve_seed

# Pre-registered, and written down before the first run for the reason this
# project has had to relearn twice: deciding what counts as a pass after seeing
# the number is not a measurement.
#
# Cost must fall by at least this much. 20% is the figure `P8d-COST` committed to
# when the phase was written, carried over unchanged rather than tuned to
# whatever the implementation turned out to deliver.
MIN_COST_REDUCTION = 0.20

# ...and probes must not rise by more than this. The guard against buying cost
# with volume: an arm that halves the bill by spending many more cheap probes has
# moved the number without helping anyone.
MAX_PROBE_INCREASE = 0.10

# Both arms must still solve everything. A cheaper search that stops solving is
# not a cheaper search.
REQUIRED_SOLVE_RATE = 1.0


@dataclass(frozen=True)
class ArmSummary:
    """Aggregate for one arm over the scored seeds."""

    name: str
    solved: int
    episodes: int
    mean_probes: float
    mean_cost: float

    @property
    def solve_rate(self) -> float:
        return self.solved / self.episodes if self.episodes else 0.0


def _summarise(name: str, results: list[SelfPlayResult]) -> ArmSummary:
    return ArmSummary(
        name=name,
        solved=sum(1 for r in results if r.solved),
        episodes=len(results),
        mean_probes=statistics.mean([r.probes for r in results]) if results else 0.0,
        mean_cost=statistics.mean([r.cost for r in results]) if results else 0.0,
    )


def score(blind: list[SelfPlayResult], aware: list[SelfPlayResult]) -> dict[str, object]:
    """Apply the pre-registered thresholds to a paired run.

    Paired by seed throughout: the seeds differ enormously in how expensive they
    happen to be, so comparing arm means across unmatched sets would report the
    luck of the draw. Every figure below is a within-seed comparison.
    """
    by_seed = {r.seed: r for r in blind}
    pairs = [(by_seed[a.seed], a) for a in aware if a.seed in by_seed]

    cost_deltas = [(b.cost - a.cost) / b.cost for b, a in pairs if b.cost > 0]
    probe_deltas = [(a.probes - b.probes) / b.probes for b, a in pairs if b.probes > 0]
    wins = sum(1 for b, a in pairs if a.cost < b.cost)
    losses = sum(1 for b, a in pairs if a.cost > b.cost)

    blind_arm = _summarise("cost-blind", blind)
    aware_arm = _summarise("cost-aware", aware)

    cost_reduction = (
        (blind_arm.mean_cost - aware_arm.mean_cost) / blind_arm.mean_cost
        if blind_arm.mean_cost
        else 0.0
    )
    probe_increase = (
        (aware_arm.mean_probes - blind_arm.mean_probes) / blind_arm.mean_probes
        if blind_arm.mean_probes
        else 0.0
    )

    # Headroom: how much of the achievable saving was captured. The floor is a
    # sweep that probes every question cheapest-answer-first, which is the best a
    # navigator can do knowing costs and nothing else about correctness. Scoring
    # against this rather than against zero is what separates "the mechanism
    # works" from "the mechanism is optimal" — only the first is claimed.
    achievable = [b.cost - b.optimal_cost for b, _ in pairs]
    captured = [b.cost - a.cost for b, a in pairs]
    capture = sum(captured) / sum(achievable) if sum(achievable) > 0 else 0.0

    criteria = {
        "cost_reduction": {
            "value": cost_reduction,
            "threshold": MIN_COST_REDUCTION,
            "passed": cost_reduction >= MIN_COST_REDUCTION,
            "claim": (
                "cost-aware selection reduces total cost to goal by at least "
                f"{MIN_COST_REDUCTION:.0%} against the same engine ranking on "
                "posterior alone"
            ),
        },
        "probe_discipline": {
            "value": probe_increase,
            "threshold": MAX_PROBE_INCREASE,
            "passed": probe_increase <= MAX_PROBE_INCREASE,
            "claim": (
                "...without buying that saving with more probes: probe count "
                f"rises by no more than {MAX_PROBE_INCREASE:.0%}"
            ),
        },
        "still_solves": {
            "value": aware_arm.solve_rate,
            "threshold": REQUIRED_SOLVE_RATE,
            "passed": aware_arm.solve_rate >= REQUIRED_SOLVE_RATE,
            "claim": "the cost-aware arm still reaches the goal on every seed",
        },
    }
    passed = all(c["passed"] for c in criteria.values())

    return {
        "decision": "CONFIRMED" if passed else "FALSIFIED",
        "episodes": len(pairs),
        "criteria": criteria,
        "arms": {"blind": asdict(blind_arm), "aware": asdict(aware_arm)},
        "paired": {
            "median_cost_reduction": statistics.median(cost_deltas) if cost_deltas else 0.0,
            "median_probe_change": statistics.median(probe_deltas) if probe_deltas else 0.0,
            "seeds_cheaper": wins,
            "seeds_dearer": losses,
            "seeds_unchanged": len(pairs) - wins - losses,
            "headroom_captured": capture,
            "mean_optimal_cost": (
                statistics.mean([b.optimal_cost for b, _ in pairs]) if pairs else 0.0
            ),
        },
    }


def _render(report: dict[str, object]) -> str:
    arms = report["arms"]  # type: ignore[index]
    paired = report["paired"]  # type: ignore[index]
    blind, aware = arms["blind"], arms["aware"]
    lines = [
        f"  episodes paired: {report['episodes']}",
        f"  cost-blind: {blind['mean_cost']:8.1f}s mean, "
        f"{blind['mean_probes']:5.2f} probes, solved {blind['solved']}/{blind['episodes']}",
        f"  cost-aware: {aware['mean_cost']:8.1f}s mean, "
        f"{aware['mean_probes']:5.2f} probes, solved {aware['solved']}/{aware['episodes']}",
        f"  achievable floor: {paired['mean_optimal_cost']:8.1f}s mean "
        f"(cheapest-answer-first sweep)",
        "",
        f"  cheaper on {paired['seeds_cheaper']} seeds, dearer on "
        f"{paired['seeds_dearer']}, unchanged on {paired['seeds_unchanged']}",
        f"  median per-seed cost reduction: {paired['median_cost_reduction']:.1%}",
        f"  headroom captured: {paired['headroom_captured']:.1%}",
        "",
    ]
    for name, c in report["criteria"].items():  # type: ignore[union-attr]
        flag = "PASS" if c["passed"] else "FAIL"
        lines.append(f"  [{flag}] {name}: {c['value']:.1%} (threshold {c['threshold']:.0%})")
        lines.append(f"         {c['claim']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.cost_gate",
        description="Score the P8d-COST falsifier on the cost-weighted landscape.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(TASK_SEEDS),
        help="seeds to play (default: the pre-registered set)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    blind = [solve_seed(s, cost_aware=False) for s in args.seeds]
    aware = [solve_seed(s, cost_aware=True) for s in args.seeds]
    report = score(blind, aware)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("P8d-COST falsifier — cost-weighted landscape, engine self-play")
        print(_render(report))
        print()
        print(f"  decision: {report['decision']}")

    if report["decision"] != "CONFIRMED":
        print(
            "P8d-COST FALSIFIED: cost-aware selection did not pay on its own landscape.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
