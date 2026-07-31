"""TS-quality ablation: Thompson Sampling vs random vs greedy selection.

Criterion 2 asks a single question: is the navigator's Thompson-Sampling
selection genuinely *better* than naive alternatives? Answering it honestly
requires a **stochastic** reward environment. Under a deterministic regime one
probe reveals the full truth, so pure exploitation (greedy) is optimal and
exploration is wasted effort — TS can never win. The explore/exploit trade-off
that TS exists to solve only pays off when each probe is *noisy* and belief must
be accumulated over many samples.

So this ablation is a seeded **multi-armed bandit**, not a walk over the
LLM-facing landscape topology. A verified root gateway exposes a fixed set of
arms whose hidden Bernoulli success rates form a deliberate *big-gap* structure:

- **winner** — one moderately-successful arm. It is the unique optimum, but its
  rate is low enough that a single unlucky early sample looks indistinguishable
  from a failure.
- **decoys** — a cluster of many identically low-rate arms, far below the winner.
  Committing to any of them is a large, permanent mistake.

Each pull draws a fresh Bernoulli outcome around the selected arm's hidden rate
and records it as logical evidence; the engine folds it into the Beta posterior.
The metric is **cumulative regret over a fixed horizon**: the sum, across every
pull, of ``winner_rate - selected_rate``. Lower is better. No arm ever verifies
(the convergence gate is tuned so belief never resolves within the horizon), so
every strategy spends the full horizon on the bandit — there is no early stop.

**What this measures, honestly.** On the *typical* seed a pure-exploitation
greedy is hard to beat: it locks the winner after one lucky pull and pays almost
no exploration tax, so it wins the median regret. Thompson Sampling's genuine
advantage is at the *tail*: greedy has no recovery path, so on the seeds where
the winner's first sample fails it locks a decoy and bleeds linear regret for the
rest of the horizon (a catastrophic lock-in). TS keeps a live posterior over
every arm and always re-discovers the winner, so its worst-case regret stays
bounded. The gate therefore checks two things (see analyse_gate.py): ``TS
decisively beats RANDOM`` (typical-case) and ``TS's worst-case regret is
materially below greedy's`` (no catastrophic commitment).

Strategies:

- **ts**: Thompson Sampling (draw theta ~ Beta, argmax) — the hypotree navigator.
- **random**: uniform random pick from the frontier.
- **greedy**: pure round-robin exploitation — pull each arm once, then always
  pull the arm with the highest *empirical* success rate, breaking ties uniformly
  at random. This is ε-greedy with ε=0: no optimism, no recovery once committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from eval.runner.config import AblationConfig
from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status
from hypotree.navigator.sampler import ThompsonSampler

# -- Bandit arm rates ---------------------------------------------------------
# A big-gap bandit: one moderate winner and a cluster of far-worse decoys. The
# gap is what makes a greedy lock-in catastrophic — committing to a decoy loses
# WINNER_RATE - DECOY_RATE reward on every remaining pull.

# The unique optimum. Kept moderate (not near 1.0) so that its first sample fails
# often enough (~45% of the time) to regularly trap a pure-exploitation greedy.
WINNER_RATE = 0.55

# The decoy cluster rate — far below the winner. Any strategy that commits here
# bleeds a large, constant per-pull regret.
DECOY_RATE = 0.12

# Number of identical decoy arms. Enough that when the winner's first sample
# fails, some decoy has almost certainly looked good early, giving greedy a
# high-rate-looking arm to latch onto.
N_DECOYS = 10

# Convergence tuning for the ablation's sampler. epsilon_ci is set extremely
# tight and n_max is set above the horizon so NO arm ever converges/verifies
# within a run. That keeps the arm set fixed for the whole horizon (a verified
# node would leave the frontier and change the bandit mid-run) and turns the
# ablation into a pure cumulative-regret measurement with no early termination.
ABLATION_EPSILON_CI = 0.001

# Fixed horizon: the number of pulls every strategy spends on one bandit. Long
# enough that a greedy catastrophic lock-in accumulates decisively more regret
# than TS's bounded exploration cost.
ABLATION_HORIZON = 300

# n_max ceiling, set one above the horizon so the convergence gate is never
# forced by the sample-count ceiling within a run.
ABLATION_N_MAX = ABLATION_HORIZON + 1

# RNG stream offsets. Selection randomness (greedy/random tie-breaks) and probe
# noise (Bernoulli outcomes) draw from independent streams so that consuming one
# never perturbs the other across strategies — keeping the comparison fair.
_NOISE_SEED_OFFSET = 100_003
_SELECT_SEED_OFFSET = 200_003


@dataclass
class AblationResult:
    """Outcome of one strategy on one seed.

    ``cumulative_regret`` is the primary metric: the summed per-pull regret
    (``winner_rate - selected_rate``) over the full horizon. Lower is better.
    """

    strategy: str
    seed: int
    rng_seed: int
    cumulative_regret: float
    total_pulls: int = 0


def _empirical_mean(node: Node) -> float | None:
    """Empirical success rate of an arm, or ``None`` if it has never been pulled.

    Nodes carry a Beta(1, 1) prior, so ``alpha - 1`` counts observed successes
    and ``beta - 1`` counts observed failures. A never-pulled arm has zero
    observations and no defined empirical mean — the signal round-robin greedy
    uses to know it must sample the arm at least once before exploiting.
    """
    successes = node.alpha - 1.0
    failures = node.beta - 1.0
    count = successes + failures
    if count <= 0:
        return None
    return successes / count


def _populate_bandit(engine: HypoTreeEngine) -> dict[str, float]:
    """Build the seeded bandit: a verified root gateway → stochastic arms.

    Returns a mapping ``{arm_node_id: hidden_success_rate}`` used to draw noisy
    probe outcomes. The arms are stochastic-regime nodes so the engine folds
    each outcome into a Beta posterior.
    """
    root_id = "root"
    engine.create_hypothesis("Root gateway", node_id=root_id, if_exists="overwrite")

    arm_rates: dict[str, float] = {}
    arms: list[tuple[str, float]] = [("arm_winner", WINNER_RATE)]
    arms += [(f"arm_decoy_{i}", DECOY_RATE) for i in range(N_DECOYS)]

    for node_id, rate in arms:
        engine.create_hypothesis(
            statement=node_id,
            parent_ids=[root_id],
            edge_type=EdgeType.DEPENDENCY,
            evidence_regime="stochastic",
            node_id=node_id,
            if_exists="overwrite",
        )
        arm_rates[node_id] = rate

    # Verify the root so its DEPENDENCY children enter the frontier.
    engine.update_status(root_id, Status.VERIFIED, reason="root gateway auto-verified")
    engine._sync_graph_from_store()
    return arm_rates


def _select_target(
    engine: HypoTreeEngine,
    frontier: list[Node],
    strategy: str,
    pulls: int,
    select_rng: np.random.Generator,
) -> tuple[str | None, str | None]:
    """Pick the next arm + issue a claim for it. Returns (node_id, claim_id).

    Returns ``(None, None)`` when TS reports the frontier is exhausted (the
    "DONE" sentinel) — which cannot happen here since no arm ever verifies.
    """
    if strategy == "ts":
        # Use the full engine path so the claim is created in the store and TS
        # draws from the engine's own seeded sampler.
        targets = engine.get_next_targets()
        if targets[0].status == "DONE":
            return None, None
        return targets[0].node_id, targets[0].claim_id

    if strategy == "random":
        pick = frontier[int(select_rng.integers(len(frontier)))]
    elif strategy == "greedy":
        # Pure round-robin exploitation. Sample every arm once (no defined
        # empirical mean yet), then always exploit the highest empirical rate,
        # breaking ties uniformly at random. No optimism and no recovery: once
        # every arm has a sample, a committed decoy is never abandoned unless its
        # own running mean falls below another arm's.
        unpulled = [n for n in frontier if _empirical_mean(n) is None]
        if unpulled:
            pick = unpulled[int(select_rng.integers(len(unpulled)))]
        else:
            means = {n.id: _empirical_mean(n) for n in frontier}
            best_mean = max(means.values())  # type: ignore[type-var]
            top = [n for n in frontier if means[n.id] >= best_mean - 1e-12]  # type: ignore[operator]
            pick = top[int(select_rng.integers(len(top)))]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    claim_id = f"ablation-claim-{pulls}"
    engine._store.create_claim(claim_id, pick.id, datetime.now(), 900)
    return pick.id, claim_id


def _run_strategy(
    engine: HypoTreeEngine,
    arm_rates: dict[str, float],
    strategy: str,
    *,
    seed: int,
    rng_seed: int,
    horizon: int,
    noise_rng: np.random.Generator,
    select_rng: np.random.Generator,
) -> AblationResult:
    """Run one selection strategy for the full horizon, accumulating regret.

    Each iteration selects a frontier arm via the strategy, draws a noisy
    Bernoulli outcome around that arm's hidden rate, and records it as logical
    evidence. Regret for the pull is ``winner_rate - selected_rate``; the total
    over the horizon is the primary metric. No arm ever verifies, so the frontier
    stays fixed and the loop always spends the full horizon.
    """
    best_rate = max(arm_rates.values())
    cumulative_regret = 0.0
    pulls = 0

    for _ in range(horizon):
        engine._store.expire_stale_claims(datetime.now())
        engine._sync_graph_from_store()
        frontier = engine._frontier_nodes()
        if not frontier:
            break

        target, claim_id = _select_target(engine, frontier, strategy, pulls, select_rng)
        if target is None:
            break

        rate = arm_rates[target]
        cumulative_regret += best_rate - rate

        # Noisy probe: a Bernoulli trial around the arm's hidden success rate.
        outcome = 1.0 if float(noise_rng.random()) < rate else 0.0
        engine.record_evidence(target, LogicalEvidence(success=outcome), claim_id=claim_id)
        pulls += 1

    return AblationResult(
        strategy=strategy,
        seed=seed,
        rng_seed=rng_seed,
        cumulative_regret=cumulative_regret,
        total_pulls=pulls,
    )


def run_ablation(config: AblationConfig) -> list[AblationResult]:
    """Run all three strategies on one (task_seed, rng_seed) bandit.

    Each strategy gets a fresh in-memory engine and freshly-seeded RNG streams,
    so the only variable is the selection algorithm. Returns results for ts,
    random, greedy.
    """
    results: list[AblationResult] = []

    for strategy in ("ts", "random", "greedy"):
        engine = HypoTreeEngine(":memory:", rng_seed=config.rng_seed)
        # Tune the convergence gate so no arm ever resolves within the horizon:
        # the arm set stays fixed and the ablation is a pure regret measurement.
        engine._sampler = ThompsonSampler(
            np.random.default_rng(config.rng_seed),
            epsilon_ci=ABLATION_EPSILON_CI,
            n_max=ABLATION_N_MAX,
        )
        arm_rates = _populate_bandit(engine)

        noise_rng = np.random.default_rng(config.rng_seed + _NOISE_SEED_OFFSET)
        select_rng = np.random.default_rng(config.rng_seed + _SELECT_SEED_OFFSET)

        result = _run_strategy(
            engine,
            arm_rates,
            strategy,
            seed=config.seed,
            rng_seed=config.rng_seed,
            horizon=ABLATION_HORIZON,
            noise_rng=noise_rng,
            select_rng=select_rng,
        )
        results.append(result)
        engine.close()

    return results


def run_ablation_all(
    eval_dir: Path,
    run_id: str,
    *,
    task_seeds: list[int] | None = None,
    rng_seeds: list[int] | None = None,
    write_logs: bool = True,
) -> dict[str, list[AblationResult]]:
    """Run the ablation across all pre-registered seeds.

    Writes one ``ablation-seed-*-rng-*.jsonl`` per seed under
    ``eval_dir/runs/<run_id>`` (unless ``write_logs=False``) so analyse_gate.py
    can score criterion 2, and returns a dict keyed by strategy for the CLI
    summary.
    """
    from eval.runner.config import NAVIGATOR_RNG_SEEDS, TASK_SEEDS, make_ablation_config

    task_seeds = task_seeds or TASK_SEEDS
    rng_seeds = rng_seeds or NAVIGATOR_RNG_SEEDS

    by_strategy: dict[str, list[AblationResult]] = {"ts": [], "random": [], "greedy": []}

    for task_seed, rng_seed in zip(task_seeds, rng_seeds, strict=True):
        config = make_ablation_config(task_seed, rng_seed, eval_dir, run_id)
        results = run_ablation(config)
        for r in results:
            by_strategy[r.strategy].append(r)
            if write_logs:
                log_ablation_result(r, config.log_path)

    return by_strategy


def log_ablation_result(result: AblationResult, log_path: Path) -> None:
    """Append one ablation result as a JSONL line."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "event_type": "ablation_result",
        "strategy": result.strategy,
        "seed": result.seed,
        "rng_seed": result.rng_seed,
        "cumulative_regret": result.cumulative_regret,
        "total_pulls": result.total_pulls,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="eval.runner.ablation_navigator",
        description="Run the Thompson-Sampling quality ablation (criterion 2).",
    )
    parser.add_argument("eval_dir", type=Path, help="path to the eval/ directory")
    parser.add_argument("--run-id", required=True, help="identifier isolating this batch's logs")
    args = parser.parse_args()

    results = run_ablation_all(args.eval_dir, args.run_id)

    # Print summary: median/mean over the typical seed, plus the worst-case
    # (max) regret — the statistic where TS's advantage over greedy shows up.
    for strategy, res_list in results.items():
        regrets = [r.cumulative_regret for r in res_list]
        print(
            f"{strategy}: median={np.median(regrets):.1f} "
            f"mean={np.mean(regrets):.1f} max={np.max(regrets):.1f}"
        )
