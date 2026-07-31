---
name: seed-reader
description: 'Render evaluation run reports and interpret gate metrics. Use for: reading eval JSONL logs as a markdown report (uv run python -m eval.seed_reader --run-id ID), listing available run-ids, scoring the GO/STOP/ITERATE gate decision, understanding what each metric means, diagnosing why an arm won or lost, reviewing probe economy and belief-state mechanics. Keywords: eval report, seed_reader, analyse_gate, run-id, gate decision, criterion 1a/1b/2/3/4, moat, ablation, exclusion groups, conflicts, stratified analysis, confirmation depth, step counts.'
---

# seed-reader — Evaluation Report Generation & Metric Interpretation

Render one evaluation run's JSONL logs as a self-contained markdown report, and
understand every metric in it. Also covers the gate-scoring companion
(`analyse_gate`) that emits the frozen GO / STOP / ITERATE decision.

## When to Use

- After an eval run completes and you need to read its results
- When asked "how did the run go?" or "did hypotree win?"
- When debugging why an arm performed a certain way (duplicates, conflicts, recovery, etc.)
- When you need the formal gate decision (GO / STOP / ITERATE)
- When comparing two runs or diagnosing data-quality issues

## How to List Available Run-Ids

Every run's logs live in `eval/runs/<run-id>/`. The directory names are the run-ids.

```bash
# List all run-ids (directories that contain episode logs)
ls eval/runs/

# Check which runs have completed (have a gate_decision.json)
ls eval/runs/*/gate_decision.json 2>/dev/null

# Check a specific run's contents
ls eval/runs/<run-id>/
```

Run-ids are validated: 1–64 chars of `[A-Za-z0-9._-]`. Typical convention:
`YYYY-MM-DD<x>` (e.g., `2026-07-28a`).

## Step 1 — Generate the Report

```bash
# Render to stdout
uv run python -m eval.seed_reader --run-id <RUN_ID>

# Write to a file (recommended — the report is long)
uv run python -m eval.seed_reader --run-id <RUN_ID> --output eval/runs/<RUN_ID>/REPORT.md

# Non-default eval directory
uv run python -m eval.seed_reader --run-id <RUN_ID> --eval-dir /path/to/eval/
```

**`--run-id` is mandatory.** Reports are only comparable within a run — mixing
two runs' logs would silently invalidate every aggregate. The tool enforces this
by checking that each log's own `run_id` field matches the directory.

## Step 2 — Score the Gate Decision

```bash
# Emits JSON: {"decision": "GO"|"STOP"|"ITERATE", "criteria": {...}, ...}
uv run python -m eval.analyse_gate eval/ --run-id <RUN_ID>

# Save alongside the report
uv run python -m eval.analyse_gate eval/ --run-id <RUN_ID> | tee eval/runs/<RUN_ID>/gate_decision.json
```

`eval.sh` runs both automatically (steps 5 and 6), but you can re-run them
manually for re-rendering without re-running the full evaluation.

## Step 3 — Interpret the Report

The report has 10 sections, ordered from conclusion to evidence. An agent that
reads only sections 1–2 gets the correct headline.

---

### §1 Headline — Steps to Target

Per-arm outcome table. **This is the most important table.**

| Column | Meaning |
|--------|---------|
| `arm` | A = manual scratchpad, B = hypotree treatment, F = flat auto-persisted log |
| `n` | Number of completed episodes (episodes without `run_end` are excluded) |
| `mean` / `median` / `stdev` | Steps (dispatched experiments) to reach the goal |
| `range` | Min–max steps |
| `goals met` | How many episodes actually reached the target (vs budget-censored) |
| `waste` | Steps above the perfect-recall reference strategy for that seed |

**Key reading:** lower steps = better. `waste` = 0 means the agent was optimal.
Arm B should beat F (criterion 1b) and A (criterion 1a).

### §2 Paired Comparisons

Seed-by-seed paired differences. **This is where the gate statistics live.**

| Column | Meaning |
|--------|---------|
| `n` | Seeds where BOTH arms completed |
| `win/loss/tie` | Treatment wins / loses / ties vs baseline (per-seed) |
| `mean diff` / `median diff` | `baseline_steps − treatment_steps` (positive = treatment faster) |
| `p` | Two-sided exact sign test (non-parametric; step counts are skewed) |

Three comparisons: **B vs F** (criterion 1b — the hypothesis under test), **B vs A**
(criterion 1a — ergonomic), **F vs A** (baselines against each other).

**Per-seed `F − B` line:** the raw per-seed diffs, so you can see which seeds
drove the result.

### §3 Per-Episode Detail

One row per (seed, arm) episode. Use this to find outliers and diagnose specific
seeds.

| Column | Meaning |
|--------|---------|
| `steps` | Total experiments dispatched |
| `goal` | Whether the target metric was reached |
| `dup` | Duplicate probes (re-probed a config already probed) |
| `waste` | Steps above the reference strategy |
| `revision` | Genuinely destructive propagation only (`NEEDS_REVISION`, `PRUNED`) |
| `excluded` | UNTESTED→EXHAUSTED — a question retired by a confirmed answer (the mechanism working) |
| `reopened` | EXHAUSTED→UNTESTED transitions (exclusion retracted after a refutation) |
| `deduced` | Members confirmed by elimination, without spending a probe |
| `conflicts` | Conflict sets recorded (multi-assumption failures, not blamed yet) |
| `group adoption` | % of nodes that declared an `exclusion_group` |
| `best score` | Highest success metric achieved |

### §4 Probe Economy

Where the budget went. **Duplicates are the key efficiency signal.**

| Column | Meaning |
|--------|---------|
| `probes` | Total experiments |
| `duplicates` | Re-probes of already-tested configs (should be ~0 for arm B) |
| `premise` | Single-axis probes (`component=v2`, etc.) |
| `combination` | Full multi-axis config probes |
| `decoy` | Probes containing the seed's decoy value (confirms shallow, refutes deep) |
| `best axes` | How many axes the closest probe got right (max = number of axes) |
| `mean waste` | Average steps above reference strategy |

**Key reading:** High `duplicates` in arm B = memory/revision failure. Low
`group adoption` = the LLM didn't declare exclusion groups. High `decoy` probes
that fail at depth = the trap fired.

### §5 Belief-State Mechanics (Arm B only)

What the hypotree engine actually did. **This validates the mechanism, not just
the step count.**

| Metric | Meaning |
|--------|---------|
| `nodes created` | Total hypothesis nodes (single + bulk) |
| `declared an exclusion group` | % of nodes with `exclusion_group` set — the biggest efficiency lever |
| `episodes using exclusion groups` | How many episodes adopted exclusion at all |
| `navigator targets handed out` | `get_next_targets` entries that returned a node |
| `targets already settled` | Should be **0** — a settled node handed back is wasted budget |
| `competing answers in one batch` | Should be **0** — two members of one exclusion group dispatched together, so the first result cannot retire the second |
| `exclusions applied` | Questions retired by a confirmed answer — the efficiency mechanism working |
| `members deduced by elimination` | Confirmed with no probe because every alternative was ruled out |
| `alternatives reopened by an interaction effect` | A conflict whose members all held individually — the answer must be among the alternatives they retired |
| `destructive revisions` | `NEEDS_REVISION`/`PRUNED` — belief withdrawn because something above it failed |
| `exclusions retracted` | Siblings handed back after their justification was withdrawn |
| `conflicts recorded` | Multi-assumption failures recorded as nogoods (not blamed yet) |
| `conflicts resolved to a culprit` | Nogoods narrowed to a single guilty premise |
| `pruned re-executions` | Should be **0** — re-running settled work |
| `results filed against a goal (refused)` | Should be **0** — the engine rejects these, so nothing is corrupted, but each one is a probe paid for whose hypothesis is still untested and whose failure raised no conflict |
| `evidence-regime overrides` | Agent asked for a regime the environment doesn't have |

**Status transitions table** at the bottom shows every `OLD->NEW` transition and
its count.

### §6 Stratified by Conflict (Arm B)

**The most explanatory table in the report.** Splits the paired B-vs-F result by
whether the episode hit an indeterminate multi-assumption failure.

| stratum | What it measures |
|---------|------------------|
| `no conflict` | How well the belief state *eliminates candidates* |
| `conflict fired` | How well it *recovers from an indeterminate failure* |

These are two different competences and pooling them hides which one is limiting.
On run `20260727a` the arms looked level overall (+11.1%) while the strata were
**+25.0% (16/17 wins)** and **−12.3% (1/13 wins)** — the entire criterion-1b
failure lived in the recovery path.

The `Recovery health` line underneath reports conflicts narrowed to a culprit,
alternatives reopened after a conflict was shown to be an interaction effect,
alternatives reopened per conflict episode, and duplicate probes per conflict
episode.

A conflict has **two** honest endings and only one names a culprit. Either a
member is refuted (culprit found), or every member survives a re-test at the
failing depth — in which case nobody is guilty, the failure is an *interaction*,
and the answer lies among the alternatives those members retired. **If neither
counter moves, the recovery never completed and the agent searched blind**, which
is what a large negative `conflict fired` stratum looks like from the inside.

Note that `conflicts resolved to a culprit` alone is not the health metric: a run
can legitimately reach `0/N` resolved while every conflict was correctly settled
as an interaction effect.

### §7 Memory Maintenance Across Context Resets

How each arm survived the enforced context resets.

| Column | Meaning |
|--------|---------|
| `resets/episode` | How many context clears happened |
| `mean summary chars` | The belief-state summary injected after each reset |
| `note writes` | Manual scratchpad writes (arms A and F) |
| `mean note chars` | Scratchpad size |
| `never wrote notes` | Episodes where the agent maintained zero manual memory |

**Key reading:** If arm A has many "never wrote notes", the A-vs-B comparison is
confounded (memory vs no-memory, not structure vs flat).

### §8 Cost per Arm

LLM turns, latency, and tokens. **Step counts hide this overhead.**

| Column | Meaning |
|--------|---------|
| `turns/episode` | Mean LLM round-trips per episode |
| `turns/step` | LLM calls per dispatched experiment (arm B > arm F due to bookkeeping) |
| `mean latency` | Seconds per LLM call |
| `wall/episode` | Wall-clock minutes per episode |
| `prompt tok/turn` | Prompt tokens per LLM call |
| `completion tok/turn` | Completion tokens per LLM call |

Empty if the run used the mock backend (no real LLM).

### §9 Navigator Ablation (Criterion 2)

TS vs random vs greedy on a synthetic bandit (no LLM, no landscape).

| Column | Meaning |
|--------|---------|
| `mean regret` / `median` | Lower = better |
| `max` | Worst-case regret — where TS should beat greedy (bounded failure) |

**Key reading:** TS beats random typically. TS does NOT beat greedy typically
(greedy wins the median). The gated property is **worst-case bounded regret** —
TS never catastrophically locks a decoy, greedy sometimes does.

### §10 Data-Quality Warnings

Everything that makes a number above less trustworthy. **Read this before
quoting any result.** Common warnings:

- Logs from multiple runs mixed in one directory (invalidates all aggregates)
- Episodes ran against different models (not comparable)
- Incomplete episodes (crashed, excluded from paired comparisons)
- Missing seeds (unbalanced comparisons)
- Budget-censored episodes (ended without meeting goal — step count understates)
- Navigator handed back settled nodes (sampler bug)
- No `llm_call` events (mock backend — step counts reflect a scripted agent)

---

## Gate Decision Reference

`analyse_gate` emits `{"decision": "...", "criteria": {...}}`. The decision rule:

| C1 (moat) | C2 (TS) | C3 (revision) | C4 (status) | Decision |
|-----------|---------|---------------|-------------|----------|
| FAIL | any | any | any | **STOP** |
| PASS | FAIL | any | any | **STOP** |
| PASS | PASS | FAIL | any | **ITERATE** |
| PASS | PASS | PASS | FAIL | **ITERATE** |
| PASS | PASS | PASS | PASS | **GO** |

### Criterion Details

**C1 — Moat (gated on 1b):**
- **1a (ergonomic):** B vs A — ≥25% median step reduction, strict majority wins, Cliff's δ ≥ 0.33. *Reported, not gated.*
- **1b (informational):** B vs F — same thresholds. **This gates the decision.** Both arms keep every fact for free; the only difference is structure.

**C2 — TS Quality:**
- TS vs random: ≥20% median regret reduction on ≥ majority of seeds
- TS vs greedy: ≥20% worst-case (max) regret reduction

**C3 — Revision:**
- Zero pruned re-executions (hard gate)
- Revision events > 0: upstream propagations + conflicts recorded + culprits identified

**C4 — Status Utility:**
- χ² on action taxonomy; defaults to collapse when underpowered (<5 BLOCKED/NEEDS_REVISION events)

---

## Quick Reference Commands

```bash
# List run-ids
ls eval/runs/

# Generate report for a run
uv run python -m eval.seed_reader --run-id <RUN_ID> --output eval/runs/<RUN_ID>/REPORT.md

# Score the gate
uv run python -m eval.analyse_gate eval/ --run-id <RUN_ID>

# Both at once (what eval.sh does)
uv run python -m eval.analyse_gate eval/ --run-id <RUN_ID> | tee eval/runs/<RUN_ID>/gate_decision.json
uv run python -m eval.seed_reader --run-id <RUN_ID> --output eval/runs/<RUN_ID>/REPORT.md

# Full evaluation run (on the GPU server)
./eval.sh --run-id <RUN_ID> --llm-model <MODEL> --llm-base-url <URL>
```

## File Layout

```
eval/
├── seed_reader.py          # Report generator (this skill's subject)
├── analyse_gate.py         # Gate scorer (companion)
├── runner/
│   └── config.py           # TASK_SEEDS (1201–1230), ALL_ARMS (B, F, A), run-id validation
├── environment/
│   ├── landscape_scoring.py # TARGET_METRIC, scoring logic, reference strategy
│   ├── landscapes/          # Ground-truth JSON per seed
│   └── briefings/           # Human-readable task briefings per seed
└── runs/
    └── <run-id>/
        ├── seed-1201-arm-B.jsonl    # Episode logs (3 arms × 30 seeds)
        ├── ablation-seed-1201-rng-2001.jsonl  # Criterion 2 ablation
        ├── gate_decision.json       # analyse_gate output (if eval.sh ran)
        └── REPORT.md                # seed_reader output (if eval.sh ran)