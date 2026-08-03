# Changelog

All notable changes to hypotree are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-08-02

### Performance
- **Selecting a target is 41× faster on a large belief state**: 1200 nodes went from
  **4 494 ms to 110 ms** per dispatch, and the curve from ~3.8× to ~2.3× per doubling of
  nodes. None of this was visible at the 40–60 nodes an evaluation episode produces, which
  is why it survived 662 tests — but every long-lived workspace lands on the wrong side of
  it, and "the belief state survives across sessions" is the product claim.
  - **The dominant cost was an N+1 query.** `get_all_nodes` derives each node's `parent_ids`
    from the edges table and was issuing **one query per node**; the navigator calls it
    several times per dispatch, so handing out two targets cost ~6 000 SQL round-trips.
    Parents are now resolved for every row in a single query.
  - `HypoTreeGraph.add_edge` ran a full `is_directed_acyclic_graph()` check **per edge**, so
    reloading *E* edges cost *O(E·(V+E))*. New `add_edges_bulk` checks the finished set once
    and rolls back if it is cyclic. The per-edge path stays for interactive creation, where
    naming the offending edge is worth the cost.
  - `HypoTreeGraph.parents()` scanned every edge in the graph and the frontier calls it once
    per node. Parents are now indexed by `dst`.
  - `get_next_targets` syncs the in-memory graph **once per batch** instead of once per pick.
    The batch loop takes leases, which change node rows but never the topology.
  - A long-lived incremental graph — invalidating on write, never rebuilding — is
    deliberately **not** done. A cached graph that disagrees with SQLite would produce
    silently wrong selections, and at 110 ms there is no case for taking that risk.

### Added
- **`record_evidence` is batch-native.** Pass `results=[{node_id, success, depth, claim_id}, …]`
  to report every experiment from one turn in one call. `create_hypotheses`, `update_status`
  and `get_next_targets` have all been batch-native for a while; `record_evidence` was the
  last singular high-frequency tool, and it is what pinned a synchronous agent at two turns
  per experiment — one to probe, one to report. The single-result form still works and is the
  `k=1` shorthand.
  - Results are applied **in order**, so a refutation's cascade lands before the next result
    in the same batch is read and the belief state ends where it would have ended had the
    results arrived separately.
  - A refused report does **not** abort the batch. Every result was paid for by an experiment
    that already ran, so discarding k-1 of them to punish one bad `node_id` destroys evidence.
    The refusals come back under `failed`, named and explained.
  - The fused dispatch runs **once**, after the whole batch, and is still a top-up.
- `get_conflicts` now reports `skipped_no_substitute` per member.

### Fixed
- **A member the diagnosis could not test was reported as cleared.** Substitution progress was
  an integer cursor, which can only say "the first k were dealt with". A member the plan had
  to pass over — because its question had no live alternative left to swap in — sat *behind*
  the cursor the moment a later member was cleared, and was reported as exonerated by an
  experiment that never ran. It could also never be revisited if an alternative freed up
  later. Progress is now a persisted **set of cleared ids**: cleared and skipped are recorded
  as the opposite claims they are, and the plan reconsiders a skipped member on every call.
  Schema `"9"` → `"10"` adds `nogoods.cleared_ids`; the migration is additive and the set is
  back-filled from the old cursor on read, so a diagnosis in flight across the upgrade keeps
  the members it had genuinely cleared. `probe_index` is kept in step and **deprecated**;
  `HypoTreeStore.advance_nogood_probe` warns and will be removed in 0.5.0.
- **Schema versions were ordered as strings.** `"9" > "10"` and even `"3" > "10"` are true
  lexicographically, so the first two-digit schema version made a database with a corrupt or
  older stamp report itself as "written by a newer hypotree" — sending the user to upgrade
  the package instead of keeping the file. Ordered numerically now; a stamp that is not a
  number is not treated as newer, which routes it to the "back it up and open an issue"
  branch where it belongs.
- **A success could resurrect a pruned branch.** Recording a passing result on a `PRUNED` node
  flipped it to `VERIFIED`, leaving the belief state asserting both that the branch is dead
  and that it holds — a confirmed node sitting on an invalidated parent. A pruned node's
  status is a statement about its ancestry, so the measurement and the posterior are kept and
  the status is left alone. Rare while probe and record alternated in fixed pairs; routine
  once results arrive in batches, because one refutation in a batch can prune the node a
  later result in the same batch belongs to.
- **`source_ref` was advertised and dropped.** 0.3.2 added the field, documented it on the
  `record_evidence` tool schema, and never read it in the dispatch — so every value sent over
  MCP was silently discarded. The single and batch paths now build evidence through one
  helper, so they cannot drift apart again.
- **The shortfall recovery fired on a search that had already won.** When every question is
  answered and the assembled answer falls short, the engine hands back the alternatives those
  answers had retired — the reading being that one of the confirmations must be wrong. It did
  not check whether a *different* assembly had since cleared the bar, and a composition that
  succeeded withdraws exactly that reading: the answers were right, the earlier assembly was
  simply the wrong combination of them. One real episode reopened six settled questions on the
  turn it found its answer, because its goal node had never been wired to the winning
  composition and an empty frontier looks the same from the inside either way. The recovery
  now stands down once any composition is `VERIFIED`.
- **Two different recoveries wrote the same reason marker.** `_recover_from_underperformance`
  reused `INTERACTION_REOPEN_PREFIX`, so a history could not distinguish "a conflict turned
  out to be an interaction effect" from "every answer was in and the assembly still missed" —
  and a run report claimed every conflict had been narrowed to a culprit *and* that six
  alternatives had been reopened because a conflict was an interaction effect, which are
  mutually exclusive endings. New `UNDERPERFORMANCE_REOPEN_PREFIX`. Additive: histories written
  before 0.4.0 keep the old prefix and still read as interaction reopens, which is the only
  label that ever existed for them.

### Changed
- `/hypotree-next` and the server-level instructions now teach the batch shape.

### Eval harness
- **One bad result no longer discards the rest of a batch.** The harness looped the
  single-result call, so a batch containing an unknown node id aborted: the results already
  applied stayed in the belief state while the whole call was reported as an error, and every
  result *after* the bad one was never attempted. Each of those is an experiment that has
  already been paid for — which is exactly what `engine.record_results` per-item isolation
  exists to prevent, and the harness was silently overriding it. Never fired in run H (zero
  record failures), so no result is affected.
- Rejected batch items are logged (`record_rejected`) and reported. Per-item isolation makes
  the containing call succeed, so without this a refusal appeared nowhere: not in the tool
  census, which sees an ok call, and not in the evidence count, which never saw the result.
- **`seed_reader` reports probes to name the culprit.** The report said how many conflicts
  were narrowed and never what they cost, so the member ordering — the thing that decides
  that cost, one probe per position — was invisible. Run G: **1.73**. Run H: **2.13**. A run
  can resolve every conflict and still have got slower, and nothing said so.
- **A win no longer cuts the episode off mid-protocol.** The environment decides the win the
  instant a probe clears the target, but the agent is holding an unreported result at that
  moment. On the seeds where the swap that identifies a culprit *is also* the winning
  combination, the engine resolved the conflict and the scoreboard recorded it as unresolved,
  because the record that proves it was never allowed to land — run G's "11 of 13 conflicts
  resolved" was really 13 of 13. The agent now gets a bounded wind-down (2 turns) to file what
  it holds. Further experiments are refused and the step counter is frozen, so this cannot
  move the headline metric; arms with nothing outstanding wind down in zero turns.
- The harness `record_evidence` accepts and instruments the batch shape, attributing every
  belief-revision side effect to the result that caused it rather than to the last in the
  batch. `seed_reader` reports **results per record call** — 1.00 means the batch shape is not
  being used and turns/step cannot fall.
- **One HTTP error no longer kills a 90-episode sweep.** `_call_openai_api` wrapped `urlopen`
  in nothing at all, so a single 5xx from the inference server propagated out of `run()` and,
  under `eval.sh`'s `set -e`, aborted everything that had not run yet. Run `I` died at seed
  1210 of 30 and took 61 episodes with it. Transient faults (5xx, 429, connection failures,
  read timeouts) are now retried with capped exponential backoff plus jitter — five attempts,
  2 s to 60 s — and every retry is logged so a run with odd numbers can be checked against how
  much trouble the transport was having. A 4xx other than 429 still fails immediately: a
  malformed request does not become well-formed by being sent again.
- **An episode that dies is excluded from scoring, not censored at budget.** Exhausted retries
  raise `LLMUnavailableError`, which becomes a clean `run_end` carrying `infra_failed`. The
  gate drops those episodes from the paired set and the reader names them as transport faults.
  Censoring assumes the agent had its full budget and did not solve the task; an episode killed
  by a dropped connection had neither, and scoring it at budget charges one arm for the
  server's bad minute.
- **`eval.sh` survives a failed episode.** The runner call no longer aborts the sweep, failed
  episodes are collected and reported at the end with the `--resume` command that recovers
  them, and the resume check now requires a terminal record *and* the absence of an infra
  marker — otherwise `--resume` would skip precisely the episodes that need re-running.
- `seed_reader` reports **alternatives reopened after a shortfall** separately from those
  reopened by an interaction effect, and lists infrastructure failures apart from episodes
  with no terminal record: one needs the server looked at, the other needs a re-run.

## [0.3.2] - 2026-08-01

### Fixed
- **Conflict diagnosis now interrogates the least-corroborated question first.** v0.3.1 replaced
  the naming-dependent alphabetical order with a hash, which removed the bias but carried no
  signal — the culprit landed at position 3.75 of 5, and conflict episodes cost +1.9
  probes. A member is only as trustworthy as the question behind it was searched: beating four
  refuted rivals is a real search, being the first thing tried while the exclusion inference
  quietly retired the rest is not. Ordering by how many siblings were eliminated *on their own
  evidence* moves the culprit to position 2.38 and is still independent of what nodes are named.
  Self-play: 16.2 → **15.9** probes, better than the alphabetical order it replaced (16.0).

### Added
- `list_nodes(view=...)` — named presets (`frontier`, `settled`, `verified`, `revision`, `stale`)
  instead of hand-assembled status filters that silently return an empty table.
- `list_nodes(stale_only=True)` and a `Stale` column: VERIFIED nodes whose newest evidence names
  a commit that is no longer checked out. `context_hash` had been captured on every evidence row.
- `LogicalEvidence.source_ref` — what was actually run to produce a number (path, URL, CI run id).
- `AGENT_GUIDE.md` now documents the 3 MCP prompts and 2 resources.

### Changed
- **Schema upgrades are now migrated**: the belief state is the accumulated
  record of a month of experiments, and surviving across sessions is the entire product claim.
  Opening a database written by an older release now walks a forward migration chain, applying
  each step and its version stamp in one transaction — an interrupted upgrade rolls back to the
  version it started from rather than stranding the file between two. Every migration is written
  to the `events` audit log as `SchemaMigrated`, so "why does this differ from my backup?" is
  answerable from the same place as every other question about the workspace.
- Schema `"8"` → `"9"` adds `evidence.source_ref`; the 8→9 migration is additive
  (`ALTER TABLE ... ADD COLUMN`) and cannot lose data.
- A database written by a **newer** hypotree is refused rather than downgraded — the newer code
  may have stored things this version cannot represent, and dropping them silently is worse than
  stopping. A version with no route forward now says to keep the file and open an issue.

### Known limitation
- A conflict member skipped for want of a usable substitute is left behind the `probe_index`
  cursor and reported as cleared. Distinguishing "cleared" from "skipped" needs a settled-set
  rather than an integer cursor, so it is scoped as a schema change rather than patched.

## [0.3.1] - 2026-08-01

### Fixed
- Windows CI: e2e stdio test timeout increased to 900s for slow GitHub Actions runners
- Conflict diagnosis no longer depends on hypothesis **naming**. Members were stored
  alphabetically and interrogated in that order, so the number of diagnostic swaps was
  decided by the id prefix rather than by the evidence — a workspace could be made to
  converge faster by renaming its hypotheses. Members are now ordered weakest-claim-first
  (`confirmed_depth`, then `evidence_count`), tie-broken on a hash of the id, and frozen
  at conflict creation so `probe_index` keeps counting into a stable list.
- `AGENT_GUIDE.md`: tool table said 17 and listed 17 while the server registers 18 —
  `get_workspace_info` was missing. The identity section documented a 5-layer resolution
  including a `UV_PROJECT_ROOT` layer that no longer exists, a 64-character name limit
  that is now 128, and `~/.local/share` as the only store location; rewritten to the real
  four layers, the `[a-z0-9][a-z0-9._~-]{0,127}` rule, the cross-platform Windows
  reserved-name rejection, and `%LOCALAPPDATA%` / `hypotree --info`.

### Changed
- `HypoTreeStore.add_nogood` now preserves the caller's member order verbatim instead of
  re-sorting it. The order is the diagnosis order and `probe_index` counts into it, so
  re-sorting silently overrode the engine's ranking.

### Eval harness
- `pruned_reexecution` no longer fires when a result was *obtained before* the branch was
  pruned. It could not distinguish spending a fresh probe on a dead branch from filing a
  measurement already paid for, and the batch-probe protocol makes the latter routine —
  a single such event flipped a run's hard gate. Transcript entries now carry a UTC
  timestamp; unknown timings still fail open so fabricated results are caught.
- Status transitions are now diffed on `(status, reason)` rather than status alone.
  Ruling a substitute out retracts its exclusion and re-settles it at the same instant,
  so a status-only diff saw no change and the mechanism reported zero forever.
- Criterion 4 derives its p-value from a seeded 20 000-draw permutation test when the
  smallest expected cell falls below 5, instead of reporting a verdict from an asymptotic
  χ² that does not apply at that sparsity.

## [0.3.0] - 2026-07-31

### Added — The adversarial eval gate passed (GO)
- **18 MCP tools**: `generate_learning_path`, `get_workspace_info` added (was 17)
- 3 MCP prompts as slash commands: `/hypotree-init`, `/hypotree-next`, `/hypotree-status`
- 2 MCP resources: `hypotree://guide`, `hypotree://state`
- Server-level `instructions` delivered at MCP handshake
- `AGENT_GUIDE.md` shipped inside the package, not as a repo-root file
- CLI: `hypotree --version`, `--help`, `--info`
- Cross-platform CI: matrix over ubuntu/windows/macOS × py3.10-3.13
- Cross-platform identity resolution: uses `%LOCALAPPDATA%` on Windows, `$XDG_DATA_HOME` on Linux/macOS
- `eval/runner/config.py`: same XDG_DATA_HOME pattern fixed for cross-platform
- PyPI metadata: authors, license, classifiers, project.urls, py.typed
- `hypotree.yaml.template` updated with cross-platform storage paths

### Gate results (Run D, qwen3.6:27b-q8_0, 30 seeds)
- C1b (B vs F): 7.5 steps median reduction (28.9%), δ=0.89, 30/0/0 wins — **PASS**
- C1a (B vs A): 25.0 steps (57.5%), δ=1.0, 30/0/0 — **PASS**
- C2 (navigator): worst case 55% below greedy, 0/30 lock-ins — **PASS**
- C3 (revision): 0 pruned re-executions, 104 revision events, 14/15 conflicts resolved — **PASS**
- C4 (status utility): χ²=45.2, V=0.185, decision=KEEP — **PASS**

### Engine changes
- Differential ablation: `_eliminate_substitute` now mirrors `_confirm_substitute`
- Fabricated claim_id handling: dropped silently, result kept
- Double-counting fix: `node_created` events no longer added to `bulk_create` counts
- Group adoption denominator: compositions without exclusion_group correctly counted
- Record→dispatch fusion: `count_next_targets` parameter on `record_evidence`

## [0.2.0] - 2026-07-25

### Added
- Thompson Sampling navigator with Beta distributions
- Cascading prune, exclusion-group inference, deduction by elimination
- ATMS-style conflict sets with differential ablation
- SQLite-WAL schema v8 (9 tables, bi-temporal history)
- 16 MCP tools over stdio
- Pre-registered adversarial eval harness (30 seeds × 3 arms)

## [0.1.0] - 2026-07-20

### Added
- Core engine: HypoTreeEngine, DAG model, status lifecycle
- SQLite store with WAL mode
- Basic MCP server (stdio transport)
- Initial test suite

[0.4.0]: https://github.com/tygryso/hypotree/releases/tag/v0.4.0
[0.3.1]: https://github.com/tygryso/hypotree/releases/tag/v0.3.1
[0.3.0]: https://github.com/tygryso/hypotree/releases/tag/v0.3.0