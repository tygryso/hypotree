# Changelog

All notable changes to hypotree are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.3.1]: https://github.com/tygryso/hypotree/releases/tag/v0.3.1
[0.3.0]: https://github.com/tygryso/hypotree/releases/tag/v0.3.0