# hypotree — Memory That Forgets

[![CI](https://github.com/tygryso/hypotree/actions/workflows/ci.yml/badge.svg)](https://github.com/tygryso/hypotree/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-lightgrey.svg)](CHANGELOG.md)
[![Tests: 669](https://img.shields.io/badge/tests-669-brightgreen.svg)](tests/)
[![Version: 0.4.0](https://img.shields.io/badge/version-0.4.0-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/hypotree.svg)](https://pypi.org/project/hypotree/)

A persistent, self-revising **hypothesis DAG** for agentic R&D — exposed as an [MCP server](https://modelcontextprotocol.io/).

Current agent memory is passive: vector stores and scratchpads accumulate facts but never revise them. Hypotree structures the agent's working knowledge as a **directed acyclic graph** of hypotheses backed by SQLite-WAL. When an experiment fails, the engine walks the dependency edges and retracts what rested on it. When a premise collapses, every dependent subtree is pruned automatically.

---

## What it does

- **Write-back belief revision** — an ATMS-style engine (de Kleer, 1986) that propagates evidence failures upstream through the dependency graph.
- **Cascading prune** — invalidating a parent hypothesis instantly transitions its entire subtree to `PRUNED`. No tokens spent on dead branches.
- **Exclusion-group inference** — confirming one member of a mutually exclusive group retires the rest as `EXHAUSTED` without probing them.
- **Deduction by elimination** — last-man-standing: when all but one alternative in an exclusion group are refuted, the survivor is `VERIFIED` without a probe.
- **Thompson Sampling navigation** — Beta-distribution sampling over the open frontier, giving bounded worst-case regret (no catastrophic lock-in).
- **Conflict resolution via differential ablation** — when an integration test fails but every component passes alone, the engine rebuilds the failing combination one swap at a time to pinpoint the culprit.
- **A derivation trail, not just a state** — `generate_learning_path` narrates what was settled, in order, separating what an experiment paid for from what the engine inferred for free, and calling out beliefs that were later withdrawn.
- **Persistent across sessions, models, agents, users, and projects** — the belief state is a SQLite database, not a context window.

---

## Install

```bash
# From PyPI
uvx hypotree
# or
pip install hypotree

# From source
git clone https://github.com/tygryso/hypotree.git
cd hypotree
uv sync
```

**Requires:** Python 3.10+ · **Runs on:** Linux, macOS, Windows

Check the install without wiring up a client — the server speaks JSON-RPC on stdin, so starting it in a terminal otherwise looks like a hang:

```bash
hypotree --version   # or: uvx hypotree --version
hypotree --info      # which belief state am I connected to, and where is it?
```

---

## Quick start

### 1. Connect to an MCP client

Add hypotree to your MCP client config (Cursor, Cline, Claude Desktop, etc.):

```json
{
  "mcpServers": {
    "hypotree": {
      "command": "uvx",
      "args": ["hypotree"],
      "env": {
        "HYPOTREE_WORKSPACE_ID": "my-project"
      }
    }
  }
}
```

Or run directly:

```bash
uvx hypotree
```

### 2. Create hypotheses

The agent creates a tree with `parent_ids` wiring combinations to their premises and `exclusion_group` declaring competing answers to one question:

```python
# Agent calls over MCP:
create_hypotheses(hypotheses=[
    {"node_id": "catalyst_A", "statement": "Pd/C catalyst works", "exclusion_group": "catalyst"},
    {"node_id": "catalyst_B", "statement": "Pt catalyst works",   "exclusion_group": "catalyst"},
    {"node_id": "catalyst_C", "statement": "Ni catalyst works",   "exclusion_group": "catalyst"},
    {"node_id": "yield_target", "statement": "reach 90% yield",
     "is_goal": True, "target_metric": 0.9, "parent_ids": ["catalyst_A"]},
])
```

### 3. Record evidence and let the engine infer

```python
# Probe catalyst_A → fails outright, catalyst_B → fails outright.
# Two experiments, one call:
record_evidence(results=[
    {"node_id": "catalyst_A", "success": 0.0},
    {"node_id": "catalyst_B", "success": 0.0},
])
# Engine: catalyst_A, catalyst_B → INVALIDATED; anything depending on them → PRUNED
#         catalyst_C → VERIFIED by elimination — no probe spent
```

### 4. Ask what you learned

```python
generate_learning_path()
# → markdown briefing + counters:
#   probes_spent = 2, conclusions = 3, conclusions_without_a_probe = 1
```

---

## MCP tools (18)

| Tool | What it does |
|------|-------------|
| `create_hypotheses` | Create one or many nodes with `parent_ids`, `exclusion_group`, `is_goal` |
| `get_next_targets` | Thompson Sampling — returns the next hypothesis to test, under a lease |
| `record_evidence` | Record one result — or every result from a turn at once with `results=[…]` — and trigger write-back propagation |
| `generate_learning_path` | What we learned, in order, and what it cost — separates conclusions an experiment paid for from ones the engine inferred free |
| `get_workspace_info` | Which belief state you are connected to and which layer chose it — start here when the graph is unexpectedly empty |
| `update_status` | Manually set node status (rarely needed — the engine does it) |
| `get_dag_context` | Get a subgraph view for the agent's context window |
| `render_dag_map` | Mermaid.js diagram of the current belief state |
| `get_goal_status` | Check whether the goal node is met |
| `get_conflicts` | List unresolved conflicts (integration failures) |
| `suggest_discriminating_experiment` | For a conflict, suggest the swap that separates the culprits |
| `list_nodes` | List/filter nodes by status, depth, or exclusion group |
| `get_evidence_history` | Full evidence trail for a node |
| `get_active_claims` | List nodes with active leases |
| `renew_claim` | Extend a lease on a node |
| `release_claims` | Release one or all leases |
| `invalidate_upstream` | Revert `VERIFIED` status from parents based on child failures |
| `verify_upstream` | Propagate confirmation up the dependency chain |

---

## Slash commands

The server ships three MCP **prompts**. Clients that support them (Cursor, Claude Desktop, Cline) surface them as slash commands, so a human can steer the loop without retyping the protocol — and, more usefully, without the agent paraphrasing it.

| Command | What it does |
|---------|-------------|
| `/hypotree-init` | Create the goal node and the first 3–5 hypotheses under it, with exclusion groups where the hypotheses are competing answers to one question |
| `/hypotree-next` | Get the next target, actually test it, and record the result against that same node — including what to do for each DONE reason |
| `/hypotree-status` | Brief you on what is established, what was ruled out, what changed, and how many conclusions cost no experiment |

`/hypotree-init` takes an optional `task` argument. Exact invocation depends on the client (Cursor and Claude Desktop namespace prompts under the server, e.g. `/hypotree:hypotree-init`).

---

## Resources

Two MCP **resources**, pulled on demand rather than carried in context:

| URI | What it is |
|-----|-----------|
| `hypotree://guide` | The full agent contract — every tool, the status lifecycle, exclusion groups, leases, confirmation depth, conflict sets, and the rules. ~23 KB, so it belongs nowhere near a system prompt |
| `hypotree://state` | The current belief state as a narrative: what was established, how, and what it cost |

---

## Agent rules — how your agent learns to use this

The operating contract reaches the model through four channels. You do not have to wire any of them up; they are listed so you know what is already in context and what is not.

1. **Server instructions.** MCP hands a server-level `instructions` block to the client during `initialize`, and every major client puts it in front of the model. Hypotree uses it for four rules: one hypothesis per node, mark the goal with `is_goal=True` and wire it to the work, record against the node you actually tested, and report what you were leased. Nothing to configure.
2. **Tool descriptions.** Each tool description carries the one rule that tool is misused without — that a goal never accepts evidence, that a lease reserves a node until you report it, that confirming one member of an exclusion group retires the rest. These are the only text guaranteed to be in context at the moment a tool is chosen.
3. **Resources.** The full guide is `hypotree://guide`. An agent that hits something surprising can read it without you pasting 23 KB into a system prompt.
4. **Your project rules file** — optional, and the only part you touch. If you want the agent to reach for hypotree *unprompted* on multi-day work, add the block below.

### Optional: `.cursorrules` / `AGENTS.md` / `CLAUDE.md`

```markdown
## Long-running R&D: use hypotree

For any task that spans more than one session, branches into competing
approaches, or where an early assumption could turn out wrong later, keep the
belief state in hypotree rather than in the conversation.

- Before starting, call `generate_learning_path`. Something may already be
  settled, and re-deriving it costs an experiment you do not have to run.
- Create the objective with `is_goal=True` and wire hypotheses to it with
  `parent_ids`. Progress is then derived, not asserted.
- Competing answers to one question share an `exclusion_group`. Confirming one
  retires the rest without testing them — this is where most of the saving is.
- Ask `get_next_targets` for work and record every result you were handed. A
  target is leased to you; anything you hold and never report is work nobody
  can do. Probed several things in one turn? Report them in one call with
  `record_evidence(results=[...])`.
- Record against the node whose statement you actually tested. A composition's
  failure filed against a premise destroys a confirmation that is still true.
- When `get_next_targets` returns DONE, read the reason. Only `all_goals_met`
  and `empty_frontier` mean stop; the rest are instructions.
```

---

---

## Architecture

```
┌─────────────────────────────────────────┐
│           MCP Client (agent)            │
│   Cursor / Cline / Claude Desktop       │
└──────────────┬──────────────────────────┘
               │  MCP protocol (stdio/HTTP)
┌──────────────▼──────────────────────────┐
│         hypotree MCP Server             │
│  ┌─────────────────────────────────┐    │
│  │     Engine (18 tools)           │    │
│  │  • Write-back propagation       │    │
│  │  • Cascading prune              │    │
│  │  • Exclusion-group inference    │    │
│  │  • Differential ablation        │    │
│  │  • Thompson Sampling navigator  │    │
│  └──────────┬──────────────────────┘    │
└─────────────┼───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│   SQLite-WAL (Schema v10, 9 tables)      │
│   • Bi-temporal history                 │
│   • Belief state + evidence + conflicts │
│   • Keyed by workspace_id               │
└─────────────────────────────────────────┘
```

---

## Evaluation

Hypotree is validated by a **pre-registered adversarial benchmark** using *qwen3.6:27b-q8_0* and *gemma4:31b-it-q4_K_M*. The benchmark is a set of 30 seeded combinatorial R&D problems, each with 3125 combinations (5 axes × 5 values). Each arm is run on all seeds, and the gate criteria are scored against the pre-registered thresholds.

**Three arms across 30 seeded combinatorial R&D problems:**

- **Arm A** — LLM agent with a manual Markdown scratchpad (ergonomic floor)
- **Arm F** — LLM agent with perfect-recall auto-transcript (steel-man baseline)
- **Arm B** — LLM agent on the full hypotree DAG belief state

**The moat is inferential, not mnemonic.** Arm F remembered every fact (0% duplicates) and still lost 30/0/0 because **hypotree deduced answers without spending probes**: 362 exclusion inferences, 19 deductions by elimination.


### Running the eval

```bash
# Pre-flight: confirm the engine solves every seed (no GPU)
uv run python -m eval.runner.engine_selfplay

# Full gate: 30 seeds × 3 arms
./eval.sh --run-iteration <X> --llm-model <model>
```

The eval harness lives in [`eval/`](eval/) and includes the frozen landscape generators, the agent runner, and the gate scorer. Run artifacts are gitignored (`eval/runs/`).

`eval.sh` is bash — on Windows, run it under WSL or Git Bash. The Python parts of the harness (`engine_selfplay`, `runner`, `analyse_gate`, `seed_reader`) are cross-platform and can be driven directly.

---

## Configuration

### Workspace identity

The belief-state database is isolated by workspace. Four resolution layers, highest priority first:

1. **`HYPOTREE_WORKSPACE_ID` env var** — an explicit name. Use this for global MCP configs, where the server's working directory is not your project.
2. **`hypotree.yaml`** — copy `hypotree.yaml.template` to your project root:
   ```yaml
   workspace_id: my-project-name
   ```
3. **Git remote hash** — SSH and HTTPS spellings of one remote resolve to the same id.
4. **Project path hash** — the fallback, and the weakest: it changes if the project moves or is mounted differently.

Layer 4 is where nearly every "my belief state is empty" report comes from. Run `hypotree --info`, or have the agent call `get_workspace_info`, to see which layer actually fired:

```bash
$ hypotree --info
{
  "workspace_id": "d94da5f61c664f94",
  "resolved_from": "git_remote",
  "database": "/home/you/.local/share/mcp_hypotree/d94da5f61c664f94/state.db",
  "database_exists": true,
  "warnings": []
}
```

Workspace names are lowercase `[a-z0-9._~-]`, up to 128 characters.

### Where state is stored

| Platform | Location |
|----------|----------|
| Linux / macOS | `$XDG_DATA_HOME/mcp_hypotree/<workspace_id>/` — defaults to `~/.local/share` |
| Windows | `%LOCALAPPDATA%\mcp_hypotree\<workspace_id>\` |

`XDG_DATA_HOME` overrides on every platform, Windows included — that is how you run isolated instances side by side.

> **Keep it on a local disk.** SQLite runs in WAL mode, which needs shared memory that network shares and most mapped drives do not provide. Pointing `XDG_DATA_HOME` at a UNC path or a mounted share will fail or corrupt the database. `hypotree --info` warns when it detects one.

### Windows notes

- Everything but `eval.sh` runs natively; the evaluation harness is a bash script and needs WSL or Git Bash.
- Git is optional. Without it on `PATH`, layers 3 and 4 both fall through to the path hash — pin the workspace with layer 1 or 2 instead.

---

## Development

```bash
# Install in dev mode
uv sync

# Run tests
uv run pytest tests/ -x -q

# Lint + format
uv run ruff check src/ tests/ eval/
uv run ruff format src/ tests/ eval/

# Type check
uv run mypy src/hypotree/
```

---

## License

[MIT](LICENSE) — Copyright © 2026 Damian Borowski

---

## Links

- **GitHub:** [github.com/tygryso/hypotree](https://github.com/tygryso/hypotree)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) — version history with gate results
- **Agent guide:** [`src/hypotree/AGENT_GUIDE.md`](src/hypotree/AGENT_GUIDE.md) — the full contract, also served live as the `hypotree://guide` MCP resource