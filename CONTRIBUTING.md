# Contributing to hypotree

Contributions welcome — bug reports, feature ideas, eval improvements, docs fixes. This is a guide to get you productive fast.

## Setup

```bash
git clone https://github.com/tygryso/hypotree.git
cd hypotree
uv sync          # creates .venv, installs all deps + dev deps
```

**Requires:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

## Running tests

```bash
uv run pytest tests/ -x -q              # full suite
uv run pytest tests/unit/ -x -q         # unit tests only
uv run pytest tests/unit/test_engine.py # one file
```

Tests use `tmp_path` fixtures — every SQLite database is isolated. Never mutate the global workspace DB.

## Linting & type checking

```bash
uv run ruff check src/ tests/ eval/     # lint
uv run ruff format src/ tests/ eval/    # format
uv run mypy src/hypotree/               # type check
```

All three must pass before a PR is merged.

## Adding a new MCP tool

1. **Define the tool** in `src/hypotree/mcp_server.py` — add the `@mcp.tool()` decorator with a typed signature and docstring (the docstring is the agent's instruction manual).
2. **Implement the engine logic** in `src/hypotree/engine.py` — any state mutation must write an event log within the same SQLite transaction.
3. **Write tests** in `tests/unit/` or `tests/e2e/` — every new function needs coverage.
4. **Update `AGENT_GUIDE.md`** (`src/hypotree/AGENT_GUIDE.md`) if the tool changes how agents should interact.

## Adding a new eval seed

Landscape generators live in `eval/environment/landscapes/`. Each seed is a frozen JSON with the search space, planted decoy, and reference strategy cost.

## Commit style

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).
- Keep commits small and atomic.
- Include the issue number in the PR title.

## Schema changes

The project is entering production — **schema changes require migrations**. SQLite schema versions must be bumped and old databases upgraded automatically. Never break existing belief-state databases.

## Reporting bugs

Open an issue with:
- The `uvx hypotree` version (or git commit)
- The MCP client you're using (Cursor, Cline, Claude Desktop)
- The workspace ID (if relevant)
- A minimal reproduction

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).