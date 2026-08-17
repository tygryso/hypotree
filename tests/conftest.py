"""Shared fixtures.

Kept deliberately thin: the suite's isolation comes from `tmp_path` databases,
not from shared state, and anything global enough to live here is usually a
smell. The one exception is genuine process-level memoisation, which by
definition outlives the test that populated it.
"""

from __future__ import annotations

import pytest

from hypotree.store.identity import reset_identity_cache


@pytest.fixture(autouse=True)
def _isolate_identity_cache() -> None:
    """Stop one test's git-remote resolution answering another's question.

    `_resolve_remote` is memoised per process so the MCP server does not pay
    three git subprocesses on every start. That cache is keyed by path, and
    `tmp_path` is unique per test, so collisions are unlikely rather than
    impossible — a test that resolves a path *before* creating a remote under it
    would otherwise poison every later lookup of the same path with `None`.
    """
    reset_identity_cache()
