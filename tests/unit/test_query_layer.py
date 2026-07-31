"""Tests for the read/query layer: list_nodes, get_evidence_history,
get_active_claims, enriched get_goal_status, evidence_count, transition
timestamps, and git-context auto-capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.engine import (
    HypoTreeEngine,
    NodeNotFoundError,
    _translate_like_wildcards,
)
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.status import Status


@pytest.fixture
def engine(tmp_path: Path) -> HypoTreeEngine:
    """Fresh engine on a temp DB with no project_path (non-git fallback)."""
    eng = HypoTreeEngine(tmp_path / "state.db", rng_seed=42, project_path=tmp_path)
    yield eng
    eng.close()


# -- wildcard translation --------------------------------------------------


@pytest.mark.unit
def test_wildcard_star_becomes_percent() -> None:
    assert _translate_like_wildcards("hello*") == "hello%"


@pytest.mark.unit
def test_wildcard_literal_percent_escaped() -> None:
    result = _translate_like_wildcards("50% done")
    assert "50\\% done" in result


@pytest.mark.unit
def test_wildcard_literal_underscore_escaped() -> None:
    result = _translate_like_wildcards("node_id")
    assert "node\\_id" in result


@pytest.mark.unit
def test_wildcard_star_and_literal_percent() -> None:
    result = _translate_like_wildcards("test*100%")
    assert "test%" in result
    assert "100\\%" in result


# -- list_nodes ------------------------------------------------------------


@pytest.mark.unit
def test_list_nodes_returns_markdown_table(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("alpha", node_id="n1")
    engine.create_hypothesis("beta", node_id="n2")
    result = engine.list_nodes()
    assert "| ID |" in result
    assert "|----|" in result
    assert "n1" in result
    assert "n2" in result


@pytest.mark.unit
def test_list_nodes_status_filter(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("verified one", node_id="n1")
    engine.create_hypothesis("untested two", node_id="n2")
    engine.update_status("n1", Status.VERIFIED, reason="manual")
    result = engine.list_nodes(status_filter=["VERIFIED"])
    assert "n1" in result
    assert "n2" not in result


@pytest.mark.unit
def test_list_nodes_query_filter(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("Thompson Sampling test", node_id="n1")
    engine.create_hypothesis("SQLite store", node_id="n2")
    result = engine.list_nodes(query_filter="thompson")
    lines = result.strip().split("\n")
    # n1 appears in the data rows; n2 should be absent
    data_lines = [ln for ln in lines if "| n1 |" in ln or "| n2 |" in ln]
    assert any("| n1 |" in ln for ln in data_lines)
    assert not any("| n2 |" in ln for ln in data_lines)


@pytest.mark.unit
def test_list_nodes_query_wildcard(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("Phase 1a models", node_id="n1")
    engine.create_hypothesis("Phase 1b store", node_id="n2")
    result = engine.list_nodes(query_filter="Phase*")
    assert "n1" in result
    assert "n2" in result


@pytest.mark.unit
def test_list_nodes_order_desc(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("first", node_id="n1")
    engine.create_hypothesis("second", node_id="n2")
    result = engine.list_nodes(order_by="created_at", ascending=False)
    lines = result.strip().split("\n")
    assert "n2" in lines[2]


@pytest.mark.unit
def test_list_nodes_limit(engine: HypoTreeEngine) -> None:
    for i in range(5):
        engine.create_hypothesis(f"node {i}", node_id=f"n{i}")
    result = engine.list_nodes(limit=2, offset=0)
    lines = result.strip().split("\n")
    assert len(lines) == 4  # 2 header + 2 data


# -- get_evidence_history --------------------------------------------------


@pytest.mark.unit
def test_evidence_history_returns_rows(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    engine.record_evidence("n1", LogicalEvidence(success=0.8))
    history = engine.get_evidence_history("n1")
    assert len(history) == 2
    assert history[0].success == 0.8
    assert history[1].success == 0.5


@pytest.mark.unit
def test_evidence_history_unknown_node(engine: HypoTreeEngine) -> None:
    with pytest.raises(NodeNotFoundError):
        engine.get_evidence_history("nonexistent")


@pytest.mark.unit
def test_evidence_history_limit(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    for i in range(5):
        engine.record_evidence("n1", LogicalEvidence(success=0.1 * i))
    history = engine.get_evidence_history("n1", limit=2)
    assert len(history) == 2


# -- get_active_claims -----------------------------------------------------


@pytest.mark.unit
def test_active_claims_empty(engine: HypoTreeEngine) -> None:
    assert engine.get_active_claims() == []


@pytest.mark.unit
def test_active_claims_returns_live(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.get_next_targets()[0]
    claims = engine.get_active_claims()
    assert len(claims) == 1
    assert claims[0].node_id == "n1"
    assert claims[0].expires_in_s > 0


# -- enriched get_goal_status ----------------------------------------------


@pytest.mark.unit
def test_goal_status_enriched(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("goal", node_id="g1", is_goal=True, target_metric=0.8)
    engine.create_hypothesis("regular", node_id="n1")
    engine.create_hypothesis("pruned", node_id="n2")
    engine.update_status("n2", Status.PRUNED, reason="test")
    resp = engine.get_goal_status()
    assert resp.goals_met_count == 0
    assert resp.goals_total_count == 1
    assert resp.total_nodes == 3
    assert resp.frontier_size >= 1
    assert "PRUNED" in resp.status_breakdown
    assert resp.status_breakdown["PRUNED"] == 1


# -- evidence_count --------------------------------------------------------


@pytest.mark.unit
def test_evidence_count_increments(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    engine.record_evidence("n1", LogicalEvidence(success=0.8))
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.evidence_count == 2


@pytest.mark.unit
def test_evidence_count_excludes_infra(engine: HypoTreeEngine) -> None:
    from hypotree.models.evidence import InfraError

    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    engine.record_evidence("n1", InfraError(error_type="OOM", message="killed"))
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.evidence_count == 1


# -- transition timestamps -------------------------------------------------


@pytest.mark.unit
def test_invalidated_at_populated(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.update_status("n1", Status.INVALIDATED, reason="test")
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.invalidated_at is not None


@pytest.mark.unit
def test_pruned_at_populated(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.update_status("n1", Status.PRUNED, reason="test")
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.pruned_at is not None


@pytest.mark.unit
def test_verified_at_populated(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("test", node_id="n1")
    engine.update_status("n1", Status.VERIFIED, reason="test")
    node = engine._store.get_node("n1")  # noqa: SLF001
    assert node is not None
    assert node.verified_at is not None


# -- git context auto-capture ----------------------------------------------


@pytest.mark.unit
def test_git_context_non_git_dir(engine: HypoTreeEngine) -> None:
    """In a non-git directory, context_hash/git_branch are None (no crash)."""
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5))
    history = engine.get_evidence_history("n1")
    assert len(history) == 1
    assert history[0].context_hash is None
    assert history[0].git_branch is None


@pytest.mark.unit
def test_git_context_explicit_preserved(engine: HypoTreeEngine) -> None:
    """If evidence already carries context_hash, it is not overwritten."""
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence(
        "n1", LogicalEvidence(success=0.5, context_hash="abc123", git_branch="feature")
    )
    history = engine.get_evidence_history("n1")
    assert len(history) == 1
    assert history[0].context_hash == "abc123"
    assert history[0].git_branch == "feature"


@pytest.mark.unit
def test_capture_git_context_in_repo(tmp_path: Path) -> None:
    """capture_git_context returns (hash, branch) inside a real git repo."""
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "README.md").write_text("# test")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)
    from hypotree.store.identity import capture_git_context

    ctx_hash, branch = capture_git_context(tmp_path)
    assert ctx_hash is not None
    assert len(ctx_hash) == 16
    assert branch in ("main", "master")


@pytest.mark.unit
def test_capture_git_context_non_git(tmp_path: Path) -> None:
    """capture_git_context returns (None, None) outside a git repo."""
    from hypotree.store.identity import capture_git_context

    ctx_hash, branch = capture_git_context(tmp_path)
    assert ctx_hash is None
    assert branch is None


# -- schema version --------------------------------------------------------


@pytest.mark.unit
def test_schema_version_stamped_on_fresh_db(tmp_path: Path) -> None:
    """Fresh DB is stamped with schema_version=2."""
    eng = HypoTreeEngine(tmp_path / "state.db", rng_seed=42)
    eng.close()
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "state.db"))
    row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    conn.close()
    from hypotree.store.schema import SCHEMA_VERSION

    assert row[0] == SCHEMA_VERSION


@pytest.mark.unit
def test_old_schema_v1_rejected(tmp_path: Path) -> None:
    """A v1 DB raises SchemaVersionError when code expects v2."""
    import sqlite3

    from hypotree.store.store import SchemaVersionError

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError, match="schema_version='1'"):
        HypoTreeEngine(db_path, rng_seed=42)


# -- LIKE escaping end-to-end (literal % / _ are not wildcards) -------------


@pytest.mark.unit
def test_list_nodes_literal_underscore_matches_literally(engine: HypoTreeEngine) -> None:
    """A literal `_` in the query must match `_`, not act as a single-char wildcard."""
    engine.create_hypothesis("node_id field", node_id="a1")
    engine.create_hypothesis("nodeXid field", node_id="a2")
    result = engine.list_nodes(query_filter="node_id")
    assert "| a1 |" in result
    assert "| a2 |" not in result


@pytest.mark.unit
def test_list_nodes_literal_percent_matches_literally(engine: HypoTreeEngine) -> None:
    """A literal `%` in the query must match `%`, not act as a multi-char wildcard."""
    engine.create_hypothesis("50% complete", node_id="a1")
    engine.create_hypothesis("halfway complete", node_id="a2")
    result = engine.list_nodes(query_filter="50%")
    assert "| a1 |" in result
    assert "| a2 |" not in result


@pytest.mark.unit
def test_list_nodes_explicit_prefix_wildcard_anchors(engine: HypoTreeEngine) -> None:
    """An explicit trailing `*` anchors to the prefix (no implicit substring wrap)."""
    engine.create_hypothesis("Phase report", node_id="a1")
    engine.create_hypothesis("report on Phase", node_id="a2")
    result = engine.list_nodes(query_filter="Phase*")
    assert "| a1 |" in result
    assert "| a2 |" not in result


# -- staleness ordering ----------------------------------------------------


@pytest.mark.unit
def test_list_nodes_staleness_surfaces_oldest_first(engine: HypoTreeEngine) -> None:
    """Default staleness ordering surfaces the most-stale (oldest-updated) node first."""
    engine.create_hypothesis("first", node_id="n1")
    engine.create_hypothesis("second", node_id="n2")
    # Touch n2 so its updated_at advances → n1 is now the most stale.
    engine.update_status("n2", Status.IN_PROGRESS, reason="touch")
    result = engine.list_nodes(order_by="staleness")
    lines = result.strip().split("\n")
    assert "| n1 |" in lines[2]
