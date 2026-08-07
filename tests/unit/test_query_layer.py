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
    (tmp_path / "README.md").write_text("# test", encoding="utf-8")
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


# -- Q1 staleness / Q6 views / Q4 source_ref -------------------------------


@pytest.mark.unit
def test_stale_is_silent_outside_a_git_checkout(engine: HypoTreeEngine) -> None:
    """A project with no commit to compare against has no drift to report.

    Inventing one would mark every confirmation permanently suspect, which is
    worse than saying nothing.
    """
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=1.0))

    assert engine.stale_node_ids() == set()
    assert "Stale" in engine.list_nodes()


@pytest.mark.unit
def test_stale_flags_a_confirmation_made_against_another_commit(
    engine: HypoTreeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hash was captured on every evidence row and never once read back."""
    import hypotree.engine as engine_mod

    engine.create_hypothesis("moved", node_id="n1")
    engine.create_hypothesis("current", node_id="n2")
    engine.record_evidence("n1", LogicalEvidence(success=1.0, context_hash="oldsha"))
    engine.record_evidence("n2", LogicalEvidence(success=1.0, context_hash="headsha"))
    engine.update_status("n1", Status.VERIFIED, reason="t")
    engine.update_status("n2", Status.VERIFIED, reason="t")
    monkeypatch.setattr(engine_mod, "capture_git_context", lambda p: ("headsha", "main"))

    assert engine.stale_node_ids() == {"n1"}
    assert "n1" in engine.list_nodes(stale_only=True)
    assert "n2" not in engine.list_nodes(stale_only=True)


@pytest.mark.unit
def test_only_verified_nodes_can_go_stale(
    engine: HypoTreeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untested hypothesis does not become false because the code moved."""
    import hypotree.engine as engine_mod

    engine.create_hypothesis("untested", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.5, context_hash="oldsha"))
    monkeypatch.setattr(engine_mod, "capture_git_context", lambda p: ("headsha", "main"))

    assert engine.stale_node_ids() == set()


@pytest.mark.unit
def test_view_preset_selects_the_frontier(engine: HypoTreeEngine) -> None:
    engine.create_hypothesis("open", node_id="n1")
    engine.create_hypothesis("done", node_id="n2")
    engine.update_status("n2", Status.VERIFIED, reason="t")

    frontier = engine.list_nodes(view="frontier")
    assert "| n1 |" in frontier
    assert "| n2 |" not in frontier


@pytest.mark.unit
def test_unknown_view_names_the_accepted_ones(engine: HypoTreeEngine) -> None:
    """A silently-empty table reads as 'nothing to do', not 'wrong question'."""
    with pytest.raises(ValueError, match="Accepted views"):
        engine.list_nodes(view="nonsense")


@pytest.mark.unit
def test_evidence_carries_the_artifact_it_came_from(engine: HypoTreeEngine) -> None:
    """'0.85' and '0.85, from pytest run #4412' are different audit trails."""
    engine.create_hypothesis("test", node_id="n1")
    engine.record_evidence("n1", LogicalEvidence(success=0.85, source_ref="ci://run/4412"))

    assert engine.get_evidence_history("n1")[0].source_ref == "ci://run/4412"


@pytest.mark.unit
def test_staleness_resolves_every_node_in_one_query(tmp_path: Path, monkeypatch) -> None:
    """One evidence query per VERIFIED node is the shape that cost 41x on dispatch.

    Asserted by counting round-trips rather than by timing, so it cannot pass
    again by accident on a fast machine.
    """
    import hypotree.engine as engine_mod

    engine = HypoTreeEngine(tmp_path / "stale.db", rng_seed=7)
    try:
        for i in range(12):
            engine.create_hypothesis(f"n{i}", node_id=f"n{i}")
            engine.record_evidence(
                f"n{i}", LogicalEvidence(success=1.0, depth=1, context_hash="old")
            )
        monkeypatch.setattr(engine_mod, "capture_git_context", lambda p: ("head", "main"))

        calls = 0
        original = engine._store.get_evidence_paginated

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(engine._store, "get_evidence_paginated", counted)
        assert len(engine.stale_node_ids()) == 12
        assert calls == 0, f"expected a single bulk query, saw {calls} per-node reads"
    finally:
        engine.close()


@pytest.mark.unit
def test_a_batch_captures_git_context_once_not_once_per_result(tmp_path: Path, monkeypatch) -> None:
    """HEAD cannot move between results that arrived in one call.

    Resolving it per result spawned two subprocesses each with a 5s timeout, so
    an eight-result batch spawned sixteen to learn one commit hash.
    """
    import hypotree.engine as engine_mod
    from hypotree.engine import EvidenceReport

    calls = 0

    def counting_capture(path):
        nonlocal calls
        calls += 1
        return "abc123", "main"

    monkeypatch.setattr(engine_mod, "capture_git_context", counting_capture)
    engine = HypoTreeEngine(tmp_path / "batch-git.db", rng_seed=7)
    try:
        for i in range(8):
            engine.create_hypothesis(f"n{i}", node_id=f"n{i}")
        engine.record_results(
            [
                EvidenceReport(node_id=f"n{i}", evidence=LogicalEvidence(success=0.5, depth=1))
                for i in range(8)
            ]
        )
        assert calls == 1, f"expected one capture for the batch, saw {calls}"

        # Lazy as well as memoised: evidence carrying its own context asks git nothing.
        calls = 0
        engine.record_results(
            [
                EvidenceReport(
                    node_id="n0",
                    evidence=LogicalEvidence(
                        success=0.5, depth=1, context_hash="x", git_branch="y"
                    ),
                )
            ]
        )
        assert calls == 0
    finally:
        engine.close()


@pytest.mark.unit
def test_recorded_artifacts_are_read_back(tmp_path: Path) -> None:
    """Written since the first release and consumed by nothing.

    An audit trail that cannot produce the log it refers to is not an audit
    trail — the same defect that was fixed for `context_hash` and left here.
    """
    engine = HypoTreeEngine(tmp_path / "artifacts.db", rng_seed=7)
    try:
        engine.create_hypothesis("n", node_id="n")
        engine.record_evidence(
            "n",
            LogicalEvidence(
                success=1.0, depth=1, artifacts=[Path("/tmp/run.log"), Path("/tmp/plot.png")]
            ),
        )
        history = engine.get_evidence_history("n")
        assert history[0].artifacts == ["/tmp/run.log", "/tmp/plot.png"]
    finally:
        engine.close()
