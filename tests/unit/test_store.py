"""Unit tests for the SQLite-WAL store: atomicity, same-txn events, history
tables, claim lifecycle, schema_version fail-fast, normalized identity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from hypotree.models.edge import Edge, EdgeType
from hypotree.models.evidence import InfraError, LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status, utcnow
from hypotree.store.identity import _normalize_remote, store_root, workspace_id
from hypotree.store.schema import SCHEMA_VERSION
from hypotree.store.store import HypoTreeStore, SchemaVersionError, _dt_to_str


@pytest.fixture
def store(tmp_path: Path) -> HypoTreeStore:
    return HypoTreeStore(tmp_path / "test.db")


@pytest.fixture
def node() -> Node:
    return Node(id="n1", statement="Test hypothesis")


# ---------------------------------------------------------------------------
# Schema setup and schema_version fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schema_version_stamped_on_creation(store: HypoTreeStore) -> None:
    row = store._conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    assert row["value"] == SCHEMA_VERSION


@pytest.mark.unit
def test_schema_version_fail_fast_on_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    s = HypoTreeStore(db)
    s.close()
    # Tamper with the schema_version
    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE schema_meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        HypoTreeStore(db)


@pytest.mark.unit
def test_all_eight_tables_exist(store: HypoTreeStore) -> None:
    tables = {
        r[0]
        for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    expected = {
        "schema_meta",
        "nodes",
        "edges",
        "evidence",
        "status_history",
        "posterior_history",
        "claims",
        "events",
    }
    assert expected <= tables


# ---------------------------------------------------------------------------
# Node CRUD + parent_ids derived from edges
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_and_get_node(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.id == "n1"
    assert loaded.statement == "Test hypothesis"
    assert loaded.status == Status.UNTESTED
    assert loaded.parent_ids == []


@pytest.mark.unit
def test_parent_ids_derived_from_edges(store: HypoTreeStore) -> None:
    parent = Node(id="p1", statement="Parent")
    child = Node(id="c1", statement="Child")
    store.add_node(parent)
    store.add_node(child)
    store.add_edge(Edge(src="p1", dst="c1", type=EdgeType.DEPENDENCY))
    loaded = store.get_node("c1")
    assert loaded is not None
    assert loaded.parent_ids == ["p1"]


@pytest.mark.unit
def test_get_all_nodes_roundtrip(store: HypoTreeStore) -> None:
    n1 = Node(id="a", statement="A")
    n2 = Node(id="b", statement="B", is_goal=True, target_metric=0.9)
    store.add_node(n1)
    store.add_node(n2)
    nodes = store.get_all_nodes()
    assert len(nodes) == 2
    by_id = {n.id: n for n in nodes}
    assert by_id["b"].is_goal is True
    assert by_id["b"].target_metric == 0.9


@pytest.mark.unit
def test_save_node_updates_cache(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    node.statement = "Updated"
    node.alpha = 5.0
    store.save_node(node)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.statement == "Updated"
    assert loaded.alpha == 5.0


# ---------------------------------------------------------------------------
# Same-transaction events
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_node_writes_event_same_txn(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    events = store.get_events()
    assert len(events) == 1
    assert events[0]["type"] == "NodeCreated"
    payload = json.loads(events[0]["payload"])
    assert payload["node"]["id"] == "n1"
    assert events[0]["txn_id"] is not None


@pytest.mark.unit
def test_add_edge_writes_event_same_txn(store: HypoTreeStore) -> None:
    n1 = Node(id="x1", statement="X1")
    n2 = Node(id="x2", statement="X2")
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(src="x1", dst="x2", type=EdgeType.REFINEMENT))
    events = store.get_events()
    edge_event = [e for e in events if e["type"] == "EdgeAdded"]
    assert len(edge_event) == 1
    payload = json.loads(edge_event[0]["payload"])
    assert payload["edge"]["type"] == "REFINEMENT"


@pytest.mark.unit
def test_change_status_writes_event_same_txn(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    now = datetime.now(timezone.utc)
    store.change_status("n1", Status.IN_PROGRESS, reason="dispatched", now=now)
    status_events = [e for e in store.get_events() if e["type"] == "StatusChanged"]
    assert len(status_events) == 1
    payload = json.loads(status_events[0]["payload"])
    assert payload["old"] == "UNTESTED"
    assert payload["new"] == "IN_PROGRESS"
    assert payload["reason"] == "dispatched"


@pytest.mark.unit
def test_atomicity_rollback_on_error(store: HypoTreeStore, node: Node) -> None:
    """If a mutation inside a transaction fails, nothing should be written."""
    store.add_node(node)
    # Simulate a failure inside the transaction
    with pytest.raises(RuntimeError), store.transaction() as conn:
        conn.execute("UPDATE nodes SET status='VERIFIED' WHERE id='n1'")
        raise RuntimeError("boom")
    # The status should remain unchanged (rolled back)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.status == Status.UNTESTED


# ---------------------------------------------------------------------------
# History tables
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_status_history_intervals(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    t0 = datetime.now(timezone.utc)
    store.change_status("n1", Status.IN_PROGRESS, reason="r1", now=t0)
    t1 = t0 + timedelta(seconds=10)
    store.change_status("n1", Status.VERIFIED, reason="r2", now=t1)

    history = store.get_status_history("n1")
    # 3 intervals: initial UNTESTED, then IN_PROGRESS, then VERIFIED
    assert len(history) == 3
    assert history[0]["status"] == "UNTESTED"
    assert history[0]["valid_to"] is not None  # closed by first transition
    assert history[1]["status"] == "IN_PROGRESS"
    assert history[1]["valid_to"] is not None  # closed by second transition
    assert history[2]["status"] == "VERIFIED"
    assert history[2]["valid_to"] is None  # currently open


@pytest.mark.unit
def test_posterior_history_snapshotted_at_transitions(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    # Update posterior without a status change — should NOT create a history row
    store.update_posterior("n1", alpha=3.0, beta=2.0)
    history_before = store.get_posterior_history("n1")
    assert len(history_before) == 1  # only the initial row

    # Now change status — should snapshot the posterior
    store.change_status("n1", Status.IN_PROGRESS, reason="test")
    history_after = store.get_posterior_history("n1")
    assert len(history_after) == 2
    # The new interval should carry the updated alpha/beta
    new_row = history_after[1]
    assert new_row["alpha"] == 3.0
    assert new_row["beta"] == 2.0


@pytest.mark.unit
def test_verified_at_set_on_verify(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    now = datetime.now(timezone.utc)
    store.change_status("n1", Status.VERIFIED, reason="passed", now=now)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.verified_at is not None


# ---------------------------------------------------------------------------
# Claim lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_claim(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    now = datetime.now(timezone.utc)
    store.create_claim("claim-1", "n1", now, ttl_s=300)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.active_claim_id == "claim-1"
    assert loaded.claimed_at is not None
    assert loaded.first_dispatched_at is not None  # set on first claim


@pytest.mark.unit
def test_consume_claim(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    now = datetime.now(timezone.utc)
    store.create_claim("claim-1", "n1", now, ttl_s=300)
    assert store.consume_claim("claim-1") is True
    # Second consume should fail (already consumed)
    assert store.consume_claim("claim-1") is False
    assert store.is_claim_consumed("claim-1") is True


@pytest.mark.unit
def test_consume_nonexistent_claim(store: HypoTreeStore) -> None:
    assert store.consume_claim("nope") is False
    assert store.is_claim_consumed("nope") is False


@pytest.mark.unit
def test_expire_stale_claims_frees_nodes(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    claimed = datetime.now(timezone.utc) - timedelta(seconds=600)
    store.create_claim("stale-claim", "n1", claimed, ttl_s=60)
    now = datetime.now(timezone.utc)
    freed = store.expire_stale_claims(now)
    assert freed == ["n1"]
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.active_claim_id is None
    assert loaded.claimed_at is None


@pytest.mark.unit
def test_active_claim_not_expired(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    claimed = datetime.now(timezone.utc)
    store.create_claim("fresh-claim", "n1", claimed, ttl_s=300)
    now = datetime.now(timezone.utc)
    freed = store.expire_stale_claims(now)
    assert freed == []
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.active_claim_id == "fresh-claim"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_append_logical_evidence(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    ev = LogicalEvidence(success=0.9, metrics={"accuracy": 0.9})
    store.append_evidence("n1", ev)
    rows = store.get_evidence_for_node("n1")
    assert len(rows) == 1
    assert rows[0]["kind"] == "logical"
    assert rows[0]["success"] == 0.9
    assert json.loads(rows[0]["metrics"]) == {"accuracy": 0.9}
    # first_evidence_at should be set
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.first_evidence_at is not None


@pytest.mark.unit
def test_append_infra_error(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    ev = InfraError(error_type="OOM", message="Out of memory")
    store.append_evidence("n1", ev)
    rows = store.get_evidence_for_node("n1")
    assert len(rows) == 1
    assert rows[0]["kind"] == "infra"
    assert rows[0]["success"] is None


@pytest.mark.unit
def test_evidence_writes_event_same_txn(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    ev = LogicalEvidence(success=0.5)
    store.append_evidence("n1", ev)
    ev_events = [e for e in store.get_events() if e["type"] == "EvidenceRecorded"]
    assert len(ev_events) == 1


# ---------------------------------------------------------------------------
# events.jsonl dump
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dump_events_jsonl(store: HypoTreeStore, node: Node, tmp_path: Path) -> None:
    store.add_node(node)
    dump_path = tmp_path / "events.jsonl"
    store.dump_events_jsonl(dump_path)
    lines = dump_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "NodeCreated"
    assert record["seq"] == 1


# ---------------------------------------------------------------------------
# Normalized git-remote identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_remote_ssh_vs_https_collide() -> None:
    ssh = _normalize_remote("git@github.com:tygryso/hypotree.git")
    https = _normalize_remote("https://github.com/tygryso/hypotree.git")
    assert ssh == https
    assert ssh == "github.com/tygryso/hypotree"


@pytest.mark.unit
def test_normalize_remote_case_insensitive() -> None:
    a = _normalize_remote("HTTPS://GitHub.COM/Foo/Bar.git")
    b = _normalize_remote("git@github.com:foo/bar.git")
    assert a == b


@pytest.mark.unit
def test_workspace_id_ssh_https_same(tmp_path: Path) -> None:
    with patch("hypotree.store.identity.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "git@github.com:tygryso/hypotree.git\n"
        ssh_id = workspace_id(tmp_path)
        mock_run.return_value.stdout = "https://github.com/tygryso/hypotree.git\n"
        https_id = workspace_id(tmp_path)
    assert ssh_id == https_id


@pytest.mark.unit
def test_workspace_id_fallback_no_remote(tmp_path: Path) -> None:
    with patch("hypotree.store.identity.subprocess.run") as mock_run:
        mock_run.return_value.stdout = ""
        wid = workspace_id(tmp_path)
    assert len(wid) == 16


@pytest.mark.unit
def test_store_root_uses_xdg_data_home(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {"XDG_DATA_HOME": str(tmp_path / "xdg")}),
        patch("hypotree.store.identity.workspace_id", return_value="abc123"),
    ):
        root = store_root(Path("/some/project"))
    assert root == tmp_path / "xdg" / "mcp_hypotree" / "abc123"
    assert root.exists()


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_node_roundtrip_all_fields(store: HypoTreeStore) -> None:
    original = Node(
        id="full",
        statement="Full spec node",
        status=Status.IN_PROGRESS,
        evidence_regime="stochastic",
        is_parametric=True,
        param_config={"lr": 0.01},
        is_goal=True,
        target_metric=0.95,
        alpha=3.5,
        beta=1.5,
        infra_retry_count=2,
    )
    store.add_node(original)
    loaded = store.get_node("full")
    assert loaded is not None
    assert loaded.id == original.id
    assert loaded.statement == original.statement
    assert loaded.status == original.status
    assert loaded.evidence_regime == "stochastic"
    assert loaded.is_parametric is True
    assert loaded.param_config == {"lr": 0.01}
    assert loaded.is_goal is True
    assert loaded.target_metric == 0.95
    assert loaded.alpha == 3.5
    assert loaded.beta == 1.5
    assert loaded.infra_retry_count == 2


# ---------------------------------------------------------------------------
# Edge cases / defensive paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_node_missing_returns_none(store: HypoTreeStore) -> None:
    assert store.get_node("ghost") is None


@pytest.mark.unit
def test_change_status_missing_node_raises(store: HypoTreeStore) -> None:
    with pytest.raises(KeyError):
        store.change_status("ghost", Status.VERIFIED)


@pytest.mark.unit
def test_update_posterior_reflected_on_node(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    store.update_posterior("n1", alpha=7.0, beta=3.0)
    loaded = store.get_node("n1")
    assert loaded is not None
    assert loaded.alpha == 7.0
    assert loaded.beta == 3.0


@pytest.mark.unit
def test_updated_at_advances_across_status_change(store: HypoTreeStore, node: Node) -> None:
    """A node rebuilt after a status change must report the advanced updated_at."""
    store.add_node(node)
    later = node.created_at + timedelta(hours=2)
    store.change_status("n1", Status.IN_PROGRESS, now=later)
    loaded = store.get_node("n1")
    assert loaded is not None
    # The store round-trip must preserve the advanced timestamp (not reset to created_at).
    assert loaded.updated_at == later
    assert loaded.updated_at > loaded.created_at


@pytest.mark.unit
def test_get_claims_for_node(store: HypoTreeStore, node: Node) -> None:
    store.add_node(node)
    now = datetime.now(timezone.utc)
    store.create_claim("claim-1", "n1", now, ttl_s=300)
    claims = store.get_claims_for_node("n1")
    assert len(claims) == 1
    assert claims[0]["claim_id"] == "claim-1"


@pytest.mark.unit
def test_workspace_id_fallback_on_subprocess_error(tmp_path: Path) -> None:
    """If git invocation raises, fall back to an absolute-path hash (16 hex)."""
    with patch("hypotree.store.identity.subprocess.run", side_effect=OSError("git missing")):
        wid = workspace_id(tmp_path)
    assert len(wid) == 16


@pytest.mark.unit
def test_exclusion_group_round_trips(store: HypoTreeStore) -> None:
    """exclusion_group must survive the store, or the inference cannot rehydrate."""
    store.add_node(Node(id="a", statement="a", exclusion_group="axis"))
    store.add_node(Node(id="b", statement="b", exclusion_group="axis"))
    store.add_node(Node(id="c", statement="c"))

    assert store.get_node("a").exclusion_group == "axis"
    assert store.get_node("c").exclusion_group is None

    members = {n.id for n in store.get_nodes_in_exclusion_group("axis", exclude_id="a")}
    assert members == {"b"}
    # An ungrouped node is never swept into someone else's exclusion set.
    assert store.get_nodes_in_exclusion_group("other") == []


# -- schema migration ------------------------------------------------------


def _write_v8_database(db_path: Path) -> None:
    """A genuine v8 database: the current DDL minus the column v9 added.

    Built from the real DDL rather than a hand-written fixture so it cannot
    drift away from what v0.3.1 actually shipped, and stamped v8 so the store
    takes the migration path rather than the fresh-stamp path.
    """
    from hypotree.store.schema import SCHEMA_DDL

    v8_ddl = SCHEMA_DDL.replace("    source_ref    TEXT,\n", "")
    assert "source_ref" not in v8_ddl, "the v8 fixture must not carry the v9 column"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(v8_ddl)
    conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '8')")
    conn.commit()
    conn.close()


@pytest.mark.unit
def test_a_v8_database_is_migrated_rather_than_rejected(tmp_path: Path) -> None:
    """The belief state is the product; an upgrade may not require discarding it."""
    db = tmp_path / "old.db"
    _write_v8_database(db)

    store = HypoTreeStore(db)
    try:
        version = store._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        columns = {r["name"] for r in store._conn.execute("PRAGMA table_info(evidence)")}
    finally:
        store.close()

    assert version == SCHEMA_VERSION
    assert "source_ref" in columns


@pytest.mark.unit
def test_migration_preserves_the_rows_that_were_already_there(tmp_path: Path) -> None:
    """A migration that loses a recorded experiment is worse than no migration."""
    db = tmp_path / "old.db"
    _write_v8_database(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO nodes (id, statement, status, alpha, beta, is_goal, evidence_regime,"
        " is_parametric, evidence_count, created_at, updated_at)"
        " VALUES ('n1','survives me','VERIFIED',3.0,1.0,0,'deterministic',0,1,?,?)",
        (_dt_to_str(utcnow()), _dt_to_str(utcnow())),
    )
    conn.commit()
    conn.close()

    store = HypoTreeStore(db)
    try:
        node = store.get_node("n1")
    finally:
        store.close()

    assert node is not None
    assert node.statement == "survives me"
    assert node.status is Status.VERIFIED


@pytest.mark.unit
def test_migration_is_recorded_in_the_audit_log(tmp_path: Path) -> None:
    """'Why does this database differ from my backup?' is an audit question."""
    db = tmp_path / "old.db"
    _write_v8_database(db)

    store = HypoTreeStore(db)
    try:
        rows = store._conn.execute("SELECT type, payload FROM events").fetchall()
    finally:
        store.close()

    migrated = [r for r in rows if r["type"] == "SchemaMigrated"]
    assert len(migrated) == 1
    assert json.loads(migrated[0]["payload"]) == {"from": "8", "to": SCHEMA_VERSION}


@pytest.mark.unit
def test_a_database_from_a_newer_hypotree_is_refused(tmp_path: Path) -> None:
    """Downgrading could silently drop data this version cannot represent."""
    db = tmp_path / "future.db"
    HypoTreeStore(db).close()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE schema_meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError, match="newer hypotree"):
        HypoTreeStore(db)


@pytest.mark.unit
def test_an_unroutable_version_says_keep_the_file(tmp_path: Path) -> None:
    """The one instruction that must never appear again is 'delete the DB'."""
    db = tmp_path / "odd.db"
    HypoTreeStore(db).close()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE schema_meta SET value='3' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError, match="do not delete it"):
        HypoTreeStore(db)


@pytest.mark.unit
def test_a_failed_migration_leaves_the_old_version_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted upgrade must roll back, not strand the DB between versions."""
    import hypotree.store.store as store_mod

    db = tmp_path / "old.db"
    _write_v8_database(db)
    monkeypatch.setattr(
        store_mod, "MIGRATIONS", {"8": (SCHEMA_VERSION, ("ALTER TABLE nope ADD COLUMN x",))}
    )

    with pytest.raises(sqlite3.OperationalError):
        HypoTreeStore(db)

    conn = sqlite3.connect(str(db))
    stamped = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
    conn.close()
    assert stamped == "8"
