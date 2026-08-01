"""SQLite-WAL source-of-truth store for the hypothesis DAG.

Every state mutation writes its event row in the same transaction — state
cache, history rows, and event commit atomically or not at all. The events
table is for audit/replay; events.jsonl is a dump, never a live projection.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from hypotree.models.edge import Edge, EdgeType
from hypotree.models.evidence import Evidence, InfraError, LogicalEvidence
from hypotree.models.node import Node
from hypotree.models.status import Status, utcnow
from hypotree.store.schema import SCHEMA_DDL, SCHEMA_VERSION


class SchemaVersionError(RuntimeError):
    """Raised when the DB schema version does not match the code's expectation."""


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _evidence_to_row(node_id: str, ev: Evidence) -> dict:
    """Flatten a Pydantic Evidence union into columns for the evidence table."""
    if isinstance(ev, LogicalEvidence):
        return {
            "node_id": node_id,
            "kind": "logical",
            "success": ev.success,
            "depth": ev.depth,
            "metrics": json.dumps(ev.metrics),
            "artifacts": json.dumps([str(p) for p in ev.artifacts]),
            "context_hash": ev.context_hash,
            "git_branch": ev.git_branch,
            "claim_id": ev.claim_id,
            "notes": ev.notes,
            "delta_success": ev.delta_success,
            "delta_metrics": json.dumps(ev.delta_metrics),
            "monotonicity": ev.monotonicity,
        }
    return {
        "node_id": node_id,
        "kind": "infra",
        "success": None,
        "depth": 0,
        "metrics": "{}",
        "artifacts": "[]",
        "context_hash": None,
        "git_branch": None,
        "claim_id": ev.claim_id,
        "notes": f"{ev.error_type}: {ev.message}",
        "delta_success": None,
        "delta_metrics": "{}",
        "monotonicity": "first",
    }


class HypoTreeStore:
    """Persistent store backed by SQLite in WAL mode."""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._check_schema_version()

    def close(self) -> None:
        self._conn.close()

    # -- transaction management ------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit BEGIN/COMMIT/ROLLBACK — single-writer, serialized."""
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _txn_id() -> str:
        return uuid.uuid4().hex

    def _write_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        payload: str,
        txn_id: str,
    ) -> None:
        conn.execute(
            "INSERT INTO events (type, payload, txn_id, written_at) VALUES (?,?,?,?)",
            (event_type, payload, txn_id, _dt_to_str(utcnow())),
        )

    # -- schema setup ----------------------------------------------------------

    def _init_schema(self) -> None:
        # executescript issues an implicit COMMIT before running, so it must
        # run outside the explicit transaction() context manager.
        self._conn.executescript(SCHEMA_DDL)

    def _check_schema_version(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            # Fresh DB — stamp it
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
        elif row["value"] != SCHEMA_VERSION:
            db_ver = row["value"]
            raise SchemaVersionError(
                f"DB schema_version='{db_ver}' does not match "
                f"code expected='{SCHEMA_VERSION}'. "
                "Under active development — no migrations. Delete the DB to reset."
            )

    # -- node CRUD -------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Insert a node + initial history rows + NodeCreated event (one txn)."""
        txn = self._txn_id()
        with self.transaction() as conn:
            self._insert_node_row(conn, node)
            # Initial status_history interval (open-ended)
            conn.execute(
                "INSERT INTO status_history (node_id, status, valid_from, valid_to, reason) "
                "VALUES (?,?,?,?,?)",
                (node.id, node.status.value, _dt_to_str(node.created_at), None, "created"),
            )
            # Initial posterior_history interval (open-ended)
            conn.execute(
                "INSERT INTO posterior_history "
                "(node_id, alpha, beta, valid_from, valid_to, reason) "
                "VALUES (?,?,?,?,?,?)",
                (
                    node.id,
                    node.alpha,
                    node.beta,
                    _dt_to_str(node.created_at),
                    None,
                    "created",
                ),
            )
            self._write_event(
                conn,
                "NodeCreated",
                json.dumps({"node": node.model_dump(mode="json")}),
                txn,
            )

    @staticmethod
    def _insert_node_row(conn: sqlite3.Connection, node: Node) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO nodes (
                id, statement, status, evidence_regime, is_parametric, param_config,
                is_goal, target_metric, exclusion_group, confirmed_depth,
                alpha, beta, evidence_count,
                active_claim_id, claimed_at, infra_retry_count,
                created_at, first_dispatched_at, first_evidence_at,
                verified_at, invalidated_at, pruned_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.id,
                node.statement,
                node.status.value,
                node.evidence_regime,
                int(node.is_parametric),
                json.dumps(node.param_config) if node.param_config is not None else None,
                int(node.is_goal),
                node.target_metric,
                node.exclusion_group,
                node.confirmed_depth,
                node.alpha,
                node.beta,
                node.evidence_count,
                node.active_claim_id,
                _dt_to_str(node.claimed_at),
                node.infra_retry_count,
                _dt_to_str(node.created_at),
                _dt_to_str(node.first_dispatched_at),
                _dt_to_str(node.first_evidence_at),
                _dt_to_str(node.verified_at),
                _dt_to_str(node.invalidated_at),
                _dt_to_str(node.pruned_at),
                _dt_to_str(node.updated_at),
            ),
        )

    def save_node(self, node: Node) -> None:
        """Update the node cache row (no history/event — pure cache refresh)."""
        with self.transaction() as conn:
            self._insert_node_row(conn, node)

    def delete_node(self, node_id: str) -> None:
        """Discard a node's own data for an overwrite-replace (one transaction).

        Drops the node's evidence, status/posterior history, claims, incoming
        (parent) edges, any conflict set it appears in, and the node row, writing
        a NodeDeleted event. Outgoing edges to children are **intentionally
        preserved** so overwriting a node keeps its place in the DAG (children
        stay attached to the reused id); the new create call redefines the parent
        edges. Used by create_hypothesis(if_exists='overwrite') so the recreated
        node has clean single-open history intervals and no duplicate creation
        event.

        Conflicts are dropped rather than rewritten: a conflict asserts that a
        specific set of hypotheses cannot all hold, and once one of them has been
        redefined that assertion is about a node that no longer exists. Keeping it
        would leave a suspect who can never be exonerated, so the conflict could
        never narrow to a culprit again.
        """
        stale_nogoods = [
            row["id"]
            for row in self._conn.execute(
                "SELECT id, source_node_id, member_ids FROM nogoods"
            ).fetchall()
            if row["source_node_id"] == node_id or node_id in json.loads(row["member_ids"])
        ]
        txn = self._txn_id()
        with self.transaction() as conn:
            conn.execute("DELETE FROM edges WHERE dst=?", (node_id,))
            conn.execute("DELETE FROM evidence WHERE node_id=?", (node_id,))
            conn.execute("DELETE FROM status_history WHERE node_id=?", (node_id,))
            conn.execute("DELETE FROM posterior_history WHERE node_id=?", (node_id,))
            conn.execute("DELETE FROM claims WHERE node_id=?", (node_id,))
            for nogood_id in stale_nogoods:
                conn.execute("DELETE FROM nogoods WHERE id=?", (nogood_id,))
            conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            self._write_event(conn, "NodeDeleted", json.dumps({"node_id": node_id}), txn)

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_all_nodes(self) -> list[Node]:
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_nodes_in_exclusion_group(self, group: str, exclude_id: str | None = None) -> list[Node]:
        """Every node sharing a mutual-exclusion group, optionally excluding one.

        Used to propagate a confirmation into its competing alternatives (and to
        retract that propagation when the confirmation is withdrawn).
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE exclusion_group = ? AND id != ?",
            (group, exclude_id or ""),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        parent_ids = self.get_parent_ids(row["id"])
        # created_at and updated_at are NOT NULL in the schema
        created_at = _str_to_dt(row["created_at"])
        updated_at = _str_to_dt(row["updated_at"])
        assert created_at is not None and updated_at is not None

        return Node(
            id=row["id"],
            statement=row["statement"],
            status=Status(row["status"]),
            parent_ids=parent_ids,
            evidence_regime=row["evidence_regime"],
            is_parametric=bool(row["is_parametric"]),
            param_config=json.loads(row["param_config"]) if row["param_config"] else None,
            is_goal=bool(row["is_goal"]),
            target_metric=row["target_metric"],
            exclusion_group=row["exclusion_group"],
            confirmed_depth=row["confirmed_depth"],
            alpha=row["alpha"],
            beta=row["beta"],
            evidence_count=row["evidence_count"],
            active_claim_id=row["active_claim_id"],
            claimed_at=_str_to_dt(row["claimed_at"]),
            infra_retry_count=row["infra_retry_count"],
            created_at=created_at,
            first_dispatched_at=_str_to_dt(row["first_dispatched_at"]),
            first_evidence_at=_str_to_dt(row["first_evidence_at"]),
            verified_at=_str_to_dt(row["verified_at"]),
            invalidated_at=_str_to_dt(row["invalidated_at"]),
            pruned_at=_str_to_dt(row["pruned_at"]),
            updated_at=updated_at,
        )

    # -- edge CRUD -------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        txn = self._txn_id()
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO edges (src, dst, type) VALUES (?,?,?)",
                (edge.src, edge.dst, edge.type.value),
            )
            self._write_event(
                conn,
                "EdgeAdded",
                json.dumps({"edge": edge.model_dump(mode="json")}),
                txn,
            )

    def get_all_edges(self) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges").fetchall()
        return [Edge(src=r["src"], dst=r["dst"], type=EdgeType(r["type"])) for r in rows]

    def get_parent_ids(self, node_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT src FROM edges WHERE dst=?", (node_id,)
        ).fetchall()
        return [r["src"] for r in rows]

    # -- status transitions ----------------------------------------------------

    def change_status(
        self, node_id: str, new: Status, reason: str = "", now: datetime | None = None
    ) -> None:
        """Transition a node's status: update cache, close/open history, write event.

        Closes the currently-open status_history and posterior_history intervals,
        opens new ones, updates the nodes cache, and writes a StatusChanged event —
        all in one transaction.
        """
        if now is None:
            now = utcnow()
        now_str = _dt_to_str(now)
        txn = self._txn_id()

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status, alpha, beta FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Node not found: {node_id}")
            old_status = Status(row["status"])

            # Re-asserting the status a node already holds is not a transition.
            # Writing it as one would open a second history interval at the same
            # instant as the one it closes, which violates the (node_id,
            # valid_from) key and, when timestamps do differ, litters the record
            # with zero-length intervals that mean nothing. Re-confirming an
            # existing belief is a normal action — it is how a shallow
            # confirmation is deepened — so it must not corrupt the history.
            if new == old_status:
                conn.execute("UPDATE nodes SET updated_at=? WHERE id=?", (now_str, node_id))
                return

            # Close the currently-open status_history interval
            conn.execute(
                "UPDATE status_history SET valid_to=? WHERE node_id=? AND valid_to IS NULL",
                (now_str, node_id),
            )
            # Open new status_history interval.
            #
            # OR REPLACE because one logical action can legitimately move a node
            # twice at the same instant — a conflict puts an assumption under
            # review and the very same evidence then clears it again. The
            # intermediate state existed for zero time, so the interval that
            # should survive is the last one written at that instant; keeping
            # both is impossible under the (node_id, valid_from) key and
            # recording a zero-length interval would be a lie either way.
            conn.execute(
                "INSERT OR REPLACE INTO status_history "
                "(node_id, status, valid_from, valid_to, reason) "
                "VALUES (?,?,?,?,?)",
                (node_id, new.value, now_str, None, reason),
            )

            # Snapshot posterior: close old interval, open new one
            conn.execute(
                "UPDATE posterior_history SET valid_to=? WHERE node_id=? AND valid_to IS NULL",
                (now_str, node_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO posterior_history "
                "(node_id, alpha, beta, valid_from, valid_to, reason) "
                "VALUES (?,?,?,?,?,?)",
                (node_id, row["alpha"], row["beta"], now_str, None, reason),
            )

            # Update the nodes cache
            update_fields = ["status=?", "updated_at=?"]
            params: list = [new.value, now_str]
            if new == Status.VERIFIED:
                update_fields.append("verified_at=?")
                params.append(now_str)
            elif new == Status.INVALIDATED:
                update_fields.append("invalidated_at=?")
                params.append(now_str)
            elif new == Status.PRUNED:
                update_fields.append("pruned_at=?")
                params.append(now_str)
            params.append(node_id)
            conn.execute(
                f"UPDATE nodes SET {', '.join(update_fields)} WHERE id=?",
                params,
            )

            self._write_event(
                conn,
                "StatusChanged",
                json.dumps(
                    {
                        "node_id": node_id,
                        "old": old_status.value,
                        "new": new.value,
                        "reason": reason,
                        "valid_from": now_str,
                    }
                ),
                txn,
            )

    # -- posterior cache -------------------------------------------------------

    def update_posterior(self, node_id: str, alpha: float, beta: float) -> None:
        """Update the posterior cache columns on the node (no history/event)."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET alpha=?, beta=?, updated_at=? WHERE id=?",
                (alpha, beta, _dt_to_str(utcnow()), node_id),
            )

    def set_confirmed_depth(self, node_id: str, depth: int | None) -> None:
        """Record the depth of the observation that confirmed this node.

        Kept separate from the posterior because it answers a different question:
        the posterior says *how likely* the hypothesis is, this says *how
        demanding the test was that produced that belief*. A composition failing
        at a greater depth is evidence against exactly the assumptions whose
        confirmation was shallower than the failure.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET confirmed_depth=?, updated_at=? WHERE id=?",
                (depth, _dt_to_str(utcnow()), node_id),
            )

    # -- evidence --------------------------------------------------------------

    def append_evidence(
        self, node_id: str, ev: Evidence, recorded_at: datetime | None = None
    ) -> None:
        """Insert an evidence row + EvidenceRecorded event (one txn)."""
        if recorded_at is None:
            recorded_at = utcnow()
        txn = self._txn_id()
        row_data = _evidence_to_row(node_id, ev)
        row_data["recorded_at"] = _dt_to_str(recorded_at)

        with self.transaction() as conn:
            columns = ", ".join(row_data.keys())
            placeholders = ", ".join("?" * len(row_data))
            conn.execute(
                f"INSERT INTO evidence ({columns}) VALUES ({placeholders})",
                tuple(row_data.values()),
            )
            # Set first_evidence_at if this is the first evidence for the node
            conn.execute(
                "UPDATE nodes SET first_evidence_at = COALESCE(first_evidence_at, ?), "
                "updated_at = ? WHERE id = ?",
                (_dt_to_str(recorded_at), _dt_to_str(recorded_at), node_id),
            )
            self._write_event(
                conn,
                "EvidenceRecorded",
                json.dumps(
                    {
                        "node_id": node_id,
                        "evidence": (
                            ev.model_dump(mode="json")
                            if isinstance(ev, LogicalEvidence | InfraError)
                            else None
                        ),
                    }
                ),
                txn,
            )

    def get_evidence_for_node(self, node_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM evidence WHERE node_id=? ORDER BY id", (node_id,)
        ).fetchall()

    def get_evidence_paginated(
        self, node_id: str, limit: int = 20, offset: int = 0
    ) -> list[sqlite3.Row]:
        """Return evidence rows newest-first with pagination."""
        return self._conn.execute(
            "SELECT * FROM evidence WHERE node_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
            (node_id, limit, offset),
        ).fetchall()

    # -- claims ----------------------------------------------------------------

    def create_claim(
        self,
        claim_id: str,
        node_id: str,
        claimed_at: datetime,
        ttl_s: int,
    ) -> None:
        """Insert a claim row + update node cache + TargetClaimed event (one txn).

        Any claim this node already holds is expired first. A lease is a
        statement that one dispatch owns this node right now; two live leases on
        the same node mean two callers can each record evidence against it and
        neither knows about the other, which is precisely the double-dispatch the
        lease exists to prevent.
        """
        txn = self._txn_id()
        claimed_str = _dt_to_str(claimed_at)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE claims SET expired=1 WHERE node_id=? AND consumed_at IS NULL AND expired=0",
                (node_id,),
            )
            conn.execute(
                "INSERT INTO claims (claim_id, node_id, claimed_at, lease_ttl_s) VALUES (?,?,?,?)",
                (claim_id, node_id, claimed_str, ttl_s),
            )
            conn.execute(
                "UPDATE nodes SET active_claim_id=?, claimed_at=?, "
                "first_dispatched_at=COALESCE(first_dispatched_at, ?), updated_at=? "
                "WHERE id=?",
                (claim_id, claimed_str, claimed_str, claimed_str, node_id),
            )
            self._write_event(
                conn,
                "TargetClaimed",
                json.dumps(
                    {
                        "node_id": node_id,
                        "claim_id": claim_id,
                        "claimed_at": claimed_str,
                    }
                ),
                txn,
            )

    def consume_claim(self, claim_id: str, consumed_at: datetime | None = None) -> bool:
        """Spend a claim. Returns False if it was already spent or is no longer live.

        A claim is a one-shot capability to report on one dispatch. Expiry —
        whether by TTL, by being superseded, or by an explicit release — revokes
        it, so an expired claim must be as unspendable as a consumed one.
        Checking only ``consumed_at`` let a caller report against a lease that had
        already been handed to someone else.

        Also clears the node's ``active_claim_id``. That column names the lease a
        node is currently under, and a consumed lease is no longer current —
        leaving it set made the column unusable as the answer to "is this node
        already dispatched?", which is the one question it exists to answer.
        """
        if consumed_at is None:
            consumed_at = utcnow()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT node_id, consumed_at, expired FROM claims WHERE claim_id=?",
                (claim_id,),
            ).fetchone()
            if row is None:
                return False
            if row["consumed_at"] is not None or row["expired"]:
                return False
            conn.execute(
                "UPDATE claims SET consumed_at=? WHERE claim_id=?",
                (_dt_to_str(consumed_at), claim_id),
            )
            conn.execute(
                "UPDATE nodes SET active_claim_id=NULL, claimed_at=NULL "
                "WHERE id=? AND active_claim_id=?",
                (row["node_id"], claim_id),
            )
            return True

    def is_claim_consumed(self, claim_id: str) -> bool:
        row = self._conn.execute(
            "SELECT consumed_at FROM claims WHERE claim_id=?", (claim_id,)
        ).fetchone()
        if row is None:
            return False
        return row["consumed_at"] is not None

    def get_claim(self, claim_id: str) -> sqlite3.Row | None:
        """Fetch a single claim row (or None if it does not exist)."""
        return self._conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()

    def expire_stale_claims(self, now: datetime) -> list[str]:
        """Mark claims past their TTL as expired; return freed node_ids.

        The comparison is inclusive so a zero-second lease expires at once,
        which is what a caller asking for one means.
        """
        now_str = _dt_to_str(now)
        freed: list[str] = []
        with self.transaction() as conn:
            stale = conn.execute(
                "SELECT claim_id, node_id FROM claims "
                "WHERE consumed_at IS NULL AND expired=0 "
                "AND datetime(claimed_at, '+' || lease_ttl_s || ' seconds') <= datetime(?)",
                (now_str,),
            ).fetchall()
            for r in stale:
                conn.execute("UPDATE claims SET expired=1 WHERE claim_id=?", (r["claim_id"],))
                conn.execute(
                    "UPDATE nodes SET active_claim_id=NULL, claimed_at=NULL WHERE id=?",
                    (r["node_id"],),
                )
                freed.append(r["node_id"])
        return freed

    # -- events ----------------------------------------------------------------

    def get_events(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM events ORDER BY seq").fetchall()

    def dump_events_jsonl(self, path: Path | str) -> None:
        """Dump the events table to a JSONL file (human/DR copy, not a live source)."""
        rows = self.get_events()
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                record = {
                    "seq": r["seq"],
                    "type": r["type"],
                    "payload": json.loads(r["payload"]),
                    "txn_id": r["txn_id"],
                    "written_at": r["written_at"],
                }
                f.write(json.dumps(record) + "\n")

    # -- conflict sets (nogoods) -------------------------------------------

    def add_nogood(
        self,
        source_node_id: str,
        member_ids: list[str],
        now: datetime,
        conflict_depth: int = 0,
    ) -> int:
        """Record that ``member_ids`` cannot all hold at ``conflict_depth``.

        The depth is part of the claim, not decoration: the members were each
        confirmed by some test, and what the failure shows is that those tests
        were not jointly sufficient *at this depth*. Returns the conflict id.

        The given order is preserved verbatim, because it is the order in which
        diagnosis will interrogate the members and ``probe_index`` counts into
        it. Re-sorting here would silently override the caller's ranking.
        """
        txn = self._txn_id()
        payload = json.dumps(list(member_ids))
        with self.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO nogoods (source_node_id, member_ids, conflict_depth, recorded_at)
                   VALUES (?,?,?,?)""",
                (source_node_id, payload, conflict_depth, _dt_to_str(now)),
            )
            nogood_id = int(cur.lastrowid or 0)
            self._write_event(
                conn,
                "ConflictRecorded",
                json.dumps(
                    {
                        "source_node_id": source_node_id,
                        "member_ids": list(member_ids),
                        "conflict_depth": conflict_depth,
                    }
                ),
                txn,
            )
        return nogood_id

    def get_nogoods(self, open_only: bool = False) -> list[dict[str, Any]]:
        """Return recorded conflict sets, newest first.

        ``open_only`` restricts to conflicts whose culprit has not been pinned
        down yet — the ones a discriminating experiment could still resolve.
        """
        sql = "SELECT * FROM nogoods"
        if open_only:
            sql += " WHERE resolved_at IS NULL"
        sql += " ORDER BY id DESC"
        return [
            {
                "id": row["id"],
                "source_node_id": row["source_node_id"],
                "member_ids": json.loads(row["member_ids"]),
                "conflict_depth": row["conflict_depth"],
                "probe_index": row["probe_index"],
                "resolved_culprit_id": row["resolved_culprit_id"],
                "reopened_at": row["reopened_at"],
                "recorded_at": row["recorded_at"],
                "resolved_at": row["resolved_at"],
            }
            for row in self._conn.execute(sql).fetchall()
        ]

    def advance_nogood_probe(self, nogood_id: int, probe_index: int, now: datetime) -> None:
        """Record that one more member has been cleared by substitution.

        Swapping a member out and watching the composition fail anyway rules that
        member out as the sole cause. The cursor is stored rather than recomputed
        so the diagnosis survives a context reset: without it every reset would
        restart the elimination and re-run experiments already paid for.
        """
        txn = self._txn_id()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nogoods SET probe_index=? WHERE id=?",
                (probe_index, nogood_id),
            )
            self._write_event(
                conn,
                "ConflictMemberCleared",
                json.dumps({"nogood_id": nogood_id, "probe_index": probe_index}),
                txn,
            )

    def mark_nogood_reopened(self, nogood_id: int, now: datetime) -> None:
        """Record that a conflict's alternatives have been reopened.

        The conflict stays *open* — no culprit was identified — but the recovery
        it triggers must not repeat, or every subsequent observation would
        re-reopen the same questions and undo answers found since.
        """
        txn = self._txn_id()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nogoods SET reopened_at=? WHERE id=?",
                (_dt_to_str(now), nogood_id),
            )
            self._write_event(
                conn,
                "ConflictReopened",
                json.dumps({"nogood_id": nogood_id}),
                txn,
            )

    def resolve_nogood(self, nogood_id: int, culprit_id: str, now: datetime) -> None:
        """Pin a conflict on a single culprit once every other member is cleared."""
        txn = self._txn_id()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nogoods SET resolved_culprit_id=?, resolved_at=? WHERE id=?",
                (culprit_id, _dt_to_str(now), nogood_id),
            )
            self._write_event(
                conn,
                "ConflictResolved",
                json.dumps({"nogood_id": nogood_id, "culprit_id": culprit_id}),
                txn,
            )

    # -- history queries -------------------------------------------------------

    def get_status_history(self, node_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM status_history WHERE node_id=? ORDER BY valid_from",
            (node_id,),
        ).fetchall()

    def get_all_status_history(self) -> list[sqlite3.Row]:
        """Every status transition in the workspace, oldest first.

        The per-node accessor answers "what happened to this hypothesis"; this
        one answers "what happened, in order" — which is the only way to tell a
        conclusion that was observed from one the engine inferred afterwards.
        """
        return self._conn.execute(
            "SELECT * FROM status_history ORDER BY valid_from, node_id"
        ).fetchall()

    def get_posterior_history(self, node_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM posterior_history WHERE node_id=? ORDER BY valid_from",
            (node_id,),
        ).fetchall()

    def get_claims_for_node(self, node_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM claims WHERE node_id=? ORDER BY claimed_at", (node_id,)
        ).fetchall()

    # -- read/query layer ------------------------------------------------------

    # Columns that support ordering in list_nodes. Maps the public name to the
    # SQL column expression. "staleness" and "posterior_mean" are derived.
    _ORDER_COLUMNS: dict[str, str] = {
        "created_at": "created_at",
        "updated_at": "updated_at",
        "verified_at": "verified_at",
        "pruned_at": "pruned_at",
        "invalidated_at": "invalidated_at",
        "posterior_mean": "alpha / (alpha + beta)",
        "evidence_count": "evidence_count",
        # staleness sorts on updated_at but its direction is inverted below
        # (oldest update = most stale), so the default surfaces stale nodes first.
        "staleness": "updated_at",
    }

    def list_nodes(
        self,
        status_filter: list[str] | None = None,
        query_filter: str | None = None,
        order_by: str = "created_at",
        ascending: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        """Query nodes with optional status/text filter, ordering, and pagination.

        query_filter uses SQL LIKE semantics: the caller translates `*`→`%` and
        escapes literal `%`/`_` before calling this method.
        """
        clauses: list[str] = []
        params: list = []

        if status_filter:
            placeholders = ", ".join("?" * len(status_filter))
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_filter)

        if query_filter:
            # ESCAPE '\' makes the caller's backslash-escaping of literal % and _
            # take effect — without it SQLite treats every _ and % in the pattern
            # as a wildcard, so a search for literal "node_id" matches nothing.
            clauses.append("LOWER(statement) LIKE LOWER(?) ESCAPE '\\'")
            params.append(query_filter)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        sort_col = self._ORDER_COLUMNS.get(order_by, "created_at")
        direction = "ASC" if ascending else "DESC"
        # staleness is the inverse of updated_at (oldest = most stale): flip the
        # direction so the default (descending staleness) surfaces stale nodes first.
        if order_by == "staleness":
            direction = "DESC" if ascending else "ASC"
        sql = f"SELECT * FROM nodes{where} ORDER BY {sort_col} {direction} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self._conn.execute(sql, params).fetchall()

    def count_nodes_by_status(self) -> dict[str, int]:
        """Return a {status_value: count} mapping for all statuses."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM nodes GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def count_all_nodes(self) -> int:
        """Total node count (all statuses)."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM nodes").fetchone()
        return row["cnt"]

    def increment_evidence_count(self, node_id: str) -> None:
        """Bump the logical-evidence counter on a node (one txn)."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET evidence_count = evidence_count + 1 WHERE id=?",
                (node_id,),
            )

    def get_active_claims(self, now: datetime) -> list[sqlite3.Row]:
        """Return live (unconsumed, unexpired) claims as of *now*."""
        now_str = _dt_to_str(now)
        return self._conn.execute(
            "SELECT c.claim_id, c.node_id, c.claimed_at, c.lease_ttl_s "
            "FROM claims c "
            "WHERE c.consumed_at IS NULL AND c.expired = 0 "
            "AND datetime(c.claimed_at, '+' || c.lease_ttl_s || ' seconds') >= datetime(?) "
            "ORDER BY c.claimed_at DESC",
            (now_str,),
        ).fetchall()

    def release_claims(self, now: datetime, claim_ids: list[str] | None = None) -> list[str]:
        """Expire live claims, returning the node ids handed back.

        With no argument this releases *everything*, which is what a caller that
        has demonstrably lost its context needs — an agent whose conversation was
        reset cannot record results it no longer remembers, so keeping its nodes
        reserved would strand them until the TTL elapses.

        With ``claim_ids`` it releases exactly those, which is what a caller that
        has decided not to run one dispatched experiment needs. Without it the
        only ways to hand a single node back were to record a result that was
        never observed or to wait out the lease, and on a long-running experiment
        that wait is measured in days.

        Distinct from expire_stale_claims, which only collects leases that ran
        out on their own.
        """
        released: list[str] = []
        with self.transaction() as conn:
            if claim_ids is None:
                rows = conn.execute(
                    "SELECT claim_id, node_id FROM claims WHERE consumed_at IS NULL AND expired=0"
                ).fetchall()
            else:
                placeholders = ",".join("?" * len(claim_ids))
                rows = (
                    conn.execute(
                        f"SELECT claim_id, node_id FROM claims WHERE consumed_at IS NULL "  # noqa: S608 — placeholders are generated, values are bound
                        f"AND expired=0 AND claim_id IN ({placeholders})",
                        tuple(claim_ids),
                    ).fetchall()
                    if claim_ids
                    else []
                )
            for row in rows:
                conn.execute("UPDATE claims SET expired=1 WHERE claim_id=?", (row["claim_id"],))
                conn.execute(
                    "UPDATE nodes SET active_claim_id=NULL, claimed_at=NULL, updated_at=? "
                    "WHERE id=?",
                    (_dt_to_str(now), row["node_id"]),
                )
                released.append(row["node_id"])
        return released

    def renew_claim(self, claim_id: str, now: datetime, ttl_s: int) -> bool:
        """Restart a live lease's clock at ``now`` with a fresh TTL.

        An experiment that outlives its lease is the ordinary case outside a
        synchronous agent loop: a training run measured in days will always
        outlast any TTL short enough to reclaim work from a crashed caller. The
        alternative to renewal is to choose the TTL for the worst case, which
        makes every genuinely abandoned node unreclaimable for just as long.
        Renewal separates the two: the TTL stays short enough to be a useful
        liveness signal, and a caller that is still working says so.

        Returns False for a claim that is unknown, consumed, or already expired
        — a lease that is gone cannot be extended, only re-issued by a new
        dispatch, and silently re-arming it would resurrect a node another
        caller may already hold.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT consumed_at, expired FROM claims WHERE claim_id=?", (claim_id,)
            ).fetchone()
            if row is None or row["consumed_at"] is not None or row["expired"]:
                return False
            now_str = _dt_to_str(now)
            conn.execute(
                "UPDATE claims SET claimed_at=?, lease_ttl_s=? WHERE claim_id=?",
                (now_str, ttl_s, claim_id),
            )
            conn.execute(
                "UPDATE nodes SET claimed_at=?, updated_at=? WHERE active_claim_id=?",
                (now_str, now_str, claim_id),
            )
            return True
