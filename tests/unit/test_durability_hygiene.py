"""Durability, audit-integrity and observability guarantees the engine relies on.

Each test here pins a property that something else in the system quietly assumes
and that nothing else checks: the revision counter's meaning, the audit log's
truthfulness, the difference between converging and running out of budget, and
the isolation of a read from the write that follows it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import Edge, EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.models.node import Node
from hypotree.navigator.convergence import (
    CONVERGED_CEILING,
    CONVERGED_DETERMINISTIC,
    CONVERGED_INTERVAL,
    NOT_CONVERGED,
    convergence_gate,
    convergence_verdict,
)
from hypotree.navigator.sampler import ThompsonSampler
from hypotree.store.store import HypoTreeStore


def _store(tmp_path: Path) -> HypoTreeStore:
    return HypoTreeStore(tmp_path / "t.db")


# -- audit integrity ----------------------------------------------------------


@pytest.mark.unit
def test_a_duplicate_edge_writes_no_event_and_does_not_move_the_revision(tmp_path: Path) -> None:
    """`events.seq` means "something changed", and two readers depend on that.

    The dashboard keys its entire snapshot cache and its SSE invalidation on the
    revision, so a write that changes nothing but bumps the counter makes every
    connected browser refetch and relayout the whole graph for no reason. The
    audit log is the second victim: an `EdgeAdded` row for an insert that
    `INSERT OR IGNORE` discarded asserts a change that did not happen.
    """
    store = _store(tmp_path)
    try:
        store.add_node(Node(id="a", statement="A"))
        store.add_node(Node(id="b", statement="B"))
        edge = Edge(src="a", dst="b", type=EdgeType.DEPENDENCY)

        store.add_edge(edge)
        after_first = store.latest_event_seq()
        assert len(store.get_all_edges()) == 1

        store.add_edge(edge)
        assert store.latest_event_seq() == after_first, "a no-op write moved the revision"
        assert len(store.get_all_edges()) == 1

        added = [e for e in store.get_events() if e["type"] == "EdgeAdded"]
        assert len(added) == 1, "the audit log claimed an edge was added twice"
    finally:
        store.close()


@pytest.mark.unit
def test_a_genuinely_new_edge_still_writes_its_event(tmp_path: Path) -> None:
    """The duplicate guard must not suppress real additions."""
    store = _store(tmp_path)
    try:
        for nid in ("a", "b", "c"):
            store.add_node(Node(id=nid, statement=nid.upper()))
        store.add_edge(Edge(src="a", dst="b", type=EdgeType.DEPENDENCY))
        before = store.latest_event_seq()
        store.add_edge(Edge(src="a", dst="c", type=EdgeType.DEPENDENCY))
        assert store.latest_event_seq() > before
        assert len([e for e in store.get_events() if e["type"] == "EdgeAdded"]) == 2
    finally:
        store.close()


# -- durability settings ------------------------------------------------------


@pytest.mark.unit
def test_the_connection_is_tuned_for_a_long_running_write_path(tmp_path: Path) -> None:
    """WAL alone left an fsync on every belief change.

    Every mutation writes its audit event in the same transaction, so the
    default `synchronous=FULL` was paying for a durability guarantee no caller
    asked for on a path that runs thousands of times in a session. NORMAL under
    WAL can lose only the last transaction and cannot corrupt the file.
    """
    store = _store(tmp_path)
    try:
        conn = store._conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1, "1 == NORMAL"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
    finally:
        store.close()


@pytest.mark.unit
def test_a_read_only_store_is_not_given_write_pragmas(tmp_path: Path) -> None:
    """An observer must stay physically unable to write, tuning or not."""
    HypoTreeStore(tmp_path / "t.db").close()
    reader = HypoTreeStore(tmp_path / "t.db", read_only=True)
    try:
        with pytest.raises(Exception, match="readonly|attempt to write"):
            reader._conn.execute("CREATE TABLE nope (x INTEGER)")
    finally:
        reader.close()


# -- lost-update hazard -------------------------------------------------------


@pytest.mark.unit
def test_an_infra_retry_does_not_revert_a_concurrent_column(tmp_path: Path) -> None:
    """The counter was bumped by rewriting every column from a stale snapshot.

    Whatever was written between the read and the save was silently reverted —
    and an infra error arrives interleaved with exactly the posterior and claim
    writes that would be lost.
    """
    store = _store(tmp_path)
    try:
        store.add_node(Node(id="n", statement="N"))
        stale = store.get_node("n")
        assert stale is not None

        store.update_posterior("n", 4.0, 2.0)  # happens after the snapshot above
        new_count = store.increment_infra_retries("n", datetime.now(timezone.utc))

        node = store.get_node("n")
        assert node is not None
        assert new_count == 1
        assert node.infra_retry_count == 1
        assert (node.alpha, node.beta) == (4.0, 2.0), "a whole-row write reverted the posterior"
    finally:
        store.close()


# -- convergence observability ------------------------------------------------


@pytest.mark.unit
def test_the_verdict_separates_a_tight_interval_from_a_spent_budget() -> None:
    """Settled by measurement and settled by ceiling used to be indistinguishable."""
    # Deterministic: one observation is the whole truth.
    assert convergence_verdict("deterministic", 1, 1.0, 1.0) == (True, CONVERGED_DETERMINISTIC)
    assert convergence_verdict("deterministic", 0, 1.0, 1.0) == (False, NOT_CONVERGED)

    # Stochastic, still wide and under the ceiling.
    assert convergence_verdict("stochastic", 3, 3.0, 2.0) == (False, NOT_CONVERGED)

    # Stochastic, interval genuinely tight.
    converged, reason = convergence_verdict("stochastic", 10, 900.0, 900.0)
    assert (converged, reason) == (True, CONVERGED_INTERVAL)

    # Stochastic, wide interval but out of budget.
    converged, reason = convergence_verdict("stochastic", 50, 5.0, 5.0, n_max=50)
    assert (converged, reason) == (True, CONVERGED_CEILING)


@pytest.mark.unit
def test_the_boolean_gate_is_unchanged_by_the_verdict_split() -> None:
    """The reason is additive; nothing about *whether* a node settles moved."""
    for regime in ("deterministic", "stochastic"):
        for count in (0, 1, 5, 49, 50, 51):
            for alpha, beta in ((1.0, 1.0), (5.0, 5.0), (900.0, 900.0)):
                expected = convergence_verdict(regime, count, alpha, beta)[0]
                assert convergence_gate(regime, count, alpha, beta) is expected


@pytest.mark.unit
def test_the_settling_note_names_only_the_ceiling_case() -> None:
    """Annotating the ordinary path would bury the one case worth noticing."""
    sampler = ThompsonSampler(rng=None)

    ceiling = Node(id="c", statement="C", evidence_regime="stochastic", alpha=5.0, beta=5.0)
    note = sampler.settling_note(ceiling, 50)
    assert note is not None
    assert "sample ceiling" in note and "CI width" in note

    tight = Node(id="t", statement="T", evidence_regime="stochastic", alpha=900.0, beta=900.0)
    assert sampler.settling_note(tight, 50) is None

    deterministic = Node(id="d", statement="D", evidence_regime="deterministic")
    assert sampler.settling_note(deterministic, 1) is None

    unsettled = Node(id="u", statement="U", evidence_regime="stochastic", alpha=2.0, beta=2.0)
    assert sampler.settling_note(unsettled, 3) is None


@pytest.mark.unit
def test_the_ceiling_reaches_the_status_history_the_agent_can_read(tmp_path: Path) -> None:
    """ "Why did this settle?" has to be answerable from the audit log.

    A stochastic node whose interval never tightened leaves the same status a
    well-measured one does. The reason is where the difference has to live.
    """
    engine = HypoTreeEngine(tmp_path / "t.db", rng_seed=1, project_path=tmp_path)
    try:
        engine.create_hypothesis("noisy", node_id="noisy", evidence_regime="stochastic")
        # Alternating results keep the interval wide, so only the ceiling can
        # settle this node.
        for i in range(60):
            engine.record_evidence("noisy", LogicalEvidence(success=float(i % 2), summary="run"))

        node = engine._store.get_node("noisy")
        assert node is not None
        assert node.status.value in {"VERIFIED", "EXHAUSTED", "INVALIDATED"}

        reasons = [
            str(r["reason"])
            for r in engine._store.get_status_history("noisy")
            if "sample ceiling" in str(r["reason"])
        ]
        assert reasons, "nothing recorded that the node settled on budget, not on evidence"
    finally:
        engine.close()


# -- dispatch snapshot --------------------------------------------------------


@pytest.mark.unit
def test_a_handed_in_snapshot_selects_exactly_what_a_fresh_read_would(tmp_path: Path) -> None:
    """Sharing one read inside a pass must not change which nodes are eligible.

    The read is shared only where no write intervenes; if the two ever disagreed
    the result would be a silently wrong selection rather than a slow one, which
    is why nothing is cached across calls.
    """
    engine = HypoTreeEngine(tmp_path / "t.db", rng_seed=3, project_path=tmp_path)
    try:
        engine.create_hypotheses(
            [
                {"node_id": "g", "statement": "goal", "is_goal": True},
                {"node_id": "a", "statement": "A", "parent_ids": []},
                {"node_id": "b", "statement": "B", "parent_ids": []},
            ]
        )
        engine._sync_graph_from_store()

        fresh = [n.id for n in engine._frontier_nodes()]
        handed = [n.id for n in engine._frontier_nodes(engine._store.get_all_nodes())]
        assert fresh == handed
        assert "g" not in fresh, "a goal is never dispatchable"
    finally:
        engine.close()


# -- rewound posterior --------------------------------------------------------


@pytest.mark.unit
def test_posterior_history_can_be_read_for_the_whole_workspace_at_once(tmp_path: Path) -> None:
    """The rewound dashboard view asked per node; the live one never did.

    Same shape as `get_all_status_history`, and it exists for the same reason:
    a scrubber drag is one question about every node, and the snapshot cache
    cannot amortise it because the instant is part of the cache key.
    """
    engine = HypoTreeEngine(tmp_path / "t.db", rng_seed=1, project_path=tmp_path)
    try:
        engine.create_hypotheses(
            [{"node_id": "a", "statement": "A"}, {"node_id": "b", "statement": "B"}]
        )
        engine.record_evidence("a", LogicalEvidence(success=1.0, summary="ok"))
        rows = engine._store.get_all_posterior_history()
    finally:
        engine.close()

    assert rows, "no posterior intervals were recorded"
    assert {str(r["node_id"]) for r in rows} >= {"a"}
    # Oldest-first, so a later interval overwrites an earlier one when a reader
    # folds them into a point-in-time view.
    stamps = [str(r["valid_from"]) for r in rows]
    assert stamps == sorted(stamps)


# -- identity resolution cost -------------------------------------------------


@pytest.mark.unit
def test_the_git_remote_is_resolved_once_per_process(tmp_path: Path, monkeypatch) -> None:
    """Three git subprocesses sat on the path before the first MCP handshake.

    Each carries a five-second timeout, so on a machine where a remote lives
    behind a hung mount the server can spend fifteen seconds looking unstarted.
    """
    from hypotree.store import identity

    calls: list[str] = []
    real = identity.subprocess.run

    def counting_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "git" in cmd[0]:
            calls.append(" ".join(map(str, cmd)))
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(identity.subprocess, "run", counting_run)
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)

    identity.reset_identity_cache()
    identity.workspace_id(tmp_path)
    after_first = len(calls)
    identity.workspace_id(tmp_path)
    assert len(calls) == after_first, "the remote lookup ran again for the same path"

    identity.reset_identity_cache()
    identity.workspace_id(tmp_path)
    assert len(calls) > after_first, "resetting the cache must force re-resolution"


@pytest.mark.unit
def test_two_paths_do_not_share_a_cached_remote(tmp_path: Path, monkeypatch) -> None:
    """The cache is keyed by path; a shared entry would merge two belief states."""
    from hypotree.store import identity

    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    identity.reset_identity_cache()
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    assert identity.workspace_id(one) != identity.workspace_id(two)


# -- layering -----------------------------------------------------------------


@pytest.mark.unit
def test_frontier_eligibility_is_a_public_question(tmp_path: Path) -> None:
    """The engine reached into the graph's privates to ask it.

    A settled node must stay off the frontier: EXHAUSTED is the conclusiveness
    guard that stops the navigator re-selecting what it already knows.
    """
    from hypotree.graph.dag import HypoTreeGraph
    from hypotree.models.status import Status

    graph = HypoTreeGraph()
    assert graph.is_frontier_status(Status.UNTESTED) is True
    assert graph.is_frontier_status(Status.IN_PROGRESS) is True
    assert graph.is_frontier_status(Status.EXHAUSTED) is False
    assert graph.is_frontier_status(Status.VERIFIED) is False
    assert not hasattr(graph, "_is_frontier_status")


# -- packaging ----------------------------------------------------------------


@pytest.mark.unit
def test_the_declared_dependencies_are_all_actually_imported() -> None:
    """A dependency nothing imports is install weight and a false signal.

    `sqlalchemy` was declared for long enough to imply an ORM was in play, in a
    package whose entire product is a hand-written SQLite schema.

    Parsed with a regex rather than `tomllib` so this runs on 3.10 too — the
    floor the package claims to support, and the interpreter a skip here would
    quietly stop checking.
    """
    import re

    root = Path(__file__).resolve().parents[2]
    block = re.search(
        r"^dependencies = \[(.*?)^\]", (root / "pyproject.toml").read_text(), re.S | re.M
    )
    assert block is not None, "could not find the dependencies block"
    names = {
        m.split(">")[0].split("=")[0].split("[")[0].strip()
        for m in re.findall(r'"([^"]+)"', block.group(1))
    }
    assert names, "parsed no dependencies"
    assert "sqlalchemy" not in names

    sources = "\n".join(
        p.read_text() for p in (root / "src").rglob("*.py") if "__pycache__" not in str(p)
    )
    # The one distribution whose import name differs from its package name.
    import_names = {"pyyaml": "yaml"}
    for name in names:
        module = import_names.get(name, name)
        assert re.search(rf"^\s*(?:from|import)\s+{re.escape(module)}\b", sources, re.M), (
            f"{name} is declared as a dependency but never imported"
        )


# -- lease hygiene ------------------------------------------------------------


@pytest.mark.unit
def test_a_lease_that_outlives_its_ttl_is_reclaimed(tmp_path: Path) -> None:
    """The reclaim sweep is inclusive, which is why a zero TTL is refused."""
    engine = HypoTreeEngine(tmp_path / "t.db", rng_seed=1, project_path=tmp_path)
    try:
        engine.create_hypothesis("h", node_id="h")
        first = engine.get_next_targets(lease_ttl_s=30)
        assert first[0].node_id == "h"
        assert engine.get_next_targets()[0].status == "DONE", "a live lease must block re-dispatch"

        engine._store.expire_stale_claims(datetime.now(timezone.utc) + timedelta(seconds=31))
        assert engine.get_next_targets()[0].node_id == "h"
    finally:
        engine.close()
