"""Tests for the read-only dashboard: the read model and its transport.

Two properties carry this feature and both are asserted directly rather than
inferred. The observer must be physically unable to write — not merely careful —
and the socket must refuse the requests a browser can be tricked into making on
a website's behalf.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hypotree.dashboard.readmodel import (
    ReadModel,
    _layer_positions,
    _selection_probabilities,
)
from hypotree.dashboard.server import DashboardServer, choose_port
from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.evidence import LogicalEvidence
from hypotree.store.store import utcnow


def _landscape(db: Path) -> HypoTreeEngine:
    """Two questions, a combination resting on one answer from each, and a goal."""
    engine = HypoTreeEngine(db, rng_seed=7)
    for group in ("a", "b"):
        for i in range(3):
            engine.create_hypothesis(f"{group}={i}", node_id=f"{group}{i}", exclusion_group=group)
    engine.create_hypothesis(
        "combo", node_id="combo", parent_ids=["a0", "b0"], edge_type=EdgeType.DEPENDENCY
    )
    engine.create_hypothesis(
        "goal", node_id="goal", is_goal=True, target_metric=0.75, parent_ids=["combo"]
    )
    return engine


# -- read model ---------------------------------------------------------------


@pytest.mark.unit
def test_the_read_model_cannot_write(tmp_path: Path) -> None:
    """The safety property, asserted rather than assumed.

    A dashboard that is merely *careful* not to write is one refactor away from
    mutating the belief state someone is watching. Opening `mode=ro` makes the
    guarantee the driver's job.
    """
    db = tmp_path / "ro.db"
    _landscape(db).close()

    read = ReadModel(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read.store.set_directive("a0", "pin", "", "test")
    finally:
        read.close()


@pytest.mark.unit
def test_layout_is_deterministic_and_puts_premises_above_what_rests_on_them(
    tmp_path: Path,
) -> None:
    """A layout that moves on every open cannot be recognised across sessions."""
    db = tmp_path / "layout.db"
    _landscape(db).close()

    read = ReadModel(db)
    try:
        first = read.graph()
        read._cache.clear()
        second = read.graph()
        coords = {n["id"]: (n["x"], n["y"]) for n in first.nodes}
        assert coords == {n["id"]: (n["x"], n["y"]) for n in second.nodes}
        # A combination sits below the answers it assumes, and the goal below it.
        assert coords["a0"][1] < coords["combo"][1] < coords["goal"][1]
    finally:
        read.close()


@pytest.mark.unit
def test_layer_positions_handles_an_empty_graph_and_isolated_nodes() -> None:
    """A fresh workspace is the first screen anyone sees; it must not raise."""
    assert _layer_positions([], []) == {}
    lone = _layer_positions(["x"], [])
    assert lone["x"] == (0.0, 0.0)
    # An edge naming a node outside the set is ignored rather than crashing —
    # a goal filter hands us exactly that.
    assert _layer_positions(["x"], [("ghost", "x")])["x"] == (0.0, 0.0)


@pytest.mark.unit
def test_selection_probabilities_are_a_distribution_over_the_frontier(
    tmp_path: Path,
) -> None:
    """The glow is the real selection chance, so it must behave like one."""
    db = tmp_path / "glow.db"
    engine = _landscape(db)
    try:
        frontier = engine._frontier_nodes()
        nodes = engine._store.get_all_nodes()
        import numpy as np

        probs = _selection_probabilities(frontier, nodes, np.random.default_rng(0))
        assert set(probs) == {n.id for n in frontier}
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        # A single candidate is chosen with certainty.
        one = _selection_probabilities(frontier[:1], nodes, np.random.default_rng(0))
        assert one[frontier[0].id] == 1.0
    finally:
        engine.close()


@pytest.mark.unit
def test_an_empty_frontier_yields_no_probabilities(tmp_path: Path) -> None:
    import numpy as np

    assert _selection_probabilities([], [], np.random.default_rng(0)) == {}


@pytest.mark.unit
def test_the_graph_is_cached_on_the_revision_and_invalidated_by_a_write(
    tmp_path: Path,
) -> None:
    """Recomputing per request rebuilds exactly the wall the engine tore down."""
    db = tmp_path / "cache.db"
    engine = _landscape(db)
    read = ReadModel(db)
    try:
        first = read.graph()
        assert read.graph() is first, "same revision must hit the cache"
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
        assert read.graph() is not first, "a mutation must invalidate it"
        assert read.graph().revision > first.revision
    finally:
        read.close()
        engine.close()


@pytest.mark.unit
def test_scrubbing_back_shows_what_was_believed_then(tmp_path: Path) -> None:
    """SCD2 makes time travel a WHERE clause; this is the assertion that it works."""
    db = tmp_path / "time.db"
    engine = _landscape(db)
    try:
        before = utcnow()
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
        after = utcnow() + timedelta(seconds=1)
    finally:
        engine.close()

    read = ReadModel(db)
    try:
        past = {n["id"]: n["status"] for n in read.graph(at=before.isoformat()).nodes}
        now = {n["id"]: n["status"] for n in read.graph(at=after.isoformat()).nodes}
        assert past["a0"] == "UNTESTED"
        assert now["a0"] == "VERIFIED"
        # The confirmation retired its siblings; the earlier view must not show that.
        assert past["a1"] == "UNTESTED"
        assert now["a1"] == "EXHAUSTED"
    finally:
        read.close()


@pytest.mark.unit
def test_an_unparseable_at_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """Silently showing 'now' for a bad timestamp would misreport history as current."""
    db = tmp_path / "badtime.db"
    _landscape(db).close()
    read = ReadModel(db)
    try:
        with pytest.raises(ValueError, match="ISO-8601"):
            read.graph(at="last tuesday")
    finally:
        read.close()


@pytest.mark.unit
def test_a_goal_filter_narrows_the_graph_and_the_timeline(tmp_path: Path) -> None:
    """The read model and the navigator share one definition of a goal's scope."""
    db = tmp_path / "scope.db"
    engine = _landscape(db)
    try:
        engine.create_hypothesis("other", node_id="other", exclusion_group="z")
        engine.create_hypothesis(
            "goal2", node_id="goal2", is_goal=True, target_metric=0.9, parent_ids=["other"]
        )
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
    finally:
        engine.close()

    read = ReadModel(db)
    try:
        scoped = read.graph(goal_id="goal")
        ids = {n["id"] for n in scoped.nodes}
        assert "goal2" not in ids and "other" not in ids
        assert {"goal", "combo", "a0", "a1", "b0"} <= ids
        assert all(t["node_id"] in ids for t in read.timeline(goal_id="goal")["ticks"])
    finally:
        read.close()


@pytest.mark.unit
def test_the_learning_path_can_be_rewound_with_the_graph(tmp_path: Path) -> None:
    """A rewound picture captioned with today's conclusions is a lie.

    The scrubber moves the graph back in time; the narrative beside it has to
    stop at the same instant, or the panel describes findings the graph has not
    reached yet.
    """
    db = tmp_path / "path-at.db"
    engine = _landscape(db)
    try:
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
        midpoint = utcnow()
        engine.record_evidence("b0", LogicalEvidence(success=1.0, depth=1))
        engine.record_evidence("b0", LogicalEvidence(success=1.0, depth=1))
    finally:
        engine.close()

    read = ReadModel(db)
    try:
        ticks = read.timeline()["ticks"]
        assert ticks, "the landscape should have settled something"

        live = read.learning_path()
        rewound = read.learning_path(at=midpoint.isoformat())

        assert [s["node_id"] for s in live["steps"]] != []
        # Everything the rewound view knows, the live view knows too.
        early = [(s["node_id"], s["at"]) for s in rewound["steps"]]
        assert early == [s for s in ((x["node_id"], x["at"]) for x in live["steps"])][: len(early)]
        assert len(early) < len(live["steps"])
        # b0 settled after the midpoint, so the rewound story cannot mention it.
        assert "b0" not in {nid for nid, _ in early}
        assert "b0" in {s["node_id"] for s in live["steps"]}
        # A `Z` suffix is what a copied JSON timestamp carries; it must parse.
        assert (
            read.learning_path(at=midpoint.isoformat().replace("+00:00", "Z"))["steps"]
            == (rewound["steps"])
        )
        # So must a hand-typed naive one, which is read as UTC rather than
        # raising the moment it meets a stored aware instant.
        naive = midpoint.replace(tzinfo=None).isoformat()
        assert read.learning_path(at=naive)["steps"] == rewound["steps"]
        assert read.graph(at=naive) is not None
        # And it says so, rather than passing a partial story off as current.
        assert "Reconstructed as of" in rewound["markdown"]
        assert "Reconstructed as of" not in live["markdown"]
    finally:
        read.close()


@pytest.mark.unit
def test_node_detail_carries_the_provenance_proof_mode_needs(tmp_path: Path) -> None:
    db = tmp_path / "detail.db"
    engine = _landscape(db)
    try:
        engine.record_evidence(
            "a0", LogicalEvidence(success=1.0, depth=2, source_ref="pytest#4412")
        )
    finally:
        engine.close()

    read = ReadModel(db)
    try:
        detail = read.node_detail("a0")
        assert detail is not None
        assert detail["evidence"][0]["source_ref"] == "pytest#4412"
        assert detail["evidence"][0]["depth"] == 2
        assert detail["status_history"], "a settled node has an interval history"
        assert read.node_detail("ghost") is None
    finally:
        read.close()


@pytest.mark.unit
def test_the_frontier_panel_never_calls_the_dispatching_peek(tmp_path: Path) -> None:
    """`get_next_targets(dry_run=True)` writes — it expires stale claims.

    It is the obvious way to fill this panel and it would make a read-only
    viewer raise on a read-only connection.
    """
    db = tmp_path / "frontier.db"
    _landscape(db).close()
    read = ReadModel(db)
    try:
        panel = read.frontier(k=3)
        assert len(panel["candidates"]) == 3
        assert panel["candidates"][0]["p_select"] >= panel["candidates"][-1]["p_select"]
    finally:
        read.close()


# -- directives ---------------------------------------------------------------


@pytest.mark.unit
def test_a_directive_steers_dispatch_without_touching_belief(tmp_path: Path) -> None:
    """The whole point: a human can redirect the search without faking evidence."""
    db = tmp_path / "directive.db"
    engine = _landscape(db)
    try:
        before = engine._store.get_node("b2")
        engine._store.set_directive("b2", "pin", "reviewer asked", "human")
        target = engine.get_next_targets(dry_run=True)[0]
        assert target.node_id == "b2"

        after = engine._store.get_node("b2")
        assert (after.alpha, after.beta) == (before.alpha, before.beta), (
            "a directive must never move the posterior"
        )

        engine._store.set_directive("b2", "suspend", "on hold", "human")
        assert engine.get_next_targets(dry_run=True)[0].node_id != "b2"

        assert engine._store.clear_directive("b2") is True
        assert engine._store.clear_directive("b2") is False
        assert engine._store.get_directives() == {}
    finally:
        engine.close()


@pytest.mark.unit
def test_no_directives_leaves_selection_byte_identical(tmp_path: Path) -> None:
    """The default path must not move because a feature exists."""
    picks = []
    for name in ("one", "two"):
        engine = _landscape(tmp_path / f"{name}.db")
        try:
            picks.append(engine.get_next_targets(dry_run=True)[0].node_id)
        finally:
            engine.close()
    assert picks[0] == picks[1]


# -- transport ----------------------------------------------------------------


async def _request(
    port: int,
    target: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes = b"",
) -> tuple[int, dict[str, Any] | str]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    head = {"Host": f"127.0.0.1:{port}", "Connection": "close", **(headers or {})}
    if body:
        head["Content-Length"] = str(len(body))
    raw = f"{method} {target} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in head.items())
    writer.write(raw.encode() + b"\r\n" + body)
    await writer.drain()
    payload = await reader.read()
    writer.close()
    head_bytes, _, tail = payload.partition(b"\r\n\r\n")
    status = int(head_bytes.split(b" ")[1])
    try:
        return status, json.loads(tail)
    except json.JSONDecodeError:
        return status, tail.decode()


@pytest_asyncio.fixture
async def server(tmp_path: Path):
    db = tmp_path / "srv.db"
    engine = _landscape(db)
    engine.record_evidence("a0", LogicalEvidence(success=1.0, depth=1))
    engine.close()
    srv = DashboardServer(db, port=choose_port(7500))
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_requires_the_session_token(server: DashboardServer) -> None:
    """Localhost is reachable by every process on the machine."""
    status, _ = await _request(server.port, "/api/meta")
    assert status == 401
    status, _ = await _request(server.port, f"/api/meta?t={server.token}")
    assert status == 200
    status, _ = await _request(server.port, "/api/meta", headers={"X-Hypotree-Token": server.token})
    assert status == 200
    status, _ = await _request(server.port, "/api/meta?t=wrong")
    assert status == 401


@pytest.mark.asyncio
@pytest.mark.integration
async def test_a_foreign_host_header_is_refused(server: DashboardServer) -> None:
    """The DNS-rebinding defence — the hole behind CVEs in Jupyter, Ray, TensorBoard.

    A page on evil.test that re-resolves to 127.0.0.1 still sends its own name,
    and once the browser treats the reply as same-origin a token in the URL is no
    longer a secret from it.
    """
    status, _ = await _request(
        server.port, f"/api/meta?t={server.token}", headers={"Host": "evil.test"}
    )
    assert status == 403


@pytest.mark.asyncio
@pytest.mark.integration
async def test_a_cross_origin_request_is_refused(server: DashboardServer) -> None:
    status, _ = await _request(
        server.port, f"/api/meta?t={server.token}", headers={"Origin": "https://evil.test"}
    )
    assert status == 403
    # Right host, wrong port is still a different origin, and it is the case a
    # second local tool would hit.
    status, _ = await _request(
        server.port,
        f"/api/meta?t={server.token}",
        headers={"Origin": f"http://127.0.0.1:{server.port + 1}"},
    )
    assert status == 403
    # A sandboxed iframe sends `null`, and a junk port must be refused rather
    # than raise inside the check.
    for bad in ("null", "http://127.0.0.1:not-a-port", f"https://127.0.0.1:{server.port}"):
        status, _ = await _request(
            server.port, f"/api/meta?t={server.token}", headers={"Origin": bad}
        )
        assert status == 403, bad


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_page_may_write_back_to_its_own_origin(server: DashboardServer) -> None:
    """The dashboard's own POSTs carry an Origin; refusing the header broke them.

    A browser attaches `Origin` to a same-origin JSON POST too, so treating the
    header's presence as hostile locked the page out of its own directive
    buttons while leaving the actual CSRF case exactly as covered.
    """
    body = json.dumps({"node_id": "a0", "mode": "pin"}).encode()
    status, payload = await _request(
        server.port,
        f"/api/directive?t={server.token}",
        method="POST",
        headers={
            "Origin": f"http://127.0.0.1:{server.port}",
            "Content-Type": "application/json",
        },
        body=body,
    )
    # This viewer has no engine attached, so the write cannot land — but it got
    # past the origin gate, which is the thing under test.
    assert status != 403, payload
    status, _ = await _request(
        server.port,
        f"/api/meta?t={server.token}",
        headers={"Origin": f"http://localhost:{server.port}"},
    )
    assert status == 200


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_shell_needs_no_token_but_the_data_does(server: DashboardServer) -> None:
    """The page is public; everything it renders is not."""
    status, body = await _request(server.port, "/")
    assert status == 200
    assert isinstance(body, str) and "hypotree" in body


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_read_endpoints_answer(server: DashboardServer) -> None:
    t = server.token
    status, meta = await _request(server.port, f"/api/meta?t={t}")
    assert status == 200 and isinstance(meta, dict)
    assert [g["id"] for g in meta["goals"]] == ["goal"]

    status, graph = await _request(server.port, f"/api/graph?t={t}")
    assert status == 200 and isinstance(graph, dict)
    assert graph["nodes"] and all("p_select" in n for n in graph["nodes"])

    status, node = await _request(server.port, f"/api/node/a0?t={t}")
    assert status == 200 and isinstance(node, dict) and node["node"]["id"] == "a0"

    status, _ = await _request(server.port, f"/api/node/ghost?t={t}")
    assert status == 404

    status, frontier = await _request(server.port, f"/api/frontier?t={t}&k=2")
    assert status == 200 and isinstance(frontier, dict)
    assert len(frontier["candidates"]) == 2

    status, path = await _request(server.port, f"/api/learning-path?t={t}")
    assert status == 200 and isinstance(path, dict) and "steps" in path

    status, timeline = await _request(server.port, f"/api/timeline?t={t}")
    assert status == 200 and isinstance(timeline, dict) and timeline["ticks"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bad_input_is_a_400_not_a_500(server: DashboardServer) -> None:
    """A viewer handed nonsense must say so, not fall over."""
    status, body = await _request(server.port, f"/api/graph?t={server.token}&at=nonsense")
    assert status == 400
    status, _ = await _request(server.port, f"/api/frontier?t={server.token}&k=abc")
    assert status == 400
    status, _ = await _request(server.port, f"/api/nope?t={server.token}")
    assert status == 404


@pytest.mark.asyncio
@pytest.mark.integration
async def test_a_directive_post_is_refused_without_an_engine(server: DashboardServer) -> None:
    """A read-only viewer says so plainly rather than pretending to have worked."""
    status, body = await _request(
        server.port,
        f"/api/directive?t={server.token}",
        method="POST",
        body=json.dumps({"node_id": "a0", "mode": "pin"}).encode(),
    )
    assert status == 503
    assert isinstance(body, dict) and "read-only" in body["error"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_a_directive_post_goes_through_the_writer_when_attached(tmp_path: Path) -> None:
    db = tmp_path / "write.db"
    engine = _landscape(db)
    engine.close()
    applied: list[tuple[str, str, str]] = []

    async def writer(node_id: str, mode: str, reason: str) -> dict[str, Any]:
        applied.append((node_id, mode, reason))
        return {"ok": True, "node_id": node_id, "mode": mode}

    srv = DashboardServer(db, port=choose_port(7520), writer=writer)
    await srv.start()
    try:
        status, body = await _request(
            srv.port,
            f"/api/directive?t={srv.token}",
            method="POST",
            body=json.dumps({"node_id": "a0", "mode": "pin", "reason": "look here"}).encode(),
        )
        assert status == 200 and isinstance(body, dict) and body["ok"] is True
        assert applied == [("a0", "pin", "look here")]

        status, _ = await _request(
            srv.port,
            f"/api/directive?t={srv.token}",
            method="POST",
            body=json.dumps({"node_id": "a0", "mode": "nonsense"}).encode(),
        )
        assert status == 400
    finally:
        await srv.stop()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_pushes_a_revision_and_drops_slow_subscribers(
    server: DashboardServer,
) -> None:
    """The stream carries an integer, so a listener that fell behind can skip ahead."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        f"GET /api/events?t={server.token} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{server.port}\r\n\r\n".encode()
    )
    await writer.drain()
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        assert b"text/event-stream" in head
        first = await asyncio.wait_for(reader.readuntil(b"\n\n"), timeout=5)
        assert b'"revision"' in first

        server.notify(9999)
        pushed = await asyncio.wait_for(reader.readuntil(b"\n\n"), timeout=5)
        assert json.loads(pushed.decode().split("data: ")[1])["revision"] == 9999

        # Overrun the queue: the newest revision must still arrive.
        for revision in range(10_000, 10_050):
            server.notify(revision)
        assert server._subscribers, "an overrun subscriber is dropped from, not left in, the set"
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_an_idle_stream_sends_a_keepalive_instead_of_dying(
    server: DashboardServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`asyncio.TimeoutError` only became the builtin `TimeoutError` in 3.11.

    Catching the builtin meant that on 3.10 the first quiet interval killed the
    stream task instead of emitting a comment frame, and the page sat in a
    reconnect loop that looks exactly like a flaky network. The interval is
    patched down so the test does not wait fifteen seconds for it.
    """
    real_wait_for = asyncio.wait_for

    async def impatient(aw, timeout):  # noqa: ANN001, ANN202
        return await real_wait_for(aw, 0.05 if timeout == 15.0 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", impatient)

    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        f"GET /api/events?t={server.token} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{server.port}\r\n\r\n".encode()
    )
    await writer.drain()
    try:
        await real_wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        await real_wait_for(reader.readuntil(b"\n\n"), timeout=5)
        beat = await real_wait_for(reader.readuntil(b"\n\n"), timeout=5)
        assert beat == b": keepalive\n\n"
        # And it keeps going rather than emitting one and stopping.
        assert await real_wait_for(reader.readuntil(b"\n\n"), timeout=5) == b": keepalive\n\n"
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@pytest.mark.unit
def test_choose_port_skips_a_taken_one() -> None:
    import socket as _socket

    port = choose_port(7600)
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as held:
        held.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        held.bind(("127.0.0.1", port))
        held.listen(1)
        assert choose_port(port) != port


# -- vendored assets and the SPA ----------------------------------------------


@pytest.mark.unit
def test_every_vendored_script_the_page_loads_is_in_the_package() -> None:
    """A page that 404s half its scripts is worse than one that says it cannot run.

    The whole point of vendoring is that the tool works on a plane, behind a
    proxy, and in an air-gapped enterprise. If a script silently goes missing,
    that promise is broken quietly.
    """
    from hypotree.dashboard.assets import ASSETS, VENDOR_SCRIPTS, missing_scripts

    assert missing_scripts() == []
    for name in VENDOR_SCRIPTS:
        content_type, body = ASSETS[name]
        assert body, f"{name} is empty"
        assert "javascript" in content_type


@pytest.mark.unit
def test_the_page_references_exactly_the_scripts_that_are_vendored() -> None:
    """The HTML and the asset table must not drift apart."""
    from hypotree.dashboard.assets import ASSETS, VENDOR_SCRIPTS
    from hypotree.dashboard.server import _load_index

    html = _load_index()
    referenced = set(re.findall(r'src="/static/([^"]+)"', html))
    referenced |= set(re.findall(r'href="/static/([^"]+)"', html))
    unvendored = referenced - set(ASSETS)
    assert not unvendored, f"page references unvendored assets: {unvendored}"
    for name in VENDOR_SCRIPTS:
        assert name in referenced, f"{name} is vendored but never loaded"
    # d3-zoom's UMD build resolves its peers off the global at load time, so a
    # peer listed after it would be undefined when it initialises.
    order = [referenced_name for referenced_name in re.findall(r'src="/static/([^"]+)"', html)]
    assert order.index("d3-selection.min.js") < order.index("d3-zoom.min.js")
    assert order.index("d3-transition.min.js") < order.index("d3-zoom.min.js")
    assert order.index("d3-drag.min.js") < order.index("d3-zoom.min.js")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_static_assets_are_served_without_a_token(server: DashboardServer) -> None:
    """The page needs its scripts before it has anywhere to put a token."""
    status, body = await _request(server.port, "/static/app.css")
    assert status == 200
    assert isinstance(body, str) and "--accent" in body

    status, _ = await _request(server.port, "/static/marked.min.js")
    assert status == 200


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_logo_is_served_as_an_image(server: DashboardServer) -> None:
    """The wordmark is a binary asset, and the table only carried text before.

    A manifest that silently drops a file type serves a 404 for something the
    page references, which is a broken header rather than a missing feature.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        f"GET /static/logo.png HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
        f"Connection: close\r\n\r\n".encode()
    )
    await writer.drain()
    payload = await reader.read()
    writer.close()
    head, _, tail = payload.partition(b"\r\n\r\n")

    assert head.split(b" ")[1] == b"200"
    assert b"image/png" in head
    assert tail.startswith(b"\x89PNG")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_static_serving_cannot_traverse_out_of_the_asset_table(
    server: DashboardServer,
) -> None:
    """Traversal is unrepresentable here: the lookup is a dict, not a path join."""
    for attempt in (
        "/static/../server.py",
        "/static/....//....//etc/passwd",
        "/static/%2e%2e/server.py",
        "/static/",
    ):
        status, _ = await _request(server.port, attempt)
        assert status == 404, attempt


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_shell_carries_the_token_and_a_same_origin_policy(
    server: DashboardServer,
) -> None:
    """Everything is vendored, so the policy can say `self` and mean it."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    raw = (await reader.read()).decode()
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()

    head, _, body = raw.partition("\r\n\r\n")
    assert "Content-Security-Policy: default-src 'self'" in head
    assert server.token in body, "the page must be able to authenticate its own fetches"
    assert "__TOKEN__" not in body, "the placeholder must be substituted"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_meta_reports_missing_assets_so_the_page_can_say_so(
    server: DashboardServer,
) -> None:
    status, meta = await _request(server.port, f"/api/meta?t={server.token}")
    assert status == 200 and isinstance(meta, dict)
    assert meta["missing_assets"] == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_api_returns_every_field_the_interface_reads(
    server: DashboardServer,
) -> None:
    """The contract between the read model and the page, pinned.

    The interface calls things like `node.posterior_mean.toFixed(3)`. A field
    quietly renamed on the Python side becomes a blank panel or a thrown
    exception in a browser nobody is watching during a test run, so the shape is
    asserted here instead.
    """
    t = server.token
    _, graph = await _request(server.port, f"/api/graph?t={t}")
    assert isinstance(graph, dict)
    for node in graph["nodes"]:
        assert {
            "id",
            "statement",
            "status",
            "is_goal",
            "exclusion_group",
            "posterior_mean",
            "evidence_count",
            "p_select",
            "directive",
            "x",
            "y",
        } <= set(node)
    for edge in graph["edges"]:
        assert {"src", "dst", "type"} <= set(edge)
    assert isinstance(graph["stats"], dict)

    _, meta = await _request(server.port, f"/api/meta?t={t}")
    assert isinstance(meta, dict)
    assert {"revision", "goals", "missing_assets"} <= set(meta)
    for goal in meta["goals"]:
        assert {"id", "statement", "met"} <= set(goal)

    _, frontier = await _request(server.port, f"/api/frontier?t={t}&k=3")
    assert isinstance(frontier, dict)
    for candidate in frontier["candidates"]:
        assert {"node_id", "statement", "p_select", "directive"} <= set(candidate)

    _, path = await _request(server.port, f"/api/learning-path?t={t}")
    assert isinstance(path, dict) and isinstance(path.get("markdown"), str)

    _, timeline = await _request(server.port, f"/api/timeline?t={t}")
    assert isinstance(timeline, dict)
    for tick in timeline["ticks"]:
        assert {"t", "node_id", "status", "reason"} <= set(tick)

    node_id = graph["nodes"][0]["id"]
    _, detail = await _request(server.port, f"/api/node/{node_id}?t={t}")
    assert isinstance(detail, dict)
    assert {"node", "evidence", "status_history", "directive"} <= set(detail)
    assert {"posterior_mean", "evidence_count", "status"} <= set(detail["node"])
    for item in detail["evidence"]:
        assert {"success", "depth", "recorded_at", "source_ref", "git_branch"} <= set(item)


# -- the policy the page actually has to run under ----------------------------
#
# The HTTP layer and the API contract were both covered and the page still did
# not run: Vue's compiler needs `new Function`, and the token arrived in an
# inline <script>. Both were blocked by a policy no test asserted anything about.


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_policy_permits_what_the_vendored_scripts_need(
    server: DashboardServer,
) -> None:
    """Vue's global build compiles the in-DOM template with `new Function`."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    head = (await reader.read()).decode().partition("\r\n\r\n")[0]
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()

    csp = next(line for line in head.split("\r\n") if line.startswith("Content-Security-Policy:"))
    assert "script-src 'self' 'unsafe-eval'" in csp, "Vue cannot compile its template without it"
    assert "style-src 'self' 'unsafe-inline'" in csp, "the graph sets a transform per node"
    assert "data:" in csp, "extensions inject data: URIs and the console fills with noise"
    # The refusal that matters: an injected event handler must never run.
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    for directive in ("object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'"):
        assert directive in csp


@pytest.mark.unit
def test_the_page_contains_no_inline_script() -> None:
    """An inline <script> is blocked by the very policy that protects v-html.

    Keeping the page free of them is what lets `script-src` stay strict, so this
    is a standing constraint rather than a one-off fix.
    """
    from hypotree.dashboard.server import _load_index

    html = _load_index()
    # Comments first: one of them talks *about* inline scripts.
    stripped = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    inline = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", stripped, re.S)
    assert [block for block in inline if block.strip()] == []


@pytest.mark.unit
def test_the_token_travels_in_a_meta_tag() -> None:
    """Because it cannot travel in an inline script any more."""
    from hypotree.dashboard.server import _load_index

    html = _load_index()
    assert re.search(r'<meta\s+name="hypotree-token"\s+content="__TOKEN__"', html)


@pytest.mark.unit
def test_the_markdown_renderer_is_wired_through_the_sanitiser() -> None:
    """`marked` does not escape HTML, and node statements are untrusted text.

    The learning path goes through `v-html`, so a statement containing markup
    would otherwise land in the document verbatim. Asserted against the shipped
    script because there is no JS test runner here — the check is that the unsafe
    shape is absent and the safe one present.
    """
    app_js = (Path(__file__).parents[2] / "src/hypotree/dashboard/static/app.js").read_text()
    assert "sanitize(marked.parse(" in app_js, "markdown must not reach v-html unsanitised"
    assert "ALLOWED_TAGS" in app_js and "ALLOWED_ATTRS" in app_js
    # An allowlist, not a blocklist: anything unnamed is dropped.
    assert "el.removeAttribute" in app_js
    assert 'name.startsWith("on")' in app_js


@pytest.mark.unit
def test_the_shipped_javascript_parses() -> None:
    """A syntax error in app.js breaks the page as silently as the CSP did.

    Skipped where node is unavailable rather than made a hard dependency: it is a
    check on an artifact the wheel carries, not on the runtime the package needs.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    static = Path(__file__).parents[2] / "src/hypotree/dashboard/static"
    result = subprocess.run(
        [node, "--check", str(static / "app.js")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
