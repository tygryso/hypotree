"""A localhost HTTP window onto the belief state, on the MCP server's own loop.

Hand-rolled HTTP/1.1 rather than a framework or `http.server`. The route table is
fixed and tiny, `http.server` is blocking and would need a thread to sit
alongside an asyncio MCP server, and a web framework is a dependency the package
does not otherwise need. What is left is request-line and header parsing against
hard caps, which is bounded work.

**Localhost is not a security model.** A socket on 127.0.0.1 is reachable by every
process on the machine and, through a browser the user already has open, by any
website they visit. Three defences, because each covers a case the others do not:

* an unguessable token generated at startup, required on every `/api/*` call, so
  a process that cannot read the URL cannot read the belief state;
* `Host` validation, which is what actually stops DNS rebinding — the attack that
  has produced CVEs in Jupyter, Ray and TensorBoard, and the one a token alone
  does not stop once a page can read same-origin responses;
* no CORS headers at all, so a cross-origin fetch cannot read a reply even if it
  is somehow sent.

The SPA is served from a string held in the process. Path traversal is not
filtered, it is unrepresentable: there is no path join anywhere in this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from hypotree.dashboard.assets import ASSETS, missing_scripts
from hypotree.dashboard.readmodel import ReadModel

# Default port and how far to probe upward when it is taken. Failing with a list
# of what was tried beats failing with "address in use" and no next step.
DEFAULT_PORT = 7331
PORT_PROBE_RANGE = 10

# `SO_REUSEADDR` means opposite things on the two platforms. On POSIX it only
# waives TIME_WAIT; on Windows it lets a bind succeed against a port another
# socket is actively listening on, which would make the probe below report every
# taken port as free. `asyncio.create_server` draws the same line, so matching it
# keeps the probe's answer true of the bind it is predicting.
_PROBE_REUSES_ADDRESS = sys.platform != "win32"

# Hard caps on anything a client controls before it has authenticated. A request
# line long enough to matter is an attack, not a browser.
_MAX_REQUEST_LINE = 8 * 1024
_MAX_HEADERS = 64
_MAX_HEADER_BYTES = 32 * 1024
_MAX_BODY_BYTES = 64 * 1024

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})

# Everything is vendored and same-origin, so the policy can say `self` and mean
# it. Two deliberate relaxations, and one deliberate refusal:
#
#   script-src 'unsafe-eval'  Vue's global build compiles the in-DOM template
#                             with `new Function`. The alternative is the runtime
#                             build plus precompiled render functions, which
#                             needs the npm toolchain this package exists partly
#                             to avoid. Every script that can run is one we ship.
#   style-src 'unsafe-inline' the graph sets a transform per node.
#   font-src/img-src data:    browser extensions inject data: URIs into pages;
#                             blocking them only fills the console with noise
#                             about content this app never asked for.
#
# `script-src` deliberately does *not* get 'unsafe-inline'. That is what stops an
# event handler injected through the learning path from ever running, and it is
# why the session token travels in a meta tag rather than an inline script.
_CSP = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)

# Served when the SPA is not in the package — a source checkout with the static
# directory stripped. Deliberately functional rather than blank: the first screen
# of a tool judged on adoption is worth more than a 404, and it keeps the
# transport demonstrable end to end without the interface existing.
_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>hypotree</title>
<style>
 body{background:#2D3142;color:#E8E6E3;font:15px/1.6 ui-sans-serif,system-ui,sans-serif;
      margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center}
 main{max-width:44rem;padding:2.5rem;background:#5B1A2E;border-radius:14px;
      box-shadow:0 18px 50px rgba(0,0,0,.45)}
 h1{margin:0 0 .4rem;font-size:1.5rem;letter-spacing:.01em}
 code{background:rgba(0,0,0,.28);padding:.15rem .4rem;border-radius:5px;color:#C9A227}
 a{color:#C9A227}
 ul{padding-left:1.1rem} li{margin:.25rem 0}
</style></head><body><main>
<h1>hypotree dashboard</h1>
<p>The read model and its API are live. The interface itself lands with Track D.</p>
<ul>
<li><code>GET /api/meta</code> — workspace identity and the goal list</li>
<li><code>GET /api/graph?goal_id=&amp;at=</code> — laid-out nodes and edges</li>
<li><code>GET /api/node/&lt;id&gt;</code> — detail and provenance</li>
<li><code>GET /api/frontier?goal_id=&amp;k=5</code> — what is likely next, and how likely</li>
<li><code>GET /api/learning-path?goal_id=</code> — the narrative</li>
<li><code>GET /api/timeline?goal_id=</code> — scrubber ticks</li>
<li><code>GET /api/events</code> — server-sent revision notifications</li>
</ul>
<p>Every <code>/api/*</code> call needs the session token, as
<code>?t=&lt;token&gt;</code> or an <code>X-Hypotree-Token</code> header.</p>
</main></body></html>
"""


def _load_index() -> str:
    """The SPA shell, or a page that explains itself if it is not in the package."""
    path = Path(__file__).parent / "static" / "index.html"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_HTML


def choose_port(preferred: int = DEFAULT_PORT, host: str = "127.0.0.1") -> int:
    """First free port at or above ``preferred``, or a message naming what was tried."""
    tried = []
    for port in range(preferred, preferred + PORT_PROBE_RANGE):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if _PROBE_REUSES_ADDRESS:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                tried.append(port)
    raise OSError(f"no free port on {host} in {tried[0]}..{tried[-1]}")


class _Request:
    """A parsed request. Only what the routes actually use."""

    __slots__ = ("method", "path", "query", "headers", "body")

    def __init__(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body

    def one(self, key: str) -> str | None:
        values = self.query.get(key)
        return values[0] if values else None


class DashboardServer:
    """Serves one belief state, read-only, to localhost.

    ``writer`` is optional and is the only route to a mutation. When the
    dashboard runs beside a live MCP server it is a callable that applies a
    scheduling directive under the engine's write lock; in standalone viewer mode
    it is absent and the directive endpoint reports that plainly rather than
    pretending to have worked.
    """

    def __init__(
        self,
        db_path: Any,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        token: str | None = None,
        writer: Callable[[str, str, str], Awaitable[dict[str, Any]]] | None = None,
        html: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self._read = ReadModel(db_path)
        self._writer = writer
        self._html = html if html is not None else _load_index()
        self._server: asyncio.AbstractServer | None = None
        self._subscribers: set[asyncio.Queue[int]] = set()
        self._notifier: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?t={self.token}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        self._notifier = asyncio.create_task(self._watch_revision())

    async def stop(self) -> None:
        if self._notifier is not None:
            self._notifier.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._notifier
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._read.close()

    # -- change notification ---------------------------------------------------

    def notify(self, revision: int) -> None:
        """Fan a revision out to every listener, dropping any that cannot keep up.

        A slow reader must never apply backpressure to the agent. Dropping is
        safe precisely because the payload is a bare integer and the client
        refetches: a subscriber that misses ten revisions and catches the
        eleventh is already showing the right thing.
        """
        for queue in list(self._subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(revision)

    async def _watch_revision(self) -> None:
        """Poll the revision counter so changes from *any* writer are seen.

        The in-process engine could call `notify` directly, but a second client
        on the same database is a supported configuration and its writes would
        then be invisible. One `SELECT MAX(seq)` a second against a WAL reader is
        far cheaper than the graph rebuild it triggers, and it makes the observer
        correct regardless of who wrote.
        """
        last = self._read.revision()
        while True:
            await asyncio.sleep(1.0)
            current = self._read.revision()
            if current != last:
                last = current
                self.notify(current)

    # -- HTTP ------------------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await self._parse(reader)
            if request is None:
                await self._send(writer, 400, {"error": "malformed request"})
                return
            if not self._host_ok(request):
                # Rejecting an unexpected Host is the DNS-rebinding defence: a
                # page on evil.test that re-resolves to 127.0.0.1 still sends its
                # own name here, and a token in the URL cannot save us once the
                # browser treats the response as same-origin.
                await self._send(writer, 403, {"error": "host not allowed"})
                return
            if not self._origin_ok(request):
                await self._send(writer, 403, {"error": "cross-origin requests are refused"})
                return
            await self._route(request, writer)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    def _host_ok(self, request: _Request) -> bool:
        host = request.headers.get("host", "")
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name in _ALLOWED_HOSTS

    def _origin_ok(self, request: _Request) -> bool:
        """Accept a same-origin request, refuse a cross-origin one.

        Browsers attach `Origin` to same-origin POSTs too, so rejecting the
        header's mere presence rejected the page's own writes — the dashboard
        could not use its own pin button. What has to be refused is an origin
        that is not ours, which is the CSRF case; an absent header (a plain GET,
        or a non-browser client) is not a browser form post and is left to the
        token and the Host check.
        """
        origin = request.headers.get("origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        try:
            port = parsed.port or 80
        except ValueError:
            # A port that is not a number cannot match ours, and asking for it
            # again elsewhere would raise.
            return False
        return (
            parsed.scheme == "http"
            and (parsed.hostname or "") in _ALLOWED_HOSTS
            and port == self.port
        )

    def _authorised(self, request: _Request) -> bool:
        supplied = request.headers.get("x-hypotree-token") or request.one("t") or ""
        return secrets.compare_digest(supplied, self.token)

    async def _parse(self, reader: asyncio.StreamReader) -> _Request | None:
        try:
            line = await reader.readuntil(b"\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            return None
        if len(line) > _MAX_REQUEST_LINE:
            return None
        parts = line.decode("latin-1").strip().split()
        if len(parts) != 3:
            return None
        method, target, _version = parts

        headers: dict[str, str] = {}
        total = 0
        while True:
            try:
                raw = await reader.readuntil(b"\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
                return None
            total += len(raw)
            if raw == b"\r\n":
                break
            if len(headers) >= _MAX_HEADERS or total > _MAX_HEADER_BYTES:
                return None
            name, _, value = raw.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()

        body = b""
        declared = headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                return None
            if length < 0 or length > _MAX_BODY_BYTES:
                return None
            with contextlib.suppress(asyncio.IncompleteReadError):
                body = await reader.readexactly(length)

        split = urlsplit(target)
        return _Request(method, split.path, parse_qs(split.query), headers, body)

    async def _route(self, request: _Request, writer: asyncio.StreamWriter) -> None:
        path = request.path
        if path == "/" and request.method == "GET":
            await self._send_html(writer, self._html.replace("__TOKEN__", self.token))
            return
        if path.startswith("/static/") and request.method == "GET":
            # A dict lookup, not a path join. Traversal is unrepresentable rather
            # than filtered, and the assets are public anyway: the page needs
            # them before it has a token to send.
            asset = ASSETS.get(path[len("/static/") :])
            if asset is None:
                await self._send(writer, 404, {"error": "not found"})
                return
            await self._send_asset(writer, *asset)
            return
        if not path.startswith("/api/"):
            await self._send(writer, 404, {"error": "not found"})
            return
        if not self._authorised(request):
            await self._send(writer, 401, {"error": "missing or invalid token"})
            return
        if path == "/api/events" and request.method == "GET":
            await self._stream_events(writer)
            return
        try:
            status, payload = await self._dispatch(request)
        except ValueError as exc:
            status, payload = 400, {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a viewer must not take the agent down
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        await self._send(writer, status, payload)

    async def _dispatch(self, request: _Request) -> tuple[int, dict[str, Any]]:
        path, method = request.path, request.method
        goal_id = request.one("goal_id") or None

        if method == "GET":
            if path == "/api/meta":
                return 200, {**self._read.meta(), "missing_assets": missing_scripts()}
            if path == "/api/graph":
                return 200, self._read.graph(goal_id, request.one("at")).to_json()
            if path.startswith("/api/node/"):
                detail = self._read.node_detail(path[len("/api/node/") :])
                return (200, detail) if detail is not None else (404, {"error": "no such node"})
            if path == "/api/frontier":
                return 200, self._read.frontier(goal_id, _int(request.one("k"), 5))
            if path == "/api/learning-path":
                return 200, self._read.learning_path(
                    goal_id,
                    _int(request.one("limit"), 200),
                    request.one("at"),
                    request.one("since"),
                )
            if path == "/api/timeline":
                return 200, self._read.timeline(goal_id)
            return 404, {"error": "not found"}

        if method == "POST" and path == "/api/directive":
            if self._writer is None:
                return 503, {"error": "read-only viewer: no engine is attached to write through"}
            body = json.loads(request.body or b"{}")
            node_id = str(body.get("node_id") or "")
            mode = str(body.get("mode") or "")
            if not node_id or mode not in ("pin", "suspend", "clear"):
                return 400, {"error": "node_id and mode ('pin'|'suspend'|'clear') are required"}
            return 200, await self._writer(node_id, mode, str(body.get("reason") or ""))

        return 405, {"error": f"{method} not allowed on {path}"}

    async def _stream_events(self, writer: asyncio.StreamWriter) -> None:
        """One SSE stream carrying revision numbers, never payloads.

        Pushing whole graphs would make every step of the agent's work an
        unbounded serialisation on a socket nobody is necessarily reading. An
        integer plus a refetch keeps the stream trivial and lets a client that
        fell behind skip straight to current.
        """
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=8)
        self._subscribers.add(queue)
        try:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Cache-Control: no-store\r\nConnection: close\r\n"
                b"X-Content-Type-Options: nosniff\r\n\r\n"
            )
            writer.write(f"data: {json.dumps({'revision': self._read.revision()})}\n\n".encode())
            await writer.drain()
            while True:
                try:
                    revision = await asyncio.wait_for(queue.get(), timeout=15.0)
                    frame = f"data: {json.dumps({'revision': revision})}\n\n".encode()
                except asyncio.TimeoutError:
                    # `asyncio.TimeoutError` only became an alias of the builtin
                    # in 3.11. Catching the builtin here meant that on 3.10 the
                    # keepalive never fired: the stream task died on the first
                    # quiet fifteen seconds and the page sat in a reconnect loop,
                    # which looks exactly like a flaky network.
                    #
                    # A comment frame. Keeps idle sockets and proxies from
                    # deciding a quiet search means a dead connection.
                    frame = b": keepalive\n\n"
                writer.write(frame)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            self._subscribers.discard(queue)

    async def _send(self, writer: asyncio.StreamWriter, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode()
        writer.write(
            f"HTTP/1.1 {status} {_REASONS.get(status, 'OK')}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"X-Content-Type-Options: nosniff\r\n"
            f"Connection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()

    async def _send_html(self, writer: asyncio.StreamWriter, html: str) -> None:
        body = html.encode()
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"X-Content-Type-Options: nosniff\r\n"
            f"Content-Security-Policy: {_CSP}\r\n"
            f"Connection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()

    async def _send_asset(
        self, writer: asyncio.StreamWriter, content_type: str, body: bytes
    ) -> None:
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"X-Content-Type-Options: nosniff\r\n"
            f"Cache-Control: max-age=86400\r\n"
            f"Connection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()


_REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


def _int(raw: str | None, default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"expected an integer, got {raw!r}") from exc


__all__ = ["DEFAULT_PORT", "DashboardServer", "choose_port"]
