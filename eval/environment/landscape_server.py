"""Pure Python HTTP landscape server — black-box R&D evaluation environment.

Uses stdlib http.server. No external dependencies. Listens on 127.0.0.1:8080.
Exposes POST /evaluate which takes {"c": "config_string", "depth": int} and
returns {"success": float, "metrics": {...}}.

The landscape is loaded from a seeded JSON config. The agent never sees this
file or the JSON — it only sees the HTTP response.

Sandbox protocol: after the harness starts this server, the source file is
deleted (rm) and restored via git restore after the run completes. The agent
has no read_file or execute_bash utilities, so it cannot inspect the server
logic or the landscape topology.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# The combinatorial scorer is shared with the generator so the two can never
# drift. The server is launched as a bare script (``python eval/environment/
# landscape_server.py``), which puts its own directory — not the repo root — on
# sys.path, so add the repo root before importing the eval package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.environment.landscape_scoring import (  # noqa: E402
    is_premise_probe,
    score_config,
)

# Global state — loaded once at startup from the landscape JSON file.
_LANDSCAPE: dict = {}
_PROBE_COUNT: dict[str, int] = {}


def _log(message: str) -> None:
    """Print a timestamped log line to stderr.

    All server logging goes to stderr so stdout is never polluted.
    """
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}", file=sys.stderr, flush=True)


def load_landscape(config_path: Path) -> None:
    """Load the landscape configuration from a JSON file."""
    global _LANDSCAPE
    _LANDSCAPE = json.loads(config_path.read_text(encoding="utf-8"))


def _evaluate(config: str, depth: int) -> dict:
    """Score a config string against the hidden landscape.

    All scoring — premise resolution, the synergy interaction, and the decoy's
    shallow mirage versus its hard failure at depth — lives in ``score_config``,
    so the server is a pure transport with no scoring rules of its own and
    cannot drift from the generator's ground truth.
    """
    seed = _LANDSCAPE.get("seed", 0)
    probe_key = f"{config}:{depth}"
    _PROBE_COUNT[probe_key] = _PROBE_COUNT.get(probe_key, 0) + 1

    return {
        "success": round(score_config(config, seed, depth), 4),
        "metrics": {
            "config": config,
            "depth": depth,
            # Echoing the probe mode makes the two scoring regimes legible to the
            # agent without revealing which values are correct.
            "probe_mode": "premise" if is_premise_probe(config) else "combination",
            "probe_number": _PROBE_COUNT[probe_key],
        },
    }


class LandscapeHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing POST /evaluate."""

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path != "/evaluate":
            _log(f"POST {self.path} → 404 (unknown path)")
            self.send_error(404, "Not found")
            return

        try:
            data = json.loads(body)
            config = data.get("c", "")
            depth = int(data.get("depth", 0))

            _log(f"POST /evaluate payload={data}")

            result = _evaluate(config, depth)
            response = json.dumps(result).encode()

            _log(f"POST /evaluate → 200 success={result['success']}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            error_msg = json.dumps({"error": str(e)}).encode()
            _log(f"POST /evaluate → 400 error={e}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)

    def do_GET(self) -> None:
        """Health check endpoint."""
        if self.path == "/health":
            _log("GET /health → 200")
            resp = json.dumps({"status": "ok", "seed": _LANDSCAPE.get("seed")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            _log(f"GET {self.path} → 404")
            self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        """Suppress default http.server logging — custom _log handles all output."""
        pass


def main() -> None:
    """Start the landscape server.

    Usage: python landscape_server.py <path_to_landscape.json> [port]
    """
    if len(sys.argv) < 2:
        print("Usage: python landscape_server.py <landscape.json> [port]", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    load_landscape(config_path)
    _log(
        f"server starting on http://127.0.0.1:{port} "
        f"(seed={_LANDSCAPE.get('seed')}, "
        f"nodes={len(_LANDSCAPE.get('nodes', []))})"
    )

    server = HTTPServer(("127.0.0.1", port), LandscapeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
