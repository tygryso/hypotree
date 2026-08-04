"""Vendored browser assets, served from the package.

Everything the interface needs ships in the wheel. No CDN, for three reasons
that all matter more than the 276 KB: a tool that breaks on a plane or behind a
corporate proxy is not finished, the paying segment is frequently air-gapped,
and a third-party script tag is write access to the page rendering someone's
proprietary hypotheses — the supply-chain integrity failure in OWASP A08.

Loaded once at import into an in-memory table keyed by name. Nothing here joins a
path, so directory traversal is not filtered, it is unrepresentable.

| file                    | licence | why                                       |
|-------------------------|---------|-------------------------------------------|
| vue.global.prod.js      | MIT     | reactivity and `<transition-group>`       |
| d3-selection, d3-zoom   | ISC     | hardware-accelerated pan/zoom             |
| d3-{drag,dispatch,...}  | ISC     | peer modules d3-zoom's UMD build requires |
| marked.min.js           | MIT     | the learning path renders as markdown     |

d3 micromodules rather than the full build: 63 KB against 273 KB, and the layout
is computed server-side so nothing else in d3 is wanted.
"""

from __future__ import annotations

from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"

# Order matters: d3-zoom's UMD build resolves its peers off the global at load.
VENDOR_SCRIPTS: tuple[str, ...] = (
    "vue.global.prod.js",
    "d3-dispatch.min.js",
    "d3-selection.min.js",
    "d3-timer.min.js",
    "d3-color.min.js",
    "d3-interpolate.min.js",
    "d3-ease.min.js",
    "d3-transition.min.js",
    "d3-drag.min.js",
    "d3-zoom.min.js",
    "marked.min.js",
)

_CONTENT_TYPES = {
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def _load() -> dict[str, tuple[str, bytes]]:
    assets: dict[str, tuple[str, bytes]] = {}
    if not _STATIC_DIR.is_dir():
        return assets
    for path in sorted(_STATIC_DIR.iterdir()):
        if path.is_file() and path.suffix in _CONTENT_TYPES:
            assets[path.name] = (_CONTENT_TYPES[path.suffix], path.read_bytes())
    return assets


ASSETS: dict[str, tuple[str, bytes]] = _load()


def missing_scripts() -> list[str]:
    """Vendored files the wheel should carry and does not.

    A source checkout without them still serves a working page — degraded, and
    saying so — rather than a blank screen and a console full of 404s.
    """
    return [name for name in VENDOR_SCRIPTS if name not in ASSETS]


__all__ = ["ASSETS", "VENDOR_SCRIPTS", "missing_scripts"]
