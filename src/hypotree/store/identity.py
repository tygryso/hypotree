"""Workspace identity — 4-layer project resolution.

Resolves which project the MCP server should operate on and derives a stable
workspace_id for the SQLite belief-state store:

1. **HYPOTREE_WORKSPACE_ID env var** — explicit human-chosen name (no hashing). The
   value is validated and used directly as the workspace directory name.
2. **hypotree.{yaml,yml} config file** — per-repo config checked into the
   consumer repo. The ``workspace_id:`` field is read and used as a literal
   workspace name.
3. **Git walk-up from cwd** — walks up from the process cwd until ``.git`` is
   found; that root is used for git-remote-hash resolution.
4. **cwd fallback** — path-hash of the current directory (weakest; last resort).

Layers 1-2 are name-based (human-chosen, explicit). Layers 3-4 are
discovery-based (automatic, hash-based).

Every path this module produces hangs off :func:`data_home`, which is the one
place that knows where the platform keeps per-user state: ``XDG_DATA_HOME`` when
set, ``%LOCALAPPDATA%`` on Windows, ``~/.local/share`` otherwise.

A diagnostic log at ``<data_home>/mcp_hypotree/logs.txt`` records every identity
resolution so mismatches between MCP clients can be diagnosed post-hoc.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import yaml

# Pattern for valid workspace names: lowercase alphanumeric, hyphens,
# underscores, dots, tildes. No spaces, no slashes, max 128 chars.
# The hyphen stays last in the class: written between `_` and `~` it is a range
# spanning most of printable ASCII and, worse, stops matching itself.
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9._~-]{0,127}$")

# Names Windows refuses for a file or directory, at any extension. A workspace
# name becomes a directory, so accepting one would raise OSError at server
# startup on Windows and nowhere else — the worst possible shape for a bug.
# Rejected on every platform so a workspace name stays portable rather than
# working until someone opens the project on a different machine.
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

# Suppresses the console window Windows opens for each git subprocess: under a
# GUI MCP client that is a visible flash on every identity resolution. Passed
# unconditionally because `creationflags=0` is accepted on POSIX — only a
# non-zero value raises there — so both platforms keep one call shape. Read via
# getattr: the constant does not exist in the POSIX subprocess module.
_CREATION_FLAGS: int = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def data_home() -> Path:
    """Base directory for hypotree's persistent state.

    Single source of truth: the diagnostic log, the belief-state store and the
    eval harness all derive from this. The rule previously lived in three
    places, and one of them was always the copy that had not been updated.

    ``XDG_DATA_HOME`` wins everywhere, Windows included — it is how the tests and
    anyone running several isolated instances redirect the store, and making the
    override conditional on the platform would mean it silently stopped working
    on one of them.

    Windows falls back to ``%LOCALAPPDATA%``, the per-user, non-roamed location
    the platform intends for exactly this. Everything else keeps
    ``~/.local/share``: macOS would arguably prefer
    ``~/Library/Application Support``, but moving it now would orphan every
    existing belief state on that platform in exchange for tidiness.
    """
    override = os.environ.get("XDG_DATA_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) if local else Path.home() / "AppData" / "Local"
    return Path.home() / ".local" / "share"


def _is_test_environment() -> bool:
    """Detect whether we are running inside pytest or a test harness.

    Pytest sets PYTEST_CURRENT_TEST. Test fixtures also point XDG_DATA_HOME at a
    temporary directory, and those paths are meaningless in a production
    diagnostic log. The temp root is asked of the platform rather than assumed
    to be ``/tmp``: on Windows it is ``%TEMP%``, so a hardcoded check suppressed
    nothing there and every test run wrote to the real log.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    base = os.environ.get("XDG_DATA_HOME", "")
    if base:
        try:
            return Path(base).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        except (ValueError, OSError):
            return False
    return False


def _log_path() -> Path:
    """Return the diagnostic log location (one shared file for all workspaces)."""
    return data_home() / "mcp_hypotree" / "logs.txt"


def _log(message: str) -> None:
    """Append a timestamped diagnostic line to the shared log file.

    Suppressed inside test environments to avoid polluting the production log
    with ephemeral temp paths.

    Best-effort: if the write fails (e.g. read-only filesystem), the error is
    silently swallowed so it never breaks MCP server startup.
    """
    if _is_test_environment():
        return
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        # Explicit encoding: the default is the locale codepage on Windows, so a
        # project path holding one non-ASCII character raises UnicodeEncodeError
        # inside the diagnostic logger itself.
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{ts} {message}\n")
    except Exception:
        pass


def _validate_name(name: str) -> str | None:
    """Validate a human-chosen workspace name.

    Must be lowercase alphanumeric + hyphens/underscores/dots/tildes, 1-128
    chars, starting with an alphanumeric character, and usable as a directory
    name on every supported platform. Returns the validated name or None.
    """
    name = name.strip().lower()
    if not _VALID_NAME.match(name):
        return None
    # Windows silently strips a trailing dot, so `proj.` and `proj` would land
    # on the same directory and merge two belief states into one.
    if name.endswith("."):
        return None
    # Windows reserves these ahead of the extension: `aux` and `aux.db` both fail.
    if name.split(".")[0] in _WINDOWS_RESERVED:
        return None
    return name


def _looks_like_path(value: str) -> bool:
    """Heuristic: does this value look like a filesystem path rather than a name?

    If it contains ``/``, ``\\``, starts with ``~``, or has a drive-letter prefix
    (Windows), it's treated as a path (legacy behavior — resolved as a project
    root, then hashed).
    """
    return bool(
        "/" in value
        or "\\" in value
        or value.startswith("~")
        or re.match(r"^[a-zA-Z]:", value) is not None
    )


def _read_yaml_config(project_path: Path) -> str | None:
    """Read the ``workspace_id:`` field from a hypotree.yaml/yml config file.

    Searched in the given directory only (no walk-up — the caller has already
    resolved the project root). Returns the validated workspace name or None.
    """
    for ext in ("yaml", "yml"):
        config_file = project_path / f"hypotree.{ext}"
        if config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    name = data.get("workspace_id") or data.get("project")
                    if name:
                        validated = _validate_name(str(name))
                        if validated:
                            _log(
                                f"identity: workspace name '{validated}' from "
                                f"{config_file.name} (dir={project_path})"
                            )
                            return validated
                        _log(
                            f"identity: invalid workspace name '{name}' in "
                            f"{config_file} — must match {_VALID_NAME.pattern}"
                        )
            except Exception as exc:
                _log(f"identity: failed to read {config_file}: {exc}")
    return None


def _normalize_remote(url: str) -> str:
    """Normalize a git remote URL so SSH/HTTPS variants produce the same key."""
    url = url.strip().lower()
    url = re.sub(r"\.git$", "", url)
    if url.startswith("git@"):
        url = url.replace("git@", "", 1)
    url = re.sub(r"^[a-z]+://", "", url)  # strip scheme
    url = url.replace(":", "/", 1)  # scp-like host:path → host/path
    url = re.sub(r"/+", "/", url)
    return url.strip("/")


@cache
def _resolve_remote(project_path: Path) -> str | None:
    """Try ``origin``, then fall back to the first remote listed by git.

    Logs each attempt so that mismatches between MCP clients can be traced.

    Memoised for the life of the process: this is up to three git subprocesses
    with a five-second timeout each, and it sits on the server's startup path
    before the first handshake — on a machine where git is slow or a remote
    lives behind a hung mount, that delay is what an MCP client renders as a
    failed start. Re-pointing a remote mid-session deliberately does *not* move
    the belief state; migrating a running session to a different workspace is
    not something a `git remote set-url` should silently do. Call
    :func:`reset_identity_cache` to force re-resolution.
    """
    # Try `origin` first — the most common case.
    try:
        origin = subprocess.run(
            ["git", "-C", str(project_path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=_CREATION_FLAGS,
        ).stdout.strip()
        if origin:
            _log(f"identity: resolved origin remote: {origin} (path={project_path})")
            return origin
        _log(f"identity: no origin remote found (path={project_path})")
    except Exception as exc:
        _log(f"identity: git origin lookup failed: {exc} (path={project_path})")
        return None

    # No `origin` — try `git remote` and pick the first one.
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), "remote"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=_CREATION_FLAGS,
        )
        remotes = result.stdout.strip().splitlines()
        if remotes:
            first_name = remotes[0].strip()
            url = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project_path),
                    "config",
                    "--get",
                    f"remote.{first_name}.url",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                creationflags=_CREATION_FLAGS,
            ).stdout.strip()
            if url:
                _log(f"identity: resolved first-remote '{first_name}': {url} (path={project_path})")
                return url
            _log(f"identity: first-remote '{first_name}' has no URL (path={project_path})")
        else:
            _log(f"identity: no git remotes at all (path={project_path})")
    except Exception as exc:
        _log(f"identity: git remote list failed: {exc} (path={project_path})")

    return None


def _git_walk_up(start: Path) -> Path | None:
    """Walk up from ``start`` until a ``.git`` directory or file is found.

    Returns the directory containing ``.git``, or None if no repo is found.
    """
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_project_root() -> Path:
    """Resolve the project root directory using discovery layers 3-4.

    Resolution order:
    3. Git walk-up from cwd until ``.git`` is found.
    4. cwd fallback.

    Does NOT handle layers 1-2 (name-based) — those are handled by
    ``workspace_id()`` which may call this to find the root for config reading.
    """
    # Layer 3: git walk-up from cwd
    git_root = _git_walk_up(Path.cwd())
    if git_root:
        _log(f"identity: project root from git walk-up={git_root}")
        return git_root

    # Layer 4: cwd fallback
    _log(f"identity: project root from cwd fallback={Path.cwd()}")
    return Path.cwd()


def resolve_project_path() -> Path:
    """Resolve the project directory the MCP server should operate on.

    Full resolution including HYPOTREE_WORKSPACE_ID-as-path (legacy):
    1. HYPOTREE_WORKSPACE_ID as a path → use as project root.
    2. Git walk-up from cwd.
    3. cwd fallback.

    Name-based HYPOTREE_WORKSPACE_ID (layer 1) and hypotree.yaml (layer 2) are
    handled by ``workspace_id()``, not here — they produce a name, not a path.
    """
    hypotree_workspace_id = os.environ.get("HYPOTREE_WORKSPACE_ID")
    if hypotree_workspace_id and _looks_like_path(hypotree_workspace_id):
        _log(f"identity: project path from HYPOTREE_WORKSPACE_ID (path)={hypotree_workspace_id}")
        return Path(hypotree_workspace_id)

    return _resolve_project_root()


def workspace_id(project_path: Path) -> str:
    """Stable identifier for this project's belief state.

    See :func:`resolve_workspace_id` for the resolution order; this is the same
    thing without the provenance.
    """
    return resolve_workspace_id(project_path)[0]


def reset_identity_cache() -> None:
    """Forget the memoised git-remote lookups.

    Exists for callers that genuinely change the git state underneath a live
    process — tests, and anything simulating a second session. Ordinary code
    should not need it: within one session the workspace is meant to be fixed.
    """
    _resolve_remote.cache_clear()


def resolve_workspace_id(project_path: Path) -> tuple[str, str, str]:
    """Resolve the workspace id and report *which layer produced it*.

    Returns ``(workspace_id, source, detail)``.

    The provenance is not decoration. Every hard support question about a
    per-project belief state is the same one — "why is it empty?" / "why do my
    two clients disagree?" — and the answer is always that the two resolved
    through different layers. Returning the layer makes that self-diagnosable
    instead of a hunt through a log file whose location the user also has to
    work out.

    4-layer resolution:
    1. **HYPOTREE_WORKSPACE_ID env var** (name) — used directly as the workspace
       directory name. No hashing.
    2. **hypotree.{yaml,yml}** — ``workspace_id:`` field read from the config file
       in the project root. Used directly as the workspace name.
    3. **Git remote hash** — normalized git remote URL → SHA-256[:16].
    4. **Path hash** — canonical project path → SHA-256[:16] (weakest).
    """
    # Layer 1: HYPOTREE_WORKSPACE_ID as a name (not a path)
    hypotree_workspace_id = os.environ.get("HYPOTREE_WORKSPACE_ID")
    if hypotree_workspace_id and not _looks_like_path(hypotree_workspace_id):
        name = _validate_name(hypotree_workspace_id)
        if name:
            _log(f"identity: workspace_id='{name}' from HYPOTREE_WORKSPACE_ID (name)")
            return name, "env", "HYPOTREE_WORKSPACE_ID"
        _log(
            f"identity: HYPOTREE_WORKSPACE_ID='{hypotree_workspace_id}' is not a valid name "
            f"(must match {_VALID_NAME.pattern}), falling through"
        )

    # Layer 2: hypotree.yaml/yml config file
    config_name = _read_yaml_config(project_path)
    if config_name:
        _log(f"identity: workspace_id='{config_name}' from hypotree.yaml")
        return config_name, "config", "hypotree.yaml"

    # Layer 3-4: git remote hash
    remote = _resolve_remote(project_path)
    if remote:
        normalized = _normalize_remote(remote)
        wid = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        _log(
            f"identity: workspace_id={wid} from remote "
            f"(normalized='{normalized}', raw='{remote}', path={project_path})"
        )
        return wid, "git_remote", normalized

    # Layer 5: path fallback
    path_key = _canonical_path_key(project_path)
    wid = hashlib.sha256(path_key.encode()).hexdigest()[:16]
    _log(
        f"identity: workspace_id={wid} from path fallback "
        f"(key='{path_key}', path={project_path}) "
        f"WARNING: path-based identity is mount-path fragile"
    )
    return wid, "path", path_key


def workspace_diagnostics(project_path: Path) -> dict[str, object]:
    """Everything needed to answer "why is my belief state not the one I expect?".

    Reports which of the four layers actually produced the id, where the store
    landed, and the two conditions that silently split one project into two
    belief states: a path-derived id (changes if the project moves or is mounted
    differently) and a store on a network share (SQLite WAL needs shared-memory
    the share cannot provide).

    A pure read \u2014 it never creates the store directory, so asking the question
    cannot itself be what brings a workspace into existence.
    """
    wid, source, detail = resolve_workspace_id(project_path)
    root = data_home() / "mcp_hypotree" / wid
    db = root / "state.db"

    warnings: list[str] = []
    if source == "path":
        warnings.append(
            "Workspace id derived from the project path: it changes if the project "
            "moves, is cloned elsewhere, or is mounted differently. Set "
            "HYPOTREE_WORKSPACE_ID or add workspace_id: to hypotree.yaml to pin it."
        )
    root_str = str(root)
    # UNC spelling only, and only where it means what it looks like: a leading
    # `//` is legal on POSIX and would raise a warning about nothing.
    if sys.platform == "win32" and root_str.startswith("\\\\"):
        warnings.append(
            "Belief state is on a network share. SQLite WAL mode needs shared "
            "memory the share cannot provide and will fail or corrupt. Point "
            "XDG_DATA_HOME at a local disk."
        )

    return {
        "workspace_id": wid,
        "resolved_from": source,
        "resolved_detail": detail,
        "project_path": str(project_path),
        "data_home": str(data_home()),
        "store_root": root_str,
        "database": str(db),
        "database_exists": db.exists(),
        "platform": sys.platform,
        "diagnostic_log": str(_log_path()),
        "warnings": warnings,
    }


def _canonical_path_key(project_path: Path) -> str:
    """The string a path-derived workspace id is hashed from.

    On Windows, separators are normalised and case is folded: ``C:\\proj`` and
    ``c:/proj`` are the same directory spelled differently by two MCP clients,
    and hashing the raw string gives two ids — the project silently acquires two
    belief states that never see each other's evidence.

    Deliberately Windows-only. macOS is usually case-insensitive too, but
    ``Path.resolve()`` already returns the on-disk casing there, so folding
    would buy almost nothing while changing the id of every existing
    path-derived workspace and orphaning its belief state. POSIX paths are
    case-sensitive and must not be folded at all.
    """
    resolved = str(project_path.resolve())
    if sys.platform == "win32":
        return resolved.replace("\\", "/").lower()
    return resolved


def store_root(project_path: Path) -> Path:
    """Return the directory holding state.db for this project.

    Lives under :func:`data_home` rather than in the working tree so the belief
    state survives git branch switches and working-tree resets.
    """
    wid = workspace_id(project_path)
    root = data_home() / "mcp_hypotree" / wid
    root.mkdir(parents=True, exist_ok=True)
    _log(f"identity: store_root resolved to {root} (workspace_id={wid})")
    return root


def capture_git_context(project_path: Path) -> tuple[str | None, str | None]:
    """Best-effort capture of (context_hash, git_branch) from the working tree.

    Returns (None, None) when not inside a git repo or when git is unavailable.
    Never raises — the staleness flag is a best-effort convenience.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(project_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=_CREATION_FLAGS,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(project_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=_CREATION_FLAGS,
        ).stdout.strip()
        # Quantized hash of HEAD — never store the raw commit SHA in the
        # belief-state DB.
        context_hash = hashlib.sha256(head.encode()).hexdigest()[:16] if head else None
        return context_hash, (branch if branch else None)
    except Exception:
        return None, None
