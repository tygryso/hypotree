"""Cross-platform behaviour of workspace identity resolution.

The engine ran on Linux only, and every place it assumed so was the same shape:
a POSIX path constant written inline instead of asked of the platform. These
tests pin the platform-dependent decisions by faking the platform, so a
Linux-only CI still catches a Windows regression.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from hypotree.store.identity import (
    _canonical_path_key,
    _is_test_environment,
    _validate_name,
    data_home,
    resolve_workspace_id,
    workspace_diagnostics,
)


@pytest.mark.unit
def test_xdg_override_wins_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The override is how tests and multi-instance setups redirect the store.

    Making it conditional on the platform would mean it silently stopped working
    on one of them, which is worse than not supporting it at all.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    for platform in ("linux", "darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        assert data_home() == tmp_path


@pytest.mark.unit
def test_windows_uses_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`~/.local/share` is a POSIX convention that means nothing on Windows."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))

    assert data_home() == tmp_path / "AppData" / "Local"


@pytest.mark.unit
def test_windows_without_localappdata_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stripped environment must not take the server down at startup."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    assert data_home() == Path.home() / "AppData" / "Local"


@pytest.mark.unit
def test_posix_keeps_its_existing_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing this would orphan every belief state already on disk."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    assert data_home() == Path.home() / ".local" / "share"


@pytest.mark.unit
def test_the_temp_root_is_asked_of_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardcoding `/tmp` suppressed nothing on Windows, where it is `%TEMP%`.

    Every test run there would have appended to the real diagnostic log.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    monkeypatch.setenv("XDG_DATA_HOME", str(Path(tempfile.gettempdir()) / "ht-check"))
    assert _is_test_environment() is True

    monkeypatch.setenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    assert _is_test_environment() is False


@pytest.mark.unit
@pytest.mark.parametrize("name", ["con", "prn", "aux", "nul", "com1", "lpt9", "AUX.db"])
def test_windows_reserved_device_names_are_rejected(name: str) -> None:
    """A workspace name becomes a directory, and Windows refuses these.

    Accepting one produces an OSError at server startup on Windows and nowhere
    else \u2014 a bug that only appears when a colleague opens the project.
    """
    assert _validate_name(name) is None


@pytest.mark.unit
def test_a_trailing_dot_is_rejected() -> None:
    """Windows strips it, so `proj.` and `proj` would merge into one belief state."""
    assert _validate_name("proj.") is None
    assert _validate_name("proj") == "proj"


@pytest.mark.unit
def test_ordinary_names_still_validate() -> None:
    assert _validate_name("my-project") == "my-project"
    assert _validate_name("  My_Project.v2  ") == "my_project.v2"
    assert _validate_name("console") == "console"  # not reserved; only exact `con` is


@pytest.mark.unit
def test_path_identity_is_case_folded_on_windows_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two spellings of one directory must not become two belief states.

    On Windows a client may launch with `C:\\proj` and another with `c:/proj`;
    hashing the raw string gives two ids and the project silently acquires two
    belief states. Folding elsewhere would change the id of every existing
    path-derived workspace and orphan it, for almost no benefit.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    key = _canonical_path_key(tmp_path)
    assert key == key.lower()
    assert "\\" not in key

    for platform in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        assert _canonical_path_key(tmp_path) == str(tmp_path.resolve())


@pytest.mark.unit
def test_git_subprocesses_are_windowless_only_on_windows() -> None:
    """A GUI MCP client on Windows flashes a console for every git call.

    The flag itself only exists on Windows, so it must be selected at import
    time rather than passed unconditionally.
    """
    from hypotree.store.identity import _CREATION_FLAGS

    assert bool(_CREATION_FLAGS) == (sys.platform == "win32")
    assert hasattr(subprocess, "CREATE_NO_WINDOW") == (sys.platform == "win32")


@pytest.mark.unit
def test_diagnostics_report_which_layer_resolved_the_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The answer to "why is my belief state empty?" is always the layer."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "pinned-name")

    wid, source, detail = resolve_workspace_id(tmp_path)
    assert (wid, source) == ("pinned-name", "env")

    info = workspace_diagnostics(tmp_path)
    assert info["workspace_id"] == "pinned-name"
    assert info["resolved_from"] == "env"
    assert info["warnings"] == []


@pytest.mark.unit
def test_diagnostics_warn_when_identity_is_path_derived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The weakest layer is the one that silently changes when a project moves."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)

    info = workspace_diagnostics(tmp_path)

    assert info["resolved_from"] == "path"
    assert any("moves" in w for w in info["warnings"])


@pytest.mark.unit
def test_diagnostics_do_not_create_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Asking where the belief state is must not be what brings it into being."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)

    info = workspace_diagnostics(tmp_path)

    assert info["database_exists"] is False
    assert not Path(str(info["store_root"])).exists()
