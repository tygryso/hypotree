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


# An en dash, an em dash and a maths symbol. `≥` has no cp1252 slot at all, and
# Python before 3.15 opens text files in the locale codepage on Windows.
NON_ASCII_TEXT = "latency ≥ 200ms — the p99 tail, not the mean – measured cold"


@pytest.mark.unit
def test_plain_text_round_trips_regardless_of_the_locale_codepage(tmp_path: Path) -> None:
    """The failure that took out 21 tests on the first Windows CI run.

    Markdown written with no explicit encoding raises UnicodeEncodeError on
    Windows the moment it contains a character cp1252 lacks — which the
    generated task briefings do (`≥`). JSON writers escape non-ASCII and were
    never at risk; plain text is where this bites.
    """
    assert any(ord(c) > 127 for c in NON_ASCII_TEXT)
    with pytest.raises(UnicodeEncodeError):
        NON_ASCII_TEXT.encode("cp1252")

    path = tmp_path / "briefing.md"
    path.write_text(NON_ASCII_TEXT, encoding="utf-8")

    assert path.read_text(encoding="utf-8") == NON_ASCII_TEXT


@pytest.mark.unit
def test_the_packaged_agent_guide_is_readable() -> None:
    """It is full of arrows and set symbols, and an agent reads it at runtime.

    Read without an explicit encoding this raises on Windows the first time an
    agent asks for `hypotree://guide`.
    """
    from hypotree.mcp_server import _agent_guide

    guide = _agent_guide()

    assert "exclusion_group" in guide
    assert any(ord(c) > 127 for c in guide)


@pytest.mark.unit
def test_a_non_ascii_workspace_config_is_readable(tmp_path: Path) -> None:
    """A `hypotree.yaml` written as UTF-8 must not be read in the locale codepage."""
    from hypotree.store.identity import workspace_id

    (tmp_path / "hypotree.yaml").write_text(
        "# projekt: pomiar opóźnień ≥ 200ms\nworkspace_id: latency-probe\n",
        encoding="utf-8",
    )

    assert workspace_id(tmp_path) == "latency-probe"


@pytest.mark.unit
def test_every_text_file_operation_declares_its_encoding() -> None:
    """The rule, enforced rather than remembered.

    This class of bug is invisible on Linux and fatal on Windows, so a reviewer
    working on Linux cannot catch it by reading the diff. Cheaper to assert here
    than to rediscover it on the next Windows CI run.
    """
    import re

    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    pattern = re.compile(r"""(\.write_text\(|\.read_text\(\)|[^.\w]open\()""")

    for sub in ("src", "eval"):
        for path in sorted((root / sub).rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "urlopen" in line or not pattern.search(line):
                    continue
                # A call left open continues on the next line, where the
                # encoding argument lives.
                if "encoding=" in line or line.rstrip().endswith("("):
                    continue
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")

    assert not offenders, "text file operations without an explicit encoding:\n" + "\n".join(
        offenders
    )
