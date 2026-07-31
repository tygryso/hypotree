"""Tests for the diagnostic logging in identity.py.

Every identity resolution appends a timestamped line to logs.txt so that
workspace-id mismatches between MCP clients can be diagnosed post-hoc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.store.identity import _is_test_environment, _log, _log_path, workspace_id


@pytest.mark.unit
def test_log_appends_timestamped_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_log writes an ISO-timestamped line to logs.txt.

    The _is_test_environment guard normally suppresses logging during tests.
    We disable the guard here so the log write can be verified.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("hypotree.store.identity._is_test_environment", lambda: False)
    _log("test diagnostic message")
    log_file = _log_path()
    assert log_file.exists()
    content = log_file.read_text()
    assert "test diagnostic message" in content
    # Each line starts with an ISO timestamp
    first_line = content.strip().split("\n")[0]
    assert "T" in first_line  # ISO format contains T separator


@pytest.mark.unit
def test_log_suppressed_in_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_log is a no-op when _is_test_environment returns True."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("hypotree.store.identity._is_test_environment", lambda: True)
    _log("this should not be written")
    log_file = _log_path()
    assert not log_file.exists()


@pytest.mark.unit
def test_is_test_environment_detects_pytest() -> None:
    """PYTEST_CURRENT_TEST env var triggers test-environment detection."""
    import os

    old = os.environ.get("PYTEST_CURRENT_TEST")
    os.environ["PYTEST_CURRENT_TEST"] = "test_something (call)"
    try:
        assert _is_test_environment() is True
    finally:
        if old is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = old


@pytest.mark.unit
def test_log_never_raises_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_log silently swallows errors (e.g. read-only filesystem)."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "nonexistent" / "nested"))
    # Even if the parent can't be created in some edge case, _log must not raise
    _log("should not crash")
    # No assertion on the file — the point is it doesn't raise


@pytest.mark.unit
def test_workspace_id_logs_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """workspace_id writes a diagnostic line showing the resolution path."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("hypotree.store.identity._is_test_environment", lambda: False)
    wid = workspace_id(tmp_path)
    log_file = _log_path()
    assert log_file.exists()
    content = log_file.read_text()
    assert f"workspace_id={wid}" in content
    assert "path fallback" in content  # tmp_path has no git repo
    assert "WARNING" in content  # path-based identity warns about fragility


@pytest.mark.unit
def test_workspace_id_logs_remote_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a git repo with a remote exists, the log records the remote URL."""
    import subprocess

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("hypotree.store.identity._is_test_environment", lambda: False)
    repo = tmp_path / "myrepo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:test/repo.git"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    wid = workspace_id(repo)
    log_file = _log_path()
    content = log_file.read_text()
    assert f"workspace_id={wid}" in content
    assert "git@github.com:test/repo.git" in content
    assert "from remote" in content
