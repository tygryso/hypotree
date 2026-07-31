"""Tests for 4-layer project identity resolution.

Covers HYPOTREE_WORKSPACE_ID (name + path), hypotree.yaml config,
git walk-up, and cwd fallback. Also tests name validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypotree.store.identity import (
    _validate_name,
    resolve_project_path,
    workspace_id,
)

# -- name validation ------------------------------------------------------


@pytest.mark.unit
def test_validate_name_accepts_valid() -> None:
    assert _validate_name("hypotree") == "hypotree"
    assert _validate_name("ml-model-v2") == "ml-model-v2"
    assert _validate_name("project_123") == "project_123"
    assert _validate_name("my.project.name") == "my.project.name"
    # The generated run-id shape: tildes separate key~value pairs, and the
    # hyphen must still match itself rather than opening an ASCII range.
    assert _validate_name("v0.3.0_run-iteration~a") == "v0.3.0_run-iteration~a"


@pytest.mark.unit
def test_validate_name_rejects_invalid() -> None:
    assert _validate_name("") is None
    assert _validate_name("My Project") is None
    assert _validate_name("has space") is None
    assert _validate_name("has/slash") is None
    assert _validate_name("x" * 129) is None
    assert _validate_name("-starts-with-dash") is None
    assert _validate_name(".starts-with-dot") is None
    # Characters that fall inside `_`-to-`~` if the hyphen is misplaced in the
    # character class — the exact way this pattern broke once.
    assert _validate_name("has{brace") is None
    assert _validate_name("has|pipe") is None


# -- workspace_id: layer 1 (HYPOTREE_WORKSPACE_ID as name) --------------------


@pytest.mark.unit
def test_workspace_id_from_env_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HYPOTREE_WORKSPACE_ID as a name is used directly as workspace_id."""
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "my-cool-project")
    wid = workspace_id(tmp_path)
    assert wid == "my-cool-project"


@pytest.mark.unit
def test_workspace_id_env_name_uppercase_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uppercase env name is lowercased."""
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "MyProject")
    wid = workspace_id(tmp_path)
    assert wid == "myproject"


@pytest.mark.unit
def test_workspace_id_env_name_invalid_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid env name falls through to other layers."""
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "has space!")
    wid = workspace_id(tmp_path)
    assert wid != "has space!"
    assert len(wid) == 16


# -- workspace_id: layer 2 (hypotree.yaml) -------------------------------


@pytest.mark.unit
def test_workspace_id_from_yaml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hypotree.yaml 'project:' field is used as workspace_id."""
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    (tmp_path / "hypotree.yaml").write_text("project: yaml-project-name\n")
    wid = workspace_id(tmp_path)
    assert wid == "yaml-project-name"


@pytest.mark.unit
def test_workspace_id_from_yml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hypotree.yml (short extension) also works."""
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    (tmp_path / "hypotree.yml").write_text("project: short-ext\n")
    wid = workspace_id(tmp_path)
    assert wid == "short-ext"


@pytest.mark.unit
def test_workspace_id_yaml_ignored_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HYPOTREE_WORKSPACE_ID (name) takes priority over yaml."""
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", "env-wins")
    (tmp_path / "hypotree.yaml").write_text("project: yaml-loses\n")
    wid = workspace_id(tmp_path)
    assert wid == "env-wins"


@pytest.mark.unit
def test_workspace_id_yaml_invalid_name_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid project name in yaml falls through to git/path layers."""
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    (tmp_path / "hypotree.yaml").write_text("project: 'has space'\n")
    wid = workspace_id(tmp_path)
    assert wid != "has space"
    assert len(wid) == 16


# -- workspace_id: layers 3-4 (git + path) --------------------------------


@pytest.mark.unit
def test_workspace_id_git_remote_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Git remote produces a normalized hash."""
    import subprocess

    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:test/repo.git"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    wid_ssh = workspace_id(repo)

    subprocess.run(
        ["git", "remote", "set-url", "origin", "https://github.com/test/repo.git"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    wid_https = workspace_id(repo)

    assert wid_ssh == wid_https
    assert len(wid_ssh) == 16


@pytest.mark.unit
def test_workspace_id_path_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No git, no config → path-based hash."""
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    wid = workspace_id(tmp_path)
    assert len(wid) == 16


# -- resolve_project_path (path layers) -----------------------------------


@pytest.mark.unit
def test_resolve_project_path_from_env_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HYPOTREE_WORKSPACE_ID as a path is used as project root."""
    monkeypatch.setenv("HYPOTREE_WORKSPACE_ID", str(tmp_path))
    result = resolve_project_path()
    assert result == tmp_path


@pytest.mark.unit
def test_resolve_project_path_git_walk_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Git walk-up finds the repo root from a subdirectory."""
    repo = tmp_path / "myrepo"
    subdir = repo / "src" / "deep"
    subdir.mkdir(parents=True)
    (repo / ".git").mkdir()
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    monkeypatch.chdir(subdir)
    result = resolve_project_path()
    assert result == repo


@pytest.mark.unit
def test_resolve_project_path_cwd_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no git repo is found, falls back to cwd."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("HYPOTREE_WORKSPACE_ID", raising=False)
    monkeypatch.chdir(empty)
    result = resolve_project_path()
    assert result == empty
