"""Smoke test: verify the package imports and the version is set."""

import pytest

import hypotree


@pytest.mark.unit
def test_package_importable() -> None:
    """The package must import cleanly with its declared dependencies installed."""
    assert hypotree.__version__ == "0.6.0"
