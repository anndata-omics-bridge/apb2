from __future__ import annotations

import importlib


def test_package_is_importable() -> None:
    """The generated distribution exposes its configured import package."""
    assert importlib.import_module("apb2") is not None
