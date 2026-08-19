"""Contracts for loading and composing vendor parsing-rule documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from apb2.vendor_parse_rules.rules import load_document


def _document_payload() -> dict[str, object]:
    return {
        "schema_version": "0.2",
        "file_version": "2",
        "software_name": "Test",
        "software_version_pattern": "^1$",
        "input": {"shape": "long"},
        "base": {
            "axis": {
                "obs_keys": ["sample"],
                "var_keys": ["feature"],
                "x_layer": "quantity",
            },
            "columns": {"obs": {"select": {"sample": "Sample"}}},
            "layers": [{"name": "quantity", "source": "Quantity"}],
        },
        "levels": {
            "ion": {
                "columns": {"var": {"select": {"feature": "Feature"}}},
            }
        },
    }


def _write_document(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_document_retains_only_its_validated_shell(tmp_path: Path) -> None:
    path = _write_document(tmp_path, _document_payload())

    document = load_document(path)

    assert vars(document).keys() == {"_shell"}
    assert document.path == path
    assert document.software_name == "Test"
    assert document.software_version_pattern == "^1$"
    assert document.levels == ("ion",)


def test_falsey_invalid_level_block_reaches_rule_validation(tmp_path: Path) -> None:
    payload = _document_payload()
    levels = payload["levels"]
    assert isinstance(levels, dict)
    ion = levels["ion"]
    assert isinstance(ion, dict)
    ion["axis"] = 0
    document = load_document(_write_document(tmp_path, payload))

    with pytest.raises(ValidationError, match="axis"):
        document.declared("ion")


def test_unknown_nested_column_block_reaches_rule_validation(tmp_path: Path) -> None:
    payload = _document_payload()
    levels = payload["levels"]
    assert isinstance(levels, dict)
    ion = levels["ion"]
    assert isinstance(ion, dict)
    columns = ion["columns"]
    assert isinstance(columns, dict)
    columns["unknown_axis"] = {}
    document = load_document(_write_document(tmp_path, payload))

    with pytest.raises(ValidationError, match="unknown_axis"):
        document.declared("ion")
