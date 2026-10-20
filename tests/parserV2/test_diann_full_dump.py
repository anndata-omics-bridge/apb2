"""The DIA-NN rules are passive dumps: they convert every column of the committed report headers.

A rule converts what the software writes, independent of any consumer; meanings a consumer
needs belong in apb-catalog. When a DIA-NN version adds a column, this test fails until the
rule converts it or ``_NOT_CONVERTED`` names it with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parserV2.fixtures import DATA_DIR

_DOCUMENTS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "apb2"
    / "parserV2"
    / "vendor_parse_rules"
    / "documents"
)
_KEYS = ("diann/v1_7", "diann/v1_8", "diann/v2")
# Columns a rule cannot convert, with the reason; empty while every header column is converted.
_NOT_CONVERTED: dict[str, dict[str, str]] = {key: {} for key in _KEYS}


def _converted_sources(node: object) -> set[str]:
    """Every vendor column a rule document reads: sources, inputs, packed values and labels."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("source", "label_column") and isinstance(value, str):
                found.add(value)
            if key in ("inputs", "value_columns") and isinstance(value, list):
                found.update(item for item in value if isinstance(item, str))
            found |= _converted_sources(value)
    elif isinstance(node, list):
        for value in node:
            found |= _converted_sources(value)
    return found


@pytest.mark.parametrize("key", _KEYS)
def test_diann_rule_converts_every_header_column(key: str) -> None:
    rule = json.loads((_DOCUMENTS / key / "rules.json").read_text(encoding="utf-8"))
    header = (DATA_DIR / key / "header.txt").read_text(encoding="utf-8").splitlines()
    converted = _converted_sources(rule)
    missing = [column for column in header if column not in converted]
    assert missing == sorted(_NOT_CONVERTED[key], key=header.index)
