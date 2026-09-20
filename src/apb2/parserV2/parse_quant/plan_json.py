"""Documentation-only JSON projection of one resolved level plan."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass

from apb2.parserV2.parse_quant.parameters.level import JsonValue

PLAN_JSON_KEY = "plan_json"
"""The provenance key the serialized plan is stored under, beside ``rule_json``."""


def resolved_plan_json(plan: Mapping[str, object]) -> str:
    """Serialize source-specific decisions into the documentation-only snapshot."""
    document = as_json_value(plan)
    if not isinstance(document, dict):
        raise TypeError("a resolved plan must serialize as an object")
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def as_json_value(value: object) -> JsonValue:
    """Return the JSON form of one plan value without interpreting it."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: as_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_text(key): as_json_value(item) for key, item in value.items()}
    if isinstance(value, Set):
        ordered: list[JsonValue] = []
        ordered.extend(sorted(_text(item) for item in value))
        return ordered
    if isinstance(value, Sequence):
        return [as_json_value(item) for item in value]
    raise TypeError(
        f"a resolved plan holds a {type(value).__name__}, which has no JSON form: {value!r}"
    )


def _text(value: object) -> str:
    """Return a mapping key or set member as text."""
    if isinstance(value, str):
        return value
    raise TypeError(f"expected text, got a {type(value).__name__}: {value!r}")
