"""Documentation-only JSON projection of one resolved level plan."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass

from apb2.parserV2.parse_quant.axis_columns import (
    CoalesceColumn,
    JoinNonemptyColumn,
    ProformaFragmentColumn,
    ProformaIonColumn,
)
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    PlainSequenceStripper,
    SequenceColumn,
    SiteListNormalizer,
    TokenRegexNormalizer,
    TokenRegexStripper,
)
from apb2.parserV2.parse_quant.parameters.level import JsonValue

_COMPUTATIONS: dict[type, str] = {
    CoalesceColumn: "coalesce",
    JoinNonemptyColumn: "join_nonempty",
    ProformaIonColumn: "proforma_ion",
    ProformaFragmentColumn: "proforma_fragment",
    TokenRegexNormalizer: "token_regex",
    SiteListNormalizer: "site_list",
    EmbeddedSiteListNormalizer: "embedded_site_list",
}

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
    if isinstance(value, SequenceColumn | TokenRegexStripper):
        return as_json_value(_sequence_snapshot(value))
    if isinstance(value, PlainSequenceStripper):
        return {
            "kind": "stripped_sequence",
            "name": value.name,
            "inputs": list(value.inputs),
            "syntax": {"kind": "plain_sequence"},
        }
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, JsonValue] = {}
        if type(value) in _COMPUTATIONS:
            result["kind"] = _COMPUTATIONS[type(value)]
        result.update(
            {field.name: as_json_value(getattr(value, field.name)) for field in fields(value)}
        )
        return result
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


def _sequence_snapshot(value: SequenceColumn | TokenRegexStripper) -> dict[str, object]:
    """Document an executable sequence operation without another runtime record."""
    operation = value.operation if isinstance(value, SequenceColumn) else value
    payload: Mapping[str, object]
    if isinstance(
        operation, TokenRegexNormalizer | SiteListNormalizer | EmbeddedSiteListNormalizer
    ):
        kind, payload = "proforma_sequence", {"normalization": operation}
    elif isinstance(operation, TokenRegexStripper):
        syntax = {
            "kind": "token_regex",
            "token_pattern": operation.token_pattern,
            "token_position": operation.token_position,
        }
        kind, payload = "stripped_sequence", {"syntax": syntax}
    else:
        raise TypeError(f"sequence operation has no plan JSON form: {type(operation).__name__}")
    return {"kind": kind, "name": value.name, "inputs": value.inputs, **payload}


def _text(value: object) -> str:
    """Return a mapping key or set member as text."""
    if isinstance(value, str):
        return value
    raise TypeError(f"expected text, got a {type(value).__name__}: {value!r}")
