"""One resolved level plan as JSON text, so a written result records how it was produced.

The rule states what any source of one vendor may look like; the plan states what *this*
source turned out to be — which columns were projected and at which read dtype, which dialect
and number notation won, which optional layers this export could not provide, which vendor
modification tokens resolved to which accessions, and what the AnnData encoders were told.
None of that is recoverable from the rule afterwards, which is why the plan travels with the
output instead of staying in the process that produced it.

Text rather than a nested mapping, for three reasons: the ``rule_json`` sitting beside it in
the same provenance is already a string; one string needs no storage schema for the empty
tuples and sets a plan legitimately contains; and two runs of the same pipeline are then
comparable with a diff.

The record is complete — every field of the plan, including its own ``provenance``, which
therefore repeats the ``rule_json`` stored next to it. That duplication is deliberate: a
serialization that dropped a field would be a different value, and no reader could tell it
from a plan whose field was genuinely empty.

Nothing here knows a storage backend or a vendor. It converts the four shapes a plan is built
from — frozen dataclass, mapping, sequence, set — and refuses anything else rather than
inventing a representation for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass

from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
from apb2.parserV2.parse_quant.parameters.working import JsonValue

PLAN_JSON_KEY = "plan_json"
"""The provenance key the serialized plan is stored under, beside ``rule_json``."""


def resolved_plan_json(plan: ResolvedLevelPlan) -> str:
    """One resolved plan as a JSON object, field for field, with every set given an order."""
    return json.dumps(as_json_value(plan), ensure_ascii=False, allow_nan=False)


def as_json_value(value: object) -> JsonValue:
    """The JSON form of one plan value; raises rather than guessing at an unknown shape."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: as_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_text(key): as_json_value(item) for key, item in value.items()}
    if isinstance(value, Set):
        # A set has no order of its own, and a plan is meant to be diffable between runs.
        ordered: list[JsonValue] = []
        ordered.extend(sorted(_text(item) for item in value))
        return ordered
    if isinstance(value, Sequence):
        return [as_json_value(item) for item in value]
    raise TypeError(
        f"a resolved plan holds a {type(value).__name__}, which has no JSON form: {value!r}"
    )


def _text(value: object) -> str:
    """A mapping key or set member, which a plan only ever spells as text."""
    if isinstance(value, str):
        return value
    raise TypeError(f"expected text, got a {type(value).__name__}: {value!r}")
