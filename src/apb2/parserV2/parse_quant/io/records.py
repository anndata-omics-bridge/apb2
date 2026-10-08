"""The summary and details every APB metadata record shares, checked without its producer.

A record sits at ``apb.<producer>``, or one key deeper when a producer keeps several named
records (``apb.catalog.<catalogue>``). Its ``summary`` lists display metrics; its ``details``
name the tables that hold the complete evidence. Everything else in a record is the
producer's payload and is not inspected here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.io.errors import InvalidResultError

_RECORD_FIELDS = frozenset({"schema_version", "provenance", "result", "summary", "details"})
_ENTRY_FIELDS = frozenset({"name", "label", "value", "unit", "status"})
_STATUSES = frozenset({"ok", "attention", "not_checked"})
_DETAIL_SLOTS = frozenset({"varm", "layers", "annotation_tables", "feature_relations"})


class RecordContract:
    """Reject malformed summaries and detail references in one root or level part."""

    __slots__ = ()

    def validate_part(self, part: Mapping[str, JsonValue], where: str, /) -> None:
        """Check every record of one part: ``{producer: record}`` or named records below it."""
        for producer, record in part.items():
            if not isinstance(record, dict):
                continue
            self._validate_record(record, f"{where} {producer}")
            for name, child in record.items():
                if name not in _RECORD_FIELDS and isinstance(child, dict):
                    self._validate_record(child, f"{where} {producer}.{name}")

    def _validate_record(self, record: Mapping[str, JsonValue], where: str) -> None:
        if "summary" in record:
            self._validate_summary(record["summary"], where)
        if "details" in record:
            self._validate_details(record["details"], where)

    def _validate_summary(self, summary: JsonValue, where: str) -> None:
        if not isinstance(summary, list):
            raise InvalidResultError(f"{where} summary is not a list")
        seen: set[tuple[str, str | None]] = set()
        for entry in summary:
            name, layer = self._validate_entry(entry, where)
            if (name, layer) in seen:
                raise InvalidResultError(f"{where} summary repeats metric {name!r} {layer or ''}")
            seen.add((name, layer))

    def _validate_entry(self, entry: JsonValue, where: str) -> tuple[str, str | None]:
        if not isinstance(entry, dict):
            raise InvalidResultError(f"{where} summary entry is not an object")
        fields = set(entry)
        if not fields >= _ENTRY_FIELDS or fields - _ENTRY_FIELDS - {"layer"}:
            raise InvalidResultError(
                f"{where} summary entry fields {sorted(fields)} are not "
                f"{sorted(_ENTRY_FIELDS)} plus an optional layer"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise InvalidResultError(f"{where} summary entry has no name")
        for text in ("label", "unit"):
            if not isinstance(entry[text], str):
                raise InvalidResultError(f"{where} summary {name!r} {text} is not text")
        layer = entry.get("layer")
        if "layer" in entry and not isinstance(layer, str):
            raise InvalidResultError(f"{where} summary {name!r} layer is not text")
        self._validate_value(entry["value"], entry["status"], f"{where} summary {name!r}")
        return name, layer if isinstance(layer, str) else None

    def _validate_value(self, value: JsonValue, status: JsonValue, where: str) -> None:
        if not isinstance(status, str) or status not in _STATUSES:
            raise InvalidResultError(f"{where} has unknown status {status!r}")
        if isinstance(value, dict | list):
            raise InvalidResultError(f"{where} value is not a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidResultError(f"{where} value is not finite; store null")
        if status == "not_checked" and value is not None:
            raise InvalidResultError(f"{where} is not_checked but has a value")

    def _validate_details(self, details: JsonValue, where: str) -> None:
        if not isinstance(details, list):
            raise InvalidResultError(f"{where} details is not a list")
        for reference in details:
            if (
                not isinstance(reference, dict)
                or set(reference) != {"slot", "name"}
                or not isinstance(reference["slot"], str)
                or reference["slot"] not in _DETAIL_SLOTS
                or not isinstance(reference["name"], str)
                or not reference["name"]
            ):
                raise InvalidResultError(
                    f"{where} details entry {reference!r} is not a slot in "
                    f"{sorted(_DETAIL_SLOTS)} and a name"
                )
