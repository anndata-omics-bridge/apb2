"""The ``parse`` record one parsed level carries in ``uns``, exactly as it is stored."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from apb2.parserV2.parse_quant.data.parsed import JsonValue


@dataclass(frozen=True, slots=True)
class ConversionEvidence:
    """What parsing one level found worth reporting.

    ``single_valued`` and ``effectively_empty`` map each location, ``obs``, ``var`` or
    ``layers``, to the fields left out of the level there, with what each held.
    ``var_disagreements`` names the vendor columns whose repeated rows gave a feature
    different values, where the rule kept the first.
    """

    unknown_mod_tokens: tuple[str, ...]
    unreadable_numeric: Mapping[str, JsonValue]
    single_valued: Mapping[str, Mapping[str, JsonValue]]
    effectively_empty: Mapping[str, Mapping[str, JsonValue]]
    var_disagreements: Mapping[str, JsonValue]

    def record(self, provenance: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Provenance, complete evidence and its summary, in the stored shape."""
        return {
            "provenance": dict(provenance),
            "result": {
                "unknown_mod_tokens": list(self.unknown_mod_tokens),
                "layer_diagnostics": {"unreadable_numeric": dict(self.unreadable_numeric)},
                "uninformative_fields": {
                    "single_valued": _nested(self.single_valued),
                    "effectively_empty": _nested(self.effectively_empty),
                },
                "var_disagreements": dict(self.var_disagreements),
            },
            "summary": [
                self._count(
                    "unknown_modification_tokens",
                    "Unknown modification tokens",
                    len(self.unknown_mod_tokens),
                    "tokens",
                ),
                self._count(
                    "unreadable_cells",
                    "Unreadable numeric cells",
                    sum(
                        cast(int, cast(dict[str, JsonValue], evidence)["cell_count"])
                        for evidence in self.unreadable_numeric.values()
                    ),
                    "cells",
                ),
                self._count(
                    "single_valued_fields",
                    "Single-valued columns and layers left out",
                    sum(len(fields) for fields in self.single_valued.values()),
                    "fields",
                ),
                self._count(
                    "effectively_empty_fields",
                    "Empty columns and layers left out",
                    sum(len(fields) for fields in self.effectively_empty.values()),
                    "fields",
                ),
                self._count(
                    "var_disagreements",
                    "Var columns whose repeated rows disagree",
                    len(self.var_disagreements),
                    "columns",
                ),
            ],
        }

    @staticmethod
    def _count(name: str, label: str, value: int | None, unit: str) -> JsonValue:
        """A problem count: ``attention`` above zero, ``not_checked`` when not judged."""
        status = "not_checked" if value is None else "attention" if value else "ok"
        return {"name": name, "label": label, "value": value, "unit": unit, "status": status}


def _nested(fields: Mapping[str, Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    return {location: dict(found) for location, found in fields.items()}
