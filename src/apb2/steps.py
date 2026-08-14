"""Plan steps that adapt reused legacy runtime strategies to the V2 parser contracts.

Each factory here consumes rule-schema objects at construction time — translating them is
construction's purpose — and returns a strategy holding only plain values and behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

import pandas as pd

from apb2.convert.checks import check_layer_occupancy
from apb2.convert.conversion import Conversion, make_conversion
from apb2.convert.preprocess import (
    FragmentExploder as LegacyFragmentExploder,
)
from apb2.convert.preprocess import (
    make_fragment_exploder,
)
from apb2.identity import NoFragments
from apb2.result import ParsedData
from apb2.serialization import JsonValue
from apb2.vendor_parse_rules.model import LongRule, WideRule


@dataclass(frozen=True, slots=True)
class TrimmedExplode:
    """Trim the frame to the columns the rule reads, then run the legacy explode.

    The trim exists only because the explode multiplies the row count ~12x; it mirrors the
    legacy ``_columns_read_by`` set, precomputed here at construction.
    """

    exploder: LegacyFragmentExploder
    needed: frozenset[str]

    def packed_columns(self) -> tuple[str, ...]:
        """The packed source columns the read must supply for this explode."""
        return self.exploder.packed_columns()

    def explode(self, table: pd.DataFrame) -> pd.DataFrame:
        keep = [column for column in table.columns if column in self.needed]
        return self.exploder.explode(table, keep)


def make_fragment_step(
    rule: LongRule | WideRule,
    modification_sources: frozenset[str],
) -> TrimmedExplode | NoFragments:
    """Return the configured explode for a fragment rule, or the identity step."""
    exploder = make_fragment_exploder(rule)
    packed = exploder.packed_columns()
    if not packed:
        return NoFragments()
    needed: set[str] = set(modification_sources) | set(packed)
    for _axis, group in rule.named_column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
    needed.update(layer.source for layer in rule.layers)
    return TrimmedExplode(exploder, frozenset(needed))


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Run the reused long/wide conversion and shape its pieces into ``ParsedData``.

    ``strict`` is the caller's error-severity policy for non-``X`` layer-contract
    findings, mirroring the legacy ``convert_table`` keyword.
    """

    conversion: Conversion
    x_layer: str
    strict: bool

    def parse(self, table: pd.DataFrame) -> ParsedData:
        pieces = self.conversion.convert(table)
        check_layer_occupancy(pieces.layers, x_layer=self.x_layer, strict=self.strict)
        return ParsedData(
            X=pieces.X,
            obs=pieces.obs,
            var=pieces.var,
            uns={},
            layers=pieces.layers,
        )


def make_result_conversion(rule: LongRule | WideRule, *, strict: bool) -> BuildResult:
    """Translate the rule's shape once into the conversion that builds the result."""
    return BuildResult(conversion=make_conversion(rule), x_layer=rule.axis.x_layer, strict=strict)


@dataclass(frozen=True, slots=True)
class AttachProvenance:
    """Attach the precomputed, JSON-typed output provenance to the result."""

    payload: Mapping[str, JsonValue]

    def attach(self, result: ParsedData) -> ParsedData:
        return replace(result, uns={**result.uns, **self.payload})


def make_output_metadata(rule: LongRule | WideRule) -> AttachProvenance:
    """Serialize the rule's provenance once; the returned step retains no model."""
    return AttachProvenance(
        payload={
            "rule_json": json.dumps(rule.model_dump(mode="json")),
            "schema_version": rule.schema_version,
            "software_name": rule.software_name,
            "shape": rule.shape,
            "quantification_level": rule.quantification_level,
        }
    )
