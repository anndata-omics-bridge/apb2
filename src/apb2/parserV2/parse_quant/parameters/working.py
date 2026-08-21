"""Working parse parameters: the rule with its search-parameter evidence already consumed.

This is the pre-source stage. Levels are chosen, gates and overrides applied, and every
Pydantic model gone — but physical column matches, dialects, read dtypes, and optional-source
presence are all still open. ``ParseRuleFacade`` constructs these values and then resolves
them against observed evidence; nothing here reads a file.

``JsonScalar`` and ``JsonValue`` are declared locally for the same reason as in
``data/parsed.py``: provenance enters as data, and a shared parent alias module would force
this child to import upward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnDeclaration,
    ModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import DuplicateMode
from apb2.parserV2.parse_quant.parameters.source import InputContract

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
"""The parsing-owned level vocabulary, structurally equal to the rule package's own."""

LEVELS: tuple[QuantificationLevel, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)
"""Canonical level order, which ``compile_parsers`` preserves."""


# ---------------------------------------------------------------------------- source layout


@dataclass(frozen=True, slots=True)
class LongSourceLayout:
    """One physical row per (observation, feature)."""

    kind: Literal["long"]


@dataclass(frozen=True, slots=True)
class WideSourceLayout:
    """One physical row per feature; observations are header captures."""

    kind: Literal["wide"]


@dataclass(frozen=True, slots=True)
class PositionalFragmentLayout:
    """Long rows whose packed fragment lists carry no labels; labels are positional."""

    kind: Literal["positional_fragment"]
    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentLayout:
    """Long rows whose fragment labels are packed in parallel in ``label_source``."""

    kind: Literal["column_labeled_fragment"]
    label_source: str
    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]


type SourceLayoutDeclaration = (
    LongSourceLayout | WideSourceLayout | PositionalFragmentLayout | ColumnLabeledFragmentLayout
)


# ------------------------------------------------------------------------ layer declarations


@dataclass(frozen=True, slots=True)
class NullOnlyRawValuePresenceDeclaration:
    """No sentinel and no structure: only null claims nothing."""

    kind: Literal["null_only"]


@dataclass(frozen=True, slots=True)
class PlainNumericRawValuePresenceDeclaration:
    """Declared numeric sentinels claim nothing, alongside null and blank text."""

    kind: Literal["plain_numeric"]
    missing_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegexNumericRawValuePresenceDeclaration:
    """The comparable number is one capture of a structured token."""

    kind: Literal["regex_numeric"]
    missing_values: tuple[float, ...]
    pattern: str


type RawValuePresenceDeclaration = (
    NullOnlyRawValuePresenceDeclaration
    | PlainNumericRawValuePresenceDeclaration
    | RegexNumericRawValuePresenceDeclaration
)


@dataclass(frozen=True, slots=True)
class PlainNumericEncodingDeclaration:
    """A layer whose cells are directly parseable numbers."""

    kind: Literal["plain_numeric"]
    missing_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegexNumericEncodingDeclaration:
    """A layer whose numeric value is one capture group of a structured cell."""

    kind: Literal["regex_numeric"]
    missing_values: tuple[float, ...]
    pattern: str


@dataclass(frozen=True, slots=True)
class FactorEncodingDeclaration:
    """A layer whose cells are category labels with declared codes."""

    kind: Literal["factor"]
    categories: tuple[tuple[str, int], ...]


type AnnDataLayerEncodingDeclaration = (
    PlainNumericEncodingDeclaration | RegexNumericEncodingDeclaration | FactorEncodingDeclaration
)


# ------------------------------------------------------------------- the working configuration


@dataclass(frozen=True, slots=True)
class WorkingAxisConfiguration:
    """One axis's authored identity and the columns it declares."""

    final_key_columns: tuple[str, ...]
    columns: AxisColumnDeclaration


@dataclass(frozen=True, slots=True)
class WorkingMeasurementLayer:
    """One named measurement: where its values come from and what they mean."""

    name: str
    source: str
    raw_presence: RawValuePresenceDeclaration
    ann_data_encoding: AnnDataLayerEncodingDeclaration


@dataclass(frozen=True, slots=True)
class WorkingMeasurements:
    """Every declared measurement, with the primary layer already promoted to required.

    Required and optional are separate collections because a missing source means different
    things for each: incompatible for one, omitted for the other. Splitting them loses the
    authored interleaving, which the parsed result must still preserve, so ``authored_order``
    states it — the order the document declared, not a flag on a record.
    """

    primary_layer_name: str
    duplicate_mode: DuplicateMode
    required_layers: tuple[WorkingMeasurementLayer, ...]
    optional_layers: tuple[WorkingMeasurementLayer, ...]
    authored_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkingParseConfiguration:
    """One level's complete rule, storage-model free and not yet source-resolved."""

    level: QuantificationLevel
    input: InputContract
    source_layout: SourceLayoutDeclaration
    obs: WorkingAxisConfiguration
    var: WorkingAxisConfiguration
    measurements: WorkingMeasurements
    modifications: tuple[ModificationConfig, ...]
    provenance: Mapping[str, JsonValue]
