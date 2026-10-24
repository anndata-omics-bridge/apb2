"""Every capability ``Parser`` exercises, declared once, by the client that exercises it.

A Protocol lives with its consumer, not with its implementations: that is what lets the
parser be typed structurally without importing a single concrete strategy, and what lets a
test inject fakes to assert the call order. Each names the smallest capability the parser
actually uses and has at least two real implementations.

Concrete providers do not import these Protocols to claim conformance. Strict structural
typing proves it where ``compile.py`` performs the wiring.

Execution laws:

- axis selections are Polars expressions; computations preserve frame rows and columns
  while assigning their declared output and returning explicit diagnostic metadata;
- ``LayerValueParser.present`` returns a Boolean expression that evaluates to one non-null mask
  value per input row, in input order, and never a converted measurement value;
- a duplicate policy receives a layer's repeated cells, absent values already null, and returns
  one row per cell, keeping the placed position columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import ObservationGrid
from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, JsonValue, ObsFinal, VarFinal
from apb2.parserV2.parse_quant.data.raw import LongLayer, RawLevel
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan
from apb2.parserV2.parse_quant.parameters.source import LevelSourceTable


class BoundInputReader(Protocol):
    """Read one already bound source through one resolved level projection."""

    def read(self) -> LevelSourceTable: ...


class SourceDecomposer(Protocol):
    """Convert one physical table shape into common raw axes and long layers."""

    def decompose(self, table: LevelSourceTable, /) -> RawLevel: ...


class FragmentTableSeparator(Protocol):
    """Turn one packed-fragment table into scalar-long rows, in two steps.

    ``listed`` splits the packed columns and drops the rows that hold no scalar; ``exploded``
    gives each scalar its row and the ``label_output`` column, keeping every other column.
    """

    @property
    def label_output(self) -> str: ...

    @property
    def packed_sources(self) -> tuple[str, ...]: ...

    def listed(self, frame: pl.DataFrame, /) -> pl.DataFrame: ...

    def exploded(self, frame: pl.DataFrame, /) -> pl.DataFrame: ...


class AxisValueCoercer(Protocol):
    """Build and validate an expression selecting one declared logical column."""

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr: ...


class ColumnComputer(Protocol):
    """Materialize one declared computed column from its exact ordered inputs.

    The operation owns its named expressions; the parser never extracts or reassembles Series.
    """

    @property
    def name(self) -> str: ...

    @property
    def inputs(self) -> tuple[str, ...]: ...

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]: ...


class DuplicatePolicy(Protocol):
    """Resolve the repeated values of each placed cell into one value."""

    def resolve(self, layer: LongLayer, /) -> LongLayer: ...


class FeatureValuePolicy(Protocol):
    """Resolve the values a feature's repeated rows give one var payload column.

    Both expressions run inside one aggregation over the raw var keys, on the columns as the
    reader produced them, before any coercion; ``schema`` gives their dtypes.
    """

    @property
    def source(self) -> str: ...

    def value(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr: ...

    def disagrees(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr: ...


class LayerValueParser(Protocol):
    """One layer's settings for raw presence and final value parsing, in separate phases.

    ``present`` marks the raw scalars that claim a cell, without converting them; ``parse``
    turns the resolved cells into canonical values and widens them onto the grid.
    """

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr: ...

    def parse(
        self, layer: LongLayer, grid: ObservationGrid, /
    ) -> tuple[FinalLayerTable, dict[str, JsonValue]]: ...


class LayerSetValidator(Protocol):
    """Validate relationships across the complete set of canonical layers.

    Returns the effectively empty layers, or ``None`` when no layer is populated enough to
    judge emptiness against.
    """

    def validate(self, layers: Mapping[str, FinalLayerTable]) -> dict[str, JsonValue] | None: ...


@dataclass(frozen=True, slots=True)
class PrunedFields:
    """A level's axes and layers without their uninformative fields, and what those held.

    ``single_valued`` and ``effectively_empty`` map each location, ``obs``, ``var`` or
    ``layers``, to the fields left out there and their findings.
    """

    obs: ObsFinal
    var: VarFinal
    layers: dict[str, FinalLayerTable]
    single_valued: dict[str, dict[str, JsonValue]]
    effectively_empty: dict[str, dict[str, JsonValue]]


class FieldPruner(Protocol):
    """Leave out the columns and layers of one level that carry no information.

    ``effectively_empty`` is the layer validator's verdict: the layers to leave out as empty.
    """

    def prune(
        self,
        obs: ObsFinal,
        var: VarFinal,
        layers: Mapping[str, FinalLayerTable],
        effectively_empty: Mapping[str, JsonValue],
        /,
    ) -> PrunedFields: ...


# ------------------------------------------------------------------------- runtime axis plans


@dataclass(frozen=True, slots=True)
class SelectedAxisColumn:
    """One axis column with its configured coercion already chosen."""

    name: str
    source: str
    coercer: AxisValueCoercer


@dataclass(frozen=True, slots=True)
class AxisPhaseRuntimePlan:
    """One materialization phase's configured operations, in execution order."""

    selections: tuple[SelectedAxisColumn, ...]
    computers: tuple[ColumnComputer, ...]


@dataclass(frozen=True, slots=True)
class AxisRuntimePlan:
    """One axis, fully configured: identity, both phases, and its retained outputs.

    Only executable operations are here. An absent optional source removed its selection
    and every computation it blocked during source resolution, so no runtime object carries
    a required flag or a skipped-name set.
    """

    keys: AxisKeyPlan
    key_phase: AxisPhaseRuntimePlan
    output_phase: AxisPhaseRuntimePlan
    outputs: tuple[str, ...]
