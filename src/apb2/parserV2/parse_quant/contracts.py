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
- ``RawValuePresence.present`` returns a Boolean expression that evaluates to one non-null mask
  value per input row, in input order, and never a converted measurement value;
- a duplicate policy preserves the raw var-key columns and the input group order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, ParsedLevel
from apb2.parserV2.parse_quant.data.raw import DecomposedDataRaw, RawLayerTable
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan


class BoundInputReader(Protocol):
    """Read one already bound source through one resolved level projection."""

    def read(self) -> LevelSourceTable: ...


class SourceDecomposer(Protocol):
    """Convert one physical table shape into common raw axes and wide raw layers."""

    def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw: ...


class FragmentTableSeparator(Protocol):
    """Turn one packed-fragment table into scalar-long rows."""

    def separate(self, table: LevelSourceTable, /) -> LevelSourceTable: ...


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


class RawValuePresence(Protocol):
    """Mark the raw layer scalars that semantically claim a cell, without converting them."""

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr: ...


class DuplicatePolicy(Protocol):
    """Resolve repeated values of each raw wide cell into one value."""

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable: ...


class LayerValueParser(Protocol):
    """One layer's settings for raw presence and final value parsing, in separate phases."""

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr: ...

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable: ...


class LayerSetValidator(Protocol):
    """Validate relationships across the complete set of canonical layers."""

    def validate(self, layers: Mapping[str, FinalLayerTable], /) -> None: ...


class ParsedLevelWriter(Protocol):
    """Persist one parsed level."""

    def write(self, parsed: ParsedLevel, target: Path, /) -> None: ...


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
