"""Physical-source parameters: what the caller supplies, what a rule allows, what was found.

Three stages live here and stay distinct. A caller supplies an ``InputSource``. A rule contributes
an ``InputContract`` containing one optional folder file name and its bounded physical
interpretations. Binding them yields ``SourceEvidence``: the dialect and header actually observed,
which is the only physical fact source resolution may consult.

``LevelReadPlan`` and the decomposition configurations are the resolved output side: exactly
which columns one level reads, which of them must stay text, and which physical shape its
table has.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

type TextEncoding = Literal["utf8", "utf8-lossy", "windows-1252"]
"""The delimited encodings the reader supports; a rule may permit several, in preference order."""


@dataclass(frozen=True, slots=True)
class NumericTextFormat:
    """How one physical file writes numbers."""

    decimal_mark: Literal[".", ","]
    thousands_marks: tuple[str, ...]


# ------------------------------------------------------------------ caller-supplied sources


@dataclass(frozen=True, slots=True)
class SingleFile:
    """One file whose declared format is resolved from its extension and evidence."""

    path: Path


@dataclass(frozen=True, slots=True)
class DelimitedFile:
    """One delimited file whose dialect the caller states explicitly.

    The escape hatch for a file whose detection is ambiguous. The stated dialect is still
    checked against the rule's declared policy and against the header it exposes.
    """

    path: Path
    delimiter: str
    encoding: TextEncoding
    numbers: NumericTextFormat
    quote_char: str = '"'


@dataclass(frozen=True, slots=True)
class Folder:
    """A folder in which exactly one declared candidate file name must resolve."""

    path: Path


@dataclass(frozen=True, slots=True)
class InputFiles:
    """Explicit physical inputs; keys may bind renamed files to canonical vendor names."""

    path: Path
    files: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class PreparedTable:
    """One preparation result shared by detection and every requested level."""

    path: Path
    frame: pl.DataFrame
    how: str
    source_paths: tuple[Path, ...]
    duration_seconds: float


type InputSource = SingleFile | DelimitedFile | Folder | InputFiles | PreparedTable


# ------------------------------------------------------------------- rule-permitted formats


@dataclass(frozen=True, slots=True)
class DelimitedFormatContract:
    """One delimited interpretation a rule permits, with its candidates already flattened.

    A fixed declaration arrives as one candidate and a detected one as its ordered
    candidates, so the binder tries one uniform bounded set and never reads a stored
    fixed/detect mode.
    """

    extensions: tuple[str, ...]
    encoding_candidates: tuple[TextEncoding, ...]
    quote_char: str
    delimiter_candidates: tuple[str, ...]
    number_format_candidates: tuple[NumericTextFormat, ...]


@dataclass(frozen=True, slots=True)
class ParquetFormatContract:
    """A Parquet interpretation a rule permits; its physical schema needs no dialect."""

    extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExcelFormatContract:
    """One named sheet in an Excel workbook, independent of its filename suffix."""

    extensions: tuple[str, ...]
    sheet_name: str


type PhysicalFormatContract = DelimitedFormatContract | ParquetFormatContract | ExcelFormatContract


@dataclass(frozen=True, slots=True)
class InputContract:
    """The single table one level reads and its permitted physical form."""

    file_name: str | None
    formats: tuple[PhysicalFormatContract, ...]


# ---------------------------------------------------------------- projected source layouts


@dataclass(frozen=True, slots=True)
class LongSourceLayout:
    """One physical row per observation and feature."""

    def packed_sources(self) -> tuple[str, ...]:
        """Return physical columns that require packed-value splitting."""
        return ()

    def has_layer_source(self, source: str, header: Collection[str]) -> bool:
        """Whether an exact long-layer source occurs in the header."""
        return source in header

    def synthesized_var_columns(self) -> tuple[str, ...]:
        """Return feature columns synthesized before axis materialization."""
        return ()


@dataclass(frozen=True, slots=True)
class WideSourceLayout:
    """One physical row per feature; observations are header captures."""

    def packed_sources(self) -> tuple[str, ...]:
        """Return physical columns that require packed-value splitting."""
        return ()

    def has_layer_source(self, source: str, header: Collection[str]) -> bool:
        """Whether a wide-layer pattern matches at least one header column."""
        pattern = re.compile(source)
        return any(pattern.match(name) for name in header)

    def synthesized_var_columns(self) -> tuple[str, ...]:
        """Return feature columns synthesized before axis materialization."""
        return ()


@dataclass(frozen=True, slots=True)
class PositionalFragmentLayout:
    """Long rows whose packed fragment lists carry positional labels."""

    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]

    def packed_sources(self) -> tuple[str, ...]:
        """Return the value columns split in parallel."""
        return self.packed_value_sources

    def has_layer_source(self, source: str, header: Collection[str]) -> bool:
        """Whether an exact packed layer source occurs in the header."""
        return source in header

    def synthesized_var_columns(self) -> tuple[str, ...]:
        """Return the fragment label synthesized during separation."""
        return (self.label_output,)


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentLayout:
    """Long rows whose fragment labels and values are packed in parallel."""

    label_source: str
    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]

    def packed_sources(self) -> tuple[str, ...]:
        """Return the label column and value columns split in parallel."""
        return (self.label_source, *self.packed_value_sources)

    def has_layer_source(self, source: str, header: Collection[str]) -> bool:
        """Whether an exact packed layer source occurs in the header."""
        return source in header

    def synthesized_var_columns(self) -> tuple[str, ...]:
        """Return the fragment label synthesized during separation."""
        return (self.label_output,)


type SourceLayoutDeclaration = (
    LongSourceLayout | WideSourceLayout | PositionalFragmentLayout | ColumnLabeledFragmentLayout
)


# ------------------------------------------------------------------------ observed evidence


@dataclass(frozen=True, slots=True)
class DelimitedSourceEvidence:
    """The already selected, unambiguous dialect of one delimited file, plus its header."""

    columns: tuple[str, ...]
    delimiter: str
    quote_char: str
    encoding: TextEncoding
    number_format: NumericTextFormat


@dataclass(frozen=True, slots=True)
class FrameSourceEvidence:
    """Native schema from a Parquet file or a prepared frame, in column order."""

    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, pl.DataType], ...]


@dataclass(frozen=True, slots=True)
class ExcelSourceEvidence:
    """The header exposed by one named workbook sheet."""

    columns: tuple[str, ...]
    sheet_name: str
    number_format: NumericTextFormat


type SourceEvidence = DelimitedSourceEvidence | FrameSourceEvidence | ExcelSourceEvidence


# --------------------------------------------------------------------- resolved read + shape


@dataclass(frozen=True, slots=True)
class LevelReadPlan:
    """Exactly what one level reads, with every projected column's read dtype decided.

    For delimited input ``text_sources`` and ``native_numeric_sources`` are disjoint and
    their union is ``projected_columns``: no column is left to inference.
    """

    projected_columns: tuple[str, ...]
    text_sources: frozenset[str]
    native_numeric_sources: frozenset[str]


@dataclass(frozen=True, slots=True)
class LongRawLayerSource:
    """One long layer and the exact physical column holding its values."""

    name: str
    source_column: str


@dataclass(frozen=True, slots=True)
class WideRawLayerSource:
    """One resolved wide header column and the observation it belongs to."""

    source_column: str
    sample: str


@dataclass(frozen=True, slots=True)
class WideRawLayerPlan:
    """One wide layer's resolved header columns, in stable header order."""

    name: str
    sources: tuple[WideRawLayerSource, ...]


@dataclass(frozen=True, slots=True)
class LongDecompositionConfig:
    """One row per (observation, feature); every layer names an exact column."""

    kind: Literal["long"]
    primary_layer_name: str
    layer_sources: tuple[LongRawLayerSource, ...]


@dataclass(frozen=True, slots=True)
class WideDecompositionConfig:
    """One row per feature; observations came from resolved header captures."""

    kind: Literal["wide"]
    primary_layer_name: str
    layer_plans: tuple[WideRawLayerPlan, ...]


@dataclass(frozen=True, slots=True)
class PositionalFragmentSeparationConfig:
    """Packed fragments with no label column: labels are the index within the precursor."""

    kind: Literal["positional"]
    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentSeparationConfig:
    """Packed fragments whose labels are packed in parallel in ``label_source``."""

    kind: Literal["column"]
    label_source: str
    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]


type FragmentSeparationConfig = (
    PositionalFragmentSeparationConfig | ColumnLabeledFragmentSeparationConfig
)


@dataclass(frozen=True, slots=True)
class DelimitedFragmentDecompositionConfig:
    """Separate the packed fragments, then decompose the scalar rows as ordinary long."""

    kind: Literal["delimited_fragment"]
    separator: FragmentSeparationConfig
    long: LongDecompositionConfig


type DecompositionConfig = (
    LongDecompositionConfig | WideDecompositionConfig | DelimitedFragmentDecompositionConfig
)
