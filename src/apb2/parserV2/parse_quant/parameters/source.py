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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

type TextEncoding = Literal["utf8", "utf8-lossy"]
"""The delimited encodings the reader supports; a rule may permit either."""


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


type InputSource = SingleFile | DelimitedFile | Folder


# ------------------------------------------------------------------- rule-permitted formats


@dataclass(frozen=True, slots=True)
class DelimitedFormatContract:
    """One delimited interpretation a rule permits, with its candidates already flattened.

    A fixed declaration arrives as one candidate and a detected one as its ordered
    candidates, so the binder tries one uniform bounded set and never reads a stored
    fixed/detect mode.
    """

    extensions: tuple[str, ...]
    encoding: TextEncoding
    quote_char: str
    delimiter_candidates: tuple[str, ...]
    number_format_candidates: tuple[NumericTextFormat, ...]


@dataclass(frozen=True, slots=True)
class ParquetFormatContract:
    """A Parquet interpretation a rule permits; its physical schema needs no dialect."""

    extensions: tuple[str, ...]


type PhysicalFormatContract = DelimitedFormatContract | ParquetFormatContract


@dataclass(frozen=True, slots=True)
class InputContract:
    """The single table one level reads and its permitted physical form."""

    file_name: str | None
    formats: tuple[PhysicalFormatContract, ...]


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
class ParquetSourceEvidence:
    """One Parquet file's physical schema; ``dtypes`` names and order match ``columns``."""

    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, pl.DataType], ...]


type SourceEvidence = DelimitedSourceEvidence | ParquetSourceEvidence


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
