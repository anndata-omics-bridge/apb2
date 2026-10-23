"""Physical-source parameters: what the caller supplies, what a rule allows, what was found.

Three stages live here and stay distinct. A caller supplies an ``InputSource``. A rule contributes
an ``InputContract`` containing one optional folder file name and its bounded physical
interpretations. Binding them yields ``SourceEvidence``: the dialect and header actually observed,
which is the only physical fact source resolution may consult.

``LevelReadPlan`` and the decomposition configurations are the resolved output side: exactly
which columns one level reads, which of them must stay text, and which physical shape its
table has. A reader returns a ``LevelSourceTable`` with the ``InputTable`` of each file it read.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from apb2.parserV2.parse_quant.parameters.vendor_file import (
    GZIP_SUFFIX,
    VendorFile,
    make_vendor_file,
)

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

    @property
    def file(self) -> VendorFile:
        """The file as apb2 reads it, plain or gzipped."""
        return make_vendor_file(self.path)


@dataclass(frozen=True, slots=True)
class Folder:
    """A folder in which exactly one declared candidate file name must resolve."""

    path: Path

    def member(self, name: str) -> VendorFile | None:
        """The table a rule declares by ``name``, stored plain or gzipped; ``None`` if absent."""
        for candidate in (self.path / name, self.path / f"{name}{GZIP_SUFFIX}"):
            if candidate.is_file():
                return make_vendor_file(candidate)
        return None


@dataclass(frozen=True, slots=True)
class InputFiles:
    """Explicit physical inputs; keys may bind renamed files to canonical vendor names."""

    path: Path
    files: Mapping[str, Path]


type InputFileRecord = dict[str, int | str | list[str]]
"""One entry of root parse provenance ``input_files``."""


@dataclass(frozen=True, slots=True)
class InputFile:
    """One file a conversion read, in the role and format its reader gave it."""

    path: Path
    role: str
    format: str

    def record(self) -> InputFileRecord:
        """Hash and size the file's bytes on disk; only its base name enters the record."""
        with self.path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "role": self.role,
            "file_name": self.path.name,
            "size_bytes": self.path.stat().st_size,
            "sha256": sha256,
            "compression": "gzip" if self.path.suffix.lower() == GZIP_SUFFIX else "none",
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class InputTable:
    """One table a reader read: its file, full header and data rows before any filter."""

    file: InputFile
    columns: tuple[str, ...]
    nrows: int

    def record(self) -> InputFileRecord:
        """The file's record with the reader's header and row count."""
        return {
            **self.file.record(),
            "nrows": self.nrows,
            "ncols": len(self.columns),
            "columns": list(self.columns),
        }


@dataclass(slots=True)
class LevelSourceTable:
    """One level's projected physical rows: what a reader returns and a decomposer accepts.

    The columns one level's read plan asked for, in that order, with no axis identity yet.
    Which columns are keys and which are measurements is the decomposer's decision.
    """

    frame: pl.DataFrame
    input_tables: tuple[InputTable, ...] = ()
    """The physical tables read, for root ``input_files``; none for an in-memory frame."""


@dataclass(frozen=True, slots=True)
class PreparedTable:
    """One preparation result shared by detection and every requested level."""

    path: Path
    frame: pl.DataFrame
    how: str
    input_tables: tuple[InputTable, ...]
    duration_seconds: float


type InputSource = SingleFile | Folder | InputFiles | PreparedTable


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

    def required_packed_sources(self) -> tuple[str, ...]:
        """Return the packed columns a header must carry."""
        return ()

    def has_layer_source(self, source: str, header: Collection[str]) -> bool:
        """Whether an exact long-layer source occurs in the header."""
        return source in header

    def synthesized_var_columns(self) -> tuple[str, ...]:
        """Return feature columns synthesized before axis materialization."""
        return ()


@dataclass(frozen=True, slots=True)
class WideSourceLayout:
    """One physical row per feature; observations are one layer's header captures."""

    sample_layer_name: str

    def packed_sources(self) -> tuple[str, ...]:
        """Return physical columns that require packed-value splitting."""
        return ()

    def required_packed_sources(self) -> tuple[str, ...]:
        """Return the packed columns a header must carry."""
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

    def required_packed_sources(self) -> tuple[str, ...]:
        """Return the packed columns a header must carry."""
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
    label_required: bool = True

    def packed_sources(self) -> tuple[str, ...]:
        """Return the label column and value columns split in parallel."""
        return (self.label_source, *self.packed_value_sources)

    def required_packed_sources(self) -> tuple[str, ...]:
        """Return the packed columns a header must carry; an optional label is not one."""
        return self.packed_sources() if self.label_required else self.packed_value_sources

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
