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
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs

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

    def tables(self, file_name: str | None) -> tuple[VendorFile, ...]:
        """The one table this file is, whatever name a rule declares for it."""
        return (self.file,)


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

    def tables(self, file_name: str | None) -> tuple[VendorFile, ...]:
        """The member a rule declares by name, else every table file the folder holds."""
        member = None if file_name is None else self.member(file_name)
        if member is not None:
            return (member,)
        files = (make_vendor_file(path) for path in sorted(self.path.iterdir()) if path.is_file())
        return tuple(file for file in files if file.suffix in _TABLE_SUFFIXES)


_TABLE_SUFFIXES = frozenset({".txt", ".tsv", ".csv", ".parquet", ".xlsx"})


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


class LevelSourceTable:
    """One level's projected physical rows: what a reader returns and a decomposer accepts.

    The columns one level's read plan asked for, in that order, with no axis identity yet.
    Which columns are keys and which are measurements is the decomposer's decision.
    ``input_tables`` are the physical tables read, for root ``input_files``; none for an
    in-memory frame.
    """

    __slots__ = ("frame", "input_tables")

    def __init__(self, frame: pl.DataFrame, input_tables: tuple[InputTable, ...] = ()) -> None:
        """Hold every absent cell as null and drop the rows without a value.

        Absent is null, a float ``NaN``, or text without a non-whitespace character, whichever
        reader or preparation produced the frame.
        """
        frame = frame.with_columns(
            cs.float().fill_nan(None),
            pl.when(cs.string().str.contains(r"\S")).then(cs.string()),
            *(
                pl.when(pl.col(name).is_in(blank.implode()))
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name, blank in _blank_categories(frame).items()
            ),
        )
        if frame.width:
            frame = frame.filter(pl.any_horizontal(pl.all().is_not_null()))
        self.frame = frame
        self.input_tables = input_tables


def _blank_categories(frame: pl.DataFrame) -> dict[str, pl.Series]:
    """Per categorical column, its distinct texts without a non-whitespace character.

    Found among the column's distinct values, so the column itself is never cast to text.
    """
    blanks: dict[str, pl.Series] = {}
    for name in frame.select(cs.categorical()).columns:
        distinct = frame.get_column(name).unique().drop_nulls()
        blank = distinct.filter(~distinct.cast(pl.String).str.contains(r"\S"))
        if blank.len():
            blanks[name] = blank
    return blanks


type InputSource = SingleFile | Folder


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
    """Native schema from a Parquet file or a prepared frame, in column order.

    Its numeric columns are already numbers; text values are read with a dot decimal mark.
    """

    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, pl.DataType], ...]
    number_format: NumericTextFormat = NumericTextFormat(decimal_mark=".", thousands_marks=())


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
    ``categorical_sources`` are text sources that only name keys and feature or run metadata;
    they are read as ``pl.Categorical``, which holds each distinct text once.
    ``converted_sources`` are text sources of plain numeric layers; a delimited reader turns
    them into numbers while reading, and keeps a column as text when any of its values needs
    the text.
    """

    projected_columns: tuple[str, ...]
    text_sources: frozenset[str]
    native_numeric_sources: frozenset[str]
    categorical_sources: frozenset[str] = frozenset()
    converted_sources: frozenset[str] = frozenset()

    def casts(self) -> tuple[pl.Expr, ...]:
        """The casts a reader of native-typed input applies: numbers and categorical keys."""
        return tuple(
            pl.col(sorted(columns)).cast(dtype)
            for columns, dtype in (
                (self.native_numeric_sources, pl.Float64()),
                (self.categorical_sources, pl.Categorical()),
            )
            if columns
        )


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
