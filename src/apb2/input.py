"""The input side: recognise the file's format, bind a dialect, read a projection.

One module for everything between a path and a DataFrame. ``format_for(path)`` returns a
path-initialized reader for header-level questions; ``bind_source`` resolves one typed
``InputSource`` into one concrete bound table for a rule (the single dispatch over the
source union); ``compile_read_plan`` fixes the exact projected columns; the two table
readers execute the plan. Detection is not an unrestricted guess: a candidate delimiter
is viable only when the header it exposes satisfies the rule's required sources.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from apb2.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.sources import (
    DelimitedDialect,
    DelimitedFile,
    FileRoles,
    Folder,
    InputSource,
    SingleFile,
    UngroupedNumbers,
)
from apb2.vendor_parse_rules.model import LongRule, WideRule
from apb2.vendor_parse_rules.runtime import Recognition, recognition_for

# ------------------------------------------------------------------- tabular file formats

_FIXED_DELIMITERS = {".csv": ",", ".tsv": "\t"}
_TEXT_SUFFIX = ".txt"
_PARQUET_SUFFIX = ".parquet"
_TEXT_DELIMITERS = ",\t"
_COMMA_DECIMAL_RE = re.compile(r"^-?\d+,(\d+)$")
_THOUSANDS_GROUP_WIDTH = 3
_DECIMAL_SAMPLE_LINES = 500
_DELIMITER_CANDIDATES: dict[str, tuple[str, ...]] = {
    ".csv": (",",),
    ".tsv": ("\t",),
    ".txt": ("\t", ",", ";"),
}


class UnknownFormat(ValueError):
    """Raised when a file extension has no registered format."""


class DelimitedText:
    """One delimited text file with its resolved separator."""

    def __init__(self, path: Path, delimiter: str) -> None:
        self.path = path
        self.delimiter = delimiter

    def columns(self) -> list[str]:
        """Read only the header row."""
        return list(
            pd.read_csv(self.path, sep=self.delimiter, encoding="utf-8-sig", nrows=0).columns
        )

    def decimal_separator(self) -> str:
        """Detect whether this file writes numbers with a comma decimal mark.

        Only the shape of the number distinguishes the two readings of ``1,234``: a
        thousands separator always groups exactly three digits, so a field whose comma is
        followed by three digits is ambiguous and never counted as evidence. A
        comma-delimited file cannot carry bare comma decimals at all and is reported as
        dot-decimal without inspection.
        """
        if self.delimiter == ",":
            return "."
        decimal_like = 0
        # Tolerant decoding: this scan only looks for digits and commas, and must not
        # turn a file pandas can still read into a decode failure before pandas sees it.
        with self.path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
            handle.readline()
            for line in islice(handle, _DECIMAL_SAMPLE_LINES):
                for field in line.rstrip("\n").split(self.delimiter):
                    match = _COMMA_DECIMAL_RE.match(field)
                    if match is not None and len(match.group(1)) != _THOUSANDS_GROUP_WIDTH:
                        decimal_like += 1
        return "," if decimal_like else "."


class Parquet:
    """One Parquet file; its physical schema replaces textual dialect resolution."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def columns(self) -> list[str]:
        """Read only the column names from the physical schema."""
        return list(pq.read_schema(self.path).names)


type TabularFile = DelimitedText | Parquet


def format_for(path: Path) -> TabularFile:
    """Return the reader for a path's extension, its delimiter already resolved.

    Raises UnknownFormat if the extension is not registered.
    """
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return Parquet(path)
    fixed = _FIXED_DELIMITERS.get(suffix)
    if fixed is not None:
        return DelimitedText(path, fixed)
    if suffix == _TEXT_SUFFIX:
        return DelimitedText(path, _sniff_text_delimiter(path))
    known = sorted((*_FIXED_DELIMITERS, _TEXT_SUFFIX, _PARQUET_SUFFIX))
    raise UnknownFormat(f"unsupported extension {path.suffix!r} for {path}; known: {known}")


def _sniff_text_delimiter(path: Path) -> str:
    """Detect comma- versus tab-delimited text from a bounded content sample."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65536)
    if not any(delimiter in sample for delimiter in _TEXT_DELIMITERS):
        # A one-column text file has no delimiter to detect. Preserve the historical
        # generic-.txt default so it still reads as one tabular column.
        return "\t"
    return csv.Sniffer().sniff(sample, delimiters=_TEXT_DELIMITERS).delimiter


# ----------------------------------------------------------------------------- read plans


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """Projected source columns in header order, and which of them stay textual."""

    columns: tuple[str, ...]
    string_sources: frozenset[str]


def compile_read_plan(
    recognition: Recognition,
    rule: LongRule | WideRule,
    header: Sequence[str],
    modification_sources: Iterable[str],
    packed_columns: Iterable[str],
) -> ReadPlan:
    """Compile the exact projection one rule needs from one inspected header.

    Required sources are validated separately (``recognition.matches`` during
    construction); the projection is the intersection of the header with everything the
    rule can read: selected and optional sources, computed-column inputs, layer sources
    (exact for long rules, regex-expanded for wide ones), modification sources, and
    packed fragment columns.
    """
    needed: set[str] = set(modification_sources) | set(packed_columns)
    for _axis, group in recognition.column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
        for column in group.computed:
            needed.update(column.inputs)
    needed.update(recognition.layer_source_columns(list(header)))
    columns = tuple(name for name in header if name in needed)
    strings = string_sources_for_rules([rule]) & set(columns)
    return ReadPlan(columns=columns, string_sources=frozenset(strings))


def string_sources_for_rules(rules: Iterable[LongRule | WideRule]) -> frozenset[str]:
    """Return real vendor sources whose exact textual tokens must survive reading."""
    source_types: dict[str, str] = {}
    for rule in rules:
        for _axis, group in recognition_for(rule).column_groups():
            selected = {**group.select, **group.optional_select}
            for output_name, source_name in selected.items():
                logical_type = group.types.get(output_name, "string")
                if source_name in source_types and source_types[source_name] != logical_type:
                    raise ValueError(
                        "conflicting logical types for vendor source "
                        f"{source_name!r}: {source_types[source_name]!r} and "
                        f"{logical_type!r}"
                    )
                source_types[source_name] = logical_type
    return frozenset(
        source for source, logical_type in source_types.items() if logical_type == "string"
    )


# -------------------------------------------------------------------------- bound readers


class DelimitedTableReader:
    """Read one delimited file with its concrete dialect and exact projection."""

    def __init__(self, path: Path, dialect: DelimitedDialect, plan: ReadPlan) -> None:
        self.path = path
        self.dialect = dialect
        self.plan = plan

    def read(self) -> pd.DataFrame:
        dtypes = dict.fromkeys(sorted(self.plan.string_sources), "string")
        return pd.read_csv(
            self.path,
            sep=self.dialect.delimiter,
            encoding="utf-8-sig",
            usecols=list(self.plan.columns),
            dtype=dtypes,
            **self.dialect.numbers.read_csv_options(),
        )


class ParquetTableReader:
    """Read one Parquet file's projected columns; its schema already types them."""

    def __init__(self, path: Path, plan: ReadPlan) -> None:
        self.path = path
        self.plan = plan

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=list(self.plan.columns))


# --------------------------------------------------------------------------- bind sources


class BoundDelimited:
    """One delimited file with its completely resolved dialect."""

    def __init__(self, path: Path, dialect: DelimitedDialect) -> None:
        self.path = path
        self.dialect = dialect

    def header(self) -> list[str]:
        return DelimitedText(self.path, self.dialect.delimiter).columns()

    def make_reader(self, plan: ReadPlan) -> DelimitedTableReader:
        return DelimitedTableReader(self.path, self.dialect, plan)


class BoundParquet:
    """One Parquet file; its physical schema replaces textual dialect resolution."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def header(self) -> list[str]:
        return Parquet(self.path).columns()

    def make_reader(self, plan: ReadPlan) -> ParquetTableReader:
        return ParquetTableReader(self.path, plan)


type BoundTable = BoundDelimited | BoundParquet


def bind_source(
    source: InputSource,
    rule: LongRule | WideRule,
    recognition: Recognition,
) -> BoundTable:
    """Resolve one typed source into one concrete bound table for ``rule``."""
    match source:
        case DelimitedFile(path=path, dialect=dialect):
            return BoundDelimited(path, dialect)
        case SingleFile(path=path):
            return _bind_single_file(path, rule, recognition)
        case Folder() | FileRoles():
            raise IncompatibleSourceError(
                f"rule {rule.software_name!r} level {rule.quantification_level!r} reads one "
                "table; folder and file-role sources need a file-set rule, and no packaged "
                "rule declares one yet (plan stage 7)"
            )


def _bind_single_file(
    path: Path,
    rule: LongRule | WideRule,
    recognition: Recognition,
) -> BoundTable:
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return BoundParquet(path)
    candidates = _DELIMITER_CANDIDATES.get(suffix)
    if candidates is None:
        known = sorted((_PARQUET_SUFFIX, *_DELIMITER_CANDIDATES))
        raise UnknownFormat(f"unsupported extension {path.suffix!r} for {path}; known: {known}")
    viable = [
        delimiter
        for delimiter in candidates
        if recognition.matches(DelimitedText(path, delimiter).columns())
    ]
    if not viable:
        raise IncompatibleSourceError(
            f"{path} does not expose the columns required by {rule.software_name!r} level "
            f"{rule.quantification_level!r} under any candidate delimiter {candidates!r}"
        )
    if len(viable) > 1:
        raise AmbiguousDialectError(
            f"{path} satisfies {rule.software_name!r} level "
            f"{rule.quantification_level!r} under several delimiters {viable!r}; bind an "
            "explicit DelimitedFile dialect instead"
        )
    delimiter = viable[0]
    decimal = DelimitedText(path, delimiter).decimal_separator()
    return BoundDelimited(path, DelimitedDialect(delimiter, UngroupedNumbers(decimal)))
