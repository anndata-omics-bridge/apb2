"""``BoundInputReader``: everything between a path and one assembled DataFrame.

One module for everything between a path and a DataFrame. ``format_for(path)`` returns a
path-initialized reader for header-level questions; ``bind_source`` resolves one typed
``InputSource`` into one concrete bound table (the single dispatch over the source union);
the two table readers execute a ``ReadPlan`` that ``selectors.compile_read_plan`` compiled
from the rule. Detection is not an unrestricted guess: a candidate delimiter is viable only
when the header it exposes is accepted, which is how the rule's required sources enter here
— as a predicate over headers, never as a rule.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from apb2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parse_quant.parse_strategy import BoundInputReader
from apb2.parse_quant.sources import (
    DelimitedDialect,
    DelimitedFile,
    FileRoles,
    Folder,
    InputSource,
    SingleFile,
    UngroupedNumbers,
)

type HeaderPredicate = Callable[[list[str]], bool]
"""Whether one inspected header satisfies the rule being constructed for."""

# ------------------------------------------------------------------- tabular file formats

_PARQUET_SUFFIX = ".parquet"
_TEXT_DELIMITERS = ",\t"
_COMMA_DECIMAL_RE = re.compile(r"^-?\d+,(\d+)$")
_THOUSANDS_GROUP_WIDTH = 3
_DECIMAL_SAMPLE_LINES = 500
# The one extension table: a single candidate is the extension's fixed delimiter; the
# multi-candidate ``.txt`` entry is resolved by sniffing (format_for) or by trying each
# candidate against the rule's required header (_bind_single_file).
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

    def header(self) -> list[str]:
        """The bound-table view of ``columns``: parquet needs no dialect resolution."""
        return self.columns()

    def make_reader(self, plan: ReadPlan) -> ParquetTableReader:
        return ParquetTableReader(self.path, plan)


type TabularFile = DelimitedText | Parquet


def format_for(path: Path) -> TabularFile:
    """Return the reader for a path's extension, its delimiter already resolved.

    Raises UnknownFormat if the extension is not registered.
    """
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return Parquet(path)
    candidates = _DELIMITER_CANDIDATES.get(suffix)
    if candidates is None:
        raise _unknown_format(path)
    if len(candidates) == 1:
        return DelimitedText(path, candidates[0])
    return DelimitedText(path, _sniff_text_delimiter(path))


def _unknown_format(path: Path) -> UnknownFormat:
    known = sorted((_PARQUET_SUFFIX, *_DELIMITER_CANDIDATES))
    return UnknownFormat(f"unsupported extension {path.suffix!r} for {path}; known: {known}")


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


_IMPLEMENTS: tuple[type[BoundInputReader], ...] = (DelimitedTableReader, ParquetTableReader)
"""Pyright checks each class against the protocol here, at its definition site."""


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


type BoundTable = BoundDelimited | Parquet


def bind_source(source: InputSource, *, accepts: HeaderPredicate, rule_label: str) -> BoundTable:
    """Resolve one typed source into one concrete bound table.

    ``accepts`` decides whether a candidate header satisfies the rule under construction,
    and ``rule_label`` names that rule in the errors raised when none does.
    """
    match source:
        case DelimitedFile(path=path, dialect=dialect):
            return BoundDelimited(path, dialect)
        case SingleFile(path=path):
            return _bind_single_file(path, accepts, rule_label)
        case Folder() | FileRoles():
            raise IncompatibleSourceError(
                f"rule {rule_label} reads one table; folder and file-role sources need a "
                "file-set rule, and no packaged rule declares one yet (plan stage 7)"
            )


def _bind_single_file(path: Path, accepts: HeaderPredicate, rule_label: str) -> BoundTable:
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return Parquet(path)
    candidates = _DELIMITER_CANDIDATES.get(suffix)
    if candidates is None:
        raise _unknown_format(path)
    viable = [
        delimiter for delimiter in candidates if accepts(DelimitedText(path, delimiter).columns())
    ]
    if not viable:
        raise IncompatibleSourceError(
            f"{path} does not expose the columns required by {rule_label} under any "
            f"candidate delimiter {candidates!r}"
        )
    if len(viable) > 1:
        raise AmbiguousDialectError(
            f"{path} satisfies {rule_label} under several delimiters {viable!r}; bind an "
            "explicit DelimitedFile dialect instead"
        )
    delimiter = viable[0]
    decimal = DelimitedText(path, delimiter).decimal_separator()
    return BoundDelimited(path, DelimitedDialect(delimiter, UngroupedNumbers(decimal)))
