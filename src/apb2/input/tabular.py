"""Recognise a file's tabular format and read its header through that format's class.

One file, one factory, three public classes: ``format_for(path)`` resolves the extension —
and, for ``.txt``, sniffs the separator — and returns a reader initialized with the path,
so every operation is a no-argument method on the file it belongs to. Full-table reads
live in ``input/readers.py`` behind a ``ReadPlan`` projection; this module stops at the
header and the number-format facts detection needs.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_FIXED_DELIMITERS = {".csv": ",", ".tsv": "\t"}
_TEXT_SUFFIX = ".txt"
_PARQUET_SUFFIX = ".parquet"
_TEXT_DELIMITERS = ",\t"
_COMMA_DECIMAL_RE = re.compile(r"^-?\d+,(\d+)$")
_THOUSANDS_GROUP_WIDTH = 3
_DECIMAL_SAMPLE_LINES = 500


class UnknownFormat(ValueError):
    """Raised when a file extension has no registered format."""


@dataclass(frozen=True, slots=True)
class DelimitedText:
    """One delimited text file with its resolved separator."""

    path: Path
    delimiter: str

    def columns(self) -> list[str]:
        """Read only the header row."""
        return list(
            pd.read_csv(self.path, sep=self.delimiter, encoding="utf-8-sig", nrows=0).columns
        )

    def decimal_separator(self) -> str:
        """Detect whether this file writes numbers with a comma decimal mark.

        Vendors export numbers in the regional format of the machine that produced the
        file, and nothing in the file declares which one, so this is inferred from
        content. Only the shape of the number distinguishes the two readings of
        ``1,234``: a thousands separator always groups exactly three digits, so a field
        whose comma is followed by three digits is ambiguous and never counted as
        evidence. A comma-delimited file cannot carry bare comma decimals at all and is
        reported as dot-decimal without inspection.
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


@dataclass(frozen=True, slots=True)
class Parquet:
    """One Parquet file; its physical schema replaces textual dialect resolution."""

    path: Path

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
