"""Typed source values and the physical dialect they may bind.

The source is a required argument of ``make_parse_strategy``; each variant owns the
resolution of concrete tables for a rule-declared layout, so construction never branches
on which variant it received. A ``DelimitedDialect`` is one delimited file's complete
physical notation; the absence of a thousands separator is the ``UngroupedNumbers``
member, never an empty string or ``str | None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class ReadCsvNumbers(TypedDict, total=False):
    """The numeric arguments one notation contributes to a pandas read."""

    decimal: str
    thousands: str


@dataclass(frozen=True, slots=True)
class UngroupedNumbers:
    """Numeric notation with a decimal mark and no digit grouping."""

    decimal: str

    def __post_init__(self) -> None:
        if not self.decimal:
            raise ValueError("decimal mark must be non-empty")

    def read_csv_options(self) -> ReadCsvNumbers:
        """Ungrouped notation configures only the decimal mark."""
        return {"decimal": self.decimal}


@dataclass(frozen=True, slots=True)
class GroupedNumbers:
    """Numeric notation with a decimal mark and a thousands separator."""

    decimal: str
    thousands: str

    def __post_init__(self) -> None:
        if not self.decimal:
            raise ValueError("decimal mark must be non-empty")
        if not self.thousands:
            raise ValueError("thousands mark must be non-empty")
        if self.decimal == self.thousands:
            raise ValueError(f"decimal and thousands marks must differ, both are {self.decimal!r}")

    def read_csv_options(self) -> ReadCsvNumbers:
        """Grouped notation configures the decimal mark and the thousands separator."""
        return {"decimal": self.decimal, "thousands": self.thousands}


type NumericNotation = UngroupedNumbers | GroupedNumbers


@dataclass(frozen=True, slots=True)
class DelimitedDialect:
    """One delimited file's complete physical notation."""

    delimiter: str
    numbers: NumericNotation

    def __post_init__(self) -> None:
        if not self.delimiter:
            raise ValueError("delimiter must be non-empty")


@dataclass(frozen=True, slots=True)
class SingleFile:
    """One vendor report file; its physical dialect is resolved during construction."""

    path: Path


@dataclass(frozen=True, slots=True)
class DelimitedFile:
    """One delimited vendor report with an explicitly bound dialect.

    The escape hatch for files whose dialect detection fails or is ambiguous: the bound
    dialect is still validated against the file and rule contract during construction.
    """

    path: Path
    dialect: DelimitedDialect


@dataclass(frozen=True, slots=True)
class Folder:
    """A folder satisfying a file-set rule through the rule's declared filenames."""

    root: Path


@dataclass(frozen=True, slots=True)
class FileRoles:
    """Explicit mapping from rule-declared table roles to concrete files.

    Per-role values compose: a role bound as ``DelimitedFile`` carries its own explicit
    dialect, so distinct files in one logical source may use distinct notations.
    """

    tables: Mapping[str, SingleFile | DelimitedFile]


type InputSource = SingleFile | DelimitedFile | Folder | FileRoles
