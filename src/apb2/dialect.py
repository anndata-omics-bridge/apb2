"""Concrete physical dialect of one delimited resource.

The absence of a thousands separator is the ``UngroupedNumbers`` member, never an empty
string or ``str | None``: one concrete dialect per delimited resource is the runtime
invariant. Whether a dialect fits an actual file and rule is judged jointly over the
``(delimiter, numbers)`` pair during construction (plan stage 3), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
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
