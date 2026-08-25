"""Reading values as numbers under the notation they were written in.

Two boundaries need this and must agree: raw presence, which compares a value against a
declared missing sentinel, and AnnData encoding, which turns the value into a float. Both
receive whatever the reader produced — text a vendor localized, or a number the file already
typed — and both must answer the same way about the same cell.

The numeric fast path is not an optimization. Sending a float through its own text form and
back is a lossy detour: a ``Float32`` column round-trips through the shortest repr *of a
float32*, which is not the value pandas would have widened. Numbers stay numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_BOOLEAN_SPELLINGS = {
    "true": "1",
    "false": "0",
    "TRUE": "1",
    "FALSE": "0",
    "True": "1",
    "False": "0",
}
"""A vendor writing ``True``/``False`` in a column its own rule calls numeric means 1 and 0.

Which is what an inferring reader produced from the same file, so reading it any other way
would lose values these rules were written to keep.
"""


@dataclass(frozen=True, slots=True)
class NumberNotation:
    """How the tokens being read were written down."""

    decimal_mark: str
    thousands_marks: tuple[str, ...]


def as_numbers(
    values: pl.Expr,
    dtype: pl.DataType,
    notation: NumberNotation,
) -> pl.Expr:
    """Read one column expression as numbers under its declared physical dtype and notation."""
    if dtype.is_numeric():
        return values.cast(pl.Float64, strict=False)
    text = values.cast(pl.String, strict=False).replace(_BOOLEAN_SPELLINGS)
    for mark in notation.thousands_marks:
        text = text.str.replace_all(mark, "", literal=True)
    if notation.decimal_mark != ".":
        text = text.str.replace_all(notation.decimal_mark, ".", literal=True)
    return text.cast(pl.Float64, strict=False)


def blank(values: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Whether each value holds nothing: null, ``NaN``, or text that is only whitespace."""
    if dtype.is_numeric():
        return absent(values, dtype)
    text = values.cast(pl.String, strict=False).str.strip_chars()
    return text.is_null() | (text == "")


def absent(values: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Whether each value is missing, counting ``NaN`` as the float spelling of null.

    A vendor writing ``NaN`` in a measurement column has written "not a number", and summing
    it would poison the whole cell rather than reporting one absent contribution.
    """
    if dtype.is_float():
        return values.is_null() | values.is_nan().fill_null(value=True)
    return values.is_null()
