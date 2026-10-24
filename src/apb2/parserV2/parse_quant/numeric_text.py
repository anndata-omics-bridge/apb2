"""Reading values as numbers under the notation they were written in.

Two phases need this and must agree: raw presence, which compares a value against a
declared missing sentinel, and canonical value parsing, which turns the value into a float. Both
receive whatever the reader produced — text a vendor localized, or a number the file already
typed — and both must answer the same way about the same cell.

The numeric fast path is not an optimization. Sending a float through its own text form and
back is a lossy detour: a ``Float32`` column round-trips through the shortest repr *of a
float32*, which is not the value pandas would have widened. Numbers stay numbers.
"""

from __future__ import annotations

import polars as pl

from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

_BOOLEAN_SPELLINGS = (["true", "TRUE", "True"], ["false", "FALSE", "False"])
"""A vendor writing ``True``/``False`` in a column its own rule calls numeric means 1 and 0.

Which is what an inferring reader produced from the same file, so reading it any other way
would lose values these rules were written to keep.
"""


def as_numbers(
    values: pl.Expr,
    dtype: pl.DataType,
    notation: NumericTextFormat,
) -> pl.Expr:
    """Read one column expression as numbers under its declared physical dtype and notation.

    The boolean spellings are matched elementwise after the cast: ``replace`` would join the
    whole column against its mapping, which costs more than the column itself.
    """
    if dtype.is_numeric():
        return values.cast(pl.Float64, strict=False)
    text = values.cast(pl.String, strict=False)
    written = text
    for mark in notation.thousands_marks:
        written = written.str.replace_all(mark, "", literal=True)
    if notation.decimal_mark != ".":
        written = written.str.replace_all(notation.decimal_mark, ".", literal=True)
    true, false = _BOOLEAN_SPELLINGS
    return (
        pl.when(text.is_in([*true, *false]))
        .then(text.is_in(true).cast(pl.Float64))
        .otherwise(written.cast(pl.Float64, strict=False))
    )


def absent(values: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Whether each value is missing, counting ``NaN`` as the float spelling of null.

    A source table already holds every absent cell as null; a ``NaN`` can still arise from
    reading the text ``NaN`` as a number, and summing it would poison the whole cell rather
    than reporting one absent contribution.
    """
    if dtype.is_float():
        return values.is_null() | values.is_nan().fill_null(value=True)
    return values.is_null()
