"""Explode packed parallel-list fragment columns into one row per fragment.

DIA-NN-style reports pack per-fragment values as delimiter-joined lists inside each
precursor row (``Fragment.Info`` plus parallel ``Fragment.Quant.*`` lists, aligned by
index, often terminated by a trailing delimiter). A fragment-level AnnData needs one row
per fragment, so these helpers split the packed columns and coerce the exploded values.

The explosion itself lives on the two exploder types in ``converters/preprocess.py``,
because the positional and column-labelled strategies do genuinely different work — a
single function branched on ``label_strategy`` twice, once to choose the packed columns
and again to explode them.
"""

from __future__ import annotations

import math

import pandas as pd


def _split_packed(value: object, delimiter: str) -> list[str]:
    """Split one packed cell into tokens, dropping a trailing empty terminator."""
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    ):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.rstrip(delimiter)  # drop DIA-NN's trailing-delimiter terminator
    if not text:
        return []
    return [token.strip() for token in text.split(delimiter)]


def split_packed_columns(
    df: pd.DataFrame,
    packed_columns: tuple[str, ...],
    delimiter: str,
) -> pd.DataFrame:
    """Copy ``df`` with each packed column split into a token list, rejecting absent ones."""
    missing = [column for column in packed_columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"[fragments] references column(s) missing from the input: {missing}; "
            f"available: {list(df.columns)[:10]}…"
        )
    work = df.copy()
    for column in packed_columns:
        work[column] = work[column].map(lambda value: _split_packed(value, delimiter))
    return work


def coerce_fragment_values(df: pd.DataFrame, value_columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce exploded values to numeric so they ride the frame as float64, not strings."""
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fragment_positions(tokens: list[str]) -> list[int]:
    """Return positions for the token list produced by ``_split_packed``."""
    return list(range(len(tokens)))
