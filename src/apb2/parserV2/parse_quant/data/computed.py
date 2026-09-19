"""A computed column and its explicit, non-column diagnostic metadata."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ColumnComputation:
    """One row-aligned series and unresolved tokens in first-observed order."""

    values: pl.Series
    unknown_mod_tokens: tuple[str, ...] = ()
