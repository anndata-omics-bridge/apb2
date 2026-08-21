"""Packed fragment lists become scalar rows, before anything else looks at the table.

DIA-NN-style reports pack per-fragment values as delimiter-joined lists inside each precursor
row, aligned by index and often terminated by a trailing delimiter. Separating them is a
physical operation on one table: it produces more rows of the same shape and one synthesized
label column, and then the ordinary long decomposer takes over. Nothing here builds an axis,
normalizes a modification, resolves a duplicate, or converts a number.

The scalar split keeps the vendor semantics exactly: a null or whitespace-only cell holds zero
tokens; outer whitespace and each token's surrounding whitespace are removed; trailing
delimiter terminators go before the split; and an interior empty token stays an empty scalar
at its aligned position. A row with zero tokens contributes no scalar row.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.data.source import LevelSourceTable

_LENGTH = "_packed_length"
_POSITION = "_packed_position"


class PackedLengthError(ValueError):
    """Parallel packed cells in one row do not hold the same number of scalars."""


def _tokens(name: str, delimiter: str) -> pl.Expr:
    """One packed column as a list of trimmed scalars, empty when the cell holds none."""
    trimmed = pl.col(name).cast(pl.String).str.strip_chars().str.strip_chars_end(delimiter)
    return (
        pl.when(trimmed.is_null() | (trimmed == ""))
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(trimmed.str.split(delimiter).list.eval(pl.element().str.strip_chars()))
        .alias(name)
    )


def _require_aligned(frame: pl.DataFrame, packed: tuple[str, ...]) -> None:
    """Parallel packed cells must have equal cardinality; a row that does not is a defect."""
    if len(packed) < 2:
        return
    first = pl.col(packed[0]).list.len()
    mismatched = frame.filter(
        ~pl.all_horizontal([pl.col(name).list.len() == first for name in packed[1:]])
    )
    if mismatched.height:
        lengths = {
            name: mismatched.get_column(name).list.len().head(3).to_list() for name in packed
        }
        raise PackedLengthError(
            f"{mismatched.height} row(s) pack different numbers of fragment scalars across "
            f"{list(packed)}; first rows hold {lengths}"
        )


def _separated(frame: pl.DataFrame, packed: tuple[str, ...], delimiter: str) -> pl.DataFrame:
    """Split every packed column, drop the rows that hold nothing, and explode the rest."""
    listed = frame.with_columns([_tokens(name, delimiter) for name in packed])
    _require_aligned(listed, packed)
    # Rows holding nothing are removed before the explode, so no scalar row is ever
    # invented for a precursor with no fragments.
    populated = listed.filter(pl.col(packed[0]).list.len() > 0)
    return populated.with_columns(
        pl.int_ranges(0, pl.col(packed[0]).list.len()).alias(_POSITION)
    ).explode([*packed, _POSITION], empty_as_null=True)


@dataclass(frozen=True, slots=True)
class PositionalFragmentTableSeparator:
    """Label each scalar by its index within the precursor: ``frag_0``, ``frag_1``, ..."""

    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]

    def separate(self, table: LevelSourceTable, /) -> LevelSourceTable:
        separated = _separated(table.frame, self.packed_value_sources, self.delimiter)
        labelled = separated.with_columns(
            (pl.lit("frag_") + pl.col(_POSITION).cast(pl.String)).alias(self.label_output)
        )
        return LevelSourceTable(frame=labelled.drop(_POSITION))


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentTableSeparator:
    """Take each scalar's label from a packed label column, up to the first ``/``."""

    label_source: str
    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]

    def separate(self, table: LevelSourceTable, /) -> LevelSourceTable:
        packed = (self.label_source, *self.packed_value_sources)
        separated = _separated(table.frame, packed, self.delimiter)
        labelled = separated.with_columns(
            pl.col(self.label_source).str.split("/").list.first().alias(self.label_output)
        )
        # The packed label column has done its work; carrying it would make every scalar row
        # hold a redundant long string.
        return LevelSourceTable(frame=labelled.drop(_POSITION, self.label_source))
