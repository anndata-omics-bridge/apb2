"""Packed fragment lists become scalar rows, in one streaming explode the decomposer runs.

DIA-NN-style reports pack per-fragment values as delimiter-joined lists inside each precursor
row, aligned by index and often terminated by a trailing delimiter. Separating them is a
physical operation on one table: ``token_counts`` counts each row's scalars without splitting
anything, and ``exploded`` turns each list position into a row carrying one label column. The
decomposer replaces the key text by row numbers first, so the explode multiplies integers
rather than strings. Nothing here builds an axis, normalizes a modification or resolves a
duplicate.

The scalar split keeps the vendor semantics exactly: a null or whitespace-only cell holds zero
tokens; outer whitespace and each token's surrounding whitespace are removed; trailing
delimiter terminators go before the split; and an interior empty token stays an empty scalar
at its aligned position, where it is a missing scalar. A row with zero tokens contributes no
scalar row.

A packed column of a plain numeric layer is read as numbers inside its lists, before the
explode, so 390 million scalars never exist as text. A column holding a token no number reads,
or the text ``NaN``, is exploded as text instead, so it reaches the parser exactly as an
unconverted layer does — the delimited reader's rule for scalar columns. The label travels as
its list position, or as a Categorical ``Fragment.Info`` label, and becomes text only on the
feature axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.numeric_text import as_numbers
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

_POSITION_LABEL_PREFIX = "frag_"
_NEEDS_TEXT = "_needs_text_"
_PLAIN_NUMBERS = NumericTextFormat(decimal_mark=".", thousands_marks=())


class PackedLengthError(ConversionError):
    """Parallel packed cells in one row do not hold the same number of scalars."""


def _trimmed(name: str, delimiter: str) -> pl.Expr:
    """One packed cell without its outer whitespace and trailing terminators."""
    return pl.col(name).cast(pl.String).str.strip_chars().str.strip_chars_end(delimiter)


def _count(name: str, delimiter: str) -> pl.Expr:
    """How many scalars one packed cell holds, without splitting it."""
    trimmed = _trimmed(name, delimiter)
    return (
        pl.when(trimmed.is_null() | (trimmed == ""))
        .then(pl.lit(0, dtype=pl.UInt32))
        .otherwise(trimmed.str.count_matches(delimiter, literal=True).cast(pl.UInt32) + 1)
        .alias(name)
    )


def _tokens(name: str, delimiter: str) -> pl.Expr:
    """One packed cell as its list of tokens, each without surrounding whitespace."""
    return _trimmed(name, delimiter).str.split(delimiter).list.eval(pl.element().str.strip_chars())


def _text(name: str, delimiter: str) -> pl.Expr:
    """The tokens as text; a token without a non-whitespace character is a missing scalar."""
    token = pl.element()
    return _tokens(name, delimiter).list.eval(pl.when(token != "").then(token)).alias(name)


@dataclass(frozen=True, slots=True)
class _PackedColumns:
    """The parallel packed columns of one table, how they are split and which read as numbers."""

    delimiter: str
    sources: tuple[str, ...]
    values: tuple[str, ...]
    converted: frozenset[str]
    notation: NumericTextFormat

    def token_counts(self, frame: pl.DataFrame) -> pl.Series:
        """Each row's scalar count; parallel packed cells must agree, or the file is defective."""
        counts = frame.select(_count(name, self.delimiter) for name in self.sources)
        first, others = pl.col(self.sources[0]), self.sources[1:]
        if not others:
            return counts.get_column(self.sources[0])
        mismatched = counts.filter(~pl.all_horizontal(pl.col(name) == first for name in others))
        if mismatched.height:
            lengths = mismatched.head(3).to_dict(as_series=False)
            raise PackedLengthError(
                f"{mismatched.height} row(s) pack different numbers of fragment scalars across "
                f"{list(self.sources)}; first rows hold {lengths}"
            )
        return counts.get_column(self.sources[0])

    def exploded(self, rows: pl.DataFrame, label: str, labels: pl.Expr) -> pl.DataFrame:
        """One row per scalar, the carried columns repeated, ``label`` from ``labels``.

        Converted columns that need their text are exploded again as text, the whole table at
        once, as the delimited reader re-reads such a column.
        """
        flagged = self._explode(rows, label, labels, self.converted)
        needs_text = frozenset(
            name for name in self.converted if flagged.get_column(_NEEDS_TEXT + name).any()
        )
        if not needs_text:
            return flagged.drop(_NEEDS_TEXT + name for name in self.converted)
        del flagged
        converted = self.converted - needs_text
        return self._explode(rows, label, labels, converted).drop(
            _NEEDS_TEXT + name for name in converted
        )

    def _explode(
        self, rows: pl.DataFrame, label: str, labels: pl.Expr, converted: frozenset[str]
    ) -> pl.DataFrame:
        carried = [name for name in rows.columns if name not in self.sources]
        token = pl.element()
        listed: list[pl.Expr] = []
        flags: list[pl.Expr] = []
        for name in self.values:
            if name not in converted:
                listed.append(_text(name, self.delimiter))
                continue
            number = as_numbers(token, pl.String(), self.notation)
            listed.append(_tokens(name, self.delimiter).list.eval(number).alias(name))
            needs_text = token.str.contains(r"\S") & (number.is_null() | number.is_nan())
            flags.append(
                _tokens(name, self.delimiter).list.eval(needs_text).alias(_NEEDS_TEXT + name)
            )
        exploded = [*self.values, *(_NEEDS_TEXT + name for name in converted), label]
        return (
            rows.lazy()
            .filter(_count(self.sources[0], self.delimiter) > 0)
            .select(*carried, *listed, *flags, labels.alias(label))
            .explode(exploded, empty_as_null=True)
            .collect(engine="streaming")
        )


@dataclass(frozen=True, slots=True)
class PositionalFragmentTableSeparator:
    """Label each scalar by its index within the precursor: ``frag_0``, ``frag_1``, ...

    The index travels as a small integer and is spelled out only on the feature axis.
    """

    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]
    converted_sources: frozenset[str] = frozenset()
    notation: NumericTextFormat = field(default=_PLAIN_NUMBERS)

    @property
    def packed_sources(self) -> tuple[str, ...]:
        """The packed columns split in parallel."""
        return self.packed_value_sources

    def token_counts(self, frame: pl.DataFrame, /) -> pl.Series:
        return self._columns.token_counts(frame)

    def exploded(self, rows: pl.DataFrame, /) -> pl.DataFrame:
        first = _count(self.packed_value_sources[0], self.delimiter)
        positions = pl.int_ranges(0, first, dtype=pl.UInt16)
        return self._columns.exploded(rows, self.label_output, positions)

    def labelled(self, fragments: pl.DataFrame, /) -> pl.DataFrame:
        """The feature axis's label column, spelled ``frag_<index>``."""
        label = pl.col(self.label_output)
        return fragments.with_columns(
            (pl.lit(_POSITION_LABEL_PREFIX) + label.cast(pl.String)).alias(self.label_output)
        )

    @property
    def _columns(self) -> _PackedColumns:
        return _PackedColumns(
            self.delimiter,
            self.packed_value_sources,
            self.packed_value_sources,
            self.converted_sources,
            self.notation,
        )


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentTableSeparator:
    """Take each scalar's label from a packed label column, up to the first ``/``.

    The label travels as a Categorical and becomes text on the feature axis.
    """

    label_source: str
    label_output: str
    delimiter: str
    packed_value_sources: tuple[str, ...]
    converted_sources: frozenset[str] = frozenset()
    notation: NumericTextFormat = field(default=_PLAIN_NUMBERS)

    @property
    def packed_sources(self) -> tuple[str, ...]:
        """The label column and the value columns, split in parallel."""
        return (self.label_source, *self.packed_value_sources)

    def token_counts(self, frame: pl.DataFrame, /) -> pl.Series:
        return self._columns.token_counts(frame)

    def exploded(self, rows: pl.DataFrame, /) -> pl.DataFrame:
        token = pl.element()
        present = pl.when(token.str.contains(r"\S")).then(token)
        label = present.str.split("/").list.first()
        labels = _tokens(self.label_source, self.delimiter).list.eval(
            pl.when(label.str.contains(r"\S")).then(label).cast(pl.Categorical)
        )
        return self._columns.exploded(rows, self.label_output, labels)

    def labelled(self, fragments: pl.DataFrame, /) -> pl.DataFrame:
        """The feature axis's label column as text."""
        return fragments.with_columns(pl.col(self.label_output).cast(pl.String))

    @property
    def _columns(self) -> _PackedColumns:
        return _PackedColumns(
            self.delimiter,
            self.packed_sources,
            self.packed_value_sources,
            self.converted_sources,
            self.notation,
        )
