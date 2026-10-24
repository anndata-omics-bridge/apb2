"""Two questions about one raw measurement cell, deliberately kept apart.

*Presence* asks whether a raw scalar claims its cell — is it missing, is it the sentinel the
vendor writes for "not measured"? The layer's value parser answers it, and the parser nulls
every absent scalar before any policy here runs. Nothing here converts a value, because
converting is the value parser's job and doing it here would decide, silently, which value
survived.

*Resolution* asks how several claiming scalars become one. A policy receives only the placed
rows that repeat a cell, groups them by (``var_pos``, ``obs_pos``), and copies the scalar it
selected through unchanged. A token that cannot be read stays present on purpose: keep-first
must not be able to hide a value that will fail to encode later.

The feature-value policies at the end ask the resolution question of a ``var`` column: which
of a feature's repeated rows gives its one value, the first present one, an extreme, or the one
another column ranks best, and whether the rows disagreed.

What is *not* here: any comparison of final keys. Two different raw keys that canonicalize to
one final key are an information loss, not a duplicate, and axis preparation rejects them
before any of this runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import polars as pl

from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS
from apb2.parserV2.parse_quant.data.raw import LongLayer
from apb2.parserV2.parse_quant.numeric_text import absent, as_numbers
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

_EXAMPLE_LIMIT = 5
_CLAIMS = "claims"
_RANK = "rank"


class DuplicateCellError(ConversionError):
    """Several raw scalars claim one measurement cell and the rule forbids that."""


class AggregateTypeError(ConversionError, TypeError):
    """A numeric aggregate received values that are not numbers."""


def _cells(layer: LongLayer, *values: pl.Expr) -> pl.DataFrame:
    """One row per placed cell, in first-appearance order, with the given aggregations."""
    return layer.rows.group_by(VAR_POS, OBS_POS, maintain_order=True).agg(*values)


class ErrorOnDuplicates:
    """More than one claiming scalar in one cell is a rule error, not a value to choose."""

    __slots__ = ()

    def resolve(self, layer: LongLayer, /) -> LongLayer:
        cells = _cells(
            layer,
            pl.col(VALUE).first(ignore_nulls=True),
            pl.col(VALUE).count().alias(_CLAIMS),
        )
        offending = cells.filter(pl.col(_CLAIMS) > 1).get_column(VAR_POS).unique().sort()
        if offending.len():
            examples = layer.var_keys[offending.head(_EXAMPLE_LIMIT)].to_dicts()
            raise DuplicateCellError(
                f"layer {layer.layer_name!r}: {offending.len()} raw key(s) claim one "
                f"measurement cell more than once; examples: {examples}"
            )
        return replace(layer, rows=cells.drop(_CLAIMS))


class KeepFirstDuplicate:
    """The first claiming scalar wins, in physical order."""

    __slots__ = ()

    def resolve(self, layer: LongLayer, /) -> LongLayer:
        return replace(layer, rows=_cells(layer, pl.col(VALUE).first(ignore_nulls=True)))


@dataclass(frozen=True, slots=True)
class AggregateNumericDuplicates:
    """Claiming scalars reduce to one number; a cell with none stays null rather than zero.

    ``pl.Expr.sum`` adds the claiming scalars into a new one; ``pl.Expr.max`` selects the
    largest of them. Every layer reduces on its own, so with several layers one feature's
    cells may come from different repeated rows.
    """

    reduction: Callable[[pl.Expr], pl.Expr]

    def resolve(self, layer: LongLayer, /) -> LongLayer:
        dtype = layer.rows.schema[VALUE]
        if not (dtype.is_numeric() or dtype == pl.Null):
            # Defence in depth: compilation rejects a plan that cannot deliver numbers, but a
            # malformed file can still deliver text where the rule promised numbers.
            raise AggregateTypeError(
                f"layer {layer.layer_name!r} aggregates duplicate cells, which needs numeric "
                f"values; its values are {dtype}"
            )
        value = pl.col(VALUE)
        return replace(
            layer,
            rows=_cells(layer, pl.when(value.count() > 0).then(self.reduction(value))),
        )


@dataclass(frozen=True, slots=True)
class KeepBestDuplicate:
    """keep_best before ranking: which layer ranks repeated rows, and whether high wins.

    It resolves nothing itself. The parser first ranks the ``by`` layer once (``ranked``),
    and the resulting policy then resolves every layer from the same winning rows.
    """

    by: str
    highest: bool
    summed: frozenset[str] = frozenset()

    def ranked(self, ranking: LongLayer) -> RankedDuplicates:
        """Read the ranking layer's numbers; text that is not a plain number is an error.

        Every claimed value is read, not only the repeated ones, so a single cell's bad
        ranking value fails just as a repeated one does.
        """
        try:
            ranks = ranking.rows.get_column(VALUE).cast(pl.Float64, strict=True).fill_nan(None)
        except pl.exceptions.InvalidOperationError as error:
            raise AggregateTypeError(
                f"keep_best ranks by layer {self.by!r}, whose values are not plain numbers: {error}"
            ) from error
        repeated = ranking.repeats()
        return RankedDuplicates(
            by=self.by,
            highest=self.highest,
            positions=ranking.rows.select(VAR_POS, OBS_POS).filter(repeated),
            ranks=ranks.filter(repeated),
            summed=self.summed,
        )


@dataclass(frozen=True, slots=True)
class RankedDuplicates:
    """keep_best after ranking: in each cell, the repeated row whose ranking value is best.

    Every layer copies the winning row's scalar, so all layers of one cell describe one
    source row; a ``summed`` layer adds up all its repeated rows instead. A row without a
    ranking value loses to any row with one; ties, and cells where no row has one, keep file
    order.
    """

    by: str
    highest: bool
    positions: pl.DataFrame
    ranks: pl.Series
    summed: frozenset[str] = frozenset()

    def resolve(self, layer: LongLayer, /) -> LongLayer:
        if layer.layer_name in self.summed:
            return AggregateNumericDuplicates(pl.Expr.sum).resolve(layer)
        if not layer.rows.select(VAR_POS, OBS_POS).equals(self.positions):
            raise ConversionError(
                f"layer {layer.layer_name!r} does not repeat the rows of ranking layer {self.by!r}"
            )
        rank = pl.col(_RANK)
        ranked = replace(layer, rows=layer.rows.with_columns(self.ranks.alias(_RANK)))
        return replace(
            layer,
            rows=_cells(
                ranked,
                pl.col(VALUE)
                .sort_by(
                    [rank.is_null(), rank], descending=[False, self.highest], maintain_order=True
                )
                .first(),
            ),
        )


@dataclass(frozen=True, slots=True)
class FirstFeatureValue:
    """A feature's first present value; rows that give another value are a disagreement."""

    source: str

    def value(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        present = ~absent(pl.col(self.source), schema[self.source])
        return pl.col(self.source).filter(present).first()

    def disagrees(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        present = ~absent(pl.col(self.source), schema[self.source])
        return pl.col(self.source).filter(present).n_unique() > 1


@dataclass(frozen=True, slots=True)
class ExtremeFeatureValue:
    """A feature's smallest or largest number, kept as the vendor wrote it.

    For a tool that writes a placeholder into some of a feature's rows, such as DIA-NN's
    protein-group q-value of 1.0 on precursors without one; such rows disagree by design.
    """

    source: str
    largest: bool
    notation: NumericTextFormat

    def value(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        number = _rank(self.source, schema, self.notation)
        return (
            pl.col(self.source)
            .sort_by(number, descending=self.largest, nulls_last=True, maintain_order=True)
            .first()
        )

    def disagrees(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        return pl.lit(value=False)


@dataclass(frozen=True, slots=True)
class RankedFeatureValue:
    """The value of the feature's row whose ``by`` number ranks best, as the vendor wrote it.

    For columns that describe one of several rows a feature repeats, such as the protein set
    a shared peptide is listed under: every column ranked by the same score takes its value
    from the same row, even an absent one. Ties and rows without a rank keep file order.
    """

    source: str
    by: str
    highest: bool
    notation: NumericTextFormat

    def value(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        rank = _rank(self.by, schema, self.notation)
        return (
            pl.col(self.source)
            .sort_by(rank, descending=self.highest, nulls_last=True, maintain_order=True)
            .first()
        )

    def disagrees(self, schema: Mapping[str, pl.DataType], /) -> pl.Expr:
        return pl.lit(value=False)


def _rank(source: str, schema: Mapping[str, pl.DataType], notation: NumericTextFormat) -> pl.Expr:
    """One column's values as numbers; ``NaN`` ranks as absent, not above every number."""
    return as_numbers(pl.col(source), schema[source], notation).fill_nan(None)
