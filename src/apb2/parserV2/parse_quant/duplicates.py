"""Two questions about one raw measurement cell, deliberately kept apart.

*Presence* asks whether a raw scalar claims its cell — is it null, is it blank text, is it the
sentinel the vendor writes for "not measured"? It answers with a Boolean mask and nothing
else. It never converts a value, because converting is the storage boundary's job and doing it
here would decide, silently, which value survived.

*Resolution* asks how several claiming scalars become one. It groups by the raw var keys only,
resolves each observation column independently, and copies the scalar it selected through
unchanged. A nonblank token that cannot be read stays present on purpose: keep-first must not
be able to hide a value that will fail to encode later.

What is *not* here: any comparison of final keys. Two different raw keys that canonicalize to
one final key are an information loss, not a duplicate, and axis preparation rejects them
before any of this runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl
import polars.selectors as cs

from apb2.parserV2.parse_quant.contracts import RawValuePresence
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.raw import RawLayerTable

_EXAMPLE_LIMIT = 5


class DuplicateCellError(ConversionError):
    """Several raw scalars claim one measurement cell and the rule forbids that."""


class AggregateTypeError(ConversionError, TypeError):
    """A numeric aggregate received values that are not numbers."""


def _masked(layer: RawLayerTable, presence: RawValuePresence) -> pl.DataFrame:
    """Null out absent scalars, retaining the original dtype and every claiming token."""
    return layer.values.with_columns(
        pl.when(presence.present(pl.col(name), dtype))
        .then(pl.col(name))
        .otherwise(None)
        .alias(name)
        for name, dtype in layer.values.select(pl.exclude(layer.raw_var_key_columns)).schema.items()
    )


class ErrorOnDuplicates:
    """More than one claiming scalar in one cell is a rule error, not a value to choose."""

    __slots__ = ()

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        masked = _masked(layer, presence)
        keys = layer.raw_var_key_columns
        duplicate = "_duplicate"
        while duplicate in masked.columns:
            duplicate += "_"
        values = pl.exclude(keys)
        resolved = masked.group_by(keys, maintain_order=True).agg(
            values.first(ignore_nulls=True),
            pl.any_horizontal(pl.lit(False), values.count() > 1).alias(duplicate),
        )
        offending = resolved.filter(pl.col(duplicate))
        if offending.height:
            examples = offending.select(keys).head(_EXAMPLE_LIMIT).to_dicts()
            raise DuplicateCellError(
                f"layer {layer.layer_name!r}: {offending.height} raw key(s) claim one "
                f"measurement cell more than once; examples: {examples}"
            )
        return replace(layer, values=resolved.drop(duplicate))


class KeepFirstDuplicate:
    """The first claiming scalar wins, independently per observation column."""

    __slots__ = ()

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        return replace(
            layer,
            values=_masked(layer, presence)
            .group_by(layer.raw_var_key_columns, maintain_order=True)
            .first(ignore_nulls=True),
        )


@dataclass(frozen=True, slots=True)
class AggregateNumericDuplicates:
    """Claiming scalars reduce to one number; a cell with none stays null rather than zero.

    ``pl.Expr.sum`` adds the claiming scalars into a new one; ``pl.Expr.max`` selects the
    largest of them. Every layer and observation column reduces on its own, so with several
    layers one feature's cells may come from different repeated rows.
    """

    reduction: Callable[[pl.Expr], pl.Expr]

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        masked = _masked(layer, presence)
        self._require_numeric(layer)
        values = pl.exclude(layer.raw_var_key_columns)
        reduced = masked.group_by(layer.raw_var_key_columns, maintain_order=True).agg(
            pl.when(values.count() > 0).then(self.reduction(values)).otherwise(None)
        )
        return replace(layer, values=reduced)

    @staticmethod
    def _require_numeric(layer: RawLayerTable) -> None:
        """Defence in depth: compilation rejects a plan that cannot deliver numbers.

        A malformed file can still deliver text where the rule promised numbers, and reducing
        text has no defined answer, so this fails at its own boundary rather than inventing
        one.
        """
        offenders = sorted(
            layer.values.select(
                ~(cs.by_name(layer.raw_var_key_columns) | cs.numeric() | cs.by_dtype(pl.Null))
            ).columns
        )
        if offenders:
            raise AggregateTypeError(
                f"layer {layer.layer_name!r} aggregates duplicate cells, which needs numeric "
                f"values; these columns hold {layer.values.schema[offenders[0]]}: {offenders}"
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

    def ranked(self, ranking: RawLayerTable, presence: RawValuePresence) -> RankedDuplicates:
        """Read the ranking layer's numbers; text that is not a plain number is an error."""
        masked = _masked(ranking, presence)
        keys = ranking.raw_var_key_columns
        try:
            ranks = masked.select(pl.exclude(keys).cast(pl.Float64, strict=True).fill_nan(None))
        except pl.exceptions.InvalidOperationError as error:
            raise AggregateTypeError(
                f"keep_best ranks by layer {self.by!r}, whose values are not plain numbers: {error}"
            ) from error
        return RankedDuplicates(
            by=self.by,
            highest=self.highest,
            keys=masked.select(keys),
            ranks=ranks,
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
    keys: pl.DataFrame
    ranks: pl.DataFrame
    summed: frozenset[str] = frozenset()

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        if layer.layer_name in self.summed:
            return AggregateNumericDuplicates(pl.Expr.sum).resolve(layer, presence)
        keys = layer.raw_var_key_columns
        masked = _masked(layer, presence)
        columns = [name for name in masked.columns if name not in keys]
        if not masked.select(keys).equals(self.keys) or not set(columns) <= set(self.ranks.columns):
            raise ConversionError(
                f"layer {layer.layer_name!r} does not repeat the rows of ranking layer {self.by!r}"
            )
        taken = set(masked.columns)
        ranks: dict[str, str] = {}
        for name in columns:
            alias = f"_rank_{name}"
            while alias in taken:
                alias += "_"
            taken.add(alias)
            ranks[name] = alias
        combined = masked.hstack(
            self.ranks.select(pl.col(name).alias(alias) for name, alias in ranks.items())
        )
        resolved = combined.group_by(keys, maintain_order=True).agg(
            pl.col(name)
            .sort_by(
                [pl.col(alias).is_null(), pl.col(alias)],
                descending=[False, self.highest],
                maintain_order=True,
            )
            .first()
            for name, alias in ranks.items()
        )
        return replace(layer, values=resolved)
