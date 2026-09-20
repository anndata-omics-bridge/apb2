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

from dataclasses import dataclass, replace

import polars as pl
import polars.selectors as cs

from apb2.parserV2.parse_quant.contracts import RawValuePresence
from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation, absent, as_numbers, blank
from apb2.parserV2.parse_quant.data.raw import RawLayerTable

_EXAMPLE_LIMIT = 5


class DuplicateCellError(ValueError):
    """Several raw scalars claim one measurement cell and the rule forbids that."""


class AggregateTypeError(TypeError):
    """A numeric aggregate received values that are not numbers."""


def _sentinel(numbers: pl.Expr, missing_values: tuple[float, ...]) -> pl.Expr:
    """Whether each readable number is one the vendor writes to mean "not measured"."""
    if not missing_values:
        return pl.lit(value=False)
    return numbers.is_in(list(missing_values)).fill_null(value=False)


class NullOnlyRawValuePresence:
    """Only absence claims nothing: a factor label or a native number needs no interpretation.

    ``NaN`` counts as absence because it is what a float column says instead of null; a
    factor label, including an empty one, is a label and claims its cell.
    """

    __slots__ = ()

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        return ~absent(values, dtype)


@dataclass(frozen=True, slots=True)
class PlainNumericRawValuePresence:
    """Null, blank text, and the declared missing values claim nothing."""

    missing_values: tuple[float, ...]
    number_format: NumberNotation

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        numbers = as_numbers(values, dtype, self.number_format)
        return ~(blank(values, dtype) | _sentinel(numbers, self.missing_values))


@dataclass(frozen=True, slots=True)
class RegexNumericRawValuePresence:
    """As plain numeric, but the comparable number is one capture of a structured token."""

    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumberNotation

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        extracted = values.cast(pl.String, strict=False).str.extract(self.pattern, 1)
        numbers = as_numbers(extracted, pl.String(), self.number_format)
        return ~(blank(values, dtype) | _sentinel(numbers, self.missing_values))


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


class AggregateNumericDuplicates:
    """Claiming scalars are summed; a cell with none stays null rather than becoming zero."""

    __slots__ = ()

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        masked = _masked(layer, presence)
        self._require_numeric(layer)
        values = pl.exclude(layer.raw_var_key_columns)
        summed = masked.group_by(layer.raw_var_key_columns, maintain_order=True).agg(
            pl.when(values.count() > 0).then(values.sum()).otherwise(None)
        )
        return replace(layer, values=summed)

    @staticmethod
    def _require_numeric(layer: RawLayerTable) -> None:
        """Defence in depth: compilation rejects a plan that cannot deliver numbers.

        A malformed file can still deliver text where the rule promised numbers, and summing
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
