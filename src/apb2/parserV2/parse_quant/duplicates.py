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

from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.contracts import RawValuePresence
from apb2.parserV2.parse_quant.data.layer_columns import presence_labels
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


@dataclass(frozen=True, slots=True)
class NullOnlyRawValuePresence:
    """Only absence claims nothing: a factor label or a native number needs no interpretation.

    ``NaN`` counts as absence because it is what a float column says instead of null; a
    factor label, including an empty one, is a label and claims its cell.
    """

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


@dataclass(frozen=True, slots=True)
class _MaskedLayer:
    """One raw layer beside its presence masks, for the length of one resolution."""

    frame: pl.DataFrame
    keys: list[str]
    values: tuple[str, ...]
    masks: tuple[str, ...]

    def grouped(self, aggregations: list[pl.Expr]) -> pl.DataFrame:
        """Reduce each raw-key group, keeping the groups in the order they first appeared."""
        return self.frame.group_by(self.keys, maintain_order=True).agg(aggregations)


def _masked(layer: RawLayerTable, presence: RawValuePresence) -> _MaskedLayer:
    """Ask the layer's presence strategy which of its scalars claim their cells."""
    keys = list(layer.raw_var_key_columns)
    values = tuple(layer.values.columns[len(keys) :])
    masks = presence_labels(len(values), reserved=layer.values.columns)
    frame = layer.values.with_columns(
        [
            presence.present(pl.col(value), layer.values.schema[value]).alias(mask)
            for value, mask in zip(values, masks, strict=True)
        ]
    )
    return _MaskedLayer(frame=frame, keys=keys, values=values, masks=masks)


def _first_present(masked: _MaskedLayer) -> pl.DataFrame:
    """The first claiming scalar of each cell, copied through unchanged."""
    return masked.grouped(
        [
            pl.col(value).filter(pl.col(mask)).first().alias(value)
            for value, mask in zip(masked.values, masked.masks, strict=True)
        ]
    )


@dataclass(frozen=True, slots=True)
class ErrorOnDuplicates:
    """More than one claiming scalar in one cell is a rule error, not a value to choose."""

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        masked = _masked(layer, presence)
        counts = masked.grouped([pl.col(mask).sum().alias(mask) for mask in masked.masks])
        offending = counts.filter(
            pl.any_horizontal([pl.col(mask) > 1 for mask in masked.masks])
            if masked.masks
            else pl.lit(value=False)
        )
        if offending.height:
            examples = offending.select(masked.keys).head(_EXAMPLE_LIMIT).to_dicts()
            raise DuplicateCellError(
                f"layer {layer.layer_name!r}: {offending.height} raw key(s) claim one "
                f"measurement cell more than once; examples: {examples}"
            )
        return RawLayerTable(
            layer_name=layer.layer_name,
            raw_var_key_columns=layer.raw_var_key_columns,
            values=_first_present(masked),
        )


@dataclass(frozen=True, slots=True)
class KeepFirstDuplicate:
    """The first claiming scalar wins, independently per observation column."""

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        return RawLayerTable(
            layer_name=layer.layer_name,
            raw_var_key_columns=layer.raw_var_key_columns,
            values=_first_present(_masked(layer, presence)),
        )


@dataclass(frozen=True, slots=True)
class AggregateNumericDuplicates:
    """Claiming scalars are summed; a cell with none stays null rather than becoming zero."""

    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        masked = _masked(layer, presence)
        self._require_numeric(layer, masked)
        summed = masked.grouped(
            [
                pl.when(pl.col(mask).any())
                .then(pl.col(value).filter(pl.col(mask)).sum())
                .otherwise(None)
                .alias(value)
                for value, mask in zip(masked.values, masked.masks, strict=True)
            ]
        )
        return RawLayerTable(
            layer_name=layer.layer_name,
            raw_var_key_columns=layer.raw_var_key_columns,
            values=summed,
        )

    @staticmethod
    def _require_numeric(layer: RawLayerTable, masked: _MaskedLayer) -> None:
        """Defence in depth: compilation rejects a plan that cannot deliver numbers.

        A malformed file can still deliver text where the rule promised numbers, and summing
        text has no defined answer, so this fails at its own boundary rather than inventing
        one.
        """
        offenders = sorted(
            name
            for name in masked.values
            if not (masked.frame.schema[name].is_numeric() or masked.frame.schema[name] == pl.Null)
        )
        if offenders:
            raise AggregateTypeError(
                f"layer {layer.layer_name!r} aggregates duplicate cells, which needs numeric "
                f"values; these columns hold {masked.frame.schema[offenders[0]]}: {offenders}"
            )
