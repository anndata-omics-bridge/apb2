"""Parse one aligned measurement layer into final canonical values."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation, as_numbers, blank
from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerTable,
    MeasurementLayerRole,
    NumericLayerType,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.errors import LayerValueError

UNKNOWN_CATEGORY_CODE = -1
_EXAMPLE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class PlainNumericLayerParser:
    """Parse directly readable numeric scalars and declared missing sentinels."""

    layer_name: str
    missing_values: tuple[float, ...]
    number_format: NumberNotation
    numeric_type: NumericLayerType

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        values = _value_block(layer)
        columns = tuple(values.columns)
        if not columns:
            canonical = values
        else:
            number_labels = _temporary_labels("_number", len(columns), reserved=columns)
            mask_labels = _temporary_labels(
                "_unreadable", len(columns), reserved=(*columns, *number_labels)
            )
            prepared = values.with_columns(
                [
                    as_numbers(pl.col(column), values.schema[column], self.number_format).alias(
                        label
                    )
                    for column, label in zip(columns, number_labels, strict=True)
                ]
            ).with_columns(
                [
                    (
                        ~blank(pl.col(column), values.schema[column])
                        & pl.col(number_label).is_null()
                    ).alias(mask_label)
                    for column, number_label, mask_label in zip(
                        columns, number_labels, mask_labels, strict=True
                    )
                ]
            )
            unreadable = [
                str(token)
                for column, mask_label in zip(columns, mask_labels, strict=True)
                for token in prepared.get_column(column)
                .filter(prepared.get_column(mask_label))
                .unique(maintain_order=True)
                .head(_EXAMPLE_LIMIT)
            ]
            if unreadable:
                logger.warning(
                    "layer {!r} declares numeric values; {} distinct unreadable token(s) "
                    "became missing, examples={}",
                    self.layer_name,
                    len(set(unreadable)),
                    sorted(set(unreadable))[:_EXAMPLE_LIMIT],
                )
            canonical = prepared.select(
                [
                    _masked(pl.col(label), self.missing_values).alias(column)
                    for column, label in zip(columns, number_labels, strict=True)
                ]
            )
        return _parsed_layer(
            layer,
            _validate_numeric_type(self.layer_name, canonical, self.numeric_type),
            semantics=QuantitativeLayerSemantics(logical_type=self.numeric_type),
            role=MeasurementLayerRole(),
        )


@dataclass(frozen=True, slots=True)
class RegexNumericLayerParser:
    """Extract one numeric capture from each structured token and parse it."""

    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumberNotation
    numeric_type: NumericLayerType

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        values = _value_block(layer)
        canonical = values.select(
            [
                _masked(
                    as_numbers(
                        pl.col(column).cast(pl.String, strict=False).str.extract(self.pattern, 1),
                        pl.String(),
                        self.number_format,
                    ),
                    self.missing_values,
                ).alias(column)
                for column in values.columns
            ]
        )
        return _parsed_layer(
            layer,
            _validate_numeric_type(self.layer_name, canonical, self.numeric_type),
            semantics=QuantitativeLayerSemantics(logical_type=self.numeric_type),
            role=MeasurementLayerRole(),
        )


@dataclass(frozen=True, slots=True)
class FactorLayerParser:
    """Replace declared category labels with their final integer codes."""

    categories: tuple[tuple[str, int], ...]

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        semantics = CategoricalLayerSemantics(
            categories=self.categories,
            missing_code=UNKNOWN_CATEGORY_CODE,
        )
        mapping = dict(self.categories)
        values = _value_block(layer)
        canonical = values.select(
            [
                pl.col(column)
                .cast(pl.String, strict=False)
                .replace_strict(
                    mapping,
                    default=UNKNOWN_CATEGORY_CODE,
                    return_dtype=pl.Int64,
                )
                .alias(column)
                for column in values.columns
            ]
        )
        return _parsed_layer(
            layer,
            canonical,
            semantics=semantics,
            role=AuxiliaryLayerRole(),
        )


def _value_block(layer: FinalLayerTable, /) -> pl.DataFrame:
    return layer.values.select(layer.values.columns[len(layer.var_key_columns) :])


def _parsed_layer(
    layer: FinalLayerTable,
    values: pl.DataFrame,
    /,
    *,
    semantics: QuantitativeLayerSemantics | CategoricalLayerSemantics,
    role: MeasurementLayerRole | AuxiliaryLayerRole,
) -> FinalLayerTable:
    keys = layer.values.select(list(layer.var_key_columns))
    return FinalLayerTable(
        layer_name=layer.layer_name,
        var_key_columns=layer.var_key_columns,
        values=pl.concat([keys, values], how="horizontal_extend"),
        role=role,
        semantics=semantics,
    )


def _masked(numbers: pl.Expr, missing_values: tuple[float, ...]) -> pl.Expr:
    if not missing_values:
        return numbers
    return pl.when(numbers.is_in(list(missing_values))).then(None).otherwise(numbers)


def _validate_numeric_type(
    layer_name: str,
    values: pl.DataFrame,
    numeric_type: NumericLayerType,
    /,
) -> pl.DataFrame:
    if numeric_type == "number":
        return values
    examples: list[float] = []
    for column in values.get_columns():
        invalid = (
            column.is_not_null()
            & ~column.is_nan().fill_null(False)
            & (~column.is_finite().fill_null(False) | (column != column.floor()).fill_null(False))
        )
        examples.extend(column.filter(invalid).head(_EXAMPLE_LIMIT - len(examples)).to_list())
        if len(examples) == _EXAMPLE_LIMIT:
            break
    if examples:
        raise LayerValueError(
            f"integer layer {layer_name!r} contains fractional or infinite values; "
            f"examples={examples}"
        )
    return values.select(
        [
            pl.when(pl.col(name).is_nan())
            .then(None)
            .otherwise(pl.col(name))
            .cast(pl.Int64, strict=True)
            .alias(name)
            for name in values.columns
        ]
    )


def _temporary_labels(
    prefix: str,
    count: int,
    *,
    reserved: tuple[str, ...],
) -> tuple[str, ...]:
    taken = set(reserved)
    while True:
        labels = tuple(f"{prefix}_{index}" for index in range(count))
        if not taken.intersection(labels):
            return labels
        prefix += "_"
