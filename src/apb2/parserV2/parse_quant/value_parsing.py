"""Parse one aligned measurement layer into final canonical values."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation, absent, as_numbers, blank
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

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Without sentinels blank text claims a cell; unreadable tokens always do."""
        if not self.missing_values:
            return ~absent(values, dtype)
        numbers = as_numbers(values, dtype, self.number_format)
        sentinel = numbers.is_in(self.missing_values).fill_null(False)
        return ~(blank(values, dtype) | sentinel)

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        values = _value_block(layer)
        if not values.width:
            canonical = values
        else:
            prepared = values.select(
                pl.struct(
                    pl.col(column).alias("raw"),
                    as_numbers(pl.col(column), dtype, self.number_format).alias("number"),
                    blank(pl.col(column), dtype).alias("blank"),
                ).alias(column)
                for column, dtype in values.schema.items()
            )
            unreadable = (
                prepared.select(
                    pl.concat_list(
                        pl.all()
                        .struct.field("raw")
                        .cast(pl.String)
                        .filter(
                            ~pl.all().struct.field("blank")
                            & pl.all().struct.field("number").is_null()
                        )
                        .unique(maintain_order=True)
                        .head(_EXAMPLE_LIMIT)
                        .implode()
                    )
                    .list.explode(empty_as_null=False)
                    .unique()
                    .sort()
                )
                .to_series()
                .to_list()
            )
            if unreadable:
                logger.warning(
                    "layer {!r} declares numeric values; {} distinct unreadable token(s) "
                    "became missing, examples={}",
                    self.layer_name,
                    len(unreadable),
                    unreadable[:_EXAMPLE_LIMIT],
                )
            canonical = prepared.select(
                _masked(pl.all().struct.field("number"), self.missing_values).name.keep()
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

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Inspect the numeric capture while retaining each original claiming token."""
        extracted = values.cast(pl.String, strict=False).str.extract(self.pattern, 1)
        numbers = as_numbers(extracted, pl.String(), self.number_format)
        sentinel = numbers.is_in(self.missing_values).fill_null(False)
        return ~(blank(values, dtype) | sentinel)

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        values = _value_block(layer)
        canonical = values.select(
            _masked(
                as_numbers(
                    pl.all().cast(pl.String, strict=False).str.extract(self.pattern, 1),
                    pl.String(),
                    self.number_format,
                ),
                self.missing_values,
            ).name.keep()
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

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Every non-missing label claims a cell, including blank or unknown labels."""
        return ~absent(values, dtype)

    def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
        semantics = CategoricalLayerSemantics(
            categories=self.categories,
            missing_code=UNKNOWN_CATEGORY_CODE,
        )
        mapping = dict(self.categories)
        values = _value_block(layer)
        canonical = values.select(
            pl.all()
            .cast(pl.String, strict=False)
            .replace_strict(
                mapping,
                default=UNKNOWN_CATEGORY_CODE,
                return_dtype=pl.Int64,
            )
        )
        return _parsed_layer(
            layer,
            canonical,
            semantics=semantics,
            role=AuxiliaryLayerRole(),
        )


def _value_block(layer: FinalLayerTable, /) -> pl.DataFrame:
    return layer.values.select(pl.exclude(layer.var_key_columns))


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
    if numeric_type == "number" or not values.width:
        return values
    invalid = (
        pl.all().is_not_null()
        & ~pl.all().is_nan().fill_null(False)
        & (~pl.all().is_finite().fill_null(False) | (pl.all() != pl.all().floor()).fill_null(False))
    )
    examples = (
        values.select(
            pl.concat_list(pl.all().filter(invalid).head(_EXAMPLE_LIMIT).implode())
            .list.explode(empty_as_null=False)
            .head(_EXAMPLE_LIMIT)
        )
        .to_series()
        .to_list()
    )
    if examples:
        raise LayerValueError(
            f"integer layer {layer_name!r} contains fractional or infinite values; "
            f"examples={examples}"
        )
    return values.select(pl.all().fill_nan(None).cast(pl.Int64, strict=True))
