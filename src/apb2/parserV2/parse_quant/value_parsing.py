"""Parse one aligned measurement layer into final canonical values."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    MeasurementLayerRole,
    NumericLayerType,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.errors import LayerValueError
from apb2.parserV2.parse_quant.numeric_text import absent, as_numbers, blank
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

UNKNOWN_CATEGORY_CODE = -1
_EXAMPLE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class PlainNumericLayerParser:
    """Parse directly readable numeric scalars and declared missing sentinels."""

    layer_name: str
    missing_values: tuple[float, ...]
    missing_at_or_below: float | None
    number_format: NumericTextFormat
    numeric_type: NumericLayerType

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Without sentinels blank text claims a cell; unreadable tokens always do."""
        if not self.missing_values and self.missing_at_or_below is None:
            return ~absent(values, dtype)
        numbers = as_numbers(values, dtype, self.number_format)
        sentinel = _declared_missing(numbers, self.missing_values, self.missing_at_or_below)
        return ~(blank(values, dtype) | sentinel.fill_null(False))

    def parse(self, layer: FinalLayerTable) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        values = layer.values
        evidence: dict[str, JsonValue] = {}
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
            evidence = NumericTokenDiagnostics(self.layer_name).collect(prepared)
            canonical = prepared.select(
                _masked(
                    pl.all().struct.field("number"), self.missing_values, self.missing_at_or_below
                ).name.keep()
            )
        return (
            _parsed_layer(
                layer,
                _validate_numeric_type(self.layer_name, canonical, self.numeric_type),
                semantics=QuantitativeLayerSemantics(logical_type=self.numeric_type),
                role=MeasurementLayerRole(),
            ),
            evidence,
        )


@dataclass(frozen=True, slots=True)
class RegexNumericLayerParser:
    """Extract one numeric capture from each structured token and parse it."""

    layer_name: str
    missing_values: tuple[float, ...]
    missing_at_or_below: float | None
    pattern: str
    number_format: NumericTextFormat
    numeric_type: NumericLayerType

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Inspect the numeric capture while retaining each original claiming token."""
        extracted = values.cast(pl.String, strict=False).str.extract(self.pattern, 1)
        numbers = as_numbers(extracted, pl.String(), self.number_format)
        sentinel = _declared_missing(numbers, self.missing_values, self.missing_at_or_below)
        return ~(blank(values, dtype) | sentinel.fill_null(False))

    def parse(self, layer: FinalLayerTable) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        values = layer.values
        prepared = values.select(
            pl.struct(
                pl.col(column).alias("raw"),
                as_numbers(
                    pl.col(column).cast(pl.String, strict=False).str.extract(self.pattern, 1),
                    pl.String(),
                    self.number_format,
                ).alias("number"),
                blank(pl.col(column), dtype).alias("blank"),
            ).alias(column)
            for column, dtype in values.schema.items()
        )
        evidence = NumericTokenDiagnostics(self.layer_name).collect(prepared)
        canonical = prepared.select(
            _masked(
                pl.all().struct.field("number"), self.missing_values, self.missing_at_or_below
            ).name.keep()
        )
        return (
            _parsed_layer(
                layer,
                _validate_numeric_type(self.layer_name, canonical, self.numeric_type),
                semantics=QuantitativeLayerSemantics(logical_type=self.numeric_type),
                role=MeasurementLayerRole(),
            ),
            evidence,
        )


@dataclass(frozen=True, slots=True)
class FactorLayerParser:
    """Replace declared category labels with their final integer codes."""

    categories: tuple[tuple[str, int], ...]

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Every non-missing label claims a cell, including blank or unknown labels."""
        return ~absent(values, dtype)

    def parse(self, layer: FinalLayerTable) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        semantics = CategoricalLayerSemantics(
            categories=self.categories,
            missing_code=UNKNOWN_CATEGORY_CODE,
        )
        mapping = dict(self.categories)
        values = layer.values
        canonical = values.select(
            pl.all()
            .cast(pl.String, strict=False)
            .replace_strict(
                mapping,
                default=UNKNOWN_CATEGORY_CODE,
                return_dtype=pl.Int64,
            )
        )
        return (
            _parsed_layer(layer, canonical, semantics=semantics, role=AuxiliaryLayerRole()),
            {},
        )


@dataclass(frozen=True, slots=True)
class NumericTokenDiagnostics:
    """Count failed numeric cells exactly and export a bounded token sample."""

    layer_name: str

    def collect(self, prepared: pl.DataFrame) -> dict[str, JsonValue]:
        if not prepared.width:
            return {}
        invalid = ~pl.all().struct.field("blank") & pl.all().struct.field("number").is_null()
        cell_count = prepared.select(pl.sum_horizontal(invalid.cast(pl.UInt64).sum())).item()
        if not cell_count:
            return {}
        tokens = prepared.select(
            pl.concat_list(
                pl.all().struct.field("raw").cast(pl.String).filter(invalid).unique().implode()
            )
            .list.explode(empty_as_null=False)
            .unique()
            .sort()
            .alias("token")
        )
        examples = tokens.head(_EXAMPLE_LIMIT).to_series().to_list()
        logger.warning(
            "layer {!r} declares numeric values; {} distinct unreadable token(s) "
            "in {} cell(s) became missing, examples={}",
            self.layer_name,
            tokens.height,
            cell_count,
            examples,
        )
        return {
            "cell_count": cell_count,
            "distinct_token_count": tokens.height,
            "examples": examples,
        }


def _parsed_layer(
    layer: FinalLayerTable,
    values: pl.DataFrame,
    /,
    *,
    semantics: QuantitativeLayerSemantics | CategoricalLayerSemantics,
    role: MeasurementLayerRole | AuxiliaryLayerRole,
) -> FinalLayerTable:
    return FinalLayerTable(
        layer_name=layer.layer_name,
        values=values,
        role=role,
        semantics=semantics,
    )


def _declared_missing(
    numbers: pl.Expr, missing_values: tuple[float, ...], missing_at_or_below: float | None
) -> pl.Expr:
    """Whether a number is a declared sentinel or at or below the declared threshold."""
    missing = numbers.is_in(list(missing_values))
    return missing if missing_at_or_below is None else missing | (numbers <= missing_at_or_below)


def _masked(
    numbers: pl.Expr, missing_values: tuple[float, ...], missing_at_or_below: float | None
) -> pl.Expr:
    if not missing_values and missing_at_or_below is None:
        return numbers
    missing = _declared_missing(numbers, missing_values, missing_at_or_below)
    return pl.when(missing).then(None).otherwise(numbers)


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
