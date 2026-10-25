"""Parse one layer's resolved cells into final canonical values, still as long rows.

The cells arrive as long rows at final positions, one per cell a layer measured; a cell no row
names is absent, exactly as an absent cell of the wide layer would be. The parsed layer keeps
those positions, so nothing here widens a layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS, ObservationGrid
from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    MeasurementLayerRole,
    NumericLayerType,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.data.raw import LongLayer
from apb2.parserV2.parse_quant.errors import LayerValueError
from apb2.parserV2.parse_quant.numeric_text import absent, as_numbers
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

UNKNOWN_CATEGORY_CODE = -1
_EXAMPLE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class PlainNumericLayerParser:
    """Parse directly readable numeric scalars and declared missing sentinels."""

    layer_name: str
    missing_values: tuple[float, ...]
    missing_at_or_below: float | None
    missing_tokens: tuple[str, ...]
    number_format: NumericTextFormat
    numeric_type: NumericLayerType

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Without sentinels every present value claims a cell; unreadable tokens always do."""
        if not self.missing_values and self.missing_at_or_below is None and not self.missing_tokens:
            return ~absent(values, dtype)
        numbers = as_numbers(values, dtype, self.number_format)
        sentinel = _declared_missing(numbers, self.missing_values, self.missing_at_or_below)
        return ~(_missing_text(values, dtype, self.missing_tokens) | sentinel.fill_null(False))

    def parse(
        self, layer: LongLayer, grid: ObservationGrid, /
    ) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        dtype = layer.rows.schema[VALUE]
        return _numeric_layer(
            self.layer_name,
            layer.rows,
            as_numbers(pl.col(VALUE), dtype, self.number_format),
            _missing_text(pl.col(VALUE), dtype, self.missing_tokens),
            self.missing_values,
            self.missing_at_or_below,
            self.numeric_type,
            grid,
        )


@dataclass(frozen=True, slots=True)
class RegexNumericLayerParser:
    """Extract one numeric capture from each structured token and parse it."""

    layer_name: str
    missing_values: tuple[float, ...]
    missing_at_or_below: float | None
    missing_tokens: tuple[str, ...]
    pattern: str
    number_format: NumericTextFormat
    numeric_type: NumericLayerType

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Inspect the numeric capture while retaining each original claiming token."""
        extracted = values.cast(pl.String, strict=False).str.extract(self.pattern, 1)
        numbers = as_numbers(extracted, pl.String(), self.number_format)
        sentinel = _declared_missing(numbers, self.missing_values, self.missing_at_or_below)
        return ~(_missing_text(values, dtype, self.missing_tokens) | sentinel.fill_null(False))

    def parse(
        self, layer: LongLayer, grid: ObservationGrid, /
    ) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        dtype = layer.rows.schema[VALUE]
        return _numeric_layer(
            self.layer_name,
            layer.rows,
            as_numbers(
                pl.col(VALUE).cast(pl.String, strict=False).str.extract(self.pattern, 1),
                pl.String(),
                self.number_format,
            ),
            _missing_text(pl.col(VALUE), dtype, self.missing_tokens),
            self.missing_values,
            self.missing_at_or_below,
            self.numeric_type,
            grid,
        )


@dataclass(frozen=True, slots=True)
class FactorLayerParser:
    """Replace declared category labels with their final integer codes."""

    categories: tuple[tuple[str, int], ...]

    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        """Every non-missing label claims a cell, including unknown labels."""
        return ~absent(values, dtype)

    def parse(
        self, layer: LongLayer, grid: ObservationGrid, /
    ) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
        semantics = CategoricalLayerSemantics(
            categories=self.categories,
            missing_code=UNKNOWN_CATEGORY_CODE,
        )
        codes = (
            layer.rows.get_column(VALUE)
            .cast(pl.String, strict=False)
            .replace_strict(
                dict(self.categories), default=UNKNOWN_CATEGORY_CODE, return_dtype=pl.Int64
            )
        )
        return (
            FinalLayerTable(
                layer.layer_name,
                layer.rows.select(VAR_POS, OBS_POS).with_columns(codes.alias(VALUE)),
                grid.height,
                grid.labels,
                role=AuxiliaryLayerRole(),
                semantics=semantics,
            ),
            {},
        )


@dataclass(frozen=True, slots=True)
class NumericTokenDiagnostics:
    """Count failed numeric cells exactly and export a bounded token sample."""

    layer_name: str

    def collect(self, tokens: pl.Series) -> dict[str, JsonValue]:
        """``tokens`` holds the value of every resolved cell with text but no number."""
        if not tokens.len():
            return {}
        distinct = tokens.cast(pl.String).unique().sort()
        examples = distinct.head(_EXAMPLE_LIMIT).to_list()
        logger.warning(
            "layer {!r} declares numeric values; {} distinct unreadable token(s) "
            "in {} cell(s) became missing, examples={}",
            self.layer_name,
            distinct.len(),
            tokens.len(),
            examples,
        )
        return {
            "cell_count": tokens.len(),
            "distinct_token_count": distinct.len(),
            "examples": examples,
        }


def _numeric_layer(
    layer_name: str,
    rows: pl.DataFrame,
    number: pl.Expr,
    blank: pl.Expr,
    missing_values: tuple[float, ...],
    missing_at_or_below: float | None,
    numeric_type: NumericLayerType,
    grid: ObservationGrid,
) -> tuple[FinalLayerTable, dict[str, JsonValue]]:
    """Read each resolved cell's number and report the unreadable ones."""
    read = rows.select(pl.col(VALUE), number.alias("number"), blank.alias("blank"))
    unreadable = read.select(~pl.col("blank") & pl.col("number").is_null()).to_series()
    evidence = NumericTokenDiagnostics(layer_name).collect(
        read.get_column(VALUE).filter(unreadable)
    )
    numbers = read.select(
        _masked(pl.col("number"), missing_values, missing_at_or_below)
    ).to_series()
    values = _validate_numeric_type(layer_name, rows, numbers, numeric_type)
    return (
        FinalLayerTable(
            layer_name,
            rows.select(VAR_POS, OBS_POS).with_columns(values.alias(VALUE)),
            grid.height,
            grid.labels,
            role=MeasurementLayerRole(),
            semantics=QuantitativeLayerSemantics(logical_type=numeric_type),
        ),
        evidence,
    )


def _missing_text(values: pl.Expr, dtype: pl.DataType, tokens: tuple[str, ...]) -> pl.Expr:
    """Absent cells, and cells holding a declared missing-value token such as ``-``."""
    if not tokens or dtype.is_numeric():
        return absent(values, dtype)
    return absent(values, dtype) | values.cast(pl.String, strict=False).str.strip_chars().is_in(
        tokens
    ).fill_null(False)


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
    rows: pl.DataFrame,
    numbers: pl.Series,
    numeric_type: NumericLayerType,
    /,
) -> pl.Series:
    """Integer layers hold whole finite numbers; examples follow observation, then var order."""
    if numeric_type == "number":
        return numbers
    invalid = (
        numbers.is_not_null()
        & ~numbers.is_nan().fill_null(False)
        & (~numbers.is_finite().fill_null(False) | (numbers != numbers.floor()).fill_null(False))
    )
    if invalid.any():
        examples = (
            rows.select(OBS_POS, VAR_POS)
            .with_columns(numbers)
            .filter(invalid)
            .sort(OBS_POS, VAR_POS)
            .get_column(numbers.name)
            .head(_EXAMPLE_LIMIT)
            .to_list()
        )
        raise LayerValueError(
            f"integer layer {layer_name!r} contains fractional or infinite values; "
            f"examples={examples}"
        )
    return numbers.fill_nan(None).cast(pl.Int64, strict=True)
