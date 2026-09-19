"""Bounded summaries of already encoded APB2 matrix layers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import singledispatch
from typing import Literal, cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    CategoricalLayerSemantics,
    FinalLayerSemantics,
    JsonValue,
    QuantitativeLayerSemantics,
)

OBSERVATION_SUMMARY_LIMIT = 100
QUANTILE_SAMPLE_LIMIT = 100_000
QUARTILE_INTERPOLATION = "linear"


@singledispatch
def represent_semantics(
    semantics: FinalLayerSemantics,
    values: pl.DataFrame,
    /,
    *,
    observation_limit: int,
) -> dict[str, JsonValue]:
    """Describe canonical values according to their attached semantics."""
    raise TypeError(f"unsupported layer semantics {type(semantics).__name__}")


@represent_semantics.register
def represent_quantitative_semantics(
    semantics: QuantitativeLayerSemantics,
    values: pl.DataFrame,
    /,
    *,
    observation_limit: int,
) -> dict[str, JsonValue]:
    return quantitative_representation(
        values,
        logical_type=semantics.logical_type,
        observation_limit=observation_limit,
    )


@represent_semantics.register
def represent_categorical_semantics(
    semantics: CategoricalLayerSemantics,
    values: pl.DataFrame,
    /,
    *,
    observation_limit: int,
) -> dict[str, JsonValue]:
    del observation_limit
    return categorical_representation(
        values,
        category_count=len(semantics.categories),
        valid_codes=tuple(code for _label, code in semantics.categories),
    )


def quantitative_representation(
    values: pl.DataFrame,
    /,
    *,
    logical_type: Literal["number", "integer"] = "number",
    observation_limit: int = OBSERVATION_SUMMARY_LIMIT,
    quantile_sample_limit: int = QUANTILE_SAMPLE_LIMIT,
) -> dict[str, JsonValue]:
    """Describe a quantitative value block without flattening its complete matrix."""
    if observation_limit < 0:
        raise ValueError("observation_limit must not be negative")
    if quantile_sample_limit <= 0:
        raise ValueError("quantile_sample_limit must be positive")

    columns = values.get_columns()
    accumulator = _NumericAccumulator()
    observation_items: list[JsonValue] = []

    for index, column in enumerate(columns):
        accumulator.add(column)
        if index < observation_limit:
            observation_items.append(
                {
                    "observation_index": index,
                    "statistics": _series_statistics(column),
                }
            )

    sample = _bounded_finite_sample(
        columns,
        finite_count=accumulator.finite_count,
        limit=quantile_sample_limit,
    )
    sample_exact = accumulator.finite_count <= quantile_sample_limit
    quartile_method = "linear_exact" if sample_exact else "linear_deterministic_grid_sample"
    return {
        "value_kind": "quantitative",
        "type": logical_type,
        "dtype": _numeric_dtype(values),
        "statistics": accumulator.statistics(
            sample,
            quartile_method=quartile_method,
            quartile_sample_limit=quantile_sample_limit,
        ),
        "observation_summaries": {
            "total_count": values.width,
            "emitted_count": len(observation_items),
            "truncated": len(observation_items) < values.width,
            "items": observation_items,
        },
    }


def categorical_representation(
    values: pl.DataFrame,
    /,
    *,
    category_count: int,
    valid_codes: tuple[int, ...],
) -> dict[str, JsonValue]:
    """Describe a factor block through fixed-size counts, never numeric moments."""
    total_count = values.height * values.width
    known_count = sum(
        int(column.is_in(valid_codes).fill_null(False).sum()) for column in values.get_columns()
    )
    return {
        "value_kind": "categorical",
        "dtype": "categorical",
        "category_count": category_count,
        "counts": {
            "total_count": total_count,
            "known_count": known_count,
            "missing_or_unknown_count": total_count - known_count,
        },
    }


@dataclass(slots=True)
class _NumericAccumulator:
    """Columnwise exact counts and moments for one complete matrix."""

    total_count: int = 0
    finite_count: int = 0
    null_count: int = 0
    nan_count: int = 0
    positive_infinity_count: int = 0
    negative_infinity_count: int = 0
    zero_count: int = 0
    mean: float = 0.0
    second_moment: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, values: pl.Series, /) -> pl.Series:
        """Accumulate one observation column and return its finite values."""
        floats = values.cast(pl.Float64, strict=True)
        finite = floats.filter(floats.is_finite().fill_null(False))
        count = len(finite)

        self.total_count += len(floats)
        self.null_count += int(floats.is_null().sum())
        self.nan_count += int(floats.is_nan().fill_null(False).sum())
        self.positive_infinity_count += int((floats == math.inf).fill_null(False).sum())
        self.negative_infinity_count += int((floats == -math.inf).fill_null(False).sum())
        self.zero_count += int((finite == 0).sum())
        if not count:
            return finite

        column_mean = float(cast(float | int, finite.mean()))
        column_variance = float(cast(float | int, finite.var(ddof=0))) if count > 1 else 0.0
        column_minimum = float(cast(float | int, finite.min()))
        column_maximum = float(cast(float | int, finite.max()))
        combined_count = self.finite_count + count
        delta = column_mean - self.mean
        self.second_moment += (
            column_variance * count + delta * delta * self.finite_count * count / combined_count
        )
        self.mean += delta * count / combined_count
        self.finite_count = combined_count
        self.minimum = column_minimum if self.minimum is None else min(self.minimum, column_minimum)
        self.maximum = column_maximum if self.maximum is None else max(self.maximum, column_maximum)
        return finite

    def statistics(
        self,
        quartile_sample: pl.Series,
        /,
        *,
        quartile_method: str,
        quartile_sample_limit: int,
    ) -> dict[str, JsonValue]:
        """Return exact moments plus explicitly qualified sampled quartiles."""
        raw_standard_deviation = (
            math.sqrt(max(0.0, self.second_moment) / (self.finite_count - 1))
            if self.finite_count > 1
            else None
        )
        return {
            "total_count": self.total_count,
            "finite_count": self.finite_count,
            "null_count": self.null_count,
            "nan_count": self.nan_count,
            "positive_infinity_count": self.positive_infinity_count,
            "negative_infinity_count": self.negative_infinity_count,
            "zero_count": self.zero_count,
            "mean": _finite_number(self.mean) if self.finite_count else None,
            "standard_deviation": _finite_number(raw_standard_deviation),
            "minimum": _finite_number(self.minimum),
            "first_quartile": _quantile(quartile_sample, 0.25),
            "median": _quantile(quartile_sample, 0.5),
            "third_quartile": _quantile(quartile_sample, 0.75),
            "maximum": _finite_number(self.maximum),
            "quartile_interpolation": QUARTILE_INTERPOLATION,
            "quartile_method": quartile_method,
            "quartile_sample_count": len(quartile_sample),
            "quartile_sample_limit": quartile_sample_limit,
        }


def _series_statistics(values: pl.Series) -> dict[str, JsonValue]:
    accumulator = _NumericAccumulator()
    finite = accumulator.add(values)
    return accumulator.statistics(
        finite,
        quartile_method="linear_exact",
        quartile_sample_limit=len(values),
    )


def _numeric_dtype(values: pl.DataFrame) -> str:
    names = {str(dtype) for dtype in values.dtypes}
    return next(iter(names)) if len(names) == 1 else "mixed"


def _quantile(values: pl.Series, probability: float) -> float | None:
    if values.is_empty():
        return None
    result = cast(
        float | int | None,
        values.quantile(probability, interpolation=QUARTILE_INTERPOLATION),
    )
    return _finite_number(result)


def _finite_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bounded_finite_sample(
    columns: list[pl.Series],
    /,
    *,
    finite_count: int,
    limit: int,
) -> pl.Series:
    """Sample deterministic positions from the finite-cell stream with bounded memory."""
    targets = _even_indices(finite_count, min(finite_count, limit))
    sampled: list[float] = []
    target_position = 0
    stream_offset = 0
    for column in columns:
        floats = column.cast(pl.Float64, strict=True)
        finite = floats.filter(floats.is_finite().fill_null(False))
        next_offset = stream_offset + len(finite)
        local_indices: list[int] = []
        while target_position < len(targets) and targets[target_position] < next_offset:
            local_indices.append(targets[target_position] - stream_offset)
            target_position += 1
        if local_indices:
            sampled.extend(cast(list[float], finite.gather(local_indices).to_list()))
        stream_offset = next_offset
    return pl.Series("", sampled, dtype=pl.Float64)


def _even_indices(length: int, count: int) -> tuple[int, ...]:
    if not count:
        return ()
    if count == 1:
        return (0,)
    return tuple(index * (length - 1) // (count - 1) for index in range(count))
