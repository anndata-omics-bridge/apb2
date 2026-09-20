"""Bounded summaries of already encoded APB2 matrix layers."""

from __future__ import annotations

import math
from functools import singledispatch
from typing import Literal, cast

import polars as pl
import polars.selectors as cs

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

    columns = _column_statistics(values)
    count = pl.col("finite_count")
    finite_count = int(columns.select(count.sum()).item())
    sample = _bounded_finite_sample(values, columns, limit=quantile_sample_limit)
    sample_exact = finite_count <= quantile_sample_limit
    quartile_method = "linear_exact" if sample_exact else "linear_deterministic_grid_sample"
    mean = (pl.col("mean") * (count / finite_count)).sum()
    variance = ((pl.col("variance") + (pl.col("mean") - mean).pow(2)) * count).sum() / (
        finite_count - 1
    )
    statistics = columns.select(
        cs.ends_with("_count").sum(),
        pl.when(finite_count > 0).then(mean).alias("mean"),
        pl.when(finite_count > 1)
        .then(variance.clip(lower_bound=0).sqrt())
        .alias("standard_deviation"),
        pl.col("minimum").min(),
        pl.col("maximum").max(),
    ).hstack(sample.to_frame().select(**_quartiles(pl.all())))
    observations = (
        _column_statistics(
            values.select(cs.by_index(range(min(observation_limit, values.width)))), quartiles=True
        )
        .drop("variance")
        .with_columns(
            quartile_interpolation=pl.lit(QUARTILE_INTERPOLATION),
            quartile_method=pl.lit("linear_exact"),
            quartile_sample_count=count,
            quartile_sample_limit=pl.col("total_count"),
        )
    )
    observation_items = (
        _finite_moments(observations)
        .with_row_index("observation_index")
        .select("observation_index", pl.struct(pl.exclude("observation_index")).alias("statistics"))
        .to_dicts()
    )
    return {
        "value_kind": "quantitative",
        "type": logical_type,
        "dtype": _numeric_dtype(values),
        "statistics": {
            **_finite_moments(statistics).row(0, named=True),
            "quartile_interpolation": QUARTILE_INTERPOLATION,
            "quartile_method": quartile_method,
            "quartile_sample_count": len(sample),
            "quartile_sample_limit": quantile_sample_limit,
        },
        "observation_summaries": {
            "total_count": values.width,
            "emitted_count": len(observation_items),
            "truncated": len(observation_items) < values.width,
            "items": cast(list[JsonValue], observation_items),
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
    known_count = values.select(
        pl.sum_horizontal(
            pl.lit(0, dtype=pl.UInt64), pl.all().is_in(valid_codes).cast(pl.UInt64).sum()
        )
    ).item()
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


def _column_statistics(values: pl.DataFrame, *, quartiles: bool = False) -> pl.DataFrame:
    """Reduce all columns together, then reshape only the small summary table."""
    floats = pl.all().cast(pl.Float64)
    finite = floats.filter(floats.is_finite())
    reductions = {
        "total_count": floats.len().cast(pl.UInt64),
        "finite_count": finite.len().cast(pl.UInt64),
        "null_count": floats.null_count().cast(pl.UInt64),
        "nan_count": floats.is_nan().cast(pl.UInt64).sum(),
        "positive_infinity_count": (floats == math.inf).cast(pl.UInt64).sum(),
        "negative_infinity_count": (floats == -math.inf).cast(pl.UInt64).sum(),
        "zero_count": (finite == 0).cast(pl.UInt64).sum(),
        "mean": finite.mean(),
        "variance": finite.var(ddof=0),
        "standard_deviation": finite.std(ddof=1),
        "minimum": finite.min(),
        "maximum": finite.max(),
        **(_quartiles(finite) if quartiles else {}),
    }
    # An empty typed column lets expression expansion retain the summary's schema.
    source = values if values.width else pl.DataFrame(schema={"empty": pl.Float64})
    return (
        source.select(
            pl.concat_list(expression).alias(name) for name, expression in reductions.items()
        )
        .explode(cs.all(), empty_as_null=False)
        .head(values.width)
    )


def _numeric_dtype(values: pl.DataFrame) -> str:
    names = {str(dtype) for dtype in values.dtypes}
    return next(iter(names)) if len(names) == 1 else "mixed"


def _quartiles(values: pl.Expr) -> dict[str, pl.Expr]:
    return {
        name: values.quantile(probability, interpolation=QUARTILE_INTERPOLATION)
        for name, probability in (
            ("first_quartile", 0.25),
            ("median", 0.5),
            ("third_quartile", 0.75),
        )
    }


def _finite_moments(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.with_columns(
        pl.when(cs.float().is_finite()).then(cs.float()).otherwise(None).name.keep()
    )


def _bounded_finite_sample(
    values: pl.DataFrame,
    statistics: pl.DataFrame,
    /,
    *,
    limit: int,
) -> pl.Series:
    """Sample deterministic positions from the finite-cell stream with bounded memory."""
    total = int(statistics.select(pl.col("finite_count").sum()).item())
    count = min(total, limit)
    if not count:
        return pl.Series("sample", [], dtype=pl.Float64)
    offsets = (
        statistics.select(pl.col("finite_count").cum_sum().shift(fill_value=0))
        .to_series()
        .implode()
    )
    # Expand one scalar offset per column; gather at most `limit` values in total.
    start = pl.lit(offsets).list.to_struct(fields=values.columns).struct.field("*")
    targets = pl.int_range(0, count, dtype=pl.UInt64) * (total - 1) // max(count - 1, 1)
    floats = pl.all().cast(pl.Float64)
    finite = floats.filter(floats.is_finite())
    local = targets.filter((targets >= start) & (targets < start + finite.len())) - start
    return (
        values.select(pl.concat_list(finite.gather(local).implode()).alias("sample"))
        .explode("sample", empty_as_null=False)
        .to_series()
    )
