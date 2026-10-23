"""Columns and layers that carry no information are not created; a finding records them.

A single-valued field holds one value in every filled cell, as the ion-mobility layer of runs
without ion mobility does; it is judged from two filled cells on, so a one-run result keeps
its obs columns. An empty column or layer has no filled cell at all, and an effectively empty
layer is the layer validator's verdict. Each is left out of the parsed level; its finding keeps
what it held: the one value, or the filled count and a summary of the values.

Kept whatever they hold: the axis keys, the columns a role names, the computed columns and the
inputs of the decoy and contaminant flags, which consumers read by name, and the primary and
sample layers, which define the level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import polars as pl

from apb2.parserV2.parse_quant.contracts import PrunedFields
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonScalar,
    JsonValue,
    ObsFinal,
    VarFinal,
)
from apb2.parserV2.parse_quant.numeric_text import blank

_SINGLE_FROM = 2
"""Filled cells from which one repeated value is a finding rather than a lone measurement."""


@dataclass(frozen=True, slots=True)
class _Values:
    """What one field's filled cells hold: how many, and their distinct extreme values."""

    filled: int
    extremes: frozenset[JsonScalar]

    @property
    def single(self) -> bool:
        return self.filled >= _SINGLE_FROM and len(self.extremes) == 1

    @property
    def value(self) -> JsonScalar:
        return next(iter(self.extremes))


@dataclass(frozen=True, slots=True)
class UninformativeFields:
    """Leave out a level's single-valued and empty fields, keeping the retained ones."""

    retained_obs: frozenset[str]
    retained_var: frozenset[str]
    retained_layers: frozenset[str]

    def prune(
        self,
        obs: ObsFinal,
        var: VarFinal,
        layers: Mapping[str, FinalLayerTable],
        effectively_empty: Mapping[str, JsonValue],
        /,
    ) -> PrunedFields:
        single: dict[str, dict[str, JsonValue]] = {"obs": {}, "var": {}, "layers": {}}
        empty: dict[str, dict[str, JsonValue]] = {"obs": {}, "var": {}, "layers": {}}
        for location, frame, retained in (
            ("obs", obs.frame, self.retained_obs),
            ("var", var.frame, self.retained_var),
        ):
            for name, values in _column_values(frame, retained).items():
                if values.filled == 0:
                    empty[location][name] = {"filled": 0}
                elif values.single:
                    single[location][name] = {"value": values.value}
        for name, layer in layers.items():
            if name in self.retained_layers:
                continue
            values = _layer_values(layer)
            if name in effectively_empty:
                verdict = effectively_empty[name]
                assert isinstance(verdict, dict)
                empty["layers"][name] = {**verdict, "filled": values.filled, **_summary(layer)}
            elif values.filled == 0:
                empty["layers"][name] = {"filled": 0}
            elif values.single:
                single["layers"][name] = {"value": values.value}
        left_out = {location: {*single[location], *empty[location]} for location in single}
        return PrunedFields(
            obs=replace(obs, frame=obs.frame.drop(sorted(left_out["obs"]))),
            var=replace(var, frame=var.frame.drop(sorted(left_out["var"]))),
            layers={
                name: layer for name, layer in layers.items() if name not in left_out["layers"]
            },
            single_valued=single,
            effectively_empty=empty,
        )


def _comparable(dtype: pl.DataType) -> bool:
    """Whether one smallest and one largest value say if a column holds a single value.

    A column of nulls alone has the dtype ``Null``, as a Parquet column without a value does.
    """
    return dtype.is_numeric() or dtype in (pl.String(), pl.Boolean(), pl.Null())


def _filled(name: str, dtype: pl.DataType) -> pl.Expr:
    return ~blank(pl.col(name), dtype)


def _column_values(frame: pl.DataFrame, retained: frozenset[str]) -> dict[str, _Values]:
    """The filled count and value range of each judged column of one axis frame."""
    judged = [
        name for name, dtype in frame.schema.items() if name not in retained and _comparable(dtype)
    ]
    if not judged:
        return {}
    schema = frame.schema
    row = frame.select(
        *(
            _filled(name, schema[name]).sum().alias(f"n{index}")
            for index, name in enumerate(judged)
        ),
        *(
            pl.col(name).filter(_filled(name, schema[name])).min().alias(f"lo{index}")
            for index, name in enumerate(judged)
        ),
        *(
            pl.col(name).filter(_filled(name, schema[name])).max().alias(f"hi{index}")
            for index, name in enumerate(judged)
        ),
    ).row(0, named=True)
    return {
        name: _Values(int(row[f"n{index}"]), frozenset({row[f"lo{index}"], row[f"hi{index}"]}))
        for index, name in enumerate(judged)
    }


def _layer_values(layer: FinalLayerTable) -> _Values:
    """The filled count and extreme values of one layer across all its runs, as labels."""
    columns = _column_values(layer.decoded_values(), frozenset()).values()
    return _Values(
        sum(values.filled for values in columns),
        frozenset(extreme for values in columns if values.filled for extreme in values.extremes),
    )


def _summary(layer: FinalLayerTable) -> dict[str, JsonValue]:
    """The six-number summary of an effectively empty layer's finite values."""
    values = layer.quantitative_values()
    if not values.width:
        return {}
    numbers = pl.concat([values.get_column(name).cast(pl.Float64) for name in values.columns])
    finite = pl.DataFrame({"value": numbers.filter(numbers.is_finite())})
    if finite.is_empty():
        return {}
    value = pl.col("value")
    row = finite.select(
        min=value.min(),
        first_quartile=value.quantile(0.25, "linear"),
        median=value.median(),
        mean=value.mean(),
        third_quartile=value.quantile(0.75, "linear"),
        max=value.max(),
    ).row(0, named=True)
    return {"summary": dict(row)}
