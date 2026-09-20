"""Validate the complete canonical layer set produced by one parse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable
from apb2.parserV2.parse_quant.errors import LayerContractError


@dataclass(frozen=True, slots=True)
class LayerContractValidator:
    """Validate required names and relative occupancy across canonical layers."""

    primary_layer_name: str
    required_names: tuple[str, ...]
    empty_ratio: float
    populated_ratio: float
    strict: bool

    def validate(self, layers: Mapping[str, FinalLayerTable], /) -> None:
        missing = [
            name for name in (self.primary_layer_name, *self.required_names) if name not in layers
        ]
        if missing:
            raise LayerContractError(
                f"the canonical layers are missing required name(s) {missing}; present: "
                f"{sorted(layers)}"
            )

        candidates: dict[str, pl.DataFrame] = {}
        for name, layer in layers.items():
            candidates.update(layer.role.occupancy_candidates(name, _value_block(layer)))
        ratios = {name: _occupancy(values) for name, values in candidates.items()}
        populated = [name for name, ratio in ratios.items() if ratio >= self.populated_ratio]
        empty = [name for name, ratio in ratios.items() if ratio < self.empty_ratio]
        if not populated:
            return
        reference = ", ".join(populated[:3])
        for name in empty:
            message = (
                f"layer {name!r} is effectively empty ({ratios[name]:.2%}) while {reference} is "
                "populated — the source column was read but its values did not parse; check the "
                "vendor number format and the missing-value sentinels"
            )
            if self.strict or name == self.primary_layer_name:
                raise LayerContractError(message)
            logger.warning(message)


def _value_block(layer: FinalLayerTable, /) -> pl.DataFrame:
    return layer.values.select(pl.exclude(layer.var_key_columns))


def _occupancy(values: pl.DataFrame, /) -> float:
    cells = values.height * values.width
    if not cells:
        return 0.0
    usable = (
        values.fill_nan(None)
        .select(pl.sum_horizontal(pl.all().is_not_null().cast(pl.UInt64).sum()))
        .item()
    )
    return usable / cells
