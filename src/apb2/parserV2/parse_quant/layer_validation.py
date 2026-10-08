"""Validate the complete canonical layer set produced by one parse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, JsonValue
from apb2.parserV2.parse_quant.errors import LayerContractError


@dataclass(frozen=True, slots=True)
class LayerContractValidator:
    """Validate required names and relative occupancy across canonical layers."""

    primary_layer_name: str
    required_names: tuple[str, ...]
    empty_ratio: float
    populated_ratio: float
    strict: bool

    def validate(self, layers: Mapping[str, FinalLayerTable]) -> dict[str, JsonValue] | None:
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
            candidates.update(layer.role.occupancy_candidates(name, layer.values))
        ratios = {name: _occupancy(values) for name, values in candidates.items()}
        populated = [name for name, ratio in ratios.items() if ratio >= self.populated_ratio]
        empty = [name for name, ratio in ratios.items() if ratio < self.empty_ratio]
        if not populated:
            return None
        diagnostics: dict[str, JsonValue] = {}
        reference = ", ".join(populated[:3])
        for name in empty:
            message = (
                f"layer {name!r} is effectively empty ({ratios[name]:.2%}) while {reference} is "
                "populated: its source column holds no readable values; unreadable tokens, if "
                "any, are reported separately"
            )
            if self.strict or name == self.primary_layer_name:
                raise LayerContractError(message)
            logger.warning(message)
            diagnostics[name] = {
                "occupancy": ratios[name],
                "empty_ratio": self.empty_ratio,
                "populated_ratio": self.populated_ratio,
                "reference_layers": list(populated),
            }
        return diagnostics


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
