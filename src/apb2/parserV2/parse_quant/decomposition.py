"""Physical shape becomes the common raw contract: two small axes and wide raw layers.

A decomposer converts layout and nothing else. It receives source-resolved column names and
configured key sets — never a rule, a vendor, a level, an unresolved regex, a duplicate mode,
an encoder, or a writer — and every shape reduces to the same three values, so everything
downstream is written once.

Two invariants are easy to lose and are the reason this module exists at all. A raw axis holds
exactly one stable-first row per complete raw-key tuple, keeping the first payload metadata it
saw. And a raw layer *keeps* repeated cells: a repeated ``(var key, obs key)`` pair becomes a
repeated var row, because deciding what several values for one cell mean is the duplicate
policy's question, and collapsing them here would destroy the evidence it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.contracts import FragmentTableSeparator, SourceDecomposer
from apb2.parserV2.parse_quant.data.raw import (
    DecomposedDataRaw,
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.layer_labels import observation_labels
from apb2.parserV2.parse_quant.parameters.axis import AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.source import LongRawLayerSource, WideRawLayerPlan

_OCCURRENCE = "_occurrence"
_VAR_SLOT = "_var_slot"
_OBS_SLOT = "_obs_slot"


def _value_column(label: str, dtype: pl.DataType, *, observed: bool) -> pl.Expr:
    """One layer value column: the pivoted values, or nulls for an unobserved observation."""
    return pl.col(label) if observed else pl.lit(None, dtype=dtype).alias(label)


def _axis_frame(frame: pl.DataFrame, plan: AxisSourcePlan) -> pl.DataFrame:
    """One stable-first row per complete raw-key tuple, with its payload columns.

    "Complete" describes the tuple, not its values: distinctness is judged on the whole raw
    key. A missing component stays, because a declared operation such as ``coalesce`` may
    still turn it into a valid final key — and if it does not, axis preparation removes the
    row and the layer cells that pointed at it.

    Conflicting payload metadata for one raw key keeps the first physical value. The
    implementation may report that; it must never answer with a second identity row.
    """
    keys = plan.keys.raw_key_columns
    columns = [*keys, *(name for name in plan.payload_sources if name not in set(keys))]
    return frame.select(columns).unique(subset=list(keys), keep="first", maintain_order=True)


def _ordered_by_axis(
    values: pl.DataFrame, axis: pl.DataFrame, keys: tuple[str, ...]
) -> pl.DataFrame:
    """Sort layer rows into axis order, keeping repeated cells adjacent and in file order."""
    slots = axis.select(list(keys)).with_row_index(_VAR_SLOT)
    return (
        values.join(slots, on=list(keys), how="left", nulls_equal=True, maintain_order="left")
        .sort([_VAR_SLOT, _OCCURRENCE])
        .drop(_VAR_SLOT, _OCCURRENCE)
    )


@dataclass(frozen=True, slots=True)
class LongSourceDecomposer:
    """One physical row per (observation, feature); every layer names an exact column."""

    primary_layer_name: str
    layer_sources: tuple[LongRawLayerSource, ...]
    obs: AxisSourcePlan
    var: AxisSourcePlan

    def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
        frame = table.frame
        obs_frame = _axis_frame(frame, self.obs)
        var_frame = _axis_frame(frame, self.var)
        labels = observation_labels(obs_frame.height, reserved=var_frame.columns)
        return DecomposedDataRaw(
            obs=ObsRaw(frame=obs_frame, raw_key_columns=self.obs.keys.raw_key_columns),
            var=VarRaw(frame=var_frame, raw_key_columns=self.var.keys.raw_key_columns),
            layers=LayersRaw(
                primary_layer_name=self.primary_layer_name,
                values=tuple(
                    self._layer(frame, obs_frame, var_frame, labels, source)
                    for source in self.layer_sources
                ),
            ),
        )

    def _layer(
        self,
        frame: pl.DataFrame,
        obs_frame: pl.DataFrame,
        var_frame: pl.DataFrame,
        labels: tuple[str, ...],
        source: LongRawLayerSource,
    ) -> RawLayerTable:
        """Turn one long value column into a wide layer, repeated cells intact."""
        var_keys = self.var.keys.raw_key_columns
        obs_keys = self.obs.keys.raw_key_columns
        slots = obs_frame.select(list(obs_keys)).with_columns(
            pl.Series(_OBS_SLOT, list(labels), dtype=pl.String)
        )
        placed = frame.join(
            slots, on=list(obs_keys), how="inner", nulls_equal=True, maintain_order="left"
        )
        # The occurrence counter exists only so the pivot cannot collapse a repeated cell.
        # It is dropped again below and never reaches a caller.
        counted = placed.with_columns(
            (pl.cum_count(_OBS_SLOT).over([*var_keys, _OBS_SLOT]) - 1).alias(_OCCURRENCE)
        )
        pivoted = counted.pivot(
            on=_OBS_SLOT,
            index=[*var_keys, _OCCURRENCE],
            values=source.source_column,
        )
        dtype = frame.schema[source.source_column]
        present = set(pivoted.columns)
        values = _ordered_by_axis(pivoted, var_frame, var_keys).select(
            [
                *var_keys,
                *(_value_column(label, dtype, observed=label in present) for label in labels),
            ]
        )
        return RawLayerTable(layer_name=source.name, raw_var_key_columns=var_keys, values=values)


@dataclass(frozen=True, slots=True)
class WideSourceDecomposer:
    """One physical row per feature; the observations came from resolved header captures."""

    primary_layer_name: str
    layer_plans: tuple[WideRawLayerPlan, ...]
    obs: AxisSourcePlan
    var: AxisSourcePlan

    def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
        frame = table.frame
        samples = self._samples()
        obs_frame = pl.DataFrame({name: list(samples) for name in self.obs.keys.raw_key_columns})
        var_frame = _axis_frame(frame, self.var)
        labels = observation_labels(len(samples), reserved=var_frame.columns)
        return DecomposedDataRaw(
            obs=ObsRaw(frame=obs_frame, raw_key_columns=self.obs.keys.raw_key_columns),
            var=VarRaw(frame=var_frame, raw_key_columns=self.var.keys.raw_key_columns),
            layers=LayersRaw(
                primary_layer_name=self.primary_layer_name,
                values=tuple(
                    self._layer(frame, var_frame, samples, labels, plan)
                    for plan in self.layer_plans
                ),
            ),
        )

    def _samples(self) -> tuple[str, ...]:
        """The primary layer defines the observation axis, in stable header order."""
        primary = next(plan for plan in self.layer_plans if plan.name == self.primary_layer_name)
        return tuple(dict.fromkeys(source.sample for source in primary.sources))

    def _layer(
        self,
        frame: pl.DataFrame,
        var_frame: pl.DataFrame,
        samples: tuple[str, ...],
        labels: tuple[str, ...],
        plan: WideRawLayerPlan,
    ) -> RawLayerTable:
        """Align one layer's resolved columns to the primary samples, in that order.

        Several physical columns claiming one sample become repeated rows, so long and wide
        inputs reach the duplicate policy in the same shape.
        """
        var_keys = self.var.keys.raw_key_columns
        by_sample: dict[str, list[str]] = {}
        for source in plan.sources:
            by_sample.setdefault(source.sample, []).append(source.source_column)
        depth = max((len(columns) for columns in by_sample.values()), default=1)
        blocks = [
            frame.select(
                [
                    *var_keys,
                    *(
                        self._value(by_sample.get(sample, []), occurrence, label)
                        for sample, label in zip(samples, labels, strict=True)
                    ),
                ]
            ).with_columns(pl.lit(occurrence, dtype=pl.UInt32).alias(_OCCURRENCE))
            for occurrence in range(depth)
        ]
        stacked = pl.concat(blocks) if blocks else frame.select(list(var_keys))
        return RawLayerTable(
            layer_name=plan.name,
            raw_var_key_columns=var_keys,
            values=_ordered_by_axis(stacked, var_frame, var_keys),
        )

    @staticmethod
    def _value(columns: list[str], occurrence: int, label: str) -> pl.Expr:
        """One sample's value at one occurrence, or a null of no particular type.

        A required layer whose pattern matched only non-primary samples has no resolved
        column at all; ``Null`` says exactly that, and both writers accept it.
        """
        if occurrence < len(columns):
            return pl.col(columns[occurrence]).alias(label)
        return _value_column(label, pl.Null(), observed=False)


@dataclass(frozen=True, slots=True)
class DelimitedFragmentSourceDecomposer:
    """Separate the packed fragments, then decompose the scalar rows as ordinary long input.

    Composition, not a third algorithm: the long decomposer injected here is the same
    implementation direct long input uses, so a fragment level cannot drift from it.
    """

    separator: FragmentTableSeparator
    long_decomposer: SourceDecomposer

    def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
        return self.long_decomposer.decompose(self.separator.separate(table))
