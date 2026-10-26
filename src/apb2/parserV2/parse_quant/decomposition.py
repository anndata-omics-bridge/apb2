"""Physical shape becomes the common raw contract: two small axes and long layers.

A decomposer converts layout and nothing else. It receives source-resolved column names and
configured key sets — never a rule, a vendor, a level, an unresolved regex, a duplicate mode,
an encoder, or a writer — and every shape reduces to the same three values, so everything
downstream is written once.

Two invariants are easy to lose and are the reason this module exists at all. A raw axis holds
exactly one row per complete raw-key tuple: the obs axis keeps the first payload metadata it
saw, and each var payload column applies the value policy the resolver configured for it,
reporting features whose rows disagree. And a layer *keeps* repeated cells: each physical value
becomes one long row naming its axis rows, so two values for one cell are two rows, because
deciding what they mean is the duplicate policy's question and collapsing them here would
destroy the evidence it needs. Layer rows carry no key text, only the two row numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import polars as pl

from apb2.parserV2.parse_quant.contracts import (
    FeatureValuePolicy,
    FragmentTableSeparator,
)
from apb2.parserV2.parse_quant.data.layer_columns import VALUE
from apb2.parserV2.parse_quant.data.raw import (
    DISAGREEING_VALUES,
    OBS_ROW,
    VAR_ROW,
    LongLayer,
    ObsRaw,
    RawLevel,
    VarRaw,
)
from apb2.parserV2.parse_quant.numeric_text import absent
from apb2.parserV2.parse_quant.parameters.axis import AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.source import (
    LevelSourceTable,
    LongRawLayerSource,
    WideRawLayerPlan,
)

_VALUE_LIMIT = 5
_DISAGREES = "_disagrees_"
_ROW_INDEX = "_row"


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
    return frame.select(pl.col(keys), pl.col(plan.payload_sources).exclude(keys)).unique(
        subset=keys, keep="first", maintain_order=True
    )


def _var_raw(
    frame: pl.DataFrame, plan: AxisSourcePlan, values: tuple[FeatureValuePolicy, ...]
) -> VarRaw:
    """One row per complete raw var-key tuple, each payload column resolved by its policy.

    A long report repeats a feature's payload in each of its rows. Each policy reduces those
    values to one and says whether they disagreed; a disagreement is reported, never turned
    into a second identity row. The policies are the payload: one per ``payload_sources``.
    """
    keys = plan.keys.raw_key_columns
    flags = tuple(f"{_DISAGREES}{index}" for index in range(len(values)))
    # Lazily, so only the columns the policies name are touched: an eager group_by first
    # rechunks every column of the read, layers included.
    resolved = (
        frame.lazy()
        .group_by(keys, maintain_order=True)
        .agg(
            *(policy.value(frame.schema).alias(policy.source) for policy in values),
            *(
                policy.disagrees(frame.schema).alias(flag)
                for policy, flag in zip(values, flags, strict=True)
            ),
        )
        .collect()
    )
    disagreements = {
        policy.source: _disagreeing(frame, resolved.filter(pl.col(flag)), keys, policy.source)
        for policy, flag in zip(values, flags, strict=True)
        if resolved.get_column(flag).any()
    }
    return VarRaw(frame=resolved.drop(flags), raw_key_columns=keys, disagreements=disagreements)


def _disagreeing(
    frame: pl.DataFrame, flagged: pl.DataFrame, keys: tuple[str, ...], source: str
) -> pl.DataFrame:
    """Each disagreeing feature's raw key, with the first distinct values its rows give."""
    return (
        frame.select(*keys, source)
        .join(flagged.select(keys), on=list(keys), nulls_equal=True, maintain_order="right")
        .filter(~absent(pl.col(source), frame.schema[source]))
        .group_by(keys, maintain_order=True)
        .agg(
            pl.col(source).unique(maintain_order=True).head(_VALUE_LIMIT).alias(DISAGREEING_VALUES)
        )
    )


def _row_numbers(frame: pl.DataFrame, axis: pl.DataFrame, keys: tuple[str, ...]) -> pl.Series:
    """Each physical row's row in one raw axis, by one join on the raw key columns alone.

    Every physical row's key tuple is in the axis it built, so no row number is null.
    """
    index = _ROW_INDEX
    while index in keys:
        index += "_"
    return (
        frame.lazy()
        .select(keys)
        .join(
            axis.lazy().select(keys).with_row_index(index),
            on=list(keys),
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        .select(index)
        .collect()
        .to_series()
    )


@dataclass(frozen=True, slots=True)
class LongSourceDecomposer:
    """One physical row per (observation, feature); every layer names an exact column."""

    primary_layer_name: str
    layer_sources: tuple[LongRawLayerSource, ...]
    obs: AxisSourcePlan
    var: AxisSourcePlan
    var_values: tuple[FeatureValuePolicy, ...]

    def decompose(self, table: LevelSourceTable, /) -> RawLevel:
        """Both raw axes, then every layer as the read column beside two shared row numbers."""
        frame = table.frame
        obs_frame = _axis_frame(frame, self.obs)
        var = _var_raw(frame, self.var, self.var_values)
        return self.raw_level(
            frame,
            obs_frame,
            var,
            _row_numbers(frame, var.frame, self.var.keys.raw_key_columns),
            _row_numbers(frame, obs_frame, self.obs.keys.raw_key_columns),
        )

    def raw_level(
        self,
        frame: pl.DataFrame,
        obs_frame: pl.DataFrame,
        var: VarRaw,
        var_row: pl.Series,
        obs_row: pl.Series,
    ) -> RawLevel:
        """The level whose layers are ``frame``'s source columns at the given row numbers."""
        var_row, obs_row = var_row.alias(VAR_ROW), obs_row.alias(OBS_ROW)
        return RawLevel(
            obs=ObsRaw(frame=obs_frame, raw_key_columns=self.obs.keys.raw_key_columns),
            var=var,
            primary_layer_name=self.primary_layer_name,
            layers=tuple(
                LongLayer(
                    source.name,
                    pl.DataFrame(
                        [var_row, obs_row, frame.get_column(source.source_column).alias(VALUE)]
                    ),
                )
                for source in self.layer_sources
            ),
        )


@dataclass(frozen=True, slots=True)
class WideSourceDecomposer:
    """One physical row per feature; the observations came from resolved header captures."""

    primary_layer_name: str
    layer_plans: tuple[WideRawLayerPlan, ...]
    obs: AxisSourcePlan
    var: AxisSourcePlan
    var_values: tuple[FeatureValuePolicy, ...]

    def decompose(self, table: LevelSourceTable, /) -> RawLevel:
        """The samples as the obs axis, then every layer as one block of rows per column."""
        frame = table.frame
        samples = self._samples()
        obs_frame = pl.DataFrame({name: list(samples) for name in self.obs.keys.raw_key_columns})
        var = _var_raw(frame, self.var, self.var_values)
        var_row = _row_numbers(frame, var.frame, self.var.keys.raw_key_columns).alias(VAR_ROW)
        return RawLevel(
            obs=ObsRaw(frame=obs_frame, raw_key_columns=self.obs.keys.raw_key_columns),
            var=var,
            primary_layer_name=self.primary_layer_name,
            layers=tuple(self._layer(frame, var_row, samples, plan) for plan in self.layer_plans),
        )

    def _samples(self) -> tuple[str, ...]:
        """The primary layer defines the observation axis, in stable header order."""
        primary = next(plan for plan in self.layer_plans if plan.name == self.primary_layer_name)
        return tuple(dict.fromkeys(source.sample for source in primary.sources))

    @staticmethod
    def _layer(
        frame: pl.DataFrame, var_row: pl.Series, samples: tuple[str, ...], plan: WideRawLayerPlan
    ) -> LongLayer:
        """One block of rows per resolved column, the first column of every sample first.

        Several physical columns claiming one sample are repeated cells, so long and wide
        inputs reach the duplicate policy in the same shape; a cell's repeats keep header
        order, then row order. A sample the layer lacks contributes no rows.
        """
        by_sample: dict[str, list[str]] = {}
        for source in plan.sources:
            by_sample.setdefault(source.sample, []).append(source.source_column)
        depth = max((len(columns) for columns in by_sample.values()), default=0)
        rows = pl.DataFrame([var_row])
        blocks = [
            rows.with_columns(
                pl.lit(position, dtype=pl.UInt32).alias(OBS_ROW),
                frame.get_column(by_sample[sample][occurrence]).alias(VALUE),
            )
            for occurrence in range(depth)
            for position, sample in enumerate(samples)
            if occurrence < len(by_sample.get(sample, ()))
        ]
        empty = pl.DataFrame(schema={VAR_ROW: pl.UInt32, OBS_ROW: pl.UInt32, VALUE: pl.Null})
        return LongLayer(plan.name, pl.concat(blocks, how="vertical_relaxed") if blocks else empty)


@dataclass(frozen=True, slots=True)
class DelimitedFragmentSourceDecomposer:
    """Precursor rows packing fragment lists: axes and row numbers first, then the explode.

    The run and precursor keys are joined to their axes on the precursor rows, so the explode
    multiplies two integers and the scalars, never key or payload text. The var payload is a
    precursor's: its value policies run once per precursor on the rows that hold fragments, and
    each fragment of the precursor takes the result. The fragment axis is one row per
    (precursor, label) in order of first appearance, its precursor key columns restored
    afterwards. Everything else is the long decomposer's: its plans and its layers.
    """

    separator: FragmentTableSeparator
    long_decomposer: LongSourceDecomposer

    def decompose(self, table: LevelSourceTable, /) -> RawLevel:
        long = self.long_decomposer
        label = self.separator.label_output
        frame = table.frame
        precursor_keys = tuple(key for key in long.var.keys.raw_key_columns if key != label)
        # Only the axis columns are filtered, so the packed text is never copied.
        axis_columns = dict.fromkeys(
            (
                *long.obs.keys.raw_key_columns,
                *long.obs.payload_sources,
                *precursor_keys,
                *long.var.payload_sources,
            )
        )
        holding = frame.select(list(axis_columns)).filter(self.separator.token_counts(frame) > 0)
        obs_frame = _axis_frame(holding, long.obs)
        precursors = _var_raw(
            holding,
            AxisSourcePlan(
                keys=replace(long.var.keys, raw_key_columns=precursor_keys),
                payload_sources=long.var.payload_sources,
            ),
            long.var_values,
        )
        del holding
        carried = (*long.var.payload_sources, *self.separator.packed_sources, label)
        precursor, obs_row = _free("_precursor_row", carried), _free("_obs_row", carried)
        exploded = self.separator.exploded(
            pl.DataFrame(
                [
                    _row_numbers(frame, precursors.frame, precursor_keys).alias(precursor),
                    _row_numbers(frame, obs_frame, long.obs.keys.raw_key_columns).alias(obs_row),
                    *frame.select(self.separator.packed_sources).get_columns(),
                ]
            )
        )
        fragments, var_row = _FragmentKeys.of(exploded, precursor, label).axis(
            exploded, precursors.frame.height
        )
        named = precursors.frame.with_row_index(precursor)
        keys = long.var.keys.raw_key_columns
        var_frame = _with_precursor_keys(self.separator.labelled(fragments), named, precursor, keys)
        return long.raw_level(
            exploded,
            obs_frame,
            VarRaw(
                frame=var_frame,
                raw_key_columns=keys,
                disagreements={
                    source: var_frame.select(keys).join(
                        disagreeing,
                        on=list(precursor_keys),
                        how="inner",
                        nulls_equal=True,
                        maintain_order="left",
                    )
                    for source, disagreeing in precursors.disagreements.items()
                },
            ),
            var_row,
            exploded.get_column(obs_row),
        )


_FRAGMENT_SLICE_ROWS = 32_000_000
"""Exploded rows keyed at a time: one slice's UInt64 keys take about 256 MB."""


@dataclass(frozen=True, slots=True)
class _FragmentKeys:
    """One integer per (precursor row, label): precursor row times the label count, plus a code.

    A fragment's identity is two small columns, so its axis needs no hash over every scalar:
    first appearance is the order of the distinct keys, and each scalar's ``var_row`` is a
    lookup into a dense table of one slot per possible key. A two-column join over the 54 M
    scalars of a 150-run DIA-NN report took 4 GB on top of them. The label code is its index
    among the distinct labels; the last code is a missing label.
    """

    precursor: str
    label: str
    labels: pl.Series
    codes: pl.Series

    @classmethod
    def of(cls, exploded: pl.DataFrame, precursor: str, label: str) -> _FragmentKeys:
        """The distinct labels in order of appearance, and each physical label's code."""
        labels = exploded.get_column(label).drop_nulls().unique(maintain_order=True)
        physical = labels.to_physical()
        size = labels.len() and physical.to_frame().select(pl.first().max() + 1).item()
        codes = pl.repeat(None, size, dtype=pl.UInt32, eager=True).scatter(
            physical, pl.int_range(0, labels.len(), dtype=pl.UInt32, eager=True)
        )
        return cls(precursor, label, labels, codes)

    @property
    def width(self) -> int:
        """Codes per precursor: one per distinct label, and one for a missing label."""
        return self.labels.len() + 1

    def axis(self, exploded: pl.DataFrame, precursors: int) -> tuple[pl.DataFrame, pl.Series]:
        """The fragment axis in order of first appearance, and each scalar's row in it."""
        slices = [
            exploded.slice(offset, _FRAGMENT_SLICE_ROWS)
            for offset in range(0, exploded.height, _FRAGMENT_SLICE_ROWS)
        ]
        first = pl.concat(
            [self._keys(part).unique(maintain_order=True) for part in slices],
            how="vertical",
        ).unique(maintain_order=True)
        rows = pl.int_range(0, first.len(), dtype=pl.UInt32, eager=True)
        lookup = pl.repeat(None, precursors * self.width, dtype=pl.UInt32, eager=True).scatter(
            first, rows
        )
        var_row = pl.concat([lookup.gather(self._keys(part)) for part in slices], how="vertical")
        missing = pl.Series([None], dtype=self.labels.dtype)
        fragments = pl.DataFrame(
            [
                (first // self.width).cast(pl.UInt32).alias(self.precursor),
                pl.concat([self.labels, missing]).gather(first % self.width).alias(self.label),
            ]
        )
        return fragments, var_row

    def _keys(self, rows: pl.DataFrame) -> pl.Series:
        code = self.codes.gather(rows.get_column(self.label).to_physical())
        return rows.get_column(self.precursor).cast(pl.UInt64) * self.width + code.fill_null(
            self.width - 1
        ).cast(pl.UInt64)


def _free(name: str, taken: tuple[str, ...]) -> str:
    """``name``, lengthened until no carried vendor column holds it."""
    while name in taken:
        name += "_"
    return name


def _with_precursor_keys(
    frame: pl.DataFrame, precursors: pl.DataFrame, precursor: str, keys: tuple[str, ...]
) -> pl.DataFrame:
    """A frame keyed by precursor row number, with the precursor's columns instead."""
    restored = frame.join(precursors, on=precursor, how="left", maintain_order="left")
    return restored.select(*keys, pl.exclude(precursor, *keys))
