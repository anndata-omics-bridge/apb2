"""Decomposed physical shape: small raw axes and each layer as long rows of integer positions.

These are the states between one physical table and one parsed level. Identity is explicit:
every axis states the ordered raw-key columns that distinguish its rows. A layer never repeats
those keys; each of its rows names its cell by the row it points at in each raw axis
(``var_row``, ``obs_row``), assigned once by joining the raw keys against the axis.

After axis preparation ``FinalPositions`` turns those rows into final positions (``var_pos``,
``obs_pos``) and drops the rows whose axis row did not survive. Duplicates and value parsing
work on these long rows; a layer becomes wide only when it is widened, last.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import pairwise

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS, ObservationGrid

DISAGREEING_VALUES = "_disagreeing_values"
"""The column of a disagreement frame listing the distinct values one feature's rows gave."""

VAR_ROW = "var_row"
"""A long row's row in the raw var axis frame."""

OBS_ROW = "obs_row"
"""A long row's row in the raw obs axis frame."""

_REPEAT_BLOCK_ROWS = 32_000_000
"""About how many placed rows one repeated-cell check hashes at a time: about 1.7 GB."""


@dataclass(slots=True)
class ObsRaw:
    """One stable-first row per complete raw obs-key tuple."""

    frame: pl.DataFrame
    # pl.DataFrame({"sample": ["A", "B", "C"]})

    raw_key_columns: tuple[str, ...]
    # ("sample",)


@dataclass(slots=True)
class VarRaw:
    """One row per complete raw var-key tuple, with its payload metadata.

    ``disagreements`` holds, per payload source whose repeated rows gave a feature different
    values, each such feature's raw key and the values (``DISAGREEING_VALUES``). Axis
    preparation may still drop a feature, so the parser counts only those that remain.
    """

    frame: pl.DataFrame
    # pl.DataFrame({
    #     "sequence": ["PEPMIDE", "OTHER"],
    #     "charge": ["2", "3"],
    #     "genes": ["GENE1", "GENE2"],
    # })

    raw_key_columns: tuple[str, ...]
    # ("sequence", "charge")

    disagreements: dict[str, pl.DataFrame] = field(default_factory=dict[str, pl.DataFrame])
    # {"genes": pl.DataFrame({"sequence": ["PEPMIDE"], ..., "_disagreeing_values": [["G1", "G2"]]})}


@dataclass(slots=True)
class LongLayer:
    """One measurement as long rows: two axis references and a value, in physical order.

    Decomposition gives ``var_row`` and ``obs_row``; placing gives ``var_pos`` and ``obs_pos``.
    A repeated pair is a repeated cell, which is the duplicate policy's question. ``var_keys``
    holds, once placed, the raw var keys of the final var rows, for error examples only.
    """

    layer_name: str
    # "Intensity"

    rows: pl.DataFrame
    # pl.DataFrame({"var_row": [0, 0, 1], "obs_row": [0, 0, 1], "value": ["1.5", "2", "7"]})

    var_keys: pl.DataFrame = field(default_factory=pl.DataFrame)
    # pl.DataFrame({"sequence": ["PEPMIDE", "OTHER"], "charge": ["2", "3"]})

    def repeats(self) -> pl.Series:
        """Which placed rows share their cell with another row.

        Rows sharing a cell share its ``var_pos``, so a large layer is checked one ``var_pos``
        range at a time, each on one UInt64 cell key: hashing both position columns of a
        fragment layer at once (54 M rows at 150 runs) took 9 GB, more than the rows; one range
        of 32 M rows takes about 1.7 GB.
        """
        cell = pl.col(VAR_POS).cast(pl.UInt64) * (1 << 32) + pl.col(OBS_POS).cast(pl.UInt64)
        blocks = -(-self.rows.height // _REPEAT_BLOCK_ROWS)
        if blocks <= 1:
            return self.rows.select(cell.is_duplicated()).to_series()
        edges = pl.linear_space(
            0, self.rows.select(pl.col(VAR_POS).max() + 1).item(), blocks + 1, eager=True
        ).ceil()
        repeated = pl.repeat(value=False, n=self.rows.height, eager=True)
        indexed = self.rows.lazy().select(VAR_POS, OBS_POS).with_row_index("_row")
        for start, stop in pairwise(edges):
            within = (pl.col(VAR_POS) >= start) & (pl.col(VAR_POS) < stop)
            found = (
                indexed.filter(within)
                .filter(cell.is_duplicated())
                .select("_row")
                .collect()
                .to_series()
            )
            if found.len():
                repeated = repeated.scatter(found, values=True)
        return repeated

    def split(self) -> tuple[LongLayer, LongLayer]:
        """The placed rows alone in their cell, and those that repeat a cell."""
        repeated = self.repeats()
        if not repeated.any():
            return self, replace(self, rows=self.rows.clear())
        return (
            replace(self, rows=self.rows.filter(~repeated)),
            replace(self, rows=self.rows.filter(repeated)),
        )


@dataclass(slots=True)
class RawLevel:
    """What every physical shape reduces to: two small raw axes and long layers."""

    obs: ObsRaw
    var: VarRaw
    primary_layer_name: str
    layers: tuple[LongLayer, ...]
    # (intensity_long, q_value_long), in authored order


@dataclass(frozen=True, slots=True)
class FinalPositions:
    """Where each raw axis row ended up: its final position, or null when it left its axis."""

    var: pl.Series
    # pl.Series("var_pos", [0, None, 1], dtype=pl.UInt32)

    obs: pl.Series
    # pl.Series("obs_pos", [0, 1], dtype=pl.UInt32)

    var_keys: pl.DataFrame
    # The raw var keys of the final var rows, in final order.

    grid: ObservationGrid
    # ObservationGrid(height=2, labels=("obs_0", "obs_1"))

    def place(self, layer: LongLayer, like: LongLayer | None = None) -> LongLayer:
        """The layer's rows at final positions, without those pointing at a dropped axis row.

        ``like`` is a layer already placed from the same raw rows: when it kept every row, its
        position columns are reused, so layers read from one table share one copy of them.
        """
        if like is not None and like.rows.height == layer.rows.height:
            positions = like.rows.select(VAR_POS, OBS_POS)
            return LongLayer(
                layer.layer_name,
                positions.with_columns(layer.rows.get_column(VALUE)),
                self.var_keys,
            )
        rows = pl.DataFrame(
            [
                self.var.gather(layer.rows.get_column(VAR_ROW)).alias(VAR_POS),
                self.obs.gather(layer.rows.get_column(OBS_ROW)).alias(OBS_POS),
                layer.rows.get_column(VALUE),
            ]
        )
        kept = pl.all_horizontal(pl.col(VAR_POS, OBS_POS).is_not_null())
        if not rows.select(kept.all()).item():
            rows = rows.filter(kept)
        return LongLayer(layer.layer_name, rows, self.var_keys)
