"""A layer's wide matrix one block at a time: feature rows for writers, observations for summaries.

``FinalLayerTable.wide`` builds the whole matrix, which is right for a reader asking for it and
wrong for a writer: a DIA-NN fragment layer of 2.8 M features and 1,236 runs is 27.6 GB wide.
Writers take it as blocks of feature rows, and the JSON summaries as blocks of observation
columns, each about ``BLOCK_BYTES`` and each widened by ``ObservationGrid``, so every block
holds exactly the values ``wide()`` would give at its rows and columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS, ObservationGrid
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerRole,
    FinalLayerSemantics,
    FinalLayerTable,
    layer_fill,
)

BLOCK_BYTES = 1 << 30
"""About how many bytes of wide values one block holds."""

READ_BLOCK_BYTES = 256 << 20
"""About how many bytes of a stored matrix a reader takes at a time; a block's cells cost more."""

_CELL_BYTES = 8


@dataclass(frozen=True, slots=True)
class LayerBlocks:
    """One layer's wide matrix in consecutive blocks of rows or of columns."""

    layer: FinalLayerTable
    block_bytes: int | None = None  # BLOCK_BYTES

    @property
    def row_step(self) -> int:
        """Feature rows per block: at least one."""
        width = max(len(self.layer.labels), 1)
        return max(self._bytes // (width * _CELL_BYTES), 1)

    @property
    def column_step(self) -> int:
        """Observation columns per block: at least one."""
        return max(self._bytes // (max(self.layer.height, 1) * _CELL_BYTES), 1)

    @property
    def _bytes(self) -> int:
        return self.block_bytes or BLOCK_BYTES

    def rows(self, step: int | None = None) -> Iterator[tuple[int, pl.DataFrame]]:
        """Each block's first feature row and its wide frame, one ``step`` of rows at a time.

        A layer without feature rows is one empty block, so a writer still sees its columns.
        Parked cells are sorted by ``var_pos``, so a block is one slice of them.
        """
        layer = self.layer
        step = step or self.row_step
        cells = layer.cells
        positions = cells.get_column(VAR_POS)
        ordered = positions.is_sorted()
        for start in range(0, max(layer.height, 1), step):
            stop = min(start + step, layer.height)
            if ordered:
                low, high = positions.search_sorted([start, stop], side="left").to_list()
                block = cells.slice(low, high - low)
            else:
                block = cells.filter(pl.col(VAR_POS).is_between(start, stop, closed="left"))
            yield (
                start,
                ObservationGrid(stop - start, layer.labels).widen(
                    block.get_column(VAR_POS) - start,
                    block.get_column(OBS_POS),
                    block.get_column(VALUE),
                    layer_fill(layer.semantics),
                ),
            )

    def columns(self) -> Iterator[pl.DataFrame]:
        """The wide frame one block of observation columns at a time, in column order.

        A layer without observations is one block without columns.
        """
        layer = self.layer
        step = self.column_step
        width = len(layer.labels)
        for start in range(0, max(width, 1), step):
            stop = min(start + step, width)
            block = layer.cells.filter(pl.col(OBS_POS).is_between(start, stop, closed="left"))
            yield ObservationGrid(layer.height, layer.labels[start:stop]).widen(
                block.get_column(VAR_POS),
                block.get_column(OBS_POS) - start,
                block.get_column(VALUE),
                layer_fill(layer.semantics),
            )

    @staticmethod
    def stored(
        layer_name: str,
        blocks: Iterable[pl.DataFrame],
        *,
        labels: tuple[str, ...],
        role: FinalLayerRole,
        semantics: FinalLayerSemantics,
    ) -> FinalLayerTable:
        """A layer from consecutive blocks of its wide rows, as ``from_wide`` of all of them.

        Each block keeps the cells that do not hold the fill, sorted by ``var_pos`` and
        ``obs_pos``, one column at a time: unpivoting a block would give every cell a label
        string first.
        """
        fill = layer_fill(semantics)
        parts: list[pl.DataFrame] = []
        height = 0
        for block in blocks:
            columns: list[pl.DataFrame] = []
            for position, column in enumerate(block.get_columns()):
                kept = (
                    column.is_not_null()
                    if fill is None
                    else column.is_null() | (column != fill).fill_null(value=True)
                )
                columns.append(
                    pl.DataFrame(
                        [
                            pl.int_range(height, height + block.height, dtype=pl.UInt32, eager=True)
                            .filter(kept)
                            .alias(VAR_POS),
                            pl.repeat(position, int(kept.sum()), dtype=pl.UInt32, eager=True).alias(
                                OBS_POS
                            ),
                            column.filter(kept).alias(VALUE),
                        ]
                    )
                )
            if columns:
                parts.append(pl.concat(columns, how="vertical_relaxed").sort(VAR_POS, OBS_POS))
            height += block.height
        cells = (
            pl.concat(parts, how="vertical_relaxed")
            if parts
            else pl.DataFrame(schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Null})
        )
        return FinalLayerTable(layer_name, cells, height, labels, role=role, semantics=semantics)

    def has_missing(self) -> bool:
        """Whether the wide matrix holds a null: an empty cell of a layer filled by null."""
        layer = self.layer
        if layer_fill(layer.semantics) is not None:
            return False
        cells = layer.cells
        return cells.height < layer.height * len(layer.labels) or bool(
            cells.get_column(VALUE).null_count()
        )
