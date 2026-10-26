"""Layer blocks: each block of rows or columns is exactly that part of the wide matrix."""

from __future__ import annotations

import polars as pl
import pytest

from apb2.parserV2.parse_quant.data.layer_blocks import LayerBlocks
from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS
from apb2.parserV2.parse_quant.data.parsed import (
    CategoricalLayerSemantics,
    FinalLayerSemantics,
    FinalLayerTable,
    MeasurementLayerRole,
    QuantitativeLayerSemantics,
)

LABELS = ("obs_0", "obs_1", "obs_2")


def _layer(*, ordered: bool, semantics: FinalLayerSemantics) -> FinalLayerTable:
    rows = [(0, 0, 1), (0, 2, 2), (2, 1, 3), (3, 0, 4), (4, 2, None)]
    if not ordered:
        rows = rows[::-1]
    var, obs, value = zip(*rows, strict=True)
    cells = pl.DataFrame(
        {VAR_POS: var, OBS_POS: obs, VALUE: value},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Int64},
    )
    return FinalLayerTable("L", cells, height=6, labels=LABELS, semantics=semantics)


SEMANTICS = [
    QuantitativeLayerSemantics(logical_type="integer"),
    CategoricalLayerSemantics(categories=(("a", 1), ("b", 2), ("c", 3), ("d", 4))),
]


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("step", [1, 2, 4, 100])
def test_row_blocks_are_the_wide_rows(
    ordered: bool, semantics: FinalLayerSemantics, step: int
) -> None:
    layer = _layer(ordered=ordered, semantics=semantics)
    wide = layer.wide()

    blocks = list(LayerBlocks(layer).rows(step))

    assert [start for start, _block in blocks] == list(range(0, 6, step))
    assert pl.concat([block for _start, block in blocks]).equals(wide)


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("block_bytes", [1, 8 * 6 * 2, 1 << 20])
def test_column_blocks_are_the_wide_columns(
    semantics: FinalLayerSemantics, block_bytes: int
) -> None:
    layer = _layer(ordered=False, semantics=semantics)

    blocks = list(LayerBlocks(layer, block_bytes=block_bytes).columns())

    assert pl.concat(blocks, how="horizontal").equals(layer.wide())


def test_an_empty_layer_is_one_empty_block_of_each_kind() -> None:
    empty = FinalLayerTable(
        "L",
        pl.DataFrame(schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64}),
        height=0,
        labels=LABELS,
    )

    rows = list(LayerBlocks(empty).rows())
    columns = list(LayerBlocks(empty).columns())

    assert [(start, block.shape) for start, block in rows] == [(0, (0, 3))]
    assert [block.shape for block in columns] == [(0, 3)]


def test_a_layer_misses_cells_unless_it_fills_them_with_a_code() -> None:
    quantitative = _layer(ordered=True, semantics=SEMANTICS[0])
    categorical = _layer(ordered=True, semantics=SEMANTICS[1])

    assert LayerBlocks(quantitative).has_missing()
    assert not LayerBlocks(categorical).has_missing()


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("step", [1, 2, 6])
def test_a_layer_stored_from_row_blocks_is_the_layer_from_wide(
    semantics: FinalLayerSemantics, step: int
) -> None:
    wide = _layer(ordered=False, semantics=semantics).wide()
    blocks = [wide.slice(start, step) for start in range(0, wide.height, step)]

    stored = LayerBlocks.stored(
        "L", blocks, labels=LABELS, role=MeasurementLayerRole(), semantics=semantics
    )

    expected = FinalLayerTable.from_wide("L", wide, semantics=semantics)
    assert stored.height == expected.height
    assert stored.wide().equals(expected.wide())
    assert stored.cells.sort(VAR_POS, OBS_POS).equals(expected.cells.sort(VAR_POS, OBS_POS))
    assert stored.cells.get_column(VAR_POS).is_sorted()


def test_a_layer_without_observations_is_stored_without_cells() -> None:
    stored = LayerBlocks.stored(
        "L",
        [pl.DataFrame(), pl.DataFrame()],
        labels=(),
        role=MeasurementLayerRole(),
        semantics=QuantitativeLayerSemantics(),
    )

    assert stored.cells.schema == FinalLayerTable.from_wide("L", pl.DataFrame()).cells.schema
