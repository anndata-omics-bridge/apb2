"""Parking: a parsed level's layer cells move to memory-mapped files and read as before."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VALUE, VAR_POS
from apb2.parserV2.parse_quant.data.parking import LayerParking
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    VarFinal,
)


def _level(cells: pl.DataFrame) -> ParsedLevel:
    return ParsedLevel(
        obs=ObsFinal(frame=pl.DataFrame({"run": ["A", "B"]}), key_columns=("run",)),
        var=VarFinal(frame=pl.DataFrame({"feature": ["F", "G", "H"]}), key_columns=("feature",)),
        primary_layer_name="Intensity",
        uns={},
        layers={
            "Intensity": FinalLayerTable("Intensity", cells, height=3, labels=("obs_0", "obs_1")),
            "Empty": FinalLayerTable(
                "Empty",
                pl.DataFrame(schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64}),
                height=3,
                labels=("obs_0", "obs_1"),
            ),
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
    )


def test_parked_cells_are_sorted_by_position_and_widen_as_before(tmp_path: Path) -> None:
    cells = pl.DataFrame(
        {VAR_POS: [2, 0, 1, 0], OBS_POS: [1, 1, 0, 0], VALUE: [4.0, None, 2.0, 1.0]},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64},
    )
    level = _level(cells)
    wide = level.layers["Intensity"].wide()

    LayerParking(tmp_path).park(level)

    parked = level.layers["Intensity"]
    assert parked.cells.rows() == [(0, 0, 1.0), (0, 1, None), (1, 0, 2.0), (2, 1, 4.0)]
    assert parked.cells.schema == cells.schema
    assert parked.wide().equals(wide)
    assert level.layers["Empty"].cells.height == 0


def test_a_parked_file_is_gone_once_mapped(tmp_path: Path) -> None:
    cells = pl.DataFrame(
        {VAR_POS: [0], OBS_POS: [0], VALUE: [1.0]},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64},
    )
    level = _level(cells)

    LayerParking(tmp_path).park(level)

    assert list(tmp_path.iterdir()) == []
    assert level.layers["Intensity"].cells.get_column(VALUE).to_list() == [1.0]
