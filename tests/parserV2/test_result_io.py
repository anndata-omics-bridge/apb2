"""Result I/O laws: exact columnar round-trips and canonical AnnData projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apb2.cli import reformat as reformat_command
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.duckdb_io import METADATA_TABLE
from apb2.parserV2.parse_quant.errors import InvalidResultError, UnsupportedResultFormatError
from apb2.parserV2.parse_quant.parquet_writer import MANIFEST_NAME
from apb2.parserV2.parse_quant.result_io import (
    ParsedLevelsReader,
    ParsedLevelsWriter,
    ResultFormat,
    read_parsed_levels,
    reader_for,
    reformat,
    result_format_for,
    write_parsed_levels,
    writer_for,
)


def _plan(layer_names: tuple[str, ...]) -> str:
    encodings: list[dict[str, object]] = []
    for name in layer_names:
        if name == "Status":
            encodings.append(
                {
                    "kind": "factor",
                    "layer_name": name,
                    "categories": [["MS/MS", 1], ["MBR", 2]],
                }
            )
        else:
            encodings.append(
                {
                    "kind": "plain_numeric",
                    "layer_name": name,
                    "missing_values": [0.0] if name == "Intensity" else [],
                    "number_format": {"decimal_mark": ".", "thousands_marks": []},
                }
            )
    return json.dumps(
        {
            "ann_data": {
                "layer_encodings": encodings,
                "layer_contract": {
                    "primary_layer_name": "Intensity",
                    "required_names": ["Intensity"],
                    "empty_ratio": 0.001,
                    "populated_ratio": 0.5,
                },
            }
        }
    )


def _level(name: str, feature_column: str) -> ParsedLevel:
    obs = pl.DataFrame(
        {
            "Run": ["Zürich", "東京"],
            "Batch": ["one", "one"],
            "NumericText": ["001", "002"],
        }
    )
    var_key_columns = (feature_column,) if name == "ion" else (feature_column, "Isoform")
    var_values: dict[str, list[object]] = {
        feature_column: ["F1", "F2"],
        "Charge": [2, None],
        "Decoy": [False, True],
    }
    if len(var_key_columns) == 2:
        var_values["Isoform"] = ["canonical", "alternative"]
    var = pl.DataFrame(var_values)
    layer_keys = {column: var.get_column(column) for column in var_key_columns}
    layers = {
        "Intensity": FinalLayerTable(
            layer_name="Intensity",
            var_key_columns=var_key_columns,
            values=pl.DataFrame(
                {
                    **layer_keys,
                    "obs_0": ["100.5", "0"],
                    "obs_1": ["200.5", None],
                }
            ),
        ),
        "Status": FinalLayerTable(
            layer_name="Status",
            var_key_columns=var_key_columns,
            values=pl.DataFrame(
                {
                    **layer_keys,
                    "obs_0": ["MS/MS", "MBR"],
                    "obs_1": ["MBR", None],
                }
            ),
        ),
    }
    return ParsedLevel(
        obs=ObsFinal(frame=obs, key_columns=("Run",)),
        var=VarFinal(frame=var, key_columns=var_key_columns),
        primary_layer_name="Intensity",
        layers=layers,
        obsm={
            "A/B": pl.DataFrame(
                {"condition": pl.Series(["treated", "control"]).cast(pl.Categorical)}
            ),
            "A?B": pl.DataFrame({"score": [1.0, float("nan")]}),
        },
        varm={"annotation": pl.DataFrame({"gene": ["G1", None]})},
        obsp={
            "correlation": pl.DataFrame({"row": [1, 0], "column": [0, 1], "value": [0.75, 0.75]})
        },
        varp={"similarity": pl.DataFrame({"row": [0], "column": [1], "value": [0.25]})},
        uns={
            "quantification_level": name,
            "software_name": "Synthetic Ω",
            "plan_json": _plan(tuple(layers)),
            "nested": {"tokens": ["x", "y"], "enabled": True},
        },
    )


def rich_result() -> ParsedLevels:
    return ParsedLevels(
        levels={
            "ion": _level("ion", "Ion"),
            "protein": _level("protein", "Protein"),
        },
        uns={"produced_by": "apb2", "quantification_levels": ["ion", "protein"]},
    )


def _assert_result_equal(actual: ParsedLevels, expected: ParsedLevels) -> None:
    assert list(actual.levels) == list(expected.levels)
    assert actual.uns == expected.uns
    for name, wanted in expected.levels.items():
        got = actual.levels[name]
        assert got.primary_layer_name == wanted.primary_layer_name
        assert got.obs.key_columns == wanted.obs.key_columns
        assert got.var.key_columns == wanted.var.key_columns
        assert got.uns == wanted.uns
        assert_frame_equal(got.obs.frame, wanted.obs.frame)
        assert_frame_equal(got.var.frame, wanted.var.frame)
        _assert_layer_mapping(got.layers, wanted.layers)
        _assert_frame_mapping(got.obsm, wanted.obsm)
        _assert_frame_mapping(got.varm, wanted.varm)
        _assert_frame_mapping(got.obsp, wanted.obsp)
        _assert_frame_mapping(got.varp, wanted.varp)


def _assert_layer_mapping(
    actual: Mapping[str, FinalLayerTable], expected: Mapping[str, FinalLayerTable]
) -> None:
    assert list(actual) == list(expected)
    for name, wanted in expected.items():
        got = actual[name]
        assert got.layer_name == wanted.layer_name
        assert got.var_key_columns == wanted.var_key_columns
        assert_frame_equal(got.values, wanted.values)


def _assert_frame_mapping(
    actual: Mapping[str, pl.DataFrame], expected: Mapping[str, pl.DataFrame]
) -> None:
    assert list(actual) == list(expected)
    for name, wanted in expected.items():
        assert_frame_equal(actual[name], wanted)


@pytest.mark.parametrize("result_format", [ResultFormat.PARQUET, ResultFormat.DUCKDB])
def test_columnar_formats_round_trip_exactly(result_format: ResultFormat, tmp_path: Path) -> None:
    suffix = ".parquet" if result_format is ResultFormat.PARQUET else ".duckdb"
    target = tmp_path / f"result{suffix}"
    writer: ParsedLevelsWriter = writer_for(result_format)
    reader: ParsedLevelsReader = reader_for(result_format)

    writer.write(rich_result(), target)

    _assert_result_equal(reader.read(target), rich_result())


@pytest.mark.parametrize(
    ("source_format", "target_format"),
    [
        (ResultFormat.PARQUET, ResultFormat.DUCKDB),
        (ResultFormat.DUCKDB, ResultFormat.PARQUET),
    ],
)
def test_columnar_crossings_are_exact(
    source_format: ResultFormat,
    target_format: ResultFormat,
    tmp_path: Path,
) -> None:
    suffix = {ResultFormat.PARQUET: ".parquet", ResultFormat.DUCKDB: ".duckdb"}
    source = tmp_path / f"source{suffix[source_format]}"
    target = tmp_path / f"target{suffix[target_format]}"
    writer_for(source_format).write(rich_result(), source)

    reformat(source, target)

    _assert_result_equal(reader_for(target_format).read(target), rich_result())


def test_h5mu_projection_is_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.h5mu"
    second = tmp_path / "second.h5mu"
    writer_for(ResultFormat.H5MU).write(rich_result(), first)
    projected = reader_for(ResultFormat.H5MU).read(first)

    writer_for(ResultFormat.H5MU).write(projected, second)

    _assert_result_equal(reader_for(ResultFormat.H5MU).read(second), projected)
    assert projected.levels["ion"].layers["Status"].values.get_column("obs_0").to_list() == [
        1.0,
        2.0,
    ]


def test_h5ad_accepts_exactly_one_level_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "ion.h5ad"
    single = ParsedLevels(
        levels={"ion": rich_result().levels["ion"]},
        uns={"selected": "ion"},
    )
    writer_for(ResultFormat.H5AD).write(single, target)
    projected = reader_for(ResultFormat.H5AD).read(target)
    second = tmp_path / "again.h5ad"

    writer_for(ResultFormat.H5AD).write(projected, second)

    _assert_result_equal(reader_for(ResultFormat.H5AD).read(second), projected)
    untouched = tmp_path / "untouched.h5ad"
    untouched.write_bytes(b"previous")
    with pytest.raises(InvalidResultError, match="exactly one"):
        writer_for(ResultFormat.H5AD).write(rich_result(), untouched)
    assert untouched.read_bytes() == b"previous"


@pytest.mark.parametrize("source_format", [ResultFormat.PARQUET, ResultFormat.DUCKDB])
def test_columnar_to_h5mu_yields_the_declared_matrix_projection(
    source_format: ResultFormat,
    tmp_path: Path,
) -> None:
    suffix = ".parquet" if source_format is ResultFormat.PARQUET else ".duckdb"
    source = tmp_path / f"raw{suffix}"
    projected_path = tmp_path / "expected.h5mu"
    crossed_path = tmp_path / "crossed.h5mu"
    writer_for(source_format).write(rich_result(), source)
    writer_for(ResultFormat.H5MU).write(rich_result(), projected_path)
    expected = reader_for(ResultFormat.H5MU).read(projected_path)

    reformat(source, crossed_path)

    _assert_result_equal(reader_for(ResultFormat.H5MU).read(crossed_path), expected)


@pytest.mark.parametrize("target_format", [ResultFormat.PARQUET, ResultFormat.DUCKDB])
def test_h5mu_to_columnar_preserves_the_represented_projection(
    target_format: ResultFormat,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5mu"
    suffix = ".parquet" if target_format is ResultFormat.PARQUET else ".duckdb"
    target = tmp_path / f"target{suffix}"
    writer_for(ResultFormat.H5MU).write(rich_result(), source)
    expected = reader_for(ResultFormat.H5MU).read(source)

    reformat(source, target)

    _assert_result_equal(reader_for(target_format).read(target), expected)


@pytest.mark.parametrize("columnar_format", [ResultFormat.PARQUET, ResultFormat.DUCKDB])
def test_one_level_h5ad_crosses_columnar_formats_in_both_directions(
    columnar_format: ResultFormat,
    tmp_path: Path,
) -> None:
    raw = ParsedLevels(levels={"ion": rich_result().levels["ion"]}, uns={"selected": "ion"})
    suffix = ".parquet" if columnar_format is ResultFormat.PARQUET else ".duckdb"
    columnar = tmp_path / f"raw{suffix}"
    h5ad = tmp_path / "matrix.h5ad"
    restored = tmp_path / f"restored{suffix}"
    writer_for(columnar_format).write(raw, columnar)

    reformat(columnar, h5ad)
    projection = reader_for(ResultFormat.H5AD).read(h5ad)
    reformat(h5ad, restored)

    _assert_result_equal(reader_for(columnar_format).read(restored), projection)


def test_path_inference_conveniences_use_the_same_adapters(tmp_path: Path) -> None:
    parquet = tmp_path / "result.parquet"
    duckdb = tmp_path / "result.duckdb"

    write_parsed_levels(rich_result(), parquet)
    write_parsed_levels(read_parsed_levels(parquet), duckdb)

    _assert_result_equal(read_parsed_levels(duckdb), rich_result())
    assert result_format_for(Path("x.h5ad")) is ResultFormat.H5AD
    assert result_format_for(Path("x.h5mu")) is ResultFormat.H5MU
    with pytest.raises(UnsupportedResultFormatError, match="unsupported result suffix"):
        result_format_for(Path("x.tsv"))


def test_writers_validate_aligned_and_pairwise_values_before_touching_target(
    tmp_path: Path,
) -> None:
    parsed = rich_result()
    parsed.levels["ion"].obsm["bad"] = pl.DataFrame({"x": [1]})
    target = tmp_path / "result.duckdb"
    target.write_bytes(b"previous")

    with pytest.raises(InvalidResultError, match="axis has 2"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert target.read_bytes() == b"previous"


def test_a_vendor_parquet_file_is_not_an_apb2_result(tmp_path: Path) -> None:
    source = tmp_path / "vendor.parquet"
    pl.DataFrame({"Intensity": [1.0]}).write_parquet(source)

    with pytest.raises(InvalidResultError, match="directory"):
        reader_for(ResultFormat.PARQUET).read(source)


def test_readers_reject_unsupported_columnar_versions(tmp_path: Path) -> None:
    parquet = tmp_path / "result.parquet"
    duckdb_path = tmp_path / "result.duckdb"
    write_parsed_levels(rich_result(), parquet)
    write_parsed_levels(rich_result(), duckdb_path)

    manifest_path = parquet / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "unsupported"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with duckdb.connect(str(duckdb_path)) as connection:
        raw = connection.execute(f"SELECT manifest_json FROM {METADATA_TABLE}").fetchone()
        assert raw is not None
        duckdb_manifest = json.loads(raw[0])
        duckdb_manifest["format_version"] = "unsupported"
        connection.execute(
            f"UPDATE {METADATA_TABLE} SET manifest_json = ?",
            [json.dumps(duckdb_manifest)],
        )

    with pytest.raises(InvalidResultError, match="version"):
        read_parsed_levels(parquet)
    with pytest.raises(InvalidResultError, match="version"):
        read_parsed_levels(duckdb_path)


@pytest.mark.parametrize(
    ("result_format", "suffix"),
    [
        (ResultFormat.H5AD, ".h5ad"),
        (ResultFormat.H5MU, ".h5mu"),
        (ResultFormat.PARQUET, ".parquet"),
        (ResultFormat.DUCKDB, ".duckdb"),
    ],
)
def test_every_writer_rejects_an_empty_collection_before_touching_the_target(
    result_format: ResultFormat,
    suffix: str,
    tmp_path: Path,
) -> None:
    target = tmp_path / f"result{suffix}"
    target.write_bytes(b"previous")

    with pytest.raises(InvalidResultError, match="at least one level"):
        writer_for(result_format).write(ParsedLevels(levels={}, uns={}), target)

    assert target.read_bytes() == b"previous"


def test_h5_writer_requires_the_stored_matrix_plan_before_touching_target(tmp_path: Path) -> None:
    parsed = rich_result()
    del parsed.levels["protein"].uns["plan_json"]
    target = tmp_path / "result.h5mu"
    target.write_bytes(b"previous")

    with pytest.raises(InvalidResultError, match="plan_json"):
        writer_for(ResultFormat.H5MU).write(parsed, target)

    assert target.read_bytes() == b"previous"


def test_pairwise_coordinates_are_validated_independently(tmp_path: Path) -> None:
    parsed = rich_result()
    parsed.levels["ion"].obsp["bad"] = pl.DataFrame({"row": [0], "column": [2], "value": [1.0]})
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="outside"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_reformat_cli_command_delegates_to_the_result_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    target = tmp_path / "target.duckdb"
    write_parsed_levels(rich_result(), source)

    assert reformat_command(source, target) == 0
    _assert_result_equal(read_parsed_levels(target), rich_result())
    assert reformat_command(source, tmp_path / "bad.tsv") == 1
