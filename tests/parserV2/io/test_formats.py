"""Result I/O laws: exact columnar round-trips and canonical AnnData projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apb2.cli import reformat as reformat_command
from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    AuxiliaryLayerRole,
    FeatureRelation,
    FinalLayerTable,
    MeasurementLayerRole,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.anndata_writer import (
    AnnDataPlanError,
    numeric_result_level,
    quantitative_layer_values,
)
from apb2.parserV2.parse_quant.io.duckdb import METADATA_TABLE
from apb2.parserV2.parse_quant.io.errors import InvalidResultError, UnsupportedResultFormatError
from apb2.parserV2.parse_quant.io.formats import (
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
from apb2.parserV2.parse_quant.io.metadata import MATRIX_PROJECTED_KEY
from apb2.parserV2.parse_quant.io.parquet_writer import MANIFEST_NAME


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
            role=AuxiliaryLayerRole(),
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
        metadata={"level_extension": {"name": name}},
    )


def rich_result() -> ParsedLevels:
    return ParsedLevels(
        levels={
            "ion": _level("ion", "Ion"),
            "protein": _level("protein", "Protein"),
        },
        uns={"produced_by": "apb2", "quantification_levels": ["ion", "protein"]},
        metadata={"collection_extension": {"enabled": True}},
    )


def annotated_result() -> ParsedLevels:
    parsed = rich_result()
    parsed.annotation_tables["protein_group_members"] = AnnotationTable(
        frame=pl.DataFrame(
            {
                "member_id": ["F1:0", "F1:1", "F2:0"],
                "protein_accession": ["P1", "P2", "P3"],
                "description": ["one", "two", None],
            }
        ),
        key_columns=("member_id",),
        metadata={"producer": "test"},
    )
    parsed.feature_relations["protein_group_membership"] = FeatureRelation(
        annotation_table="protein_group_members",
        target_level="protein",
        coordinates=pl.DataFrame(
            {
                "row": [0, 1, 2],
                "column": [0, 0, 1],
                "value": [1.0, 1.0, 1.0],
            }
        ),
        metadata={"semantic": "member_of"},
    )
    return parsed


def _assert_result_equal(actual: ParsedLevels, expected: ParsedLevels) -> None:
    assert list(actual.levels) == list(expected.levels)
    assert actual.uns == expected.uns
    assert actual.metadata == expected.metadata
    assert list(actual.annotation_tables) == list(expected.annotation_tables)
    assert list(actual.feature_relations) == list(expected.feature_relations)
    for name, wanted in expected.annotation_tables.items():
        got = actual.annotation_tables[name]
        assert got.key_columns == wanted.key_columns
        assert got.metadata == wanted.metadata
        assert_frame_equal(got.frame, wanted.frame)
    for name, wanted in expected.feature_relations.items():
        got = actual.feature_relations[name]
        assert got.annotation_table == wanted.annotation_table
        assert got.target_level == wanted.target_level
        assert got.metadata == wanted.metadata
        assert_frame_equal(got.coordinates, wanted.coordinates)
    for name, wanted in expected.levels.items():
        got = actual.levels[name]
        assert got.primary_layer_name == wanted.primary_layer_name
        assert got.obs.key_columns == wanted.obs.key_columns
        assert got.var.key_columns == wanted.var.key_columns
        assert got.uns == wanted.uns
        assert got.metadata == wanted.metadata
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
        assert got.role == wanted.role
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
    ("result_format", "suffix"),
    [
        (ResultFormat.H5MU, ".h5mu"),
        (ResultFormat.PARQUET, ".parquet"),
        (ResultFormat.DUCKDB, ".duckdb"),
    ],
)
def test_annotation_tables_and_relations_round_trip(
    result_format: ResultFormat,
    suffix: str,
    tmp_path: Path,
) -> None:
    target = tmp_path / f"annotated{suffix}"

    writer_for(result_format).write(annotated_result(), target)
    restored = reader_for(result_format).read(target)

    if result_format is ResultFormat.H5MU:
        second = tmp_path / "annotated-again.h5mu"
        writer_for(result_format).write(restored, second)
        _assert_result_equal(reader_for(result_format).read(second), restored)
    else:
        _assert_result_equal(restored, annotated_result())


def test_h5ad_rejects_annotation_tables_before_touching_target(tmp_path: Path) -> None:
    parsed = annotated_result()
    parsed.levels = {"protein": parsed.levels["protein"]}
    target = tmp_path / "annotated.h5ad"

    with pytest.raises(InvalidResultError, match="cannot store annotation tables"):
        writer_for(ResultFormat.H5AD).write(parsed, target)

    assert not target.exists()


@pytest.mark.parametrize(
    ("result_format", "suffix"),
    [
        (ResultFormat.H5AD, ".h5ad"),
        (ResultFormat.H5MU, ".h5mu"),
        (ResultFormat.PARQUET, ".parquet"),
        (ResultFormat.DUCKDB, ".duckdb"),
    ],
)
def test_measurement_and_auxiliary_roles_round_trip_through_every_result_format(
    result_format: ResultFormat,
    suffix: str,
    tmp_path: Path,
) -> None:
    source = rich_result().levels["ion"]
    parsed = ParsedLevels(levels={"ion": source}, uns={"selected": "ion"})
    target = tmp_path / f"role{suffix}"

    writer_for(result_format).write(parsed, target)
    restored = reader_for(result_format).read(target).levels["ion"]

    assert isinstance(restored.layers["Intensity"].role, MeasurementLayerRole)
    assert isinstance(restored.layers["Status"].role, AuxiliaryLayerRole)


def test_parquet_metadata_without_a_layer_role_defaults_to_measurement(tmp_path: Path) -> None:
    target = tmp_path / "legacy.parquet"
    writer_for(ResultFormat.PARQUET).write(rich_result(), target)
    manifest_path = target / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["levels"]["ion"]["layers"]["Status"]["role"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = reader_for(ResultFormat.PARQUET).read(target)

    assert isinstance(restored.levels["ion"].layers["Status"].role, MeasurementLayerRole)


@pytest.mark.parametrize(
    ("persisted_role", "message"),
    [("unknown", "unknown role"), (None, "role is not text")],
)
def test_parquet_reader_rejects_an_explicit_invalid_layer_role(
    persisted_role: str | None,
    message: str,
    tmp_path: Path,
) -> None:
    target = tmp_path / "invalid-role.parquet"
    writer_for(ResultFormat.PARQUET).write(rich_result(), target)
    manifest_path = target / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["levels"]["ion"]["layers"]["Status"]["role"] = persisted_role
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InvalidResultError, match=message):
        reader_for(ResultFormat.PARQUET).read(target)


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

    assert projected.metadata == rich_result().metadata
    assert projected.levels["ion"].metadata == rich_result().levels["ion"].metadata

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
        metadata={"collection_extension": {"enabled": True}},
    )
    writer_for(ResultFormat.H5AD).write(single, target)
    projected = reader_for(ResultFormat.H5AD).read(target)
    assert projected.metadata == single.metadata
    assert projected.levels["ion"].metadata == single.levels["ion"].metadata
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


def test_a_layer_with_permuted_var_keys_is_rejected_before_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    parsed.levels["ion"].layers["Intensity"].values = (
        parsed.levels["ion"].layers["Intensity"].values.reverse()
    )
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="do not match var row-for-row"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_a_layer_with_a_different_var_key_is_rejected_before_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    layer = parsed.levels["ion"].layers["Intensity"]
    layer.values = layer.values.with_columns(pl.Series("Ion", ["different", "F2"]))
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="do not match var row-for-row"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_a_layer_whose_var_keys_are_not_leading_columns_is_rejected(tmp_path: Path) -> None:
    parsed = rich_result()
    layer = parsed.levels["protein"].layers["Intensity"]
    layer.values = layer.values.select("Isoform", "Protein", "obs_0", "obs_1")
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="must begin with var keys"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_a_duplicate_final_axis_key_is_rejected_before_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    parsed.levels["ion"].obs.frame = parsed.levels["ion"].obs.frame.with_columns(
        pl.Series("Run", ["same", "same"])
    )
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="obs contains a duplicate key"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_an_incomplete_final_axis_key_is_rejected_before_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    parsed.levels["ion"].var.frame = parsed.levels["ion"].var.frame.with_columns(
        pl.Series("Ion", ["F1", None])
    )
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="var contains an incomplete key"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


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


def test_quantitative_layer_projection_reuses_the_stored_vendor_encoding() -> None:
    ion = _level("ion", "Ion")

    projected = quantitative_layer_values(ion, "Intensity")

    assert projected.to_dict(as_series=False) == {
        "obs_0": [100.5, None],
        "obs_1": [200.5, None],
    }


def test_numeric_result_level_accepts_numeric_and_null_values_without_mutating_input() -> None:
    ion = _level("ion", "Ion")
    ion.layers = {
        "Intensity": FinalLayerTable(
            layer_name="Intensity",
            var_key_columns=("Ion",),
            values=pl.DataFrame(
                {
                    "Ion": ["F1", "F2"],
                    "obs_0": pl.Series([10, 20], dtype=pl.Int64),
                    "obs_1": pl.Series([None, None], dtype=pl.Null),
                }
            ),
        )
    }
    original_provenance = dict(ion.uns)

    projected = numeric_result_level(ion)

    assert ion.uns == original_provenance
    assert MATRIX_PROJECTED_KEY not in ion.uns
    assert projected is not ion
    assert projected.layers is ion.layers
    assert projected.uns is not ion.uns
    assert projected.uns[MATRIX_PROJECTED_KEY] is True


@pytest.mark.parametrize(
    "invalid_values",
    [
        pl.Series("obs_0", [True, False], dtype=pl.Boolean),
        pl.Series("obs_0", ["10", "20"], dtype=pl.String),
    ],
    ids=["boolean", "string"],
)
def test_numeric_result_level_rejects_nonnumeric_values(invalid_values: pl.Series) -> None:
    ion = _level("ion", "Ion")
    ion.layers = {
        "Intensity": FinalLayerTable(
            layer_name="Intensity",
            var_key_columns=("Ion",),
            values=pl.DataFrame(
                {
                    "Ion": ["F1", "F2"],
                    "obs_0": invalid_values,
                    "obs_1": [10.0, 20.0],
                }
            ),
        )
    }

    with pytest.raises(AnnDataPlanError, match=r"not already numeric.*obs_0"):
        numeric_result_level(ion)


def test_h5_writer_accepts_an_added_numeric_layer_missing_from_the_parse_plan(
    tmp_path: Path,
) -> None:
    parsed = rich_result()
    ion = parsed.levels["ion"]
    ion.layers["medpolish_from_fragment"] = FinalLayerTable(
        layer_name="medpolish_from_fragment",
        var_key_columns=ion.var.key_columns,
        values=pl.DataFrame(
            {
                "Ion": ["F1", "F2"],
                "obs_0": [10.0, 20.0],
                "obs_1": [11.0, None],
            }
        ),
    )
    target = tmp_path / "result.h5mu"

    write_parsed_levels(parsed, target)

    restored = read_parsed_levels(target).levels["ion"]
    np.testing.assert_allclose(
        restored.layers["medpolish_from_fragment"].values.select("obs_0", "obs_1").to_numpy(),
        np.array([[10.0, 11.0], [20.0, np.nan]]),
        equal_nan=True,
    )


def test_h5_writer_accepts_a_planless_matrix_projected_derived_level(tmp_path: Path) -> None:
    protein = _level("protein", "Protein")
    protein.var = VarFinal(
        frame=pl.DataFrame({"Protein": ["F1", "F2"]}),
        key_columns=("Protein",),
    )
    protein.uns = {
        "produced_by": "apb-aggregate",
        "quantification_level": "protein",
        MATRIX_PROJECTED_KEY: True,
    }
    protein.primary_layer_name = "medpolish_from_ion"
    protein.layers = {
        "medpolish_from_ion": FinalLayerTable(
            layer_name="medpolish_from_ion",
            var_key_columns=protein.var.key_columns,
            values=pl.DataFrame(
                {
                    "Protein": ["F1", "F2"],
                    "obs_0": [10.0, 20.0],
                    "obs_1": [11.0, None],
                }
            ),
        )
    }
    target = tmp_path / "protein.h5ad"

    write_parsed_levels(ParsedLevels(levels={"protein": protein}, uns={}), target)

    restored = read_parsed_levels(target).levels["protein"]
    assert restored.primary_layer_name == "medpolish_from_ion"
    assert restored.layers["medpolish_from_ion"].values.height == 2


def test_h5_writer_rejects_an_unplanned_nonnumeric_layer(tmp_path: Path) -> None:
    parsed = rich_result()
    ion = parsed.levels["ion"]
    ion.layers["derived_text"] = FinalLayerTable(
        layer_name="derived_text",
        var_key_columns=ion.var.key_columns,
        values=pl.DataFrame(
            {
                "Ion": ["F1", "F2"],
                "obs_0": ["one", "two"],
                "obs_1": ["three", "four"],
            }
        ),
    )

    with pytest.raises(AnnDataPlanError, match=r"unplanned layer.*not already numeric"):
        write_parsed_levels(parsed, tmp_path / "result.h5mu")


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
