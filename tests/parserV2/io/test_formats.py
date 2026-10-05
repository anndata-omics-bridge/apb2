"""Result I/O laws: exact columnar round-trips and canonical AnnData projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import anndata
import duckdb
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apb2.cli.app import reformat as reformat_command
from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FeatureRelation,
    FinalLayerTable,
    JsonValue,
    LevelHierarchy,
    MeasurementLayerRole,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    QuantitativeLayerSemantics,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.duckdb import METADATA_TABLE
from apb2.parserV2.parse_quant.io.errors import InvalidResultError, UnsupportedResultFormatError
from apb2.parserV2.parse_quant.io.formats import (
    ParsedLevelsReader,
    ParsedLevelsWriter,
    ResultFormat,
    read_parsed_levels,
    reader_for,
    result_format_for,
    write_parsed_levels,
    writer_for,
)
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
            "canonicalization": {
                "layer_values": encodings,
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
            values=(
                pl.DataFrame(
                    {
                        **layer_keys,
                        "obs_0": [100.5, None],
                        "obs_1": [200.5, None],
                    }
                )
            ).drop(var_key_columns, strict=False),
            semantics=QuantitativeLayerSemantics(),
            semantic_roles=("abundance",),
        ),
        "Status": FinalLayerTable(
            layer_name="Status",
            values=(
                pl.DataFrame(
                    {
                        **layer_keys,
                        "obs_0": [1, 2],
                        "obs_1": [2, -1],
                    }
                )
            ).drop(var_key_columns, strict=False),
            role=AuxiliaryLayerRole(),
            semantics=CategoricalLayerSemantics(
                categories=(("MS/MS", 1), ("MBR", 2)),
                missing_code=-1,
            ),
        ),
        "Count": FinalLayerTable(
            layer_name="Count",
            values=(
                pl.DataFrame(
                    {
                        **layer_keys,
                        "obs_0": pl.Series([1, None], dtype=pl.Int64),
                        "obs_1": pl.Series([2, 3], dtype=pl.Int64),
                    }
                )
            ).drop(var_key_columns, strict=False),
            role=AuxiliaryLayerRole(),
            semantics=QuantitativeLayerSemantics(logical_type="integer"),
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
            "hierarchy": "lfq",
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
        assert got.semantic_roles == wanted.semantic_roles
        assert type(got.role) is type(wanted.role)
        assert got.semantics == wanted.semantics
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


def test_parquet_metadata_without_a_layer_role_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "invalid.parquet"
    writer_for(ResultFormat.PARQUET).write(rich_result(), target)
    manifest_path = target / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["levels"]["ion"]["layers"]["Status"]["role"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InvalidResultError, match="role"):
        reader_for(ResultFormat.PARQUET).read(target)


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

    write_parsed_levels(read_parsed_levels(source), target)

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

    write_parsed_levels(read_parsed_levels(source), crossed_path)

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

    write_parsed_levels(read_parsed_levels(source), target)

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

    write_parsed_levels(read_parsed_levels(columnar), h5ad)
    projection = reader_for(ResultFormat.H5AD).read(h5ad)
    write_parsed_levels(read_parsed_levels(h5ad), restored)

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


def test_layer_rows_are_positional_after_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    layer = parsed.levels["ion"].layers["Intensity"]
    layer.values = layer.values.reverse()
    target = tmp_path / "result.duckdb"
    write_parsed_levels(parsed, target)
    assert_frame_equal(
        read_parsed_levels(target).levels["ion"].layers["Intensity"].values, layer.values
    )


def test_a_layer_with_a_different_var_key_is_rejected_before_writing(tmp_path: Path) -> None:
    parsed = rich_result()
    layer = parsed.levels["ion"].layers["Intensity"]
    layer.values = layer.values.with_columns(pl.Series("Ion", ["different", "F2"]))
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="observation columns"):
        writer_for(ResultFormat.DUCKDB).write(parsed, target)

    assert not target.exists()


def test_a_layer_whose_var_keys_are_not_leading_columns_is_rejected(tmp_path: Path) -> None:
    parsed = rich_result()
    layer = parsed.levels["protein"].layers["Intensity"]
    layer.values = layer.values.with_columns(pl.lit("unexpected").alias("Protein"))
    target = tmp_path / "result.duckdb"

    with pytest.raises(InvalidResultError, match="observation columns"):
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


def test_h5_writer_ignores_missing_and_corrupt_plan_json(tmp_path: Path) -> None:
    baseline = rich_result()
    baseline_target = tmp_path / "baseline.h5mu"
    writer_for(ResultFormat.H5MU).write(baseline, baseline_target)
    expected = reader_for(ResultFormat.H5MU).read(baseline_target)

    parsed = rich_result()
    del parsed.levels["protein"].uns["plan_json"]
    parsed.levels["ion"].uns["plan_json"] = "not json"
    target = tmp_path / "result.h5mu"

    writer_for(ResultFormat.H5MU).write(parsed, target)

    restored = reader_for(ResultFormat.H5MU).read(target)
    expected.levels["protein"].uns.pop("plan_json")
    expected.levels["ion"].uns["plan_json"] = "not json"
    _assert_result_equal(restored, expected)


def test_quantitative_values_returns_canonical_values_directly() -> None:
    ion = _level("ion", "Ion")

    projected = ion.layers["Intensity"].quantitative_values()

    assert projected.to_dict(as_series=False) == {
        "obs_0": [100.5, None],
        "obs_1": [200.5, None],
    }


def test_h5_writer_accepts_an_added_numeric_layer_missing_from_the_parse_plan(
    tmp_path: Path,
) -> None:
    parsed = rich_result()
    ion = parsed.levels["ion"]
    ion.layers["medpolish_from_fragment"] = FinalLayerTable(
        layer_name="medpolish_from_fragment",
        values=(
            pl.DataFrame(
                {
                    "Ion": ["F1", "F2"],
                    "obs_0": [10.0, 20.0],
                    "obs_1": [11.0, None],
                }
            )
        ).drop(ion.var.key_columns, strict=False),
        semantic_roles=("abundance",),
    )
    target = tmp_path / "result.h5mu"

    write_parsed_levels(parsed, target)

    restored = read_parsed_levels(target).levels["ion"]
    np.testing.assert_allclose(
        restored.layers["medpolish_from_fragment"].values.select("obs_0", "obs_1").to_numpy(),
        np.array([[10.0, 11.0], [20.0, np.nan]]),
        equal_nan=True,
    )


def test_h5_writer_accepts_a_planless_derived_level(tmp_path: Path) -> None:
    protein = _level("protein", "Protein")
    protein.var = VarFinal(
        frame=pl.DataFrame({"Protein": ["F1", "F2"]}),
        key_columns=("Protein",),
    )
    protein.uns = {
        "hierarchy": "lfq",
        "produced_by": "apb-aggregate",
        "quantification_level": "protein",
    }
    protein.primary_layer_name = "medpolish_from_ion"
    protein.layers = {
        "medpolish_from_ion": FinalLayerTable(
            layer_name="medpolish_from_ion",
            values=(
                pl.DataFrame(
                    {
                        "Protein": ["F1", "F2"],
                        "obs_0": [10.0, 20.0],
                        "obs_1": [11.0, None],
                    }
                )
            ).drop(protein.var.key_columns, strict=False),
            semantic_roles=("abundance",),
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
        values=(
            pl.DataFrame(
                {
                    "Ion": ["F1", "F2"],
                    "obs_0": ["one", "two"],
                    "obs_1": ["three", "four"],
                }
            )
        ).drop(ion.var.key_columns, strict=False),
        semantic_roles=("abundance",),
    )

    with pytest.raises(InvalidResultError, match=r"not numeric"):
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


@pytest.mark.parametrize("suffix", [".h5ad", ".h5mu", ".parquet", ".duckdb"])
def test_tool_namespaces_preserve_overlapping_ownership_and_empty_objects(
    tmp_path: Path, suffix: str
) -> None:
    # Start from the canonical numeric projection, so all four formats share values/dtypes.
    initial = tmp_path / "initial.h5ad"
    write_parsed_levels(ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={}), initial)
    level = read_parsed_levels(initial).levels["ion"]
    level.layers["Intensity"].semantic_roles = ("abundance",)
    level.metadata = {
        "tool": {"annotation": {"count": 2}, "scoring": {"Intensity": {"score": 0.5}}},
        "extension": {"empty": {}, "level": None, "items": [], "nested": {"level": {}}},
    }
    root: dict[str, JsonValue] = {
        "tool": {"provenance": {"annotation": {"source": "input.tsv"}}},
        "extension": {"empty": {"root": 1}, "z": {}, "a": 2, "nested": {}},
    }
    parsed = ParsedLevels(levels={"ion": level}, uns={"produced_by": "test"}, metadata=root)
    target = tmp_path / f"result{suffix}"
    write_parsed_levels(parsed, target)
    restored = read_parsed_levels(target)
    _assert_result_equal(restored, parsed)
    representation = json.loads(Path(f"{target}.apb.json").read_text())
    assert representation["format_version"] == "4"
    assert "shared" not in representation
    assert "storage" not in representation["levels"][0]["apb"]
    if suffix == ".h5ad":
        stored = anndata.read_h5ad(target)
        apb = stored.uns["apb"]
        assert not {"shared", "level"}.intersection(apb)
        assert set(apb["tool"]) == {"provenance", "annotation", "scoring"}
        assert stored.X is not None and "Intensity" not in stored.layers
        assert representation["root"] is None
        assert set(representation["levels"][0]["apb"]["tool"]) == set(apb["tool"])
        ownership = json.loads(apb["storage"])["metadata_ownership"]
        assert "input.tsv" not in json.dumps(ownership), "ownership stores paths, not values"
    else:
        assert representation["root"]["apb"]["tool"] == root["tool"]
        assert "provenance" not in representation["levels"][0]["apb"]["tool"]


@pytest.mark.parametrize("suffix", [".h5ad", ".h5mu", ".parquet", ".duckdb"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"levels": [{"level": "ion", "fingerprint": None}]}, id="record-list"),
        pytest.param([{}], id="empty-record"),
        pytest.param(["a", None], id="text-with-null"),
        pytest.param([[1, 2], [3]], id="ragged-lists"),
        pytest.param([1, "a"], id="integer-and-text"),
        pytest.param([1, 2.5], id="integer-and-float"),
        pytest.param([True, 1], id="boolean-and-integer"),
        pytest.param(2**70, id="integer-beyond-64-bits"),
        pytest.param([2**70, 1], id="integers-beyond-64-bits"),
        pytest.param("a\x00b", id="text-with-nul"),
        pytest.param({"LFQ/Intensity": 1, ".": 2, "": 3, "a\x00b": 4}, id="unlinkable-keys"),
        pytest.param('[1, "a"]', id="json-looking-text"),
        pytest.param(["a", "b"], id="native-text-list"),
    ],
)
def test_extension_json_values_round_trip_through_every_result_format(
    tmp_path: Path, suffix: str, value: JsonValue
) -> None:
    level = _level("ion", "Ion")
    level.metadata = {"tool": {"local": value}}
    parsed = ParsedLevels(levels={"ion": level}, uns={}, metadata={"tool": {"shared": value}})
    target = tmp_path / f"result{suffix}"
    write_parsed_levels(parsed, target)
    restored = read_parsed_levels(target)
    # JSON text distinguishes 1 from 1.0 and True, which list equality does not.
    assert json.dumps(restored.metadata, sort_keys=True) == json.dumps(
        parsed.metadata, sort_keys=True
    )
    assert json.dumps(restored.levels["ion"].metadata, sort_keys=True) == json.dumps(
        level.metadata, sort_keys=True
    )


def test_h5ad_stores_only_unstorable_values_as_recorded_json_text(tmp_path: Path) -> None:
    parsed = ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={})
    parsed.metadata["catalog"] = {"levels": [{"level": "ion"}], "names": ["a", "b"]}
    target = tmp_path / "result.h5ad"
    write_parsed_levels(parsed, target)
    apb = anndata.read_h5ad(target).uns["apb"]
    assert json.loads(apb["catalog"]["levels"]) == [{"level": "ion"}]
    assert isinstance(apb["catalog"]["names"], np.ndarray)
    assert json.loads(apb["storage"])["json_values"] == [["catalog", "levels"]]


@pytest.mark.parametrize(
    ("json_values", "message"),
    [
        ([["catalog", "levels"], ["catalog", "levels"]], "does not name stored text"),
        ([["catalog", "levels"], ["catalog", "names"]], "does not name stored text"),
        ([["catalog", "levels"], ["catalog", "absent"]], "does not name stored text"),
        ([["catalog", "levels"], ["catalog", "text"]], "invalid JSON value"),
        ([["catalog", "levels"], ["catalog", "number"]], "do not describe"),
    ],
)
def test_h5_reader_rejects_json_value_paths_that_do_not_match_the_metadata(
    tmp_path: Path, json_values: list[list[str]], message: str
) -> None:
    parsed = ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={})
    parsed.metadata["catalog"] = {
        "levels": [{"level": "ion"}],
        "names": ["a", "b"],
        "text": "plain",
        "number": "1",
    }
    target = tmp_path / "result.h5ad"
    write_parsed_levels(parsed, target)
    stored = anndata.read_h5ad(target)
    descriptor = json.loads(stored.uns["apb"]["storage"])
    descriptor["json_values"] = json_values
    stored.uns["apb"]["storage"] = json.dumps(descriptor)
    stored.write_h5ad(target)
    with pytest.raises(InvalidResultError, match=message):
        read_parsed_levels(target)


def test_h5_writer_rejects_a_section_name_hdf5_cannot_link(tmp_path: Path) -> None:
    parsed = ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={}, metadata={"a/b": 1})
    target = tmp_path / "result.h5ad"
    with pytest.raises(InvalidResultError, match="cannot name an HDF5 group"):
        write_parsed_levels(parsed, target)
    assert not target.exists()


def test_single_level_export_from_h5mu_retains_root_provenance(tmp_path: Path) -> None:
    source = tmp_path / "collection.h5mu"
    target = tmp_path / "ion.h5ad"
    parsed = rich_result()
    parsed.metadata["tool"] = {"provenance": {"source": "input.tsv"}}
    parsed.levels["ion"].metadata["tool"] = {"annotation": {"count": 2}}
    write_parsed_levels(parsed, source)
    selected = read_parsed_levels(source)
    selected.levels = {"ion": selected.levels["ion"]}
    write_parsed_levels(selected, target)
    _assert_result_equal(read_parsed_levels(target), selected)


def test_conflicting_h5ad_metadata_is_rejected_before_publication(tmp_path: Path) -> None:
    parsed = ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={})
    parsed.metadata["tool"] = {"value": 1}
    parsed.levels["ion"].metadata["tool"] = {"value": 1}
    target = tmp_path / "conflict.h5ad"
    with pytest.raises(InvalidResultError, match="conflicting APB metadata"):
        write_parsed_levels(parsed, target)
    assert not target.exists()
    assert not Path(f"{target}.apb.json").exists()


def test_previous_hdf_metadata_layout_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "old.h5ad"
    write_parsed_levels(ParsedLevels(levels={"ion": _level("ion", "Ion")}, uns={}), target)
    stored = anndata.read_h5ad(target)
    descriptor = json.loads(stored.uns["apb"]["storage"])
    descriptor["format_version"] = "2"
    stored.uns["apb"]["storage"] = json.dumps(descriptor)
    stored.write_h5ad(target)
    with pytest.raises(InvalidResultError, match="version"):
        read_parsed_levels(target)


@pytest.mark.parametrize("suffix", [".h5mu", ".parquet", ".duckdb"])
def test_open_hierarchy_and_typed_roles_round_trip(suffix: str, tmp_path: Path) -> None:
    level = _level("site", "site_id")
    level.var.roles = {"site_identity": "site_id"}
    parsed = ParsedLevels(
        levels={"site": level},
        uns={},
        hierarchy=LevelHierarchy(
            "custom_enrichment", (("peptidoform", "form_id"), ("site", "site_identity"))
        ),
    )
    path = tmp_path / f"sites{suffix}"
    write_parsed_levels(parsed, path)
    restored = read_parsed_levels(path)
    assert restored.hierarchy == parsed.hierarchy
    assert restored.levels["site"].var.roles == level.var.roles
    assert restored.levels["site"].layers["Intensity"].semantic_roles == ("abundance",)
    assert restored.levels["site"].layers["Intensity"].values.width == level.obs.frame.height
    assert "hierarchy" not in restored.metadata
    assert "column_roles" not in restored.levels["site"].uns


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Categorical])
def test_var_role_requires_string_values(
    dtype: pl.DataType | type[pl.DataType], tmp_path: Path
) -> None:
    parsed = rich_result()
    level = parsed.levels["ion"]
    level.var.frame = level.var.frame.with_columns(
        pl.Series("assignment", [1, 2]).cast(pl.String).cast(dtype)
    )
    level.var.roles = {"protein_assignment": "assignment"}
    with pytest.raises(InvalidResultError, match="requires String"):
        write_parsed_levels(parsed, tmp_path / "invalid.parquet")


def test_result_requires_same_observation_values_and_order(tmp_path: Path) -> None:
    parsed = rich_result()
    parsed.levels["protein"].obs.frame = parsed.levels["protein"].obs.frame.reverse()
    with pytest.raises(InvalidResultError, match="share the observation axis"):
        write_parsed_levels(parsed, tmp_path / "invalid.h5mu")
