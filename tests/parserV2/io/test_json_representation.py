"""The compact APB JSON representation exposes structure and summaries, never matrices."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FeatureRelation,
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    QuantitativeLayerSemantics,
    VarFinal,
)
from apb2.parserV2.parse_quant.io import formats, json_representation
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels
from apb2.parserV2.parse_quant.io.json_representation import (
    FORMAT,
    FORMAT_VERSION,
    project_result,
    sidecar_path,
    write_result_representation,
)
from apb2.parserV2.parse_quant.io.layer_representation import quantitative_representation


def _parsed() -> ParsedLevels:
    level = ParsedLevel(
        obs=ObsFinal(
            frame=pl.DataFrame(
                {
                    "sample": ["sample A", "sample B"],
                    "batch": ["one", None],
                }
            ),
            key_columns=("sample",),
        ),
        var=VarFinal(
            frame=pl.DataFrame(
                {
                    "feature": ["F1", "F2", "F3", "F4"],
                    "score": [1.0, None, 3.0, 4.0],
                }
            ),
            key_columns=("feature",),
        ),
        primary_layer_name="Intensity",
        layers={
            "Intensity": FinalLayerTable(
                layer_name="Intensity",
                var_key_columns=("feature",),
                values=pl.DataFrame(
                    {
                        "feature": ["F1", "F2", "F3", "F4"],
                        "obs_0": [1.0, None, float("nan"), float("inf")],
                        "obs_1": [0.0, 2.0, 3.0, float("-inf")],
                    }
                ),
            ),
            "Count": FinalLayerTable(
                layer_name="Count",
                var_key_columns=("feature",),
                values=pl.DataFrame(
                    {
                        "feature": ["F1", "F2", "F3", "F4"],
                        "obs_0": [1, 2, 3, 4],
                        "obs_1": [4, 3, 2, 1],
                    }
                ),
                role=AuxiliaryLayerRole(),
                semantics=QuantitativeLayerSemantics(logical_type="integer"),
            ),
        },
        obsm={"design": pl.DataFrame({"group": ["x", "y"]})},
        varm={"loadings": pl.DataFrame({"loading": [0.1, 0.2, 0.3, 0.4]})},
        obsp={"neighbors": pl.DataFrame({"row": [0], "column": [1], "value": [0.5]})},
        varp={},
        uns={
            "quantification_level": "ion",
            "source_path": "/private/input/vendor.tsv",
        },
        metadata={
            "layer_descriptors": {"Intensity": {"unit": "arbitrary units", "scale": "linear"}}
        },
    )
    return ParsedLevels(
        levels={"ion": level},
        uns={"produced_by": "apb2"},
        annotation_tables={
            "proteins": AnnotationTable(
                frame=pl.DataFrame({"id": ["P1", "P2"], "description": ["one", None]}),
                key_columns=("id",),
                metadata={"kind": "lookup"},
            )
        },
        feature_relations={
            "membership": FeatureRelation(
                annotation_table="proteins",
                target_level="ion",
                coordinates=pl.DataFrame({"row": [0], "column": [1], "value": [1.0]}),
                metadata={"relation": "member_of"},
            )
        },
    )


def _empty_level() -> ParsedLevel:
    return ParsedLevel(
        obs=ObsFinal(
            frame=pl.DataFrame(schema={"sample": pl.String}),
            key_columns=("sample",),
        ),
        var=VarFinal(
            frame=pl.DataFrame(schema={"protein": pl.String}),
            key_columns=("protein",),
        ),
        primary_layer_name="Intensity",
        layers={
            "Intensity": FinalLayerTable(
                layer_name="Intensity",
                var_key_columns=("protein",),
                values=pl.DataFrame(schema={"protein": pl.String}),
            )
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
        uns={},
    )


def _factor_result() -> ParsedLevels:
    parsed = _parsed()
    level = parsed.levels["ion"]
    level.uns["plan_json"] = _factor_plan()
    level.layers["Status"] = FinalLayerTable(
        layer_name="Status",
        var_key_columns=("feature",),
        values=pl.DataFrame(
            {
                "feature": ["F1", "F2", "F3", "F4"],
                "obs_0": [1, 2, -1, -1],
                "obs_1": [2, 1, 1, -1],
            }
        ),
        role=AuxiliaryLayerRole(),
        semantics=CategoricalLayerSemantics(
            categories=(("MS/MS", 1), ("MBR", 2)),
            missing_code=-1,
        ),
    )
    return ParsedLevels(levels={"ion": level}, uns={})


def test_version_four_representation_is_structured_bounded_and_deterministic(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "result.h5mu"
    artifact.write_bytes(b"scientific result")

    document: dict[str, Any] = project_result(_parsed(), artifact)

    assert document["format"] == FORMAT
    assert document["format_version"] == FORMAT_VERSION
    assert document["artifact"] == {
        "name": "result.h5mu",
        "physical_format": "h5mu",
        "size_bytes": 17,
    }
    level = document["levels"][0]
    assert level["name"] == "ion"
    assert level["dimensions"] == {"observations": 2, "variables": 4}
    assert level["observations"] == {
        "total_count": 2,
        "emitted_count": 2,
        "truncated": False,
        "items": [
            {"index": 0, "key": {"sample": "sample A"}, "label": "sample A"},
            {"index": 1, "key": {"sample": "sample B"}, "label": "sample B"},
        ],
    }
    assert level["obs"] == {
        "row_count": 2,
        "key_columns": ["sample"],
        "columns": [
            {"name": "sample", "dtype": "String", "null_count": 0},
            {"name": "batch", "dtype": "String", "null_count": 1},
        ],
    }
    intensity = level["layers"][0]
    assert intensity["name"] == "Intensity"
    assert intensity["value_kind"] == "quantitative"
    assert intensity["type"] == "number"
    assert intensity["unit"] == "arbitrary units"
    assert intensity["scale"] == "linear"
    assert intensity["statistics"] == {
        "total_count": 8,
        "finite_count": 4,
        "null_count": 1,
        "nan_count": 1,
        "positive_infinity_count": 1,
        "negative_infinity_count": 1,
        "zero_count": 1,
        "mean": 1.5,
        "standard_deviation": pytest.approx(1.2909944487358056),
        "minimum": 0.0,
        "first_quartile": 0.75,
        "median": 1.5,
        "third_quartile": 2.25,
        "maximum": 3.0,
        "quartile_interpolation": "linear",
        "quartile_method": "linear_exact",
        "quartile_sample_count": 4,
        "quartile_sample_limit": 100_000,
    }
    summaries = intensity["observation_summaries"]
    assert summaries["total_count"] == 2
    assert summaries["emitted_count"] == 2
    assert summaries["truncated"] is False
    first_sample = summaries["items"][0]
    assert first_sample["observation_index"] == 0
    assert "key" not in first_sample
    assert "label" not in first_sample
    assert first_sample["statistics"]["finite_count"] == 1
    assert first_sample["statistics"]["standard_deviation"] is None
    assert level["aligned"]["obsm"][0]["columns"] == [
        {"name": "group", "dtype": "String", "null_count": 0}
    ]
    assert document["annotation_tables"][0]["row_count"] == 2
    assert document["feature_relations"][0]["coordinates"]["row_count"] == 1

    serialized = json.dumps(document, allow_nan=False)
    assert "vendor.tsv" in serialized
    assert "/private/input" not in serialized
    assert serialized.count("sample A") == 2  # One key value and one display label.
    for forbidden in ("F1", "P1", "0.1"):
        assert forbidden not in serialized


def test_multiple_levels_and_empty_axes_have_a_complete_json_shape(tmp_path: Path) -> None:
    artifact = tmp_path / "result.h5mu"
    artifact.write_bytes(b"collection")
    parsed = _parsed()
    parsed.levels["protein"] = _empty_level()
    parsed.uns["nonfinite_extension"] = float("nan")

    document: dict[str, Any] = project_result(parsed, artifact)

    assert [level["name"] for level in document["levels"]] == ["ion", "protein"]
    empty = document["levels"][1]
    assert empty["dimensions"] == {"observations": 0, "variables": 0}
    assert empty["observations"] == {
        "total_count": 0,
        "emitted_count": 0,
        "truncated": False,
        "items": [],
    }
    assert empty["layers"][0]["observation_summaries"]["items"] == []
    assert empty["layers"][0]["statistics"]["finite_count"] == 0
    assert empty["layers"][0]["statistics"]["median"] is None
    assert document["root"]["apb"]["parse"]["nonfinite_extension"] is None
    json.dumps(document, allow_nan=False)


def test_levels_follow_the_same_canonical_order_as_physical_mudata() -> None:
    parsed = _parsed()
    ion = parsed.levels["ion"]
    parsed.levels = {"protein": _empty_level(), "ion": ion}

    document: dict[str, Any] = project_result(parsed)

    assert [level["name"] for level in document["levels"]] == ["ion", "protein"]


def test_enum_axis_categories_and_nonfinite_descriptors_do_not_leak_into_json(
    tmp_path: Path,
) -> None:
    parsed = _parsed()
    level = parsed.levels["ion"]
    level.obs.frame = level.obs.frame.with_columns(
        pl.Series("state", ["private-a", "private-b"]).cast(pl.Enum(["private-a", "private-b"]))
    )
    level.metadata["layer_descriptors"] = {
        "Intensity": {"unit": "arbitrary units", "scale": float("nan")}
    }
    artifact = tmp_path / "result.h5mu"
    artifact.write_bytes(b"result")

    document: dict[str, Any] = project_result(parsed, artifact)
    serialized = json.dumps(document, allow_nan=False)

    state = next(
        column for column in document["levels"][0]["obs"]["columns"] if column["name"] == "state"
    )
    assert state["dtype"] == "Enum"
    assert "private-a" not in serialized
    assert document["levels"][0]["layers"][0]["scale"] is None


def test_known_embedded_json_containers_are_projected_recursively_without_mutation() -> None:
    parsed = _parsed()
    rule_text = json.dumps(
        {
            "schema_version": "0.3",
            "software_name": "Sage",
            "separator": "/",
            "value_pattern": "/foo/bar",
            "resource_paths": ["/private/rules/a.json", r"C:\private\rules\b.json"],
        }
    )
    plan_text = json.dumps(
        {
            "level": "ion",
            "canonicalization": {
                "layer_values": [
                    {
                        "kind": "plain_numeric",
                        "layer_name": name,
                        "missing_values": [],
                        "number_format": {"decimal_mark": ".", "thousands_marks": []},
                    }
                    for name in ("Intensity", "Count")
                ],
                "layer_contract": {
                    "primary_layer_name": "Intensity",
                    "required_names": ["Intensity"],
                    "empty_ratio": 0.001,
                    "populated_ratio": 0.5,
                },
            },
            "provenance": {"rule_json": rule_text},
        }
    )
    aggregate_text = json.dumps(
        [{"source_level": "ion", "target_level": "protein", "method": "mean"}]
    )
    level = parsed.levels["ion"]
    level.uns.update({"rule_json": rule_text, "plan_json": plan_text})
    level.metadata["aggregate"] = aggregate_text
    parsed.annotation_tables["proteins"].metadata.update(
        {
            "rule_json": "{not valid JSON",
            "plan_json": "42",
            "note": '{"looks":"like JSON"}',
        }
    )

    document: dict[str, Any] = project_result(parsed)

    projected_level = document["levels"][0]
    assert projected_level["apb"]["parse"]["rule_json"] == {
        "schema_version": "0.3",
        "software_name": "Sage",
        "separator": "/",
        "value_pattern": "/foo/bar",
        "resource_paths": ["a.json", "b.json"],
    }
    assert projected_level["apb"]["parse"]["plan_json"] == {
        "level": "ion",
        "canonicalization": {
            "layer_values": [
                {
                    "kind": "plain_numeric",
                    "layer_name": name,
                    "missing_values": [],
                    "number_format": {"decimal_mark": ".", "thousands_marks": []},
                }
                for name in ("Intensity", "Count")
            ],
            "layer_contract": {
                "primary_layer_name": "Intensity",
                "required_names": ["Intensity"],
                "empty_ratio": 0.001,
                "populated_ratio": 0.5,
            },
        },
        "provenance": {
            "rule_json": {
                "schema_version": "0.3",
                "software_name": "Sage",
                "separator": "/",
                "value_pattern": "/foo/bar",
                "resource_paths": ["a.json", "b.json"],
            }
        },
    }
    assert document["root"]["apb"]["parse"] == {"produced_by": "apb2"}
    assert projected_level["apb"]["aggregate"] == [
        {"source_level": "ion", "target_level": "protein", "method": "mean"}
    ]
    projected_annotation = document["annotation_tables"][0]["metadata"]
    assert projected_annotation["rule_json"] == "{not valid JSON"
    assert projected_annotation["plan_json"] == "42"
    assert projected_annotation["note"] == '{"looks":"like JSON"}'
    assert level.uns["rule_json"] == rule_text
    assert level.uns["plan_json"] == plan_text
    assert level.metadata["aggregate"] == aggregate_text


def test_sidecar_publication_replaces_atomically_and_preserves_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "result.h5mu"
    artifact.write_bytes(b"valid artifact")
    destination = sidecar_path(artifact)
    destination.write_text("previous\n", encoding="utf-8")

    written = write_result_representation(_parsed(), artifact)

    assert written == destination
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["artifact"]["name"] == artifact.name
    assert artifact.read_bytes() == b"valid artifact"
    assert not list(tmp_path.glob("*.tmp"))


def test_public_result_writer_always_emits_the_adjacent_sidecar(tmp_path: Path) -> None:
    target = tmp_path / "result.parquet"

    write_parsed_levels(_parsed(), target)

    document = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert document["artifact"]["physical_format"] == "parquet"
    assert document["artifact"]["size_bytes"] > 0


def test_sidecar_failure_is_visible_after_the_scientific_artifact_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.parquet"

    previous = sidecar_path(target)
    previous.write_text("stale representation\n", encoding="utf-8")

    def fail_projection(
        _parsed: ParsedLevels,
        _target: Path | None = None,
        /,
    ) -> dict[str, Any]:
        raise OSError("sidecar publication failed")

    monkeypatch.setattr(json_representation, "project_result", fail_projection)

    with pytest.raises(OSError, match="sidecar publication failed"):
        formats.write_parsed_levels(_parsed(), target)

    assert target.is_dir()
    assert not previous.exists()


def test_sidecar_publish_failure_removes_stale_file_and_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "result.h5mu"
    artifact.write_bytes(b"new artifact")
    destination = sidecar_path(artifact)
    destination.write_text("stale representation\n", encoding="utf-8")

    def fail_replace(_source: str | bytes, _target: str | bytes, /) -> None:
        raise OSError("atomic replacement failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="atomic replacement failed"):
        write_result_representation(_parsed(), artifact)

    assert artifact.read_bytes() == b"new artifact"
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_in_memory_projection_does_not_require_an_artifact(tmp_path: Path) -> None:
    del tmp_path

    document = project_result(_parsed())

    assert document["artifact"] is None


def test_factor_layer_has_bounded_categorical_counts_and_no_numeric_summary(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "factor.parquet"
    artifact.mkdir()
    parsed = _factor_result()

    document: dict[str, Any] = project_result(parsed, artifact)
    status = document["levels"][0]["layers"][2]

    assert status == {
        "name": "Status",
        "role": "auxiliary",
        "primary": False,
        "storage_slot": "layers",
        "shape": {"observations": 2, "variables": 4},
        "unit": None,
        "scale": None,
        "value_kind": "categorical",
        "dtype": "categorical",
        "category_count": 2,
        "counts": {
            "total_count": 8,
            "known_count": 5,
            "missing_or_unknown_count": 3,
        },
    }
    assert "statistics" not in status
    assert "observation_summaries" not in status


def test_quantitative_sidecar_preserves_declared_integer_semantics(tmp_path: Path) -> None:
    artifact = tmp_path / "integer.parquet"
    artifact.mkdir()

    document: dict[str, Any] = project_result(_factor_result(), artifact)
    count = document["levels"][0]["layers"][1]

    assert count["name"] == "Count"
    assert count["type"] == "integer"
    assert count["value_kind"] == "quantitative"


def test_corrupt_stored_plan_changes_no_layer_semantics(tmp_path: Path) -> None:
    parsed = _factor_result()
    parsed.levels["ion"].uns["plan_json"] = "not json"
    artifact = tmp_path / "corrupt-plan.parquet"
    artifact.mkdir()

    document: dict[str, Any] = project_result(parsed, artifact)

    assert [layer["type"] for layer in document["levels"][0]["layers"][:2]] == [
        "number",
        "integer",
    ]


def test_factor_missing_code_is_reserved(tmp_path: Path) -> None:
    del tmp_path

    with pytest.raises(ValueError, match="missing code -1 is reserved"):
        CategoricalLayerSemantics(categories=(("invalid", -1),), missing_code=-1)


def test_factor_semantics_survive_h5_read_projection_and_rewrite(tmp_path: Path) -> None:
    first = tmp_path / "factor.h5ad"
    write_parsed_levels(_factor_result(), first)
    restored = read_parsed_levels(first)

    document: dict[str, Any] = project_result(restored)
    status = next(layer for layer in document["levels"][0]["layers"] if layer["name"] == "Status")

    assert status["value_kind"] == "categorical"
    assert status["counts"] == {
        "total_count": 8,
        "known_count": 5,
        "missing_or_unknown_count": 3,
    }
    assert "statistics" not in status

    status_values = restored.levels["ion"].layers["Status"].values
    restored.levels["ion"].layers["Status"].values = status_values.with_columns(
        pl.Series("obs_0", [1, 2, 1, -1], dtype=pl.Int64)
    )
    second = tmp_path / "factor-again.h5ad"
    write_parsed_levels(restored, second)

    rewritten = read_parsed_levels(second)
    assert rewritten.levels["ion"].layers["Status"].values.get_column("obs_0").to_list() == [
        1,
        2,
        1,
        -1,
    ]
    rewritten_document = json.loads(sidecar_path(second).read_text(encoding="utf-8"))
    rewritten_status = next(
        layer for layer in rewritten_document["levels"][0]["layers"] if layer["name"] == "Status"
    )
    assert rewritten_status["value_kind"] == "categorical"
    assert rewritten_status["counts"] == {
        "total_count": 8,
        "known_count": 6,
        "missing_or_unknown_count": 2,
    }


def test_observation_identifiers_and_layer_summaries_share_one_fixed_cap(
    tmp_path: Path,
) -> None:
    observation_count = 105
    observation_names = [f"sample-{index}" for index in range(observation_count)]
    level = ParsedLevel(
        obs=ObsFinal(
            frame=pl.DataFrame({"sample": observation_names}),
            key_columns=("sample",),
        ),
        var=VarFinal(frame=pl.DataFrame({"feature": ["F1"]}), key_columns=("feature",)),
        primary_layer_name="Intensity",
        layers={
            "Intensity": FinalLayerTable(
                layer_name="Intensity",
                var_key_columns=("feature",),
                values=pl.DataFrame(
                    {
                        "feature": ["F1"],
                        **{f"obs_{index}": [float(index)] for index in range(observation_count)},
                    }
                ),
            )
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
        uns={},
    )
    artifact = tmp_path / "wide.parquet"
    artifact.mkdir()

    document: dict[str, Any] = project_result(
        ParsedLevels(levels={"ion": level}, uns={}),
        artifact,
    )
    projected_level = document["levels"][0]
    observations = projected_level["observations"]
    summaries = projected_level["layers"][0]["observation_summaries"]

    assert (observations["total_count"], observations["emitted_count"]) == (105, 100)
    assert observations["truncated"] is True
    assert (summaries["total_count"], summaries["emitted_count"]) == (105, 100)
    assert summaries["truncated"] is True
    assert observations["items"][-1]["label"] == "sample-99"
    assert summaries["items"][-1]["observation_index"] == 99
    assert "sample-100" not in json.dumps(document)


def test_whole_layer_quartiles_use_a_bounded_deterministic_sample() -> None:
    values = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [5, 6, 7, 8, 9]})

    first = quantitative_representation(
        values,
        observation_limit=1,
        quantile_sample_limit=4,
    )
    second = quantitative_representation(
        values,
        observation_limit=1,
        quantile_sample_limit=4,
    )

    assert first == second
    statistics = cast(dict[str, Any], first["statistics"])
    assert statistics["finite_count"] == 10
    assert statistics["mean"] == pytest.approx(4.5)
    assert statistics["quartile_method"] == "linear_deterministic_grid_sample"
    assert statistics["quartile_sample_count"] == 4
    assert statistics["quartile_sample_limit"] == 4


def test_quartile_sampling_spends_its_bound_only_on_finite_cells() -> None:
    representation = quantitative_representation(
        pl.DataFrame({"empty-a": [None], "value": [42.0], "empty-b": [float("inf")]}),
        observation_limit=0,
        quantile_sample_limit=2,
    )

    statistics = cast(dict[str, Any], representation["statistics"])
    assert statistics["finite_count"] == 1
    assert statistics["quartile_method"] == "linear_exact"
    assert statistics["quartile_sample_count"] == 1
    assert statistics["first_quartile"] == 42.0
    assert statistics["median"] == 42.0
    assert statistics["third_quartile"] == 42.0


def test_derived_nonfinite_moments_remain_valid_json() -> None:
    representation = quantitative_representation(
        pl.DataFrame({"sample": [1e308, -1e308]}),
        observation_limit=1,
    )

    statistics = cast(dict[str, Any], representation["statistics"])
    assert statistics["standard_deviation"] is None
    json.dumps(representation, allow_nan=False)


def _factor_plan() -> str:
    return json.dumps(
        {
            "canonicalization": {
                "layer_values": [
                    {
                        "kind": "plain_numeric",
                        "layer_name": name,
                        "missing_values": [],
                        "number_format": {"decimal_mark": ".", "thousands_marks": []},
                        "type": "integer" if name == "Count" else "number",
                    }
                    for name in ("Intensity", "Count")
                ]
                + [
                    {
                        "kind": "factor",
                        "layer_name": "Status",
                        "categories": [["MS/MS", 1], ["MBR", 2]],
                    }
                ],
                "layer_contract": {
                    "primary_layer_name": "Intensity",
                    "required_names": ["Intensity"],
                    "empty_ratio": 0.001,
                    "populated_ratio": 0.5,
                },
            }
        }
    )
