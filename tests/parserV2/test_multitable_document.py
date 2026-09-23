"""One vendor document selects independent tables before any output is written."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from apb2.command.conversion import (
    ConversionError,
    convert_all_from_rule_config,
    convert_from_rule_config,
)
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    AmbiguousRuleError,
    detect_rule_documents,
    select_document_levels,
)
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from apb2.parserV2.parse_quant.io.json_representation import sidecar_path
from apb2.parserV2.parse_quant.parameters.source import Folder, SingleFile
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_parse_rules.document import document_json_schema, make_rule_document
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel
from apb2.parserV2.vendor_parse_rules.schema_artifact import artifact_path


def _table(level: str, filename: str) -> dict[str, Any]:
    return {
        "input": {"shape": "long", "extensions": [".tsv"], "file_name": filename},
        "base": {
            "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
            "columns": {
                "obs": [{"name": "sample", "source": "Run"}],
                "var": [{"name": "feature", "source": "Feature"}],
            },
            "measurements": {
                "primary_layer": "Quantity",
                "layers": [{"name": "Quantity", "source": "Intensity"}],
            },
        },
        "levels": {level: {}},
    }


def _payload() -> dict[str, Any]:
    return {
        "schema_version": "0.8",
        "file_version": "1",
        "software_name": "Synthetic",
        "software_version_pattern": "^1$",
        "tables": [_table("protein", "proteins.tsv"), _table("ion", "ions.tsv")],
    }


def _rule_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    return path


def _source(path: Path, value: int = 1) -> Path:
    path.write_text(f"Run\tFeature\tIntensity\ns1\tf1\t{value}\n", encoding="utf-8")
    return path


def test_table_local_shapes_and_merges_cannot_leak() -> None:
    payload = _payload()
    table = payload["tables"][0]
    table["input"]["shape"] = "wide"
    del table["base"]["columns"]["obs"]
    table["base"]["measurements"]["layers"][0]["source"] = "^Intensity (?P<sample>.+)$"
    document = make_rule_document(Path("rules.json"), payload)

    assert document.levels == ("ion", "protein")
    assert document.table_levels == (("protein",), ("ion",))
    assert document.declared("ion").input.file_name == "ions.tsv"
    assert document.declared("ion").declaration.shape == "long"
    assert document.declared("protein").input.file_name == "proteins.tsv"
    assert document.declared("protein").declaration.shape == "wide"


@pytest.mark.parametrize(
    ("keys", "existing", "message"),
    [
        (("Experiment", "Raw_File"), True, "existing combined target"),
        (("same-key", "same_key"), False, "distinct output filenames"),
    ],
)
def test_split_output_conflicts_fail_before_any_write(
    tmp_path: Path, keys: tuple[str, str], existing: bool, message: str
) -> None:
    payload = _payload()
    for table, key in zip(payload["tables"], keys, strict=True):
        table["base"]["axis"]["obs_keys"] = [key]
        table["base"]["columns"]["obs"][0]["name"] = key
    rule = tmp_path / "rules.json"
    rule.write_text(json.dumps(payload))
    _source(tmp_path / "ions.tsv")
    _source(tmp_path / "proteins.tsv")
    target = tmp_path / "result.h5mu"
    if existing:
        target.write_text("user result")
    with pytest.raises(ConversionError, match=message):
        convert_all_from_rule_config(
            data=tmp_path,
            output=target,
            rule_config=rule,
            parameters_path=None,
            software=None,
            checks="standard",
        )
    if existing:
        assert target.read_text() == "user result"
    assert not list(tmp_path.glob("result.*.h5mu"))


@pytest.mark.parametrize("invalid", [[], [{}], [{"input": {}, "base": {}, "levels": {}}]])
def test_empty_or_incomplete_tables_are_rejected(invalid: list[dict[str, Any]]) -> None:
    payload = _payload()
    payload["tables"] = invalid
    with pytest.raises(ValidationError):
        make_rule_document(Path("rules.json"), payload)


def test_duplicate_level_ownership_is_rejected() -> None:
    payload = _payload()
    payload["tables"].append(copy.deepcopy(payload["tables"][0]))
    with pytest.raises(ValidationError, match="exactly one table"):
        make_rule_document(Path("rules.json"), payload)


def test_published_authoring_schema_describes_tables_not_runtime_paths() -> None:
    schema = document_json_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "tables" in properties
    assert not {"path", "input", "base", "levels"}.intersection(properties)
    published = artifact_path().with_name("document.schema.json")
    assert json.loads(published.read_text()) == schema


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_old_single_input_document_shape_has_no_compatibility_alias(version: str) -> None:
    payload = _payload()
    payload["schema_version"] = version
    payload.update(payload.pop("tables")[0])
    with pytest.raises(ValidationError):
        make_rule_document(Path("rules.json"), payload)


def test_declared_filename_selects_only_its_table_despite_identical_headers(tmp_path: Path) -> None:
    document = load_rule_document(_rule_file(tmp_path))
    source = SingleFile(path=_source(tmp_path / "ions.tsv"))
    selected = select_document_levels(document, source, LEVELS, UNKNOWN_SEARCH_PARAMETERS)
    assert [item.level for item in selected] == ["ion"]
    assert select_document_levels(document, source, ("protein",), UNKNOWN_SEARCH_PARAMETERS) == ()


def test_renamed_file_rejects_ambiguous_groups_but_explicit_level_resolves(tmp_path: Path) -> None:
    document = load_rule_document(_rule_file(tmp_path))
    source = SingleFile(path=_source(tmp_path / "input_file.txt"))
    with pytest.raises(AmbiguousRuleError, match="several table groups"):
        select_document_levels(document, source, LEVELS, UNKNOWN_SEARCH_PARAMETERS)
    selected = select_document_levels(document, source, ("protein",), UNKNOWN_SEARCH_PARAMETERS)
    assert [item.level for item in selected] == ["protein"]


@pytest.mark.parametrize("suffix", [".h5mu", ".parquet", ".duckdb"])
def test_explicit_multitable_conversion_round_trips_in_canonical_order(
    tmp_path: Path, suffix: str
) -> None:
    rule = _rule_file(tmp_path)
    _source(tmp_path / "ions.tsv", 11)
    _source(tmp_path / "proteins.tsv", 22)
    target = tmp_path / f"converted{suffix}"
    summary = convert_all_from_rule_config(
        data=tmp_path,
        output=target,
        rule_config=rule,
        parameters_path=None,
        software=None,
        checks="standard",
    )
    assert [item.level for item in summary.levels] == ["ion", "protein"]
    result = read_parsed_levels(target)
    assert list(result.levels) == ["ion", "protein"]
    assert result.levels["ion"].obs.frame.height == result.levels["protein"].obs.frame.height == 1
    expected_values: tuple[tuple[QuantificationLevel, int], ...] = (("ion", 11), ("protein", 22))
    for level, expected in expected_values:
        values = result.levels[level].layers["Quantity"].values.get_column("obs_0")
        assert values.cast(pl.Float64).to_list() == [expected]
    assert sidecar_path(target).exists()


@pytest.mark.parametrize("suffix", [".h5mu", ".parquet", ".duckdb"])
def test_integer_measurements_round_trip_with_canonical_integer_dtype(
    tmp_path: Path, suffix: str
) -> None:
    payload = _payload()
    for table in payload["tables"]:
        table["base"]["measurements"]["layers"][0]["type"] = "integer"
    rule = tmp_path / "rules.json"
    rule.write_text(json.dumps(payload), encoding="utf-8")
    _source(tmp_path / "ions.tsv", 11)
    _source(tmp_path / "proteins.tsv", 22)
    target = tmp_path / f"converted{suffix}"

    convert_all_from_rule_config(
        data=tmp_path,
        output=target,
        rule_config=rule,
        parameters_path=None,
        software=None,
        checks="standard",
    )

    representation = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    layers = [level["layers"][0] for level in representation["levels"]]
    assert [(layer["type"], layer["dtype"]) for layer in layers] == [
        ("integer", "Int64"),
        ("integer", "Int64"),
    ]
    result = read_parsed_levels(target)
    assert list(result.levels) == ["ion", "protein"]
    expected_values: tuple[tuple[QuantificationLevel, int], ...] = (("ion", 11), ("protein", 22))
    for level, expected in expected_values:
        values = result.levels[level].layers["Quantity"].values.get_column("obs_0")
        assert values.cast(pl.Float64).to_list() == [expected]
    assert sidecar_path(target).exists()


@pytest.mark.parametrize(
    "malformed", ["Bad\tHeader\n", "Run\tFeature\tIntensity\ns1\tf1\t1\ns1\tf1\t2\n"]
)
def test_explicit_present_malformed_table_preserves_existing_output(
    tmp_path: Path, malformed: str
) -> None:
    rule = _rule_file(tmp_path)
    _source(tmp_path / "ions.tsv")
    bad = tmp_path / "proteins.tsv"
    bad.write_text(malformed, encoding="utf-8")
    target = tmp_path / "out.h5mu"
    target.write_bytes(b"existing artifact")
    sidecar_path(target).write_text("existing sidecar", encoding="utf-8")
    with pytest.raises(ConversionError):
        convert_all_from_rule_config(
            data=tmp_path,
            output=target,
            rule_config=rule,
            parameters_path=None,
            software=None,
            checks="standard",
        )
    assert target.read_bytes() == b"existing artifact"
    assert sidecar_path(target).read_text() == "existing sidecar"


def test_explicit_requested_missing_table_has_focused_error(tmp_path: Path) -> None:
    rule = _rule_file(tmp_path)
    _source(tmp_path / "ions.tsv")
    with pytest.raises(ConversionError, match=r"proteins\.tsv"):
        convert_from_rule_config(
            data=tmp_path,
            level="protein",
            output=tmp_path / "out.h5ad",
            rule_config=rule,
            parameters_path=None,
            software=None,
            checks="standard",
        )


@pytest.mark.parametrize("level", ["ion", "protein"])
def test_packaged_filename_selection_matches_explicit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: QuantificationLevel
) -> None:
    rule = _rule_file(tmp_path)
    monkeypatch.setattr("apb2.parserV2.detect_document.PACKAGED", (rule,))
    name = "ions.tsv" if level == "ion" else "proteins.tsv"
    source = SingleFile(path=_source(tmp_path / name))
    parameters = Parameters(software_name="Synthetic", software_version="1")
    detected = detect_rule_documents(parameters, source, LEVELS)
    assert [item.level for item in detected.levels] == [level]


def test_folder_skips_absent_tables_without_losing_the_available_one(tmp_path: Path) -> None:
    document = load_rule_document(_rule_file(tmp_path))
    _source(tmp_path / "ions.tsv")
    selected = select_document_levels(
        document, Folder(path=tmp_path), LEVELS, UNKNOWN_SEARCH_PARAMETERS
    )
    assert [item.level for item in selected] == ["ion"]
