"""CLI tests: the rule-config convert route end to end on a tiny table."""

from __future__ import annotations

import json
from pathlib import Path

import anndata
import pytest

from apb2.cli import ConvertCliOptions, convert
from apb2.parserV2.conversion_facade import ConversionError
from apb2.parserV2.parse_quant.anndata_writer import NAMESPACE, PARSE_NAMESPACE

_DOCUMENT = {
    "schema_version": "0.3",
    "file_version": "1",
    "software_name": "CliTest",
    "software_version_pattern": "^1$",
    "input": {"shape": "long", "extensions": [".tsv"]},
    "base": {
        "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
        "columns": {
            "obs": {"select": {"sample": "Run"}},
            "var": {"select": {"feature": "Precursor"}},
        },
        "measurements": {
            "primary_layer": "Abundance",
            "layers": [{"name": "Abundance", "source": "Intensity"}],
        },
    },
    "levels": {"ion": {}},
}

_TSV = "Run\tPrecursor\tIntensity\ns1\tp1\t1.5\ns1\tp2\t2.5\ns2\tp1\t3.5\n"


def test_convert_with_rule_config_writes_h5ad(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
    )

    assert exit_code == 0
    written = anndata.read_h5ad(tmp_path / "out.h5ad")
    assert written.shape == (2, 2)
    namespace = written.uns[NAMESPACE][PARSE_NAMESPACE]
    assert namespace["rule_selection_method"] == "rule_config"
    assert namespace["software_name"] == "CliTest"


def test_convert_with_rule_config_preserves_the_complete_parameter_record(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")
    parameters = Path(__file__).parent / "vendor_params" / "params" / "wombat_params.yaml"

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(
            params=parameters,
            params_software="wombat",
            rule_config=rule_config,
            output=tmp_path / "out",
        ),
    )

    assert exit_code == 0
    namespace = anndata.read_h5ad(tmp_path / "out.h5ad").uns[NAMESPACE][PARSE_NAMESPACE]
    record = json.loads(str(namespace["search_parameters"]))
    assert record["software_name"] == "Wombat"
    assert record["software_version"] == "0.9.8"
    assert namespace["search_parameters_path"] == str(parameters)


def test_convert_accepts_a_dotted_basename_and_appends_its_own_suffix(tmp_path: Path) -> None:
    """A basename may contain dots: ``ion.apb2`` beside ``ion`` names two converters' outputs."""
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(rule_config=rule_config, output=tmp_path / "ion.apb2"),
    )

    assert exit_code == 0
    assert (tmp_path / "ion.apb2.h5ad").is_file()


def test_convert_rejects_output_that_already_carries_the_appended_suffix(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(output=tmp_path / "out.h5ad"),
    )

    assert exit_code == 2


def test_convert_missing_level_in_rule_config_fails(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        "protein",
        ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
    )

    assert exit_code == 1


def test_convert_incompatible_source_fails(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text("Wrong\tColumns\nvalue\t1\n", encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
    )

    assert exit_code == 1


def test_convert_invalid_parameters_fail(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text("not: a wombat parameter file\n", encoding="utf-8")

    exit_code = convert(
        report,
        "ion",
        ConvertCliOptions(
            params=parameters,
            params_software="wombat",
            rule_config=rule_config,
            output=tmp_path / "out",
        ),
    )

    assert exit_code == 1


def test_convert_reports_an_expected_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    def fail_write(**_arguments: object) -> None:
        raise ConversionError("empty primary layer")

    monkeypatch.setattr("apb2.cli.conversion_facade.convert_from_rule_config", fail_write)

    assert (
        convert(
            report,
            "ion",
            ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
        )
        == 1
    )


def test_convert_does_not_hide_an_unexpected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    def fail_unexpectedly(**_arguments: object) -> None:
        raise RuntimeError("implementation defect")

    monkeypatch.setattr("apb2.cli.conversion_facade.convert_from_rule_config", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="implementation defect"):
        convert(
            report,
            "ion",
            ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
        )
