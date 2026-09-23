"""CLI tests: the rule-config convert route end to end on a tiny table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import anndata
import mudata
import pytest

from apb2.cli import ConvertCliOptions, app, convert
from apb2.command.conversion import ConversionError
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from apb2.parserV2.parse_quant.io.json_representation import sidecar_path
from apb2.parserV2.parse_quant.io.metadata import (
    NAMESPACE,
    PARSE_NAMESPACE,
)
from parserV2.fixtures import committed_sample

_DOCUMENT = {
    "schema_version": "0.8",
    "file_version": "1",
    "software_name": "CliTest",
    "software_version_pattern": "^1$",
    "tables": [
        {
            "input": {"shape": "long", "extensions": [".tsv"]},
            "base": {
                "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
                "columns": {
                    "obs": [{"name": "sample", "source": "Run"}],
                    "var": [{"name": "feature", "source": "Precursor"}],
                },
                "measurements": {
                    "primary_layer": "Abundance",
                    "layers": [{"name": "Abundance", "source": "Intensity"}],
                },
            },
            "levels": {"ion": {}},
        }
    ],
}

_TSV = "Run\tPrecursor\tIntensity\ns1\tp1\t1.5\ns1\tp2\t2.5\ns2\tp1\t3.5\n"


def test_convert_help_exposes_one_software_hint(capsys: pytest.CaptureFixture[str]) -> None:
    app(["convert", "--help"], exit_on_error=False, result_action="return_value")
    help_text = capsys.readouterr().out
    assert "--software" in help_text
    assert "Software hint" in help_text
    assert "--params-software" not in help_text


def test_convert_diann_with_software_and_no_params(tmp_path: Path) -> None:
    source = committed_sample("diann/v2")
    assert source is not None

    exit_code = convert(
        source,
        "ion",
        ConvertCliOptions(software="DIA-NN", output=tmp_path / "diann"),
    )

    assert exit_code == 0
    result = anndata.read_h5ad(tmp_path / "diann.h5ad")
    assert result.shape[0] > 0


_MULTILEVEL_DOCUMENT = {
    "schema_version": "0.8",
    "file_version": "1",
    "software_name": "CliTest",
    "software_version_pattern": "^1$",
    "tables": [
        {
            "input": {"shape": "long", "extensions": [".tsv"]},
            "base": {
                "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
                "columns": {
                    "obs": [{"name": "sample", "source": "Run"}],
                    "var": [
                        {"name": "feature", "source": "Precursor"},
                        {"name": "protein", "source": "Protein"},
                    ],
                },
                "measurements": {
                    "primary_layer": "Abundance",
                    "layers": [{"name": "Abundance", "source": "Intensity"}],
                },
            },
            "levels": {
                "ion": {},
                "protein": {"axis": {"var_keys": ["protein"]}},
            },
        }
    ],
}

_MULTILEVEL_TSV = (
    "Run\tPrecursor\tProtein\tIntensity\ns1\tp1\tP1\t1.5\ns1\tp2\tP2\t2.5\ns2\tp1\tP1\t3.5\n"
)


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
    level = written.uns[NAMESPACE][PARSE_NAMESPACE]
    assert "rule_selection_method" not in level
    assert json.loads(str(level["rule_json"]))["software_name"] == "CliTest"
    representation = json.loads(sidecar_path(tmp_path / "out.h5ad").read_text())
    assert representation["levels"][0]["dimensions"] == {
        "observations": 2,
        "variables": 2,
    }


def test_convert_writes_optional_separate_timing_file(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")
    timings = tmp_path / "converted.timings.json"

    assert (
        convert(
            report,
            "ion",
            ConvertCliOptions(
                rule_config=rule_config,
                output=tmp_path / "out",
                timings_output=timings,
            ),
        )
        == 0
    )

    document = json.loads(timings.read_text(encoding="utf-8"))
    assert document["format"] == "apb-tool-timings"
    assert document["format_version"] == 1
    assert document["tool"] == "apb2"
    assert document["operation"] == "convert"
    assert [phase["name"] for phase in document["phases"]] == ["compile", "read", "parse", "write"]
    assert all(phase["seconds"] >= 0 for phase in document["phases"])
    assert [level["level"] for level in document["levels"]] == ["ion"]
    assert "timings" not in anndata.read_h5ad(tmp_path / "out.h5ad").uns
    assert (
        convert(
            report,
            "ion",
            ConvertCliOptions(
                rule_config=rule_config,
                output=tmp_path / "again",
                timings_output=timings,
            ),
        )
        == 2
    )
    assert not (tmp_path / "again.h5ad").exists()


def test_convert_without_a_level_writes_every_rule_level_as_mudata(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_MULTILEVEL_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_MULTILEVEL_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        options=ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
    )

    assert exit_code == 0
    written = mudata.read_h5mu(tmp_path / "out.h5mu")
    assert list(written.mod) == ["ion", "protein"]
    assert list(written.mod) == ["ion", "protein"]
    representation = json.loads(sidecar_path(tmp_path / "out.h5mu").read_text())
    assert [level["name"] for level in representation["levels"]] == ["ion", "protein"]


@pytest.mark.parametrize("storage_format", ["parquet", "duckdb"])
def test_convert_without_a_level_uses_selected_storage_writer(
    storage_format: Literal["parquet", "duckdb"],
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_MULTILEVEL_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_MULTILEVEL_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        options=ConvertCliOptions(
            rule_config=rule_config,
            output=tmp_path / "out",
            storage_format=storage_format,
        ),
    )

    target = tmp_path / f"out.{storage_format}"
    assert exit_code == 0
    assert list(read_parsed_levels(target).levels) == ["ion", "protein"]
    assert sidecar_path(target).is_file()


def test_convert_without_a_level_keeps_one_compatible_level_in_mudata(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    rule_config = tmp_path / "rules.json"
    rule_config.write_text(json.dumps(_DOCUMENT), encoding="utf-8")

    exit_code = convert(
        report,
        options=ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
    )

    assert exit_code == 0
    assert list(mudata.read_h5mu(tmp_path / "out.h5mu").mod) == ["ion"]


def test_convert_with_rule_config_does_not_embed_the_parameter_record(tmp_path: Path) -> None:
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
            software="WOMBAT",
            rule_config=rule_config,
            output=tmp_path / "out",
        ),
    )

    assert exit_code == 0
    namespace = anndata.read_h5ad(tmp_path / "out.h5ad").uns[NAMESPACE][PARSE_NAMESPACE]
    assert "search_parameters" not in namespace
    assert "search_parameters_path" not in namespace


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


def test_multilevel_convert_rejects_output_that_already_ends_in_h5mu(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")

    exit_code = convert(
        report,
        options=ConvertCliOptions(output=tmp_path / "out.h5mu"),
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
            software="wombat",
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

    monkeypatch.setattr("apb2.cli.conversion.convert_from_rule_config", fail_write)

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

    monkeypatch.setattr("apb2.cli.conversion.convert_from_rule_config", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="implementation defect"):
        convert(
            report,
            "ion",
            ConvertCliOptions(rule_config=rule_config, output=tmp_path / "out"),
        )
