"""CLI tests: the rule-config convert route end to end on a tiny table."""

from __future__ import annotations

import json
from pathlib import Path

import anndata

from apb2.cli import ConvertCliOptions, convert

_DOCUMENT = {
    "schema_version": "0.2",
    "file_version": "1",
    "software_name": "CliTest",
    "software_version_pattern": "^1$",
    "input": {"shape": "long"},
    "base": {
        "axis": {"obs_keys": ["sample"]},
        "columns": {"obs": {"select": {"sample": "Run"}}},
    },
    "levels": {
        "ion": {
            "axis": {"var_keys": ["feature"], "x_layer": "Abundance"},
            "columns": {"var": {"select": {"feature": "Precursor"}}},
            "layers": [{"name": "Abundance", "source": "Intensity"}],
        }
    },
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
    namespace = written.uns["anndata_proteomics"]
    assert namespace["rule_selection_method"] == "rule_config"
    assert namespace["software_name"] == "CliTest"


def test_convert_rejects_output_with_suffix(tmp_path: Path) -> None:
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
