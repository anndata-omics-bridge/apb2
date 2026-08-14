"""Working-parser tests for V2: real construction, dialect resolution, parse, write."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest

from apb2.errors import (
    AmbiguousDialectError,
    IncompatibleSourceError,
    NoCompatibleLevelError,
)
from apb2.input import UnknownFormat
from apb2.output import to_anndata
from apb2.parse_strategy import make_parse_strategies, make_parse_strategy
from apb2.sources import DelimitedDialect, DelimitedFile, Folder, GroupedNumbers, SingleFile
from apb2.vendor_parse_rules.model import LongRule, WideRule, validate_rule


def _ion_rule() -> LongRule | WideRule:
    return validate_rule(
        {
            "schema_version": "0.2",
            "file_version": "1",
            "software_name": "V2Test",
            "software_version_pattern": "^1$",
            "shape": "long",
            "quantification_level": "ion",
            "axis": {"obs_keys": ["sample"], "var_keys": ["feature"], "x_layer": "Abundance"},
            "columns": {
                "obs": {"select": {"sample": "Run"}},
                "var": {"select": {"feature": "Precursor"}},
            },
            "layers": [{"name": "Abundance", "source": "Intensity"}],
        }
    )


def _protein_rule() -> LongRule | WideRule:
    return validate_rule(
        {
            "schema_version": "0.2",
            "file_version": "1",
            "software_name": "V2Test",
            "software_version_pattern": "^1$",
            "shape": "long",
            "quantification_level": "protein",
            "axis": {"obs_keys": ["sample"], "var_keys": ["feature"], "x_layer": "Abundance"},
            "columns": {
                "obs": {"select": {"sample": "Run"}},
                "var": {"select": {"feature": "Protein"}},
            },
            "layers": [{"name": "Abundance", "source": "Intensity"}],
        }
    )


_TSV = "Run\tPrecursor\tProtein\tIntensity\ns1\tp1\tA\t1.5\ns1\tp2\tB\t2.5\ns2\tp1\tA\t3.5\n"


def test_make_parser_parses_a_tsv_end_to_end(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")

    parser = make_parse_strategy(_ion_rule(), SingleFile(report))
    parsed = parser.parse()

    assert list(parsed.obs["sample"]) == ["s1", "s2"]
    assert list(parsed.var["feature"]) == ["p1", "p2"]
    np.testing.assert_allclose(parsed.X, [[1.5, 2.5], [3.5, np.nan]], equal_nan=True)
    assert set(parsed.layers) == {"Abundance"}
    assert parsed.uns["software_name"] == "V2Test"
    assert parsed.uns["quantification_level"] == "ion"


def test_make_parsers_builds_compatible_levels_in_order(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")

    parsers = make_parse_strategies([_protein_rule(), _ion_rule()], SingleFile(report))

    assert [parser.level for parser in parsers] == ["ion", "protein"]
    protein = parsers[1].parse()
    assert list(protein.var["feature"]) == ["A", "B"]


def test_detection_resolves_a_semicolon_txt_export(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "Run;Precursor;Intensity\ns1;p1;1.5\ns1;p2;2.5\n",
        encoding="utf-8",
    )

    parsed = make_parse_strategy(_ion_rule(), SingleFile(report)).parse()

    np.testing.assert_allclose(parsed.X, [[1.5, 2.5]])


def test_ambiguous_delimiters_fail_construction(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "Run\tPrecursor\tIntensity\tX,Run,Precursor,Intensity\n",
        encoding="utf-8",
    )

    with pytest.raises(AmbiguousDialectError, match="explicit DelimitedFile"):
        make_parse_strategy(_ion_rule(), SingleFile(report))


def test_explicit_dialect_reads_grouped_numbers(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "Run;Precursor;Intensity\ns1;p1;100,000,000\n",
        encoding="utf-8",
    )
    source = DelimitedFile(
        report,
        DelimitedDialect(delimiter=";", numbers=GroupedNumbers(decimal=".", thousands=",")),
    )

    parsed = make_parse_strategy(_ion_rule(), source).parse()

    np.testing.assert_allclose(parsed.X, [[100_000_000.0]])


def test_incompatible_file_is_a_construction_error(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text("Other\tColumns\n1\t2\n", encoding="utf-8")

    with pytest.raises(IncompatibleSourceError, match="V2Test"):
        make_parse_strategy(_ion_rule(), SingleFile(report))


def test_folder_sources_need_a_file_set_rule(tmp_path: Path) -> None:
    with pytest.raises(IncompatibleSourceError, match="file-set"):
        make_parse_strategy(_ion_rule(), Folder(tmp_path))
    with pytest.raises(NoCompatibleLevelError):
        make_parse_strategies([_ion_rule()], Folder(tmp_path))


def test_unknown_extension_is_an_error() -> None:
    with pytest.raises(UnknownFormat, match="xlsx"):
        make_parse_strategy(_ion_rule(), SingleFile(Path("report.xlsx")))


def test_to_anndata_writes_one_file_with_the_apb_namespace(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(_TSV, encoding="utf-8")
    target = tmp_path / "ion.h5ad"

    to_anndata(make_parse_strategy(_ion_rule(), SingleFile(report)).parse(), target)

    assert target.exists()
    adata = anndata.read_h5ad(target)
    assert adata.shape == (2, 2)
    assert adata.uns["anndata_proteomics"]["software_name"] == "V2Test"
    assert "Abundance" in adata.layers
