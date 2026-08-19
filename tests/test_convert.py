"""Working-parser tests for V2: real construction, dialect resolution, parse, write."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest

from apb2.configure_parse import make_parse_strategies, make_parse_strategy
from apb2.errors import (
    AmbiguousDialectError,
    IncompatibleSourceError,
    NoCompatibleLevelError,
)
from apb2.output import to_anndata
from apb2.parse_quant.bound_input_reader import UnknownFormat
from apb2.parse_quant.sources import (
    DelimitedDialect,
    DelimitedFile,
    Folder,
    GroupedNumbers,
    SingleFile,
)
from apb2.vendor_parse_rules.model import validate_rule
from apb2.vendor_parse_rules.rules import Rule


def _ion_rule() -> Rule:
    return Rule(
        validate_rule(
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
    )


def _protein_rule() -> Rule:
    return Rule(
        validate_rule(
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


def _modification_rule(*, extra_computed: list[dict[str, object]] | None = None) -> Rule:
    return Rule(
        validate_rule(
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
                    "var": {
                        "select": {"feature": "Precursor", "modified": "Modified"},
                        "computed": [
                            {
                                "how": "stripped_sequence",
                                "name": "ProForma_peptide",
                                "inputs": ["modified"],
                            },
                            *(extra_computed or []),
                        ],
                    },
                },
                "layers": [{"name": "Abundance", "source": "Intensity"}],
                "modifications": {
                    "parser": "token_regex",
                    "source_column": "Modified",
                    "token_pattern": r"\(([^)]+)\)",
                    "map": [{"token": "ox", "accession": "UNIMOD:35"}],
                },
            }
        )
    )


def test_vendor_column_named_like_a_modification_output_does_not_skip_the_applier(
    tmp_path: Path,
) -> None:
    """A raw `stripped_sequence` column must not defeat the mods run (review round 1)."""
    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPrecursor\tModified\tstripped_sequence\tIntensity\n"
        "s1\tp1\tPEP(ox)TIDE\tVENDORJUNK\t1.5\n",
        encoding="utf-8",
    )

    parsed = make_parse_strategy(_modification_rule(), SingleFile(report)).parse()

    assert list(parsed.var["ProForma_peptide"]) == ["PEPTIDE"]


def _optional_shared_rule() -> Rule:
    return Rule(
        validate_rule(
            {
                "schema_version": "0.2",
                "file_version": "1",
                "software_name": "V2Test",
                "software_version_pattern": "^1$",
                "shape": "long",
                "quantification_level": "ion",
                "axis": {"obs_keys": ["sample"], "var_keys": ["vkey"], "x_layer": "Abundance"},
                "columns": {
                    "obs": {"select": {"sample": "Run"}},
                    "var": {
                        "select": {"pep": "Peptide"},
                        "optional_select": {"opt": "Opt"},
                        "computed": [
                            {"how": "coalesce", "name": "vkey", "inputs": ["opt", "pep"]},
                            {"how": "coalesce", "name": "anno", "inputs": ["opt", "pep"]},
                        ],
                    },
                },
                "layers": [{"name": "Abundance", "source": "Intensity"}],
            }
        )
    )


def test_optional_skipped_in_key_phase_stays_skipped_for_rest_computes(tmp_path: Path) -> None:
    """An absent optional feeding both a key and an annotation compute must not raise."""
    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPeptide\tIntensity\ns1\tAAA\t1.5\ns1\tBBB\t2.5\n",
        encoding="utf-8",
    )

    parsed = make_parse_strategy(_optional_shared_rule(), SingleFile(report)).parse()

    assert list(parsed.var["anno"]) == ["AAA", "BBB"]
    assert list(parsed.var.index) == ["AAA", "BBB"]


def _two_layer_rule() -> Rule:
    return Rule(
        validate_rule(
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
                "layers": [
                    {"name": "Abundance", "source": "Intensity"},
                    {"name": "Score", "source": "Score"},
                ],
            }
        )
    )


def test_empty_x_layer_beside_a_populated_sibling_is_a_contract_error(tmp_path: Path) -> None:
    from apb2.parse_quant.table_conversion import LayerContractError

    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPrecursor\tIntensity\tScore\ns1\tp1\tjunk\t0.9\ns1\tp2\tjunk\t0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(LayerContractError, match="'Abundance' is effectively empty"):
        make_parse_strategy(_two_layer_rule(), SingleFile(report)).parse()


def test_strict_promotes_an_empty_auxiliary_layer_to_an_error(tmp_path: Path) -> None:
    from apb2.parse_quant.table_conversion import LayerContractError

    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPrecursor\tIntensity\tScore\ns1\tp1\t1.5\tjunk\ns1\tp2\t2.5\tjunk\n",
        encoding="utf-8",
    )

    make_parse_strategy(_two_layer_rule(), SingleFile(report)).parse()  # warning only

    with pytest.raises(LayerContractError, match="'Score' is effectively empty"):
        make_parse_strategy(_two_layer_rule(), SingleFile(report), strict=True).parse()


def _fragment_rule() -> Rule:
    return Rule(
        validate_rule(
            {
                "schema_version": "0.2",
                "file_version": "1",
                "software_name": "V2Test",
                "software_version_pattern": "^1$",
                "shape": "long",
                "quantification_level": "fragment",
                "axis": {"obs_keys": ["sample"], "var_keys": ["feature"], "x_layer": "Abundance"},
                "columns": {
                    "obs": {"select": {"sample": "Run"}},
                    "var": {
                        "select": {"precursor": "Precursor"},
                        "computed": [
                            {
                                "how": "join_nonempty",
                                "name": "feature",
                                "inputs": ["precursor", "fragment_label"],
                                "separator": "/",
                            }
                        ],
                    },
                },
                "layers": [
                    {"name": "Abundance", "source": "Frag.Quant"},
                    {"name": "Correlation", "source": "Frag.Corr"},
                ],
                "fragments": {
                    "label_strategy": "positional",
                    "value_columns": ["Frag.Quant", "Frag.Corr"],
                },
            }
        )
    )


def test_missing_optional_packed_fragment_column_skips_its_layer(tmp_path: Path) -> None:
    """An absent packed column backing an optional layer must not kill the parse."""
    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPrecursor\tFrag.Quant\ns1\tp1\t1.5;2.5\n",
        encoding="utf-8",
    )

    parsed = make_parse_strategy(_fragment_rule(), SingleFile(report)).parse()

    assert set(parsed.layers) == {"Abundance"}
    assert list(parsed.var.index) == ["p1/frag_0", "p1/frag_1"]


def test_missing_required_packed_fragment_column_fails_construction(tmp_path: Path) -> None:
    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPrecursor\tFrag.Corr\ns1\tp1\t0.9;0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(IncompatibleSourceError):
        make_parse_strategy(_fragment_rule(), SingleFile(report))


def test_vendor_column_named_like_a_skipped_optional_never_reaches_the_output(
    tmp_path: Path,
) -> None:
    """A raw column bearing a skipped optional's declared name must stay excluded."""
    report = tmp_path / "report.tsv"
    report.write_text(
        "Run\tPeptide\topt\tIntensity\ns1\tAAA\tJUNK\t1.5\n",
        encoding="utf-8",
    )

    parsed = make_parse_strategy(_optional_shared_rule(), SingleFile(report)).parse()

    assert "opt" not in parsed.var.columns
    assert list(parsed.var["anno"]) == ["AAA"]
