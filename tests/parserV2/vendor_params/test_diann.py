"""DIA-NN parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.diann import extract_params
from apb2.parserV2.vendor_params.parsers.shared.model import ParamsError
from parserV2.vendor_params import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

# DIANN_1.7.16 excluded: its checked-in expected CSV predates a code change
# (charges, abundance_normalization_ions, etc.) and disagrees with what
# ProteoBench's own parser produces today. APB matches ProteoBench runtime.
CASES = [
    "DIANN_output_20240229_report.log.txt",
    "Version1_9_Predicted_Library_report.log.txt",
    "DIANN_WU304578_report.log.txt",
    "DIANN_cfg_settings.txt",
    "DIANN_cfg_MBR.txt",
    "DIA-NN_cfg_directq.txt",
]
DDA_FIXTURE = PROTEOBENCH_PARAMS / "DIANN_DDA_report.log.txt"


@pytest.mark.parametrize("txt_name", CASES)
def test_diann_matches_proteobench(txt_name: str) -> None:
    txt = PROTEOBENCH_PARAMS / txt_name
    csv = txt.with_suffix(".csv")
    if not txt.exists() or not csv.exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(txt)
    expected = proteobench_params.expected_csv(csv)

    fields = [
        "software_name",
        "software_version",
        "search_engine",
        "enable_match_between_runs",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "enzyme",
        "allowed_miscleavages",
        "min_peptide_length",
        "max_peptide_length",
        "fixed_mods",
        "variable_mods",
        "max_mods",
        "min_precursor_charge",
        "max_precursor_charge",
        "ident_fdr_psm",
        "scan_window",
        "quantification_method",
        "protein_inference",
        # abundance_normalization_ions intentionally excluded: ProteoBench's
        # checked-in expected CSVs predate a code change in extract_params,
        # so the fixtures disagree with what ProteoBench's parser produces
        # today. APB matches the current ProteoBench runtime output.
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, f"{txt_name}: " + "; ".join(mismatches)


# --- graceful degrade: a non-DIA-NN param file must not crash the parser (root-cause fix) --------


def test_extract_params_rejects_non_diann_file_cleanly(tmp_path: Path) -> None:
    # A FragPipe workflow file mis-attached to a DIA-NN submission (real ProteoBench case): no
    # `diann --` command line and no DIA-NN version banner → a clean ParamsError, not
    # InvalidVersion.
    bad = tmp_path / "param_0..workflow"
    bad.write_text("# FragPipe (22.0) runtime properties\nfragpipe.config.bin-msfragger=/x\n")
    with pytest.raises(ParamsError, match="not a DIA-NN parameter file"):
        extract_params(bad)


@pytest.mark.parametrize("txt_name", CASES)
def test_existing_diann_fixtures_are_dia(txt_name: str) -> None:
    txt = PROTEOBENCH_PARAMS / txt_name
    if not txt.exists():
        pytest.skip("ProteoBench fixture missing")

    assert extract_params(txt).acquisition_method == "DIA"


def test_diann_detects_dda_fixture() -> None:
    assert extract_params(DDA_FIXTURE).acquisition_method == "DDA"


def test_diann_detects_dda_command_line_without_log_marker(tmp_path: Path) -> None:
    params_file = tmp_path / "diann.log"
    params_file.write_text("diann --unimod4 --dda\n")

    assert extract_params(params_file).acquisition_method == "DDA"


def test_diann_detects_exact_dda_log_marker_without_flag(tmp_path: Path) -> None:
    params_file = tmp_path / "diann.log"
    params_file.write_text(
        "DIA-NN 2.6.0 Enterprise "
        "(Data-Independent Acquisition by Neural Networks)\n"
        "diann --unimod4\n"
        "All runs will be analysed as DDA runs\n"
    )

    assert extract_params(params_file).acquisition_method == "DDA"


def test_extract_params_accepts_unversioned_version_invariant_unimod4(tmp_path: Path) -> None:
    params_file = tmp_path / "diann.log"
    params_file.write_text("diann --unimod4\n")

    params = extract_params(params_file)

    assert params.software_version is None
    assert [mod.name for mod in params.fixed_mods] == ["C[Carbamidomethyl]"]


def test_extract_params_rejects_invalid_version_instead_of_guessing(tmp_path: Path) -> None:
    params_file = tmp_path / "diann.log"
    params_file.write_text(
        "DIA-NN not-a-version (Data-Independent Acquisition by Neural Networks)\ndiann --unimod4\n"
    )

    with pytest.raises(ParamsError, match="invalid DIA-NN version"):
        extract_params(params_file)


def test_unversioned_version_sensitive_unimod_is_rejected(tmp_path: Path) -> None:
    params_file = tmp_path / "diann.log"
    params_file.write_text("diann --unimod35\n")

    with pytest.raises(ParamsError, match="version is required"):
        extract_params(params_file)
