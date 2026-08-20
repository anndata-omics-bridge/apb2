"""FragPipe workflow-file parser equivalence tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

import proteobench_params
from apb2.vendor_params.parsers.fragpipe import extract_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    "fragpipe.workflow",
    "fragpipe_older.workflow",
    "fragpipe_win_paths.workflow",
    "fragpipe_v22.workflow",
    "fragpipe_fdr_test.workflow",
    "fragpipe-version.workflow",
    "fragpipe_v23_noMBR.workflow",
]


@pytest.mark.parametrize("workflow_name", CASES)
def test_fragpipe_matches_proteobench(workflow_name: str) -> None:
    workflow = PROTEOBENCH_PARAMS / workflow_name
    expected_csv = PROTEOBENCH_PARAMS / f"{Path(workflow_name).stem}_extracted_params.csv"
    if not workflow.exists() or not expected_csv.exists():
        pytest.skip("ProteoBench fixture missing")
    params = extract_params(workflow)
    expected = proteobench_params.expected_csv(expected_csv)

    fields = [
        "software_name",
        "software_version",
        "search_engine",
        "search_engine_version",
        "enzyme",
        "semi_enzymatic",
        "allowed_miscleavages",
        "fixed_mods",
        "variable_mods",
        "max_mods",
        "min_peptide_length",
        "max_peptide_length",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "ident_fdr_psm",
        "ident_fdr_protein",
        "enable_match_between_runs",
        "min_precursor_charge",
        "protein_inference",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, f"{workflow_name}: " + "; ".join(mismatches)


def test_fragpipe_exposes_embedded_diann_quantification_version() -> None:
    params = extract_params(PROTEOBENCH_PARAMS / "fragpipe.workflow")

    assert params.software_name == "FragPipe"
    assert params.software_version == "23.0"
    assert params.quantification_software == "DIA-NN"
    assert params.quantification_software_version == "1.8.2 beta 8"


def test_fragpipe_without_diann_quantification_has_no_embedded_quantifier() -> None:
    params = extract_params(PROTEOBENCH_PARAMS / "fragpipe_v23_noMBR.workflow")

    assert params.quantification_software is None
    assert params.quantification_software_version is None


def test_fragpipe_reads_diann_version_from_legacy_executable_path() -> None:
    workflow = (PROTEOBENCH_PARAMS / "fragpipe.workflow").read_text()
    workflow = workflow.replace("# DIA-NN version 1.8.2 beta 8\n", "").replace(
        "fragpipe-config.bin-diann=C\\:\\FragPipe\\FragPipe-23.0\\tools\\diann\\1.8.2_beta_8\\windows\\DiaNN.exe",
        "fragpipe-config.bin-diann=C\\:\\tools\\diann\\1.8.2_beta_8\\win\\DiaNN.exe",
    )

    parsed = extract_params(StringIO(workflow))
    assert parsed.quantification_software_version == "1.8.2 beta 8"
