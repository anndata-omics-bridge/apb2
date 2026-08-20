"""PEAKS parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import proteobench_params
from apb2.vendor_params.parsers.peaks import extract_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    "PEAKS_parameters.txt",
    "PEAKS_parameters_DDA.txt",
    "PEAKS_parameters_DIA.txt",
    "PEAKS_parameters_DDA_new.txt",
    "PEAKS_diaPASEF.txt",
]


@pytest.mark.parametrize("txt_name", CASES)
def test_peaks_matches_proteobench(txt_name: str) -> None:
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
        "search_engine_version",
        "ident_fdr_psm",
        "ident_fdr_peptide",
        "ident_fdr_protein",
        "enable_match_between_runs",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "enzyme",
        "semi_enzymatic",
        "allowed_miscleavages",
        "min_peptide_length",
        "max_peptide_length",
        "fixed_mods",
        "variable_mods",
        "max_mods",
        "min_precursor_charge",
        "max_precursor_charge",
        "quantification_method",
        "abundance_normalization_ions",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, f"{txt_name}: " + "; ".join(mismatches)
