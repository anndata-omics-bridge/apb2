"""Spectronaut parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apb2.parserV2.search_parameters.spectronaut import extract_params
from parserV2.search_parameters import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    "spectronaut_Experiment1_ExperimentSetupOverview_BGS_Factory_Settings.txt",
    "Spectronaut_dynamic.txt",
    "Spectronaut_static.txt",
    "Spectronaut_relative.txt",
]


@pytest.mark.parametrize("txt_name", CASES)
def test_spectronaut_matches_proteobench(txt_name: str) -> None:
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
        "ident_fdr_psm",
        "ident_fdr_protein",
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
        "quantification_method",
        "protein_inference",
        "abundance_normalization_ions",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, f"{txt_name}: " + "; ".join(mismatches)
