"""MetaMorpheus parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apb2.parserV2.search_parameters.metamorpheus import extract_params
from parserV2.search_parameters import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"
TOML_FILE = PROTEOBENCH_PARAMS / "metamorpheus_search_task_config.toml"
VERSION_FILE = PROTEOBENCH_PARAMS / "metamorpheus_version_result.txt"
EXPECTED_CSV = PROTEOBENCH_PARAMS / "metamorpheus_parameters.csv"


@pytest.mark.skipif(not TOML_FILE.exists(), reason="ProteoBench fixture missing")
def test_metamorpheus_matches_proteobench() -> None:
    params = extract_params(TOML_FILE, VERSION_FILE)
    expected = proteobench_params.expected_csv(EXPECTED_CSV)
    fields = [
        "software_name",
        "software_version",
        "search_engine",
        "enzyme",
        "allowed_miscleavages",
        "fixed_mods",
        "variable_mods",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "min_peptide_length",
        "max_peptide_length",
        "max_mods",
        "min_precursor_charge",
        "max_precursor_charge",
        "enable_match_between_runs",
        "quantification_method",
        "ident_fdr_psm",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, "; ".join(mismatches)


@pytest.mark.skipif(not TOML_FILE.exists(), reason="ProteoBench fixture missing")
def test_metamorpheus_input_order_insensitive() -> None:
    direct = extract_params(TOML_FILE, VERSION_FILE)
    reversed_ = extract_params(VERSION_FILE, TOML_FILE)
    assert direct == reversed_
