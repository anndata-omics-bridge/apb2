"""MSAID parameter parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.msaid import extract_params
from parserV2.vendor_params import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"


def test_msaid_matches_proteobench() -> None:
    csv = PROTEOBENCH_PARAMS / "MSAID_default_params.csv"
    expected_tsv = PROTEOBENCH_PARAMS / "MSAID_default_params.tsv"
    if not csv.exists() or not expected_tsv.exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(csv)
    expected = proteobench_params.expected_csv(expected_tsv, delimiter="\t")
    # semi_enzymatic intentionally excluded: ProteoBench's dynamic dataclass only
    # populates fields declared in the JSON template, and DIA_ion.json omits
    # semi_enzymatic, so the expected TSV has it blank. APB always emits it.
    fields = [
        "software_name",
        "search_engine",
        "search_engine_version",
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
        "quantification_method",
        "enable_match_between_runs",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, "; ".join(mismatches)
