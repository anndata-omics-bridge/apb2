"""MaxQuant XML parser equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import proteobench_params
from apb2.vendor_params.parsers.maxquant import extract_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    ("mqpar1.5.3.30_MBR.xml", "mqpar1.5.3.30_MBR_sel.json"),
    ("mqpar_MQ1.6.3.3_MBR.xml", "mqpar_MQ1.6.3.3_MBR_sel.json"),
    ("mqpar_MQ2.1.3.0_noMBR.xml", "mqpar_MQ2.1.3.0_noMBR_sel.json"),
    ("mqpar_mq2.6.2.0_1mc_MBR.xml", "mqpar_mq2.6.2.0_1mc_MBR_sel.json"),
]


@pytest.mark.parametrize(("xml_name", "expected_name"), CASES)
def test_maxquant_matches_proteobench(xml_name: str, expected_name: str) -> None:
    xml_path = PROTEOBENCH_PARAMS / xml_name
    expected_path = PROTEOBENCH_PARAMS / expected_name
    if not xml_path.exists() or not expected_path.exists():
        pytest.skip("ProteoBench fixture missing")
    expected = proteobench_params.expected_json(expected_path)
    params = extract_params(xml_path)

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
        "semi_enzymatic",
        "allowed_miscleavages",
        "min_peptide_length",
        "fixed_mods",
        "variable_mods",
        "max_mods",
        "max_precursor_charge",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, "; ".join(mismatches)
