"""Sage parser equivalence tests against ProteoBench fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.sage import extract_params
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from parserV2.vendor_params import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"
SAGE_PARAMETERFILE = PROTEOBENCH_PARAMS / "sage_parameterfile.json"
SAGE_PARAMETERFILE_CSV = PROTEOBENCH_PARAMS / "sage_parameterfile.csv"
SAGE_RESULTS = PROTEOBENCH_PARAMS / "sage_results.json"
SAGE_RESULTS_CSV = PROTEOBENCH_PARAMS / "sage_results.csv"


def _assert_matches_expected(params: Parameters, expected_csv: Path) -> None:
    expected = proteobench_params.expected_csv(expected_csv)
    fields_to_check = [
        "software_name",
        "software_version",
        "search_engine",
        "search_engine_version",
        "enzyme",
        "semi_enzymatic",
        "allowed_miscleavages",
        "min_peptide_length",
        "max_peptide_length",
        "max_mods",
        "min_precursor_charge",
        "max_precursor_charge",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "enable_match_between_runs",
        "fixed_mods",
        "variable_mods",
    ]
    mismatches = proteobench_params.compare(params, expected, fields_to_check)
    assert not mismatches, "; ".join(mismatches)


@pytest.mark.skipif(not SAGE_PARAMETERFILE.exists(), reason="ProteoBench fixture missing")
def test_sage_parameterfile_matches_proteobench_csv() -> None:
    params = extract_params(SAGE_PARAMETERFILE)
    _assert_matches_expected(params, SAGE_PARAMETERFILE_CSV)


@pytest.mark.skipif(not SAGE_RESULTS.exists(), reason="ProteoBench fixture missing")
def test_sage_results_matches_proteobench_csv() -> None:
    params = extract_params(SAGE_RESULTS)
    _assert_matches_expected(params, SAGE_RESULTS_CSV)


def test_sage_accepts_filelike_object(tmp_path: Path) -> None:
    payload = b"""{
        "version": "0.14.6",
        "database": {
            "enzyme": {
                "missed_cleavages": 1, "min_len": 7, "max_len": 50,
                "cleave_at": "KR", "restrict": null, "c_terminal": true,
                "semi_enzymatic": null
            },
            "static_mods": {"C": 57.02146},
            "variable_mods": {"M": [15.9949]},
            "max_variable_mods": 3
        },
        "precursor_tol": {"ppm": [-20.0, 20.0]},
        "fragment_tol": {"ppm": [-20.0, 20.0]},
        "precursor_charge": [1, 7]
    }"""
    buf = io.BytesIO(payload)
    params = extract_params(buf)
    assert params.software_name == "Sage"
    assert params.software_version == "0.14.6"
    assert params.precursor_mass_tolerance is not None
    assert params.precursor_mass_tolerance.value == 20.0
    assert params.precursor_mass_tolerance.unit == "ppm"
    assert params.precursor_mass_tolerance.mode == "absolute"
    # A null `restrict` is no restriction, so cleavage happens after K and R regardless
    # of a following proline: Trypsin/P.
    assert params.enzyme == "Trypsin/P"


def test_sage_trypsin_p_when_restrict_missing() -> None:
    payload = b"""{
        "version": "0.14.6",
        "database": {
            "enzyme": {
                "missed_cleavages": 1, "min_len": 7, "max_len": 50,
                "cleave_at": "KR", "semi_enzymatic": null
            },
            "static_mods": {}, "variable_mods": {}, "max_variable_mods": 3
        },
        "precursor_tol": {"ppm": [-20.0, 20.0]},
        "fragment_tol": {"ppm": [-20.0, 20.0]},
        "precursor_charge": [1, 7]
    }"""
    params = extract_params(io.BytesIO(payload))
    assert params.enzyme == "Trypsin/P"


def test_sage_trypsin_with_p_restrict() -> None:
    payload = b"""{
        "version": "0.14.6",
        "database": {
            "enzyme": {
                "missed_cleavages": 1, "min_len": 7, "max_len": 50,
                "cleave_at": "KR", "restrict": "P",
                "semi_enzymatic": null
            },
            "static_mods": {}, "variable_mods": {}, "max_variable_mods": 3
        },
        "precursor_tol": {"ppm": [-20.0, 20.0]},
        "fragment_tol": {"ppm": [-20.0, 20.0]},
        "precursor_charge": [1, 7]
    }"""
    params = extract_params(io.BytesIO(payload))
    assert params.enzyme == "Trypsin"


def test_sage_semi_enzymatic_true() -> None:
    payload = b"""{
        "version": "0.14.6",
        "database": {
            "enzyme": {
                "missed_cleavages": 1, "min_len": 7, "max_len": 50,
                "cleave_at": "KR", "semi_enzymatic": true
            },
            "static_mods": {}, "variable_mods": {}, "max_variable_mods": 3
        },
        "precursor_tol": {"ppm": [-20.0, 20.0]},
        "fragment_tol": {"ppm": [-20.0, 20.0]},
        "precursor_charge": [1, 7]
    }"""
    params = extract_params(io.BytesIO(payload))
    assert params.semi_enzymatic is True
