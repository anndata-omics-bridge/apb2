"""Equivalence tests for YAML-based parameter parsers (AlphaPept, WOMBAT)."""

from __future__ import annotations

from pathlib import Path

import pytest

import proteobench_params
from apb2.vendor_params.model import Parameters
from apb2.vendor_params.parsers.alphapept import extract_params as alphapept_extract
from apb2.vendor_params.parsers.wombat import extract_params as wombat_extract

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

COMMON_FIELDS = [
    "software_name",
    "software_version",
    "search_engine",
    "enzyme",
    "allowed_miscleavages",
    "max_mods",
    "min_peptide_length",
    "max_peptide_length",
    "min_precursor_charge",
    "max_precursor_charge",
    "precursor_mass_tolerance",
    "fragment_mass_tolerance",
    "enable_match_between_runs",
]


def _compare(params: Parameters, csv: Path, extra: tuple[str, ...] = ()) -> None:
    expected = proteobench_params.expected_csv(csv)
    mismatches = proteobench_params.compare(params, expected, COMMON_FIELDS + list(extra))
    assert not mismatches, "; ".join(mismatches)


@pytest.mark.parametrize(
    "yaml_name",
    ["alphapept_0.4.9.yaml", "alphapept_0.4.9_unnormalized.yaml"],
)
def test_alphapept_matches_proteobench(yaml_name: str) -> None:
    yaml_path = PROTEOBENCH_PARAMS / yaml_name
    csv_path = yaml_path.with_suffix(".csv")
    if not yaml_path.exists():
        pytest.skip("ProteoBench fixture missing")
    params = alphapept_extract(yaml_path)
    _compare(
        params,
        csv_path,
        extra=("ident_fdr_psm", "ident_fdr_protein", "fixed_mods", "variable_mods"),
    )


def test_wombat_matches_proteobench() -> None:
    yaml_path = PROTEOBENCH_PARAMS / "wombat_params.yaml"
    csv_path = PROTEOBENCH_PARAMS / "wombat_params.csv"
    if not yaml_path.exists():
        pytest.skip("ProteoBench fixture missing")
    params = wombat_extract(yaml_path)
    _compare(
        params,
        csv_path,
        extra=(
            "ident_fdr_psm",
            "ident_fdr_peptide",
            "ident_fdr_protein",
            "fixed_mods",
            "variable_mods",
            "abundance_normalization_ions",
        ),
    )
