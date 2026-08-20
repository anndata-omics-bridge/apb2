"""AlphaDIA run-log parameter parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from apb2.vendor_params.model import ParamsError
from apb2.vendor_params.parsers.alphadia import extract_params

PARAMS_DIR = Path(__file__).resolve().parent / "params"
FIXTURE = PARAMS_DIR / "alphadia_1.10.3.log.txt"


def test_extracts_the_config_tree_from_the_run_log() -> None:
    params = extract_params(FIXTURE)

    assert params.software_name == "AlphaDIA"
    assert params.software_version == "1.10.3"
    assert params.acquisition_method == "DIA"
    assert params.enzyme == "Trypsin/P"
    assert params.allowed_miscleavages == 1
    assert params.min_peptide_length == 6
    assert params.max_peptide_length == 30
    assert params.min_precursor_charge == 1
    assert params.max_precursor_charge == 4
    assert params.max_mods == 1
    assert params.enable_match_between_runs is False
    assert params.quantification_method == "DirectLFQ"
    assert params.predictors_library == "AlphaPeptDeep"


def test_software_version_is_the_banner_not_the_config_schema_version() -> None:
    """The config tree also carries a bare ``version: 1`` key.

    Reading ``version`` by key alone yields the schema version, so the software
    version must be anchored on the startup ``PROGRESS:`` banner.
    """
    assert extract_params(FIXTURE).software_version == "1.10.3"


def test_overridden_entries_take_the_applied_value_not_the_default() -> None:
    """``enzyme: trypsin/p [user defined, default: trypsin]`` applied trypsin/p."""
    params = extract_params(FIXTURE)

    assert params.enzyme == "Trypsin/P"
    assert params.max_mods == 1  # "1 [user defined, default: 2]"


def test_tolerances_are_read_as_ppm() -> None:
    params = extract_params(FIXTURE)

    assert params.precursor_mass_tolerance is not None
    assert params.precursor_mass_tolerance.value == 10.0
    assert params.precursor_mass_tolerance.unit == "ppm"
    assert params.fragment_mass_tolerance is not None
    assert params.fragment_mass_tolerance.value == 15.0


def test_fdr_populates_psm_and_protein() -> None:
    params = extract_params(FIXTURE)

    assert params.ident_fdr_psm is not None
    assert params.ident_fdr_psm.value == pytest.approx(0.01)
    assert params.ident_fdr_protein is not None
    assert params.ident_fdr_protein.value == pytest.approx(0.01)


def test_modifications_render_in_proforma_like_notation() -> None:
    params = extract_params(FIXTURE)

    assert [m.name for m in params.fixed_mods] == ["C[Carbamidomethyl]"]
    assert [m.name for m in params.variable_mods] == [
        "M[Oxidation]",
        "Protein N-term[Acetyl]",
    ]


def test_a_zero_tolerance_records_automatic_calibration(tmp_path: Path) -> None:
    """AlphaDIA writes ``0`` when it calibrated the tolerance from the data."""
    log = tmp_path / "log.txt"
    log.write_text(
        "0:00:00.0 PROGRESS: version: 1.12.1\n"
        "0:00:00.1 INFO: │   ├──target_ms1_tolerance: 0\n"
        "0:00:00.1 INFO: │   ├──target_ms2_tolerance: 0\n",
        encoding="utf-8",
    )

    params = extract_params(log)

    assert params.precursor_mass_tolerance is not None
    assert params.precursor_mass_tolerance.mode == "automatic"
    assert params.precursor_mass_tolerance.value is None


def test_a_malformed_tolerance_is_not_reclassified_as_missing(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text(
        "0:00:00.0 PROGRESS: version: 1.12.1\n"
        "0:00:00.1 INFO: │   ├──target_ms1_tolerance: not-a-number\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be numeric"):
        extract_params(log)


def test_a_non_alphadia_file_raises_params_error(tmp_path: Path) -> None:
    other = tmp_path / "not-alphadia.txt"
    other.write_text("some other vendor's parameter file\nkey: value\n", encoding="utf-8")

    with pytest.raises(ParamsError, match="not an AlphaDIA run log"):
        extract_params(other)
