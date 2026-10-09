"""Focused parameter parsing for the four newly supported quantification tools."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from apb2.parserV2.vendor_params.parsers.i2masschroq import extract_params
from apb2.parserV2.vendor_params.parsers.shared.model import MassTolerance
from apb2.parserV2.vendor_params.registry import parse_params

DATA = Path(__file__).resolve().parents[1] / "data"


def test_i2masschroq_parameters_are_typed() -> None:
    result = parse_params(DATA / "i2masschroq" / "param_0..txt", "i2masschroq")

    assert result.software_version == "1.0.18"
    assert result.search_engine == "X!Tandem"
    assert result.allowed_miscleavages == 2
    assert result.precursor_mass_tolerance is not None
    assert result.precursor_mass_tolerance.value == 10


def test_i2masschroq_xtandem_fragment_tolerance_reads_its_own_unit() -> None:
    excerpt = (
        "i2MassChroQ_VERSION\t9.9.9\n"
        "AnalysisSoftware_name\tX!Tandem\n"
        "spectrum, parent monoisotopic mass error minus\t15\n"
        "spectrum, parent monoisotopic mass error plus\t15\n"
        "spectrum, parent monoisotopic mass error units\tppm\n"
        "spectrum, fragment monoisotopic mass error\t0.5\n"
        "spectrum, fragment monoisotopic mass error units\tDaltons\n"
        "scoring, maximum missed cleavage sites\t1\n"
        "spectrum, maximum parent charge\t4\n"
    )

    result = extract_params(StringIO(excerpt))

    assert result.precursor_mass_tolerance == MassTolerance(mode="absolute", value=15, unit="ppm")
    assert result.fragment_mass_tolerance == MassTolerance(mode="absolute", value=0.5, unit="Da")


def test_quantms_versions_identify_the_workflow_and_engine() -> None:
    result = parse_params(DATA / "quantms" / "param_0..yml", "quantms")

    assert result.software_version == "v1.3.1dev-g70337bc"
    assert result.search_engine == "comet"
    assert result.search_engine_version == "2023.01 rev. 2"


def test_prolinestudio_reads_parameter_sheets_from_its_workbook() -> None:
    result = parse_params(DATA / "prolinestudio" / "sample.txt", "prolinestudio")

    assert result.software_version == "2.3.0"
    assert result.search_engine == "Mascot"
    assert result.min_peptide_length == 7
    assert result.enable_match_between_runs is True


def test_msangel_reads_the_mascot_workflow() -> None:
    result = parse_params(DATA / "msangel" / "param_0..json", "msangel")

    assert result.software_version == "2.2.10"
    assert result.search_engine == "Mascot"
    assert result.allowed_miscleavages == 2
    assert result.min_precursor_charge == 2
    assert result.max_precursor_charge == 4
