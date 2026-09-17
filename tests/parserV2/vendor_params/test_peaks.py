"""PEAKS parser equivalence tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.peaks import extract_params
from parserV2.vendor_params import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    "PEAKS_parameters.txt",
    "PEAKS_parameters_DDA.txt",
    "PEAKS_parameters_DIA.txt",
    "PEAKS_parameters_DDA_new.txt",
    "PEAKS_diaPASEF.txt",
]


def test_peaks_reads_astral_report_settings() -> None:
    """The newer report labels populate metadata without inventing absent settings."""
    params = extract_params(PROTEOBENCH_PARAMS / "PEAKS_astral_report.txt")

    assert params.software_name == "PEAKS"
    assert params.software_version is None
    assert params.search_engine_version is None
    assert params.enable_match_between_runs is None
    assert params.enzyme == "Trypsin/P"
    assert params.semi_enzymatic is False
    assert params.allowed_miscleavages == 1
    assert (params.min_peptide_length, params.max_peptide_length) == (6, 30)
    assert (params.min_precursor_charge, params.max_precursor_charge) == (1, 5)
    assert params.max_mods == 1
    assert params.precursor_mass_tolerance is not None
    assert params.precursor_mass_tolerance.value == 3
    assert params.precursor_mass_tolerance.unit == "ppm"
    assert params.fragment_mass_tolerance is not None
    assert params.fragment_mass_tolerance.value == 10
    assert params.fragment_mass_tolerance.unit == "ppm"
    assert params.ident_fdr_psm is not None
    assert params.ident_fdr_psm.value == 0.01
    assert params.ident_fdr_peptide is not None
    assert params.ident_fdr_peptide.value == 0.01
    assert params.ident_fdr_protein is not None
    assert params.ident_fdr_protein.value == 0.01
    assert [mod.accession for mod in params.fixed_mods] == ["UNIMOD:4"]
    assert [mod.accession for mod in params.variable_mods] == ["UNIMOD:35"]
    assert params.quantification_method == "ID-directed LFQ"
    assert params.abundance_normalization_ions == "TIC"


@pytest.mark.parametrize("txt_name", CASES)
def test_peaks_matches_proteobench(txt_name: str) -> None:
    txt = PROTEOBENCH_PARAMS / txt_name
    csv = txt.with_suffix(".csv")
    if not txt.exists() or not csv.exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(txt)
    expected = proteobench_params.expected_csv(csv)
    if txt_name in {"PEAKS_parameters_DDA.txt", "PEAKS_parameters_DDA_new.txt"}:
        # These reports never state MBR. The upstream oracle inferred False from absence.
        assert "Match Between Run:" not in txt.read_text(encoding="utf-8")
        expected["enable_match_between_runs"] = None

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


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ("Peptide Length: six to 30", "integer range"),
        ("Peptide Length: 6 to 30 extra", "integer range"),
        ("Peptide Length: [6 - 30", "integer range"),
        ("Peptide Length: 30 to 6", "minimum peptide length"),
        ("Missed Cleavage: many", "integer"),
        ("Missed Cleavage: -1", "allowed_miscleavages"),
        ("Max Variable PTM Per Peptide: 1.5", "integer"),
        ("Peptide Feature: 1<=charge<=five", "invalid charge bounds"),
        ("Peptide Feature: 1<=charge<=5.5", "invalid charge bounds"),
        ("Peptide Feature: 5<=charge<=1", "minimum charge"),
        ("Peptide Feature: 0<=charge<=5", "min_precursor_charge"),
        ("Precursor Charge between: unknown\nPeptide Feature: 1<=charge<=5", "integer range"),
        ("Protein FDR(%): many", "invalid FDR"),
        ("Protein FDR(%): 101", "invalid FDR"),
        ("Protein FDR(%): nan", "invalid FDR"),
        ("Protein FDR(%):", "invalid FDR"),
        ("Match Between Run: maybe", "requires Yes or No"),
    ],
)
def test_peaks_rejects_malformed_stated_settings(settings: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_params(StringIO(settings))


def test_peaks_reads_diapasef_report_without_peptide_or_protein_fdr() -> None:
    params = extract_params(PROTEOBENCH_PARAMS / "PEAKS_diapasef_report.txt")
    assert params.precursor_mass_tolerance is not None
    assert params.precursor_mass_tolerance.value == 15
    assert params.fragment_mass_tolerance is not None
    assert params.fragment_mass_tolerance.value == 15
    assert params.ident_fdr_psm is not None
    assert params.ident_fdr_psm.value == 0.01
    assert params.ident_fdr_peptide is None
    assert params.ident_fdr_protein is None
    assert (params.min_precursor_charge, params.max_precursor_charge) == (1, 5)
    assert [mod.accession for mod in params.fixed_mods] == ["UNIMOD:4"]
    assert [mod.accession for mod in params.variable_mods] == ["UNIMOD:35"]


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ("", {"software_version": None, "semi_enzymatic": None}),
        ("Increased Sensitivity: Enabled", {"enable_match_between_runs": None}),
        ("Match Between Run: No", {"enable_match_between_runs": False}),
        ("Match Between Run: Yes", {"enable_match_between_runs": True}),
        ("Protein FDR(%): 0", {"ident_fdr_protein": {"value": 0.0}}),
        ("Protein FDR(%): 1.00%", {"ident_fdr_protein": {"value": 0.01}}),
        ("Protein Group FDR: 0.01", {"ident_fdr_protein": {"value": 0.01}}),
        (
            "Proteins FDR(%): 0.9720\nFDR Threshold(%): 0.0",
            {"ident_fdr_protein": None, "ident_fdr_peptide": None, "ident_fdr_psm": None},
        ),
        (
            "Proteins FDR(%): 0.9720\nProtein FDR(%): 1.00\nFDR Threshold(%): 0.0",
            {"ident_fdr_protein": {"value": 0.01}},
        ),
        (
            "Precursor FDR: 0.0%\nPSM FDR: 1.0%",
            {"ident_fdr_psm": {"value": 0.0}},
        ),
        (
            "Peptide Feature: 1 <= charge <= 5\nPrecursor Charge between: 2,4",
            {"min_precursor_charge": 2, "max_precursor_charge": 4},
        ),
        (
            "Peptide Feature: 1<=charge<=5\nCharge between: [2 - 4]",
            {"min_precursor_charge": 2, "max_precursor_charge": 4},
        ),
    ],
)
def test_peaks_setting_units_absence_and_precedence(
    settings: str, expected: dict[str, object]
) -> None:
    parsed = extract_params(StringIO(settings)).model_dump(mode="json")
    assert {key: parsed[key] for key in expected} == expected


def test_peaks_modification_lists_stop_at_the_next_setting() -> None:
    text = """Fixed Modifications:
- Carbamidomethylation (+57.02)
Variable Modifications:
- Oxidation (M) (+15.99)
Database:
- Not a modification
Fixed Modifications:
Variable Modifications:
- Acetylation (Protein N-term)
Max Variable PTM Per Peptide: 1
Other List:
- Also not a modification
"""
    params = extract_params(StringIO(text))
    assert params.fixed_mods == []
    assert [mod.accession for mod in params.variable_mods] == ["UNIMOD:1"]
    old = extract_params(
        StringIO("Variable Modifications:\n- Acetylation (Protein N-term) (+42.01)")
    )
    assert params.variable_mods == old.variable_mods


# --- a fixture pinned against ProteoBench's current runtime, not a checked-in CSV -----------
#
# PEAKS_FDR_log.txt ships no expected CSV upstream. Its expected record was generated by
# running ProteoBench's own extract_params, so the oracle stays independent of this parser.

RUNTIME_CASES = [
    ("PEAKS_FDR_log.txt", "PEAKS_FDR_log_sel.json"),
]


@pytest.mark.parametrize(("txt_name", "expected_name"), RUNTIME_CASES)
def test_peaks_matches_proteobench_runtime(txt_name: str, expected_name: str) -> None:
    txt = PROTEOBENCH_PARAMS / txt_name
    expected_path = PROTEOBENCH_PARAMS / expected_name
    if not txt.exists() or not expected_path.exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(txt)
    expected = proteobench_params.expected_json(expected_path)

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
        "min_precursor_charge",
        "max_precursor_charge",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, f"{txt_name}: " + "; ".join(mismatches)
