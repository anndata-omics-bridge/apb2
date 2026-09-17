"""Coherent related MaxQuant tables, including fan-out and unmatched measurements."""

from __future__ import annotations

import polars as pl


def maxquant_tables() -> dict[str, pl.DataFrame]:
    evidence = pl.DataFrame(
        {
            "id": ["0", "1", "2"],
            "Raw file": ["raw1", "raw1", "raw2"],
            "Experiment": ["A", "A", "A"],
            "Modified sequence": ["_PEPTIDE_"] * 3,
            "Sequence": ["PEPTIDE"] * 3,
            "Modifications": ["Unmodified"] * 3,
            "Charge": ["2"] * 3,
            "Mass": ["800"] * 3,
            "Proteins": ["P;Q"] * 3,
            "Intensity": ["7", "13", "5"],
            "Protein group IDs": ["0;1"] * 3,
            "Peptide ID": ["0"] * 3,
            "Mod. peptide ID": ["0"] * 3,
        }
    )
    return {
        "evidence": evidence,
        "peptidoform": pl.DataFrame(
            {
                "id": ["0"],
                "Sequence": ["PEPTIDE"],
                "Modifications": ["Unmodified"],
                "Proteins": ["P;Q"],
                "Evidence IDs": ["0;1;2"],
                "Intensity A": ["25"],
            }
        ),
        "peptide": pl.DataFrame(
            {
                "id": ["0"],
                "Sequence": ["PEPTIDE"],
                "Proteins": ["P;Q"],
                "Leading razor protein": ["P"],
                "Mod. peptide IDs": ["0"],
                "Evidence IDs": ["0;1;2"],
                "Intensity A": ["25"],
            }
        ),
        "protein": pl.DataFrame(
            {
                "id": ["0", "1", "2"],
                "Protein IDs": ["P", "Q", "UNMATCHED"],
                "Peptide IDs": ["0", "0", None],
                "Evidence IDs": ["0;1;2", "0;1;2", None],
                "Intensity A": ["100", "200", "300"],
                "Intensity B": ["101", None, "301"],
                "LFQ intensity A": ["110", "210", "310"],
                "LFQ intensity B": ["111", None, "311"],
            }
        ),
    }
