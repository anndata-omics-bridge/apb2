"""MSAID parameter-file parser (CSV)."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from apb2.vendor_params.model import Parameters

# Map MSAID modification strings to ProForma-like notation.
MODIFICATION_MAPPING = {
    "Carbamidomethyl (C)": "C[Carbamidomethyl]",
    "Oxidation (M)": "M[Oxidation]",
    "Acetylation (N-term)": "N-term[Acetylation]",
}


def _homogenize_mods(raw_mods: str) -> str:
    """Map a comma-separated MSAID modification string to ProForma-like notation."""
    if not raw_mods.strip():
        return ""
    mapped = [MODIFICATION_MAPPING.get(mod.strip(), mod.strip()) for mod in raw_mods.split(",")]
    return ", ".join(mapped)


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse an MSAID parameter CSV into :class:`Parameters`.

    Mirrors ``proteobench.io.params.msaid.extract_params``.
    """
    df = pd.read_csv(source)
    raw: dict[str, str] = dict(df.itertuples(False, None))

    algorithm_parts = raw["Algorithm"].split(" ", 1)
    search_engine = algorithm_parts[0]
    search_engine_version = algorithm_parts[1] if len(algorithm_parts) > 1 else None

    fragment_tol = raw["Fragment Mass Tolerance"]
    quant_method = raw["Quantification Type"]
    mbr = "Quan in all file" in quant_method or "MBR" in quant_method

    return Parameters.model_validate(
        {
            "software_name": "MSAID",
            "software_version": None,
            "search_engine": search_engine,
            "search_engine_version": search_engine_version,
            "ident_fdr_psm": 0.01,
            "ident_fdr_peptide": 0.01,
            "ident_fdr_protein": 0.01,
            "enable_match_between_runs": mbr,
            "fragment_mass_tolerance": f"[-{fragment_tol}, {fragment_tol}]",
            "enzyme": raw["Enzyme"],
            "semi_enzymatic": raw["Enzyme Specificity"] != "full",
            "allowed_miscleavages": int(raw["Max. Missed Cleavage Sites"]),
            "min_peptide_length": int(raw["Min. Peptide Length"]),
            "max_peptide_length": int(raw["Max. Peptide Length"]),
            "fixed_mods": _homogenize_mods(raw["Static Modifications"]),
            "variable_mods": _homogenize_mods(raw["Variable Modifications"]),
            "max_mods": int(raw["Maximum Number of Modifications"]),
            "min_precursor_charge": int(raw["Min. Peptide Charge"]),
            "max_precursor_charge": int(raw["Max. Peptide Charge"]),
            "quantification_method": quant_method,
        }
    )
