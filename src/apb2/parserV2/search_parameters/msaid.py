"""MSAID parameter-file parser (CSV)."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from apb2.parserV2.search_parameters.common import (
    mapped_modifications,
    split_modifications,
    tolerance_from_text,
)
from apb2.parserV2.search_parameters.model import ModType, Parameters, Probability

# Map MSAID modification strings to ProForma-like notation.
MODIFICATION_MAPPING = {
    "Carbamidomethyl (C)": "C[Carbamidomethyl]",
    "Oxidation (M)": "M[Oxidation]",
    "Acetylation (N-term)": "N-term[Acetylation]",
}

# MSAID's own FDR is not stated in the settings export; ProteoBench records the 1% it runs at.
_IDENT_FDR = Probability(value=0.01)


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse an MSAID parameter CSV into :class:`Parameters`.

    Mirrors ``proteobench.io.params.msaid.extract_params``.
    """
    df = pd.read_csv(source)
    raw: dict[str, str] = dict(df.itertuples(False, None))

    algorithm_parts = raw["Algorithm"].split(" ", 1)
    quant_method = raw["Quantification Type"]

    return Parameters(
        software_name="MSAID",
        search_engine=algorithm_parts[0],
        search_engine_version=algorithm_parts[1] if len(algorithm_parts) > 1 else None,
        ident_fdr_psm=_IDENT_FDR,
        ident_fdr_peptide=_IDENT_FDR,
        ident_fdr_protein=_IDENT_FDR,
        enable_match_between_runs="Quan in all file" in quant_method or "MBR" in quant_method,
        fragment_mass_tolerance=tolerance_from_text(raw["Fragment Mass Tolerance"]),
        enzyme=raw["Enzyme"],
        semi_enzymatic=raw["Enzyme Specificity"] != "full",
        allowed_miscleavages=int(raw["Max. Missed Cleavage Sites"]),
        min_peptide_length=int(raw["Min. Peptide Length"]),
        max_peptide_length=int(raw["Max. Peptide Length"]),
        fixed_mods=mapped_modifications(
            split_modifications(raw["Static Modifications"]), MODIFICATION_MAPPING, ModType.fixed
        ),
        variable_mods=mapped_modifications(
            split_modifications(raw["Variable Modifications"]),
            MODIFICATION_MAPPING,
            ModType.variable,
        ),
        max_mods=int(raw["Maximum Number of Modifications"]),
        min_precursor_charge=int(raw["Min. Peptide Charge"]),
        max_precursor_charge=int(raw["Max. Peptide Charge"]),
        quantification_method=quant_method,
    )
