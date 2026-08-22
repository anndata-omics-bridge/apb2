"""WOMBAT-P parameter-file parser (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import yaml
from pydantic import BaseModel, ConfigDict

from apb2.parserV2.search_parameters.common import (
    modifications,
    read_text,
    split_modifications,
    tolerance_from_text,
)
from apb2.parserV2.search_parameters.model import ModType, Parameters, Probability

# Longest qualifier first: "protein n-term" also contains "n-term".
_TERMINI = {
    "protein n-term": "Protein N-term",
    "protein c-term": "Protein C-term",
    "n-term": "N-term",
    "c-term": "C-term",
}


class _WombatModel(BaseModel):
    """Strict-enough base for the consumed WOMBAT-P YAML sections."""

    model_config = ConfigDict(extra="ignore")


class _WombatParameters(_WombatModel):
    enzyme: str
    miscleavages: int
    fixed_mods: str
    variable_mods: str
    max_mods: int
    min_peptide_length: int
    max_peptide_length: int
    precursor_mass_tolerance: str
    fragment_mass_tolerance: str
    ident_fdr_protein: float
    ident_fdr_peptide: float
    ident_fdr_psm: float
    min_precursor_charge: int
    max_precursor_charge: int
    enable_match_between_runs: bool
    normalization_method: str | bool


class _WombatDocument(_WombatModel):
    version: str
    params: _WombatParameters


def _homogenize_mod_xtandem(mod_str: str) -> str:
    """Convert a WOMBAT-P X!Tandem modification spec to ProForma-like notation.

    Format: ``{modname} of {residue}``, e.g. ``Oxidation of M``,
    ``Acetyl of Protein N-term``.
    """
    mod_str = mod_str.strip()
    if " of " not in mod_str:
        return mod_str
    name, residue_part = mod_str.split(" of ", 1)
    residue_part = residue_part.strip()
    lower = residue_part.lower()
    for qualifier, target in _TERMINI.items():
        if qualifier in lower:
            return f"{target}[{name}]"
    return f"{residue_part.upper()}[{name}]"


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse a WOMBAT-P YAML configuration into :class:`Parameters`.

    Mirrors ``proteobench.io.params.wombat.extract_params``.
    """
    record = _WombatDocument.model_validate(yaml.safe_load(read_text(source)))
    parameters = record.params

    return Parameters(
        software_name="Wombat",
        software_version=record.version,
        search_engine="various",
        enzyme="Trypsin" if parameters.enzyme == "trypsin" else parameters.enzyme,
        allowed_miscleavages=parameters.miscleavages,
        fixed_mods=modifications(
            (_homogenize_mod_xtandem(mod) for mod in split_modifications(parameters.fixed_mods)),
            ModType.fixed,
        ),
        variable_mods=modifications(
            (_homogenize_mod_xtandem(mod) for mod in split_modifications(parameters.variable_mods)),
            ModType.variable,
        ),
        max_mods=parameters.max_mods,
        min_peptide_length=parameters.min_peptide_length,
        max_peptide_length=parameters.max_peptide_length,
        precursor_mass_tolerance=tolerance_from_text(parameters.precursor_mass_tolerance),
        fragment_mass_tolerance=tolerance_from_text(parameters.fragment_mass_tolerance),
        ident_fdr_protein=Probability(value=parameters.ident_fdr_protein),
        ident_fdr_peptide=Probability(value=parameters.ident_fdr_peptide),
        ident_fdr_psm=Probability(value=parameters.ident_fdr_psm),
        min_precursor_charge=parameters.min_precursor_charge,
        max_precursor_charge=parameters.max_precursor_charge,
        enable_match_between_runs=parameters.enable_match_between_runs,
        abundance_normalization_ions=parameters.normalization_method,
    )
