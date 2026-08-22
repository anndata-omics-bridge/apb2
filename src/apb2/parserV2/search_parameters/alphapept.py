"""AlphaPept parameter-file parser (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import yaml
from pydantic import BaseModel, ConfigDict

from apb2.parserV2.search_parameters.common import modifications, read_text, tolerance_unit
from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)

MODIFICATION_MAPPING = {
    "cC": "C[Carbamidomethyl]",
    "oxM": "M[Oxidation]",
    "a<^": "N-term[Acetyl]",
}


class _AlphaPeptModel(BaseModel):
    """Strict-enough base for the consumed AlphaPept YAML sections."""

    model_config = ConfigDict(extra="ignore")


class _Summary(_AlphaPeptModel):
    version: str


class _Fasta(_AlphaPeptModel):
    protease: str
    mods_fixed: list[str]
    mods_fixed_terminal: list[str]
    mods_fixed_terminal_prot: list[str]
    mods_variable: list[str]
    mods_variable_terminal: list[str]
    mods_variable_terminal_prot: list[str]
    n_missed_cleavages: int
    n_modifications_max: int
    pep_length_min: int
    pep_length_max: int


class _Search(_AlphaPeptModel):
    ppm: bool
    prec_tol: float
    frag_tol: float
    protein_fdr: float
    peptide_fdr: float


class _Features(_AlphaPeptModel):
    iso_charge_min: int
    iso_charge_max: int


class _Workflow(_AlphaPeptModel):
    match: bool


class _AlphaPeptDocument(_AlphaPeptModel):
    summary: _Summary
    fasta: _Fasta
    search: _Search
    features: _Features
    workflow: _Workflow


def _map_modifications(declared: list[str], mod_type: ModType) -> list[SearchedModification]:
    """Resolve validated AlphaPept modification names."""
    return modifications(
        (MODIFICATION_MAPPING.get(name.strip(), name.strip()) for name in declared),
        mod_type,
    )


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse an AlphaPept YAML configuration file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.alphapept.extract_params``.
    """
    record = _AlphaPeptDocument.model_validate(yaml.safe_load(read_text(source)))
    summary = record.summary
    fasta = record.fasta
    search = record.search
    features = record.features
    workflow = record.workflow

    unit = tolerance_unit("ppm" if search.ppm else "Da")

    return Parameters(
        software_name="AlphaPept",
        software_version=summary.version,
        search_engine="AlphaPept",
        search_engine_version=summary.version,
        enzyme="Trypsin" if fasta.protease == "trypsin" else fasta.protease,
        allowed_miscleavages=fasta.n_missed_cleavages,
        fixed_mods=_map_modifications(
            fasta.mods_fixed + fasta.mods_fixed_terminal + fasta.mods_fixed_terminal_prot,
            ModType.fixed,
        ),
        variable_mods=_map_modifications(
            fasta.mods_variable + fasta.mods_variable_terminal + fasta.mods_variable_terminal_prot,
            ModType.variable,
        ),
        max_mods=fasta.n_modifications_max,
        min_peptide_length=fasta.pep_length_min,
        max_peptide_length=fasta.pep_length_max,
        precursor_mass_tolerance=MassTolerance(mode="absolute", value=search.prec_tol, unit=unit),
        fragment_mass_tolerance=MassTolerance(mode="absolute", value=search.frag_tol, unit=unit),
        ident_fdr_protein=Probability(value=search.protein_fdr),
        ident_fdr_psm=Probability(value=search.peptide_fdr),
        min_precursor_charge=features.iso_charge_min,
        max_precursor_charge=features.iso_charge_max,
        enable_match_between_runs=workflow.match,
    )
