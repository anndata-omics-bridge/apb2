"""PEAKS parameter-file parser (text report)."""

from __future__ import annotations

from pathlib import Path
from typing import IO

from apb2.parserV2.search_parameters.common import (
    mapped_modifications,
    read_lines,
    required_settings_value,
    settings_value,
    tolerance_from_text,
)
from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
)


def _required(lines: list[str], term: str) -> str:
    """Read a PEAKS setting the parser cannot proceed without."""
    return required_settings_value(lines, term, software="PEAKS")


_Source = Path | IO[bytes] | IO[str]

# PEAKS modification tokens -> ProForma-style names (ports ProteoBench's
# ``peaks.MODIFICATION_MAPPING``). Applied via ``MAP.get(mod, mod)`` so
# unrecognized modifications pass through unchanged.
_MODIFICATION_MAPPING = {
    "Carbamidomethylation (+57.02)": "C[Carbamidomethyl]",
    "Oxidation (M) (+15.99)": "M[Oxidation]",
    "Acetylation (Protein N-term) (+42.01)": "Protein N-term[Acetylation]",
}


def _mass_tolerance(lines: list[str], term: str) -> MassTolerance | None:
    """Read a PEAKS tolerance, resolving its ``System Default`` to the documented 40 ppm."""
    raw = settings_value(lines, term)
    if raw is None or not raw.strip():
        return None
    return tolerance_from_text("40 ppm" if raw == "System Default" else raw)


def _fdr(lines: list[str], term: str) -> Probability | None:
    """Read an FDR PEAKS states as a percentage (``1.0%``)."""
    raw = settings_value(lines, term)
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    return Probability(value=float(text.removesuffix("%")) / 100 if "%" in text else float(text))


def _between(lines: list[str], start: str, end: str) -> list[str]:
    """Pick the ``- value`` items of the last ``start``..``end`` block in the file.

    PEAKS repeats its modification blocks per sample; the final block is the one that applied.
    """
    capturing = False
    items: list[str] = []
    pending: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith(start):
            capturing = True
            pending = []
            continue
        if capturing and line.startswith(end):
            capturing = False
            items = pending[:]
            pending = []
        if capturing and line.startswith("- "):
            pending.append(line[2:].strip())
    return pending if capturing else items


def extract_params(source: _Source) -> Parameters:
    """Parse a PEAKS settings text file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.peaks.extract_params``.
    """
    lines = read_lines(source, strip=True)

    version = settings_value(lines, "PEAKS Version:")
    psm_fdr = _fdr(lines, "Precursor FDR:") or _fdr(lines, "PSM FDR:")

    peptide_range_raw = settings_value(lines, "Peptide Length between:")
    peptide_range = (
        peptide_range_raw.split(",")
        if peptide_range_raw is not None
        else _required(lines, "Peptide Length Range:").split(" - ")
    )

    charge_range_raw = settings_value(lines, "Precursor Charge between:")
    charge_range = (
        charge_range_raw.split(",")
        if charge_range_raw is not None
        else _required(lines, "Charge between:").replace("[", "").replace("]", "").split(" - ")
    )

    min_prec_mz = max_prec_mz = min_frag_mz = max_frag_mz = None
    prec_mz_raw = settings_value(lines, "Precursor M/Z between:")
    if prec_mz_raw is not None:
        prec_mz = prec_mz_raw.split(",")
        min_prec_mz, max_prec_mz = int(prec_mz[0]), int(prec_mz[1])
        frag_mz_raw = settings_value(lines, "Fragment M/Z between:")
        if frag_mz_raw is not None:
            frag_mz = frag_mz_raw.split(",")
            min_frag_mz, max_frag_mz = int(frag_mz[0]), int(frag_mz[1])

    fixed = _between(lines, "Fixed Modifications:", "Variable Modifications:")
    variable = _between(lines, "Variable Modifications:", "Database:")

    return Parameters(
        software_name="PEAKS",
        software_version=version,
        search_engine="PEAKS",
        search_engine_version=version,
        ident_fdr_psm=psm_fdr,
        ident_fdr_peptide=_fdr(lines, "Peptide FDR:"),
        ident_fdr_protein=_fdr(lines, "Protein Group FDR:"),
        enable_match_between_runs=settings_value(lines, "Match Between Run:") == "Yes",
        precursor_mass_tolerance=_mass_tolerance(lines, "Precursor Mass Error Tolerance:"),
        fragment_mass_tolerance=_mass_tolerance(lines, "Fragment Mass Error Tolerance:"),
        enzyme=settings_value(lines, "Enzyme:"),
        semi_enzymatic=settings_value(lines, "Digest Mode:") != "Specific",
        allowed_miscleavages=int(_required(lines, "Max Missed Cleavage:")),
        min_peptide_length=int(peptide_range[0]),
        max_peptide_length=int(peptide_range[1]),
        fixed_mods=mapped_modifications(fixed, _MODIFICATION_MAPPING, ModType.fixed),
        variable_mods=mapped_modifications(variable, _MODIFICATION_MAPPING, ModType.variable),
        max_mods=int(_required(lines, "Max Variable PTM per Peptide:")),
        min_precursor_charge=int(charge_range[0]),
        max_precursor_charge=int(charge_range[1]),
        min_precursor_mz=min_prec_mz,
        max_precursor_mz=max_prec_mz,
        min_fragment_mz=min_frag_mz,
        max_fragment_mz=max_frag_mz,
        quantification_method=settings_value(lines, "LFQ Method:"),
        abundance_normalization_ions=settings_value(lines, "Normalization Method:"),
    )
