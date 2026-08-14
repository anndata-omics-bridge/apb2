"""PEAKS parameter-file parser (text report)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.vendor_params.model import Parameters
from apb2.vendor_params.parsers._common import read_lines

_Source = Path | IO[bytes] | IO[str]

# PEAKS modification tokens -> ProForma-style names (ports ProteoBench's
# ``peaks.MODIFICATION_MAPPING``). Applied via ``MAP.get(mod, mod)`` so
# unrecognized modifications pass through unchanged.
_MODIFICATION_MAPPING = {
    "Carbamidomethylation (+57.02)": "C[Carbamidomethyl]",
    "Oxidation (M) (+15.99)": "M[Oxidation]",
    "Acetylation (Protein N-term) (+42.01)": "Protein N-term[Acetylation]",
}


def _clean(text: str) -> str:
    return re.sub(r"^[\s:,\t]+|[\s:,\t]+$", "", text)


def _value(lines: list[str], term: str) -> str | None:
    for line in lines:
        if term in line:
            return _clean(line.split(term, 1)[1])
    return None


def _required_value(lines: list[str], term: str) -> str:
    """Return a required PEAKS setting."""
    value = _value(lines, term)
    if value is None:
        raise ValueError(f"PEAKS setting {term!r} is missing")
    return value


def _mass_tolerance(lines: list[str], term: str) -> str | None:
    raw = _value(lines, term)
    return "40 ppm" if raw == "System Default" else raw


def _fdr(lines: list[str], term: str) -> str | None:
    """Extract an FDR value, dropping any trailing ``%`` (e.g. ``1.0%`` -> ``1.0``)."""
    raw = _value(lines, term)
    return raw.replace("%", "").strip() if raw else raw


def _between(lines: list[str], start: str, end: str, only_last: bool = False) -> list[str]:
    """Pick ``- value`` items between ``start`` and ``end`` block markers."""
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
            if only_last:
                items = pending[:]
            else:
                items.extend(pending)
            pending = []
        if capturing and line.startswith("- "):
            pending.append(line[2:].strip())
    if only_last and capturing:
        items = pending
    return items


def extract_params(source: _Source) -> Parameters:
    """Parse a PEAKS settings text file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.peaks.extract_params``.
    """
    lines = read_lines(source, strip=True)

    version = _value(lines, "PEAKS Version:")
    psm_fdr = _fdr(lines, "Precursor FDR:") or _fdr(lines, "PSM FDR:")

    peptide_range_raw = _value(lines, "Peptide Length between:")
    peptide_range = (
        peptide_range_raw.split(",")
        if peptide_range_raw is not None
        else _required_value(lines, "Peptide Length Range:").split(" - ")
    )

    charge_range_raw = _value(lines, "Precursor Charge between:")
    charge_range = (
        charge_range_raw.split(",")
        if charge_range_raw is not None
        else _required_value(lines, "Charge between:")
        .replace("[", "")
        .replace("]", "")
        .split(" - ")
    )

    min_prec_mz = max_prec_mz = min_frag_mz = max_frag_mz = None
    prec_mz_raw = _value(lines, "Precursor M/Z between:")
    if prec_mz_raw is not None:
        prec_mz = prec_mz_raw.split(",")
        min_prec_mz, max_prec_mz = int(prec_mz[0]), int(prec_mz[1])
        frag_mz_raw = _value(lines, "Fragment M/Z between:")
        if frag_mz_raw is not None:
            frag_mz = frag_mz_raw.split(",")
            min_frag_mz, max_frag_mz = int(frag_mz[0]), int(frag_mz[1])

    fixed = _between(lines, "Fixed Modifications:", "Variable Modifications:", only_last=True)
    variable = _between(lines, "Variable Modifications:", "Database:", only_last=True)

    return Parameters.model_validate(
        {
            "software_name": "PEAKS",
            "software_version": version,
            "search_engine": "PEAKS",
            "search_engine_version": version,
            "ident_fdr_psm": psm_fdr,
            "ident_fdr_peptide": _fdr(lines, "Peptide FDR:"),
            "ident_fdr_protein": _fdr(lines, "Protein Group FDR:"),
            "enable_match_between_runs": _value(lines, "Match Between Run:") == "Yes",
            "precursor_mass_tolerance": _mass_tolerance(lines, "Precursor Mass Error Tolerance:"),
            "fragment_mass_tolerance": _mass_tolerance(lines, "Fragment Mass Error Tolerance:"),
            "enzyme": _value(lines, "Enzyme:"),
            "semi_enzymatic": _value(lines, "Digest Mode:") != "Specific",
            "allowed_miscleavages": int(_required_value(lines, "Max Missed Cleavage:")),
            "min_peptide_length": int(peptide_range[0]),
            "max_peptide_length": int(peptide_range[1]),
            "fixed_mods": ", ".join(_MODIFICATION_MAPPING.get(m.strip(), m.strip()) for m in fixed),
            "variable_mods": ", ".join(
                _MODIFICATION_MAPPING.get(m.strip(), m.strip()) for m in variable
            ),
            "max_mods": int(_required_value(lines, "Max Variable PTM per Peptide:")),
            "min_precursor_charge": int(charge_range[0]),
            "max_precursor_charge": int(charge_range[1]),
            "min_precursor_mz": min_prec_mz,
            "max_precursor_mz": max_prec_mz,
            "min_fragment_mz": min_frag_mz,
            "max_fragment_mz": max_frag_mz,
            "quantification_method": _value(lines, "LFQ Method:"),
            "abundance_normalization_ions": _value(lines, "Normalization Method:"),
        }
    )
