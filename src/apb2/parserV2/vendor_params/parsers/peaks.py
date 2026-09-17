"""PEAKS parameter-file parser (text report)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.common import (
    mapped_modifications,
    read_lines,
    tolerance_from_text,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
)


def _setting(lines: list[str], *terms: str) -> str | None:
    """Read the first stated spelling, matching the whole label rather than a substring."""
    for term in terms:
        for line in lines:
            label, separator, value = line.partition(":")
            if separator and label.casefold() == term.removesuffix(":").casefold():
                return value.strip()
    return None


_Source = Path | IO[bytes] | IO[str]

# PEAKS modification tokens -> ProForma-style names (ports ProteoBench's
# ``peaks.MODIFICATION_MAPPING``). Applied via ``MAP.get(mod, mod)`` so
# unrecognized modifications pass through unchanged.
_MODIFICATION_MAPPING = {
    "Carbamidomethylation": "C[Carbamidomethyl]",
    "Carbamidomethylation (+57.02)": "C[Carbamidomethyl]",
    "Oxidation (M)": "M[Oxidation]",
    "Oxidation (M) (+15.99)": "M[Oxidation]",
    "Acetylation (Protein N-term)": "Protein N-term[Acetylation]",
    "Acetylation (Protein N-term) (+42.01)": "Protein N-term[Acetylation]",
}


def _mass_tolerance(lines: list[str], term: str) -> MassTolerance | None:
    """Read a PEAKS tolerance, resolving its ``System Default`` to the documented 40 ppm."""
    raw = _setting(lines, term)
    if raw is None or not raw.strip():
        return None
    return tolerance_from_text("40 ppm" if raw == "System Default" else raw)


def _fdr(lines: list[str], *terms: str) -> Probability | None:
    """Read an identification FDR; a percent sign in its label also defines its unit."""
    for term in terms:
        raw = _setting(lines, term)
        if raw is not None:
            try:
                value = float(raw.removesuffix("%"))
                return Probability(value=value / 100 if "%" in term or "%" in raw else value)
            except ValueError as error:
                raise ValueError(f"PEAKS setting {term!r} has invalid FDR {raw!r}") from error
    return None


def _modification_list(lines: list[str], term: str) -> list[str]:
    """Read the last modification list, stopping before the next non-list setting.

    PEAKS repeats modification blocks. Database and LFQ lists are not modifications.
    """
    capturing = False
    items: list[str] = []
    for line in lines:
        if line == term:
            capturing = True
            items = []
        elif capturing and line.startswith("- "):
            items.append(line[2:].strip())
        elif line:
            capturing = False
    return items


def _integer_setting(lines: list[str], *terms: str) -> int | None:
    """Read an optional integer without treating malformed stated values as missing."""
    raw = _setting(lines, *terms)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"PEAKS setting {terms!r} requires an integer, got {raw!r}") from error


def _range_setting(lines: list[str], *terms: str) -> tuple[int | None, int | None]:
    """Read comma, hyphen, or 'to' bounds, optionally enclosed in brackets."""
    raw = _setting(lines, *terms)
    if raw is None:
        return None, None
    text = raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw
    match = re.fullmatch(r"\s*(\d+)\s*(?:,|-|to)\s*(\d+)\s*", text)
    if match is None:
        raise ValueError(f"PEAKS setting {terms!r} requires an integer range, got {raw!r}")
    return int(match[1]), int(match[2])


def _charge_range(lines: list[str]) -> tuple[int | None, int | None]:
    """Prefer precursor bounds, then the older or compact LFQ charge filter."""
    bounds = _range_setting(lines, "Precursor Charge between:", "Charge between:")
    if bounds[0] is not None:
        return bounds
    raw = _setting(lines, "Peptide Feature:")
    if raw is None or "charge" not in raw:
        return None, None
    match = re.search(r"(?:^|,)\s*(\d+)\s*<=\s*charge\s*<=\s*(\d+)\s*(?:,|$)", raw)
    if match is None:
        raise ValueError(f"PEAKS setting 'Peptide Feature:' has invalid charge bounds: {raw!r}")
    return int(match[1]), int(match[2])


def _match_between_runs(lines: list[str]) -> bool | None:
    """Preserve absence; Increased Sensitivity does not state match-between-runs."""
    raw = _setting(lines, "Match Between Run:")
    if raw is None:
        return None
    if raw not in {"Yes", "No"}:
        raise ValueError(f"PEAKS setting 'Match Between Run:' requires Yes or No, got {raw!r}")
    return raw == "Yes"


def extract_params(source: _Source) -> Parameters:
    """Parse a PEAKS settings text file into :class:`Parameters`.

    Extracts metadata only; these settings never filter the exported measurements.
    """
    lines = read_lines(source, strip=True)

    version = _setting(lines, "PEAKS Version:")
    peptide_range = _range_setting(
        lines, "Peptide Length between:", "Peptide Length Range:", "Peptide Length:"
    )
    charge_range = _charge_range(lines)
    digest_mode = _setting(lines, "Digest Mode:")

    min_prec_mz = max_prec_mz = min_frag_mz = max_frag_mz = None
    prec_mz_raw = _setting(lines, "Precursor M/Z between:")
    if prec_mz_raw is not None:
        prec_mz = prec_mz_raw.split(",")
        min_prec_mz, max_prec_mz = int(prec_mz[0]), int(prec_mz[1])
        frag_mz_raw = _setting(lines, "Fragment M/Z between:")
        if frag_mz_raw is not None:
            frag_mz = frag_mz_raw.split(",")
            min_frag_mz, max_frag_mz = int(frag_mz[0]), int(frag_mz[1])

    fixed = _modification_list(lines, "Fixed Modifications:")
    variable = _modification_list(lines, "Variable Modifications:")

    return Parameters(
        software_name="PEAKS",
        software_version=version,
        search_engine="PEAKS",
        search_engine_version=version,
        ident_fdr_psm=_fdr(lines, "Precursor FDR:", "Precursor FDR(%):", "PSM FDR:"),
        ident_fdr_peptide=_fdr(lines, "Peptide FDR:", "Peptide FDR(%):"),
        ident_fdr_protein=_fdr(lines, "Protein Group FDR:", "Protein FDR(%):"),
        enable_match_between_runs=_match_between_runs(lines),
        precursor_mass_tolerance=_mass_tolerance(lines, "Precursor Mass Error Tolerance:"),
        fragment_mass_tolerance=_mass_tolerance(lines, "Fragment Mass Error Tolerance:"),
        enzyme=_setting(lines, "Enzyme:"),
        semi_enzymatic=None if digest_mode is None else digest_mode != "Specific",
        allowed_miscleavages=_integer_setting(lines, "Max Missed Cleavage:", "Missed Cleavage:"),
        min_peptide_length=peptide_range[0],
        max_peptide_length=peptide_range[1],
        fixed_mods=mapped_modifications(fixed, _MODIFICATION_MAPPING, ModType.fixed),
        variable_mods=mapped_modifications(variable, _MODIFICATION_MAPPING, ModType.variable),
        max_mods=_integer_setting(lines, "Max Variable PTM per Peptide:"),
        min_precursor_charge=charge_range[0],
        max_precursor_charge=charge_range[1],
        min_precursor_mz=min_prec_mz,
        max_precursor_mz=max_prec_mz,
        min_fragment_mz=min_frag_mz,
        max_fragment_mz=max_frag_mz,
        quantification_method=_setting(lines, "LFQ Method:", "Q Method:"),
        abundance_normalization_ions=_setting(lines, "Normalization Method:", "Normalization:"),
    )
