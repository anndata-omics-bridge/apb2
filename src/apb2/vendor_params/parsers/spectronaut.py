"""Spectronaut settings-text parameter parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.vendor_params.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)
from apb2.vendor_params.parsers._common import (
    automatic_tolerance,
    homogenize_paren_mods,
    modifications,
    read_lines,
    required_settings_value,
    settings_value,
    split_modifications,
    tolerance_unit,
)

_Source = Path | IO[bytes] | IO[str]

_VENDOR_SYSTEM_MAP = {
    "Thermo": "Thermo Orbitrap",
    "Bruker": "TOF",
}
_MS1_STATIC = re.compile(r"MS1 Tolerance \(Th\):\s*(\d*)")
_MS2_STATIC = re.compile(r"MS2 Tolerance \(Th\):\s*(\d*)")
_MS1_RELATIVE = re.compile(r"MS1 Tolerance \(ppm\):\s*(\d*)")
_MS2_RELATIVE = re.compile(r"MS2 Tolerance \(ppm\):\s*(\d*)")
_MAIN_SEARCH = re.compile(r"Main Search:\s*(.*)")
# Spectronaut's calibration mode names the unit its two tolerance lines carry.
_CALIBRATION_PATTERNS = {
    "Static": ("Th", _MS1_STATIC, _MS2_STATIC),
    "Relative": ("ppm", _MS1_RELATIVE, _MS2_RELATIVE),
}

# Fallback mapping for modifications without parenthesized residue specifiers.
MODIFICATION_MAPPING = {
    "Cys-Cys": "C[Disulfide]",
    "Cysteinyl": "C[Cysteinyl]",
    "Cysteinyl - carbamidomethyl": "C[Cysteinyl + Carbamidomethyl]",
}


def _homogenize_mods(raw_mods: str | None, mod_type: ModType) -> list[SearchedModification]:
    """Map a comma-delimited ``{name} ({residues})`` string to resolved modifications."""
    if raw_mods is None:
        return []
    return modifications(
        (
            token
            for mod in split_modifications(raw_mods)
            for token in homogenize_paren_mods(mod, MODIFICATION_MAPPING)
        ),
        mod_type,
    )


def _qvalue(raw: str | None) -> Probability | None:
    """Read a Spectronaut q-value cutoff, which uses a locale decimal comma in some exports."""
    return None if raw is None else Probability(value=float(raw.replace(",", ".")))


def _anchored_value(lines: list[str], term: str) -> str | None:
    """Read a setting whose name must start the line."""
    for line in lines:
        if line.startswith(term):
            return re.sub(r"^[\s:,\t]+|[\s:,\t]+$", "", line.removeprefix(term))
    return None


def _required(lines: list[str], term: str) -> str:
    """Read a Spectronaut setting the parser cannot proceed without."""
    return required_settings_value(lines, term, software="Spectronaut")


def _extract_tolerances(
    lines: list[str],
    system: str,
) -> tuple[MassTolerance | None, MassTolerance | None]:
    """Read the MS1 and MS2 tolerances Spectronaut states under its calibration mode.

    Spectronaut nests them: the ``Pulsar Search\\Tolerances`` block, then the instrument
    system, then ``Main Search``, whose calibration mode decides which unit the two numbers
    below it carry. ``Dynamic`` means it calibrated them from the data instead.
    """
    system_lines = _tolerance_system_lines(lines, system)
    calibration, main_search_lines = _main_search_block(system_lines)
    if calibration is None:
        return None, None
    if calibration == "Dynamic":
        return automatic_tolerance(), automatic_tolerance()
    patterns = _CALIBRATION_PATTERNS.get(calibration)
    if patterns is None:
        return None, None
    unit, ms1_pattern, ms2_pattern = patterns
    ms1 = _first_regex_value(main_search_lines, ms1_pattern)
    ms2 = _first_regex_value(main_search_lines, ms2_pattern)
    if ms1 is None or ms2 is None:
        return None, None
    return (
        MassTolerance(mode="absolute", value=float(ms1), unit=tolerance_unit(unit)),
        MassTolerance(mode="absolute", value=float(ms2), unit=tolerance_unit(unit)),
    )


def _tolerance_system_lines(lines: list[str], system: str) -> list[str]:
    """Return lines following the requested system in Spectronaut's tolerance block."""
    in_tolerance_block = False
    in_system_block = False
    result: list[str] = []
    for line in lines:
        if line.startswith("Pulsar Search\\Tolerances"):
            in_tolerance_block = True
            continue
        if not in_tolerance_block:
            continue
        if line.startswith(system):
            in_system_block = True
            continue
        if in_system_block:
            result.append(line)
    return result


def _first_regex_value(lines: list[str], pattern: re.Pattern[str]) -> str | None:
    for line in lines:
        if match := pattern.search(line):
            return match.group(1).strip()
    return None


def _main_search_block(lines: list[str]) -> tuple[str | None, list[str]]:
    """Return the calibration mode and settings below ``Main Search``."""
    for index, line in enumerate(lines):
        if match := _MAIN_SEARCH.search(line):
            return match.group(1).strip(), lines[index + 1 :]
    return None, []


def extract_params(source: _Source) -> Parameters:
    """Parse a Spectronaut settings-export text file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.spectronaut.read_spectronaut_settings``.
    """
    lines = read_lines(source, strip=True)
    vendor = settings_value(lines, "Vendor:")
    if vendor not in _VENDOR_SYSTEM_MAP:
        raise ValueError(
            f"unknown Spectronaut vendor: {vendor!r}; expected one of {sorted(_VENDOR_SYSTEM_MAP)}"
        )
    system = _VENDOR_SYSTEM_MAP[vendor]

    software_version = lines[0].split()[1]

    # Strip tree-drawing characters present in some Spectronaut exports.
    lines = [re.sub(r"^[\s│├─└]*", "", line).strip() for line in lines]

    precursor_tol, fragment_tol = _extract_tolerances(lines, system)

    ident_psm = _qvalue(settings_value(lines, "Precursor Qvalue Cutoff:"))
    ident_protein = _qvalue(settings_value(lines, "Protein Qvalue Cutoff (Experiment):"))

    charge_raw = settings_value(lines, "Peptide Charge:")
    if charge_raw is None or charge_raw == "False":
        min_z = max_z = None
    else:
        min_z = max_z = int(charge_raw)

    return Parameters(
        software_name="Spectronaut",
        software_version=software_version,
        search_engine="Spectronaut",
        search_engine_version=software_version,
        ident_fdr_psm=ident_psm,
        ident_fdr_protein=ident_protein,
        enable_match_between_runs=False,
        precursor_mass_tolerance=precursor_tol,
        fragment_mass_tolerance=fragment_tol,
        enzyme=settings_value(lines, "Enzymes / Cleavage Rules:"),
        semi_enzymatic=settings_value(lines, "Digest Type:") != "Specific",
        allowed_miscleavages=int(_required(lines, "Missed Cleavages:")),
        max_peptide_length=int(_required(lines, "Max Peptide Length:")),
        min_peptide_length=int(_required(lines, "Min Peptide Length:")),
        fixed_mods=_homogenize_mods(settings_value(lines, "Fixed Modifications:"), ModType.fixed),
        # Anchored: "Variable Modifications:" also appears inside longer setting names.
        variable_mods=_homogenize_mods(
            _anchored_value(lines, "Variable Modifications:"), ModType.variable
        ),
        max_mods=int(_required(lines, "Max Variable Modifications:")),
        min_precursor_charge=min_z,
        max_precursor_charge=max_z,
        scan_window=settings_value(lines, "XIC IM Extraction Window:"),
        quantification_method=settings_value(lines, "Quantity MS Level:"),
        protein_inference=settings_value(lines, "Inference Algorithm:"),
        abundance_normalization_ions=settings_value(lines, "Cross-Run Normalization:"),
    )
