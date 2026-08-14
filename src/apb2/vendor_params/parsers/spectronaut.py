"""Spectronaut settings-text parameter parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.vendor_params.model import Parameters
from apb2.vendor_params.parsers._common import homogenize_paren_mods, read_lines

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

# Fallback mapping for modifications without parenthesized residue specifiers.
MODIFICATION_MAPPING = {
    "Cys-Cys": "C[Disulfide]",
    "Cysteinyl": "C[Cysteinyl]",
    "Cysteinyl - carbamidomethyl": "C[Cysteinyl + Carbamidomethyl]",
}


def _homogenize_mods(raw_mods: str, sep: str = ",") -> str:
    """Map a separator-delimited ``{name} ({residues})`` string to ProForma-like notation."""
    if not raw_mods.strip():
        return raw_mods
    return ", ".join(
        homogenize_paren_mods(mod, MODIFICATION_MAPPING)
        for mod in raw_mods.split(sep)
        if mod.strip()
    )


def _clean(text: str) -> str:
    return re.sub(r"^[\s:,\t]+|[\s:,\t]+$", "", text)


def _value(lines: list[str], term: str) -> str | None:
    for line in lines:
        if term in line:
            return _clean(line.split(term)[1])
    return None


def _required_value(lines: list[str], term: str) -> str:
    """Return a required Spectronaut setting."""
    value = _value(lines, term)
    if value is None:
        raise ValueError(f"Spectronaut setting {term!r} is missing")
    return value


def _value_regex(lines: list[str], pattern: str) -> str | None:
    for line in lines:
        if re.search(pattern, line):
            return _clean(re.split(pattern, line)[1])
    return None


def _extract_tolerances(
    lines: list[str],
    system: str,
) -> tuple[str | None, str | None]:
    system_lines = _tolerance_system_lines(lines, system)
    calibration, main_search_lines = _main_search_block(system_lines)
    if calibration is None:
        return None, None
    if calibration == "Dynamic":
        return "Dynamic", "Dynamic"
    patterns = _tolerance_patterns(calibration)
    if patterns is None:
        return None, None
    unit, ms1_pattern, ms2_pattern = patterns
    ms1 = _first_regex_value(main_search_lines, ms1_pattern)
    ms2 = _first_regex_value(main_search_lines, ms2_pattern)
    if ms1 is None or ms2 is None:
        return None, None
    return f"[-{ms1} {unit}, {ms1} {unit}]", f"[-{ms2} {unit}, {ms2} {unit}]"


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


def _tolerance_patterns(
    calibration: str,
) -> tuple[str, re.Pattern[str], re.Pattern[str]] | None:
    if calibration == "Static":
        return "Th", _MS1_STATIC, _MS2_STATIC
    if calibration == "Relative":
        return "ppm", _MS1_RELATIVE, _MS2_RELATIVE
    return None


def extract_params(source: _Source) -> Parameters:
    """Parse a Spectronaut settings-export text file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.spectronaut.read_spectronaut_settings``.
    """
    lines = read_lines(source, strip=True)
    vendor = _value(lines, "Vendor:")
    if vendor not in _VENDOR_SYSTEM_MAP:
        raise ValueError(
            f"unknown Spectronaut vendor: {vendor!r}; expected one of {sorted(_VENDOR_SYSTEM_MAP)}"
        )
    system = _VENDOR_SYSTEM_MAP[vendor]

    software_version = lines[0].split()[1]

    # Strip tree-drawing characters present in some Spectronaut exports.
    lines = [re.sub(r"^[\s│├─└]*", "", line).strip() for line in lines]

    precursor_tol, fragment_tol = _extract_tolerances(lines, system)

    psm_raw = _value(lines, "Precursor Qvalue Cutoff:")
    protein_raw = _value(lines, "Protein Qvalue Cutoff (Experiment):")
    ident_psm = float(psm_raw.replace(",", ".")) if psm_raw is not None else None
    ident_protein = float(protein_raw.replace(",", ".")) if protein_raw is not None else None

    charge_raw = _value(lines, "Peptide Charge:")
    if charge_raw is None or charge_raw == "False":
        min_z = max_z = None
    else:
        min_z = max_z = int(charge_raw)

    fixed_modifications = _value(lines, "Fixed Modifications:")
    variable_modifications = _value_regex(lines, r"^Variable Modifications:")

    return Parameters.model_validate(
        {
            "software_name": "Spectronaut",
            "software_version": software_version,
            "search_engine": "Spectronaut",
            "search_engine_version": software_version,
            "ident_fdr_psm": ident_psm,
            "ident_fdr_protein": ident_protein,
            "enable_match_between_runs": False,
            "precursor_mass_tolerance": precursor_tol,
            "fragment_mass_tolerance": fragment_tol,
            "enzyme": _value(lines, "Enzymes / Cleavage Rules:"),
            "semi_enzymatic": _value(lines, "Digest Type:") != "Specific",
            "allowed_miscleavages": int(_required_value(lines, "Missed Cleavages:")),
            "max_peptide_length": int(_required_value(lines, "Max Peptide Length:")),
            "min_peptide_length": int(_required_value(lines, "Min Peptide Length:")),
            "fixed_mods": (
                None if fixed_modifications is None else _homogenize_mods(fixed_modifications)
            ),
            "variable_mods": (
                None if variable_modifications is None else _homogenize_mods(variable_modifications)
            ),
            "max_mods": int(_required_value(lines, "Max Variable Modifications:")),
            "min_precursor_charge": min_z,
            "max_precursor_charge": max_z,
            "scan_window": _value(lines, "XIC IM Extraction Window:"),
            "quantification_method": _value(lines, "Quantity MS Level:"),
            "protein_inference": _value(lines, "Inference Algorithm:"),
            "abundance_normalization_ions": _value(lines, "Cross-Run Normalization:"),
        }
    )
