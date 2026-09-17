"""i2MassChroQ tabular parameter-file parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.common import (
    mapped_modifications,
    read_lines,
    symmetric_tolerance,
    tolerance_unit,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    ParamsError,
    Probability,
)

_MODIFICATIONS = {
    "57.02146@C": "C[Carbamidomethyl]",
    "15.99491@M": "M[Oxidation]",
    "+42.01056@[": "Protein N-term[Acetyl]",
    "Acetyl(N-term)": "Protein N-term[Acetyl]",
    "C:57.021465": "C[Carbamidomethyl]",
    "M:15.994915": "M[Oxidation]",
    "^E:-18.010565": "E[Pyro-glu from E]",
    "^Q:-17.026548": "Q[Pyro-glu from Q]",
}


def _record(source: Path | IO[bytes] | IO[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_lines(source):
        key, separator, value = line.partition("\t")
        if separator:
            values[key] = value.strip()
    if "i2MassChroQ_VERSION" not in values:
        raise ParamsError("not an i2MassChroQ parameter file")
    return values


def _probability(values: dict[str, str], key: str) -> Probability | None:
    raw = values.get(key)
    return Probability(value=float(raw)) if raw else None


def _tokens(values: dict[str, str], prefix: str) -> list[str]:
    return [value for key, value in values.items() if key.startswith(prefix) and value]


def _float_or_none(values: dict[str, str], key: str) -> float | None:
    raw = values.get(key)
    return float(raw) if raw else None


def _xtandem(values: dict[str, str]) -> Parameters:
    unit = values["spectrum, parent monoisotopic mass error units"]
    lower = -float(values["spectrum, parent monoisotopic mass error minus"])
    upper = float(values["spectrum, parent monoisotopic mass error plus"])
    fragment = float(values["spectrum, fragment monoisotopic mass error"])
    fixed = _tokens(values, "residue, modification mass")
    variable = _tokens(values, "residue, potential modification mass")
    if values.get("protein, quick acetyl") == "yes":
        variable.append("Acetyl(N-term)")
    missed_key = (
        "refine, maximum missed cleavage sites"
        if values.get("refine") == "yes"
        else "scoring, maximum missed cleavage sites"
    )
    return Parameters(
        software_name="i2MassChroQ",
        software_version=values["i2MassChroQ_VERSION"],
        quantification_software="i2MassChroQ",
        quantification_software_version=values["i2MassChroQ_VERSION"],
        acquisition_method="DDA",
        search_engine=values.get("AnalysisSoftware_name", "").replace("X! ", "X!"),
        search_engine_version=values.get("AnalysisSoftware_version"),
        ident_fdr_psm=_probability(values, "psm_fdr"),
        ident_fdr_peptide=_probability(values, "peptide_fdr"),
        ident_fdr_protein=_probability(values, "protein_fdr"),
        enable_match_between_runs=values.get("mcq_mbr") == "T",
        precursor_mass_tolerance=symmetric_tolerance(lower, upper, unit),
        fragment_mass_tolerance=MassTolerance(
            mode="absolute", value=fragment, unit=tolerance_unit(unit)
        ),
        enzyme=values.get("protein, cleavage site"),
        semi_enzymatic=values.get("protein, cleavage semi") == "yes",
        allowed_miscleavages=int(values[missed_key]),
        fixed_mods=mapped_modifications(fixed, _MODIFICATIONS, ModType.fixed),
        variable_mods=mapped_modifications(variable, _MODIFICATIONS, ModType.variable),
        max_precursor_charge=int(values["spectrum, maximum parent charge"]),
    )


def _sage_tolerance(raw: str) -> MassTolerance:
    lower, upper, unit = raw.split()
    return symmetric_tolerance(float(lower), float(upper), unit)


def _sage(values: dict[str, str]) -> Parameters:
    minimum, maximum = (int(value) for value in values["sage_precursor_charge"].split())
    fixed = values.get("sage_database_static_mods", "").split()
    variable = values.get("sage_database_variable_mods", "").split()
    restriction = values.get("sage_database_enzyme_restrict", "")
    enzyme = values.get("sage_database_enzyme_cleave_at", "")
    if restriction:
        enzyme = f"{enzyme}|{restriction}"
    return Parameters(
        software_name="i2MassChroQ",
        software_version=values["i2MassChroQ_VERSION"],
        quantification_software="i2MassChroQ",
        quantification_software_version=values["i2MassChroQ_VERSION"],
        acquisition_method="DDA",
        search_engine="Sage",
        search_engine_version=values.get("AnalysisSoftware_version") or values.get("sage_version"),
        ident_fdr_psm=_probability(values, "psm_fdr"),
        ident_fdr_peptide=_probability(values, "peptide_fdr"),
        ident_fdr_protein=_probability(values, "protein_fdr"),
        enable_match_between_runs=values.get("mcq_mbr") == "T",
        precursor_mass_tolerance=_sage_tolerance(values["sage_precursor_tol"]),
        fragment_mass_tolerance=_sage_tolerance(values["sage_fragment_tol"]),
        enzyme=enzyme,
        semi_enzymatic=False,
        allowed_miscleavages=int(values["sage_database_enzyme_missed_cleavages"]),
        min_peptide_length=int(values["sage_database_enzyme_min_len"]),
        max_peptide_length=int(values["sage_database_enzyme_max_len"]),
        fixed_mods=mapped_modifications(fixed, _MODIFICATIONS, ModType.fixed),
        variable_mods=mapped_modifications(variable, _MODIFICATIONS, ModType.variable),
        max_mods=int(values["sage_database_max_variable_mods"]),
        min_precursor_charge=minimum,
        max_precursor_charge=maximum,
        min_fragment_mz=_float_or_none(values, "sage_database_fragment_min_mz"),
        max_fragment_mz=_float_or_none(values, "sage_database_fragment_max_mz"),
    )


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse one i2MassChroQ parameter export."""
    values = _record(source)
    engine = re.sub(r"\s+", "", values.get("AnalysisSoftware_name", "")).lower()
    if engine == "x!tandem":
        return _xtandem(values)
    if engine == "sage":
        return _sage(values)
    raise ParamsError(f"unsupported i2MassChroQ search engine: {engine!r}")
