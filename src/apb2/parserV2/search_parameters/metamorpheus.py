"""MetaMorpheus parameter-file parser (TOML + version text)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field

from apb2.parserV2.search_parameters.common import modifications, read_text, tolerance_unit
from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)

_Source = Path | IO[bytes] | IO[str]

# MetaMorpheus names its protease in lowercase; the canonical display names are APB's.
_ENZYME_MAP = {"trypsin": "Trypsin", "trypsin/p": "Trypsin/P"}
_TERMINI = {
    "(Prot N-Term)": "Protein N-term",
    "(Pep N-Term)": "N-term",
    "(Prot C-Term)": "Protein C-term",
    "(Pep C-Term)": "C-term",
}


class _VendorModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DigestionSettings(_VendorModel):
    """MetaMorpheus digestion fields consumed by APB."""

    protease: str = Field(alias="Protease")
    max_missed_cleavages: int = Field(alias="MaxMissedCleavages")
    min_peptide_length: int = Field(alias="MinPeptideLength")
    max_peptide_length: int = Field(alias="MaxPeptideLength")
    max_mods_for_peptide: int = Field(alias="MaxModsForPeptide")


class PrecursorDeconvolutionSettings(_VendorModel):
    """MetaMorpheus precursor charge bounds consumed by APB."""

    min_assumed_charge_state: int = Field(alias="MinAssumedChargeState")
    max_assumed_charge_state: int = Field(alias="MaxAssumedChargeState")


class CommonSettings(_VendorModel):
    """MetaMorpheus common settings consumed by APB."""

    fixed_modifications: str = Field(alias="ListOfModsFixed")
    variable_modifications: str = Field(alias="ListOfModsVariable")
    precursor_mass_tolerance: str = Field(alias="PrecursorMassTolerance")
    product_mass_tolerance: str = Field(alias="ProductMassTolerance")
    q_value_threshold: float = Field(alias="QValueThreshold")
    digestion: DigestionSettings = Field(alias="DigestionParams")
    precursor_deconvolution: PrecursorDeconvolutionSettings = Field(
        alias="PrecursorDeconvolutionParameters"
    )


class SearchSettings(_VendorModel):
    """MetaMorpheus search settings consumed by APB."""

    match_between_runs: bool = Field(alias="MatchBetweenRuns")
    do_parsimony: bool = Field(alias="DoParsimony")
    normalize: bool = Field(alias="Normalize")


class MetaMorpheusSettings(_VendorModel):
    """Typed subset of a MetaMorpheus search-task TOML document."""

    common: CommonSettings = Field(alias="CommonParameters")
    search: SearchSettings = Field(alias="SearchParameters")


@dataclass(frozen=True, slots=True)
class SettingsFile:
    """An input recognized as the MetaMorpheus TOML settings file."""

    settings: MetaMorpheusSettings


@dataclass(frozen=True, slots=True)
class VersionFile:
    """An input recognized as the MetaMorpheus version report."""

    first_line: str


def _format_tolerance(tolerance: str) -> MassTolerance:
    """Read ``"±20.0000 PPM"`` as a typed symmetric tolerance."""
    value, unit = tolerance.split()
    return MassTolerance(
        mode="absolute",
        value=float(value.strip("±")),
        unit=tolerance_unit(unit),
    )


def _homogenize_mod(mod_str: str) -> str:
    """Convert a MetaMorpheus modification spec to ProForma-like notation.

    MetaMorpheus format: ``{modname} on {residue}`` with optional terminal
    qualifiers like ``(Pep N-Term)`` or ``(Prot N-Term)``.

    Examples:
        ``Carbamidomethyl on C`` -> ``C[Carbamidomethyl]``
        ``Acetylation on X (Prot N-Term)`` -> ``Protein N-term[Acetylation]``
        ``Oxidation on M`` -> ``M[Oxidation]``
    """
    mod_str = mod_str.strip()
    if " on " not in mod_str:
        return mod_str
    name, residue_part = mod_str.split(" on ", 1)
    residue_part = residue_part.strip()
    for qualifier, target in _TERMINI.items():
        if qualifier in residue_part:
            return f"{target}[{name}]"
    return f"{residue_part}[{name}]"


def _parse_modifications(mods: str, mod_type: ModType) -> list[SearchedModification]:
    """Resolve MetaMorpheus's tab-delimited modification blocks."""
    return modifications(
        (
            _homogenize_mod(parts[1])
            for parts in (entry.split("\t") for entry in mods.split("\t\t"))
            if len(parts) > 1
        ),
        mod_type,
    )


def _load_pair(file_a: _Source, file_b: _Source) -> tuple[VersionFile, SettingsFile]:
    """Identify which input is the version-text file and which is the TOML."""
    first = _try_load(file_a)
    second = _try_load(file_b)
    if isinstance(first, VersionFile) and isinstance(second, SettingsFile):
        return first, second
    if isinstance(first, SettingsFile) and isinstance(second, VersionFile):
        return second, first
    raise ValueError("expected one TOML file and one version-text file")


def _try_load(source: _Source) -> SettingsFile | VersionFile:
    """Classify and parse one MetaMorpheus input file."""
    text = read_text(source, errors="replace")
    if "[CommonParameters]" in text and "[SearchParameters]" in text:
        raw_document: object = tomllib.loads(text)
        return SettingsFile(MetaMorpheusSettings.model_validate(raw_document))
    lines = text.splitlines()
    if not lines:
        raise ValueError("MetaMorpheus version report is empty")
    return VersionFile(lines[0].strip())


def extract_params(file_a: _Source, file_b: _Source) -> Parameters:
    """Parse a MetaMorpheus TOML + version-text file pair (order-independent).

    Mirrors ``proteobench.io.params.metamorpheus.extract_params``.
    """
    version_file, settings_file = _load_pair(file_a, file_b)
    common = settings_file.settings.common
    search = settings_file.settings.search
    digestion = common.digestion
    precursor = common.precursor_deconvolution
    version_parts = version_file.first_line.split()
    if len(version_parts) < 3:
        raise ValueError("MetaMorpheus version report has no version value")

    return Parameters(
        software_name="MetaMorpheus",
        software_version=version_parts[2],
        search_engine="MetaMorpheus",
        enzyme=_ENZYME_MAP.get(digestion.protease.lower(), digestion.protease),
        allowed_miscleavages=digestion.max_missed_cleavages,
        fixed_mods=_parse_modifications(common.fixed_modifications, ModType.fixed),
        variable_mods=_parse_modifications(common.variable_modifications, ModType.variable),
        precursor_mass_tolerance=_format_tolerance(common.precursor_mass_tolerance),
        fragment_mass_tolerance=_format_tolerance(common.product_mass_tolerance),
        min_peptide_length=digestion.min_peptide_length,
        max_peptide_length=digestion.max_peptide_length,
        max_mods=digestion.max_mods_for_peptide,
        min_precursor_charge=precursor.min_assumed_charge_state,
        max_precursor_charge=precursor.max_assumed_charge_state,
        enable_match_between_runs=search.match_between_runs,
        quantification_method="FlashLFQ",
        protein_inference="Parsimony" if search.do_parsimony else None,
        abundance_normalization_ions=search.normalize,
        ident_fdr_psm=Probability(value=common.q_value_threshold),
    )
