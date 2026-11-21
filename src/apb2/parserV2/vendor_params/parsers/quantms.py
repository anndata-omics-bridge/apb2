"""quantms parameter parser (versions YAML + Nextflow params JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from apb2.parserV2.search_parameters.model import (
    AcquisitionMethod,
    MassTolerance,
    ModType,
    Parameters,
    SearchedModification,
)
from apb2.parserV2.vendor_params.parsers.shared.common import (
    homogenize_paren_mods,
    modifications,
    read_text,
    split_modifications,
    tolerance_unit,
)
from apb2.parserV2.vendor_params.parsers.shared.errors import ParamsError

_Source = Path | IO[bytes] | IO[str]

_ACQUISITION: dict[str, AcquisitionMethod] = {"dda": "DDA", "dia": "DIA"}


class QuantmsSettings(BaseModel):
    """Typed subset of the Nextflow ``params`` JSON a quantms run writes."""

    model_config = ConfigDict(extra="ignore")

    acquisition_method: Literal["dda", "dia"]
    enzyme: str
    allowed_missed_cleavages: int
    precursor_mass_tolerance: float
    precursor_mass_tolerance_unit: str
    fragment_mass_tolerance: float
    fragment_mass_tolerance_unit: str
    fixed_mods: str
    variable_mods: str
    psm_level_fdr_cutoff: float
    protein_level_fdr_cutoff: float
    min_precursor_charge: int
    max_precursor_charge: int
    min_peptide_length: int
    max_peptide_length: int
    max_mods: int
    protein_inference_method: str
    quantification_method: str


@dataclass(frozen=True, slots=True)
class SettingsFile:
    """An input recognized as the quantms params JSON."""

    settings: QuantmsSettings


@dataclass(frozen=True, slots=True)
class VersionsFile:
    """An input recognized as the nf-core software-versions YAML."""

    workflow_version: str
    search_engine: str | None
    search_engine_version: str | None


def _tolerance(value: float, unit: str) -> MassTolerance:
    return MassTolerance(mode="absolute", value=value, unit=tolerance_unit(unit))


def _parse_modifications(value: str, mod_type: ModType) -> list[SearchedModification]:
    """Resolve quantms's comma-separated ``Name (sites)`` modification list."""
    return modifications(
        (
            token
            for entry in split_modifications(value)
            for token in homogenize_paren_mods(entry, {})
        ),
        mod_type,
    )


def _versions(payload: dict[str, object]) -> VersionsFile:
    """Read the workflow and search-engine identities from the versions YAML."""
    workflow = cast("dict[str, object]", payload["Workflow"])
    version = workflow.get("bigbio/quantms") or workflow.get("nf-core/quantms")
    if not isinstance(version, str):
        raise ParamsError("quantms workflow version is missing")
    engines: list[str] = []
    engine_versions: list[str] = []
    for name, raw in payload.items():
        if not name.startswith("SEARCHENGINE"):
            continue
        engines.append(name.removeprefix("SEARCHENGINE").lower())
        if isinstance(raw, dict):
            values = cast("dict[str, object]", raw)
            candidate = values.get("Comet") or values.get("msgf_plus")
            if candidate is not None:
                engine_versions.append(str(candidate))
    return VersionsFile(
        workflow_version=version,
        search_engine=",".join(engines) or None,
        search_engine_version=",".join(engine_versions) or None,
    )


def _try_load(source: _Source) -> SettingsFile | VersionsFile:
    """Classify and parse one quantms input file by its content."""
    text = read_text(source)
    if text.lstrip().startswith("{"):
        return SettingsFile(QuantmsSettings.model_validate(json.loads(text)))
    payload: object = yaml.safe_load(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("Workflow"), dict):
        raise ParamsError("not a quantms versions file or params JSON")
    return _versions(cast("dict[str, object]", payload))


def _load_pair(file_a: _Source, file_b: _Source) -> tuple[VersionsFile, SettingsFile]:
    """Identify which input is the versions YAML and which the params JSON."""
    first = _try_load(file_a)
    second = _try_load(file_b)
    if isinstance(first, VersionsFile) and isinstance(second, SettingsFile):
        return first, second
    if isinstance(first, SettingsFile) and isinstance(second, VersionsFile):
        return second, first
    raise ParamsError("expected one quantms versions YAML and one params JSON")


def extract_params(file_a: _Source, file_b: _Source) -> Parameters:
    """Parse a quantms versions-YAML + params-JSON pair (order-independent)."""
    versions, settings_file = _load_pair(file_a, file_b)
    settings = settings_file.settings
    return Parameters(
        software_name="quantms",
        software_version=versions.workflow_version,
        quantification_software="quantms",
        quantification_software_version=versions.workflow_version,
        acquisition_method=_ACQUISITION[settings.acquisition_method],
        search_engine=versions.search_engine,
        search_engine_version=versions.search_engine_version,
        enzyme=settings.enzyme,
        allowed_miscleavages=settings.allowed_missed_cleavages,
        precursor_mass_tolerance=_tolerance(
            settings.precursor_mass_tolerance, settings.precursor_mass_tolerance_unit
        ),
        fragment_mass_tolerance=_tolerance(
            settings.fragment_mass_tolerance, settings.fragment_mass_tolerance_unit
        ),
        fixed_mods=_parse_modifications(settings.fixed_mods, ModType.fixed),
        variable_mods=_parse_modifications(settings.variable_mods, ModType.variable),
        ident_fdr_psm=settings.psm_level_fdr_cutoff,
        ident_fdr_protein=settings.protein_level_fdr_cutoff,
        min_precursor_charge=settings.min_precursor_charge,
        max_precursor_charge=settings.max_precursor_charge,
        min_peptide_length=settings.min_peptide_length,
        max_peptide_length=settings.max_peptide_length,
        max_mods=settings.max_mods,
        protein_inference=settings.protein_inference_method,
        quantification_method=settings.quantification_method,
    )
