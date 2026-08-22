"""Typed parameter models for proteomics search-engine settings.

This is a **storage schema**: it declares what a search-parameter record is, rejects illegal
states, and serializes to JSON for ``uns``. It does not interpret vendor text. Each vendor
parser reads its own format and hands over values of exactly the declared types — a
``MassTolerance``, a ``Probability``, a ``list[SearchedModification]`` — so there is one place
per concept rather than a parser that stringifies and a validator that parses the string back.
See ``TODO/Archive/TODO_vendor_params_boundary.md`` for the round trip this replaced.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)


class ParamsError(Exception):
    """A parameter file could not be parsed for the requested software.

    Raised when the file is clearly not the expected format (e.g. a FragPipe workflow handed to the
    DIA-NN parser) or lacks the markers a parser needs. It is a clean input-error signal that
    conversion propagates and interactive callers may translate into a focused diagnostic.
    """


AcquisitionMethod = Literal["DDA", "DIA", "unknown"]
ToleranceUnit = Literal["ppm", "Da"]
ToleranceMode = Literal["absolute", "automatic"]

# Vendor CSV cells and XML attributes arrive blank, as a placeholder word, or as a pandas
# ``NaN``. All of them mean "this file does not say", which is the one text interpretation the
# schema still owns because it is about absence rather than about any vendor's grammar.
_MISSING_STRINGS = {"", "-", "none", "nan", "n/a", "na", "not specified", "unknown", "placeholder"}

_ACCESSION_RE = re.compile(r"^(UNIMOD|MOD):\d+$", re.IGNORECASE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModType(StrEnum):
    """Whether a modification was searched as fixed, variable, or unknown."""

    fixed = "fixed"
    variable = "variable"
    unknown = "unknown"


class SearchedModification(_Strict):
    """A modification declared in a search-engine parameter file.

    Part of the parameter schema, not of modification domain knowledge: it is parsed from a
    vendor parameter file and carries no sequence localization — that is
    ``ModificationOccurrence``, which the sequence normalizer builds.
    """

    name: str
    accession: str | None = None
    mod_type: ModType = ModType.unknown
    target: str | None = None
    position: str | None = "Anywhere"
    mass_delta: float | None = None
    source: str | None = None

    @field_validator("accession")
    @classmethod
    def _valid_accession(cls, value: str | None) -> str | None:
        if value is not None and not _ACCESSION_RE.match(value):
            raise ValueError("accession must look like UNIMOD:35 or MOD:00425")
        return value


class Probability(_Strict):
    """A probability value constrained to the closed interval [0, 1]."""

    value: float = Field(ge=0, le=1)


class MassTolerance(_Strict):
    """Mass tolerance centered at the theoretical mass.

    Mass tolerances are physically symmetric: a vendor saying
    ``[-20 ppm, 20 ppm]`` and a vendor saying ``20 ppm`` mean the same
    thing — a ± half-width around the theoretical peak. APB stores
    only that half-width as ``value`` plus a ``unit``. Asymmetric
    ranges are rejected where they are read; we have not seen a tool
    that actually means ``[-10 ppm, +30 ppm]`` in earnest.
    """

    value: NonNegativeFloat | None = None
    unit: ToleranceUnit | None = None
    mode: ToleranceMode
    label: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> MassTolerance:
        if self.mode == "absolute":
            if self.value is None:
                raise ValueError("absolute tolerance requires value")
            if self.unit is None:
                raise ValueError("absolute tolerance requires unit")
        else:
            if self.unit is not None:
                raise ValueError("automatic tolerance cannot define unit")
            if self.value is not None:
                raise ValueError("automatic tolerance cannot define numeric bounds")
        return self


class Parameters(_Strict):
    """Proteomics search-parameter record with typed fields."""

    software_name: str | None = None
    software_version: str | None = None
    quantification_software: str | None = None
    quantification_software_version: str | None = None
    acquisition_method: AcquisitionMethod = "unknown"
    search_engine: str | None = None
    search_engine_version: str | None = None
    ident_fdr_psm: Probability | None = None
    ident_fdr_peptide: Probability | None = None
    ident_fdr_protein: Probability | None = None
    enable_match_between_runs: bool | None = None
    precursor_mass_tolerance: MassTolerance | None = None
    fragment_mass_tolerance: MassTolerance | None = None
    enzyme: str | None = None
    semi_enzymatic: bool | None = None
    allowed_miscleavages: NonNegativeInt | None = None
    min_peptide_length: NonNegativeInt | None = None
    max_peptide_length: NonNegativeInt | None = None
    fixed_mods: list[SearchedModification] = Field(default_factory=list)
    variable_mods: list[SearchedModification] = Field(default_factory=list)
    max_mods: NonNegativeInt | None = None
    min_precursor_charge: PositiveInt | None = None
    max_precursor_charge: PositiveInt | None = None
    min_precursor_mz: NonNegativeFloat | None = None
    max_precursor_mz: NonNegativeFloat | None = None
    min_fragment_mz: NonNegativeFloat | None = None
    max_fragment_mz: NonNegativeFloat | None = None
    quantification_method: str | None = None
    # Whether the quantification step merged a peptidoform's charge states into one value.
    # This decides the quantification *level* of the result table, not just its content: a
    # charge-collapsed export is peptidoform-level even when the tool's file carries a
    # charge column (Sage writes -1 there). Parsing rules gate level availability on it.
    combine_charge_states: bool | None = None
    protein_inference: str | None = None
    abundance_normalization_ions: str | bool | None = None
    predictors_library: str | None = None
    # Two vendors report unrelated quantities here: DIA-NN a scan-window radius in scans,
    # Spectronaut an ion-mobility extraction window that can read "Dynamic". Splitting them
    # is a schema question recorded in TODO/Archive/TODO_vendor_params_boundary.md §9.
    scan_window: NonNegativeInt | str | None = None

    @field_validator(
        "software_name",
        "software_version",
        "quantification_software",
        "quantification_software_version",
        "search_engine",
        "search_engine_version",
        "enzyme",
        "quantification_method",
        "protein_inference",
        "abundance_normalization_ions",
        "predictors_library",
        mode="before",
    )
    @classmethod
    def _empty_strings_to_none(cls, value: object) -> object:
        return None if _is_missing(value) else value

    @model_validator(mode="after")
    def _validate_ranges(self) -> Parameters:
        _validate_order(self.min_precursor_charge, self.max_precursor_charge, "charge")
        _validate_order(self.min_peptide_length, self.max_peptide_length, "peptide length")
        _validate_order(self.min_precursor_mz, self.max_precursor_mz, "precursor m/z")
        _validate_order(self.min_fragment_mz, self.max_fragment_mz, "fragment m/z")
        return self


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS)


def _validate_order(
    minimum: int | float | None,
    maximum: int | float | None,
    label: str,
) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"minimum {label} cannot exceed maximum {label}")
