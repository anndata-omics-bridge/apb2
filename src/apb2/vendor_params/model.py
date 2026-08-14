"""Typed parameter models for proteomics search-engine settings."""

from __future__ import annotations

import math
import re
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

from apb2.modifications import unimod_registry
from apb2.modifications.model import SearchedModification


class ParamsError(Exception):
    """A parameter file could not be parsed for the requested software.

    Raised when the file is clearly not the expected format (e.g. a FragPipe workflow handed to the
    DIA-NN parser) or lacks the markers a parser needs. It is a clean input-error signal that
    conversion propagates and interactive callers may translate into a focused diagnostic.
    """


ScalarValue = str | int | float | bool | None
AcquisitionMethod = Literal["DDA", "DIA", "unknown"]
ToleranceUnit = Literal["ppm", "Da"]
ToleranceMode = Literal["absolute", "automatic"]

_MISSING_STRINGS = {"", "-", "none", "nan", "n/a", "na", "not specified", "unknown", "placeholder"}

# Canonical enzyme-name mapping (lowercase key -> display name), ported from
# ProteoBench's io/params `_ENZYME_MAP`. Applied symmetrically to parser output
# and round-tripped CSV expectations via the `enzyme` before-validator.
# COMMENT: map is for all the tools. Should this not be per tool?
_ENZYME_MAP = {
    "trypsin": "Trypsin",
    "trypsin/p": "Trypsin/P",
    "stricttrypsin": "Trypsin/P",
    "k*,r*,!p*": "Trypsin",
    "[rk]|{p}": "Trypsin",
    "[rk]": "Trypsin/P",
    "kr": "Trypsin/P",
    "kr|p,true": "Trypsin",
    "kr|p,t": "Trypsin",
    "kr,true": "Trypsin/P",
    "kr,t": "Trypsin/P",
    "lys-c": "Lys-C",
    "lysc": "Lys-C",
    "arg-c": "Arg-C",
    "argc": "Arg-C",
    "asp-n": "Asp-N",
    "aspn": "Asp-N",
    "chymotrypsin": "Chymotrypsin",
    "gluc": "Glu-C",
    "glu-c": "Glu-C",
}

# Tolerance values that indicate automatic calibration (ported from ProteoBench
# `_AUTO_CALIBRATION_SENTINELS`); collapsed to a single canonical label.
_AUTO_CALIBRATION_LABEL = "Automatic calibration"
_AUTO_CALIBRATION_SENTINELS = frozenset(
    {
        "dynamic",
        "automatic",
        "automatic calibration",
        "auto",
        "auto detected",
        "0",
        "0 ppm",
        "[-0.0 ppm, 0.0 ppm]",
    }
)
_RANGE_RE = re.compile(
    r"^\[\s*(?P<lower>[+-]?\d+(?:\.\d+)?)\s*(?P<unit1>[A-Za-z]*)\s*,\s*"
    r"(?P<upper>[+-]?\d+(?:\.\d+)?)\s*(?P<unit2>[A-Za-z]*)\s*\]$"
)
_ABSOLUTE_RE = re.compile(r"^(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]*)$")
_SEARCH_MOD_RE = re.compile(r"^(?P<target>.*?)\[(?P<identity>[^\[\]]+)\]$")
_MASS_IDENTITY_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Probability(_Strict):
    """A probability value constrained to the closed interval [0, 1]."""

    value: float = Field(ge=0, le=1)

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("%"):
                return float(text[:-1]) / 100
            return float(text)
        return value

    @classmethod
    def parse(cls, value: object) -> Probability | None:
        if _is_missing(value):
            return None
        if isinstance(value, Probability):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        numeric = _coerce_float(value)
        if numeric is None:
            return None
        if numeric >= 1:
            numeric /= 100
        return cls(value=numeric)


class MassTolerance(_Strict):
    """Mass tolerance centered at the theoretical mass.

    Mass tolerances are physically symmetric: a vendor saying
    ``[-20 ppm, 20 ppm]`` and a vendor saying ``20 ppm`` mean the same
    thing — a ± half-width around the theoretical peak. APB stores
    only that half-width as ``value`` plus a ``unit``. Asymmetric
    ranges are rejected at parse time; we have not seen a tool that
    actually means ``[-10 ppm, +30 ppm]`` in earnest.
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

    @classmethod
    def parse(cls, value: object) -> MassTolerance | None:
        """Parse vendor tolerance values into a typed tolerance."""
        if _is_missing(value):
            return None
        if isinstance(value, MassTolerance):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, int | float):
            return cls._parse_numeric(value)
        if not isinstance(value, str):
            raise TypeError(f"unsupported mass tolerance value: {value!r}")
        return cls._parse_text(value)

    @classmethod
    def _parse_numeric(cls, value: int | float) -> MassTolerance:
        if value == 0:
            return cls(mode="automatic", label=_AUTO_CALIBRATION_LABEL)
        raise ValueError("mass tolerance numeric values require an explicit unit")

    @classmethod
    def _parse_text(cls, value: str) -> MassTolerance:
        text = value.strip()
        if text.lower() in _AUTO_CALIBRATION_SENTINELS:
            return cls(mode="automatic", label=_AUTO_CALIBRATION_LABEL)

        range_match = _RANGE_RE.match(text)
        if range_match:
            return cls._parse_range(value, range_match)

        absolute_match = _ABSOLUTE_RE.match(text)
        if absolute_match:
            return cls(
                mode="absolute",
                value=float(absolute_match.group("value")),
                unit=_normalize_unit(absolute_match.group("unit")),
            )

        raise ValueError(f"could not parse mass tolerance: {value!r}")

    @classmethod
    def _parse_range(cls, value: str, match: re.Match[str]) -> MassTolerance:
        unit = _normalize_unit(match.group("unit1") or match.group("unit2"))
        lower = float(match.group("lower"))
        upper = float(match.group("upper"))
        if not math.isclose(lower, -upper, abs_tol=1e-9):
            raise ValueError(f"asymmetric mass tolerance ranges are not supported: {value!r}")
        return cls(mode="absolute", value=abs(upper), unit=unit)


class UnparsedParameter(_Strict):
    """Explicitly retained vendor value that is outside the core schema."""

    name: str
    value: ScalarValue
    source: str | None = None


# COMMENT : as few possible NONE fields.
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
    scan_window: NonNegativeInt | str | None = None
    unparsed_parameters: list[UnparsedParameter] = Field(default_factory=list)

    @field_validator(
        "software_name",
        "software_version",
        "quantification_software",
        "quantification_software_version",
        "search_engine",
        "search_engine_version",
        "quantification_method",
        "protein_inference",
        "abundance_normalization_ions",
        "predictors_library",
        mode="before",
    )
    @classmethod
    def _empty_strings_to_none(cls, value: object) -> object:
        return None if _is_missing(value) else value

    @field_validator("enzyme", mode="before")
    @classmethod
    def _canonicalize_enzyme(cls, value: object) -> object:
        if _is_missing(value):
            return None
        if isinstance(value, str):
            return _ENZYME_MAP.get(value.strip().lower(), value)
        return value

    @field_validator(
        "ident_fdr_psm",
        "ident_fdr_peptide",
        "ident_fdr_protein",
        mode="before",
    )
    @classmethod
    def _coerce_probability(cls, value: object) -> object:
        return Probability.parse(value)

    @field_validator("precursor_mass_tolerance", "fragment_mass_tolerance", mode="before")
    @classmethod
    def _coerce_tolerance(cls, value: object) -> object:
        return MassTolerance.parse(value)

    @field_validator(
        "allowed_miscleavages",
        "min_peptide_length",
        "max_peptide_length",
        "max_mods",
        mode="before",
    )
    @classmethod
    def _coerce_non_negative_int(cls, value: object) -> object:
        if _is_missing(value):
            return None
        return int(float(str(value).strip()))

    @field_validator("min_precursor_charge", "max_precursor_charge", mode="before")
    @classmethod
    def _coerce_positive_int(cls, value: object) -> object:
        if _is_missing(value):
            return None
        return int(float(str(value).strip()))

    @field_validator(
        "min_precursor_mz",
        "max_precursor_mz",
        "min_fragment_mz",
        "max_fragment_mz",
        mode="before",
    )
    @classmethod
    def _coerce_non_negative_float(cls, value: object) -> object:
        if _is_missing(value):
            return None
        return float(str(value).strip())

    @field_validator("enable_match_between_runs", "semi_enzymatic", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> object:
        if _is_missing(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"cannot coerce boolean value: {value!r}")

    @field_validator("scan_window", mode="before")
    @classmethod
    def _coerce_scan_window(cls, value: object) -> object:
        if _is_missing(value):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else str(value)
        if isinstance(value, str):
            text = value.strip()
            if not _MASS_IDENTITY_RE.fullmatch(text):
                return text
            numeric = float(text)
            return int(numeric) if numeric.is_integer() else text
        return value

    @field_validator("fixed_mods", "variable_mods", mode="before")
    @classmethod
    def _coerce_modifications(cls, value: object) -> object:
        if _is_missing(value):
            return []
        if isinstance(value, SearchedModification):
            return [value]
        if isinstance(value, dict):
            return [
                SearchedModification(
                    name=str(target),
                    target=str(target),
                    mass_delta=_coerce_float(delta),
                    source=f"{target}: {delta}",
                )
                for target, delta in value.items()
            ]
        if isinstance(value, list | tuple | set):
            return [_modification_from_item(item) for item in value if not _is_missing(item)]
        if isinstance(value, str):
            return [_modification_from_item(part) for part in _split_mod_string(value)]
        raise TypeError(f"unsupported modification value: {value!r}")

    @field_validator("unparsed_parameters", mode="before")
    @classmethod
    def _coerce_unparsed(cls, value: object) -> object:
        if _is_missing(value):
            return []
        return value

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


def _normalize_unit(unit: str) -> ToleranceUnit:
    if not unit:
        raise ValueError("mass tolerance requires unit ppm or Da")
    lookup: dict[str, ToleranceUnit] = {"ppm": "ppm", "da": "Da", "th": "Da"}
    normalized = lookup.get(unit.strip().lower())
    if normalized is None:
        raise ValueError("mass tolerance unit must be ppm or Da")
    return normalized


def _coerce_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        return float(text)
    return None


def _split_mod_string(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        return [text]
    return [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()]


def _modification_from_item(item: object) -> SearchedModification:
    if isinstance(item, SearchedModification):
        return item
    if isinstance(item, dict):
        return SearchedModification.model_validate(item)
    return _modification_from_token(str(item))


def _modification_from_token(token: str) -> SearchedModification:
    """Canonicalize a known identity while preserving the token-shaped API."""
    match = _SEARCH_MOD_RE.fullmatch(token)
    identity = match.group("identity") if match is not None else token
    result = _find_modification(identity)
    if not isinstance(result, unimod_registry.UnimodMatch):
        return SearchedModification(name=token, source=token)
    entry = result.entry

    canonical_token = entry.name if match is None else f"{match.group('target')}[{entry.name}]"
    return SearchedModification(
        name=canonical_token,
        accession=entry.accession,
        mass_delta=entry.mass_delta,
        source=canonical_token,
    )


def _find_modification(
    identity: str,
) -> (
    unimod_registry.UnimodMatch
    | unimod_registry.UnrecognizedUnimodName
    | unimod_registry.UnrecognizedUnimodMass
):
    """Resolve a known name/accession/mass without consuming unknown tokens."""
    name_result = unimod_registry.find_by_name(identity)
    if isinstance(name_result, unimod_registry.UnimodMatch):
        return name_result
    if not _MASS_IDENTITY_RE.fullmatch(identity.strip()):
        return name_result
    return unimod_registry.find_by_mass(float(identity))


def _validate_order(
    minimum: int | float | None,
    maximum: int | float | None,
    label: str,
) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"minimum {label} cannot exceed maximum {label}")
