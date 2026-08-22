"""ProteoBench expected-output oracle for the vendor parameter parsers — TEST ONLY.

``apb2``'s :class:`Parameters` is a storage schema: it accepts the types its fields declare
and nothing else. ProteoBench's checked-in expected files are a **foreign text format** —
``NaN``, ``""``, ``"1.0%"``, ``"[-20.0 ppm, 20.0 ppm]"``, ``"KR"``, ``"Automatic
calibration"``. Interpreting that format is a parity-test concern, so the interpretation
lives here instead of running on every vendor parse inside ``model.py``; see
``TODO/Archive/TODO_vendor_params_boundary.md``.

Both sides of every comparison go through the same formatters, so a reported difference is a
difference in meaning rather than in spelling. The legacy ``Parameters.from_series`` /
``to_series`` pair this replaces lived in ``anndata_proteomics``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from apb2.parserV2.search_parameters import unimod as unimod_registry
from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    Parameters,
    SearchedModification,
    ToleranceUnit,
)

Cell = str | int | float | bool | None
Series = dict[str, str | None]

_MISSING_STRINGS: Final = frozenset(
    {"", "-", "none", "nan", "n/a", "na", "not specified", "unknown", "placeholder"}
)

# ProteoBench's own enzyme spellings, including the command-line and regex forms its CSVs
# carry. Only this oracle needs them: each vendor parser canonicalizes its own syntax.
_ENZYME_MAP: Final = {
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

_AUTOMATIC: Final = frozenset(
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
_RANGE_RE: Final = re.compile(
    r"^\[\s*(?P<lower>[+-]?\d+(?:\.\d+)?)\s*(?P<unit1>[A-Za-z]*)\s*,\s*"
    r"(?P<upper>[+-]?\d+(?:\.\d+)?)\s*(?P<unit2>[A-Za-z]*)\s*\]$"
)
_ABSOLUTE_RE: Final = re.compile(r"^(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]*)$")
_TOKEN_RE: Final = re.compile(r"^(?P<target>.*?)\[(?P<identity>[^\[\]]+)\]$")
_NUMBER_RE: Final = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

_PROBABILITY_FIELDS: Final = frozenset({"ident_fdr_psm", "ident_fdr_peptide", "ident_fdr_protein"})
_TOLERANCE_FIELDS: Final = frozenset({"precursor_mass_tolerance", "fragment_mass_tolerance"})
_MODIFICATION_FIELDS: Final = frozenset({"fixed_mods", "variable_mods"})
_BOOLEAN_FIELDS: Final = frozenset(
    {"enable_match_between_runs", "semi_enzymatic", "combine_charge_states"}
)


def actual(parameters: Parameters) -> Series:
    """Format one parsed ``Parameters`` into ProteoBench's comparison shapes."""
    out: Series = {}
    for field in Parameters.model_fields:
        value = getattr(parameters, field)
        if field in _MODIFICATION_FIELDS:
            out[field] = _format_modifications(value)
        elif field in _TOLERANCE_FIELDS:
            out[field] = None if value is None else _format_tolerance(value)
        elif field in _PROBABILITY_FIELDS:
            out[field] = None if value is None else _number(value.value)
        else:
            out[field] = _format_scalar(field, value)
    return out


def expected(raw: Mapping[str, object]) -> Series:
    """Interpret one ProteoBench expected record into the same comparison shapes."""
    out: Series = {}
    for field, value in raw.items():
        if _is_missing(value):
            out[field] = None
        elif field in _MODIFICATION_FIELDS:
            out[field] = _format_modifications(_read_modifications(str(value)))
        elif field in _TOLERANCE_FIELDS:
            out[field] = _format_tolerance(_read_tolerance(value))
        elif field in _PROBABILITY_FIELDS:
            out[field] = _number(_read_probability(value))
        elif field == "enzyme":
            out[field] = _ENZYME_MAP.get(str(value).strip().lower(), str(value))
        else:
            out[field] = _format_scalar(field, value)
    return out


def expected_csv(path: Path, *, delimiter: str = ",") -> Series:
    """Read a ProteoBench two-column expected CSV/TSV into comparison shapes."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    return expected({row[0]: row[1] for row in rows[1:] if len(row) > 1 and row[0]})


def expected_json(path: Path) -> Series:
    """Read a ProteoBench expected JSON record, tolerating its literal ``NaN`` tokens."""
    payload: object = json.loads(path.read_text(encoding="utf-8").replace("NaN", "null"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a ProteoBench expected record")
    record: dict[str, object] = payload
    return expected(record)


def compare(parameters: Parameters, reference: Series, fields: list[str]) -> list[str]:
    """Return one report line per field whose meaning differs between the two sides."""
    observed = actual(parameters)
    return [
        f"{field}: parsed {observed.get(field)!r} != expected {reference.get(field)!r}"
        for field in fields
        if observed.get(field) != reference.get(field)
    ]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS


def _number(value: float) -> str:
    return f"{float(value):g}"


def _format_scalar(field: str, value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if field in _BOOLEAN_FIELDS:
        return str(_read_boolean(text))
    return _number(float(text)) if _NUMBER_RE.fullmatch(text) else text


def _read_boolean(text: str) -> bool:
    """ProteoBench writes booleans as ``True``/``False`` text in CSV and as JSON literals."""
    lowered = text.lower()
    if lowered in {"true", "1", "1.0", "yes"}:
        return True
    if lowered in {"false", "0", "0.0", "no"}:
        return False
    raise ValueError(f"expected a boolean, got {text!r}")


def _format_tolerance(tolerance: MassTolerance) -> str:
    if tolerance.mode == "automatic":
        return "automatic"
    if tolerance.value is None or tolerance.unit is None:
        raise ValueError(f"absolute tolerance is incomplete: {tolerance!r}")
    return f"{_number(tolerance.value)} {tolerance.unit}"


def _format_modifications(modifications: list[SearchedModification]) -> str | None:
    return ",".join(mod.source or mod.name for mod in modifications) or None


def _read_probability(value: object) -> float:
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100
    number = float(text)
    return number / 100 if number > 1 else number


def _read_tolerance(value: object) -> MassTolerance:
    text = str(value).strip()
    if text.lower() in _AUTOMATIC:
        return MassTolerance(mode="automatic", label="Automatic calibration")
    if match := _RANGE_RE.match(text):
        lower, upper = float(match.group("lower")), float(match.group("upper"))
        if not math.isclose(lower, -upper, abs_tol=1e-9):
            raise ValueError(f"asymmetric expected tolerance: {value!r}")
        return MassTolerance(
            mode="absolute",
            value=abs(upper),
            unit=_read_unit(match.group("unit1") or match.group("unit2")),
        )
    if match := _ABSOLUTE_RE.match(text):
        return MassTolerance(
            mode="absolute",
            value=float(match.group("value")),
            unit=_read_unit(match.group("unit")),
        )
    raise ValueError(f"could not read expected tolerance: {value!r}")


def _read_unit(unit: str) -> ToleranceUnit:
    lookup: dict[str, ToleranceUnit] = {"ppm": "ppm", "da": "Da", "th": "Da"}
    normalized = lookup.get(unit.strip().lower())
    if normalized is None:
        raise ValueError(f"expected tolerance unit must be ppm or Da, got {unit!r}")
    return normalized


def _read_modifications(value: str) -> list[SearchedModification]:
    text = value.strip()
    if not text:
        return []
    tokens = [text] if text.startswith("{") and text.endswith("}") else re.split(r"\s*,\s*", text)
    return [_read_modification(token) for token in tokens if token.strip()]


def _read_modification(token: str) -> SearchedModification:
    match = _TOKEN_RE.fullmatch(token)
    identity = match.group("identity") if match is not None else token
    entry = _find(identity)
    if entry is None:
        return SearchedModification(name=token, source=token)
    canonical = entry.name if match is None else f"{match.group('target')}[{entry.name}]"
    return SearchedModification(
        name=canonical,
        accession=entry.accession,
        mass_delta=entry.mass_delta,
        source=canonical,
    )


def _find(identity: str) -> unimod_registry.UnimodEntry | None:
    by_name = unimod_registry.find_by_name(identity)
    if isinstance(by_name, unimod_registry.UnimodMatch):
        return by_name.entry
    if not _NUMBER_RE.fullmatch(identity.strip()):
        return None
    by_mass = unimod_registry.find_by_mass(float(identity))
    return by_mass.entry if isinstance(by_mass, unimod_registry.UnimodMatch) else None
