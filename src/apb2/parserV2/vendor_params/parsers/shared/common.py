"""Shared parser helpers: source reading, and the two vendor-agnostic value grammars."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    SearchedModification,
    ToleranceUnit,
)
from apb2.parserV2.vendor_params.parsers.shared.unimod import (
    UNIMOD_REGISTRY,
    UnimodEntry,
    UnimodMatch,
)

Source = Path | IO[bytes] | IO[str]

# A tolerance the tool calibrated from the data instead of searching at a fixed width. The
# spellings are pooled because the parsers that need them read free-text settings blocks.
_AUTOMATIC_LABEL = "Automatic calibration"
_AUTOMATIC_SPELLINGS = frozenset(
    {"dynamic", "automatic", "automatic calibration", "auto", "auto detected", "0", "0 ppm"}
)
_TOLERANCE_RE = re.compile(r"^(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)$")
_TOKEN_RE = re.compile(r"^(?P<target>.*?)\[(?P<identity>[^\[\]]+)\]$")
_MASS_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class MassModificationMatch:
    """A vendor mass table matched a modification name."""

    name: str


@dataclass(frozen=True)
class UnrecognizedModificationMass:
    """A vendor mass table contains no entry for a mass."""

    mass: float


def read_text(source: Source, *, errors: str = "strict") -> str:
    """Read a path, text file-like, or bytes file-like into text.

    Rewinds seekable streams first (a no-op on fresh sources) and decodes bytes
    as UTF-8. Centralizes the source-acquisition logic that each vendor parser
    used to re-implement; only the per-vendor parse step legitimately varies.
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors=errors)

    if source.seekable():
        source.seek(0)
    raw = source.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors=errors)
    return raw


def read_lines(source: Source, *, strip: bool = False) -> list[str]:
    """Read *source* into a list of lines, optionally stripping each line."""
    lines = read_text(source).splitlines()
    return [line.strip() for line in lines] if strip else lines


def settings_value(lines: list[str], term: str) -> str | None:
    """Read a ``Term: value`` setting from a vendor's settings-export text.

    PEAKS and Spectronaut both export their settings as labelled lines; the punctuation and
    whitespace around the value are noise in both.
    """
    for line in lines:
        if term in line:
            return re.sub(r"^[\s:,\t]+|[\s:,\t]+$", "", line.split(term, 1)[1])
    return None


def required_settings_value(lines: list[str], term: str, *, software: str) -> str:
    """Read a setting the parser cannot proceed without."""
    value = settings_value(lines, term)
    if value is None:
        raise ValueError(f"{software} setting {term!r} is missing")
    return value


def tolerance_from_text(value: str) -> MassTolerance:
    """Read a tolerance a vendor states as prose: ``20 ppm``, ``0.02Da``, ``Dynamic``.

    Vendors that state a number and a unit separately build :class:`MassTolerance` directly.
    """
    text = value.strip()
    if text.lower() in _AUTOMATIC_SPELLINGS:
        return MassTolerance(mode="automatic", label=_AUTOMATIC_LABEL)
    match = _TOLERANCE_RE.match(text)
    if match is None:
        raise ValueError(f"could not read mass tolerance: {value!r}")
    return MassTolerance(
        mode="absolute",
        value=float(match.group("value")),
        unit=tolerance_unit(match.group("unit")),
    )


def automatic_tolerance() -> MassTolerance:
    """The tolerance a tool calibrated from the data rather than searching at a fixed width."""
    return MassTolerance(mode="automatic", label=_AUTOMATIC_LABEL)


def symmetric_tolerance(lower: float, upper: float, unit: str) -> MassTolerance:
    """Build the half-width tolerance a vendor states as an explicit signed interval."""
    if not math.isclose(lower, -upper, abs_tol=1e-9):
        raise ValueError(f"asymmetric mass tolerance ranges are not supported: [{lower}, {upper}]")
    return MassTolerance(mode="absolute", value=abs(upper), unit=tolerance_unit(unit))


def tolerance_unit(unit: str) -> ToleranceUnit:
    """Narrow a vendor's unit spelling to the two units the schema declares."""
    lookup: dict[str, ToleranceUnit] = {"ppm": "ppm", "da": "Da", "th": "Da"}
    normalized = lookup.get(unit.strip().lower())
    if normalized is None:
        raise ValueError(f"mass tolerance unit must be ppm or Da, got {unit!r}")
    return normalized


def modifications(tokens: Iterable[str], mod_type: ModType) -> list[SearchedModification]:
    """Resolve ProForma-like ``Residue[Name]`` tokens against the Unimod registry.

    The notation is APB's, not any vendor's: each parser converts its own modification syntax
    into these tokens, and this resolves each identity once.
    """
    return [modification(token, mod_type) for token in tokens if token.strip()]


def mapped_modifications(
    declared: Iterable[str],
    mapping: Mapping[str, str],
    mod_type: ModType,
) -> list[SearchedModification]:
    """Resolve vendor modification labels through a per-vendor name table.

    One label can name several modifications — DIA-NN's ``UniMod:21`` is phospho on S, T, and
    Y — so a mapped label expands to one token each. Unmapped labels pass through.
    """
    return modifications(
        (
            token
            for label in declared
            for token in split_modifications(mapping.get(label.strip(), label.strip()))
        ),
        mod_type,
    )


def modification(token: str, mod_type: ModType) -> SearchedModification:
    """Canonicalize one known identity while preserving the token-shaped name."""
    match = _TOKEN_RE.fullmatch(token)
    identity = match.group("identity") if match is not None else token
    entry = _find_modification(identity)
    if entry is None:
        return SearchedModification(name=token, mod_type=mod_type, source=token)
    canonical = entry.name if match is None else f"{match.group('target')}[{entry.name}]"
    return SearchedModification(
        name=canonical,
        accession=entry.accession,
        mod_type=mod_type,
        mass_delta=entry.mass_delta,
        source=canonical,
    )


def split_modifications(value: str, separator: str = ",") -> list[str]:
    """Split a vendor's delimited modification string into tokens."""
    return [part.strip() for part in value.split(separator) if part.strip()]


def homogenize_paren_mods(mod: str, mapping: Mapping[str, str]) -> list[str]:
    """Convert a ``{name} (residues)`` modification token to ProForma-like tokens.

    ``Carbamidomethyl (C)`` -> ``["C[Carbamidomethyl]"]``;
    ``Phospho (STY)`` -> ``["S[Phospho]", "T[Phospho]", "Y[Phospho]"]``;
    ``Acetyl (Protein N-term)`` -> ``["Protein N-term[Acetyl]"]``. A multi-letter residue spec
    is one vendor entry naming several modifications, so it expands to one token each — hence
    a list, rather than a string the caller would have to split on a separator that also
    separates unrelated entries. Tokens without a parenthesized residue spec fall back to
    *mapping*; the mapping data is per-vendor, only this mechanic is shared.
    """
    mod = mod.strip()
    idx = mod.rfind("(")
    if idx == -1:
        return [mapping.get(mod, mod)]
    name = mod[:idx].strip()
    residues = mod[idx + 1 :].rstrip(")").strip()
    if "n-term" in residues.lower() or "c-term" in residues.lower():
        return [f"{residues}[{name}]"]
    return [f"{aa}[{name}]" for aa in residues]


def lookup_mass_mod(
    mass: float,
    mapping: Mapping[float, str],
    *,
    tol: float = 0.001,
) -> MassModificationMatch | UnrecognizedModificationMass:
    """Look up a modification name in a vendor mass table.

    The mass→name table and any fallback are per-vendor; only this nearest-match
    lookup is shared. An unmatched mass is an explicit domain result.
    """
    for ref_mass, name in mapping.items():
        if abs(mass - ref_mass) < tol:
            return MassModificationMatch(name)
    return UnrecognizedModificationMass(mass)


def _find_modification(identity: str) -> UnimodEntry | None:
    """Resolve a known name, accession, or mass without consuming unknown tokens."""
    by_name = UNIMOD_REGISTRY.find_by_name(identity)
    if isinstance(by_name, UnimodMatch):
        return by_name.entry
    if not _MASS_RE.fullmatch(identity.strip()):
        return None
    by_mass = UNIMOD_REGISTRY.find_by_mass(float(identity))
    return by_mass.entry if isinstance(by_mass, UnimodMatch) else None
