"""Shared parser helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

Source = Path | IO[bytes] | IO[str]


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


def format_tolerance_range(tolerance: Mapping[str, Sequence[float | int]]) -> str:
    """Format a mass-tolerance dict ``{unit: [low, high]}`` as a bracketed string.

    Output format matches ProteoBench's `_format_tolerance_range` so that
    existing expected-output CSVs round-trip unchanged.

    Examples
    --------
    >>> format_tolerance_range({"ppm": [-20.0, 20.0]})
    '[-20.0 ppm, 20.0 ppm]'
    """
    if not isinstance(tolerance, dict):
        raise ValueError(f"tolerance must be dict, got {type(tolerance).__name__}")

    unit_lookup = {"ppm": "ppm", "da": "Da"}
    for key, values in tolerance.items():
        normalized = str(key).lower()
        if normalized in unit_lookup:
            unit = unit_lookup[normalized]
            return "[" + ", ".join(f"{v} {unit}" for v in values) + "]"
    raise KeyError(f"unsupported tolerance unit(s): {list(tolerance.keys())}")


def homogenize_paren_mods(mod: str, mapping: Mapping[str, str]) -> str:
    """Convert a ``{name} (residues)`` modification token to ProForma-like notation.

    ``Carbamidomethyl (C)`` -> ``C[Carbamidomethyl]``;
    ``Phospho (STY)`` -> ``S[Phospho], T[Phospho], Y[Phospho]``;
    ``Acetyl (Protein N-term)`` -> ``Protein N-term[Acetyl]``. Multi-letter
    residue specs expand one token per letter. Tokens without a parenthesized
    residue spec fall back to *mapping* (unrecognized names pass through). The
    mapping data is per-vendor; only this mechanic is shared.
    """
    mod = mod.strip()
    idx = mod.rfind("(")
    if idx == -1:
        return mapping.get(mod, mod)
    name = mod[:idx].strip()
    residues = mod[idx + 1 :].rstrip(")").strip()
    if "n-term" in residues.lower() or "c-term" in residues.lower():
        return f"{residues}[{name}]"
    return ", ".join(f"{aa}[{name}]" for aa in residues)


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
