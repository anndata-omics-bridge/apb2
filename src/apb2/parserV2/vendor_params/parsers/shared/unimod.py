"""Built-in canonical-modification document and runtime registry.

The JSON data file (``unimod_registry.json``, sibling of this module) is the single source of
truth for the supported modifications. Its Pydantic document stays a passive storage boundary;
the shared :class:`UnimodRegistry` owns the loaded table and every lookup over it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field


class UnimodEntry(BaseModel):
    """One canonical modification record."""

    model_config = ConfigDict(extra="forbid")

    accession: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    target: list[str]  # allowed residues/termini, e.g. ["S", "T", "Y"] for Phospho
    position: str
    mass_delta: float


class _UnimodRegistryDocument(BaseModel):
    """Top-level storage shape of the packaged registry JSON."""

    model_config = ConfigDict(extra="forbid")

    entries: list[UnimodEntry] = Field(min_length=1)


@dataclass(frozen=True)
class UnimodMatch:
    """A canonical Unimod record matched a lookup query."""

    entry: UnimodEntry


@dataclass(frozen=True)
class UnrecognizedUnimodName:
    """No canonical Unimod record matched a name or accession."""

    name: str


@dataclass(frozen=True)
class UnrecognizedUnimodMass:
    """No canonical Unimod record matched a monoisotopic mass."""

    mass_delta: float


class UnimodRegistry:
    """Canonical Unimod entries and every supported lookup over them."""

    __slots__ = ("_by_accession", "_by_name")

    def __init__(self, entries: Iterable[UnimodEntry]) -> None:
        """Index validated entries by accession, canonical name, and aliases.

        Args:
            entries: Validated canonical modification entries.

        Raises:
            ValueError: An accession or normalized name identifies multiple entries.
        """
        by_accession: dict[str, UnimodEntry] = {}
        by_name: dict[str, UnimodEntry] = {}
        for entry in entries:
            if entry.accession in by_accession:
                raise ValueError(f"duplicate accession in Unimod registry: {entry.accession!r}")
            by_accession[entry.accession] = entry
            for name in (entry.accession, entry.name, *entry.aliases):
                normalized = name.strip().casefold()
                previous = by_name.get(normalized)
                if previous is not None and previous.accession != entry.accession:
                    raise ValueError(
                        f"duplicate normalized name in Unimod registry: {name!r} identifies "
                        f"{previous.accession} and {entry.accession}"
                    )
                by_name[normalized] = entry
        self._by_accession: Mapping[str, UnimodEntry] = MappingProxyType(by_accession)
        self._by_name: Mapping[str, UnimodEntry] = MappingProxyType(by_name)

    def resolve(self, accession: str) -> UnimodEntry:
        """Return the canonical record for ``accession`` or raise ``KeyError``."""
        entry = self._by_accession.get(accession)
        if entry is None:
            raise KeyError(
                f"accession {accession!r} not found in unimod_registry.json; "
                f"add it there before referencing it from a parsing rule"
            )
        return entry

    def find_by_name(self, name: str) -> UnimodMatch | UnrecognizedUnimodName:
        """Find a canonical modification by accession, name, or shared synonym."""
        entry = self._by_name.get(name.strip().casefold())
        if entry is None:
            return UnrecognizedUnimodName(name)
        return UnimodMatch(entry)

    def find_by_mass(
        self,
        mass_delta: float,
        *,
        tolerance: float = 0.001,
    ) -> UnimodMatch | UnrecognizedUnimodMass:
        """Find one canonical modification by monoisotopic mass within tolerance.

        Unknown masses return a tagged result. An ambiguous match raises because silently choosing
        one identity would corrupt the search-parameter record.

        Args:
            mass_delta: Monoisotopic mass delta to match.
            tolerance: Maximum absolute difference in daltons.

        Returns:
            A canonical match or an explicit unrecognized-mass result.

        Raises:
            ValueError: The tolerance is negative or several entries match.
        """
        if tolerance < 0:
            raise ValueError("mass tolerance must be non-negative")

        matches = [
            entry
            for entry in self._by_accession.values()
            if math.isclose(
                entry.mass_delta,
                mass_delta,
                rel_tol=0,
                abs_tol=tolerance,
            )
        ]
        if len(matches) > 1:
            accessions = ", ".join(entry.accession for entry in matches)
            raise ValueError(
                f"mass delta {mass_delta} is ambiguous within {tolerance} Da: {accessions}"
            )
        if matches:
            return UnimodMatch(matches[0])
        return UnrecognizedUnimodMass(mass_delta)


_REGISTRY_JSON = resources.files("apb2.parserV2.vendor_params.parsers.shared").joinpath(
    "unimod_registry.json"
)
UNIMOD_REGISTRY = UnimodRegistry(
    _UnimodRegistryDocument.model_validate_json(_REGISTRY_JSON.read_text(encoding="utf-8")).entries
)
"""Shared runtime registry loaded once from the packaged JSON document."""
