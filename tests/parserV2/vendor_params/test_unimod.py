"""Canonical Unimod document loading and runtime lookup behavior."""

from __future__ import annotations

import pytest

from apb2.parserV2.vendor_params.parsers.shared.unimod import (
    UNIMOD_REGISTRY,
    UnimodEntry,
    UnimodMatch,
    UnimodRegistry,
    UnrecognizedUnimodMass,
    UnrecognizedUnimodName,
)


def _entry(
    accession: str,
    name: str,
    mass_delta: float,
    *,
    aliases: list[str] | None = None,
) -> UnimodEntry:
    """Construct one focused validated registry entry."""
    return UnimodEntry(
        accession=accession,
        name=name,
        aliases=[] if aliases is None else aliases,
        target=["M"],
        position="Anywhere",
        mass_delta=mass_delta,
    )


@pytest.mark.parametrize(
    ("identity", "accession"),
    [
        ("UNIMOD:1", "UNIMOD:1"),
        ("oxidation", "UNIMOD:35"),
        (" Carbamidomethylation ", "UNIMOD:4"),
    ],
)
def test_packaged_json_loads_and_resolves_names(identity: str, accession: str) -> None:
    result = UNIMOD_REGISTRY.find_by_name(identity)

    assert isinstance(result, UnimodMatch)
    assert result.entry.accession == accession
    assert UNIMOD_REGISTRY.resolve(accession) is result.entry


def test_registry_resolves_mass_and_reports_unknown_queries() -> None:
    mass_match = UNIMOD_REGISTRY.find_by_mass(15.9949)

    assert isinstance(mass_match, UnimodMatch)
    assert mass_match.entry.accession == "UNIMOD:35"
    assert UNIMOD_REGISTRY.find_by_name("vendor-specific") == UnrecognizedUnimodName(
        "vendor-specific"
    )
    assert UNIMOD_REGISTRY.find_by_mass(999.0) == UnrecognizedUnimodMass(999.0)
    with pytest.raises(KeyError, match=r"unimod_registry\.json"):
        UNIMOD_REGISTRY.resolve("UNIMOD:999999")


def test_registry_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="mass tolerance must be non-negative"):
        UNIMOD_REGISTRY.find_by_mass(15.994915, tolerance=-0.001)


def test_registry_rejects_ambiguous_mass() -> None:
    registry = UnimodRegistry(
        (
            _entry("UNIMOD:TEST1", "First", 10.0),
            _entry("UNIMOD:TEST2", "Second", 10.0005),
        )
    )

    with pytest.raises(ValueError, match="UNIMOD:TEST1, UNIMOD:TEST2"):
        registry.find_by_mass(10.0, tolerance=0.001)


def test_registry_rejects_duplicate_accessions() -> None:
    with pytest.raises(ValueError, match="duplicate accession"):
        UnimodRegistry(
            (
                _entry("UNIMOD:TEST", "First", 10.0),
                _entry("UNIMOD:TEST", "Second", 20.0),
            )
        )


def test_registry_rejects_names_shared_by_different_entries() -> None:
    with pytest.raises(ValueError, match="duplicate normalized name"):
        UnimodRegistry(
            (
                _entry("UNIMOD:TEST1", "First", 10.0, aliases=["shared"]),
                _entry("UNIMOD:TEST2", "Shared", 20.0),
            )
        )
