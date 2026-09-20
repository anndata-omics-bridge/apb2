"""Axis parameters: what one small axis frame selects, computes, and calls its identity.

``AxisColumnSelection`` describes the physical source of one logical column. Computations
are executable objects in the compiler contract, not configuration copies. ``AxisKeyPlan`` says which columns
carry identity, at which of the three stages. ``ModificationConfig`` carries everything one
modification normalizer needs, already resolved — the accession lookups happened during rule
projection, so nothing here consults a registry or a Unimod file.

Every value is a plain immutable record. Normalizer ``kind`` tags support provenance;
normalizers reuse these settings directly but never dispatch on their tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type AxisLogicalType = Literal["string", "integer", "number", "boolean"]
"""The logical type a declared axis column is coerced to on its small axis frame."""

type ModificationTokenPosition = Literal[
    "before_residue", "after_residue", "n_term", "c_term", "embedded", "unknown"
]
type UnknownModificationPolicy = Literal["preserve", "drop", "error"]


@dataclass(frozen=True, slots=True)
class AxisColumnSelection:
    """One declared axis column read from one physical source under one logical type."""

    name: str
    source: str
    logical_type: AxisLogicalType


# ---------------------------------------------------------------------------- axis identity


@dataclass(frozen=True, slots=True)
class AxisKeyPlan:
    """The three column sets one axis identity passes through.

    ``raw_key_columns`` distinguishes physical rows before any coercion or computation;
    ``key_input_columns`` are the direct logical inputs of the authored key; and
    ``final_key_columns`` is the authored identity that reaches the result.
    """

    raw_key_columns: tuple[str, ...]
    key_input_columns: tuple[str, ...]
    final_key_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AxisSourcePlan:
    """What one axis takes off the physical table: its raw keys and its payload columns."""

    keys: AxisKeyPlan
    payload_sources: tuple[str, ...]


# ------------------------------------------------------------------------------ modifications


@dataclass(frozen=True, slots=True)
class ModificationMapEntry:
    """One vendor token and the canonical modification identity it resolved to."""

    token: str
    name: str
    accession: str
    target: tuple[str, ...]
    position: str
    mass_delta: float


@dataclass(frozen=True, slots=True)
class TokenRegexModificationConfig:
    """Inline modification tokens extracted by one regex (``PEPM[15.9949]TIDE``)."""

    kind: Literal["token_regex"]
    token_pattern: str
    token_position: ModificationTokenPosition
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    entries: tuple[ModificationMapEntry, ...]


@dataclass(frozen=True, slots=True)
class SiteListModificationConfig:
    """Parallel modification-name and site columns beside a bare sequence."""

    kind: Literal["site_list"]
    delimiter: str
    site_base: int
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    entries: tuple[ModificationMapEntry, ...]


@dataclass(frozen=True, slots=True)
class EmbeddedSiteListModificationConfig:
    """Delimited modification entries that each contain their own site."""

    kind: Literal["embedded_site_list"]
    delimiter: str
    entry_pattern: str
    site_base: int
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    entries: tuple[ModificationMapEntry, ...]


type ModificationConfig = (
    TokenRegexModificationConfig | SiteListModificationConfig | EmbeddedSiteListModificationConfig
)
