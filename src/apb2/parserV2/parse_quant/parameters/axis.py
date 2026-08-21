"""Axis parameters: what one small axis frame selects, computes, and calls its identity.

Three questions, three groups of values. ``AxisColumnSelection`` and the computed-column
configurations say how a declared column is materialized. ``AxisKeyPlan`` says which columns
carry identity, at which of the three stages. ``ModificationConfig`` carries everything one
modification normalizer needs, already resolved — the accession lookups happened during rule
projection, so nothing here consults a registry or a Unimod file.

Every value is a plain immutable record. ``kind`` tags exist only so the composition root can
construct behaviour once; no computation receives one.
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


# ------------------------------------------------------------------------ computed columns


@dataclass(frozen=True, slots=True)
class CoalesceColumnConfig:
    """Take the first non-null input value in declaration order."""

    kind: Literal["coalesce"]
    name: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JoinNonemptyColumnConfig:
    """Join the non-empty input values with a separator."""

    kind: Literal["join_nonempty"]
    name: str
    inputs: tuple[str, ...]
    separator: str


@dataclass(frozen=True, slots=True)
class StrippedSequenceColumnConfig:
    """Expose the modification-stripped peptide a normalizer already derived."""

    kind: Literal["stripped_sequence"]
    name: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProformaSequenceColumnConfig:
    """Expose the ProForma peptidoform a normalizer already derived."""

    kind: Literal["proforma_sequence"]
    name: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProformaIonColumnConfig:
    """Combine a peptidoform and a positive integer charge into a ProForma ion."""

    kind: Literal["proforma_ion"]
    name: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProformaFragmentColumnConfig:
    """Combine a ProForma ion and a fragment label into a ProForma fragment."""

    kind: Literal["proforma_fragment"]
    name: str
    inputs: tuple[str, ...]


type ComputedColumnConfig = (
    CoalesceColumnConfig
    | JoinNonemptyColumnConfig
    | StrippedSequenceColumnConfig
    | ProformaSequenceColumnConfig
    | ProformaIonColumnConfig
    | ProformaFragmentColumnConfig
)


@dataclass(frozen=True, slots=True)
class AxisColumnDeclaration:
    """One axis's authored columns before any physical source is known.

    Required and optional selections are separate collections rather than one collection of
    flagged records: an axis key may never be optional, and source resolution prunes only
    the optional side.
    """

    required_selections: tuple[AxisColumnSelection, ...]
    optional_selections: tuple[AxisColumnSelection, ...]
    computed: tuple[ComputedColumnConfig, ...]
    declared_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AxisMaterializationConfig:
    """One materialization phase: resolved selections and computers, in execution order."""

    selections: tuple[AxisColumnSelection, ...]
    computers: tuple[ComputedColumnConfig, ...]


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


@dataclass(frozen=True, slots=True)
class ResolvedAxisColumnPlan:
    """One axis fully resolved against physical evidence.

    ``skipped`` records the optional outputs this source cannot provide. It is evidence for
    a caller and for tests; the runtime phases contain only operations that can execute, so
    no strategy ever consults it.
    """

    source: AxisSourcePlan
    key_phase: AxisMaterializationConfig
    output_phase: AxisMaterializationConfig
    outputs: tuple[str, ...]
    skipped: frozenset[str]


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
    source_column: str
    token_pattern: str
    token_position: ModificationTokenPosition
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    proforma_output: str
    stripped_output: str
    entries: tuple[ModificationMapEntry, ...]


@dataclass(frozen=True, slots=True)
class SiteListModificationConfig:
    """Parallel modification-name and site columns beside a bare sequence."""

    kind: Literal["site_list"]
    sequence_column: str
    modification_column: str
    site_column: str
    delimiter: str
    site_base: int
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    proforma_output: str
    stripped_output: str
    entries: tuple[ModificationMapEntry, ...]


type ModificationConfig = TokenRegexModificationConfig | SiteListModificationConfig
