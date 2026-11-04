"""Axis identity and sourced or computed column entries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    Note,
    PerFeature,
    UnknownPolicy,
)
from apb2.parserV2.vendor_parse_rules.schema.roles import SemanticRole


class Axis(ModelBase):
    """The declared columns that identify observations and variables."""

    obs_keys: list[str] = Field(min_length=1)
    var_keys: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _keys_are_unique(self) -> Axis:
        if len(self.obs_keys) != len(set(self.obs_keys)):
            raise ValueError("axis.obs_keys must be unique")
        if len(self.var_keys) != len(set(self.var_keys)):
            raise ValueError("axis.var_keys must be unique")
        return self


class ComputedColumnBase(ModelBase):
    """Facts shared by every computed column entry.

    A computed entry declares neither a type nor whether it is required; neither is a choice.
    """

    source: str | None = None
    roles: list[SemanticRole] = Field(default_factory=list)
    note: Note = None

    @property
    def required(self) -> Literal[True]:
        """A computed entry's own source, where it has one, must exist.

        An entry whose inputs a source lacks is still left out, as its inputs decide that.
        """
        return True

    @property
    def type(self) -> Literal["string"]:
        """Every computed value is text: a ProForma identifier, a coalesced or joined value."""
        return "string"


class Coalesce(ComputedColumnBase):
    """Take the first non-null input value in declaration order."""

    how: Literal["coalesce"]
    name: str
    inputs: list[str] = Field(min_length=2)


class JoinNonempty(ComputedColumnBase):
    """Join non-empty input values with a separator."""

    how: Literal["join_nonempty"]
    name: str
    inputs: list[str] = Field(min_length=2)
    separator: str = Field(min_length=1)


class StrippedSequence(ComputedColumnBase):
    """Strip one logical sequence input using a named grammar, without a token map."""

    how: Literal["stripped_sequence"]
    name: Literal["ProForma_peptide"] = "ProForma_peptide"
    inputs: list[str] = Field(min_length=1, max_length=1)
    syntax: str


class ProformaSequence(ComputedColumnBase):
    """Normalize all declared logical inputs using named syntax and the level's token map."""

    how: Literal["proforma_sequence"]
    name: Literal["ProForma_peptidoform", "ProForma_peptidoform_unlocalized"]
    inputs: list[str] = Field(min_length=1, max_length=3)
    syntax: str
    case_sensitive: bool
    unknown_policy: UnknownPolicy


class ProformaIon(ComputedColumnBase):
    """Combine a peptidoform and charge into a ProForma ion."""

    how: Literal["proforma_ion"]
    name: Literal["ProForma_ion"] = "ProForma_ion"
    inputs: list[str] = Field(min_length=2, max_length=2)


class ProformaFragment(ComputedColumnBase):
    """Combine a ProForma ion and fragment label into a ProForma fragment."""

    how: Literal["proforma_fragment"]
    name: Literal["ProForma_fragment"] = "ProForma_fragment"
    inputs: list[str] = Field(min_length=2, max_length=2)


class _Flag(ModelBase):
    """A decoy or contaminant column: a boolean every row carries, never missing."""

    source: None = None
    type: Literal["boolean"] = "boolean"
    required: Literal[True] = True
    roles: list[SemanticRole] = Field(default_factory=list, max_length=0)


class _Marking(_Flag):
    """Mark the rows the vendor itself marks.

    ``equals`` marks a row whose input value equals it, ``prefix`` one whose input, or a
    member of its ``separator``-joined list, starts with it. With several inputs, any match
    marks the row, and inputs a source lacks are left out.
    """

    inputs: list[str] = Field(min_length=1)
    equals: str | None = Field(default=None, min_length=1)
    prefix: str | None = Field(default=None, min_length=1)
    separator: str | None = Field(default=None, min_length=1)
    note: Note = None

    @model_validator(mode="after")
    def _one_marking(self) -> _Marking:
        markings = [field for field in ("equals", "prefix") if getattr(self, field) is not None]
        if len(markings) != 1:
            raise ValueError(
                "a decoy or contaminant column declares exactly one of equals or prefix"
            )
        if self.separator is not None and self.prefix is None:
            raise ValueError(
                "a decoy or contaminant column declares a separator only with a prefix"
            )
        return self


class DecoyFlag(_Marking):
    """``apb_Decoy``: the rows the vendor marks as decoys."""

    how: Literal["decoy"]
    name: Literal["apb_Decoy"] = "apb_Decoy"


class ContaminantFlag(_Marking):
    """``apb_Contaminant``: the rows the vendor marks as contaminants it flags or adds itself."""

    how: Literal["contaminant"]
    name: Literal["apb_Contaminant"] = "apb_Contaminant"


class UnmarkedFlag(_Flag):
    """``apb_Decoy`` or ``apb_Contaminant`` for a report with no column marking such rows.

    The result still carries the column, every row ``false``.
    """

    inputs: list[str] = Field(default_factory=list, max_length=0)
    note: Note = None
    how: Literal["unmarked"]
    name: Literal["apb_Decoy", "apb_Contaminant"]


FLAG_COLUMNS = ("apb_Decoy", "apb_Contaminant")
"""The vendor markings every rule declares on its var axis, exactly once each."""

FIXED_COLUMNS = (
    "ProForma_peptide",
    "ProForma_peptidoform",
    "ProForma_peptidoform_unlocalized",
    "ProForma_ion",
    "ProForma_fragment",
    *FLAG_COLUMNS,
)
"""The column names APB2 fixes for every vendor: the ProForma identifiers and the flags."""


type ComputedColumn = Annotated[
    Coalesce
    | JoinNonempty
    | StrippedSequence
    | ProformaSequence
    | ProformaIon
    | ProformaFragment
    | DecoyFlag
    | ContaminantFlag
    | UnmarkedFlag,
    Field(discriminator="how"),
]


class RankRows(ModelBase):
    """The var column that ranks a feature's repeated rows, for its ``best_row`` columns.

    The level keeps, per feature, the row whose ``by`` column is ``best`` before the feature
    table and the layers are built, so the ``best_row`` columns and every layer describe the
    same row, such as the protein set a shared peptide is assigned to.
    """

    by: str = Field(description="the numeric var column that ranks a feature's repeated rows")
    best: Literal["highest", "lowest"] = Field(description="which value of the 'by' column wins")


class SourcedColumn(ModelBase):
    """One logical axis column read from one physical vendor column.

    A protein-list column may declare the vendor's own further list ``separator``, such as
    PEAKS's ``:``; ``;`` and ``,`` separate every protein list already. ``type`` and
    ``required`` come from the entry or from its table's ``defaults``.
    """

    name: str
    source: str
    type: AxisColumnType
    required: bool
    roles: list[SemanticRole] = Field(default_factory=list)
    separator: str | None = Field(default=None, min_length=1)
    note: Note = None

    @model_validator(mode="after")
    def _separator_lists_proteins(self) -> SourcedColumn:
        if self.separator is not None and not {"fasta_accessions", "protein_assignment"} & {
            *self.roles
        }:
            raise ValueError(f"{self.name}: only a protein-list column declares a separator")
        return self


class SourcedVarColumn(SourcedColumn):
    """A var column read from one vendor column, and how it resolves a feature's repeated rows.

    A long report repeats a feature in every row of it, one per run. ``per_feature`` ``first``
    keeps the first nonblank value and reports features whose rows disagree. ``min`` and
    ``max`` keep the smallest or largest number, for a tool that writes a placeholder into some
    of a feature's rows, such as DIA-NN's protein-group q-value of 1.0. ``best_row`` takes the
    value from the row the level's ``rank_rows`` keeps. An observation has no repeated rows to
    resolve, so obs columns carry no ``per_feature``.
    """

    per_feature: PerFeature
    # Text the vendor writes for a missing value, such as MaxQuant's NaN; absent means none.
    missing_tokens: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _extremes_are_numbers(self) -> SourcedVarColumn:
        mode = self.per_feature
        if mode in {"min", "max"} and self.type not in {"number", "integer"}:
            raise ValueError(f"{self.name}: per_feature {mode!r} needs a numeric column")
        if any(not token.strip() for token in self.missing_tokens):
            raise ValueError("missing_tokens declares blank text, which is already missing")
        return self


type ColumnEntry = SourcedColumn | ComputedColumn
"""An obs entry, or any entry for a reader of either group."""

type VarEntry = SourcedVarColumn | ComputedColumn
"""A var entry: its sourced columns say how they resolve a feature's repeated rows."""

type ColumnGroup = Sequence[ColumnEntry]
"""Either group, for a reader of both: var entries are column entries too."""


def computed_columns(group: ColumnGroup) -> tuple[ComputedColumn, ...]:
    """Computed entries in execution order."""
    return tuple(entry for entry in group if not isinstance(entry, SourcedColumn))


class LongColumns(ModelBase):
    """Observation and variable declarations for a long source."""

    obs: list[ColumnEntry]
    var: list[VarEntry]


class WideColumns(ModelBase):
    """Variable declarations for a wide source."""

    var: list[VarEntry]
