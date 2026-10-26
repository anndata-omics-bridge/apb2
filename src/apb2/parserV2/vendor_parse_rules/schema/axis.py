"""Axis identity and sourced or computed column entries."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    Note,
    UnknownPolicy,
    VarDuplicateMode,
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
    """Facts shared by every computed column entry."""

    source: str | None = None
    type: AxisColumnType = "string"
    required: bool = True
    roles: list[SemanticRole] = Field(default_factory=list)
    note: Note = None


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
    """Normalize all declared logical inputs using named syntax and a token map."""

    how: Literal["proforma_sequence"]
    name: Literal["ProForma_peptidoform", "ProForma_peptidoform_unlocalized"] = (
        "ProForma_peptidoform"
    )
    inputs: list[str] = Field(min_length=1, max_length=3)
    syntax: str
    modification_map: str
    case_sensitive: bool = False
    unknown_policy: UnknownPolicy = "preserve"


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
    """Mark the rows the vendor itself marks; never missing.

    ``equals`` marks a row whose input value equals it, ``prefix`` one whose input, or a
    member of its ``separator``-joined list, starts with it. Without inputs the vendor writes
    no such rows and every row is unmarked. With several inputs, any match marks the row, and
    inputs a source lacks are left out.
    """

    source: None = None
    type: Literal["boolean"] = "boolean"
    required: Literal[True] = True
    roles: list[SemanticRole] = Field(default_factory=list, max_length=0)
    inputs: list[str] = Field(default_factory=list)
    equals: str | None = Field(default=None, min_length=1)
    prefix: str | None = Field(default=None, min_length=1)
    separator: str | None = Field(default=None, min_length=1)
    note: Note = None

    @model_validator(mode="after")
    def _one_marking(self) -> _Flag:
        markings = [field for field in ("equals", "prefix") if getattr(self, field) is not None]
        if self.inputs and len(markings) != 1:
            raise ValueError(
                "a decoy or contaminant column with inputs declares exactly one of equals or prefix"
            )
        if not self.inputs and markings:
            raise ValueError(
                "a decoy or contaminant column without inputs declares neither equals nor prefix"
            )
        if self.separator is not None and self.prefix is None:
            raise ValueError(
                "a decoy or contaminant column declares a separator only with a prefix"
            )
        return self


class DecoyFlag(_Flag):
    """``apb_Decoy``: the rows the vendor marks as decoys."""

    how: Literal["decoy"]
    name: Literal["apb_Decoy"] = "apb_Decoy"


class ContaminantFlag(_Flag):
    """``apb_Contaminant``: the rows the vendor marks as contaminants it flags or adds itself."""

    how: Literal["contaminant"]
    name: Literal["apb_Contaminant"] = "apb_Contaminant"


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
    | ContaminantFlag,
    Field(discriminator="how"),
]


class ColumnDuplicates(ModelBase):
    """How a var column resolves the values a feature's repeated rows give it.

    A long report repeats a feature's values in every row of it, one per run. ``keep_first``
    keeps the first nonblank value and reports features whose rows disagree. ``min`` and
    ``max`` keep the smallest or largest number, for a tool that writes a placeholder into
    some of a feature's rows, such as DIA-NN's protein-group q-value of 1.0. ``keep_best``
    takes the value of the row whose ``by`` column is ``best``, so the columns ranked by one
    score describe the same row, such as the protein set a shared peptide is assigned to.
    """

    mode: VarDuplicateMode = "keep_first"
    by: str | None = Field(
        default=None, description="keep_best: the numeric var column that ranks repeated rows"
    )
    best: Literal["highest", "lowest"] = Field(
        default="highest", description="keep_best: which value of the 'by' column wins"
    )

    @model_validator(mode="after")
    def _ranking_only_for_keep_best(self) -> ColumnDuplicates:
        if (self.mode == "keep_best") != (self.by is not None):
            raise ValueError("'by' names the ranking column exactly when mode is keep_best")
        if self.mode != "keep_best" and self.best != "highest":
            raise ValueError("'best' applies only to keep_best")
        return self


class SourcedColumn(ModelBase):
    """One logical axis column read from one physical vendor column.

    A protein-list column may declare the vendor's own further list ``separator``, such as
    PEAKS's ``:``; ``;`` and ``,`` separate every protein list already. Only a var column
    declares ``duplicates``; without it, a feature keeps its first nonblank value.
    """

    name: str
    source: str
    type: AxisColumnType = "string"
    required: bool = True
    roles: list[SemanticRole] = Field(default_factory=list)
    separator: str | None = Field(default=None, min_length=1)
    duplicates: ColumnDuplicates | None = None
    note: Note = None

    @model_validator(mode="after")
    def _separator_lists_proteins(self) -> SourcedColumn:
        if self.separator is not None and not {"fasta_accessions", "protein_assignment"} & {
            *self.roles
        }:
            raise ValueError(f"{self.name}: only a protein-list column declares a separator")
        return self

    @model_validator(mode="after")
    def _extremes_are_numbers(self) -> SourcedColumn:
        mode = self.duplicates_mode
        if mode in {"min", "max"} and self.type not in {"number", "integer"}:
            raise ValueError(f"{self.name}: duplicates mode {mode!r} needs a numeric column")
        return self

    @property
    def duplicates_mode(self) -> VarDuplicateMode:
        """The declared mode, ``keep_first`` without a ``duplicates`` block."""
        return "keep_first" if self.duplicates is None else self.duplicates.mode


type ColumnEntry = SourcedColumn | ComputedColumn


type ColumnGroup = list[ColumnEntry]


def computed_columns(group: ColumnGroup) -> tuple[ComputedColumn, ...]:
    """Computed entries in execution order."""
    return tuple(entry for entry in group if not isinstance(entry, SourcedColumn))


class LongColumns(ModelBase):
    """Observation and variable declarations for a long source."""

    obs: ColumnGroup
    var: ColumnGroup


class WideColumns(ModelBase):
    """Variable declarations for a wide source."""

    var: ColumnGroup
