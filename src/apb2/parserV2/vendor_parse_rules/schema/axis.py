"""Axis identity and sourced or computed column entries."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import AxisColumnType, ModelBase, UnknownPolicy
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
    name: Literal["ProForma_peptidoform"] = "ProForma_peptidoform"
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


type ComputedColumn = Annotated[
    Coalesce | JoinNonempty | StrippedSequence | ProformaSequence | ProformaIon | ProformaFragment,
    Field(discriminator="how"),
]


class SourcedColumn(ModelBase):
    """One logical axis column read from one physical vendor column."""

    name: str
    source: str
    type: AxisColumnType = "string"
    required: bool = True
    roles: list[SemanticRole] = Field(default_factory=list)


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
