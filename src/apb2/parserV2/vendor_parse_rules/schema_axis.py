"""Axis identity, selected columns, and computed-column declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema_base import AxisColumnType, ModelBase


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


class Coalesce(ModelBase):
    """Take the first non-null input value in declaration order."""

    how: Literal["coalesce"]
    name: str
    inputs: list[str] = Field(min_length=2)


class JoinNonempty(ModelBase):
    """Join non-empty input values with a separator."""

    how: Literal["join_nonempty"]
    name: str
    inputs: list[str] = Field(min_length=2)
    separator: str = Field(min_length=1)


class StrippedSequence(ModelBase):
    """Expose the modification-stripped peptide from one sequence input."""

    how: Literal["stripped_sequence"]
    name: Literal["ProForma_peptide"] = "ProForma_peptide"
    inputs: list[str] = Field(min_length=1, max_length=1)


class ProformaSequence(ModelBase):
    """Expose the normalized ProForma peptidoform from one sequence input."""

    how: Literal["proforma_sequence"]
    name: Literal["ProForma_peptidoform"] = "ProForma_peptidoform"
    inputs: list[str] = Field(min_length=1, max_length=1)


class ProformaIon(ModelBase):
    """Combine a peptidoform and charge into a ProForma ion."""

    how: Literal["proforma_ion"]
    name: Literal["ProForma_ion"] = "ProForma_ion"
    inputs: list[str] = Field(min_length=2, max_length=2)


class ProformaFragment(ModelBase):
    """Combine a ProForma ion and fragment label into a ProForma fragment."""

    how: Literal["proforma_fragment"]
    name: Literal["ProForma_fragment"] = "ProForma_fragment"
    inputs: list[str] = Field(min_length=2, max_length=2)


type ComputedColumn = Annotated[
    Coalesce | JoinNonempty | StrippedSequence | ProformaSequence | ProformaIon | ProformaFragment,
    Field(discriminator="how"),
]


class ColumnGroup(ModelBase):
    """Required, optional, typed, and computed columns for one axis."""

    select: dict[str, str] = Field(default_factory=dict)
    optional_select: dict[str, str] = Field(default_factory=dict)
    types: dict[str, AxisColumnType] = Field(default_factory=dict)
    computed: list[ComputedColumn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_declarations(self) -> ColumnGroup:
        both = sorted(set(self.select) & set(self.optional_select))
        if both:
            raise ValueError(f"column name(s) declared in both selections: {both}")
        unknown = sorted(set(self.types) - (set(self.select) | set(self.optional_select)))
        if unknown:
            raise ValueError(f"types must name selected columns; unknown: {unknown}")
        names = [column.name for column in self.computed]
        if len(names) != len(set(names)):
            raise ValueError("computed column names must be unique")
        return self


class LongColumns(ModelBase):
    """Observation and variable declarations for a long source."""

    obs: ColumnGroup
    var: ColumnGroup


class WideColumns(ModelBase):
    """Variable declarations for a wide source."""

    var: ColumnGroup


class ColumnRoles(ModelBase):
    """Semantic variable columns needed by downstream canonical consumers."""

    protein_assignment: str | None = Field(default=None, min_length=1)
    fasta_accessions: str | None = Field(default=None, min_length=1)


def group_names(group: ColumnGroup) -> list[str]:
    """All logical names a column group declares, in stable declaration order."""
    computed = (column.name for column in group.computed)
    return list(dict.fromkeys([*group.select, *group.optional_select, *computed]))
