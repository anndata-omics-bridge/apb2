"""Axis identity and sourced or computed column entries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Discriminator, Field, RootModel, Tag, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import AxisColumnType, ModelBase
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
    """Expose the modification-stripped peptide from one sequence input."""

    how: Literal["stripped_sequence"]
    name: Literal["ProForma_peptide"] = "ProForma_peptide"
    inputs: list[str] = Field(min_length=1, max_length=1)


class ProformaSequence(ComputedColumnBase):
    """Expose the normalized ProForma peptidoform from one sequence input."""

    how: Literal["proforma_sequence"]
    name: Literal["ProForma_peptidoform"] = "ProForma_peptidoform"
    inputs: list[str] = Field(min_length=1, max_length=1)


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


def _column_kind(value: object) -> str:
    if isinstance(value, Mapping):
        return "source" if "source" in value else str(value.get("how"))
    if isinstance(value, SourcedColumn):
        return "source"
    return str(getattr(value, "how", None))


type ColumnEntry = Annotated[
    Annotated[SourcedColumn, Tag("source")]
    | Annotated[Coalesce, Tag("coalesce")]
    | Annotated[JoinNonempty, Tag("join_nonempty")]
    | Annotated[StrippedSequence, Tag("stripped_sequence")]
    | Annotated[ProformaSequence, Tag("proforma_sequence")]
    | Annotated[ProformaIon, Tag("proforma_ion")]
    | Annotated[ProformaFragment, Tag("proforma_fragment")],
    Discriminator(_column_kind),
]


class LegacyColumnGroup(ModelBase):
    """Schema-0.3 side maps retained only while packaged rules migrate."""

    select: dict[str, str] = Field(default_factory=dict)
    optional_select: dict[str, str] = Field(default_factory=dict)
    types: dict[str, AxisColumnType] = Field(default_factory=dict)
    computed: list[ComputedColumn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_declarations(self) -> LegacyColumnGroup:
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


class ColumnGroup(RootModel[list[ColumnEntry] | LegacyColumnGroup]):
    """One axis's columns, temporarily accepting schema-0.3 side maps."""

    @property
    def entries(self) -> tuple[ColumnEntry, ...]:
        """Column entries in stable authored order."""
        if isinstance(self.root, list):
            return tuple(self.root)
        types = self.root.types
        required = [
            SourcedColumn(name=name, source=source, type=types.get(name, "string"))
            for name, source in self.root.select.items()
        ]
        optional = [
            SourcedColumn(
                name=name,
                source=source,
                type=types.get(name, "string"),
                required=False,
            )
            for name, source in self.root.optional_select.items()
        ]
        computed = {column.name: column for column in self.root.computed}
        selected_names = {source.name for source in (*required, *optional)}
        entries: list[ColumnEntry] = []
        for source in (*required, *optional):
            column = computed.get(source.name)
            entries.append(
                source
                if column is None
                else column.model_copy(
                    update={
                        "source": source.source,
                        "type": source.type,
                        "required": source.required,
                    }
                )
            )
        entries.extend(column for column in self.root.computed if column.name not in selected_names)
        return tuple(entries)

    @property
    def sourced(self) -> tuple[SourcedColumn, ...]:
        """Physical selections in stable authored order."""
        return tuple(
            SourcedColumn(
                name=entry.name,
                source=entry.source,
                type=entry.type,
                required=entry.required,
                roles=entry.roles,
            )
            for entry in self.entries
            if entry.source is not None
        )

    @property
    def computed(self) -> tuple[ComputedColumn, ...]:
        """Computed entries in execution order."""
        return tuple(entry for entry in self.entries if not isinstance(entry, SourcedColumn))

    @property
    def names(self) -> tuple[str, ...]:
        """All logical names in stable authored order."""
        return tuple(entry.name for entry in self.entries)

    @model_validator(mode="after")
    def _entry_names_and_roles_are_unique(self) -> ColumnGroup:
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("column entry names must be unique")
        for entry in self.entries:
            if len(entry.roles) != len(set(entry.roles)):
                raise ValueError(f"column {entry.name!r} roles must be unique")
        return self


class LongColumns(ModelBase):
    """Observation and variable declarations for a long source."""

    obs: ColumnGroup
    var: ColumnGroup


class WideColumns(ModelBase):
    """Variable declarations for a wide source."""

    var: ColumnGroup


class ColumnRoles(ModelBase):
    """Schema-0.3 role map retained only while packaged rules migrate."""

    protein_assignment: str | None = Field(default=None, min_length=1)
    fasta_accessions: str | None = Field(default=None, min_length=1)
