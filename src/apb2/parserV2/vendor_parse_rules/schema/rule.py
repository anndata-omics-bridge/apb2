"""Complete effective-rule declarations and essential cross-block invariants."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from apb2.parserV2.vendor_parse_rules.schema.annotation import SampleAnnotation
from apb2.parserV2.vendor_parse_rules.schema.axis import (
    Axis,
    Coalesce,
    ColumnGroup,
    ComputedColumn,
    JoinNonempty,
    LongColumns,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
    WideColumns,
    computed_columns,
)
from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    QuantificationLevel,
    SchemaVersion,
)
from apb2.parserV2.vendor_parse_rules.schema.base_modifications import (
    Modifications,
    modification_outputs,
)
from apb2.parserV2.vendor_parse_rules.schema.fragments import Fragments
from apb2.parserV2.vendor_parse_rules.schema.measurements import Measurements
from apb2.parserV2.vendor_parse_rules.schema.parameters import (
    ConditionValue,
    SearchParameterField,
    SearchParameterOverride,
)
from apb2.parserV2.vendor_parse_rules.schema.roles import (
    ROLE_CONFIG,
    RoleOwner,
    SemanticRole,
)

_SAMPLE_GROUP = "sample"


class _RuleCore(ModelBase):
    """Fields shared by long and wide effective rules."""

    schema_version: SchemaVersion
    file_version: str
    software_name: str
    software_version_pattern: str
    quantification_level: QuantificationLevel
    axis: Axis
    measurements: Measurements
    modifications: Modifications | None = None
    requires_search_parameters: dict[SearchParameterField, ConditionValue] = Field(
        default_factory=dict
    )
    search_parameter_overrides: list[SearchParameterOverride] = Field(default_factory=list)
    sample_annotation: SampleAnnotation | None = None

    @model_validator(mode="after")
    def _core_consistency(self) -> _RuleCore:
        names = [layer.name for layer in self.measurements.layers]
        if len(names) != len(set(names)):
            raise ValueError("measurement layer names must be unique")
        if self.measurements.primary_layer not in set(names):
            raise ValueError(
                f"measurements.primary_layer={self.measurements.primary_layer!r} matches no "
                f"layer; available: {sorted(names)}"
            )
        _check_role_owners(
            "layer", (role for layer in self.measurements.layers for role in layer.roles)
        )
        return self


class LongRule(_RuleCore):
    """One row per observation-variable pair with exact physical source names."""

    shape: Literal["long"]
    columns: LongColumns
    fragments: Fragments | None = None

    @model_validator(mode="after")
    def _column_consistency(self) -> LongRule:
        if self.fragments is not None and self.quantification_level != "fragment":
            raise ValueError("fragments are valid only for quantification_level='fragment'")
        _check_column_group(self.axis.obs_keys, self.columns.obs, "obs")
        _check_column_group(self.axis.var_keys, self.columns.var, "var")
        _check_computed_columns(self, self.columns.var)
        self._check_obs_computed_columns(self.columns.obs)
        _check_derived_not_selected(self.modifications, (self.columns.obs, self.columns.var))
        _check_one_type_per_source((self.columns.obs, self.columns.var))
        return self

    @staticmethod
    def _check_obs_computed_columns(obs: ColumnGroup) -> None:
        sequence_derived = [
            column.name
            for column in computed_columns(obs)
            if not isinstance(column, Coalesce | JoinNonempty)
        ]
        if sequence_derived:
            raise ValueError(f"sequence-derived computed columns are var-only: {sequence_derived}")
        available = {column.name for column in obs if column.source is not None}
        for column in computed_columns(obs):
            missing = [source for source in column.inputs if source not in available]
            if missing:
                raise ValueError(
                    f"computed column {column.name!r} references undeclared obs columns: {missing}"
                )
            available.add(column.name)


class WideRule(_RuleCore):
    """One row per variable with observations captured from layer headers."""

    shape: Literal["wide"]
    columns: WideColumns
    fragments: None = None

    @model_validator(mode="after")
    def _column_consistency(self) -> WideRule:
        for layer in self.measurements.layers:
            try:
                pattern = re.compile(layer.source)
            except re.error as error:
                raise ValueError(
                    f"Layer {layer.name!r}: wide source must be a valid regex: {error}"
                ) from error
            if _SAMPLE_GROUP not in pattern.groupindex:
                raise ValueError(
                    f"Layer {layer.name!r}: wide source must contain "
                    f"'(?P<{_SAMPLE_GROUP}>...)'; got {layer.source!r}"
                )
        _check_column_group(self.axis.var_keys, self.columns.var, "var")
        _check_computed_columns(self, self.columns.var)
        _check_derived_not_selected(self.modifications, (self.columns.var,))
        _check_one_type_per_source((self.columns.var,))
        return self


type Rule = Annotated[LongRule | WideRule, Field(discriminator="shape")]

_RULE_ADAPTER: TypeAdapter[LongRule | WideRule] = TypeAdapter(Rule)


def validate_rule(payload: object) -> LongRule | WideRule:
    """Validate one composed effective-rule payload."""
    return _RULE_ADAPTER.validate_python(payload)


def rule_json_schema() -> dict[str, object]:
    """Return the JSON Schema for complete schema-0.4 effective rules."""
    return _RULE_ADAPTER.json_schema()


def _check_column_group(keys: list[str], group: ColumnGroup, owner: RoleOwner) -> None:
    names = [entry.name for entry in group]
    if len(set(names)) != len(names):
        raise ValueError("column entry names must be unique")
    declared = set(names)
    missing = [key for key in keys if key not in declared]
    if missing:
        raise ValueError(f"axis.{owner}_keys must be declared in columns.{owner}: {missing}")
    optional_names = {
        column.name for column in group if column.source is not None and not column.required
    }
    optional = [key for key in keys if key in optional_names]
    if optional:
        raise ValueError(f"axis.{owner}_keys must not be optional: {optional}")
    role_columns: list[tuple[SemanticRole, str]] = [
        (role, entry.name) for entry in group for role in entry.roles
    ]
    _check_role_owners(owner, (role for role, _name in role_columns))
    if len(dict(role_columns)) != len(role_columns):
        raise ValueError(f"roles must be unique on columns.{owner}")


def _check_role_owners(owner: RoleOwner, roles: Iterable[SemanticRole]) -> None:
    invalid = sorted(set(roles) - ROLE_CONFIG[owner])
    if invalid:
        raise ValueError(f"roles are not allowed on {owner}: {invalid}")


def _check_derived_not_selected(
    modifications: Modifications | None,
    groups: tuple[ColumnGroup, ...],
) -> None:
    if modifications is None:
        return
    selected = {
        column.source for group in groups for column in group if column.source is not None
    } & modification_outputs(modifications)
    if selected:
        raise ValueError(
            f"derived modification columns belong in computed, not select: {sorted(selected)}"
        )


def _check_one_type_per_source(groups: tuple[ColumnGroup, ...]) -> None:
    declared: dict[str, AxisColumnType] = {}
    for group in groups:
        for column in group:
            if column.source is None:
                continue
            if declared.setdefault(column.source, column.type) != column.type:
                raise ValueError(
                    f"vendor source {column.source!r} has conflicting logical types: "
                    f"{declared[column.source]!r} and {column.type!r}"
                )


def _check_computed_columns(rule: LongRule | WideRule, var: ColumnGroup) -> None:
    available = {column.name for column in var if column.source is not None}
    if rule.fragments is not None:
        available.add(rule.fragments.label_output)
    for column in computed_columns(var):
        missing = [source for source in column.inputs if source not in available]
        if missing:
            raise ValueError(
                f"computed column {column.name!r} references undeclared var columns: {missing}"
            )
        _check_computed_column(rule, column, var)
        available.add(column.name)


def _check_derived_sequence_column(
    rule: _RuleCore,
    column: StrippedSequence | ProformaSequence,
) -> None:
    if rule.modifications is None:
        raise ValueError(f"how={column.how!r} requires a modifications block")
    if (
        isinstance(column, ProformaSequence)
        and rule.modifications.output_column != "proforma_sequence"
    ):
        raise ValueError(
            "how='proforma_sequence' reads 'proforma_sequence', but modifications produces "
            f"{rule.modifications.output_column!r}"
        )


def _check_computed_column(
    rule: _RuleCore,
    column: ComputedColumn,
    var: ColumnGroup,
) -> None:
    if isinstance(column, StrippedSequence | ProformaSequence):
        _check_derived_sequence_column(rule, column)
        return
    if isinstance(column, ProformaIon):
        if rule.quantification_level not in {"ion", "fragment"}:
            raise ValueError("how='proforma_ion' is valid only for ion or fragment rules")
        source_types = {entry.name: entry.type for entry in var if entry.source is not None}
        if source_types.get(column.inputs[1]) != "integer":
            raise ValueError("how='proforma_ion' requires an integer charge source")
        if rule.quantification_level == "ion" and column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma ion must be an axis.var_keys member")
        return
    if isinstance(column, ProformaFragment):
        if rule.quantification_level != "fragment":
            raise ValueError("how='proforma_fragment' is valid only for fragment rules")
        if column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma fragment must be an axis.var_keys member")
