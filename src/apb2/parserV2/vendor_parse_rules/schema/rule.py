"""Complete effective-rule declarations and essential cross-block invariants."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from apb2.parserV2.vendor_parse_rules.schema.axis import (
    Axis,
    Coalesce,
    ColumnGroup,
    ColumnRoles,
    ComputedColumn,
    JoinNonempty,
    LongColumns,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
    WideColumns,
    group_names,
)
from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    QuantificationLevel,
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

_SAMPLE_GROUP = "sample"


class _RuleCore(ModelBase):
    """Fields shared by long and wide effective rules."""

    schema_version: str
    file_version: str
    software_name: str
    software_version_pattern: str
    quantification_level: QuantificationLevel
    axis: Axis
    measurements: Measurements
    column_roles: ColumnRoles = Field(default_factory=ColumnRoles)
    modifications: Modifications | None = None
    fragments: Fragments | None = None
    requires_search_parameters: dict[SearchParameterField, ConditionValue] = Field(
        default_factory=dict
    )
    search_parameter_overrides: list[SearchParameterOverride] = Field(default_factory=list)

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
        if self.fragments is not None and self.quantification_level != "fragment":
            raise ValueError("fragments are valid only for quantification_level='fragment'")
        return self


class LongRule(_RuleCore):
    """One row per observation-variable pair with exact physical source names."""

    shape: Literal["long"]
    columns: LongColumns

    @model_validator(mode="after")
    def _column_consistency(self) -> LongRule:
        _check_axis_keys(self.axis.obs_keys, self.columns.obs, "obs")
        _check_axis_keys(self.axis.var_keys, self.columns.var, "var")
        _check_column_roles(self.column_roles, self.columns.var)
        _check_computed_columns(self, self.columns.var)
        self._check_obs_computed_columns(self.columns.obs)
        _check_derived_not_selected(self.modifications, (self.columns.obs, self.columns.var))
        _check_one_type_per_source((self.columns.obs, self.columns.var))
        return self

    @staticmethod
    def _check_obs_computed_columns(obs: ColumnGroup) -> None:
        sequence_derived = [
            column.name
            for column in obs.computed
            if not isinstance(column, Coalesce | JoinNonempty)
        ]
        if sequence_derived:
            raise ValueError(f"sequence-derived computed columns are var-only: {sequence_derived}")
        available = set(obs.select) | set(obs.optional_select)
        for column in obs.computed:
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

    @model_validator(mode="after")
    def _no_fragments_on_wide(self) -> WideRule:
        if self.fragments is not None:
            raise ValueError("fragments require a long rule")
        return self

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
        _check_axis_keys(self.axis.var_keys, self.columns.var, "var")
        _check_column_roles(self.column_roles, self.columns.var)
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
    """Return the JSON Schema for complete schema-0.3 effective rules."""
    return _RULE_ADAPTER.json_schema()


def _check_axis_keys(keys: list[str], group: ColumnGroup, axis_name: str) -> None:
    declared = set(group_names(group))
    missing = [key for key in keys if key not in declared]
    if missing:
        raise ValueError(
            f"axis.{axis_name}_keys must be declared in columns.{axis_name}: {missing}"
        )
    optional = [key for key in keys if key in group.optional_select]
    if optional:
        raise ValueError(f"axis.{axis_name}_keys must not be optional: {optional}")


def _check_column_roles(roles: ColumnRoles, var: ColumnGroup) -> None:
    declared = set(group_names(var))
    values = (
        ("protein_assignment", roles.protein_assignment),
        ("fasta_accessions", roles.fasta_accessions),
    )
    for role, column in values:
        if column is not None and column not in declared:
            raise ValueError(f"column_roles.{role} must name a declared var column; got {column!r}")


def _check_derived_not_selected(
    modifications: Modifications | None,
    groups: tuple[ColumnGroup, ...],
) -> None:
    if modifications is None:
        return
    selected = {
        source
        for group in groups
        for source in (*group.select.values(), *group.optional_select.values())
    } & modification_outputs(modifications)
    if selected:
        raise ValueError(
            f"derived modification columns belong in computed, not select: {sorted(selected)}"
        )


def _check_one_type_per_source(groups: tuple[ColumnGroup, ...]) -> None:
    declared: dict[str, AxisColumnType] = {}
    for group in groups:
        for name, source in (group.select | group.optional_select).items():
            logical_type = group.types.get(name, "string")
            if declared.setdefault(source, logical_type) != logical_type:
                raise ValueError(
                    f"vendor source {source!r} has conflicting logical types: "
                    f"{declared[source]!r} and {logical_type!r}"
                )


def _check_computed_columns(rule: _RuleCore, var: ColumnGroup) -> None:
    available = set(var.select) | set(var.optional_select)
    if rule.fragments is not None:
        available.add(rule.fragments.label_output)
    for column in var.computed:
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
        charge_column = column.inputs[1]
        if var.types.get(charge_column, "string") != "integer":
            raise ValueError("how='proforma_ion' requires an integer charge source")
        if rule.quantification_level == "ion" and column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma ion must be an axis.var_keys member")
        return
    if isinstance(column, ProformaFragment):
        if rule.quantification_level != "fragment":
            raise ValueError("how='proforma_fragment' is valid only for fragment rules")
        if column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma fragment must be an axis.var_keys member")
