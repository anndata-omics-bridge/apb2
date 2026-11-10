"""Complete effective-rule declarations and essential cross-block invariants."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from apb2.parserV2.vendor_parse_rules.schema.annotation import SampleAnnotation
from apb2.parserV2.vendor_parse_rules.schema.axis import (
    FLAG_COLUMNS,
    Axis,
    Coalesce,
    ColumnEntry,
    ColumnGroup,
    ComputedColumn,
    JoinNonempty,
    LongColumns,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    RankRows,
    SourcedColumn,
    SourcedVarColumn,
    StrippedSequence,
    VarEntry,
    WideColumns,
    computed_columns,
)
from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    PerFeature,
    QuantificationLevel,
    SchemaVersion,
)
from apb2.parserV2.vendor_parse_rules.schema.base_modifications import (
    EmbeddedSiteListSyntax,
    ModificationMap,
    PlainSequenceSyntax,
    SequenceSyntax,
    SiteListSyntax,
    TokenRegexSyntax,
)
from apb2.parserV2.vendor_parse_rules.schema.fragments import Fragments
from apb2.parserV2.vendor_parse_rules.schema.hierarchy import HIERARCHIES
from apb2.parserV2.vendor_parse_rules.schema.measurements import (
    FactorLayer,
    Layer,
    Measurements,
    NumericLayer,
)
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
    hierarchy: str
    # Header columns that mean a table reports another level: the level skips such a table.
    forbidden_columns: list[str] = Field(default_factory=list)
    axis: Axis
    measurements: Measurements
    sequence_syntax: dict[str, SequenceSyntax] = Field(default_factory=dict)
    # Each vendor modification token and its Unimod accession; one per level, absent means none.
    modification_map: ModificationMap | None = None
    requires_search_parameters: dict[SearchParameterField, ConditionValue] = Field(
        default_factory=dict
    )
    search_parameter_overrides: list[SearchParameterOverride] = Field(default_factory=list)
    sample_annotation: SampleAnnotation | None = None
    rank_rows: RankRows | None = None

    @model_validator(mode="after")
    def _core_consistency(self) -> _RuleCore:
        if self.hierarchy not in HIERARCHIES:
            raise ValueError(f"unknown hierarchy {self.hierarchy!r}")
        if self.quantification_level not in dict(HIERARCHIES[self.hierarchy]):
            raise ValueError(
                f"level {self.quantification_level!r} is absent from hierarchy {self.hierarchy!r}"
            )
        _check_names("measurement layer names", self.measurements.layers)
        names = [layer.name for layer in self.measurements.layers]
        if self.measurements.primary_layer not in set(names):
            raise ValueError(
                f"measurements.primary_layer={self.measurements.primary_layer!r} matches no "
                f"layer; available: {sorted(names)}"
            )
        sample_layer = self.measurements.sample_layer
        if sample_layer is not None and sample_layer not in set(names):
            raise ValueError(
                f"measurements.sample_layer={sample_layer!r} matches no layer; "
                f"available: {sorted(names)}"
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
        if self.fragments is not None and self.quantification_level != "peptidoform_fragment":
            raise ValueError(
                "fragments are valid only for quantification_level='peptidoform_fragment'"
            )
        if self.measurements.sample_layer is not None:
            raise ValueError(
                "measurements.sample_layer is valid only for wide rules; long observations "
                "come from columns.obs"
            )
        _check_column_group(self.axis.obs_keys, self.columns.obs, "obs")
        _check_column_group(self.axis.var_keys, self.columns.var, "var")
        _check_per_feature(self.axis.var_keys, self.columns.var, self.rank_rows)
        _check_computed_columns(self, self.columns.var)
        self._check_obs_computed_columns(self.columns.obs)
        _check_one_type_per_source((self.columns.obs, self.columns.var))
        _check_forbidden_columns(self.forbidden_columns, self._named_sources())
        return self

    def _named_sources(self) -> tuple[str, ...]:
        """Every source column the level names exactly: its entries', layers' and fragments'."""
        entries = (*self.columns.obs, *self.columns.var)
        return (
            *(entry.source for entry in entries if entry.source is not None),
            *(layer.source for layer in self.measurements.layers),
            *(self.fragments.read_columns if self.fragments is not None else ()),
        )

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
            if layer.version is not None:
                raise ValueError(f"Layer {layer.name!r}: a wide layer carries no version")
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
        _check_per_feature(self.axis.var_keys, self.columns.var, self.rank_rows)
        _check_computed_columns(self, self.columns.var)
        _check_one_type_per_source((self.columns.var,))
        # Layer sources are patterns claiming a family of columns; a level may forbid one.
        _check_forbidden_columns(
            self.forbidden_columns,
            (entry.source for entry in self.columns.var if entry.source is not None),
        )
        return self


type Rule = Annotated[LongRule | WideRule, Field(discriminator="shape")]

_RULE_ADAPTER: TypeAdapter[LongRule | WideRule] = TypeAdapter(Rule)


def validate_rule(payload: object) -> LongRule | WideRule:
    """Validate one composed effective-rule payload."""
    return _RULE_ADAPTER.validate_python(payload)


def rule_json_schema() -> dict[str, object]:
    """Return the JSON Schema for complete schema-0.10 effective rules."""
    return _RULE_ADAPTER.json_schema()


def _check_forbidden_columns(forbidden: Sequence[str], sources: Iterable[str]) -> None:
    """A level cannot forbid a source it names: it would match no table or never read it."""
    read = sorted(set(forbidden) & set(sources))
    if read:
        raise ValueError(f"forbidden_columns {read} are sources the level reads")


def _check_names(kind: str, entries: Sequence[ColumnEntry] | Sequence[Layer]) -> None:
    """Names are unique ignoring case, as DuckDB's column names are.

    The exception is a column the vendor spells differently across versions: entries of one
    exact name, each with a ``version``, which a file picks by the source its header carries.
    They agree on all but ``source``, ``version`` and ``note``, and are optional, as no file of
    other versions carries their source.
    """
    by_name: dict[str, list[ColumnEntry | Layer]] = {}
    for entry in entries:
        if _version(entry) is not None and entry.required:
            raise ValueError(f"{entry.name!r}: an entry with a version is not required")
        by_name.setdefault(entry.name.casefold(), []).append(entry)
    for clash in (group for group in by_name.values() if len(group) > 1):
        names = sorted({entry.name for entry in clash})
        if len(names) > 1 or any(_version(entry) is None for entry in clash):
            raise ValueError(
                f"{kind} must be unique ignoring case, unless every entry of one name "
                f"carries a version: {names}"
            )
        shared = {
            json.dumps(entry.model_dump(exclude={"source", "version", "note"}), sort_keys=True)
            for entry in clash
        }
        if len(shared) > 1:
            raise ValueError(
                f"entries named {names[0]!r} differ in more than source, version and note"
            )


def _version(entry: ColumnEntry | Layer) -> str | None:
    """The versions an entry's source is written by; ``None`` for all, and on computed ones."""
    return entry.version if isinstance(entry, SourcedColumn | NumericLayer | FactorLayer) else None


def _check_column_group(keys: list[str], group: ColumnGroup, owner: RoleOwner) -> None:
    _check_names("column entry names", group)
    names = [entry.name for entry in group]
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
    role_columns: list[tuple[SemanticRole, str]] = list(
        dict.fromkeys((role, entry.name) for entry in group for role in entry.roles)
    )
    if owner == "var" and any(entry.roles and entry.type != "string" for entry in group):
        raise ValueError("every var role requires a string-typed column")
    _check_role_owners(owner, (role for role, _name in role_columns))
    if len(dict(role_columns)) != len(role_columns):
        raise ValueError(f"roles must be unique on columns.{owner}")


def _check_per_feature(
    keys: list[str], var: Sequence[VarEntry], rank_rows: RankRows | None
) -> None:
    """Only payload columns resolve repeated rows, each vendor column one way, and
    ``best_row`` columns exactly when the level ranks its rows."""
    sourced = [entry for entry in var if isinstance(entry, SourcedVarColumn)]
    declared = [entry.name for entry in sourced if entry.per_feature != "first"]
    keyed = [name for name in declared if name in keys]
    if keyed:
        raise ValueError(f"axis.var_keys identify a feature and resolve no repeated rows: {keyed}")
    resolutions: dict[str, PerFeature] = {}
    for entry in sourced:
        if resolutions.setdefault(entry.source, entry.per_feature) != entry.per_feature:
            raise ValueError(
                f"vendor source {entry.source!r} resolves repeated rows both as "
                f"{resolutions[entry.source]!r} and as {entry.per_feature!r}"
            )
    best_rows = [entry.name for entry in sourced if entry.per_feature == "best_row"]
    if best_rows and rank_rows is None:
        raise ValueError(f"per_feature 'best_row' needs the level's rank_rows: {best_rows}")
    if rank_rows is None:
        return
    if not best_rows:
        raise ValueError("rank_rows ranks rows for best_row columns, and no column is best_row")
    ranking = {entry.name: entry for entry in sourced}.get(rank_rows.by)
    if ranking is None or rank_rows.by in keys or ranking.per_feature == "best_row":
        raise ValueError(
            f"rank_rows ranks by {rank_rows.by!r}, which must be a sourced var column, "
            "neither a key nor a best_row column"
        )
    if ranking.type not in {"number", "integer"} or not ranking.required:
        raise ValueError(
            f"rank_rows ranks by {rank_rows.by!r}, which must be a required numeric column"
        )


def _check_role_owners(owner: RoleOwner, roles: Iterable[SemanticRole]) -> None:
    invalid = sorted(set(roles) - ROLE_CONFIG[owner])
    if invalid:
        raise ValueError(f"roles are not allowed on {owner}: {invalid}")


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
    names = [column.name for column in var]
    for flag in FLAG_COLUMNS:
        if names.count(flag) != 1:
            raise ValueError(
                f"var declares {flag!r} {names.count(flag)} times; every rule declares it once, "
                "with how 'unmarked' when the report marks no such rows"
            )


def _check_sequence_column(
    rule: _RuleCore,
    column: StrippedSequence | ProformaSequence,
) -> None:
    syntax = rule.sequence_syntax.get(column.syntax)
    if syntax is None:
        raise ValueError(f"column {column.name!r} references missing syntax {column.syntax!r}")
    if isinstance(column, StrippedSequence):
        if not isinstance(syntax, PlainSequenceSyntax | TokenRegexSyntax):
            raise ValueError("stripped_sequence requires token_regex or plain_sequence syntax")
        return
    if isinstance(syntax, PlainSequenceSyntax):
        raise ValueError("proforma_sequence does not accept plain_sequence syntax")
    if column.name.endswith("_unlocalized") and not isinstance(syntax, EmbeddedSiteListSyntax):
        raise ValueError(f"{column.name} requires embedded_site_list syntax")
    arity = (
        1
        if isinstance(syntax, TokenRegexSyntax)
        else 3
        if isinstance(syntax, SiteListSyntax)
        else 2
    )
    if len(column.inputs) != arity:
        raise ValueError(
            f"{syntax.parser} normalization requires {arity} inputs; got {column.inputs!r}"
        )
    if rule.modification_map is None:
        raise ValueError(f"column {column.name!r} needs its level's modification_map")


def _check_computed_column(
    rule: _RuleCore,
    column: ComputedColumn,
    var: ColumnGroup,
) -> None:
    if isinstance(column, StrippedSequence | ProformaSequence):
        _check_sequence_column(rule, column)
        return
    if isinstance(column, ProformaIon):
        if rule.quantification_level not in {"peptidoform_ion", "peptidoform_fragment"}:
            raise ValueError("how='proforma_ion' is valid only for ion or fragment rules")
        source_types = {entry.name: entry.type for entry in var if entry.source is not None}
        if source_types.get(column.inputs[1]) != "integer":
            raise ValueError("how='proforma_ion' requires an integer charge source")
        if rule.quantification_level == "peptidoform_ion" and column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma ion must be an axis.var_keys member")
        return
    if isinstance(column, ProformaFragment):
        if rule.quantification_level != "peptidoform_fragment":
            raise ValueError("how='proforma_fragment' is valid only for fragment rules")
        if column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma fragment must be an axis.var_keys member")
