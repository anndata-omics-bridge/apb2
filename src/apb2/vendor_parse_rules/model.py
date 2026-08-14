"""The rules.json module: the composed effective rule, its blocks, and the document shell.

Reading order mirrors a composed rule top to bottom: ``LongRule | WideRule``, ``Axis``
(+ ``Duplicates``), column groups (+ the ``ComputedColumn`` union), ``ColumnRoles``, the
layer union, the fragments union, the modifications union — then ``Document``, the shell
whose fragments stay raw dicts until the base-times-level merge crosses the single typed
boundary, ``validate_rule``. Search parameters are a separate schema in
``vendor_params/model.py``; ``Parameters`` is imported only for the gates.

Design rules: every block whose entries behave differently is a discriminated union;
validators exist only where structure cannot express the constraint; no field aliases —
a key that would need one is renamed at the source (``inputs``, never ``from``). This
module never touches pandas or numpy: it is the declarative side of the parser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    model_validator,
)

from apb2.vendor_params.model import Parameters

TableShape = Literal["long", "wide"]
QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
AxisColumnType = Literal["string", "integer", "number", "boolean"]
DuplicateMode = Literal["error", "aggregate", "keep_first", "keep_all_as_raw_table"]
TokenPosition = Literal[
    "before_residue", "after_residue", "n_term", "c_term", "embedded", "unknown"
]
UnknownPolicy = Literal["preserve", "drop", "error"]

LEVELS: tuple[QuantificationLevel, ...] = get_args(QuantificationLevel)
"""Quantification levels in canonical order."""

PEPTIDE_LEVELS: frozenset[QuantificationLevel] = frozenset(LEVELS) - {"protein"}
"""Quantification levels whose features carry a peptide sequence."""

_SAMPLE_GROUP = "sample"

ConditionValue = str | int | float | bool | None
"""A JSON scalar compared for equality against one parsed parameter value."""


class ModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchParameterOverride(ModelBase):
    """Swap the X layer when the parsed search parameters satisfy a condition."""

    when_search_parameters: dict[str, ConditionValue] = Field(min_length=1)
    x_layer: str


# ------------------------------------------------------------------------------- the rule


class _RuleCore(ModelBase):
    """Fields and checks shared by both rule shapes; never instantiated directly."""

    schema_version: str
    file_version: str
    software_name: str
    software_version_pattern: str
    quantification_level: QuantificationLevel
    axis: Axis
    column_roles: ColumnRoles = Field(default_factory=lambda: ColumnRoles())
    layers: list[Layer] = Field(min_length=1)
    modifications: Modifications | None = None
    fragments: Fragments | None = None
    requires_search_parameters: dict[str, ConditionValue] = Field(default_factory=dict)
    search_parameter_overrides: list[SearchParameterOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _core_consistency(self) -> _RuleCore:
        conditions = [
            self.requires_search_parameters,
            *(override.when_search_parameters for override in self.search_parameter_overrides),
        ]
        for condition in conditions:
            unknown = sorted(set(condition) - set(Parameters.model_fields))
            if unknown:
                raise ValueError(f"unknown search-parameter condition field(s): {unknown}")
        names = {layer.name for layer in self.layers}
        if self.axis.x_layer not in names:
            raise ValueError(
                f"axis.x_layer={self.axis.x_layer!r} matches no layer; available: {sorted(names)}"
            )
        if self.fragments is not None and self.quantification_level != "fragment":
            raise ValueError("[fragments] is only valid for quantification_level='fragment'.")
        return self


class LongRule(_RuleCore):
    """One row per (observation, feature): every source is an exact column name."""

    shape: Literal["long"]
    columns: LongColumns

    @model_validator(mode="after")
    def _column_consistency(self) -> LongRule:
        _check_axis_keys(self.axis.obs_keys, self.columns.obs, "obs")
        _check_axis_keys(self.axis.var_keys, self.columns.var, "var")
        _check_column_roles(self.column_roles, self.columns.var)
        _check_computed_columns(self, self.columns.var)
        _check_derived_not_selected(self.modifications, (self.columns.obs, self.columns.var))
        return self


class WideRule(_RuleCore):
    """One row per feature; the observation axis comes from the ``(?P<sample>...)``
    named group every layer ``source`` regex must carry — there is no obs column group.
    """

    shape: Literal["wide"]
    columns: WideColumns

    @model_validator(mode="after")
    def _column_consistency(self) -> WideRule:
        for layer in self.layers:
            try:
                pattern = re.compile(layer.source)
            except re.error as exc:
                raise ValueError(
                    f"Layer {layer.name!r}: wide rule 'source' must be a valid regex: {exc}"
                ) from exc
            if _SAMPLE_GROUP not in pattern.groupindex:
                raise ValueError(
                    f"Layer {layer.name!r}: wide rule 'source' must contain a "
                    f"'(?P<{_SAMPLE_GROUP}>...)' named group; got {layer.source!r}."
                )
        _check_axis_keys(self.axis.var_keys, self.columns.var, "var")
        _check_column_roles(self.column_roles, self.columns.var)
        _check_computed_columns(self, self.columns.var)
        _check_derived_not_selected(self.modifications, (self.columns.var,))
        return self


type Rule = Annotated[LongRule | WideRule, Field(discriminator="shape")]

_RULE_ADAPTER: TypeAdapter[LongRule | WideRule] = TypeAdapter(Rule)


def validate_rule(payload: object) -> LongRule | WideRule:
    """Validate one composed-rule payload: the single typed boundary of the rules system."""
    return _RULE_ADAPTER.validate_python(payload)


def rule_json_schema() -> dict[str, object]:
    """Return the JSON Schema of the effective-rule union."""
    return _RULE_ADAPTER.json_schema()


# ----------------------------------------------------------------------------------- axis


class Axis(ModelBase):
    """Which declared columns index observations and features, and which layer is X."""

    obs_keys: list[str] = Field(min_length=1)
    var_keys: list[str] = Field(min_length=1)
    x_layer: str
    duplicates: Duplicates = Field(default_factory=lambda: Duplicates())


class Duplicates(ModelBase):
    """What to do when several rows land on one (observation, feature) cell."""

    mode: DuplicateMode = "error"


# -------------------------------------------------------------------------------- columns


class LongColumns(ModelBase):
    obs: ColumnGroup
    var: ColumnGroup


class WideColumns(ModelBase):
    var: ColumnGroup


class ColumnGroup(ModelBase):
    """Declared axis columns: ``select`` sources must be present and gate recognition;
    ``optional_select`` sources are captured when present and silently skipped when absent.
    """

    select: dict[str, str] = Field(default_factory=dict)
    optional_select: dict[str, str] = Field(default_factory=dict)
    types: dict[str, AxisColumnType] = Field(default_factory=dict)
    computed: list[ComputedColumn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_declarations(self) -> ColumnGroup:
        both = sorted(set(self.select) & set(self.optional_select))
        if both:
            raise ValueError(f"column name(s) declared in both select and optional_select: {both}")
        unknown = sorted(set(self.types) - (set(self.select) | set(self.optional_select)))
        if unknown:
            raise ValueError(f"types must name selected columns; unknown: {unknown}")
        return self


class Coalesce(ModelBase):
    """Take the first non-null input value in declaration order."""

    how: Literal["coalesce"]
    name: str
    inputs: list[str] = Field(min_length=2)


class JoinNonempty(ModelBase):
    """Join the non-empty input values with a separator."""

    how: Literal["join_nonempty"]
    name: str
    inputs: list[str] = Field(min_length=2)
    separator: str = Field(min_length=1)


class StrippedSequence(ModelBase):
    """Expose the modification-stripped peptide derived from one input column."""

    how: Literal["stripped_sequence"]
    name: Literal["ProForma_peptide"] = "ProForma_peptide"
    inputs: list[str] = Field(min_length=1, max_length=1)


class ProformaSequence(ModelBase):
    """Expose the ProForma peptidoform derived from one input column."""

    how: Literal["proforma_sequence"]
    name: Literal["ProForma_peptidoform"] = "ProForma_peptidoform"
    inputs: list[str] = Field(min_length=1, max_length=1)


class ProformaIon(ModelBase):
    """Combine a peptidoform and a charge column into a ProForma ion."""

    how: Literal["proforma_ion"]
    name: Literal["ProForma_ion"] = "ProForma_ion"
    inputs: list[str] = Field(min_length=2, max_length=2)


class ProformaFragment(ModelBase):
    """Combine a ProForma ion and a fragment label into a ProForma fragment."""

    how: Literal["proforma_fragment"]
    name: Literal["ProForma_fragment"] = "ProForma_fragment"
    inputs: list[str] = Field(min_length=2, max_length=2)


type ComputedColumn = Annotated[
    Coalesce | JoinNonempty | StrippedSequence | ProformaSequence | ProformaIon | ProformaFragment,
    Field(discriminator="how"),
]
"""One computed-column declaration; each ``how`` declares only the fields it uses."""


class ColumnRoles(ModelBase):
    """Semantic locations needed by downstream canonical-data consumers."""

    protein_assignment: str | None = Field(default=None, min_length=1)
    fasta_accessions: str | None = Field(default=None, min_length=1)


# --------------------------------------------------------------------------------- layers


class NumericLayer(ModelBase):
    """A quantitative layer whose cells become floats.

    ``source`` is an exact column name in a ``LongRule``, a ``(?P<sample>...)`` header
    regex in a ``WideRule``. Layers are optional unless ``required`` or named as
    ``axis.x_layer``; a regex ``value_pattern`` extracts the numeric part of structured
    cells (PEAKS ``AScore`` is ``site:modification:score``).
    """

    encoding_mode: Literal["numeric"] = "numeric"
    name: str
    source: str
    missing_values: list[float] = Field(default_factory=list)
    value_pattern: ValuePattern = Field(default_factory=lambda: NoValuePattern())
    required: bool = False


class FactorLayer(ModelBase):
    """A categorical layer whose cells are encoded through a declared category map."""

    encoding_mode: Literal["factor"]
    name: str
    source: str
    categories: dict[str, int] = Field(min_length=1)
    required: bool = False


def _layer_encoding(value: object) -> str:
    """Tag a layer payload; an absent ``encoding_mode`` is the numeric authoring default."""
    if isinstance(value, Mapping):
        return str(value.get("encoding_mode", "numeric"))
    return str(getattr(value, "encoding_mode", "numeric"))


type Layer = Annotated[
    Annotated[NumericLayer, Tag("numeric")] | Annotated[FactorLayer, Tag("factor")],
    Discriminator(_layer_encoding),
]


class NoValuePattern(ModelBase):
    """A numeric layer contains directly parseable scalar values."""

    mode: Literal["none"] = "none"


class RegexValuePattern(ModelBase):
    """Extract one numeric capture group from each structured layer value."""

    mode: Literal["regex"] = "regex"
    pattern: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pattern(self) -> RegexValuePattern:
        try:
            compiled = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"value_pattern is not a valid regex: {exc}") from exc
        if compiled.groups != 1:
            raise ValueError(
                f"value_pattern must have exactly one capture group, found {compiled.groups}."
            )
        return self


type ValuePattern = Annotated[
    NoValuePattern | RegexValuePattern,
    Field(discriminator="mode"),
]


# ------------------------------------------------------------------------------ fragments


class PositionalFragments(ModelBase):
    """No per-fragment label column (older DIA-NN): labels are synthesised positionally
    (``frag_0``, ``frag_1``, ...) by index within the precursor.
    """

    label_strategy: Literal["positional"]
    value_columns: list[str] = Field(min_length=1)
    delimiter: str = ";"
    label_output: str = "fragment_label"


class ColumnLabeledFragments(ModelBase):
    """Fragment identities packed in ``label_column`` (DIA-NN ``Fragment.Info``, tokens
    like ``b4-unknown^1/327.16``): ``label_output`` is the token before ``/``.
    """

    label_strategy: Literal["column"]
    value_columns: list[str] = Field(min_length=1)
    label_column: str
    delimiter: str = ";"
    label_output: str = "fragment_label"


type Fragments = Annotated[
    PositionalFragments | ColumnLabeledFragments,
    Field(discriminator="label_strategy"),
]
"""Packed parallel-list fragment columns to explode before conversion."""


# -------------------------------------------------------------------------- modifications


class ModificationMapEntry(ModelBase):
    """A vendor token plus the Unimod accession it means (resolved via
    ``unimod_registry.toml`` so all tools agree on what e.g. ``UNIMOD:35`` means).
    """

    token: str
    accession: str


class TokenRegexModifications(ModelBase):
    """Extract inline modification tokens with a regex and map them to Unimod
    (``"PEPM[15.9949]TIDE"``, ``"_(ac)PEPTIDEM(ox)_"``).
    """

    parser: Literal["token_regex"]
    source_column: str
    token_pattern: str
    token_position: TokenPosition = "after_residue"
    case_sensitive: bool = False
    unknown_policy: UnknownPolicy = "preserve"
    output_column: str = "proforma_sequence"
    map: list[ModificationMapEntry] = Field(min_length=1)


class SiteListModifications(ModelBase):
    """Parallel name/site columns beside a bare sequence (alphabase layout, AlphaDIA):
    names and sites are ``delimiter``-joined and paired index-wise; sites are
    ``site_base``-indexed and site ``0`` is the N-terminus regardless of ``site_base``.
    """

    parser: Literal["site_list"]
    sequence_column: str
    modification_column: str
    site_column: str
    delimiter: str = ";"
    site_base: int = Field(default=1, ge=0, le=1)
    case_sensitive: bool = False
    unknown_policy: UnknownPolicy = "preserve"
    output_column: str = "proforma_sequence"
    map: list[ModificationMapEntry] = Field(min_length=1)


type Modifications = Annotated[
    TokenRegexModifications | SiteListModifications,
    Field(discriminator="parser"),
]


# --------------------------------------------------- whole-rule checks shared by both shapes


def _group_names(group: ColumnGroup) -> list[str]:
    computed = (column.name for column in group.computed)
    return list(dict.fromkeys([*group.select, *group.optional_select, *computed]))


def _check_axis_keys(keys: list[str], group: ColumnGroup, axis_name: str) -> None:
    """Axis keys must be declared, and never best-effort: an index cannot be optional."""
    declared = set(_group_names(group))
    missing = [key for key in keys if key not in declared]
    if missing:
        raise ValueError(
            f"axis.{axis_name}_keys must be declared in columns.{axis_name}: {missing}"
        )
    optional = [key for key in keys if key in group.optional_select]
    if optional:
        raise ValueError(f"axis.{axis_name}_keys must not name optional_select columns: {optional}")


def _check_column_roles(roles: ColumnRoles, var: ColumnGroup) -> None:
    declared_columns = set(_group_names(var))
    for role, column in ((n, c) for n, c in roles if c is not None):
        if column not in declared_columns:
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
    } & {modifications.output_column, "stripped_sequence"}
    if selected:
        raise ValueError(
            "apb2-derived modification columns must be declared in "
            f"columns.var.computed, not select: {sorted(selected)}"
        )


def _check_computed_columns(rule: _RuleCore, var: ColumnGroup) -> None:
    """Check what needs the whole rule; each column's own shape is a field declaration."""
    available = set(var.select) | set(var.optional_select)
    if rule.fragments is not None:
        # Fragment expansion injects this source before computed columns materialize.
        available.add(rule.fragments.label_output)
    for column in var.computed:
        missing = [source for source in column.inputs if source not in available]
        if missing:
            raise ValueError(
                f"computed column {column.name!r} references undeclared var column(s): {missing}"
            )
        _check_computed_column(rule, column, var)
        available.add(column.name)


def _check_computed_column(rule: _RuleCore, column: ComputedColumn, var: ColumnGroup) -> None:
    if isinstance(column, StrippedSequence | ProformaSequence):
        if rule.modifications is None:
            raise ValueError(f"how={column.how!r} requires a [modifications] block.")
        return
    if isinstance(column, ProformaIon):
        # At fragment level ProForma_ion is an intermediate for ProForma_fragment.
        if rule.quantification_level not in {"ion", "fragment"}:
            raise ValueError("how='proforma_ion' is valid only for ion or fragment rules.")
        charge_column = column.inputs[1]
        if var.types.get(charge_column, "string") != "integer":
            raise ValueError(
                "how='proforma_ion' requires its charge source to declare type='integer'; "
                f"got {charge_column!r}"
            )
        if rule.quantification_level == "ion" and column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma ion columns must be used in axis.var_keys.")
        return
    if isinstance(column, ProformaFragment):
        if rule.quantification_level != "fragment":
            raise ValueError("how='proforma_fragment' is valid only for fragment rules.")
        if column.name not in rule.axis.var_keys:
            raise ValueError("computed ProForma fragment columns must be used in axis.var_keys.")


# ------------------------------------------------------------------------------ documents


type JsonDict = dict[str, object]

_FIELDWISE_BLOCKS = (
    "axis",
    "column_roles",
    "modifications",
    "fragments",
    "requires_search_parameters",
)
_CONCAT_BLOCKS = ("layers", "search_parameter_overrides")


class _InputBlock(ModelBase):
    shape: TableShape


class Document(ModelBase):
    """One parsed rules.json: a validated shell around raw dict fragments.

    The fragments stay raw dicts through the base-times-level merge — merging dicts needs no
    models, presence is key membership — and cross the single typed boundary,
    ``validate_rule``, only once composed. That boundary is also the only validator:
    unknown keys and wrong types ride through the merge and are reported there with paths.
    """

    path: Path
    schema_version: str
    file_version: str
    software_name: str
    software_version_pattern: str
    input: _InputBlock
    base: JsonDict
    levels: dict[QuantificationLevel, JsonDict] = Field(min_length=1)


def compose_rule(document: Document, level: QuantificationLevel) -> LongRule | WideRule:
    """Merge ``level`` over the document's base and validate the composed rule."""
    try:
        merged = _merge_fragments(document.base, document.levels[level])
    except TypeError as error:
        raise ValueError(
            f"{document.path}: level {level!r} fragments do not merge: {error}"
        ) from error
    return validate_rule(
        {
            "schema_version": document.schema_version,
            "file_version": document.file_version,
            "software_name": document.software_name,
            "software_version_pattern": document.software_version_pattern,
            "quantification_level": level,
            "shape": document.input.shape,
            **merged,
        }
    )


def load_document(path: Path) -> Document:
    """Parse one rules.json shell; every fragment is validated later, by ``rule``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: document must be a JSON object")
    return Document.model_validate({"path": path, **cast("JsonDict", raw)})


def _merge_fragments(base: JsonDict, level: JsonDict) -> JsonDict:
    """Merge one level fragment over a base fragment.

    ``_FIELDWISE_BLOCKS`` merge field-wise (a level field replaces the base field
    wholesale); inside ``columns.obs`` / ``columns.var`` the ``select`` /
    ``optional_select`` / ``types`` mappings merge key-wise and ``computed`` concatenates;
    ``_CONCAT_BLOCKS`` concatenate. Keys outside the known blocks ride through unchanged
    so that ``validate_rule``'s ``extra="forbid"`` rejects them with a field path.
    """
    merged: JsonDict = {**base, **level}
    for key in _FIELDWISE_BLOCKS:
        block = {**_mapping(base, key), **_mapping(level, key)}
        if block:
            merged[key] = block
    columns: JsonDict = {}
    for part in ("obs", "var"):
        group = _merge_group(
            _mapping(_mapping(base, "columns"), part),
            _mapping(_mapping(level, "columns"), part),
        )
        if group:
            columns[part] = group
    if columns:
        merged["columns"] = columns
    for key in _CONCAT_BLOCKS:
        entries = [*_sequence(base, key), *_sequence(level, key)]
        if entries:
            merged[key] = entries
    return merged


def _merge_group(base_group: JsonDict, level_group: JsonDict) -> JsonDict:
    """Merge one column group: mappings union key-wise, ``computed`` concatenates."""
    merged: JsonDict = {**base_group, **level_group}
    for key in ("select", "optional_select", "types"):
        mapping = {**_mapping(base_group, key), **_mapping(level_group, key)}
        if mapping:
            merged[key] = mapping
    computed = [*_sequence(base_group, "computed"), *_sequence(level_group, "computed")]
    if computed:
        merged["computed"] = computed
    return merged


def _mapping(fragment: JsonDict, key: str) -> JsonDict:
    return cast("JsonDict", fragment.get(key) or {})


def _sequence(fragment: JsonDict, key: str) -> list[object]:
    return cast("list[object]", fragment.get(key) or [])
