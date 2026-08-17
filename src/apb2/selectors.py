"""Every rules.json class selector, and the strategy it names.

rules.json keys are of three kinds. *Section selectors* pick which rule is in force (the
level key, ``software_version_pattern``, the parameter gates) and are resolved before
anything here runs. *Class selectors* are the discriminator literals — ``shape``,
``encoding_mode``, ``computed[].how``, ``modifications.parser``,
``fragments.label_strategy``, ``value_pattern.mode``, ``duplicates.mode``, plus the logical
``types`` of an axis column — and each one names a class in ``parse_quant``. Everything
else is *configuration*: constructor arguments for the class its selector named.

This module is the whole class-selector half: one function per selector, reading its
literal once and returning a constructed strategy. It is also the only place where a rule
declaration and a ``parse_quant`` class meet. The strategies take ordinary typed values and
import no schema package, so translating configuration into them happens here, beside the
composition root that orders them — never inside the strategies themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from apb2.modifications.apply_rules import MapEntry, ModificationRule, SiteListRule
from apb2.modifications.unimod_registry import resolve
from apb2.parse_quant.bound_input_reader import ReadPlan
from apb2.parse_quant.column_plan import (
    AXIS_COERCERS,
    AxisCoercer,
    AxisMaterialization,
    CoalesceColumn,
    ColumnComputer,
    ColumnMaterialization,
    DerivedSequenceColumn,
    JoinNonEmptyColumn,
    MaterializationPass,
    ProformaFragmentColumn,
    ProformaIonColumn,
)
from apb2.parse_quant.duplicates import (
    DuplicatePolicy,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
    SumDuplicates,
)
from apb2.parse_quant.errors import IncompatibleSourceError
from apb2.parse_quant.fragment_exploder import (
    ColumnLabeledExplode,
    FragmentExploder,
    NoFragments,
    PositionalExplode,
)
from apb2.parse_quant.layers import (
    FactorCoercion,
    LayerCoercion,
    LayerPlan,
    PlainNumericCoercion,
    RegexNumericCoercion,
)
from apb2.parse_quant.modifications import (
    ModificationApplier,
    NoModifications,
    SequenceColumns,
    SiteListApplier,
    TokenRegexApplier,
)
from apb2.parse_quant.table_conversion import Conversion, LongConversion, WideConversion
from apb2.vendor_parse_rules.model import (
    AxisColumnType,
    Coalesce,
    ColumnGroup,
    ColumnLabeledFragments,
    ComputedColumn,
    DuplicateMode,
    Duplicates,
    Fragments,
    JoinNonempty,
    Layer,
    LongRule,
    Modifications,
    ProformaIon,
    ProformaSequence,
    RegexValuePattern,
    SiteListModifications,
    StrippedSequence,
    WideRule,
    group_names,
    modification_outputs,
)
from apb2.vendor_parse_rules.runtime import (
    Recognition,
    declared_source_columns,
    layer_required,
    modification_sources,
    rule_label,
)

# ------------------------------------------------------------------------- columns.types


def coercer_for(logical_type: AxisColumnType) -> AxisCoercer:
    """Read a declared column's logical type, and return the coercion it names."""
    return AXIS_COERCERS[logical_type]


# ---------------------------------------------------------------------- columns.computed


def computer_for(column: ComputedColumn) -> ColumnComputer:
    """Read a computed-column declaration's ``how`` once; return the computer it names."""
    if isinstance(column, Coalesce):
        return CoalesceColumn(name=column.name, sources=tuple(column.inputs))
    if isinstance(column, JoinNonempty):
        return JoinNonEmptyColumn(
            name=column.name, sources=tuple(column.inputs), separator=column.separator
        )
    if isinstance(column, StrippedSequence | ProformaSequence):
        return DerivedSequenceColumn(name=column.name, source_key=column.how)
    if isinstance(column, ProformaIon):
        sequence, charge = column.inputs
        return ProformaIonColumn(name=column.name, sequence_key=sequence, charge_key=charge)
    ion, label = column.inputs
    return ProformaFragmentColumn(name=column.name, ion_key=ion, label_key=label)


def column_plan_for(
    rule: LongRule | WideRule, recognition: Recognition, applier: ModificationApplier
) -> ColumnMaterialization:
    """Build both materialization passes for every axis the rule declares."""
    keys_by_axis = {"obs": rule.axis.obs_keys, "var": rule.axis.var_keys}
    return ColumnMaterialization(
        groups={
            axis: _axis_materialization(group, keys_by_axis[axis])
            for axis, group in recognition.column_groups()
        },
        applier=applier,
    )


def _axis_materialization(group: ColumnGroup, keys: Sequence[str]) -> AxisMaterialization:
    """Split one group's declarations at the axis-key closure.

    The closure is prepared on the flat table because the pivot cannot group without it;
    everything else is materialized afterwards, on the deduplicated axis frame.
    """
    declared = tuple(group_names(group))
    closure = _key_closure(group, keys)
    return AxisMaterialization(
        declared=declared,
        keys=_materialization_pass(group, closure),
        rest=_materialization_pass(group, {name for name in declared if name not in closure}),
    )


def _key_closure(group: ColumnGroup, keys: Sequence[str]) -> set[str]:
    """Every declared column the axis keys are computed from, keys included."""
    computed_inputs = {column.name: set(column.inputs) for column in group.computed}
    closure = set(keys)
    changed = True
    while changed:
        changed = False
        for name, inputs in computed_inputs.items():
            if name in closure and not inputs.issubset(closure):
                closure |= inputs
                changed = True
    return closure


def _materialization_pass(group: ColumnGroup, names: AbstractSet[str]) -> MaterializationPass:
    """One pass over the declarations whose output name is in ``names``."""
    select = tuple((name, source) for name, source in group.select.items() if name in names)
    optional = tuple(
        (name, source) for name, source in group.optional_select.items() if name in names
    )
    return MaterializationPass(
        select=select,
        optional=optional,
        coercers={
            name: coercer_for(group.types.get(name, "string"))
            for name, _source in (*select, *optional)
        },
        computers=tuple(computer_for(column) for column in group.computed if column.name in names),
    )


# ---------------------------------------------------------------------------- layers[]


def coercion_for(layer: Layer) -> LayerCoercion:
    """Read a layer declaration's encoding flag once, and return the coercion it names."""
    if layer.encoding_mode == "factor":
        return FactorCoercion(layer.categories)
    missing_values = tuple(layer.missing_values)
    if isinstance(layer.value_pattern, RegexValuePattern):
        return RegexNumericCoercion(missing_values, layer.value_pattern.pattern)
    return PlainNumericCoercion(missing_values)


def _layer_plans(rule: LongRule | WideRule) -> tuple[LayerPlan, ...]:
    """One plan per declared layer, with the rule-level ``required`` question folded in."""
    return tuple(
        LayerPlan(
            name=layer.name,
            source=layer.source,
            required=layer_required(rule, layer),
            coercion=coercion_for(layer),
        )
        for layer in rule.layers
    )


# -------------------------------------------------------------------- axis.duplicates


POLICY_BY_MODE: Mapping[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": SumDuplicates(),
    # "keep_all_as_raw_table" is absent on purpose: the schema accepts the mode and no
    # policy implements it, so policy_for raises once, naming it, before any conversion
    # work starts.
}


def policy_for(duplicates: Duplicates) -> DuplicatePolicy:
    """Select the policy a rule's duplicate mode names.

    Raises NotImplementedError for a mode the schema permits but no policy implements.
    """
    policy = POLICY_BY_MODE.get(duplicates.mode)
    if policy is None:
        raise NotImplementedError(f"duplicates.mode={duplicates.mode!r} is not yet supported")
    return policy


# ------------------------------------------------------------------------- fragments


def exploder_for(rule: LongRule | WideRule, header: list[str]) -> FragmentExploder:
    """Read the rule's ``label_strategy`` once, and return the exploder it names.

    Packed columns resolve against the header here: a missing column backing a required
    layer fails construction; a missing optional one is dropped so the conversion skips
    its layer, exactly as it does on the non-fragment path.
    """
    fragments: Fragments | None = rule.fragments
    if fragments is None:
        return NoFragments()
    header_set = set(header)
    required_sources = {layer.source for layer in rule.layers if layer_required(rule, layer)}
    missing_required = [
        column
        for column in fragments.value_columns
        if column not in header_set and column in required_sources
    ]
    if missing_required:
        raise IncompatibleSourceError(
            f"input lacks the packed fragment column(s) {missing_required} required by "
            f"{rule_label(rule)}"
        )
    value_columns = tuple(column for column in fragments.value_columns if column in header_set)
    if not value_columns:
        raise IncompatibleSourceError(
            f"input carries none of the packed fragment columns {list(fragments.value_columns)} "
            f"declared by {rule_label(rule)}"
        )
    if isinstance(fragments, ColumnLabeledFragments):
        return ColumnLabeledExplode(
            label_column=fragments.label_column,
            label_output=fragments.label_output,
            delimiter=fragments.delimiter,
            value_columns=value_columns,
        )
    return PositionalExplode(
        label_output=fragments.label_output,
        delimiter=fragments.delimiter,
        value_columns=value_columns,
    )


# --------------------------------------------------------------------- modifications


def applier_for(rule: LongRule | WideRule) -> ModificationApplier:
    """Read the ``parser`` selector once; return the applier it names, or the identity.

    The absence questions — is a block declared, does anything read its output — are
    asked once here instead of on every table. Resolving the Unimod map is part of the
    same step, so an unknown accession fails before any table is read.
    """
    modifications = rule.modifications
    if modifications is None:
        return NoModifications()
    consumed = any(
        isinstance(column, ProformaSequence | StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return NoModifications()
    columns = SequenceColumns(
        output_column=modifications.output_column,
        sources=modification_sources(modifications),
        outputs=modification_outputs(modifications),
    )
    if isinstance(modifications, SiteListModifications):
        return SiteListApplier(
            columns,
            SiteListRule(
                delimiter=modifications.delimiter,
                site_base=modifications.site_base,
                case_sensitive=modifications.case_sensitive,
                unknown_policy=modifications.unknown_policy,
                entries=_map_entries(modifications),
            ),
        )
    return TokenRegexApplier(
        columns,
        ModificationRule(
            token_pattern=modifications.token_pattern,
            token_position=modifications.token_position,
            case_sensitive=modifications.case_sensitive,
            unknown_policy=modifications.unknown_policy,
            entries=_map_entries(modifications),
        ),
    )


def _map_entries(modifications: Modifications) -> tuple[MapEntry, ...]:
    """Fill ``name``, ``target``, ``position``, ``mass_delta`` from the bundled registry.

    Raises ``KeyError`` if an entry references an unknown accession.
    """
    entries: list[MapEntry] = []
    for entry in modifications.map:
        record = resolve(entry.accession)
        entries.append(
            MapEntry(
                token=entry.token,
                name=record.name,
                accession=record.accession,
                target=tuple(record.target),
                position=record.position,
                mass_delta=record.mass_delta,
            )
        )
    return tuple(entries)


# --------------------------------------------------------------------------- shape


def conversion_for(rule: LongRule | WideRule, *, strict: bool) -> Conversion:
    """Read a rule's shape once, and return the conversion it names."""
    layers = _layer_plans(rule)
    var_carry = _carry_columns(rule.axis.var_keys, rule.columns.var, _var_extras(rule))
    if isinstance(rule, LongRule):
        return LongConversion(
            obs_keys=rule.axis.obs_keys,
            var_keys=rule.axis.var_keys,
            obs_carry=_carry_columns(rule.axis.obs_keys, rule.columns.obs, ()),
            var_carry=var_carry,
            layers=layers,
            x_layer=rule.axis.x_layer,
            duplicates=policy_for(rule.axis.duplicates),
            strict=strict,
        )
    return WideConversion(
        obs_outputs=rule.axis.obs_keys,
        var_keys=rule.axis.var_keys,
        var_carry=var_carry,
        layers=layers,
        x_layer=rule.axis.x_layer,
        duplicates=policy_for(rule.axis.duplicates),
        software_name=rule.software_name,
        strict=strict,
    )


def _carry_columns(
    keys: Sequence[str], group: ColumnGroup, extras: tuple[str, ...]
) -> tuple[str, ...]:
    """Everything an axis frame must take off the flat table, in stable order.

    Declared names cover the already-prepared key-closure columns; raw sources cover
    everything the post-pivot pass materializes afterwards. Absent names (skipped
    optionals, columns not yet materialized) drop out where the frame is projected.
    """
    return tuple(
        dict.fromkeys(
            [
                *keys,
                *group_names(group),
                *group.select.values(),
                *group.optional_select.values(),
                *extras,
            ]
        )
    )


def _var_extras(rule: LongRule | WideRule) -> tuple[str, ...]:
    """Raw modification sources/outputs and the fragment label the var frame may need."""
    extras: list[str] = []
    if rule.modifications is not None:
        extras.extend(modification_sources(rule.modifications))
        extras.extend(sorted(modification_outputs(rule.modifications)))
    if rule.fragments is not None:
        extras.append(rule.fragments.label_output)
    return tuple(extras)


# ----------------------------------------------------------------------- the read plan


def compile_read_plan(
    recognition: Recognition,
    header: Sequence[str],
    modification_columns: Iterable[str],
    packed_columns: Iterable[str],
) -> ReadPlan:
    """Compile the exact projection one rule needs from one inspected header.

    Required sources are validated separately (``recognition.matches`` during
    construction); the projection is the intersection of the header with everything the
    rule can read: selected and optional sources, computed-column inputs, layer sources
    (exact for long rules, regex-expanded for wide ones), modification sources, and
    packed fragment columns.
    """
    needed = set(modification_columns) | set(packed_columns) | declared_source_columns(recognition)
    needed.update(recognition.layer_source_columns(list(header)))
    columns = tuple(name for name in header if name in needed)
    strings = _string_sources(recognition) & set(columns)
    return ReadPlan(columns=columns, string_sources=frozenset(strings))


def _string_sources(recognition: Recognition) -> frozenset[str]:
    """Real vendor sources whose exact textual tokens must survive reading."""
    source_types: dict[str, str] = {}
    for _axis, group in recognition.column_groups():
        selected = {**group.select, **group.optional_select}
        for output_name, source_name in selected.items():
            logical_type = group.types.get(output_name, "string")
            if source_name in source_types and source_types[source_name] != logical_type:
                raise ValueError(
                    "conflicting logical types for vendor source "
                    f"{source_name!r}: {source_types[source_name]!r} and "
                    f"{logical_type!r}"
                )
            source_types[source_name] = logical_type
    return frozenset(
        source for source, logical_type in source_types.items() if logical_type == "string"
    )
