"""Turn one rule and one input into one running ``Parser``.

This is the only module that depends on all three sides — the strategies in
``parse_quant``, the rule in ``vendor_parse_rules``, and the search-parameter evidence in
``vendor_params`` — and therefore the only place a configuration value is read.

``make_parse_strategy`` is the composition root; everything below it is one function per
rules.json **class selector** (the discriminator literals of TODO item 5: ``shape``,
``encoding_mode``, ``computed[].how``, ``modifications.parser``,
``fragments.label_strategy``, ``value_pattern.mode``, ``duplicates.mode``, and a declared
column's logical ``types``). Each reads its literal once and returns the strategy it names,
constructed from plain values — so no strategy imports a schema package, and the map from a
rules.json key to the class implementing it is this file, top to bottom.

Imports are module-qualified on purpose: at every call site below you can see which side of
the boundary a name comes from — ``model.Coalesce`` in, ``column_plan.CoalesceColumn`` out.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from apb2.modifications import apply_rules
from apb2.parse_quant import (
    bound_input_reader,
    column_plan,
    duplicates,
    fragment_exploder,
    layers,
    modifications,
    parse_strategy,
    table_conversion,
)
from apb2.parse_quant.errors import IncompatibleSourceError, NoCompatibleLevelError
from apb2.parse_quant.sources import InputSource
from apb2.serialization import JsonValue
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules import model, runtime

type Rule = model.LongRule | model.WideRule


# ------------------------------------------------------------------- the composition root


def make_parse_strategy(
    rule: Rule,
    source: InputSource,
    parameters: Parameters | None = None,
    *,
    strict: bool = False,
) -> parse_strategy.Parser:
    """Construct one fully injected parser for one quantification level.

    Raises ``IncompatibleSourceError`` when ``source`` cannot satisfy ``rule`` or when the
    rule's parameter gate is not satisfied by ``parameters``. ``strict`` promotes
    non-``X`` layer-contract warnings to errors.
    """
    if not runtime.available_for(rule, parameters):
        raise IncompatibleSourceError(
            f"{runtime.rule_label(rule)} is not available for the supplied search parameters "
            f"(requires {rule.requires_search_parameters})"
        )
    rule = runtime.resolved_for(rule, parameters)
    label = runtime.rule_label(rule)
    recognition = runtime.recognition_for(rule)
    binding = bound_input_reader.bind_source(source, accepts=recognition.matches, rule_label=label)
    header = binding.header()
    if not recognition.matches(header):
        raise IncompatibleSourceError(
            f"{binding.path} does not carry the columns required by {label}"
        )
    applier = applier_for(rule)
    missing = [column for column in applier.sources if column not in set(header)]
    if missing:
        raise IncompatibleSourceError(
            f"{binding.path} lacks the [modifications] source column(s) {missing} required "
            f"by {label}"
        )
    exploder = exploder_for(rule, header)
    return parse_strategy.Parser(
        level=rule.quantification_level,
        input=binding.make_reader(_read_plan(recognition, header, applier, exploder)),
        fragments=exploder,
        columns=column_plan_for(rule, recognition, applier),
        conversion=conversion_for(rule, strict=strict),
        provenance=_provenance(rule),
    )


def make_parse_strategies(
    rules: Iterable[Rule],
    source: InputSource,
    parameters: Parameters | None = None,
    *,
    strict: bool = False,
) -> list[parse_strategy.Parser]:
    """Construct one parser per rule the source satisfies, in ``LEVELS`` order.

    Incompatible rules never gate construction of the compatible ones; an empty result
    is an error because a source that satisfies nothing was the wrong source.
    """
    ordered = sorted(rules, key=lambda rule: model.LEVELS.index(rule.quantification_level))
    parsers: list[parse_strategy.Parser] = []
    for rule in ordered:
        try:
            parsers.append(make_parse_strategy(rule, source, parameters, strict=strict))
        except IncompatibleSourceError:
            continue
    if not parsers:
        levels = [rule.quantification_level for rule in ordered]
        raise NoCompatibleLevelError(
            f"no rule among levels {levels} is satisfied by the bound source {source!r}"
        )
    return parsers


def _read_plan(
    recognition: runtime.Recognition,
    header: Sequence[str],
    applier: modifications.ModificationApplier,
    exploder: parse_strategy.FragmentExploder,
) -> bound_input_reader.ReadPlan:
    """Fix the projection: everything the rule reads, plus what the two resolved
    strategies resolved against the same header.
    """
    columns = runtime.projected_columns(
        recognition, header, [*applier.source_columns(), *exploder.packed_columns()]
    )
    return bound_input_reader.ReadPlan(
        columns=columns,
        string_sources=runtime.string_typed_sources(recognition) & frozenset(columns),
    )


def _provenance(rule: Rule) -> dict[str, JsonValue]:
    """Serialize the rule's provenance once; the parser retains no model."""
    return {
        "rule_json": json.dumps(rule.model_dump(mode="json")),
        "schema_version": rule.schema_version,
        "software_name": rule.software_name,
        "shape": rule.shape,
        "quantification_level": rule.quantification_level,
    }


# ------------------------------------------------------------------------------ columns


def coercer_for(logical_type: model.AxisColumnType) -> column_plan.AxisCoercer:
    """Read a declared column's logical type, and return the coercion it names."""
    return column_plan.AXIS_COERCERS[logical_type]


def computer_for(column: model.ComputedColumn) -> column_plan.ColumnComputer:
    """Read a computed-column declaration's ``how`` once; return the computer it names."""
    if isinstance(column, model.Coalesce):
        return column_plan.CoalesceColumn(name=column.name, sources=tuple(column.inputs))
    if isinstance(column, model.JoinNonempty):
        return column_plan.JoinNonEmptyColumn(
            name=column.name, sources=tuple(column.inputs), separator=column.separator
        )
    if isinstance(column, model.StrippedSequence | model.ProformaSequence):
        return column_plan.DerivedSequenceColumn(name=column.name, source_key=column.how)
    if isinstance(column, model.ProformaIon):
        sequence, charge = column.inputs
        return column_plan.ProformaIonColumn(
            name=column.name, sequence_key=sequence, charge_key=charge
        )
    ion, label = column.inputs
    return column_plan.ProformaFragmentColumn(name=column.name, ion_key=ion, label_key=label)


def column_plan_for(
    rule: Rule, recognition: runtime.Recognition, applier: modifications.ModificationApplier
) -> column_plan.ColumnMaterialization:
    """Build both materialization passes for every axis the rule declares."""
    keys_by_axis = {"obs": rule.axis.obs_keys, "var": rule.axis.var_keys}
    return column_plan.ColumnMaterialization(
        groups={
            axis: _axis_materialization(group, keys_by_axis[axis])
            for axis, group in recognition.column_groups()
        },
        applier=applier,
    )


def _axis_materialization(
    group: model.ColumnGroup, keys: Sequence[str]
) -> column_plan.AxisMaterialization:
    """Split one group's declarations at the axis-key closure: prepared before the pivot,
    the rest afterwards on the deduplicated axis frame.
    """
    declared = tuple(model.group_names(group))
    closure = runtime.key_closure(group, keys)
    return column_plan.AxisMaterialization(
        declared=declared,
        keys=_materialization_pass(group, closure),
        rest=_materialization_pass(group, {name for name in declared if name not in closure}),
    )


def _materialization_pass(
    group: model.ColumnGroup, names: AbstractSet[str]
) -> column_plan.MaterializationPass:
    """One pass over the declarations whose output name is in ``names``."""
    select = tuple((name, source) for name, source in group.select.items() if name in names)
    optional = tuple(
        (name, source) for name, source in group.optional_select.items() if name in names
    )
    return column_plan.MaterializationPass(
        select=select,
        optional=optional,
        coercers={
            name: coercer_for(group.types.get(name, "string"))
            for name, _source in (*select, *optional)
        },
        computers=tuple(computer_for(column) for column in group.computed if column.name in names),
    )


# ------------------------------------------------------------------------------ layers[]


def coercion_for(layer: model.Layer) -> layers.LayerCoercion:
    """Read a layer declaration's encoding flag once, and return the coercion it names."""
    if layer.encoding_mode == "factor":
        return layers.FactorCoercion(layer.categories)
    missing_values = tuple(layer.missing_values)
    if isinstance(layer.value_pattern, model.RegexValuePattern):
        return layers.RegexNumericCoercion(missing_values, layer.value_pattern.pattern)
    return layers.PlainNumericCoercion(missing_values)


def _layer_plans(rule: Rule) -> tuple[layers.LayerPlan, ...]:
    """One plan per declared layer, with the rule-level ``required`` question folded in."""
    return tuple(
        layers.LayerPlan(
            name=layer.name,
            source=layer.source,
            required=runtime.layer_required(rule, layer),
            coercion=coercion_for(layer),
        )
        for layer in rule.layers
    )


# ----------------------------------------------------------------------- axis.duplicates


POLICY_BY_MODE: Mapping[model.DuplicateMode, duplicates.DuplicatePolicy] = {
    "error": duplicates.ErrorOnDuplicates(),
    "keep_first": duplicates.KeepFirstDuplicate(),
    "aggregate": duplicates.SumDuplicates(),
    # "keep_all_as_raw_table" is absent on purpose: the schema accepts the mode and no
    # policy implements it, so policy_for raises once, naming it, before any conversion
    # work starts.
}


def policy_for(declared: model.Duplicates) -> duplicates.DuplicatePolicy:
    """Select the policy a rule's duplicate mode names.

    Raises NotImplementedError for a mode the schema permits but no policy implements.
    """
    policy = POLICY_BY_MODE.get(declared.mode)
    if policy is None:
        raise NotImplementedError(f"duplicates.mode={declared.mode!r} is not yet supported")
    return policy


# ---------------------------------------------------------------------------- fragments


def exploder_for(rule: Rule, header: list[str]) -> parse_strategy.FragmentExploder:
    """Read the rule's ``label_strategy`` once, and return the exploder it names.

    Packed columns resolve against the header here: a missing column backing a required
    layer fails construction; a missing optional one is dropped so the conversion skips
    its layer, exactly as it does on the non-fragment path.
    """
    declared: model.Fragments | None = rule.fragments
    if declared is None:
        return fragment_exploder.NoFragments()
    present = set(header)
    required_sources = {
        layer.source for layer in rule.layers if runtime.layer_required(rule, layer)
    }
    missing = [
        column
        for column in declared.value_columns
        if column not in present and column in required_sources
    ]
    if missing:
        raise IncompatibleSourceError(
            f"input lacks the packed fragment column(s) {missing} required by "
            f"{runtime.rule_label(rule)}"
        )
    value_columns = tuple(column for column in declared.value_columns if column in present)
    if not value_columns:
        raise IncompatibleSourceError(
            f"input carries none of the packed fragment columns {list(declared.value_columns)} "
            f"declared by {runtime.rule_label(rule)}"
        )
    if isinstance(declared, model.ColumnLabeledFragments):
        return fragment_exploder.ColumnLabeledExplode(
            label_column=declared.label_column,
            label_output=declared.label_output,
            delimiter=declared.delimiter,
            value_columns=value_columns,
        )
    return fragment_exploder.PositionalExplode(
        label_output=declared.label_output,
        delimiter=declared.delimiter,
        value_columns=value_columns,
    )


# ------------------------------------------------------------------------ modifications


def applier_for(rule: Rule) -> modifications.ModificationApplier:
    """Read the ``parser`` selector once; return the applier it names, or the identity.

    The absence questions — is a block declared, does anything read its output — are
    asked once here instead of on every table, and the Unimod map is resolved in the same
    step, so an unknown accession fails before any table is read.
    """
    declared = rule.modifications
    if declared is None:
        return modifications.NoModifications()
    consumed = any(
        isinstance(column, model.ProformaSequence | model.StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return modifications.NoModifications()
    columns = modifications.SequenceColumns(
        output_column=declared.output_column,
        sources=runtime.modification_sources(declared),
        outputs=model.modification_outputs(declared),
    )
    entries = apply_rules.map_entries((entry.token, entry.accession) for entry in declared.map)
    if isinstance(declared, model.SiteListModifications):
        return modifications.SiteListApplier(
            columns,
            apply_rules.SiteListRule(
                delimiter=declared.delimiter,
                site_base=declared.site_base,
                case_sensitive=declared.case_sensitive,
                unknown_policy=declared.unknown_policy,
                entries=entries,
            ),
        )
    return modifications.TokenRegexApplier(
        columns,
        apply_rules.ModificationRule(
            token_pattern=declared.token_pattern,
            token_position=declared.token_position,
            case_sensitive=declared.case_sensitive,
            unknown_policy=declared.unknown_policy,
            entries=entries,
        ),
    )


# -------------------------------------------------------------------------------- shape


def conversion_for(rule: Rule, *, strict: bool) -> parse_strategy.TableConversion:
    """Read a rule's shape once, and return the conversion it names."""
    layer_plans = _layer_plans(rule)
    policy = policy_for(rule.axis.duplicates)
    var_carry = runtime.carried_columns(
        rule.axis.var_keys, rule.columns.var, runtime.var_extras(rule)
    )
    if isinstance(rule, model.LongRule):
        return table_conversion.LongConversion(
            obs_keys=rule.axis.obs_keys,
            var_keys=rule.axis.var_keys,
            obs_carry=runtime.carried_columns(rule.axis.obs_keys, rule.columns.obs, ()),
            var_carry=var_carry,
            layers=layer_plans,
            x_layer=rule.axis.x_layer,
            duplicates=policy,
            strict=strict,
        )
    return table_conversion.WideConversion(
        obs_outputs=rule.axis.obs_keys,
        var_keys=rule.axis.var_keys,
        var_carry=var_carry,
        layers=layer_plans,
        x_layer=rule.axis.x_layer,
        duplicates=policy,
        software_name=rule.software_name,
        strict=strict,
    )
