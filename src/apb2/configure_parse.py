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

from apb2 import rule_reading
from apb2.errors import IncompatibleSourceError, NoCompatibleLevelError, RuleNotApplicable
from apb2.parse_quant import (
    bound_input_reader,
    column_plan,
    duplicates,
    fragment_exploder,
    layers,
    parse_strategy,
    table_conversion,
)
from apb2.parse_quant.modifications import applier as modification_applier
from apb2.parse_quant.modifications import normalize_sequence
from apb2.parse_quant.sources import InputSource
from apb2.serialization import JsonValue
from apb2.vendor_parse_rules import model
from apb2.vendor_parse_rules.rules import Recognition, Rule

type RuleConfig = model.LongRule | model.WideRule
"""The validated declaration inside a ``Rule``; every selector below reads one."""


# ------------------------------------------------------------------- the composition root


def make_parse_strategy(
    rule: Rule,
    source: InputSource,
    *,
    strict: bool = False,
) -> parse_strategy.Parser:
    """Construct one fully injected parser for one quantification level.

    ``rule`` is what the rules door returned: the declaration the evidence selected, and the
    recognition built with it. Nothing here reads a search parameter, and nothing rebuilds
    the recognition. Raises ``IncompatibleSourceError`` when ``source`` cannot satisfy the
    rule; ``strict`` promotes non-``X`` layer-contract warnings to errors.
    """
    config, recognition = rule.config, rule.recognition
    label = rule_reading.rule_label(config)
    binding = bound_input_reader.bind_source(source, accepts=recognition.matches, rule_label=label)
    header = binding.header()
    if not recognition.matches(header):
        raise IncompatibleSourceError(
            f"{binding.path} does not carry the columns required by {label}"
        )
    applier = modifications_for(config)
    missing = [column for column in applier.sources if column not in set(header)]
    if missing:
        raise IncompatibleSourceError(
            f"{binding.path} lacks the [modifications] source column(s) {missing} required "
            f"by {label}"
        )
    exploder = exploder_for(config, header)
    columns_read = [*applier.source_columns(), *exploder.packed_columns()]
    return parse_strategy.Parser(
        level=config.quantification_level,
        input=binding.make_reader(_read_plan(recognition, header, columns_read)),
        fragments=exploder,
        columns=column_plan_for(config, recognition, applier),
        conversion=conversion_for(config, strict=strict),
        provenance=_provenance(config),
    )


def make_parse_strategies(
    rules: Iterable[Rule],
    source: InputSource,
    *,
    strict: bool = False,
) -> list[parse_strategy.Parser]:
    """Construct one parser per rule the source satisfies, in ``LEVELS`` order.

    Incompatible rules never gate construction of the compatible ones; an empty result
    is an error because a source that satisfies nothing was the wrong source.
    """
    ordered = sorted(rules, key=lambda rule: model.LEVELS.index(rule.config.quantification_level))
    parsers: list[parse_strategy.Parser] = []
    for rule in ordered:
        try:
            parsers.append(make_parse_strategy(rule, source, strict=strict))
        except RuleNotApplicable:
            continue
    if not parsers:
        levels = [rule.config.quantification_level for rule in ordered]
        raise NoCompatibleLevelError(
            f"no rule among levels {levels} is satisfied by the bound source {source!r}"
        )
    return parsers


def _read_plan(
    recognition: Recognition, header: Sequence[str], also: Iterable[str]
) -> bound_input_reader.ReadPlan:
    """Fix the projection: everything the rule reads, plus ``also``.

    ``also`` is what the resolved strategies decided they touch against this same header —
    modification sources and outputs, packed fragment columns. They are asked for those
    names at the call site: a projection is names, so nothing here holds a strategy.
    """
    columns = rule_reading.projected_columns(recognition, header, also)
    return bound_input_reader.ReadPlan(
        columns=columns,
        string_sources=rule_reading.string_typed_sources(recognition) & frozenset(columns),
    )


def _provenance(rule: RuleConfig) -> dict[str, JsonValue]:
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
    rule: RuleConfig, recognition: Recognition, applier: modification_applier.ModificationApplier
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
    closure = rule_reading.key_closure(group, keys)
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


def _layer_plans(rule: RuleConfig) -> tuple[layers.LayerPlan, ...]:
    """One plan per declared layer, with the rule-level ``required`` question folded in."""
    return tuple(
        layers.LayerPlan(
            name=layer.name,
            source=layer.source,
            required=model.layer_required(rule, layer),
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


def exploder_for(rule: RuleConfig, header: list[str]) -> parse_strategy.FragmentExploder:
    """Read the rule's ``label_strategy`` once, and return the exploder it names.

    Packed columns resolve against the header here: a missing column backing a required
    layer fails construction; a missing optional one is dropped so the conversion skips
    its layer, exactly as it does on the non-fragment path.
    """
    declared: model.Fragments | None = rule.fragments
    if declared is None:
        return fragment_exploder.NoFragments()
    present = set(header)
    required_sources = {layer.source for layer in rule.layers if model.layer_required(rule, layer)}
    missing = [
        column
        for column in declared.value_columns
        if column not in present and column in required_sources
    ]
    if missing:
        raise IncompatibleSourceError(
            f"input lacks the packed fragment column(s) {missing} required by "
            f"{rule_reading.rule_label(rule)}"
        )
    value_columns = tuple(column for column in declared.value_columns if column in present)
    if not value_columns:
        raise IncompatibleSourceError(
            f"input carries none of the packed fragment columns {list(declared.value_columns)} "
            f"declared by {rule_reading.rule_label(rule)}"
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


def modifications_for(rule: RuleConfig) -> modification_applier.ModificationApplier:
    """Read the ``[modifications]`` block, and hand its settings to the applier factory.

    Everything rule-shaped happens here — is a block declared, does anything read its
    output, what do its fields say — including resolving the Unimod map, so an unknown
    accession fails before any table is read. Which class those settings name is
    ``applier_for``'s decision, and it lives beside the classes.
    """
    declared = rule.modifications
    if declared is None:
        return modification_applier.NoModifications()
    consumed = any(
        isinstance(column, model.ProformaSequence | model.StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return modification_applier.NoModifications()
    columns = modification_applier.SequenceColumns(
        output_column=declared.output_column,
        sources=rule_reading.modification_sources(declared),
        outputs=model.modification_outputs(declared),
    )
    entries = normalize_sequence.map_entries(
        (entry.token, entry.accession) for entry in declared.map
    )
    settings: normalize_sequence.TokenRegexSettings | normalize_sequence.SiteListSettings
    if isinstance(declared, model.SiteListModifications):
        settings = normalize_sequence.SiteListSettings(
            delimiter=declared.delimiter,
            site_base=declared.site_base,
            case_sensitive=declared.case_sensitive,
            unknown_policy=declared.unknown_policy,
            entries=entries,
        )
    else:
        settings = normalize_sequence.TokenRegexSettings(
            token_pattern=declared.token_pattern,
            token_position=declared.token_position,
            case_sensitive=declared.case_sensitive,
            unknown_policy=declared.unknown_policy,
            entries=entries,
        )
    return modification_applier.applier_for(columns, settings)


# -------------------------------------------------------------------------------- shape


def conversion_for(rule: RuleConfig, *, strict: bool) -> parse_strategy.TableConversion:
    """Read a rule's shape once, and return the conversion it names."""
    layer_plans = _layer_plans(rule)
    policy = policy_for(rule.axis.duplicates)
    var_carry = rule_reading.carried_columns(
        rule.axis.var_keys, rule.columns.var, rule_reading.var_extras(rule)
    )
    if isinstance(rule, model.LongRule):
        return table_conversion.LongConversion(
            obs_keys=rule.axis.obs_keys,
            var_keys=rule.axis.var_keys,
            obs_carry=rule_reading.carried_columns(rule.axis.obs_keys, rule.columns.obs, ()),
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
