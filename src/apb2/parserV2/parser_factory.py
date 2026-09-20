"""Construct parser runtime collaborators from resolved rule declarations.

Every declarative discriminator selects behavior here, exactly once. Settings may retain tags
for provenance; past this boundary nothing dispatches on what
vendor, level, layout, value form, duplicate mode, or output format it is dealing with, because
the answer has already become behaviour.

Two kinds of dispatch appear below, and the difference is deliberate. Where a tag selects among
stateless implementations, a table maps the tag to the instance. Where construction needs the
declaration's own fields, one function per family narrows the closed union — which is the same
single dispatch point, with exhaustiveness checked by the type checker instead of by a string
key.

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from apb2.parserV2.parse_quant.axis_columns import (
    BooleanAxisCoercer,
    CoalesceColumn,
    IntegerAxisCoercer,
    JoinNonemptyColumn,
    NumberAxisCoercer,
    ProformaFragmentColumn,
    ProformaIonColumn,
    StringAxisCoercer,
)
from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    AxisValueCoercer,
    ColumnComputer,
    DuplicatePolicy,
    FragmentTableSeparator,
    LayerSetValidator,
    LayerValueParser,
    RawValuePresence,
    SelectedAxisColumn,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
    NullOnlyRawValuePresence,
    PlainNumericRawValuePresence,
    RegexNumericRawValuePresence,
)
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.io.formats import ParsedLevelFormatWriter
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    PlainSequenceStripper,
    SequenceColumn,
    SequenceOperation,
    SiteListNormalizer,
    TokenRegexNormalizer,
    TokenRegexStripper,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisLogicalType,
    AxisMaterializationConfig,
    AxisSourcePlan,
    CoalesceColumnConfig,
    ComputedColumnConfig,
    EmbeddedSiteListModificationConfig,
    JoinNonemptyColumnConfig,
    ModificationConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    ResolvedAxisColumnPlan,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
    StrippingSyntaxConfig,
    TokenRegexSyntaxConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerContractConfig,
    LayerValueConfig,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DecompositionConfig,
    FragmentSeparationConfig,
    FrameSourceEvidence,
    InputSource,
    LongDecompositionConfig,
    NumericTextFormat,
    PositionalFragmentSeparationConfig,
    PreparedTable,
    WideDecompositionConfig,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.plan_json import PLAN_JSON_KEY, resolved_plan_json
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.prepare_source import prepare_source
from apb2.parserV2.source_binding import BoundTable

# ----------------------------------------------------------------------------- registries

_DUPLICATE_POLICIES: Mapping[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": AggregateNumericDuplicates(),
}
"""One policy per executable duplicate mode; schema 0.8 declares no others."""


def _notation(config: NumericTextFormat, /) -> NumberNotation:
    return NumberNotation(
        decimal_mark=config.decimal_mark,
        thousands_marks=config.thousands_marks,
    )


def make_axis_coercer(
    logical_type: AxisLogicalType,
    number_format: NumericTextFormat,
) -> AxisValueCoercer:
    """Construct the coercion one logical type names under this source's number notation."""
    if logical_type == "integer":
        return IntegerAxisCoercer(notation=_notation(number_format))
    if logical_type == "number":
        return NumberAxisCoercer(notation=_notation(number_format))
    if logical_type == "boolean":
        return BooleanAxisCoercer()
    return StringAxisCoercer()


def duplicate_policy_for(mode: DuplicateMode) -> DuplicatePolicy:
    """Select the policy one resolved duplicate mode names."""
    return _DUPLICATE_POLICIES[mode]


def make_column_computer(config: ComputedColumnConfig) -> ColumnComputer:
    """Construct the computed column one declaration describes."""
    if isinstance(config, CoalesceColumnConfig):
        return CoalesceColumn(name=config.name, inputs=config.inputs)
    if isinstance(config, JoinNonemptyColumnConfig):
        return JoinNonemptyColumn(
            name=config.name, inputs=config.inputs, separator=config.separator
        )
    if isinstance(config, StrippedSequenceColumnConfig):
        return SequenceColumn(
            name=config.name, inputs=config.inputs, operation=make_sequence_stripper(config.syntax)
        )
    if isinstance(config, ProformaSequenceColumnConfig):
        return SequenceColumn(
            name=config.name,
            inputs=config.inputs,
            operation=make_sequence_normalizer(config.normalization),
        )
    if isinstance(config, ProformaIonColumnConfig):
        return ProformaIonColumn(name=config.name, inputs=config.inputs)
    return ProformaFragmentColumn(name=config.name, inputs=config.inputs)


def make_sequence_stripper(config: StrippingSyntaxConfig) -> SequenceOperation:
    """Construct residue extraction without looking up modification identities."""
    if isinstance(config, TokenRegexSyntaxConfig):
        return TokenRegexStripper(config.token_pattern, config.token_position)
    return PlainSequenceStripper()


def make_sequence_normalizer(config: ModificationConfig) -> SequenceOperation:
    """Construct the normalizer one modification declaration describes."""
    if isinstance(config, SiteListModificationConfig):
        return SiteListNormalizer(rules=config)
    if isinstance(config, EmbeddedSiteListModificationConfig):
        return EmbeddedSiteListNormalizer(rules=config)
    return TokenRegexNormalizer(rules=config)


def make_layer_operations(
    config: LayerValueConfig, numbers: NumericTextFormat
) -> tuple[RawValuePresence, LayerValueParser]:
    """Select raw presence and canonical parsing once from the same layer declaration."""
    value = config.value
    notation = _notation(numbers)
    if isinstance(value, PlainNumericLayerDeclaration):
        presence = (
            PlainNumericRawValuePresence(value.missing_values, notation)
            if value.missing_values
            else NullOnlyRawValuePresence()
        )
        return presence, PlainNumericLayerParser(
            layer_name=config.layer_name,
            missing_values=value.missing_values,
            number_format=notation,
            numeric_type=value.type,
        )
    if isinstance(value, RegexNumericLayerDeclaration):
        return (
            RegexNumericRawValuePresence(value.missing_values, value.pattern, notation),
            RegexNumericLayerParser(
                layer_name=config.layer_name,
                missing_values=value.missing_values,
                pattern=value.pattern,
                number_format=notation,
                numeric_type=value.type,
            ),
        )
    return NullOnlyRawValuePresence(), FactorLayerParser(categories=value.categories)


def make_layer_validator(
    config: LayerContractConfig,
    checks: Literal["standard", "strict"],
) -> LayerSetValidator:
    """Construct validation for relationships across the final canonical layer set."""
    return LayerContractValidator(
        primary_layer_name=config.primary_layer_name,
        required_names=config.required_names,
        empty_ratio=config.empty_ratio,
        populated_ratio=config.populated_ratio,
        strict=checks == "strict",
    )


def make_fragment_table_separator(config: FragmentSeparationConfig) -> FragmentTableSeparator:
    """Construct the separator one packed-fragment declaration describes."""
    if isinstance(config, PositionalFragmentSeparationConfig):
        return PositionalFragmentTableSeparator(
            label_output=config.label_output,
            delimiter=config.delimiter,
            packed_value_sources=config.packed_value_sources,
        )
    return ColumnLabeledFragmentTableSeparator(
        label_source=config.label_source,
        label_output=config.label_output,
        delimiter=config.delimiter,
        packed_value_sources=config.packed_value_sources,
    )


def make_source_decomposer(
    config: DecompositionConfig, obs: AxisSourcePlan, var: AxisSourcePlan
) -> SourceDecomposer:
    """Construct the one physical-shape strategy this level's table needs.

    The fragment path is composition, not a third algorithm: it receives a separator and the
    same ordinary long decomposer direct long input uses.
    """
    if isinstance(config, WideDecompositionConfig):
        return WideSourceDecomposer(
            primary_layer_name=config.primary_layer_name,
            layer_plans=config.layer_plans,
            obs=obs,
            var=var,
        )
    if isinstance(config, LongDecompositionConfig):
        return _long_decomposer(config, obs, var)
    return DelimitedFragmentSourceDecomposer(
        separator=make_fragment_table_separator(config.separator),
        long_decomposer=_long_decomposer(config.long, obs, var),
    )


def _long_decomposer(
    config: LongDecompositionConfig, obs: AxisSourcePlan, var: AxisSourcePlan
) -> LongSourceDecomposer:
    return LongSourceDecomposer(
        primary_layer_name=config.primary_layer_name,
        layer_sources=config.layer_sources,
        obs=obs,
        var=var,
    )


def make_axis_runtime_plan(
    resolved: ResolvedAxisColumnPlan,
    number_format: NumericTextFormat,
) -> AxisRuntimePlan:
    """Turn one resolved axis plan into configured behaviour, phase by phase."""
    return AxisRuntimePlan(
        keys=resolved.source.keys,
        key_phase=_runtime_phase(resolved.key_phase, number_format),
        output_phase=_runtime_phase(resolved.output_phase, number_format),
        outputs=resolved.outputs,
    )


def _runtime_phase(
    phase: AxisMaterializationConfig,
    number_format: NumericTextFormat,
) -> AxisPhaseRuntimePlan:
    return AxisPhaseRuntimePlan(
        selections=tuple(
            SelectedAxisColumn(
                name=selection.name,
                source=selection.source,
                coercer=make_axis_coercer(selection.logical_type, number_format),
            )
            for selection in phase.selections
        ),
        computers=tuple(make_column_computer(config) for config in phase.computers),
    )


# ------------------------------------------------------------------- the fixed compilation


def compile_level(
    facade: ParseRuleFacade,
    source: InputSource,
    checks: Literal["standard", "strict"],
) -> Parser:
    """Resolve one selected level and construct its fully configured parser."""
    working = facade.working_parameters
    source = prepare_source(source, working.preparation)
    preparation: dict[str, JsonValue] = {}
    if isinstance(source, PreparedTable):
        evidence = FrameSourceEvidence(
            columns=tuple(source.frame.columns), dtypes=tuple(source.frame.schema.items())
        )
        resolved = facade.resolve_source(evidence)
        input_reader = PreparedInputReader(source.frame, resolved)
        preparation = {
            "input_preparation": {
                "how": source.how,
                "sources": [str(path) for path in source.source_paths],
                "duration_seconds": source.duration_seconds,
                "rows": source.frame.height,
                "estimated_size_bytes": source.frame.estimated_size(),
            }
        }
    else:
        bound = BoundTable(source, working.input)
        evidence = bound.evidence(working.accepts_header)
        resolved = facade.resolve_source(evidence)
        input_reader = bound.reader(evidence, resolved.read)
    raw_value_presence: dict[str, RawValuePresence] = {}
    layer_parsers: dict[str, LayerValueParser] = {}
    for config in resolved.layer_values:
        raw_value_presence[config.layer_name], layer_parsers[config.layer_name] = (
            make_layer_operations(config, resolved.number_format)
        )
    return Parser(
        level=resolved.level,
        input_reader=input_reader,
        decomposer=make_source_decomposer(
            resolved.decomposition, resolved.obs.source, resolved.var.source
        ),
        obs_plan=make_axis_runtime_plan(resolved.obs, resolved.number_format),
        var_plan=make_axis_runtime_plan(resolved.var, resolved.number_format),
        duplicates=duplicate_policy_for(resolved.duplicate_mode),
        raw_value_presence=raw_value_presence,
        layer_parsers=layer_parsers,
        layer_validator=make_layer_validator(resolved.layer_contract, checks),
        writer=ParsedLevelFormatWriter(),
        provenance={
            **resolved.provenance,
            **preparation,
            PLAN_JSON_KEY: resolved_plan_json(resolved),
        },
    )
