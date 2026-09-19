"""Construct parser runtime collaborators from resolved rule declarations.

Every declarative discriminator is consumed here, exactly once, and the object that comes back
carries no tag. That is the whole point of the module: past this boundary nothing asks what
vendor, level, layout, value form, duplicate mode, or output format it is dealing with, because
the answer has already become behaviour.

Two kinds of dispatch appear below, and the difference is deliberate. Where a tag selects among
stateless implementations, a table maps the tag to the instance. Where construction needs the
declaration's own fields, one function per family narrows the closed union — which is the same
single dispatch point, with exhaustiveness checked by the type checker instead of by a string
key.

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Literal

from apb2.parserV2.parse_quant.axis_columns import (
    BooleanAxisCoercer,
    CoalesceColumn,
    DerivedSequenceColumn,
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
    BoundInputReader,
    ColumnComputer,
    DuplicatePolicy,
    FragmentTableSeparator,
    LayerSetValidator,
    LayerValueParser,
    ModificationNormalizer,
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
    EmbeddedSiteListRules,
    SiteListNormalizer,
    SiteListRules,
    TokenRegexNormalizer,
    TokenRegexRules,
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
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerContractConfig,
    LayerValueConfig,
    PlainNumericLayerConfig,
    PlainNumericRawValuePresenceConfig,
    RawValuePresenceConfig,
    RegexNumericLayerConfig,
    RegexNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.plan_json import PLAN_JSON_KEY, resolved_plan_json
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
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
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.prepare_source import prepare_source
from apb2.parserV2.source_binding import (
    bind_source,
    header_predicate,
    make_reader,
    source_evidence,
)

# ----------------------------------------------------------------------------- registries

_DUPLICATE_POLICIES: Mapping[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": AggregateNumericDuplicates(),
}
"""One policy per executable duplicate mode; schema 0.7 declares no others."""


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


def policy_for(mode: DuplicateMode) -> DuplicatePolicy:
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
    if isinstance(config, StrippedSequenceColumnConfig | ProformaSequenceColumnConfig):
        return DerivedSequenceColumn(name=config.name, inputs=config.inputs)
    if isinstance(config, ProformaIonColumnConfig):
        return ProformaIonColumn(name=config.name, inputs=config.inputs)
    return ProformaFragmentColumn(name=config.name, inputs=config.inputs)


def make_modification_normalizer(config: ModificationConfig) -> ModificationNormalizer:
    """Construct the normalizer one modification declaration describes."""
    if isinstance(config, SiteListModificationConfig):
        return SiteListNormalizer(
            rules=SiteListRules(
                delimiter=config.delimiter,
                site_base=config.site_base,
                case_sensitive=config.case_sensitive,
                unknown_policy=config.unknown_policy,
                entries=config.entries,
            ),
            sources=(config.sequence_column, config.modification_column, config.site_column),
            proforma_output=config.proforma_output,
            stripped_output=config.stripped_output,
        )
    if isinstance(config, EmbeddedSiteListModificationConfig):
        return EmbeddedSiteListNormalizer(
            rules=EmbeddedSiteListRules(
                delimiter=config.delimiter,
                entry_pattern=config.entry_pattern,
                site_base=config.site_base,
                case_sensitive=config.case_sensitive,
                unknown_policy=config.unknown_policy,
                entries=config.entries,
            ),
            sources=(config.sequence_column, config.modification_column),
            proforma_output=config.proforma_output,
            stripped_output=config.stripped_output,
        )
    return TokenRegexNormalizer(
        rules=TokenRegexRules(
            token_pattern=config.token_pattern,
            token_position=config.token_position,
            case_sensitive=config.case_sensitive,
            unknown_policy=config.unknown_policy,
            entries=config.entries,
        ),
        sources=(config.source_column,),
        proforma_output=config.proforma_output,
        stripped_output=config.stripped_output,
    )


def make_raw_value_presence(config: RawValuePresenceConfig) -> RawValuePresence:
    """Construct the presence strategy one resolved layer declaration describes."""
    if isinstance(config, PlainNumericRawValuePresenceConfig):
        return PlainNumericRawValuePresence(
            missing_values=config.missing_values,
            number_format=_notation(config.number_format),
        )
    if isinstance(config, RegexNumericRawValuePresenceConfig):
        return RegexNumericRawValuePresence(
            missing_values=config.missing_values,
            pattern=config.pattern,
            number_format=_notation(config.number_format),
        )
    return NullOnlyRawValuePresence()


def make_layer_value_parser(config: LayerValueConfig) -> LayerValueParser:
    """Construct the parser for one resolved layer value declaration."""
    if isinstance(config, PlainNumericLayerConfig):
        return PlainNumericLayerParser(
            layer_name=config.layer_name,
            missing_values=config.missing_values,
            number_format=_notation(config.number_format),
            numeric_type=config.type,
        )
    if isinstance(config, RegexNumericLayerConfig):
        return RegexNumericLayerParser(
            layer_name=config.layer_name,
            missing_values=config.missing_values,
            pattern=config.pattern,
            number_format=_notation(config.number_format),
            numeric_type=config.type,
        )
    return FactorLayerParser(categories=config.categories)


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
    reader_from: Callable[[ResolvedLevelPlan], BoundInputReader]
    if isinstance(source, PreparedTable):
        evidence = FrameSourceEvidence(
            columns=tuple(source.frame.columns), dtypes=tuple(source.frame.schema.items())
        )
        reader_from = partial(PreparedInputReader, source.frame)
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
        bound = bind_source(source, working.input)
        evidence = source_evidence(source, bound, header_predicate(working))

        def read_physical(plan: ResolvedLevelPlan) -> BoundInputReader:
            return make_reader(bound, evidence, plan.read)

        reader_from = read_physical
    resolved = facade.resolve_source(evidence)
    layer_values = {config.layer_name: config for config in resolved.layer_values}
    return Parser(
        level=resolved.level,
        input_reader=reader_from(resolved),
        decomposer=make_source_decomposer(
            resolved.decomposition, resolved.obs.source, resolved.var.source
        ),
        obs_plan=make_axis_runtime_plan(resolved.obs, resolved.number_format),
        var_plan=make_axis_runtime_plan(resolved.var, resolved.number_format),
        modification_normalizers=tuple(
            make_modification_normalizer(config) for config in resolved.modifications
        ),
        duplicates=policy_for(resolved.duplicate_mode),
        raw_value_presence={
            config.layer_name: make_raw_value_presence(config)
            for config in resolved.raw_value_presence
        },
        layer_parsers={
            name: make_layer_value_parser(layer_values[name])
            for name in (config.layer_name for config in resolved.raw_value_presence)
        },
        layer_validator=make_layer_validator(resolved.layer_contract, checks),
        writer=ParsedLevelFormatWriter(),
        provenance={
            **resolved.provenance,
            **preparation,
            PLAN_JSON_KEY: resolved_plan_json(resolved),
        },
    )
