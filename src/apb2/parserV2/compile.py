"""The composition root: bind one physical source, then build one parser per level.

Every declarative discriminator is consumed here, exactly once, and the object that comes back
carries no tag. That is the whole point of the module: past this boundary nothing asks what
vendor, level, layout, encoding, duplicate mode, or output format it is dealing with, because
the answer has already become behaviour.

Two kinds of dispatch appear below, and the difference is deliberate. Where a tag selects among
stateless implementations, a table maps the tag to the instance. Where construction needs the
declaration's own fields, one function per family narrows the closed union — which is the same
single dispatch point, with exhaustiveness checked by the type checker instead of by a string
key.

Source binding is allowed to branch on evidence, because an outcome is a fact about a file
rather than a behaviour selector: no declared format accepts this extension; no delimiter
candidate exposes the required header; several candidates do; a folder holds zero or several
declared candidates. Each of those is reported as what it is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from apb2.parserV2.parse_quant import delimited_input, parquet_input
from apb2.parserV2.parse_quant.anndata_writer import (
    AnnDataLayerContractChecker,
    AnnDataLayerEncoder,
    AnnDataWriter,
    FactorAnnDataEncoder,
    OccupancyPolicy,
    PlainNumericAnnDataEncoder,
    RegexNumericAnnDataEncoder,
    StandardAnnDataLayerContract,
    StrictAnnDataLayerContract,
)
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
    ColumnComputer,
    DuplicatePolicy,
    FragmentTableSeparator,
    ModificationNormalizer,
    ParsedLevelWriter,
    RawValuePresence,
    SelectedAxisColumn,
    SourceDecomposer,
)
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
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.modifications import (
    SiteListNormalizer,
    SiteListRules,
    TokenRegexNormalizer,
    TokenRegexRules,
)
from apb2.parserV2.parse_quant.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisLogicalType,
    AxisMaterializationConfig,
    AxisSourcePlan,
    CoalesceColumnConfig,
    ComputedColumnConfig,
    JoinNonemptyColumnConfig,
    ModificationConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    ResolvedAxisColumnPlan,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    AnnDataLayerEncodingConfig,
    AnnDataSerializationConfig,
    DuplicateMode,
    FactorAnnDataEncodingConfig,
    PlainNumericRawValuePresenceConfig,
    RawValuePresenceConfig,
    RegexNumericAnnDataEncodingConfig,
    RegexNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.plan_json import PLAN_JSON_KEY, resolved_plan_json
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
from apb2.parserV2.parse_quant.parameters.source import (
    DecompositionConfig,
    DelimitedFile,
    Folder,
    FragmentSeparationConfig,
    InputContract,
    InputSource,
    LevelReadPlan,
    LongDecompositionConfig,
    NumericTextFormat,
    ParquetFormatContract,
    ParquetSourceEvidence,
    PhysicalFormatContract,
    PositionalFragmentSeparationConfig,
    SingleFile,
    SourceEvidence,
    WideDecompositionConfig,
)
from apb2.parserV2.parse_quant.parameters.working import (
    ColumnLabeledFragmentLayout,
    PositionalFragmentLayout,
    QuantificationLevel,
    WideSourceLayout,
    WorkingParseConfiguration,
)
from apb2.parserV2.parse_quant.parquet_writer import ParquetWriter
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS


class NoCompatibleLevelError(ValueError):
    """No requested level can be compiled against the supplied source."""


# ---------------------------------------------------------------- the output declaration


@dataclass(frozen=True, slots=True)
class AnnDataOutput:
    """Persist as ``.h5ad``, with the layer contract checked as strictly as asked."""

    checks: Literal["standard", "strict"] = "standard"


@dataclass(frozen=True, slots=True)
class ParquetOutput:
    """Persist as a Parquet directory dataset. Nothing to configure."""


type OutputDeclaration = AnnDataOutput | ParquetOutput


# ----------------------------------------------------------------------------- registries

_AXIS_COERCERS: Mapping[AxisLogicalType, AxisValueCoercer] = {
    "string": StringAxisCoercer(),
    "integer": IntegerAxisCoercer(),
    "number": NumberAxisCoercer(),
    "boolean": BooleanAxisCoercer(),
}
"""One coercion per declared logical axis-column type; every coercer is stateless."""

_DUPLICATE_POLICIES: Mapping[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": AggregateNumericDuplicates(),
}
"""One policy per executable duplicate mode; schema 0.3 declares no others."""


def axis_coercer_for(logical_type: AxisLogicalType) -> AxisValueCoercer:
    """Select the coercion one declared logical type names."""
    return _AXIS_COERCERS[logical_type]


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


def make_anndata_layer_encoder(config: AnnDataLayerEncodingConfig) -> AnnDataLayerEncoder:
    """Construct the encoder one resolved layer declaration describes."""
    if isinstance(config, FactorAnnDataEncodingConfig):
        return FactorAnnDataEncoder(layer_name=config.layer_name, categories=config.categories)
    if isinstance(config, RegexNumericAnnDataEncodingConfig):
        return RegexNumericAnnDataEncoder(
            layer_name=config.layer_name,
            missing_values=config.missing_values,
            pattern=config.pattern,
            number_format=_notation(config.number_format),
        )
    return PlainNumericAnnDataEncoder(
        layer_name=config.layer_name,
        missing_values=config.missing_values,
        number_format=_notation(config.number_format),
    )


def _notation(number_format: NumericTextFormat) -> NumberNotation:
    """Hand a strategy the two scalars it needs, not the parsing parameter holding them."""
    return NumberNotation(
        decimal_mark=number_format.decimal_mark,
        thousands_marks=number_format.thousands_marks,
    )


def make_anndata_layer_contract_checker(
    config: AnnDataSerializationConfig, checks: Literal["standard", "strict"]
) -> AnnDataLayerContractChecker:
    """Select the contract check the caller asked for, configured for this level."""
    policy = OccupancyPolicy(
        primary_layer_name=config.layer_contract.primary_layer_name,
        required_names=config.layer_contract.required_names,
        empty_ratio=config.layer_contract.empty_ratio,
        populated_ratio=config.layer_contract.populated_ratio,
    )
    if checks == "strict":
        return StrictAnnDataLayerContract(policy=policy)
    return StandardAnnDataLayerContract(policy=policy)


def make_parsed_level_writer(
    output: OutputDeclaration, resolved: ResolvedLevelPlan
) -> ParsedLevelWriter:
    """Consume the output choice once. A Parquet compile constructs no encoder at all."""
    if isinstance(output, ParquetOutput):
        return ParquetWriter()
    retained = tuple(config.layer_name for config in resolved.raw_value_presence)
    encodings = {config.layer_name: config for config in resolved.ann_data.layer_encodings}
    return AnnDataWriter(
        encoders={name: make_anndata_layer_encoder(encodings[name]) for name in retained},
        contract=make_anndata_layer_contract_checker(resolved.ann_data, output.checks),
    )


def make_axis_runtime_plan(resolved: ResolvedAxisColumnPlan) -> AxisRuntimePlan:
    """Turn one resolved axis plan into configured behaviour, phase by phase."""
    return AxisRuntimePlan(
        keys=resolved.source.keys,
        key_phase=_runtime_phase(resolved.key_phase),
        output_phase=_runtime_phase(resolved.output_phase),
        outputs=resolved.outputs,
    )


def _runtime_phase(phase: AxisMaterializationConfig) -> AxisPhaseRuntimePlan:
    return AxisPhaseRuntimePlan(
        selections=tuple(
            SelectedAxisColumn(
                name=selection.name,
                source=selection.source,
                coercer=axis_coercer_for(selection.logical_type),
            )
            for selection in phase.selections
        ),
        computers=tuple(make_column_computer(config) for config in phase.computers),
    )


# ------------------------------------------------------------------- the fixed compilation


@dataclass(frozen=True, slots=True)
class ParseRuleCompiler:
    """One level's compilation: bind, resolve once, construct, inject."""

    facade: ParseRuleFacade
    output: OutputDeclaration

    def compile(self, source: InputSource) -> Parser:
        """Build one fully initialized parser for this level and this source."""
        working = self.facade.working_parameters
        bound = bind_source(source, working.input)
        evidence = source_evidence(source, bound, header_predicate(working))
        resolved = self.facade.resolve_source(evidence)
        return Parser(
            level=resolved.level,
            input_reader=make_reader(bound, evidence, resolved.read),
            decomposer=make_source_decomposer(
                resolved.decomposition, resolved.obs.source, resolved.var.source
            ),
            obs_plan=make_axis_runtime_plan(resolved.obs),
            var_plan=make_axis_runtime_plan(resolved.var),
            modification_normalizers=tuple(
                make_modification_normalizer(config) for config in resolved.modifications
            ),
            duplicates=policy_for(resolved.duplicate_mode),
            raw_value_presence={
                config.layer_name: make_raw_value_presence(config)
                for config in resolved.raw_value_presence
            },
            writer=make_parsed_level_writer(self.output, resolved),
            # The plan is provenance too: what the rule permitted is already in
            # ``rule_json``, and this is what this source actually resolved to.
            provenance={**resolved.provenance, PLAN_JSON_KEY: resolved_plan_json(resolved)},
        )


def compile_parsers(
    *,
    document: RuleDocument,
    levels: Iterable[QuantificationLevel],
    parameter_evidence: SearchParameterEvidence,
    source: InputSource,
    output: OutputDeclaration,
) -> list[Parser]:
    """One fully initialized parser per requested level this source can satisfy.

    Canonical level order, whatever order the caller asked in. A level the evidence excludes
    or the source cannot satisfy is skipped without affecting the others; a request that
    satisfies nothing was the wrong source, and says so.
    """
    asked = set(levels)
    # Annotated because a generator widens a literal; the two packages spell this level
    # vocabulary separately and the values are the same.
    requested: tuple[QuantificationLevel, ...] = tuple(
        candidate for candidate in LEVELS if candidate in asked
    )
    parsers: list[Parser] = []
    skipped: dict[str, str] = {}
    for level in requested:
        try:
            facade = ParseRuleFacade(document, level, parameter_evidence)
            parsers.append(ParseRuleCompiler(facade=facade, output=output).compile(source))
        except (RuleNotApplicable, IncompatibleSourceError) as reason:
            skipped[level] = str(reason)
    if not parsers:
        raise NoCompatibleLevelError(
            f"no requested level of {document.path} is satisfied by {source!r}: {skipped}"
        )
    return parsers


def header_predicate(
    working: WorkingParseConfiguration,
) -> delimited_input.HeaderPredicate:
    """Whether one inspected header can satisfy this level, for dialect resolution.

    Only what the level cannot do without: its required selections, its required layer
    sources, the columns its modifications read, and the packed columns a separator splits.
    A wide layer source is a pattern, so it is asked whether anything matches.
    """
    exact = frozenset(
        {
            *(
                selection.source
                for axis in (working.obs, working.var)
                for selection in axis.columns.required_selections
            ),
            *_modification_sources(working),
            *_packed_sources(working),
        }
    )
    required_layers = tuple(layer.source for layer in working.measurements.required_layers)
    wide = isinstance(working.source_layout, WideSourceLayout)

    def accepts(header: tuple[str, ...]) -> bool:
        present = frozenset(header)
        if not exact <= present:
            return False
        if not wide:
            return all(source in present for source in required_layers)
        return all(
            any(re.compile(pattern).match(name) for name in header) for pattern in required_layers
        )

    return accepts


def _modification_sources(working: WorkingParseConfiguration) -> tuple[str, ...]:
    return tuple(
        column
        for config in working.modifications
        for column in (
            (config.sequence_column, config.modification_column, config.site_column)
            if isinstance(config, SiteListModificationConfig)
            else (config.source_column,)
        )
    )


def _packed_sources(working: WorkingParseConfiguration) -> tuple[str, ...]:
    layout = working.source_layout
    if isinstance(layout, ColumnLabeledFragmentLayout):
        return (layout.label_source, *layout.packed_value_sources)
    if isinstance(layout, PositionalFragmentLayout):
        return layout.packed_value_sources
    return ()


# ------------------------------------------------------------------------- source binding


@dataclass(frozen=True, slots=True)
class BoundTable:
    """One concrete file and the one declared physical interpretation selected for it."""

    path: Path
    format: PhysicalFormatContract


def bind_source(source: InputSource, contract: InputContract) -> BoundTable:
    """Resolve one complete caller-supplied source into one concrete table.

    A source value arrives complete; binding reports what it cannot satisfy rather than
    filling anything in. Raises ``IncompatibleSourceError`` when no declared format accepts
    the path and leaves dialect ambiguity to physical evidence resolution.
    """
    return _bind_path(_path_of(source, contract), contract)


def source_evidence(
    source: InputSource,
    bound: BoundTable,
    accepts: delimited_input.HeaderPredicate,
) -> SourceEvidence:
    """Observe the physical evidence one bound table exposes, before any full read."""
    if isinstance(bound.format, ParquetFormatContract):
        return parquet_input.schema_evidence(bound.path)
    if isinstance(source, DelimitedFile):
        return delimited_input.stated_evidence(source, bound.format, accepts)
    return delimited_input.detected_evidence(bound.path, bound.format, accepts)


def source_recognition_evidence(
    source: InputSource,
    bound: BoundTable,
    accepts: delimited_input.HeaderPredicate,
) -> SourceEvidence:
    """Observe only schema/header evidence for packaged-rule recognition."""
    if isinstance(bound.format, ParquetFormatContract):
        return parquet_input.schema_evidence(bound.path)
    if isinstance(source, DelimitedFile):
        return delimited_input.stated_evidence(source, bound.format, accepts)
    return delimited_input.detected_header_evidence(bound.path, bound.format, accepts)


def make_reader(
    bound: BoundTable,
    evidence: SourceEvidence,
    plan: LevelReadPlan,
) -> delimited_input.DelimitedInputReader | parquet_input.ParquetInputReader:
    """Construct the reader this bound source, its evidence, and one level plan describe."""
    if isinstance(evidence, ParquetSourceEvidence):
        return parquet_input.make_parquet_reader(bound.path, plan)
    return delimited_input.make_delimited_reader(bound.path, evidence, plan)


def _path_of(source: InputSource, contract: InputContract) -> Path:
    """The one file this source names, resolving an exact folder file name when needed."""
    if isinstance(source, SingleFile | DelimitedFile):
        return source.path
    return _folder_path(source, contract)


def _folder_path(source: Folder, contract: InputContract) -> Path:
    if contract.file_name is None:
        raise IncompatibleSourceError(
            f"{source.path} is a folder, but this rule declares no file_name"
        )
    path = source.path / contract.file_name
    if not path.is_file():
        raise IncompatibleSourceError(
            f"{source.path} does not contain the declared file {contract.file_name!r}"
        )
    return path


def _bind_path(path: Path, contract: InputContract) -> BoundTable:
    """Select a declared format, using extensions as hints when several are possible."""
    suffix = path.suffix.lower()
    claiming = [declared for declared in contract.formats if suffix in set(declared.extensions)]
    if not claiming and len(contract.formats) == 1:
        return BoundTable(path=path, format=contract.formats[0])
    if not claiming:
        declared = sorted(extension for entry in contract.formats for extension in entry.extensions)
        raise IncompatibleSourceError(
            f"{path} has extension {suffix!r}, which no declared format accepts; declared: "
            f"{declared}"
        )
    if len(claiming) > 1:
        raise AmbiguousDialectError(
            f"{path} extension {suffix!r} is claimed by several declared formats"
        )
    return BoundTable(path=path, format=claiming[0])
