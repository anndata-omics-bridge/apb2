"""``ParseRuleFacade``: one rule document becomes storage-neutral parsing parameters.

The one module that imports both sibling children, which is why it lives here and not inside
either: it consumes ``vendor_parse_rules.RuleDocument`` and produces ``parse_quant.parameters``
values. Putting it in either sibling would create the sideways dependency the folder law
forbids.

Two operations, in this order and only this order:

``__init__`` composes one effective rule — level chosen, gate satisfied, primary-layer override
applied — and immediately projects it into ``WorkingParseConfiguration``. The Pydantic
declaration is not retained: after construction the facade holds plain values, so nothing
downstream can reach a storage model through it.

``resolve_source`` binds those values to one observed header, once, and returns one complete
``ResolvedLevelPlan``. Atomic on purpose: the reader, both axes, the decomposer, the separator,
the presence strategies, and the encoders are all derived from the same projected column set,
so optional-source presence, wide sample captures, and packed source order cannot disagree
between plans resolved separately.

One boundary the architecture leaves unassigned: a rule names modifications by Unimod
accession, and the mass, target, and position those accessions denote come from the bundled
registry. Resolving an accession into plain values is declaration projection, so it happens
here — which keeps ``parse_quant`` free of both the registry and Pydantic, and keeps the
normalizers reading nothing but their configuration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnDeclaration,
    AxisColumnSelection,
    AxisKeyPlan,
    AxisMaterializationConfig,
    AxisSourcePlan,
    CoalesceColumnConfig,
    ComputedColumnConfig,
    JoinNonemptyColumnConfig,
    ModificationConfig,
    ModificationMapEntry,
    ProformaFragmentColumnConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    ResolvedAxisColumnPlan,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
    TokenRegexModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    AnnDataLayerContractConfig,
    AnnDataLayerEncodingConfig,
    AnnDataSerializationConfig,
    FactorAnnDataEncodingConfig,
    NullOnlyRawValuePresenceConfig,
    PlainNumericAnnDataEncodingConfig,
    PlainNumericRawValuePresenceConfig,
    RawValuePresenceConfig,
    RegexNumericAnnDataEncodingConfig,
    RegexNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentSeparationConfig,
    DecompositionConfig,
    DelimitedFormatContract,
    DelimitedFragmentDecompositionConfig,
    DelimitedSourceEvidence,
    FragmentSeparationConfig,
    InputContract,
    LevelReadPlan,
    LongDecompositionConfig,
    LongRawLayerSource,
    NumericTextFormat,
    ParquetFormatContract,
    ParquetSourceEvidence,
    PhysicalFormatContract,
    PositionalFragmentSeparationConfig,
    SourceEvidence,
    WideDecompositionConfig,
    WideRawLayerPlan,
    WideRawLayerSource,
)
from apb2.parserV2.parse_quant.parameters.working import (
    AnnDataLayerEncodingDeclaration,
    ColumnLabeledFragmentLayout,
    FactorEncodingDeclaration,
    JsonValue,
    LongSourceLayout,
    NullOnlyRawValuePresenceDeclaration,
    PlainNumericEncodingDeclaration,
    PlainNumericRawValuePresenceDeclaration,
    PositionalFragmentLayout,
    QuantificationLevel,
    RawValuePresenceDeclaration,
    RegexNumericEncodingDeclaration,
    RegexNumericRawValuePresenceDeclaration,
    SourceLayoutDeclaration,
    WideSourceLayout,
    WorkingAxisConfiguration,
    WorkingMeasurementLayer,
    WorkingMeasurements,
    WorkingParseConfiguration,
)
from apb2.parserV2.vendor_params.parsers.shared.unimod import UNIMOD_REGISTRY
from apb2.parserV2.vendor_parse_rules.document import (
    EffectiveRule,
    RuleDocument,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.schema.axis import (
    Coalesce,
    ColumnGroup,
    ComputedColumn,
    JoinNonempty,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
    group_names,
)
from apb2.parserV2.vendor_parse_rules.schema.base_formats import (
    DELIMITED_BASE_FORMATS,
    PARQUET_EXTENSIONS,
    BaseDelimitedFormat,
    DetectedNumberFormat,
    SupportedExtension,
)
from apb2.parserV2.vendor_parse_rules.schema.base_modifications import SiteListModifications
from apb2.parserV2.vendor_parse_rules.schema.fragments import ColumnLabeledFragments
from apb2.parserV2.vendor_parse_rules.schema.input import Input
from apb2.parserV2.vendor_parse_rules.schema.measurements import (
    FactorLayer,
    Layer,
    NoValuePattern,
    NumericLayer,
    RegexValuePattern,
    layer_required,
)
from apb2.parserV2.vendor_parse_rules.schema.rule import LongRule, WideRule

PRODUCER = "apb2"
"""What this package writes as ``uns['apb']['parse']['produced_by']``.

Which tool converted the object, stated by the tool. A reader that wants a rule document
renders it only for a producer whose schema it can validate; sniffing the payload's shape
would be a worse answer than asking who wrote it.
"""

_EMPTY_RATIO = 0.001
_POPULATED_RATIO = 0.5
_STRIPPED_OUTPUT = "stripped_sequence"
_SAMPLE_GROUP = "sample"

type RawSources = Mapping[str, tuple[str, ...]]
"""Which physical columns each declared logical name is ultimately derived from."""


@dataclass(frozen=True, slots=True)
class _ResolvedLayers:
    """The layers one source can actually provide, and how their values are laid out.

    Exactly one of ``long_sources`` and ``wide_plans`` is populated: they are the two shapes
    a decomposition configuration is built from, and the layout decided which before this
    value existed.
    """

    retained: tuple[WorkingMeasurementLayer, ...]
    required_names: tuple[str, ...]
    long_sources: tuple[LongRawLayerSource, ...]
    wide_plans: tuple[WideRawLayerPlan, ...]
    source_columns: frozenset[str]
    plain_numeric_columns: frozenset[str]


class ParseRuleFacade:
    """One simplified interface over rule composition, projection, and source resolution."""

    __slots__ = ("_configuration",)

    def __init__(
        self,
        document: RuleDocument,
        level: QuantificationLevel,
        parameter_evidence: SearchParameterEvidence,
    ) -> None:
        effective = document.rule(level, parameter_evidence)
        self._configuration = self._project_effective_rule(effective)

    @classmethod
    def from_declared_rule(
        cls,
        document: RuleDocument,
        level: QuantificationLevel,
    ) -> ParseRuleFacade:
        """Construct the header-only recognition view, without a gate or override."""
        facade = cls.__new__(cls)
        facade._configuration = cls._project_effective_rule(document.declared(level))
        return facade

    @property
    def working_parameters(self) -> WorkingParseConfiguration:
        return self._configuration

    # -------------------------------------------------------------------------- projection

    @staticmethod
    def _project_effective_rule(effective: EffectiveRule) -> WorkingParseConfiguration:
        """Turn one validated declaration into plain values, retaining no Pydantic model."""
        rule = effective.declaration
        ParseRuleFacade._require_rule_compatibility(rule)
        modifications = ParseRuleFacade._project_modifications(rule)
        obs_group = rule.columns.obs if isinstance(rule, LongRule) else None
        return WorkingParseConfiguration(
            level=rule.quantification_level,
            input=ParseRuleFacade._project_input(effective.input),
            source_layout=ParseRuleFacade._project_layout(rule),
            obs=ParseRuleFacade._project_axis(rule.axis.obs_keys, obs_group, rule),
            var=ParseRuleFacade._project_axis(rule.axis.var_keys, rule.columns.var, rule),
            measurements=ParseRuleFacade._project_measurements(rule),
            modifications=modifications,
            provenance=ParseRuleFacade._project_provenance(rule),
        )

    @staticmethod
    def _project_input(declared: Input) -> InputContract:
        return InputContract(
            file_name=declared.file_name,
            formats=tuple(
                ParseRuleFacade._project_format(extension, declared)
                for extension in dict.fromkeys(declared.extensions)
            ),
        )

    @staticmethod
    def _project_format(
        extension: SupportedExtension,
        declared: Input,
    ) -> PhysicalFormatContract:
        """Apply one shared extension default and this document's explicit exception."""
        if extension in PARQUET_EXTENSIONS:
            if declared.delimiter is not None or declared.numbers is not None:
                raise ValueError("Parquet input cannot declare text-format detection")
            return ParquetFormatContract(extensions=(extension,))
        base = DELIMITED_BASE_FORMATS[extension]
        delimiters = (
            (base.delimiter,)
            if declared.delimiter is None
            else tuple(dict.fromkeys(declared.delimiter.candidates))
        )
        return DelimitedFormatContract(
            extensions=(extension,),
            encoding=base.encoding,
            quote_char=base.quote_char,
            delimiter_candidates=delimiters,
            number_format_candidates=ParseRuleFacade._project_number_formats(
                declared.numbers,
                base,
            ),
        )

    @staticmethod
    def _project_number_formats(
        declared: DetectedNumberFormat | None,
        base: BaseDelimitedFormat,
    ) -> tuple[NumericTextFormat, ...]:
        """Use one base notation unless this document explicitly enables detection."""
        if declared is None:
            return (
                NumericTextFormat(
                    decimal_mark=base.decimal_mark,
                    thousands_marks=base.thousands_marks,
                ),
            )
        return tuple(
            NumericTextFormat(
                decimal_mark=decimal,
                thousands_marks=tuple(
                    dict.fromkeys(mark for mark in declared.thousands_candidates if mark != decimal)
                ),
            )
            for decimal in dict.fromkeys(declared.decimal_candidates)
        )

    @staticmethod
    def _require_rule_compatibility(rule: LongRule | WideRule) -> None:
        """Reject declarations for which no configured runtime strategy can be built."""
        if rule.measurements.duplicates.mode == "aggregate":
            offenders = sorted(
                layer.name
                for layer in rule.measurements.layers
                if not isinstance(layer, NumericLayer)
                or layer.missing_values
                or not isinstance(layer.value_pattern, NoValuePattern)
            )
            if offenders:
                raise ValueError(
                    "aggregate duplicates require plain numeric layers without late decoding; "
                    f"offending layers: {offenders}"
                )
        fragments = rule.fragments
        if fragments is None:
            return
        physical = {layer.source for layer in rule.measurements.layers}
        physical.update(fragments.value_columns)
        groups = (
            (rule.columns.obs, rule.columns.var)
            if isinstance(rule, LongRule)
            else (rule.columns.var,)
        )
        for group in groups:
            physical.update(group.select.values())
            physical.update(group.optional_select.values())
        if isinstance(fragments, ColumnLabeledFragments):
            physical.add(fragments.label_column)
            selected_label = [
                name
                for group in groups
                for name, source in {**group.select, **group.optional_select}.items()
                if source == fragments.label_column
            ]
            if selected_label:
                raise ValueError(
                    "columns must not select the packed fragment label source; "
                    f"offending columns: {selected_label}"
                )
        if fragments.label_output in physical:
            raise ValueError(
                f"fragments.label_output={fragments.label_output!r} collides with a physical "
                "source column"
            )

    @staticmethod
    def _project_layout(rule: LongRule | WideRule) -> SourceLayoutDeclaration:
        fragments = rule.fragments
        if fragments is None:
            return (
                LongSourceLayout(kind="long")
                if isinstance(rule, LongRule)
                else WideSourceLayout(kind="wide")
            )
        if isinstance(fragments, ColumnLabeledFragments):
            return ColumnLabeledFragmentLayout(
                kind="column_labeled_fragment",
                label_source=fragments.label_column,
                delimiter=fragments.delimiter,
                label_output=fragments.label_output,
                packed_value_sources=tuple(fragments.value_columns),
            )
        return PositionalFragmentLayout(
            kind="positional_fragment",
            delimiter=fragments.delimiter,
            label_output=fragments.label_output,
            packed_value_sources=tuple(fragments.value_columns),
        )

    @staticmethod
    def _project_axis(
        keys: Sequence[str],
        group: ColumnGroup | None,
        rule: LongRule | WideRule,
    ) -> WorkingAxisConfiguration:
        """Project one axis. A wide rule has no obs group: its keys are header captures."""
        if group is None:
            return WorkingAxisConfiguration(
                final_key_columns=tuple(keys),
                columns=AxisColumnDeclaration(
                    required_selections=(),
                    optional_selections=(),
                    computed=(),
                    declared_order=tuple(keys),
                ),
            )
        return WorkingAxisConfiguration(
            final_key_columns=tuple(keys),
            columns=AxisColumnDeclaration(
                required_selections=ParseRuleFacade._project_selections(group, group.select),
                optional_selections=ParseRuleFacade._project_selections(
                    group, group.optional_select
                ),
                computed=tuple(
                    ParseRuleFacade._project_computed(column, rule) for column in group.computed
                ),
                declared_order=tuple(group_names(group)),
            ),
        )

    @staticmethod
    def _project_selections(
        group: ColumnGroup, declared: Mapping[str, str]
    ) -> tuple[AxisColumnSelection, ...]:
        return tuple(
            AxisColumnSelection(
                name=name,
                source=source,
                logical_type=group.types.get(name, "string"),
            )
            for name, source in declared.items()
        )

    @staticmethod
    def _project_computed(
        column: ComputedColumn, rule: LongRule | WideRule
    ) -> ComputedColumnConfig:
        """Name the series each computation actually reads, not the column it documents.

        ``stripped_sequence`` and ``proforma_sequence`` expose what the modification
        normalizer already derived, so their input is that derived column; the authored
        ``inputs`` names the selected sequence the normalizer reads, which is an
        authored-consistency check the schema already made.
        """
        if isinstance(column, Coalesce):
            return CoalesceColumnConfig(
                kind="coalesce", name=column.name, inputs=tuple(column.inputs)
            )
        if isinstance(column, JoinNonempty):
            return JoinNonemptyColumnConfig(
                kind="join_nonempty",
                name=column.name,
                inputs=tuple(column.inputs),
                separator=column.separator,
            )
        if isinstance(column, StrippedSequence):
            return StrippedSequenceColumnConfig(
                kind="stripped_sequence", name=column.name, inputs=(_STRIPPED_OUTPUT,)
            )
        if isinstance(column, ProformaSequence):
            declared = rule.modifications
            if declared is None:
                raise ValueError("how='proforma_sequence' requires a [modifications] block")
            return ProformaSequenceColumnConfig(
                kind="proforma_sequence", name=column.name, inputs=(declared.output_column,)
            )
        if isinstance(column, ProformaIon):
            return ProformaIonColumnConfig(
                kind="proforma_ion", name=column.name, inputs=tuple(column.inputs)
            )
        return ProformaFragmentColumnConfig(
            kind="proforma_fragment", name=column.name, inputs=tuple(column.inputs)
        )

    @staticmethod
    def _project_measurements(rule: LongRule | WideRule) -> WorkingMeasurements:
        """Promote the primary layer into the required set, preserving authored order."""
        projected = tuple(
            ParseRuleFacade._project_layer(layer) for layer in rule.measurements.layers
        )
        required = frozenset(
            layer.name
            for layer in rule.measurements.layers
            if layer_required(rule.measurements.primary_layer, layer)
        )
        return WorkingMeasurements(
            primary_layer_name=rule.measurements.primary_layer,
            duplicate_mode=rule.measurements.duplicates.mode,
            required_layers=tuple(layer for layer in projected if layer.name in required),
            optional_layers=tuple(layer for layer in projected if layer.name not in required),
            authored_order=tuple(layer.name for layer in projected),
        )

    @staticmethod
    def _project_layer(layer: Layer) -> WorkingMeasurementLayer:
        return WorkingMeasurementLayer(
            name=layer.name,
            source=layer.source,
            raw_presence=ParseRuleFacade._project_presence(layer),
            ann_data_encoding=ParseRuleFacade._project_encoding(layer),
        )

    @staticmethod
    def _project_presence(layer: Layer) -> RawValuePresenceDeclaration:
        """What makes a raw scalar of this layer claim its cell, before any conversion."""
        if isinstance(layer, FactorLayer):
            return NullOnlyRawValuePresenceDeclaration(kind="null_only")
        if isinstance(layer.value_pattern, RegexValuePattern):
            return RegexNumericRawValuePresenceDeclaration(
                kind="regex_numeric",
                missing_values=tuple(layer.missing_values),
                pattern=layer.value_pattern.pattern,
            )
        if layer.missing_values:
            return PlainNumericRawValuePresenceDeclaration(
                kind="plain_numeric", missing_values=tuple(layer.missing_values)
            )
        return NullOnlyRawValuePresenceDeclaration(kind="null_only")

    @staticmethod
    def _project_encoding(layer: Layer) -> AnnDataLayerEncodingDeclaration:
        if isinstance(layer, FactorLayer):
            return FactorEncodingDeclaration(
                kind="factor", categories=tuple(layer.categories.items())
            )
        if isinstance(layer.value_pattern, RegexValuePattern):
            return RegexNumericEncodingDeclaration(
                kind="regex_numeric",
                missing_values=tuple(layer.missing_values),
                pattern=layer.value_pattern.pattern,
            )
        return PlainNumericEncodingDeclaration(
            kind="plain_numeric", missing_values=tuple(layer.missing_values)
        )

    @staticmethod
    def _project_modifications(rule: LongRule | WideRule) -> tuple[ModificationConfig, ...]:
        """Project the modifications block, resolving its accessions, when anything reads it."""
        declared = rule.modifications
        consumed = any(
            isinstance(column, ProformaSequence | StrippedSequence)
            for column in rule.columns.var.computed
        )
        if declared is None or not consumed:
            return ()
        entries = tuple(
            ModificationMapEntry(
                token=entry.token,
                name=record.name,
                accession=record.accession,
                target=tuple(record.target),
                position=record.position,
                mass_delta=record.mass_delta,
            )
            for entry, record in (
                (entry, UNIMOD_REGISTRY.resolve(entry.accession)) for entry in declared.map
            )
        )
        if isinstance(declared, SiteListModifications):
            return (
                SiteListModificationConfig(
                    kind="site_list",
                    sequence_column=declared.sequence_column,
                    modification_column=declared.modification_column,
                    site_column=declared.site_column,
                    delimiter=declared.delimiter,
                    site_base=declared.site_base,
                    case_sensitive=declared.case_sensitive,
                    unknown_policy=declared.unknown_policy,
                    proforma_output=declared.output_column,
                    stripped_output=_STRIPPED_OUTPUT,
                    entries=entries,
                ),
            )
        return (
            TokenRegexModificationConfig(
                kind="token_regex",
                source_column=declared.source_column,
                token_pattern=declared.token_pattern,
                token_position=declared.token_position,
                case_sensitive=declared.case_sensitive,
                unknown_policy=declared.unknown_policy,
                proforma_output=declared.output_column,
                stripped_output=_STRIPPED_OUTPUT,
                entries=entries,
            ),
        )

    @staticmethod
    def _project_provenance(rule: LongRule | WideRule) -> Mapping[str, JsonValue]:
        """What the parse section records: who wrote it, the rule, and the facts steps read.

        ``produced_by`` and ``column_roles`` are not decoration. A later APB step needs the
        quantification level and one ``var`` column per semantic role, and must not have to
        validate a schema-0.3 document to learn a column name — a reader of another generation
        cannot. Stating both as data is what lets ``apb fasta`` and ``apb proteobench`` run on
        an object this parser wrote.
        """
        provenance: dict[str, JsonValue] = {
            "produced_by": PRODUCER,
            "rule_json": json.dumps(rule.model_dump(mode="json")),
            "column_roles": {
                name: column
                for name in ("protein_assignment", "fasta_accessions")
                if (column := getattr(rule.column_roles, name)) is not None
            },
            "schema_version": rule.schema_version,
            "software_name": rule.software_name,
            "shape": rule.shape,
            "quantification_level": rule.quantification_level,
        }
        if rule.sample_annotation is not None:
            provenance["sample_annotation_matching"] = rule.sample_annotation.matching.model_dump(
                mode="json"
            )
        return provenance

    # -------------------------------------------------------------------- source resolution

    def resolve_source(self, evidence: SourceEvidence) -> ResolvedLevelPlan:
        """Bind the working parameters to one observed source, once and completely.

        Raises ``IncompatibleSourceError`` when this source cannot satisfy the level: a
        required column, layer, or final key that its header does not provide.
        """
        working = self._configuration
        present = frozenset(evidence.columns)
        modifications = self._retained_modifications(present)
        derived = self._derived_sources(modifications)
        var = self._resolve_axis(working.var, present, derived, self._var_synthesized())
        layers = self._resolve_layers(evidence.columns, present, self._accounted(var, derived))
        obs = self._resolve_axis(working.obs, present, {}, self._obs_synthesized())
        read = self._read_plan(evidence, obs, var, layers, derived)
        self._require_aggregatable(evidence, read, layers)
        numbers = _resolved_numbers(evidence)
        return ResolvedLevelPlan(
            level=working.level,
            number_format=numbers,
            read=read,
            decomposition=self._decomposition(layers),
            obs=obs,
            var=var,
            modifications=modifications,
            duplicate_mode=working.measurements.duplicate_mode,
            raw_value_presence=tuple(_presence_config(layer, numbers) for layer in layers.retained),
            ann_data=AnnDataSerializationConfig(
                layer_encodings=tuple(
                    _encoding_config(layer, numbers) for layer in layers.retained
                ),
                layer_contract=AnnDataLayerContractConfig(
                    primary_layer_name=working.measurements.primary_layer_name,
                    required_names=layers.required_names,
                    empty_ratio=_EMPTY_RATIO,
                    populated_ratio=_POPULATED_RATIO,
                ),
            ),
            provenance=working.provenance,
        )

    # ------------------------------------------------------------------------ modifications

    def _retained_modifications(self, present: frozenset[str]) -> tuple[ModificationConfig, ...]:
        """Every projected modification block, provided this source carries its sources."""
        for config in self._configuration.modifications:
            missing = [column for column in _modification_sources(config) if column not in present]
            if missing:
                raise IncompatibleSourceError(
                    f"{self._label()} reads modification source column(s) {missing} that this "
                    "source does not carry"
                )
        return self._configuration.modifications

    @staticmethod
    def _derived_sources(modifications: tuple[ModificationConfig, ...]) -> RawSources:
        """Which physical columns each modification-derived column is derived from."""
        derived: dict[str, tuple[str, ...]] = {}
        for config in modifications:
            sources = _modification_sources(config)
            for output in (config.proforma_output, config.stripped_output):
                derived[output] = sources
        return derived

    def _var_synthesized(self) -> tuple[str, ...]:
        """Names the fragment separator creates as real columns of the scalar-long table."""
        layout = self._configuration.source_layout
        if isinstance(layout, PositionalFragmentLayout | ColumnLabeledFragmentLayout):
            return (layout.label_output,)
        return ()

    def _obs_synthesized(self) -> tuple[str, ...]:
        """A wide observation axis comes from header captures, not from selected columns."""
        if isinstance(self._configuration.source_layout, WideSourceLayout):
            return self._configuration.obs.final_key_columns
        return ()

    def _accounted(self, var: ResolvedAxisColumnPlan, derived: RawSources) -> frozenset[str]:
        """Names a permissive wide layer pattern must not mistake for a sample column."""
        declaration = self._configuration.var.columns
        return frozenset(
            {
                *declaration.declared_order,
                *(selection.source for selection in declaration.required_selections),
                *(selection.source for selection in declaration.optional_selections),
                *var.source.keys.raw_key_columns,
                *var.source.payload_sources,
                *derived,
                *(column for columns in derived.values() for column in columns),
            }
        )

    # ---------------------------------------------------------------------------- axis plans

    def _resolve_axis(
        self,
        axis: WorkingAxisConfiguration,
        present: frozenset[str],
        derived: RawSources,
        synthesized: tuple[str, ...],
    ) -> ResolvedAxisColumnPlan:
        """Resolve one axis: prune what this source cannot provide, then plan both phases."""
        declaration = axis.columns
        missing = [
            selection.source
            for selection in declaration.required_selections
            if selection.source not in present
        ]
        if missing:
            raise IncompatibleSourceError(
                f"{self._label()} selects column(s) {missing} that this source does not carry"
            )
        selections = (
            *declaration.required_selections,
            *(
                selection
                for selection in declaration.optional_selections
                if selection.source in present
            ),
        )
        skipped = {
            selection.name
            for selection in declaration.optional_selections
            if selection.source not in present
        }
        raw_sources, computers = self._materializable(
            declaration.computed, selections, derived, synthesized, skipped
        )
        keys = self._axis_key_plan(axis.final_key_columns, computers, raw_sources, skipped)
        closure = _identity_closure(axis.final_key_columns, computers)
        self._require_output_phase_keeps_identity(axis.final_key_columns, closure, selections)
        return ResolvedAxisColumnPlan(
            source=AxisSourcePlan(
                keys=keys,
                payload_sources=_ordered_unique(
                    column
                    for column in (
                        *(selection.source for selection in selections),
                        *(column for columns in derived.values() for column in columns),
                    )
                    if column not in set(keys.raw_key_columns)
                ),
            ),
            key_phase=_phase(selections, computers, closure, inside=True),
            output_phase=_phase(selections, computers, closure, inside=False),
            outputs=tuple(name for name in declaration.declared_order if name not in skipped),
            skipped=frozenset(skipped),
        )

    def _require_output_phase_keeps_identity(
        self,
        final_keys: tuple[str, ...],
        closure: frozenset[str],
        selections: tuple[AxisColumnSelection, ...],
    ) -> None:
        """Metadata is materialized after the collision check, so it may not rewrite a key.

        A selection or computation naming a final key must be inside the identity closure;
        outside it, it would run after the check that just proved the keys are distinct.
        """
        offenders = sorted(
            selection.name
            for selection in selections
            if selection.name in set(final_keys) and selection.name not in closure
        )
        if offenders:
            raise ValueError(
                f"{self._label()} materializes the axis key(s) {offenders} outside the "
                "identity closure, which would overwrite them after validation"
            )

    def _materializable(
        self,
        declared: tuple[ComputedColumnConfig, ...],
        selections: tuple[AxisColumnSelection, ...],
        derived: RawSources,
        synthesized: tuple[str, ...],
        skipped: set[str],
    ) -> tuple[RawSources, tuple[ComputedColumnConfig, ...]]:
        """Walk the declarations in order, binding each name to its physical closure.

        Reading the environment before rebinding is what lets a computed column consume a
        selected column of its own name — the coalesce that widens ``Proteins`` reads the
        selected ``Proteins`` and then becomes it — without the walk chasing its own tail.
        """
        raw_sources: dict[str, tuple[str, ...]] = {name: (name,) for name in synthesized}
        raw_sources.update(derived)
        for selection in selections:
            raw_sources[selection.name] = (selection.source,)
        retained: list[ComputedColumnConfig] = []
        for computer in declared:
            resolved = _prune_inputs(computer, raw_sources)
            if resolved is None:
                skipped.add(computer.name)
                continue
            retained.append(resolved)
            raw_sources[resolved.name] = _ordered_unique(
                column for name in resolved.inputs for column in raw_sources[name]
            )
        return raw_sources, tuple(retained)

    def _axis_key_plan(
        self,
        final_keys: tuple[str, ...],
        computers: tuple[ComputedColumnConfig, ...],
        raw_sources: RawSources,
        skipped: set[str],
    ) -> AxisKeyPlan:
        """Derive the three identity column sets from the authored keys alone.

        Generic by construction: the walk asks each key how it is materialized and follows
        that answer, so no vendor or level appears anywhere in it.
        """
        by_name = {computer.name: computer for computer in computers}
        inputs: list[str] = []
        raw: list[str] = []
        for key in final_keys:
            if key in skipped or key not in raw_sources:
                raise IncompatibleSourceError(
                    f"{self._label()} cannot materialize the axis key {key!r} from this source"
                )
            computer = by_name.get(key)
            inputs.extend(computer.inputs if computer is not None else (key,))
            raw.extend(raw_sources[key])
        return AxisKeyPlan(
            raw_key_columns=_ordered_unique(raw),
            key_input_columns=_ordered_unique(inputs),
            final_key_columns=final_keys,
        )

    # -------------------------------------------------------------------------------- layers

    def _resolve_layers(
        self,
        columns: tuple[str, ...],
        present: frozenset[str],
        accounted: frozenset[str],
    ) -> _ResolvedLayers:
        """Resolve every declared measurement against this header, by physical layout."""
        layout = self._configuration.source_layout
        if isinstance(layout, WideSourceLayout):
            return self._resolve_wide_layers(columns, accounted)
        return self._resolve_long_layers(present)

    def _resolve_long_layers(self, present: frozenset[str]) -> _ResolvedLayers:
        """Every long layer names one exact column; a packed one is split before reading it."""
        measurements = self._configuration.measurements
        required = {layer.name for layer in measurements.required_layers}
        missing = [
            layer.source for layer in measurements.required_layers if layer.source not in present
        ]
        if missing:
            raise IncompatibleSourceError(
                f"{self._label()} requires layer source column(s) {missing} that this source "
                "does not carry"
            )
        retained = tuple(layer for layer in self._authored_layers() if layer.source in present)
        return _ResolvedLayers(
            retained=retained,
            required_names=tuple(layer.name for layer in retained if layer.name in required),
            long_sources=tuple(
                LongRawLayerSource(name=layer.name, source_column=layer.source)
                for layer in retained
            ),
            wide_plans=(),
            source_columns=frozenset(layer.source for layer in retained),
            plain_numeric_columns=frozenset(
                layer.source
                for layer in retained
                if isinstance(layer.ann_data_encoding, PlainNumericEncodingDeclaration)
            ),
        )

    def _resolve_wide_layers(
        self, columns: tuple[str, ...], accounted: frozenset[str]
    ) -> _ResolvedLayers:
        """Expand each layer's header regex, then align every layer to the primary samples.

        The primary layer defines the observation axis. A permissive pattern must not turn
        an accounted-for column into an extra sample, and a layer that matched only tokens
        outside that axis is not evidence of more observations.
        """
        measurements = self._configuration.measurements
        candidates = tuple(name for name in columns if name not in accounted)
        matches = {
            layer.name: _match_samples(candidates, layer.source)
            for layer in self._authored_layers()
        }
        primary = measurements.primary_layer_name
        samples = _ordered_unique(sample for _column, sample in matches[primary])
        if not samples:
            raise IncompatibleSourceError(
                f"{self._label()} matched no observation column for its primary layer {primary!r}"
            )
        required = {layer.name for layer in measurements.required_layers}
        retained: list[WorkingMeasurementLayer] = []
        plans: list[WideRawLayerPlan] = []
        for layer in self._authored_layers():
            aligned = tuple(
                WideRawLayerSource(source_column=column, sample=sample)
                for column, sample in matches[layer.name]
                if sample in set(samples)
            )
            if layer.name in required and not matches[layer.name]:
                raise IncompatibleSourceError(
                    f"{self._label()} matched no column for its required layer "
                    f"{layer.name!r} pattern {layer.source!r}"
                )
            if layer.name not in required and not aligned:
                continue
            retained.append(layer)
            plans.append(WideRawLayerPlan(name=layer.name, sources=aligned))
        return _ResolvedLayers(
            retained=tuple(retained),
            required_names=tuple(layer.name for layer in retained if layer.name in required),
            long_sources=(),
            wide_plans=tuple(plans),
            source_columns=frozenset(
                source.source_column for plan in plans for source in plan.sources
            ),
            plain_numeric_columns=frozenset(
                source.source_column
                for layer, plan in zip(retained, plans, strict=True)
                if isinstance(layer.ann_data_encoding, PlainNumericEncodingDeclaration)
                for source in plan.sources
            ),
        )

    def _authored_layers(self) -> tuple[WorkingMeasurementLayer, ...]:
        """Every declared measurement in the order the document authored it."""
        measurements = self._configuration.measurements
        by_name = {
            layer.name: layer
            for layer in (*measurements.required_layers, *measurements.optional_layers)
        }
        return tuple(by_name[name] for name in measurements.authored_order)

    # -------------------------------------------------------------------- read and structure

    def _read_plan(
        self,
        evidence: SourceEvidence,
        obs: ResolvedAxisColumnPlan,
        var: ResolvedAxisColumnPlan,
        layers: _ResolvedLayers,
        derived: RawSources,
    ) -> LevelReadPlan:
        """Project exactly this level's transitive source closure, and decide every dtype.

        A measurement column is read natively only when the rule sums its values, because
        summing needs numbers and the schema already checked that those layers are plain. Every
        other measurement stays text: reading it as a float is an encoding, the storage boundary
        owns encoding, and real exports write ``-``, ``NA``, or ``False`` in a column a rule
        calls numeric — which an eager numeric read cannot survive.
        """
        lexical = frozenset(
            {
                *obs.source.keys.raw_key_columns,
                *obs.source.payload_sources,
                *var.source.keys.raw_key_columns,
                *var.source.payload_sources,
                *(column for columns in derived.values() for column in columns),
                *self._packed_sources(),
            }
        )
        needed = lexical | layers.source_columns
        projected = tuple(name for name in evidence.columns if name in needed)
        if isinstance(evidence, ParquetSourceEvidence):
            # Parquet carries its own schema; overriding it would discard physical types.
            return LevelReadPlan(
                projected_columns=projected,
                text_sources=frozenset(),
                native_numeric_sources=frozenset(),
            )
        native = (
            frozenset()
            if evidence.number_format.thousands_marks
            or self._configuration.measurements.duplicate_mode != "aggregate"
            else frozenset(
                column
                for column in projected
                if column in layers.plain_numeric_columns and column not in lexical
            )
        )
        return LevelReadPlan(
            projected_columns=projected,
            text_sources=frozenset(projected) - native,
            native_numeric_sources=native,
        )

    def _packed_sources(self) -> tuple[str, ...]:
        """The physical columns a fragment separator splits, plus its packed label column."""
        layout = self._configuration.source_layout
        if isinstance(layout, ColumnLabeledFragmentLayout):
            return (layout.label_source, *layout.packed_value_sources)
        if isinstance(layout, PositionalFragmentLayout):
            return layout.packed_value_sources
        return ()

    def _decomposition(self, layers: _ResolvedLayers) -> DecompositionConfig:
        """Name the one physical shape this level's table has."""
        layout = self._configuration.source_layout
        primary = self._configuration.measurements.primary_layer_name
        if isinstance(layout, WideSourceLayout):
            return WideDecompositionConfig(
                kind="wide", primary_layer_name=primary, layer_plans=layers.wide_plans
            )
        long = LongDecompositionConfig(
            kind="long", primary_layer_name=primary, layer_sources=layers.long_sources
        )
        if isinstance(layout, LongSourceLayout):
            return long
        return DelimitedFragmentDecompositionConfig(
            kind="delimited_fragment",
            separator=self._separator(layout, layers),
            long=long,
        )

    def _separator(
        self,
        layout: PositionalFragmentLayout | ColumnLabeledFragmentLayout,
        layers: _ResolvedLayers,
    ) -> FragmentSeparationConfig:
        """Keep the retained packed sources in authored order; drop what is absent."""
        packed = tuple(
            column for column in layout.packed_value_sources if column in layers.source_columns
        )
        if not packed:
            raise IncompatibleSourceError(
                f"{self._label()} carries none of the packed fragment columns "
                f"{list(layout.packed_value_sources)}"
            )
        if isinstance(layout, ColumnLabeledFragmentLayout):
            return ColumnLabeledFragmentSeparationConfig(
                kind="column",
                label_source=layout.label_source,
                label_output=layout.label_output,
                delimiter=layout.delimiter,
                packed_value_sources=packed,
            )
        return PositionalFragmentSeparationConfig(
            kind="positional",
            label_output=layout.label_output,
            delimiter=layout.delimiter,
            packed_value_sources=packed,
        )

    def _require_aggregatable(
        self, evidence: SourceEvidence, read: LevelReadPlan, layers: _ResolvedLayers
    ) -> None:
        """Reject an aggregate rule whose values this source cannot deliver as numbers.

        Checked here rather than at runtime because it is a property of the rule and the
        source together, and the alternative is discovering it after reading a large table.
        """
        if self._configuration.measurements.duplicate_mode != "aggregate":
            return
        if isinstance(evidence, ParquetSourceEvidence):
            numeric = frozenset(name for name, dtype in evidence.dtypes if dtype.is_numeric())
            offenders = sorted(layers.source_columns - numeric)
        else:
            offenders = sorted(layers.source_columns - read.native_numeric_sources)
        if offenders:
            raise IncompatibleSourceError(
                f"{self._label()} aggregates duplicate cells, which requires native numeric "
                f"layer values; these resolve to text: {offenders}"
            )

    def _label(self) -> str:
        """How this level names itself in an error message."""
        working = self._configuration
        software = working.provenance.get("software_name", "rule")
        return f"{software!r} level {working.level!r}"


# ------------------------------------------------------------------ shared plain operations


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _modification_sources(config: ModificationConfig) -> tuple[str, ...]:
    if isinstance(config, SiteListModificationConfig):
        return (config.sequence_column, config.modification_column, config.site_column)
    return (config.source_column,)


def _match_samples(candidates: tuple[str, ...], pattern: str) -> tuple[tuple[str, str], ...]:
    """Every candidate column this layer pattern claims, with the sample it captured."""
    compiled = re.compile(pattern)
    matched: list[tuple[str, str]] = []
    for name in candidates:
        match = compiled.match(name)
        if match is not None:
            matched.append((name, match.group(_SAMPLE_GROUP)))
    return tuple(matched)


def _prune_inputs(
    computer: ComputedColumnConfig, available: RawSources
) -> ComputedColumnConfig | None:
    """Drop the inputs this source cannot provide, or report the computation as blocked.

    Only the two combining operations survive a missing input: coalescing or joining the
    columns that are present is the operation the rule asked for. Everything else needs
    every input it declared.
    """
    if isinstance(computer, CoalesceColumnConfig | JoinNonemptyColumnConfig):
        kept = tuple(name for name in computer.inputs if name in available)
        if not kept:
            return None
        if isinstance(computer, CoalesceColumnConfig):
            return CoalesceColumnConfig(kind="coalesce", name=computer.name, inputs=kept)
        return JoinNonemptyColumnConfig(
            kind="join_nonempty",
            name=computer.name,
            inputs=kept,
            separator=computer.separator,
        )
    if any(name not in available for name in computer.inputs):
        return None
    return computer


def _identity_closure(
    final_keys: tuple[str, ...], computers: tuple[ComputedColumnConfig, ...]
) -> frozenset[str]:
    """Every declared name that must be materialized before identity can be checked."""
    by_name = {computer.name: computer for computer in computers}
    closure = set(final_keys)
    pending = list(final_keys)
    while pending:
        computer = by_name.get(pending.pop())
        if computer is None:
            continue
        for name in computer.inputs:
            if name not in closure:
                closure.add(name)
                pending.append(name)
    return frozenset(closure)


def _phase(
    selections: tuple[AxisColumnSelection, ...],
    computers: tuple[ComputedColumnConfig, ...],
    closure: frozenset[str],
    *,
    inside: bool,
) -> AxisMaterializationConfig:
    """Split the declarations at the identity closure, keeping declaration order in each."""
    return AxisMaterializationConfig(
        selections=tuple(
            selection for selection in selections if (selection.name in closure) is inside
        ),
        computers=tuple(computer for computer in computers if (computer.name in closure) is inside),
    )


def _resolved_numbers(evidence: SourceEvidence) -> NumericTextFormat:
    """The notation a retained token must be read with; Parquet values are already numbers."""
    if isinstance(evidence, DelimitedSourceEvidence):
        return evidence.number_format
    return NumericTextFormat(decimal_mark=".", thousands_marks=())


def _presence_config(
    layer: WorkingMeasurementLayer, numbers: NumericTextFormat
) -> RawValuePresenceConfig:
    """Which raw scalars of this layer claim a cell, with its layer name attached."""
    declaration = layer.raw_presence
    if isinstance(declaration, PlainNumericRawValuePresenceDeclaration):
        return PlainNumericRawValuePresenceConfig(
            kind="plain_numeric",
            layer_name=layer.name,
            missing_values=declaration.missing_values,
            number_format=numbers,
        )
    if isinstance(declaration, RegexNumericRawValuePresenceDeclaration):
        return RegexNumericRawValuePresenceConfig(
            kind="regex_numeric",
            layer_name=layer.name,
            missing_values=declaration.missing_values,
            pattern=declaration.pattern,
            number_format=numbers,
        )
    return NullOnlyRawValuePresenceConfig(kind="null_only", layer_name=layer.name)


def _encoding_config(
    layer: WorkingMeasurementLayer, numbers: NumericTextFormat
) -> AnnDataLayerEncodingConfig:
    """How this layer's raw scalars become a dense float matrix, at the AnnData boundary."""
    declaration = layer.ann_data_encoding
    if isinstance(declaration, FactorEncodingDeclaration):
        return FactorAnnDataEncodingConfig(
            kind="factor", layer_name=layer.name, categories=declaration.categories
        )
    if isinstance(declaration, RegexNumericEncodingDeclaration):
        return RegexNumericAnnDataEncodingConfig(
            kind="regex_numeric",
            layer_name=layer.name,
            missing_values=declaration.missing_values,
            pattern=declaration.pattern,
            number_format=numbers,
        )
    return PlainNumericAnnDataEncodingConfig(
        kind="plain_numeric",
        layer_name=layer.name,
        missing_values=declaration.missing_values,
        number_format=numbers,
    )
