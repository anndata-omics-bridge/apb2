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
executable ``ParseStrategy``. Atomic on purpose: the reader, both axes, the decomposer, the separator,
the presence strategies and value parsers are all derived from the same projected column set,
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
from collections.abc import Mapping, Sequence
from typing import Literal

from apb2.parserV2.parse_quant.axis_columns import (
    CoalesceColumn,
    JoinNonemptyColumn,
    ProformaFragmentColumn,
    ProformaIonColumn,
)
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    PlainSequenceStripper,
    SequenceColumn,
    SequenceOperation,
    SiteListNormalizer,
    TokenRegexNormalizer,
    TokenRegexStripper,
)
from apb2.parserV2.parse_quant.operations import (
    ComputedOperation,
    WorkingAxisConfiguration,
    WorkingParseConfiguration,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnSelection,
    EmbeddedSiteListModificationConfig,
    ModificationMapEntry,
    SiteListModificationConfig,
    TokenRegexModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.level import (
    JsonValue,
    QuantificationLevel,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    FactorLayerDeclaration,
    LayerValueDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
    WorkingMeasurementLayer,
    WorkingMeasurements,
)
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentLayout,
    DelimitedFormatContract,
    ExcelFormatContract,
    InputContract,
    LongSourceLayout,
    NumericTextFormat,
    ParquetFormatContract,
    PhysicalFormatContract,
    PositionalFragmentLayout,
    SourceEvidence,
    SourceLayoutDeclaration,
    WideSourceLayout,
)
from apb2.parserV2.parse_quant.parser import ParseStrategy
from apb2.parserV2.parse_quant.source_resolution import SourcePlanResolver
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
    computed_columns,
)
from apb2.parserV2.vendor_parse_rules.schema.base_formats import (
    DELIMITED_BASE_FORMATS,
    PARQUET_EXTENSIONS,
    WORKBOOK_EXTENSIONS,
    BaseDelimitedFormat,
    DetectedNumberFormat,
    SupportedExtension,
)
from apb2.parserV2.vendor_parse_rules.schema.base_modifications import (
    EmbeddedSiteListSyntax,
    PlainSequenceSyntax,
    SiteListSyntax,
    TokenRegexSyntax,
)
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


def _layer_roles(
    layers: Sequence[Layer],
) -> dict[str, JsonValue]:
    roles = sorted({role for layer in layers for role in layer.roles})
    projected: dict[str, JsonValue] = {
        role: [layer.name for layer in layers if role in layer.roles] for role in roles
    }
    return projected


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
        obs_group = rule.columns.obs if isinstance(rule, LongRule) else None
        return WorkingParseConfiguration(
            level=rule.quantification_level,
            input=ParseRuleFacade._project_input(effective.input),
            source_layout=ParseRuleFacade._project_layout(rule),
            obs=ParseRuleFacade._project_axis(rule.axis.obs_keys, obs_group, rule),
            var=ParseRuleFacade._project_axis(rule.axis.var_keys, rule.columns.var, rule),
            measurements=ParseRuleFacade._project_measurements(rule),
            provenance=ParseRuleFacade._project_provenance(rule),
            preparation=effective.preparation,
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
            if (
                declared.delimiter is not None
                or declared.numbers is not None
                or declared.encoding is not None
            ):
                raise ValueError("Parquet input cannot declare text-format detection")
            return ParquetFormatContract(extensions=(extension,))
        if declared.sheet_name is not None:
            if extension not in WORKBOOK_EXTENSIONS:
                raise ValueError(f"workbook sheet input cannot use extension {extension!r}")
            return ExcelFormatContract(extensions=(extension,), sheet_name=declared.sheet_name)
        base = DELIMITED_BASE_FORMATS[extension]
        delimiters = (
            (base.delimiter,)
            if declared.delimiter is None
            else tuple(dict.fromkeys(declared.delimiter.candidates))
        )
        encodings = (
            (base.encoding,)
            if declared.encoding is None
            else tuple(dict.fromkeys(declared.encoding.candidates))
        )
        return DelimitedFormatContract(
            extensions=(extension,),
            encoding_candidates=encodings,
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
            physical.update(column.source for column in group if column.source is not None)
        if isinstance(fragments, ColumnLabeledFragments):
            physical.add(fragments.label_column)
            selected_label = [
                column.name
                for group in groups
                for column in group
                if column.source == fragments.label_column
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
            return LongSourceLayout() if isinstance(rule, LongRule) else WideSourceLayout()
        if isinstance(fragments, ColumnLabeledFragments):
            return ColumnLabeledFragmentLayout(
                label_source=fragments.label_column,
                delimiter=fragments.delimiter,
                label_output=fragments.label_output,
                packed_value_sources=tuple(fragments.value_columns),
            )
        return PositionalFragmentLayout(
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
            return WorkingAxisConfiguration(tuple(keys), (), (), (), tuple(keys))
        return WorkingAxisConfiguration(
            final_key_columns=tuple(keys),
            required_selections=ParseRuleFacade._project_selections(group, required=True),
            optional_selections=ParseRuleFacade._project_selections(group, required=False),
            computed=tuple(
                ParseRuleFacade._project_computed(column, rule)
                for column in computed_columns(group)
            ),
            declared_order=tuple(column.name for column in group),
        )

    @staticmethod
    def _project_selections(
        group: ColumnGroup, *, required: bool
    ) -> tuple[AxisColumnSelection, ...]:
        return tuple(
            AxisColumnSelection(
                name=column.name,
                source=column.source,
                logical_type=column.type,
            )
            for column in group
            if column.source is not None and column.required is required
        )

    @staticmethod
    def _project_computed(column: ComputedColumn, rule: LongRule | WideRule) -> ComputedOperation:
        """Preserve the exact authored logical inputs and bind referenced configuration."""
        if isinstance(column, Coalesce):
            return CoalesceColumn(column.name, tuple(column.inputs))
        if isinstance(column, JoinNonempty):
            return JoinNonemptyColumn(column.name, tuple(column.inputs), column.separator)
        if isinstance(column, StrippedSequence):
            return SequenceColumn(
                column.name, tuple(column.inputs), ParseRuleFacade._project_stripping(column, rule)
            )
        if isinstance(column, ProformaSequence):
            return SequenceColumn(
                column.name,
                tuple(column.inputs),
                ParseRuleFacade._project_modifications(column, rule),
            )
        if isinstance(column, ProformaIon):
            return ProformaIonColumn(column.name, tuple(column.inputs))
        return ProformaFragmentColumn(column.name, tuple(column.inputs))

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
            layers=projected,
            required_names=required,
        )

    @staticmethod
    def _project_layer(layer: Layer) -> WorkingMeasurementLayer:
        return WorkingMeasurementLayer(
            name=layer.name,
            source=layer.source,
            value=ParseRuleFacade._project_layer_value(layer),
            roles=tuple(layer.roles),
        )

    @staticmethod
    def _project_layer_value(layer: Layer) -> LayerValueDeclaration:
        if isinstance(layer, FactorLayer):
            return FactorLayerDeclaration(categories=tuple(layer.categories.items()))
        if isinstance(layer.value_pattern, RegexValuePattern):
            return RegexNumericLayerDeclaration(
                missing_values=tuple(layer.missing_values),
                pattern=layer.value_pattern.pattern,
                type=layer.type,
            )
        return PlainNumericLayerDeclaration(
            missing_values=tuple(layer.missing_values), type=layer.type
        )

    @staticmethod
    def _project_stripping(
        column: StrippedSequence, rule: LongRule | WideRule
    ) -> SequenceOperation:
        syntax = rule.sequence_syntax[column.syntax]
        if isinstance(syntax, TokenRegexSyntax):
            return TokenRegexStripper(syntax.token_pattern, syntax.token_position)
        assert isinstance(syntax, PlainSequenceSyntax)
        return PlainSequenceStripper()

    @staticmethod
    def _project_modifications(
        column: ProformaSequence, rule: LongRule | WideRule
    ) -> SequenceOperation:
        """Resolve only the explicitly referenced map of a normalization operation."""
        syntax = rule.sequence_syntax[column.syntax]
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
                (entry, UNIMOD_REGISTRY.resolve(entry.accession))
                for entry in rule.modification_maps[column.modification_map]
            )
        )
        if isinstance(syntax, SiteListSyntax):
            return SiteListNormalizer(
                SiteListModificationConfig(
                    kind="site_list",
                    delimiter=syntax.delimiter,
                    site_base=syntax.site_base,
                    case_sensitive=column.case_sensitive,
                    unknown_policy=column.unknown_policy,
                    entries=entries,
                )
            )
        if isinstance(syntax, EmbeddedSiteListSyntax):
            return EmbeddedSiteListNormalizer(
                EmbeddedSiteListModificationConfig(
                    kind="embedded_site_list",
                    delimiter=syntax.delimiter,
                    entry_pattern=syntax.entry_pattern,
                    site_base=syntax.site_base,
                    case_sensitive=column.case_sensitive,
                    unknown_policy=column.unknown_policy,
                    entries=entries,
                )
            )
        assert isinstance(syntax, TokenRegexSyntax)
        return TokenRegexNormalizer(
            TokenRegexModificationConfig(
                kind="token_regex",
                token_pattern=syntax.token_pattern,
                token_position=syntax.token_position,
                case_sensitive=column.case_sensitive,
                unknown_policy=column.unknown_policy,
                entries=entries,
            )
        )

    @staticmethod
    def _project_provenance(rule: LongRule | WideRule) -> Mapping[str, JsonValue]:
        """What the parse section records: who wrote it, the rule, and the facts steps read.

        ``produced_by`` and the role maps are not decoration. Later APB steps must not have
        to validate a schema-0.8 document to learn which columns and layers carry a meaning.
        """
        provenance: dict[str, JsonValue] = {
            "rule_json": json.dumps(rule.model_dump(mode="json")),
            "column_roles": {
                role: entry.name for entry in rule.columns.var for role in entry.roles
            },
            "layer_roles": _layer_roles(rule.measurements.layers),
            "schema_version": rule.schema_version,
            "software_name": rule.software_name,
            "shape": rule.shape,
            "quantification_level": rule.quantification_level,
        }
        if rule.sample_annotation is not None:
            provenance["sample_annotation_matching"] = rule.sample_annotation.matching.model_dump(
                mode="json",
                exclude_none=True,
            )
        return provenance

    def resolve_source(
        self, evidence: SourceEvidence, *, checks: Literal["standard", "strict"] = "standard"
    ) -> ParseStrategy:
        """Resolve this declaration against one observed physical source."""
        return SourcePlanResolver(self._configuration).resolve(evidence, checks=checks)
