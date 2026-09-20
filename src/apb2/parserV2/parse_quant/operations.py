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
    AxisValueCoercer,
    ColumnComputer,
    DuplicatePolicy,
    LayerValueParser,
    RawValuePresence,
)
from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
    NullOnlyRawValuePresence,
    PlainNumericRawValuePresence,
    RegexNumericRawValuePresence,
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
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisLogicalType,
    CoalesceColumnConfig,
    ComputedColumnConfig,
    EmbeddedSiteListModificationConfig,
    JoinNonemptyColumnConfig,
    ModificationConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
    StrippingSyntaxConfig,
    TokenRegexSyntaxConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerValueConfig,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)

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
