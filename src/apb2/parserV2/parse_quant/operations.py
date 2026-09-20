"""The compiler's semantic input: source requirements and executable column operations.

The parent facade chooses computations from authored declarations, without retaining storage
models. Source resolution binds these same objects to available inputs and execution phases.
Only coercion and layer parsing still need construction here: their numeric notation comes
from the physical source. Duplicate policies are immutable, stateless registry entries.

These contracts compose inward settings and behavior, so they live here rather than in the
independent parameters leaf. The parser itself consumes only the bound runtime collaborators.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
    DuplicatePolicy,
    LayerValueParser,
)
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
)
from apb2.parserV2.parse_quant.modifications import (
    PlainSequenceStripper,
    SequenceColumn,
    TokenRegexStripper,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnSelection,
    AxisLogicalType,
)
from apb2.parserV2.parse_quant.parameters.level import JsonValue, QuantificationLevel
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerValueDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
    WorkingMeasurements,
)
from apb2.parserV2.parse_quant.parameters.source import (
    InputContract,
    NumericTextFormat,
    SourceLayoutDeclaration,
)
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)

# ----------------------------------------------------------------------------- registries

type ComputedOperation = (
    CoalesceColumn
    | JoinNonemptyColumn
    | SequenceColumn
    | PlainSequenceStripper
    | TokenRegexStripper
    | ProformaIonColumn
    | ProformaFragmentColumn
)


@dataclass(frozen=True, slots=True)
class WorkingAxisConfiguration:
    """One axis's physical selections and already constructed computations."""

    final_key_columns: tuple[str, ...]
    required_selections: tuple[AxisColumnSelection, ...]
    optional_selections: tuple[AxisColumnSelection, ...]
    computed: tuple[ComputedOperation, ...]
    declared_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkingParseConfiguration:
    """Pydantic-free source requirements and operations produced by the facade."""

    level: QuantificationLevel
    input: InputContract
    source_layout: SourceLayoutDeclaration
    obs: WorkingAxisConfiguration
    var: WorkingAxisConfiguration
    measurements: WorkingMeasurements
    provenance: Mapping[str, JsonValue]
    preparation: str | None = None

    def accepts_header(self, header: tuple[str, ...]) -> bool:
        """Whether all required physical sources occur in a candidate header."""
        required = {
            selection.source
            for axis in (self.obs, self.var)
            for selection in axis.required_selections
        } | set(self.source_layout.packed_sources())
        return required <= set(header) and all(
            self.source_layout.has_layer_source(layer.source, header)
            for layer in self.measurements.required_layers
        )


_DUPLICATE_POLICIES: Mapping[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": AggregateNumericDuplicates(),
}
"""One policy per executable duplicate mode; schema 0.8 declares no others."""


def make_axis_coercer(
    logical_type: AxisLogicalType,
    number_format: NumericTextFormat,
) -> AxisValueCoercer:
    """Construct the coercion one logical type names under this source's number notation."""
    if logical_type == "integer":
        return IntegerAxisCoercer(notation=number_format)
    if logical_type == "number":
        return NumberAxisCoercer(notation=number_format)
    if logical_type == "boolean":
        return BooleanAxisCoercer()
    return StringAxisCoercer()


def duplicate_policy_for(mode: DuplicateMode) -> DuplicatePolicy:
    """Select the policy one resolved duplicate mode names."""
    return _DUPLICATE_POLICIES[mode]


def make_layer_parser(
    layer_name: str, value: LayerValueDeclaration, numbers: NumericTextFormat
) -> LayerValueParser:
    """Configure raw presence and canonical parsing together from one layer declaration."""
    if isinstance(value, PlainNumericLayerDeclaration):
        return PlainNumericLayerParser(
            layer_name=layer_name,
            missing_values=value.missing_values,
            number_format=numbers,
            numeric_type=value.type,
        )
    if isinstance(value, RegexNumericLayerDeclaration):
        return RegexNumericLayerParser(
            layer_name=layer_name,
            missing_values=value.missing_values,
            pattern=value.pattern,
            number_format=numbers,
            numeric_type=value.type,
        )
    return FactorLayerParser(categories=value.categories)
