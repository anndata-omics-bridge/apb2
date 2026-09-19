"""``ResolvedLevelPlan``: one level, resolved against one physical source, all at once.

Atomic on purpose. One projected source set feeds the reader, both axes, the decomposer, the
separator, and value parsers, so optional-source presence, wide captures, packed source order,
and the primary sample set cannot disagree between plans that were resolved separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from apb2.parserV2.parse_quant.parameters.axis import (
    ModificationConfig,
    ResolvedAxisColumnPlan,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerContractConfig,
    LayerValueConfig,
    RawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DecompositionConfig,
    LevelReadPlan,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.parameters.working import JsonValue, QuantificationLevel


@dataclass(frozen=True, slots=True)
class ResolvedLevelPlan:
    """Everything ``ParseRuleCompiler`` needs to construct one parser."""

    level: QuantificationLevel
    number_format: NumericTextFormat
    read: LevelReadPlan
    decomposition: DecompositionConfig
    obs: ResolvedAxisColumnPlan
    var: ResolvedAxisColumnPlan
    modifications: tuple[ModificationConfig, ...]
    duplicate_mode: DuplicateMode
    raw_value_presence: tuple[RawValuePresenceConfig, ...]
    layer_values: tuple[LayerValueConfig, ...]
    layer_contract: LayerContractConfig
    provenance: Mapping[str, JsonValue]
