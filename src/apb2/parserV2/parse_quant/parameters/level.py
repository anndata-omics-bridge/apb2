"""One quantification level before and after physical-source resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from apb2.parserV2.parse_quant.parameters.axis import (
    EmbeddedSiteListModificationConfig,
    ModificationConfig,
    ResolvedAxisColumnPlan,
    SiteListModificationConfig,
    WorkingAxisConfiguration,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    LayerContractConfig,
    LayerValueConfig,
    RawValuePresenceConfig,
    WorkingMeasurements,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DecompositionConfig,
    InputContract,
    LevelReadPlan,
    NumericTextFormat,
    SourceLayoutDeclaration,
)

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
"""The parsing-owned level vocabulary, structurally equal to the rule package's own."""

LEVELS: tuple[QuantificationLevel, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)
"""Canonical level order preserved by compiler selection."""


@dataclass(frozen=True, slots=True)
class WorkingParseConfiguration:
    """One complete projected rule before physical-source resolution."""

    level: QuantificationLevel
    input: InputContract
    source_layout: SourceLayoutDeclaration
    obs: WorkingAxisConfiguration
    var: WorkingAxisConfiguration
    measurements: WorkingMeasurements
    modifications: tuple[ModificationConfig, ...]
    provenance: Mapping[str, JsonValue]
    preparation: str | None = None

    def accepts_header(self, header: tuple[str, ...]) -> bool:
        """Whether a physical header satisfies every required source declaration."""
        present = frozenset(header)
        exact = frozenset(
            {
                *self.obs.required_sources(),
                *self.var.required_sources(),
                *self._modification_sources(),
                *self.source_layout.packed_sources(),
            }
        )
        if not exact <= present:
            return False
        return all(
            self.source_layout.has_layer_source(source, header)
            for source in self.measurements.required_sources()
        )

    def _modification_sources(self) -> tuple[str, ...]:
        return tuple(
            column
            for config in self.modifications
            for column in (
                (config.sequence_column, config.modification_column, config.site_column)
                if isinstance(config, SiteListModificationConfig)
                else (
                    (config.sequence_column, config.modification_column)
                    if isinstance(config, EmbeddedSiteListModificationConfig)
                    else (config.source_column,)
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ResolvedLevelPlan:
    """One complete level plan resolved against one physical source."""

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
