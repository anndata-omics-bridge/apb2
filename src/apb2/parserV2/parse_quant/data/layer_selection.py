"""Strict selection of layers carrying scientific abundance semantics."""

from dataclasses import dataclass
from typing import Literal, Protocol

from apb2.parserV2.parse_quant.data.parsed import JsonValue, ParsedLevel


@dataclass(frozen=True, slots=True)
class ResolvedLayerSelection:
    """Ordered layers selected by one policy, with portable provenance."""

    mode: Literal["primary", "named_abundance", "all_abundance"]
    layer_names: tuple[str, ...]

    def as_json(self) -> dict[str, JsonValue]:
        """Return the policy label used in operation provenance."""
        return {"selection_mode": self.mode}


class LayerSelection(Protocol):
    """Resolve an abundance selection against one level."""

    def resolve(self, level: ParsedLevel) -> ResolvedLayerSelection: ...


class PrimaryLayer:
    """Select the primary matrix, requiring its abundance role."""

    def resolve(self, level: ParsedLevel) -> ResolvedLayerSelection:
        """Validate and select the primary abundance layer."""
        NamedAbundanceLayer(level.primary_layer_name).resolve(level)
        return ResolvedLayerSelection("primary", (level.primary_layer_name,))


@dataclass(frozen=True, slots=True)
class NamedAbundanceLayer:
    """Select one named abundance layer."""

    name: str

    def resolve(self, level: ParsedLevel) -> ResolvedLayerSelection:
        """Require a present, quantitative layer with the abundance role."""
        if self.name not in level.layers:
            raise ValueError(f"level has no layer {self.name!r}")
        layer = level.layers[self.name]
        if "abundance" not in layer.semantic_roles or not layer.role.accepts_primary_layer():
            raise ValueError(f"layer {self.name!r} does not carry the abundance role")
        return ResolvedLayerSelection("named_abundance", (self.name,))


class AllAbundanceLayers:
    """Select every abundance layer in authored order."""

    def resolve(self, level: ParsedLevel) -> ResolvedLayerSelection:
        """Require at least one abundance layer and validate each selected layer."""
        names = tuple(
            name for name, layer in level.layers.items() if "abundance" in layer.semantic_roles
        )
        if not names:
            raise ValueError("level has no layer carrying the abundance role")
        for name in names:
            NamedAbundanceLayer(name).resolve(level)
        return ResolvedLayerSelection("all_abundance", names)


PRIMARY_LAYER = PrimaryLayer()
ALL_ABUNDANCE_LAYERS = AllAbundanceLayers()
