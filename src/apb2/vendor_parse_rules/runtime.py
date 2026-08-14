"""Runtime behaviour over rule configs: recognition, parameter gates, overrides.

The pydantic models in ``model.py`` represent the configs and nothing else; every
question asked *of* a rule at runtime is answered here, pandas-free, so document
selection (``documents/select.py``) can use it without touching the computational stack.

``recognition_for(rule)`` is the one shape dispatch on this side: past it, a
``LongRecognition`` reads sources as exact column names and a ``WideRecognition`` reads
them as ``(?P<sample>...)`` header regexes, and neither carries a flag the other could
misread.
"""

from __future__ import annotations

import re

from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.model import (
    ColumnGroup,
    ColumnLabeledFragments,
    ConditionValue,
    Layer,
    LongRule,
    WideRule,
    modification_outputs,
    validate_rule,
)

_SYNTHESIZED = frozenset({"stripped_sequence"})

type AxisName = str  # "obs" | "var"


def layer_required(rule: LongRule | WideRule, layer: Layer) -> bool:
    """A layer must be present iff it is the ``x_layer`` or explicitly ``required``."""
    return layer.required or layer.name == rule.axis.x_layer


def synthesized_columns(rule: LongRule | WideRule) -> frozenset[str]:
    """Columns apb2 creates itself, which must never be required of the input."""
    if rule.modifications is None:
        return _SYNTHESIZED
    return _SYNTHESIZED | modification_outputs(rule.modifications)


def declared_source_columns(recognition: Recognition) -> set[str]:
    """Every raw source the rule's column groups read: selected, optional, computed inputs."""
    needed: set[str] = set()
    for _axis, group in recognition.column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
        for column in group.computed:
            needed.update(column.inputs)
    return needed


def available_for(rule: LongRule | WideRule, parameters: Parameters | None) -> bool:
    """Whether the rule's parameter gate is satisfied.

    An ungated rule is always available; a gated one (Sage: ``combine_charge_states``
    decides the level) needs evidence, so it is unavailable when the caller has none.
    """
    if not rule.requires_search_parameters:
        return True
    if parameters is None:
        return False
    return _condition_matches(rule.requires_search_parameters, parameters)


def resolved_for(rule: LongRule | WideRule, parameters: Parameters | None) -> LongRule | WideRule:
    """Return the rule with every matching X-layer override applied (DIA-NN: acquisition
    mode decides the quantitative column). No evidence or no match returns the rule
    unchanged; matching overrides that disagree raise.
    """
    if parameters is None or not rule.search_parameter_overrides:
        return rule
    x_layers = {
        override.x_layer
        for override in rule.search_parameter_overrides
        if _condition_matches(override.when_search_parameters, parameters)
    }
    if not x_layers:
        return rule
    if len(x_layers) > 1:
        raise ValueError(
            f"matching search-parameter overrides disagree on x_layer: {sorted(x_layers)}"
        )
    payload = rule.model_dump(mode="python")
    payload["axis"]["x_layer"] = next(iter(x_layers))
    return validate_rule(payload)


def _condition_matches(condition: dict[str, ConditionValue], parameters: Parameters) -> bool:
    """Whether every declared parameter equality holds for the parsed values."""
    observed = parameters.model_dump(mode="json", include=set(condition))
    return observed == condition


class LongRecognition:
    """Header recognition for a long rule: every source is an exact column name."""

    def __init__(self, rule: LongRule) -> None:
        self._rule = rule
        expected = set(rule.columns.obs.select.values()) | set(rule.columns.var.select.values())
        expected.update(layer.source for layer in rule.layers if layer_required(rule, layer))
        self.required_headers: frozenset[str] = frozenset(expected - synthesized_columns(rule))

    def column_groups(self) -> tuple[tuple[AxisName, ColumnGroup], ...]:
        return (("obs", self._rule.columns.obs), ("var", self._rule.columns.var))

    def layer_source_columns(self, header: list[str]) -> set[str]:
        """Layer sources are exact column names in a long rule."""
        del header
        return {layer.source for layer in self._rule.layers}

    def matches(self, headers: list[str] | tuple[str, ...] | frozenset[str]) -> bool:
        """Whether raw input headers satisfy the rule's required sources."""
        header_set = set(headers)
        if not _fragment_label_present(self._rule, header_set):
            return False
        return self.required_headers.issubset(header_set)


class WideRecognition:
    """Header recognition for a wide rule: layer sources are sample-capturing regexes."""

    def __init__(self, rule: WideRule) -> None:
        self._rule = rule
        self._required_var = frozenset(
            set(rule.columns.var.select.values()) - synthesized_columns(rule)
        )

    def column_groups(self) -> tuple[tuple[AxisName, ColumnGroup], ...]:
        return (("var", self._rule.columns.var),)

    def layer_source_columns(self, header: list[str]) -> set[str]:
        """Expand each layer's header regex over the real header."""
        matched: set[str] = set()
        for layer in self._rule.layers:
            compiled = re.compile(layer.source)
            matched.update(name for name in header if compiled.match(name) is not None)
        return matched

    def matches(self, headers: list[str] | tuple[str, ...] | frozenset[str]) -> bool:
        """Whether raw input headers satisfy the rule's required sources."""
        header_set = set(headers)
        if not _fragment_label_present(self._rule, header_set):
            return False
        for layer in self._rule.layers:
            if layer_required(self._rule, layer) and not any(
                re.compile(layer.source).match(header) for header in header_set
            ):
                return False
        return self._required_var.issubset(header_set)


type Recognition = LongRecognition | WideRecognition


def recognition_for(rule: LongRule | WideRule) -> Recognition:
    """Read the rule's shape once and return the recognition it names."""
    if isinstance(rule, LongRule):
        return LongRecognition(rule)
    return WideRecognition(rule)


def _fragment_label_present(rule: LongRule | WideRule, header_set: set[str]) -> bool:
    if isinstance(rule.fragments, ColumnLabeledFragments):
        return rule.fragments.label_column in header_set
    return True
