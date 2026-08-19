"""Whether a real header satisfies a rule, and which of its columns a layer names.

The one shape dispatch on the rule side: past ``recognition_for`` a ``LongRecognition``
reads layer sources as exact column names and a ``WideRecognition`` reads them as
``(?P<sample>...)`` header regexes, and neither carries a flag the other could misread.

Nothing imports this module from outside ``vendor_parse_rules``: ``rules.get_rule`` builds
the recognition and hands it back with the rule it belongs to.
"""

from __future__ import annotations

import re

from apb2.vendor_parse_rules.model import (
    ColumnGroup,
    ColumnLabeledFragments,
    LongRule,
    WideRule,
    layer_required,
    modification_outputs,
)

_SYNTHESIZED = frozenset({"stripped_sequence"})

type AxisName = str  # "obs" | "var"


def synthesized_columns(rule: LongRule | WideRule) -> frozenset[str]:
    """Columns apb2 creates itself, which must never be required of the input."""
    if rule.modifications is None:
        return _SYNTHESIZED
    return _SYNTHESIZED | modification_outputs(rule.modifications)


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
