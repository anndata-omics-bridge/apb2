"""The physical read plan: exactly which source columns a parser reads, and how.

Compiled once during construction from the rule and the inspected header, so the reader
never loads an entire vendor export only to discard unused columns afterward.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from apb2.vendor_parse_rules.model import LongRule, WideRule


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """Projected source columns in header order, and which of them stay textual."""

    columns: tuple[str, ...]
    string_sources: frozenset[str]


def compile_read_plan(
    rule: LongRule | WideRule,
    header: Sequence[str],
    modification_sources: Iterable[str],
    packed_columns: Iterable[str],
) -> ReadPlan:
    """Compile the exact projection one rule needs from one inspected header.

    Required sources are validated separately (``matches_headers`` during construction);
    the projection is simply the intersection of the header with everything the rule can
    read: selected and optional sources, computed-column inputs, layer sources (exact for
    long rules, regex-expanded for wide ones), modification sources, and packed fragment
    columns.
    """
    needed: set[str] = set(modification_sources) | set(packed_columns)
    for _axis, group in rule.named_column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
        for column in group.computed:
            needed.update(column.inputs)
    needed.update(rule.layer_source_columns(header))
    columns = tuple(name for name in header if name in needed)
    strings = string_sources_for_rules([rule]) & set(columns)
    return ReadPlan(columns=columns, string_sources=frozenset(strings))


def string_sources_for_rules(rules: Iterable[LongRule | WideRule]) -> frozenset[str]:
    """Return real vendor sources whose exact textual tokens must survive reading."""
    source_types: dict[str, str] = {}
    for rule in rules:
        for _axis, group in rule.named_column_groups():
            selected = {**group.select, **group.optional_select}
            for output_name, source_name in selected.items():
                logical_type = group.type_for(output_name)
                if source_name in source_types and source_types[source_name] != logical_type:
                    raise ValueError(
                        "conflicting logical types for vendor source "
                        f"{source_name!r}: {source_types[source_name]!r} and "
                        f"{logical_type!r}"
                    )
                source_types[source_name] = logical_type
    return frozenset(
        source for source, logical_type in source_types.items() if logical_type == "string"
    )
