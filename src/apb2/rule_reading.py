"""How the parse strategy reads a rule's declarations.

Outside ``vendor_parse_rules`` on purpose: that folder deserializes rules.json and hands
back one composed rule, and every question asked *of* those declarations afterwards — which
raw sources are read, which columns an axis frame must carry, which must exist before the
pivot — is the consumer's, not the schema's. Every function here takes plain declarations or
the recognition that came with the rule, and returns names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from apb2.vendor_parse_rules.model import (
    ColumnGroup,
    LongRule,
    Modifications,
    SiteListModifications,
    WideRule,
    group_names,
    modification_outputs,
)
from apb2.vendor_parse_rules.rules import Recognition


def rule_label(rule: LongRule | WideRule) -> str:
    """How one rule names itself in an error message."""
    return f"{rule.software_name!r} level {rule.quantification_level!r}"


def modification_sources(modifications: Modifications) -> tuple[str, ...]:
    """The raw vendor columns one modifications declaration reads."""
    if isinstance(modifications, SiteListModifications):
        return (
            modifications.sequence_column,
            modifications.modification_column,
            modifications.site_column,
        )
    return (modifications.source_column,)


def declared_source_columns(recognition: Recognition) -> set[str]:
    """Every raw source the rule's column groups read: selected, optional, computed inputs."""
    needed: set[str] = set()
    for _axis, group in recognition.column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
        for column in group.computed:
            needed.update(column.inputs)
    return needed


def projected_columns(
    recognition: Recognition, header: Sequence[str], also: Iterable[str]
) -> tuple[str, ...]:
    """The exact columns one rule reads out of one inspected header, in header order.

    Required sources are validated separately (``matches``); this is the intersection of
    the header with everything the rule *can* read: selected and optional sources,
    computed-column inputs, layer sources (exact for long rules, regex-expanded for wide
    ones), and whatever else the caller resolved against the same header.
    """
    needed = set(also) | declared_source_columns(recognition)
    needed.update(recognition.layer_source_columns(list(header)))
    return tuple(name for name in header if name in needed)


def string_typed_sources(recognition: Recognition) -> frozenset[str]:
    """Vendor sources whose exact textual tokens must survive reading.

    One source cannot carry two logical types: ``_check_one_type_per_source`` rejects that
    rule at validation, so reading the first declaration of each is enough here.
    """
    return frozenset(
        source
        for _axis, group in recognition.column_groups()
        for name, source in (group.select | group.optional_select).items()
        if group.types.get(name, "string") == "string"
    )


def key_closure(group: ColumnGroup, keys: Sequence[str]) -> set[str]:
    """Every declared column the axis keys are computed from, keys included.

    This is what must be materialized before the pivot: the pivot cannot group by a key
    that does not exist yet, and a computed key needs its inputs present first.
    """
    computed_inputs = {column.name: set(column.inputs) for column in group.computed}
    closure = set(keys)
    changed = True
    while changed:
        changed = False
        for name, inputs in computed_inputs.items():
            if name in closure and not inputs.issubset(closure):
                closure |= inputs
                changed = True
    return closure


def carried_columns(
    keys: Sequence[str], group: ColumnGroup, extras: tuple[str, ...]
) -> tuple[str, ...]:
    """Everything an axis frame must take off the flat table, in stable order.

    Declared names cover the already-prepared key-closure columns; raw sources cover
    everything the post-pivot pass materializes afterwards. Absent names (skipped
    optionals, columns not yet materialized) drop out where the frame is projected.
    """
    return tuple(
        dict.fromkeys(
            [
                *keys,
                *group_names(group),
                *group.select.values(),
                *group.optional_select.values(),
                *extras,
            ]
        )
    )


def var_extras(rule: LongRule | WideRule) -> tuple[str, ...]:
    """Raw modification sources/outputs and the fragment label the var frame may need."""
    extras: list[str] = []
    if rule.modifications is not None:
        extras.extend(modification_sources(rule.modifications))
        extras.extend(sorted(modification_outputs(rule.modifications)))
    if rule.fragments is not None:
        extras.append(rule.fragments.label_output)
    return tuple(extras)
