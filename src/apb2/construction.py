"""The composition boundary: ``make_parse_strategy`` and ``make_parse_strategies``.

The only place where a rule, the concrete input, and the parameter evidence meet.
The rule's parameter conditions are resolved here — the gate is checked and matching
X-layer overrides applied — and everything past it is a fully injected ``Parser``:
concrete readers, resolved dialects, exact projections, configured strategies, and
serialized provenance. No rule model, no unresolved policy.
"""

from __future__ import annotations

from collections.abc import Iterable

from apb2.columns import make_column_plan
from apb2.convert.preprocess import make_modification_applier
from apb2.errors import (
    IncompatibleSourceError,
    NoCompatibleLevelError,
)
from apb2.input.binding import bind_source
from apb2.input.plan import compile_read_plan
from apb2.parser import Parser
from apb2.sources import InputSource
from apb2.steps import (
    make_fragment_step,
    make_output_metadata,
    make_result_conversion,
)
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.model import LEVELS, LongRule, WideRule


def make_parse_strategy(
    rule: LongRule | WideRule,
    source: InputSource,
    parameters: Parameters | None = None,
    *,
    strict: bool = False,
) -> Parser:
    """Construct one fully injected parser for one quantification level.

    Raises ``IncompatibleSourceError`` when ``source`` cannot satisfy ``rule`` or when the
    rule's parameter gate is not satisfied by ``parameters``. ``strict`` promotes non-``X``
    layer-contract warnings to errors.
    """
    if not rule.available_for(parameters):
        raise IncompatibleSourceError(
            f"{rule.software_name!r} level {rule.quantification_level!r} is not available "
            f"for the supplied search parameters (requires {rule.requires_search_parameters})"
        )
    rule = rule.resolved_for(parameters)
    binding = bind_source(source, rule)
    header = binding.header()
    if not rule.matches_headers(header):
        raise IncompatibleSourceError(
            f"{binding.path} does not carry the columns required by "
            f"{rule.software_name!r} level {rule.quantification_level!r}"
        )
    modifications = make_modification_applier(rule)
    fragments = make_fragment_step(rule, modifications.source_columns())
    plan = compile_read_plan(
        rule,
        header,
        modifications.source_columns(),
        fragments.packed_columns(),
    )
    return Parser(
        level=rule.quantification_level,
        input=binding.make_reader(plan),
        modifications=modifications,
        fragments=fragments,
        columns=make_column_plan(rule),
        conversion=make_result_conversion(rule, strict=strict),
        metadata=make_output_metadata(rule),
    )


def make_parse_strategies(
    rules: Iterable[LongRule | WideRule],
    source: InputSource,
    parameters: Parameters | None = None,
    *,
    strict: bool = False,
) -> list[Parser]:
    """Construct one parser per rule the source satisfies, in ``LEVELS`` order.

    Incompatible rules never gate construction of the compatible ones; an empty result
    is an error because a source that satisfies nothing was the wrong source.
    """
    ordered = sorted(rules, key=_level_rank)
    parsers: list[Parser] = []
    for rule in ordered:
        try:
            parsers.append(make_parse_strategy(rule, source, parameters, strict=strict))
        except IncompatibleSourceError:
            continue
    if not parsers:
        levels = [rule.quantification_level for rule in ordered]
        raise NoCompatibleLevelError(
            f"no rule among levels {levels} is satisfied by the bound source {source!r}"
        )
    return parsers


def _level_rank(rule: LongRule | WideRule) -> int:
    return LEVELS.index(rule.quantification_level)
