"""The parse strategy: the pipeline context and the composition root that fills it.

``Parser`` holds only completed runtime behavior and concrete values — no rule model, no
unresolved policy, no storage backend — and neither it nor any injected strategy branches
on which rule variant, vendor, layout, format, dialect policy, or mode was selected:
``make_parse_strategy`` is the only place where a rule, the concrete input, and the
parameter evidence meet, and the selector factories finish every choice there.

The pipeline runs conversion-first: after the read and the fragment explode, only the
axis-key closure is materialized on the flat table (the pivot cannot group without it);
every remaining declared column is materialized afterwards on the deduplicated axis
frames, where a column fix touches nrObs or nrVars rows instead of nrObs x nrVars.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Protocol

import pandas as pd

from apb2.columns import ColumnMaterialization
from apb2.conversion import conversion_for
from apb2.errors import IncompatibleSourceError, NoCompatibleLevelError
from apb2.fragments import exploder_for
from apb2.input import bind_source, compile_read_plan
from apb2.modifications.pipeline import applier_for
from apb2.result import ParsedData
from apb2.serialization import JsonValue
from apb2.sources import InputSource
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.model import LEVELS, LongRule, QuantificationLevel, WideRule
from apb2.vendor_parse_rules.runtime import available_for, recognition_for, resolved_for


class BoundInputReader(Protocol):
    """Read the bound source into one assembled table."""

    def read(self) -> pd.DataFrame: ...


class FragmentExploder(Protocol):
    """Expand packed per-fragment values into rows."""

    def explode(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


class ColumnPlan(Protocol):
    """Materialize declared columns: key closure before the pivot, the rest after."""

    def prepare_keys(self, table: pd.DataFrame, /) -> pd.DataFrame: ...

    def finish(self, result: ParsedData, /) -> ParsedData: ...


class TableConversion(Protocol):
    """Convert the key-prepared table into one backend-neutral result."""

    def parse(self, table: pd.DataFrame, /) -> ParsedData: ...


class Parser:
    """One quantification level's completed strategy graph.

    ``level`` is an output identity used for naming and provenance, never a discriminator
    consulted by any strategy. Every field is unconditional: a rule that declares no
    fragments receives the identity exploder.
    """

    def __init__(
        self,
        *,
        level: QuantificationLevel,
        input: BoundInputReader,
        fragments: FragmentExploder,
        columns: ColumnPlan,
        conversion: TableConversion,
        provenance: Mapping[str, JsonValue],
    ) -> None:
        self.level = level
        self.input = input
        self.fragments = fragments
        self.columns = columns
        self.conversion = conversion
        self.provenance = provenance

    def parse(self) -> ParsedData:
        """Run the plan once and return one backend-neutral result."""
        table = self.input.read()
        table = self.fragments.explode(table)
        table = self.columns.prepare_keys(table)
        result = self.conversion.parse(table)
        result = self.columns.finish(result)
        return replace(result, uns={**result.uns, **self.provenance})


def make_parse_strategy(
    rule: LongRule | WideRule,
    source: InputSource,
    parameters: Parameters | None = None,
    *,
    strict: bool = False,
) -> Parser:
    """Construct one fully injected parser for one quantification level.

    Raises ``IncompatibleSourceError`` when ``source`` cannot satisfy ``rule`` or when the
    rule's parameter gate is not satisfied by ``parameters``. ``strict`` promotes
    non-``X`` layer-contract warnings to errors.
    """
    if not available_for(rule, parameters):
        raise IncompatibleSourceError(
            f"{rule.software_name!r} level {rule.quantification_level!r} is not available "
            f"for the supplied search parameters (requires {rule.requires_search_parameters})"
        )
    rule = resolved_for(rule, parameters)
    recognition = recognition_for(rule)
    binding = bind_source(source, rule, recognition)
    header = binding.header()
    if not recognition.matches(header):
        raise IncompatibleSourceError(
            f"{binding.path} does not carry the columns required by "
            f"{rule.software_name!r} level {rule.quantification_level!r}"
        )
    applier = applier_for(rule)
    missing = [column for column in applier.sources if column not in set(header)]
    if missing:
        raise KeyError(f"[modifications] needs column(s) {missing} not found in {binding.path}")
    sources = applier.source_columns()
    fragments = exploder_for(rule, recognition, sources)
    plan = compile_read_plan(recognition, header, sources, fragments.packed_columns())
    return Parser(
        level=rule.quantification_level,
        input=binding.make_reader(plan),
        fragments=fragments,
        columns=ColumnMaterialization(rule, recognition, applier),
        conversion=conversion_for(rule, strict=strict),
        provenance=_provenance(rule),
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
    ordered = sorted(rules, key=lambda rule: LEVELS.index(rule.quantification_level))
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


def _provenance(rule: LongRule | WideRule) -> dict[str, JsonValue]:
    """Serialize the rule's provenance once; the parser retains no model."""
    return {
        "rule_json": json.dumps(rule.model_dump(mode="json")),
        "schema_version": rule.schema_version,
        "software_name": rule.software_name,
        "shape": rule.shape,
        "quantification_level": rule.quantification_level,
    }
