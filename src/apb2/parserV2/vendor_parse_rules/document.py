"""Load, compose and validate rule documents; physical recognition belongs to parsing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.annotation import SampleAnnotation
from apb2.parserV2.vendor_parse_rules.schema.base import (
    LEVELS,
    ModelBase,
    QuantificationLevel,
    SchemaVersion,
)
from apb2.parserV2.vendor_parse_rules.schema.input import Input
from apb2.parserV2.vendor_parse_rules.schema.parameters import (
    ConditionValue,
    SearchParameterField,
)
from apb2.parserV2.vendor_parse_rules.schema.rule import (
    LongRule,
    WideRule,
    validate_rule,
)

type JsonDict = dict[str, object]
"""A raw rules.json fragment: dicts merge without models, presence is key membership.

It must not become a wrapper class whose methods merely forward ordinary dict
operations. Malformed values ride the merge untouched and are reported, with their
authored paths, at the single effective-rule validation boundary.
"""
type MergeBlock = Callable[[JsonDict, JsonDict], JsonDict]


class RuleNotApplicable(ValueError):
    """This rule does not apply to what the caller has — try another level.

    Detection catches this while resolving level selections, so anything meaning "not this
    level" must be this class or a subclass, and
    anything meaning "the caller is wrong" must not be.
    """


@dataclass(frozen=True, slots=True)
class SearchParameterEvidence:
    """The complete parameter vocabulary permitted in schema-0.8 conditions."""

    acquisition_method: Literal["DDA", "DIA", "unknown"]
    combine_charge_states: bool | None

    def observed(self, requested: Iterable[SearchParameterField]) -> dict[str, ConditionValue]:
        """The requested fields' values, for comparison against a declared condition."""
        values: dict[SearchParameterField, ConditionValue] = {
            "acquisition_method": self.acquisition_method,
            "combine_charge_states": self.combine_charge_states,
        }
        return {name: values[name] for name in requested}


# ------------------------------------------------------------------------ the effective rule


@dataclass(frozen=True, slots=True)
class EffectiveRule:
    """One level's validated declaration and its table's input policy."""

    input: Input
    declaration: LongRule | WideRule
    preparation: str | None = None


# ------------------------------------------------------------------ the document shell


class _PreparationSchema(ModelBase):
    """Select a registered function; the level rules describe its output."""

    how: Literal["alphadia", "maxquant"]


class _RuleTableSchema(ModelBase):
    """One physical input and the level declarations composed only within that table."""

    input: Input
    base: JsonDict
    levels: dict[QuantificationLevel, JsonDict] = Field(min_length=1)
    prepare: _PreparationSchema | None = None


class RuleDocument(ModelBase):
    """One rules.json, with raw fragments validated after base/level composition.

    The fragments stay raw dicts through the base-times-level merge — merging dicts needs no
    models, presence is key membership — and cross the single typed boundary,
    ``validate_rule``, only once composed. Cross-block semantics are validated there:
    unknown keys and wrong types ride through the merge and are reported there with paths.
    """

    path: Path
    schema_version: SchemaVersion
    file_version: str
    software_name: str
    software_version_pattern: str
    sample_annotation: SampleAnnotation | None = None
    tables: list[_RuleTableSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_level_ownership(self) -> RuleDocument:
        levels = [level for table in self.tables for level in table.levels]
        if len(levels) != len(set(levels)):
            raise ValueError("each quantification level must belong to exactly one table")
        return self

    @property
    def levels(self) -> tuple[QuantificationLevel, ...]:
        declared = {level for table in self.tables for level in table.levels}
        return tuple(level for level in LEVELS if level in declared)

    @property
    def table_levels(self) -> tuple[tuple[QuantificationLevel, ...], ...]:
        """The level groups that share one input, in authored table order."""
        return tuple(tuple(table.levels) for table in self.tables)

    def declared(self, level: QuantificationLevel) -> EffectiveRule:
        """The rule this file *declares* for ``level``, gates and overrides ignored.

        For callers that have no evidence and cannot have any: recognizing a vendor from
        column headers, and the sweep over every packaged level.
        """
        table = self._table_for(level)
        return self._effective(self._payload_for(level, table), table)

    def rule(
        self,
        level: QuantificationLevel,
        evidence: SearchParameterEvidence,
    ) -> EffectiveRule:
        """The rule this file declares for ``level``, as the evidence selects it.

        Raises ``RuleNotApplicable`` — naming what went wrong — when the file has no such
        level, or when its parameter gate excludes this evidence.
        """
        effective = self.declared(level)
        rule = effective.declaration
        gate = rule.requires_search_parameters
        observed = evidence.observed(gate)
        if gate != observed:
            raise RuleNotApplicable(
                f"{self.software_name!r} level {level!r} requires search parameters {gate}, "
                f"but the supplied evidence is {observed}"
            )
        primary_layers = {
            override.primary_layer
            for override in rule.search_parameter_overrides
            if evidence.observed(override.when_search_parameters) == override.when_search_parameters
        }
        if len(primary_layers) > 1:
            raise ValueError(
                "matching search-parameter overrides disagree on primary_layer: "
                f"{sorted(primary_layers)}"
            )
        if not primary_layers:
            return effective
        payload = rule.model_dump(mode="python")
        payload["measurements"]["primary_layer"] = primary_layers.pop()
        return replace(effective, declaration=validate_rule(payload))

    def _effective(self, payload: JsonDict, table: _RuleTableSchema) -> EffectiveRule:
        declaration = validate_rule(payload)
        return EffectiveRule(
            input=table.input,
            declaration=declaration,
            preparation=table.prepare.how if table.prepare is not None else None,
        )

    def _table_for(self, level: QuantificationLevel) -> _RuleTableSchema:
        for table in self.tables:
            if level in table.levels:
                return table
        raise RuleNotApplicable(
            f"{self.path} has no level {level!r}; available: {list(self.levels)}"
        )

    def _payload_for(self, level: QuantificationLevel, table: _RuleTableSchema) -> JsonDict:
        """Compose one declared level over its own table's base."""
        level_fragment = table.levels[level]
        return {
            "schema_version": self.schema_version,
            "file_version": self.file_version,
            "software_name": self.software_name,
            "software_version_pattern": self.software_version_pattern,
            "quantification_level": level,
            "shape": table.input.shape,
            **(
                {"sample_annotation": self.sample_annotation.model_dump(mode="json")}
                if self.sample_annotation is not None
                else {}
            ),
            **_merge_fragments(table.base, level_fragment),
        }


def make_rule_document(path: Path, payload: JsonDict) -> RuleDocument:
    """Validate one raw rules.json payload and return the document it describes."""
    return RuleDocument.model_validate({"path": path, **payload})


def document_json_schema() -> dict[str, object]:
    """Describe the authored document shell; effective rules validate merged fragments."""
    schema = RuleDocument.model_json_schema()
    schema["properties"].pop("path")
    schema["required"].remove("path")
    schema["title"] = "RuleDocument"
    return schema


# ---------------------------------------------------- the file: the base-level merge


def _merge_fragments(base: JsonDict, level: JsonDict) -> JsonDict:
    """Merge one level fragment over a base fragment.

    ``columns`` and ``measurements`` descend one additional level; all other declared merge
    shapes are top-level.
    """
    merged = _merge_blocks(
        base,
        level,
        mappings=(
            "axis",
            "sequence_syntax",
            "modification_maps",
            "fragments",
            "requires_search_parameters",
        ),
        sequences=("search_parameter_overrides",),
    )
    _merge_nested(merged, base, level, "columns", _merge_columns)
    _merge_nested(merged, base, level, "measurements", _merge_measurements)
    return merged


def _merge_nested(
    merged: JsonDict,
    base: JsonDict,
    level: JsonDict,
    key: str,
    merge: MergeBlock,
) -> None:
    """Replace one nested block in ``merged`` with its own two-level merge, when present."""
    if key not in base and key not in level:
        return
    base_block = base.get(key, {})
    level_block = level.get(key, {})
    if not isinstance(base_block, dict) or not isinstance(level_block, dict):
        return
    merged[key] = merge(base_block, level_block)


def _merge_columns(base: JsonDict, level: JsonDict) -> JsonDict:
    """Merge the obs and var groups inside a columns block."""
    columns: JsonDict = {**base, **level}
    for axis in ("obs", "var"):
        if axis not in base and axis not in level:
            continue
        if axis not in base:
            columns[axis] = level[axis]
            continue
        if axis not in level:
            columns[axis] = base[axis]
            continue
        base_group = base[axis]
        level_group = level[axis]
        if isinstance(base_group, list) and isinstance(level_group, list):
            entries = [*base_group, *level_group]
            columns[axis] = [
                entry for entry in entries if isinstance(entry, dict) and "source" in entry
            ] + [entry for entry in entries if not isinstance(entry, dict) or "source" not in entry]
    return columns


def _merge_measurements(base: JsonDict, level: JsonDict) -> JsonDict:
    """Merge a measurements block: nested duplicates key-wise, layers concatenated."""
    return _merge_blocks(base, level, mappings=("duplicates",), sequences=("layers",))


def _merge_blocks(
    base: JsonDict,
    level: JsonDict,
    *,
    mappings: tuple[str, ...],
    sequences: tuple[str, ...],
) -> JsonDict:
    """Merge named mappings key-wise and concatenate named sequences.

    A malformed value remains untouched so effective-rule validation reports it.
    """
    merged: JsonDict = {**base, **level}
    for key in mappings:
        if key not in base and key not in level:
            continue
        base_block = base.get(key, {})
        level_block = level.get(key, {})
        if isinstance(base_block, dict) and isinstance(level_block, dict):
            merged[key] = {**base_block, **level_block}
    for key in sequences:
        if key not in base and key not in level:
            continue
        base_entries = base.get(key, [])
        level_entries = level.get(key, [])
        if isinstance(base_entries, list) and isinstance(level_entries, list):
            merged[key] = [*base_entries, *level_entries]
    return merged
