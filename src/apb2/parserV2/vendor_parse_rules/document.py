"""Load, compose and validate rule documents; physical recognition belongs to parsing."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from apb2.parserV2.vendor_parse_rules.schema.annotation import SampleAnnotation
from apb2.parserV2.vendor_parse_rules.schema.base import (
    AxisColumnType,
    ModelBase,
    PerFeature,
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

_SAMPLE = "{sample}"
"""The placeholder a wide layer's source writes where each column names its sample."""


class RuleNotApplicable(ValueError):
    """This rule does not apply to what the caller has — try another level.

    Detection catches this while resolving level selections, so anything meaning "not this
    level" must be this class or a subclass, and
    anything meaning "the caller is wrong" must not be.
    """


@dataclass(frozen=True, slots=True)
class SearchParameterEvidence:
    """The complete parameter vocabulary permitted in schema-0.10 conditions."""

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


# ------------------------------------------------------------------ the document shell


class _PreparationSchema(ModelBase):
    """Select a registered function; the level rules describe its output."""

    how: Literal[
        "alphadia", "maxquant_wide", "maxquant_evidence", "metamorpheus", "msangel", "wombat"
    ]


class _VarDefaults(ModelBase):
    """What a table's var columns read from a vendor column take where they state no value."""

    type: AxisColumnType
    required: bool
    per_feature: PerFeature


class _ObsDefaults(ModelBase):
    """What a long table's obs columns read from a vendor column take where they state none."""

    type: AxisColumnType
    required: bool


class _LayerDefaults(ModelBase):
    """What a table's layers take where they state no value; a factor layer takes no type."""

    encoding_mode: Literal["numeric", "factor"]
    type: Literal["number", "integer"]
    required: bool


class _TableDefaults(ModelBase):
    """A table's ``base.defaults``: each choice its entries share, stated once.

    An entry's own value wins. Computed entries take none: they have no such choice.
    """

    var: _VarDefaults
    obs: _ObsDefaults | None = None
    layers: _LayerDefaults


class _RuleTableSchema(ModelBase):
    """One physical input and the level declarations composed only within that table."""

    input: Input
    base: JsonDict
    levels: dict[QuantificationLevel, JsonDict] = Field(min_length=1)
    prepare: _PreparationSchema | None = None

    def defaults(self) -> _TableDefaults:
        """The values the table's column and layer entries take where they state none."""
        if "defaults" not in self.base:
            raise ValueError("base declares no defaults: the values its entries take by default")
        try:
            return _TableDefaults.model_validate(self.base["defaults"])
        except ValidationError as error:
            problems = "; ".join(
                ".".join(("base.defaults", *map(str, problem["loc"]))) + f": {problem['msg']}"
                for problem in error.errors()
            )
            raise ValueError(problems) from error

    @model_validator(mode="after")
    def _defaults_are_the_bases(self) -> _RuleTableSchema:
        """The base states its entries' defaults once, for the axes its table has."""
        if (self.defaults().obs is not None) != (self.input.shape == "long"):
            raise ValueError("base.defaults names obs exactly for a long table")
        levels = [level for level, fragment in self.levels.items() if "defaults" in fragment]
        if levels:
            raise ValueError(f"defaults belong to the table's base; levels {levels} declare them")
        return self

    @model_validator(mode="after")
    def _prepared_input_declares_no_detection(self) -> _RuleTableSchema:
        """A preparation reads its files by extension, so declared detection would go unread."""
        declared = (self.input.delimiter, self.input.numbers, self.input.encoding)
        if self.prepare is not None and any(value is not None for value in declared):
            raise ValueError("a prepared table reads its files by extension; declare no detection")
        return self

    @model_validator(mode="after")
    def _workbook_sheets_are_named_once(self) -> _RuleTableSchema:
        """A direct workbook table names its sheet; a preparation's join names its own."""
        sheet = self.input.sheet_name
        if self.prepare is None and ".xlsx" in self.input.extensions and sheet is None:
            raise ValueError(".xlsx input requires sheet_name")
        if self.prepare is not None and sheet is not None:
            raise ValueError("a prepared table's join names the sheets it reads; declare none")
        return self


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
    hierarchy: str
    parameter_file: Literal["required", "none"]
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
        return tuple(dict.fromkeys(level for table in self.tables for level in table.levels))

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

    def table_input(self, level: QuantificationLevel) -> Input:
        """The input policy of ``level``'s table, without composing or validating the level."""
        return self._table_for(level).input

    def preparation(self, level: QuantificationLevel) -> str | None:
        """The preparation that builds ``level``'s table from several inputs, if any."""
        prepare = self._table_for(level).prepare
        return None if prepare is None else prepare.how

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
        return EffectiveRule(input=table.input, declaration=declaration)

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
            "hierarchy": self.hierarchy,
            "shape": table.input.shape,
            **(
                {"sample_annotation": self.sample_annotation.model_dump(mode="json")}
                if self.sample_annotation is not None
                else {}
            ),
            **_compose_shorthand(
                _merge_fragments(table.base, level_fragment), table.input.shape, table.defaults()
            ),
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


# ------------------------------------------- the file: shorthand the composition resolves


def _compose_shorthand(merged: JsonDict, shape: str, defaults: _TableDefaults) -> JsonDict:
    """Resolve the authoring shorthand of one merged level into explicit declarations.

    ``source_prefix`` names the table a preparation step's join prefixes, such as MaxQuant's
    ``peptide.``: an entry's own prefix wins over the level's, which wins over the base's. A
    wide layer's ``{sample}`` stands for the table's ``sample_pattern``, the level's over the
    base's. A sequence column that names no ``syntax`` or ``modification_map`` takes the level's
    only one. Column entries read from a vendor column, and layers, take the table's
    ``defaults`` for each value they state none of. New dicts carry the result, so the
    document's fragments stay as authored.
    """
    composed = dict(merged)
    composed.pop("defaults", None)
    prefix = composed.pop("source_prefix", None)
    pattern = composed.pop("sample_pattern", None)
    measurements = composed.get("measurements")
    layers = measurements.get("layers") if isinstance(measurements, dict) else None
    _check_sample_pattern(shape, pattern, layers)
    columns = composed.get("columns")
    if isinstance(columns, dict):
        composed["columns"] = {
            group: [
                _composed_column(entry, prefix, composed, _column_defaults(defaults, group))
                for entry in entries
            ]
            if isinstance(entries, list)
            else entries
            for group, entries in columns.items()
        }
    if isinstance(measurements, dict) and isinstance(layers, list):
        composed["measurements"] = {
            **measurements,
            "layers": [
                _layer_with_defaults(
                    _wide_layer(layer, prefix, pattern)
                    if shape == "wide"
                    else _prefixed(layer, prefix),
                    defaults.layers,
                )
                for layer in layers
            ],
        }
    return composed


def _check_sample_pattern(shape: str, pattern: object, layers: object) -> None:
    """A wide table declares its ``sample_pattern`` exactly when a layer writes ``{sample}``."""
    if pattern is not None and shape != "wide":
        raise ValueError("sample_pattern applies to a wide table")
    if shape != "wide":
        return
    writing = [
        layer.get("name")
        for layer in (layers if isinstance(layers, list) else [])
        if isinstance(layer, dict) and _SAMPLE in str(layer.get("source"))
    ]
    if writing and pattern is None:
        raise ValueError(
            f"layers {writing} write {_SAMPLE}, and the table declares no sample_pattern, such "
            "as '(?P<sample>.+)'"
        )
    if pattern is not None and not writing:
        raise ValueError(f"sample_pattern is declared, and no layer writes {_SAMPLE}")


def _column_defaults(defaults: _TableDefaults, group: object) -> JsonDict:
    """The values one column group's entries take where they state none."""
    chosen = defaults.var if group == "var" else defaults.obs if group == "obs" else None
    return {} if chosen is None else chosen.model_dump()


def _layer_with_defaults(layer: object, defaults: _LayerDefaults) -> object:
    """One layer with each value it states none of taken from the table; a factor has no type."""
    if not isinstance(layer, dict):
        return layer
    filled = {"encoding_mode": defaults.encoding_mode, "required": defaults.required, **layer}
    if filled["encoding_mode"] == "numeric":
        filled.setdefault("type", defaults.type)
    return filled


def _composed_column(entry: object, prefix: object, level: JsonDict, defaults: JsonDict) -> object:
    """One column with its prefix applied, and its sequence references or defaults filled in."""
    column = _prefixed(entry, prefix)
    if not isinstance(column, dict):
        return column
    how = column.get("how")
    if how is None:
        return {**defaults, **column}
    if how in {"proforma_sequence", "stripped_sequence"} and "syntax" not in column:
        column["syntax"] = _only_name(level, "sequence_syntax", column)
    if how == "proforma_sequence" and "modification_map" not in column:
        column["modification_map"] = _only_name(level, "modification_maps", column)
    return column


def _prefixed(entry: object, prefix: object) -> object:
    """A copy of one column or long-table layer whose ``source`` carries its table prefix."""
    if not isinstance(entry, dict):
        return entry
    composed = dict(entry)
    declared = "source_prefix" in composed
    own = composed.pop("source_prefix") if declared else prefix
    source = composed.get("source")
    if own is None or (source is None and not declared):
        return composed
    if not isinstance(own, str) or not isinstance(source, str):
        raise ValueError(f"{composed.get('name')!r}: source_prefix applies to a text source")
    composed["source"] = own + source
    return composed


def _wide_layer(entry: object, prefix: object, pattern: object) -> object:
    """A copy of one wide layer whose ``source`` is the regex naming each sample's column.

    A source with ``{sample}`` is the column header as text: ``pattern`` stands in for the
    placeholder, and the text before and after it, the table prefix first, matches literally.
    Any other source is a regex already, and an escaped prefix follows its leading ``^``.
    """
    if not isinstance(entry, dict):
        return entry
    composed = dict(entry)
    own = composed.pop("source_prefix") if "source_prefix" in composed else prefix
    source = composed.get("source")
    name = composed.get("name")
    if own is not None and not isinstance(own, str):
        raise ValueError(f"layer {name!r}: source_prefix is text")
    if not isinstance(source, str):
        return composed
    if _SAMPLE in source:
        before, *after = source.split(_SAMPLE)
        if len(after) != 1:
            raise ValueError(f"layer {name!r}: a wide source names {_SAMPLE} once")
        if not isinstance(pattern, str):
            raise ValueError("sample_pattern is a regex text")
        composed["source"] = f"^{_literal((own or '') + before)}{pattern}{_literal(after[0])}$"
        return composed
    if own is None:
        return composed
    if not source.startswith("^"):
        raise ValueError(f"layer {name!r}: a wide source under source_prefix starts with '^'")
    composed["source"] = "^" + _literal(own) + source[1:]
    return composed


def _literal(text: str) -> str:
    """``text`` as a regex matching it literally; a space stays a space outside verbose mode."""
    return re.escape(text).replace("\\ ", " ")


def _only_name(level: JsonDict, registry: str, column: JsonDict) -> str:
    """The one name ``registry`` declares, for a sequence column that names none."""
    declared = level.get(registry)
    names = list(declared) if isinstance(declared, dict) else []
    if len(names) != 1:
        raise ValueError(
            f"column {column.get('name')!r} names no {registry} entry, and the level declares "
            f"{len(names)}, {names}; name one"
        )
    return str(names[0])


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
    """Merge a measurements block: nested repeated_cells key-wise, layers concatenated."""
    return _merge_blocks(base, level, mappings=("repeated_cells",), sequences=("layers",))


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
