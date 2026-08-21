"""What one rules.json answers: its levels, their effective rules, and header recognition.

``RuleDocument`` is the public API over one loaded file. It retains its validated ``_shell``
and reads through it rather than copying its members into a second field set, because two
copies of one fact are how two answers to one question start to drift.

Two rules.json keys read search parameters, and ``rule()`` is where both act:
``requires_search_parameters`` gates the level (Sage declares one level per
``combine_charge_states`` setting, so without the evidence there is no telling which of them
a file is) and ``search_parameter_overrides`` patches ``measurements.primary_layer`` (DIA-NN's
acquisition mode decides which column carries the quantity). The patch goes into the payload
*before* validation, so a rule is validated once and is applicable by construction.

``SearchParameterEvidence`` is deliberately smaller than any parameter-file model: schema 0.3
permits exactly two condition fields, this package owns that vocabulary, and the outer
application translates its own parameter model into this value before entering Parser V2.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema_axis import ColumnGroup
from apb2.parserV2.vendor_parse_rules.schema_base import (
    SCHEMA_VERSION,
    ModelBase,
    QuantificationLevel,
)
from apb2.parserV2.vendor_parse_rules.schema_base_modifications import modification_outputs
from apb2.parserV2.vendor_parse_rules.schema_fragments import ColumnLabeledFragments
from apb2.parserV2.vendor_parse_rules.schema_input import Input
from apb2.parserV2.vendor_parse_rules.schema_measurements import layer_required
from apb2.parserV2.vendor_parse_rules.schema_parameters import (
    ConditionValue,
    SearchParameterField,
)
from apb2.parserV2.vendor_parse_rules.schema_rule import (
    LongRule,
    WideRule,
    validate_rule,
)

_SYNTHESIZED = frozenset({"stripped_sequence"})

type AxisName = Literal["obs", "var"]

type JsonDict = dict[str, object]
"""A raw rules.json fragment: dicts merge without models, presence is key membership.

It must not become a wrapper class whose methods merely forward ordinary dict
operations. Malformed values ride the merge untouched and are reported, with their
authored paths, at the single effective-rule validation boundary.
"""
type MergeBlock = Callable[[JsonDict, JsonDict], JsonDict]


class RuleNotApplicable(ValueError):
    """This rule does not apply to what the caller has — try another level.

    The skip contract: ``compile_parsers`` catches this to move to the next quantification
    level, so anything meaning "not this level" must be this class or a subclass, and
    anything meaning "the caller is wrong" must not be.
    """


@dataclass(frozen=True, slots=True)
class SearchParameterEvidence:
    """The complete parameter vocabulary permitted in schema-0.3 conditions."""

    acquisition_method: Literal["DDA", "DIA", "unknown"]
    combine_charge_states: bool | None

    def observed(self, requested: Iterable[SearchParameterField]) -> dict[str, ConditionValue]:
        """The requested fields' values, for comparison against a declared condition."""
        known = {field.name for field in fields(self)}
        return {name: getattr(self, name) for name in requested if name in known}


# ------------------------------------------------------------------------ header recognition


def synthesized_columns(rule: LongRule | WideRule) -> frozenset[str]:
    """Columns Parser V2 creates itself, which must never be required of the input."""
    if rule.modifications is None:
        return _SYNTHESIZED
    return _SYNTHESIZED | modification_outputs(rule.modifications)


class LongRecognition:
    """Header recognition for a long rule: every source is an exact column name."""

    __slots__ = ("_rule", "required_headers")

    def __init__(self, rule: LongRule) -> None:
        self._rule = rule
        expected = set(rule.columns.obs.select.values()) | set(rule.columns.var.select.values())
        expected.update(
            layer.source
            for layer in rule.measurements.layers
            if layer_required(rule.measurements.primary_layer, layer)
        )
        self.required_headers: frozenset[str] = frozenset(expected - synthesized_columns(rule))

    def column_groups(self) -> tuple[tuple[AxisName, ColumnGroup], ...]:
        return (("obs", self._rule.columns.obs), ("var", self._rule.columns.var))

    def layer_source_columns(self, header: Iterable[str]) -> set[str]:
        """Layer sources are exact column names in a long rule."""
        del header
        return {layer.source for layer in self._rule.measurements.layers}

    def matches(self, headers: Iterable[str]) -> bool:
        """Whether raw input headers satisfy the rule's required sources."""
        header_set = set(headers)
        if not _fragment_label_present(self._rule, header_set):
            return False
        return self.required_headers.issubset(header_set)


class WideRecognition:
    """Header recognition for a wide rule: layer sources are sample-capturing regexes."""

    __slots__ = ("_required_var", "_rule")

    def __init__(self, rule: WideRule) -> None:
        self._rule = rule
        self._required_var = frozenset(
            set(rule.columns.var.select.values()) - synthesized_columns(rule)
        )

    def column_groups(self) -> tuple[tuple[AxisName, ColumnGroup], ...]:
        return (("var", self._rule.columns.var),)

    def layer_source_columns(self, header: Iterable[str]) -> set[str]:
        """Expand each layer's header regex over the real header."""
        names = list(header)
        matched: set[str] = set()
        for layer in self._rule.measurements.layers:
            compiled = re.compile(layer.source)
            matched.update(name for name in names if compiled.match(name) is not None)
        return matched

    def matches(self, headers: Iterable[str]) -> bool:
        """Whether raw input headers satisfy the rule's required sources."""
        header_set = set(headers)
        if not _fragment_label_present(self._rule, header_set):
            return False
        for layer in self._rule.measurements.layers:
            if layer_required(self._rule.measurements.primary_layer, layer) and not any(
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


# ------------------------------------------------------------------------ the effective rule


@dataclass(frozen=True, slots=True)
class EffectiveRule:
    """One level's validated declaration, its document's input policy, and its recognition.

    All three travel together so projection is a total function of one value: the input
    declaration is the same for every level of a document and is not copied onto
    ``RuleDocument``, and rebuilding the recognition elsewhere is how two answers to one
    question start to drift.
    """

    input: Input
    declaration: LongRule | WideRule
    recognition: Recognition


# ------------------------------------------------------------------ the document shell


class _RuleDocumentSchema(ModelBase):
    """One parsed rules.json, as a shell around raw dict fragments — private on purpose.

    The fragments stay raw dicts through the base-times-level merge — merging dicts needs no
    models, presence is key membership — and cross the single typed boundary,
    ``validate_rule``, only once composed. That boundary is also the only validator:
    unknown keys and wrong types ride through the merge and are reported there with paths.
    """

    path: Path
    schema_version: str
    file_version: str
    software_name: str
    software_version_pattern: str
    input: Input
    base: JsonDict
    levels: dict[QuantificationLevel, JsonDict] = Field(min_length=1)

    @model_validator(mode="after")
    def _is_this_generation(self) -> _RuleDocumentSchema:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path}: schema_version must be {SCHEMA_VERSION!r} for this rule "
                f"package; got {self.schema_version!r}"
            )
        return self


class RuleDocument:
    """One rules.json: what it describes, which levels it declares, and their rules."""

    __slots__ = ("_shell",)

    def __init__(self, shell: _RuleDocumentSchema) -> None:
        self._shell = shell

    @property
    def path(self) -> Path:
        return self._shell.path

    @property
    def software_name(self) -> str:
        return self._shell.software_name

    @property
    def software_version_pattern(self) -> str:
        return self._shell.software_version_pattern

    @property
    def levels(self) -> tuple[QuantificationLevel, ...]:
        return tuple(self._shell.levels)

    def declared(self, level: QuantificationLevel) -> EffectiveRule:
        """The rule this file *declares* for ``level``, gates and overrides ignored.

        For callers that have no evidence and cannot have any: recognizing a vendor from
        column headers, and the sweep over every packaged level.
        """
        return self._effective(self._payload_for(level))

    def rule(
        self,
        level: QuantificationLevel,
        evidence: SearchParameterEvidence,
    ) -> EffectiveRule:
        """The rule this file declares for ``level``, as the evidence selects it.

        Raises ``RuleNotApplicable`` — naming what went wrong — when the file has no such
        level, or when its parameter gate excludes this evidence.
        """
        payload = self._payload_for(level)
        declared = self._effective(payload)
        self._require_gate_admits(
            declared.declaration.requires_search_parameters,
            evidence,
            level,
        )
        patched = _with_primary_layer_override(payload, evidence)
        return declared if patched is payload else self._effective(patched)

    def matches(self, headers: Iterable[str]) -> bool:
        """Whether any level this file declares recognizes these headers.

        Gates are not consulted: this answers "does this look like that vendor's export",
        which is what a caller asks when it has no parameters yet — the question that decides
        which vendor's parameter parser to run.
        """
        header_set = frozenset(headers)
        return any(self.declared(level).recognition.matches(header_set) for level in self.levels)

    def _effective(self, payload: JsonDict) -> EffectiveRule:
        declaration = validate_rule(payload)
        return EffectiveRule(
            input=self._shell.input,
            declaration=declaration,
            recognition=recognition_for(declaration),
        )

    def _payload_for(self, level: QuantificationLevel) -> JsonDict:
        """Compose one declared level over the common document base."""
        try:
            level_fragment = self._shell.levels[level]
        except KeyError as error:
            raise RuleNotApplicable(
                f"{self.path} has no level {level!r}; available: {sorted(self.levels)}"
            ) from error
        return {
            "schema_version": self._shell.schema_version,
            "file_version": self._shell.file_version,
            "software_name": self._shell.software_name,
            "software_version_pattern": self._shell.software_version_pattern,
            "quantification_level": level,
            "shape": self._shell.input.shape,
            **_merge_fragments(self._shell.base, level_fragment),
        }

    def _require_gate_admits(
        self,
        gate: object,
        evidence: SearchParameterEvidence,
        level: QuantificationLevel,
    ) -> None:
        """Raise unless every gated parameter holds; the two outcomes read differently."""
        if not gate:
            return
        if not isinstance(gate, dict):
            raise ValueError(f"requires_search_parameters must be an object; got {gate!r}")
        declared: dict[str, ConditionValue] = gate
        if _condition_holds(declared, evidence):
            return
        observed = evidence.observed(_condition_fields(declared))
        raise RuleNotApplicable(
            f"{self.software_name!r} level {level!r} requires search parameters {declared}, "
            f"but the supplied evidence is {observed}"
        )


def _with_primary_layer_override(payload: JsonDict, evidence: SearchParameterEvidence) -> JsonDict:
    """Patch ``measurements.primary_layer`` when the evidence matches; validation follows."""
    declared = payload.get("search_parameter_overrides")
    if not isinstance(declared, list) or not declared:
        return payload
    primary_layers = {
        override["primary_layer"]
        for override in declared
        if isinstance(override, dict)
        and _condition_holds(override["when_search_parameters"], evidence)
    }
    if not primary_layers:
        return payload
    if len(primary_layers) > 1:
        raise ValueError(
            "matching search-parameter overrides disagree on primary_layer: "
            f"{sorted(map(str, primary_layers))}"
        )
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("measurements must be an object to carry a primary_layer override")
    patched: JsonDict = {**measurements, "primary_layer": next(iter(primary_layers))}
    return {**payload, "measurements": patched}


def _condition_holds(condition: object, evidence: SearchParameterEvidence) -> bool:
    """Whether every declared parameter equality holds for the supplied evidence."""
    if not isinstance(condition, dict):
        raise ValueError(f"search-parameter condition must be an object; got {condition!r}")
    declared: dict[str, ConditionValue] = condition
    return evidence.observed(_condition_fields(declared)) == declared


def _condition_fields(condition: dict[str, ConditionValue]) -> tuple[SearchParameterField, ...]:
    """The condition's keys, typed as the finite vocabulary the schema already accepted."""
    permitted: tuple[SearchParameterField, ...] = (
        "acquisition_method",
        "combine_charge_states",
    )
    return tuple(name for name in permitted if name in condition)


def make_rule_document(path: Path, payload: JsonDict) -> RuleDocument:
    """Validate one raw rules.json payload and return the document it describes."""
    return RuleDocument(_RuleDocumentSchema.model_validate({"path": path, **payload}))


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
            "column_roles",
            "modifications",
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
        base_group = base.get(axis, {})
        level_group = level.get(axis, {})
        if not isinstance(base_group, dict) or not isinstance(level_group, dict):
            continue
        columns[axis] = _merge_blocks(
            base_group,
            level_group,
            mappings=("select", "optional_select", "types"),
            sequences=("computed",),
        )
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
