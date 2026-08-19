"""The entry point of this package: ``load_document(path)``. Nothing else here is called.

This folder deserializes rules.json, and that is all it does. Load a file and you have the
whole of it: what software it describes, which quantification levels it
declares, whether a header looks like its output, and — the one thing anyone actually wants —
``rule(level, parameters)``, the composed and validated rule for one level. ``declared(level)``
is the same composition for a caller that has no evidence at all and therefore no gate to
satisfy: what the file says, which is what recognizing a vendor from headers asks.

``PACKAGED`` is the list of files apb2 ships, as paths: data, not a function.

The pydantic shell that parses the file is private (``_Shell``): a document answers questions,
and a schema model may not, so exactly one of them is public and it is the one with methods.
``Recognition`` is re-exported here for the same reason — it is the type of something ``Rule``
hands back, so a caller must be able to name it without reaching past this module.

Two rules.json keys read search parameters, and ``rule()`` is where both act:
``requires_search_parameters`` gates the level (Sage declares one level per
``combine_charge_states`` setting, so without the parameters there is no telling which of
them a file is) and ``search_parameter_overrides`` patches ``axis.x_layer`` (DIA-NN's
acquisition mode decides which column carries the quantity). The patch goes into the payload
*before* validation, so a rule is validated once and is applicable by construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import cast

from pydantic import Field

from apb2.errors import RuleNotApplicable
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules._recognition import Recognition as Recognition  # re-exported
from apb2.vendor_parse_rules._recognition import recognition_for
from apb2.vendor_parse_rules.model import (
    ConditionValue,
    LongRule,
    ModelBase,
    QuantificationLevel,
    TableShape,
    WideRule,
    validate_rule,
)


def _packaged() -> tuple[Path, ...]:
    root = Path(str(resources.files("apb2.vendor_parse_rules.documents")))
    return tuple(sorted(set(root.glob("*/rules.json")) | set(root.glob("*/v*/rules.json"))))


PACKAGED: tuple[Path, ...] = _packaged()
"""Every rules.json apb2 ships, in stable path order."""


class Rule:
    """One composed rules.json level: the validated declaration and its header recognition.

    Both halves travel together because every consumer needs both, and rebuilding one from
    the other is how two answers to one question start to drift.
    """

    def __init__(self, config: LongRule | WideRule) -> None:
        self.config = config
        self.recognition: Recognition = recognition_for(config)


def load_document(path: Path) -> Document:
    """Read one rules.json. The single entry point of this package.

    Raises when the file is not a readable rules document, so a ``Document`` in hand means
    the shell parsed; a level's blocks are validated when that level is composed, which is
    the only place they can be — a level is validated as a whole rule.
    """
    return Document(_load_shell(path))


class Document:
    """One rules.json: what it describes, which levels it declares, and their rules."""

    def __init__(self, shell: _Shell) -> None:
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

    def rule(self, level: QuantificationLevel, parameters: Parameters | None) -> Rule:
        """The rule this file declares for ``level``, as the evidence selects it.

        Raises ``RuleNotApplicable`` — naming what went wrong — when the file has no such
        level, or when its parameter gate excludes this evidence.
        """
        payload = self._payload_for(level)
        self._require_gate_admits(payload.get("requires_search_parameters"), parameters, level)
        return Rule(validate_rule(_with_override(payload, parameters)))

    def declared(self, level: QuantificationLevel) -> Rule:
        """The rule this file *declares* for ``level``, gates ignored.

        For callers that have no evidence and cannot have any: recognizing a vendor from
        column headers, and the test that sweeps every packaged level.
        """
        return Rule(validate_rule(self._payload_for(level)))

    def matches(self, headers: Iterable[str]) -> bool:
        """Whether any level this file declares recognizes these headers.

        Gates are not consulted: this answers "does this look like that vendor's export",
        which is what a caller asks when it has no parameters yet — the question that decides
        which vendor's parameter parser to run.
        """
        header_set = frozenset(headers)
        return any(self.declared(level).recognition.matches(header_set) for level in self.levels)

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
        self, gate: object, parameters: Parameters | None, level: QuantificationLevel
    ) -> None:
        """Raise unless every gated parameter holds; the three outcomes read differently."""
        if not gate:
            return
        if not isinstance(gate, dict):
            raise ValueError(f"requires_search_parameters must be an object; got {gate!r}")
        label = f"{self.software_name!r} level {level!r}"
        if parameters is None:
            raise RuleNotApplicable(
                f"{label} requires search parameters {gate}, and none were supplied"
            )
        if _matches(gate, parameters):
            return
        observed = parameters.model_dump(mode="json", include=set(gate))
        raise RuleNotApplicable(
            f"{label} requires search parameters {gate}, but the supplied values are {observed}"
        )


def _with_override(payload: JsonDict, parameters: Parameters | None) -> JsonDict:
    """Patch ``axis.x_layer`` when the evidence matches an override; validation follows."""
    declared = payload.get("search_parameter_overrides")
    if parameters is None or not isinstance(declared, list) or not declared:
        return payload
    x_layers = {
        override["x_layer"]
        for override in declared
        if isinstance(override, dict) and _matches(override["when_search_parameters"], parameters)
    }
    if not x_layers:
        return payload
    if len(x_layers) > 1:
        raise ValueError(
            f"matching search-parameter overrides disagree on x_layer: {sorted(map(str, x_layers))}"
        )
    axis = payload["axis"]
    if not isinstance(axis, dict):
        raise ValueError("axis must be an object to carry an x_layer override")
    return {**payload, "axis": {**axis, "x_layer": next(iter(x_layers))}}


def _matches(condition: object, parameters: Parameters) -> bool:
    """Whether every declared parameter equality holds for the parsed values."""
    if not isinstance(condition, dict):
        raise ValueError(f"search-parameter condition must be an object; got {condition!r}")
    declared: dict[str, ConditionValue] = condition
    return parameters.model_dump(mode="json", include=set(declared)) == declared


# ------------------------------------------ the file: its shell, and the base-level merge


type JsonDict = dict[str, object]
"""A raw rules.json fragment: dicts merge without models, presence is key membership."""


class _Input(ModelBase):
    shape: TableShape


class _Shell(ModelBase):
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
    input: _Input
    base: JsonDict
    levels: dict[QuantificationLevel, JsonDict] = Field(min_length=1)


def _load_shell(path: Path) -> _Shell:
    """Parse the shell; each level's blocks are validated later, when a level is composed."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: document must be a JSON object")
    return _Shell.model_validate({"path": path, **cast("JsonDict", raw)})


def _merge_fragments(base: JsonDict, level: JsonDict) -> JsonDict:
    """Merge one level fragment over a base fragment.

    Columns descend one additional level; all other declared merge shapes are top-level.
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
        sequences=("layers", "search_parameter_overrides"),
    )
    if "columns" not in base and "columns" not in level:
        return merged
    base_columns = base.get("columns", {})
    level_columns = level.get("columns", {})
    if not isinstance(base_columns, dict) or not isinstance(level_columns, dict):
        return merged
    merged["columns"] = _merge_columns(base_columns, level_columns)
    return merged


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
