"""Minimal schema-0.8 documents for the rule shapes no packaged vendor exercises.

The packaged set covers most of the architecture, but not all of it: no document declares a
column-labelled fragment table, an optional column whose absence blocks a chain, or a wide
required layer that matches only non-primary samples. Building the smallest document that
declares one is how those paths get tested without inventing a vendor.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from apb2.parserV2.parse_quant.data.parsed import ParsedLevel
from apb2.parserV2.parse_quant.parameters.source import LevelSourceTable
from apb2.parserV2.parse_quant.parser import ParseStrategy
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    SearchParameterEvidence,
    make_rule_document,
)
from apb2.parserV2.vendor_parse_rules.schema.base import SCHEMA_VERSION, QuantificationLevel

NO_EVIDENCE = SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)
FLAGS: tuple[dict[str, Any], ...] = (
    {"name": "apb_Decoy", "how": "unmarked"},
    {"name": "apb_Contaminant", "how": "unmarked"},
)
"""The vendor markings every rule declares once; a synthetic vendor writes none."""

LONG_DEFAULTS: dict[str, Any] = {
    "var": {"type": "string", "required": True, "per_feature": "first"},
    "obs": {"type": "string", "required": True},
    "layers": {"encoding_mode": "numeric", "type": "number", "required": False},
}
"""A long table's ``base.defaults``; a test's entries state only the values that differ."""
WIDE_DEFAULTS: dict[str, Any] = {key: LONG_DEFAULTS[key] for key in ("var", "layers")}
"""A wide table's ``base.defaults``: it has no obs columns."""


def without_flags(computers: tuple[Any, ...]) -> tuple[Any, ...]:
    """The planned computations a test declared, leaving out the FLAGS every rule carries."""
    names = {flag["name"] for flag in FLAGS}
    return tuple(computer for computer in computers if computer.name not in names)


def with_flags(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the undeclared FLAGS after the sourced entries, keeping both ends' indexes."""
    declared = {entry["name"] for entry in entries}
    at = next((index for index, entry in enumerate(entries) if "source" not in entry), len(entries))
    missing = [dict(flag) for flag in FLAGS if flag["name"] not in declared]
    return [*entries[:at], *missing, *entries[at:]]


def parse_table(strategy: ParseStrategy, table: LevelSourceTable) -> ParsedLevel:
    """Parse one read table as the runtime does: decompose it, then parse the raw level."""
    return strategy.parse_decomposed(strategy.decompose(table))


def plan_snapshot(strategy: ParseStrategy) -> dict[str, Any]:
    """Inspect saved decisions without retaining another runtime configuration graph."""
    encoded = strategy.provenance["plan_json"]
    assert isinstance(encoded, str)
    return json.loads(encoded)


def _column_entries(
    selected: dict[str, str],
    optional: dict[str, str] | None = None,
    types: dict[str, str] | None = None,
    computed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build entry-shaped columns while allowing a computation to replace its source."""
    logical_types = types or {}
    computers = {entry["name"]: entry for entry in computed or []}
    entries: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for declarations, required in ((selected, True), (optional or {}, False)):
        for name, source in declarations.items():
            selected_names.add(name)
            entry: dict[str, Any] = {"name": name, "source": source}
            if logical_type := logical_types.get(name):
                entry["type"] = logical_type
            if not required:
                entry["required"] = False
            if computer := computers.get(name):
                entry.update({key: value for key, value in computer.items() if key != "name"})
            entries.append(entry)
    entries.extend(entry for entry in computed or [] if entry["name"] not in selected_names)
    return entries


def document(
    *,
    shape: str,
    base: dict[str, Any],
    levels: dict[str, dict[str, Any]],
    defaults: bool = True,
) -> RuleDocument:
    """One in-memory rules.json, validated exactly as a packaged file would be.

    A base without them gets the shape's defaults, unless ``defaults`` is false, and
    repeated cells that are an error where no fragment says otherwise.
    """
    declared: dict[str, Any] = {"shape": shape, "extensions": [".tsv"]}
    return make_rule_document(
        Path("synthetic/rules.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "file_version": "1",
            "software_name": "Synthetic",
            "software_version_pattern": "^1$",
            "hierarchy": "lfq",
            "parameter_file": "required",
            "tables": [
                {
                    "input": declared,
                    "base": complete_base(shape, base, levels, defaults),
                    "levels": levels,
                }
            ],
        },
    )


def complete_base(
    shape: str, base: dict[str, Any], levels: dict[str, dict[str, Any]], defaults: bool = True
) -> dict[str, Any]:
    """``base`` with the values a test leaves to the table stated: defaults, repeated cells."""
    completed = dict(base)
    if defaults and "defaults" not in completed:
        chosen = LONG_DEFAULTS if shape == "long" else WIDE_DEFAULTS
        completed = {"defaults": copy.deepcopy(chosen), **completed}
    fragments = [completed, *levels.values()]
    if not any("repeated_cells" in fragment.get("measurements", {}) for fragment in fragments):
        measurements = completed.get("measurements", {})
        completed["measurements"] = {"repeated_cells": {"mode": "error"}, **measurements}
    return completed


def facade(
    built: RuleDocument,
    level: QuantificationLevel = "peptidoform_ion",
    evidence: SearchParameterEvidence = NO_EVIDENCE,
) -> ParseRuleFacade:
    return ParseRuleFacade(built, level, evidence)


def long_document(
    *,
    obs_select: dict[str, str],
    var_select: dict[str, str],
    var_optional: dict[str, str] | None = None,
    var_types: dict[str, str] | None = None,
    computed: list[dict[str, Any]] | None = None,
    var_keys: list[str] | None = None,
    layers: list[dict[str, Any]] | None = None,
    primary_layer: str = "Quantity",
    duplicates: str = "error",
    rank_by: str | None = None,
    rank_best: str = "highest",
    rank_rows: dict[str, Any] | None = None,
) -> RuleDocument:
    """A long rule with one obs key and whatever var declarations a test needs.

    ``rank_by`` names the layer a ``keep_best`` rule ranks repeated cells by, and
    ``rank_best`` its winning value; ``rank_rows`` the var column a level ranks a feature's
    repeated rows by.
    """
    return document(
        shape="long",
        base={
            "axis": {
                "obs_keys": list(obs_select),
                "var_keys": var_keys or list(var_select)[:1],
            },
            "columns": {
                "obs": _column_entries(obs_select),
                "var": with_flags(_column_entries(var_select, var_optional, var_types, computed)),
            },
            "measurements": {
                "primary_layer": primary_layer,
                "repeated_cells": {"mode": duplicates}
                | ({"by": rank_by, "best": rank_best} if rank_by else {}),
                "layers": layers or [{"name": "Quantity", "source": "Quantity"}],
            },
            **({"rank_rows": rank_rows} if rank_rows is not None else {}),
        },
        levels={"peptidoform_ion": {}},
    )


def wide_document(
    *,
    var_select: dict[str, str],
    layers: list[dict[str, Any]],
    primary_layer: str,
    var_keys: list[str] | None = None,
    obs_keys: list[str] | None = None,
    sample_layer: str | None = None,
) -> RuleDocument:
    """A wide rule whose observation axis comes from its sample layer's header captures."""
    return document(
        shape="wide",
        base={
            "axis": {
                "obs_keys": obs_keys or ["sample"],
                "var_keys": var_keys or list(var_select)[:1],
            },
            "columns": {"var": with_flags(_column_entries(var_select))},
            "measurements": {
                "primary_layer": primary_layer,
                **({"sample_layer": sample_layer} if sample_layer is not None else {}),
                "layers": layers,
            },
        },
        levels={"peptidoform_ion": {}},
    )
