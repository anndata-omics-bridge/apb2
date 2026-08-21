"""Minimal schema-0.3 documents for the rule shapes no packaged vendor exercises.

The packaged set covers most of the architecture, but not all of it: no document declares a
column-labelled fragment table, an optional column whose absence blocks a chain, or a wide
required layer that matches only non-primary samples. Building the smallest document that
declares one is how those paths get tested without inventing a vendor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    SearchParameterEvidence,
    make_rule_document,
)
from apb2.parserV2.vendor_parse_rules.schema.base import SCHEMA_VERSION, QuantificationLevel

NO_EVIDENCE = SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)


def document(
    *,
    shape: str,
    base: dict[str, Any],
    levels: dict[str, dict[str, Any]],
) -> RuleDocument:
    """One in-memory rules.json, validated exactly as a packaged file would be."""
    declared: dict[str, Any] = {"shape": shape, "extensions": [".tsv"]}
    return make_rule_document(
        Path("synthetic/rules.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "file_version": "1",
            "software_name": "Synthetic",
            "software_version_pattern": "^1$",
            "input": declared,
            "base": base,
            "levels": levels,
        },
    )


def facade(
    built: RuleDocument,
    level: QuantificationLevel = "ion",
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
) -> RuleDocument:
    """A long rule with one obs key and whatever var declarations a test needs."""
    var_group: dict[str, Any] = {"select": var_select}
    if var_optional:
        var_group["optional_select"] = var_optional
    if var_types:
        var_group["types"] = var_types
    if computed:
        var_group["computed"] = computed
    return document(
        shape="long",
        base={
            "axis": {
                "obs_keys": list(obs_select),
                "var_keys": var_keys or list(var_select)[:1],
            },
            "columns": {"obs": {"select": obs_select}, "var": var_group},
            "measurements": {
                "primary_layer": primary_layer,
                "duplicates": {"mode": duplicates},
                "layers": layers or [{"name": "Quantity", "source": "Quantity"}],
            },
        },
        levels={"ion": {}},
    )


def wide_document(
    *,
    var_select: dict[str, str],
    layers: list[dict[str, Any]],
    primary_layer: str,
    var_keys: list[str] | None = None,
    obs_keys: list[str] | None = None,
) -> RuleDocument:
    """A wide rule whose observation axis comes from its primary layer's header captures."""
    return document(
        shape="wide",
        base={
            "axis": {
                "obs_keys": obs_keys or ["sample"],
                "var_keys": var_keys or list(var_select)[:1],
            },
            "columns": {"var": {"select": var_select}},
            "measurements": {
                "primary_layer": primary_layer,
                "duplicates": {"mode": "error"},
                "layers": layers,
            },
        },
        levels={"ion": {}},
    )
