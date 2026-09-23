"""The public rule catalogue reports packaged producers, categories and versions."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from apb2.api import QuantificationLevel, get_rules
from apb2.parserV2.vendor_parse_rules.catalog import RuleCatalog

_CATALOG = Path(str(resources.files("apb2.parserV2.vendor_parse_rules"))) / "catalog.json"


def _payload() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(_CATALOG.read_text(encoding="utf-8")))


def _load_changed_catalog(tmp_path: Path, payload: dict[str, Any]) -> RuleCatalog:
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return RuleCatalog(source)


def test_get_rules_returns_software_and_version_specific_variants() -> None:
    dia = get_rules("DIA", level="ion")
    by_name = {software.software_name: software for software in dia}

    assert [software.software_name for software in dia] == sorted(
        by_name, key=lambda name: (name.casefold(), name)
    )
    assert set(by_name) == {
        "AlphaDIA",
        "DIA-NN",
        "FragPipe",
        "MaxQuant",
        "pb_custom",
        "PEAKS",
        "Spectronaut",
    }
    assert [
        (variant.rule, variant.software_version_pattern) for variant in by_name["DIA-NN"].variants
    ] == [
        ("diann/v1_7/rules.json", r"^1\.[0-7](\.|$)"),
        ("diann/v1_8/rules.json", r"^1\.[89](\.|$)"),
        ("diann/v2/rules.json", r"^2\..*"),
    ]
    assert all("ion" in variant.levels for software in dia for variant in software.variants)


def test_dda_query_keeps_only_version_with_dda_evidence() -> None:
    dda = {software.software_name: software for software in get_rules("DDA", level="ion")}

    assert set(dda) == {
        "AlphaPept",
        "DIA-NN",
        "FragPipe",
        "i2MassChroQ",
        "MaxQuant",
        "MSAngel",
        "pb_custom",
        "PEAKS",
        "ProlineStudio",
        "quantms",
        "Sage",
        "WOMBAT",
    }
    assert [variant.rule for variant in dda["DIA-NN"].variants] == ["diann/v2/rules.json"]
    assert "MetaMorpheus" not in dda
    assert "MSAID" not in dda


def test_level_filter_retains_version_level_association() -> None:
    dia_fragments = {
        software.software_name: software for software in get_rules("DIA", level="fragment")
    }

    assert set(dia_fragments) == {"DIA-NN", "Spectronaut"}
    assert {variant.rule for variant in dia_fragments["DIA-NN"].variants} == {
        "diann/v1_7/rules.json",
        "diann/v1_8/rules.json",
    }
    assert get_rules("DDA", level="fragment") == []


def test_query_rejects_unknown_category_and_level() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        get_rules("TMT")
    with pytest.raises(ValueError, match="unknown quantification level"):
        get_rules("DIA", level=cast("QuantificationLevel", "unknown"))


def test_catalogue_rejects_missing_and_nonexistent_rule_paths(tmp_path: Path) -> None:
    payload = _payload()
    payload["assignments"].pop()
    with pytest.raises(ValueError, match="missing="):
        _load_changed_catalog(tmp_path, payload)

    payload = _payload()
    payload["assignments"].append({"rule": "absent/rules.json", "categories": ["DIA"]})
    with pytest.raises(ValueError, match="unknown="):
        _load_changed_catalog(tmp_path, payload)


def test_catalogue_rejects_duplicate_paths_and_unknown_categories(tmp_path: Path) -> None:
    payload = _payload()
    payload["assignments"].append(payload["assignments"][0])
    with pytest.raises(ValidationError, match="duplicate rule paths"):
        _load_changed_catalog(tmp_path, payload)

    payload = _payload()
    payload["assignments"][0]["categories"] = ["Unknown"]
    with pytest.raises(ValidationError, match="unknown categories"):
        _load_changed_catalog(tmp_path, payload)


def test_unclassified_rule_requires_a_reason(tmp_path: Path) -> None:
    payload = _payload()
    payload["assignments"][0]["categories"] = []
    with pytest.raises(ValidationError, match="requires a reason"):
        _load_changed_catalog(tmp_path, payload)

    payload["assignments"][0]["reason"] = "no verified acquisition mode"
    catalog = _load_changed_catalog(tmp_path, payload)
    assert "AlphaDIA" in {software.software_name for software in catalog.get_rules("DIA")}
    assert [
        variant.rule for software in catalog.get_rules("DIA") for variant in software.variants
    ].count("alphadia/v1_10/rules.json") == 0
