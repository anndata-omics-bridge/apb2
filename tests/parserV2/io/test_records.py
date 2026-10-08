"""The summary and details contract every APB record shares."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.formats import write_parsed_levels
from apb2.parserV2.parse_quant.io.records import RecordContract


def entry(**fields: JsonValue) -> dict[str, JsonValue]:
    base: dict[str, JsonValue] = {
        "name": "unmatched_features",
        "label": "Unmatched peptides",
        "value": 3,
        "unit": "features",
        "status": "attention",
    }
    return {**base, **fields}


def level() -> ParsedLevel:
    return ParsedLevel(
        obs=ObsFinal(frame=pl.DataFrame({"run": ["A"]}), key_columns=("run",)),
        var=VarFinal(frame=pl.DataFrame({"feature": ["F1"]}), key_columns=("feature",)),
        primary_layer_name="Intensity",
        uns={},
        layers={"Intensity": FinalLayerTable("Intensity", pl.DataFrame({"obs_0": [1.0]}))},
        obsm={},
        varm={},
        obsp={},
        varp={},
    )


def test_a_record_holds_only_the_fields_it_uses() -> None:
    RecordContract().validate_part(
        {
            "fasta": {"summary": [entry(), entry(name="decoy_features", status="ok")]},
            "parse": {"provenance": {"rule_json": "{}"}},
            "catalog": {"identification_confidence": {"summary": [entry(layer="Q.Value")]}},
            "extension": {"anything": [1, {"summary": "not a record"}]},
        },
        "level 'ion'",
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"summary": {"name": "x"}}, "summary is not a list"),
        ({"summary": [entry(value=None, status="not_checked"), "x"]}, "not an object"),
        ({"summary": [{"name": "x", "label": "X", "value": 1, "unit": ""}]}, "optional layer"),
        ({"summary": [entry(extra=1)]}, "optional layer"),
        ({"summary": [entry(name="")]}, "has no name"),
        ({"summary": [entry(unit=3)]}, "unit is not text"),
        ({"summary": [entry(layer=3)]}, "layer is not text"),
        ({"summary": [entry(layer=None)]}, "layer is not text"),
        ({"summary": [entry(status=["ok"])]}, "unknown status"),
        ({"details": [{"slot": ["varm"], "name": "x"}]}, "slot in"),
        ({"summary": [entry(status="warning")]}, "unknown status"),
        ({"summary": [entry(value=[1])]}, "not a scalar"),
        ({"summary": [entry(value=math.nan)]}, "store null"),
        ({"summary": [entry(status="not_checked")]}, "not_checked but has a value"),
        ({"summary": [entry(), entry(value=4)]}, "repeats metric"),
        ({"details": [{"slot": "obs", "name": "x"}]}, "slot in"),
        ({"details": [{"slot": "varm"}]}, "slot in"),
        ({"named": {"summary": [entry(status="bad")]}}, "named summary"),
    ],
)
def test_a_malformed_summary_or_reference_is_rejected(
    record: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(InvalidResultError, match=message):
        RecordContract().validate_part({"tool": record}, "root")


def test_one_metric_may_repeat_for_different_layers() -> None:
    RecordContract().validate_part(
        {"tool": {"summary": [entry(layer="Intensity"), entry(layer="LFQ"), entry()]}}, "root"
    )


def test_undefined_values_are_null_with_any_status() -> None:
    RecordContract().validate_part(
        {
            "tool": {
                "summary": [
                    entry(name="a", value=None, status="attention"),
                    entry(name="b", value=None, status="ok"),
                    entry(name="c", value=None, status="not_checked"),
                ]
            }
        },
        "root",
    )


@pytest.mark.parametrize("scope", ["root", "level"])
def test_writers_reject_a_malformed_summary_before_publication(tmp_path: Path, scope: str) -> None:
    parsed = ParsedLevels(levels={"ion": level()}, uns={})
    record: dict[str, JsonValue] = {"summary": [entry(status="warning")]}
    if scope == "root":
        parsed.metadata["tool"] = record
    else:
        parsed.levels["ion"].metadata["tool"] = record
    target = tmp_path / "result.h5mu"
    with pytest.raises(InvalidResultError, match="unknown status"):
        write_parsed_levels(parsed, target)
    assert not target.exists()
