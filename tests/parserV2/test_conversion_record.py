"""The conversion record: provenance, complete evidence and its summary."""

from __future__ import annotations

from apb2.parserV2.parse_quant.conversion_record import ConversionEvidence
from apb2.parserV2.parse_quant.data.parsed import JsonValue


def statuses(record: dict[str, JsonValue]) -> dict[str, tuple[JsonValue, JsonValue]]:
    summary = record["summary"]
    assert isinstance(summary, list)
    result: dict[str, tuple[JsonValue, JsonValue]] = {}
    for entry in summary:
        assert isinstance(entry, dict)
        result[str(entry["name"])] = (entry["value"], entry["status"])
    return result


def test_findings_need_attention_and_their_evidence_stays_in_the_result() -> None:
    evidence = ConversionEvidence(
        unknown_mod_tokens=("Mystery@M", "Other@C"),
        unreadable_numeric={
            "Intensity": {"cell_count": 3, "distinct_token_count": 1, "examples": ["-"]},
            "Q.Value": {"cell_count": 2, "distinct_token_count": 1, "examples": ["NA"]},
        },
        effectively_empty={"MS2": {"occupancy": 0.0}},
    )

    record = evidence.record({"rule_json": "{}"})

    assert record["provenance"] == {"rule_json": "{}"}
    assert record["result"] == {
        "unknown_mod_tokens": ["Mystery@M", "Other@C"],
        "layer_diagnostics": {
            "unreadable_numeric": {
                "Intensity": {"cell_count": 3, "distinct_token_count": 1, "examples": ["-"]},
                "Q.Value": {"cell_count": 2, "distinct_token_count": 1, "examples": ["NA"]},
            },
            "effectively_empty": {"MS2": {"occupancy": 0.0}},
        },
    }
    assert statuses(record) == {
        "unknown_modification_tokens": (2, "attention"),
        "unreadable_cells": (5, "attention"),
        "effectively_empty_layers": (1, "attention"),
    }


def test_a_clean_level_reports_explicit_zeros_and_an_unjudged_check_as_not_checked() -> None:
    record = ConversionEvidence((), {}, None).record({})

    assert statuses(record) == {
        "unknown_modification_tokens": (0, "ok"),
        "unreadable_cells": (0, "ok"),
        "effectively_empty_layers": (None, "not_checked"),
    }
