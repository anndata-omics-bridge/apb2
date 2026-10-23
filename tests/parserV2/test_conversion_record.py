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


NOTHING: dict[str, dict[str, JsonValue]] = {"obs": {}, "var": {}, "layers": {}}


def test_findings_need_attention_and_their_evidence_stays_in_the_result() -> None:
    single: dict[str, dict[str, JsonValue]] = {
        "obs": {},
        "var": {"Lib_Peptidoform_Q_Value": {"value": 0.0}},
        "layers": {"Predicted_IM": {"value": 0.0}, "IM": {"value": 0.0}},
    }
    empty: dict[str, dict[str, JsonValue]] = {
        "obs": {},
        "var": {"Genes": {"filled": 0}},
        "layers": {"MS2": {"occupancy": 0.0, "filled": 0}},
    }
    evidence = ConversionEvidence(
        unknown_mod_tokens=("Mystery@M", "Other@C"),
        unreadable_numeric={
            "Intensity": {"cell_count": 3, "distinct_token_count": 1, "examples": ["-"]},
            "Q.Value": {"cell_count": 2, "distinct_token_count": 1, "examples": ["NA"]},
        },
        single_valued=single,
        effectively_empty=empty,
        var_disagreements={"Genes": {"features": 1, "examples": [{"key": ["P1"], "values": []}]}},
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
        },
        "uninformative_fields": {"single_valued": single, "effectively_empty": empty},
        "var_disagreements": {
            "Genes": {"features": 1, "examples": [{"key": ["P1"], "values": []}]}
        },
    }
    assert statuses(record) == {
        "unknown_modification_tokens": (2, "attention"),
        "unreadable_cells": (5, "attention"),
        "single_valued_fields": (3, "attention"),
        "effectively_empty_fields": (2, "attention"),
        "var_disagreements": (1, "attention"),
    }


def test_a_clean_level_reports_explicit_zeros() -> None:
    record = ConversionEvidence((), {}, NOTHING, NOTHING, {}).record({})

    assert statuses(record) == {
        "unknown_modification_tokens": (0, "ok"),
        "unreadable_cells": (0, "ok"),
        "single_valued_fields": (0, "ok"),
        "effectively_empty_fields": (0, "ok"),
        "var_disagreements": (0, "ok"),
    }
