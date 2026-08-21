"""The authored input schema adds vendor facts to one shared physical-format policy."""

from __future__ import annotations

import json

from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    NumericTextFormat,
    ParquetFormatContract,
)
from apb2.parserV2.vendor_parse_rules.schema.base_formats import (
    DELIMITED_BASE_FORMATS,
    PARQUET_EXTENSIONS,
)
from parserV2.fixtures import document_pairs

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())

EXPECTED_EXTENSIONS = {
    "alphadia/v1_10": [".tsv"],
    "alphadia/v1_12": [".tsv"],
    "alphadia/v2": [".parquet"],
    "alphapept": [".csv"],
    "diann/v1": [".tsv"],
    "diann/v2": [".parquet"],
    "fragpipe": [".tsv"],
    "maxquant": [".txt"],
    "peaks": [".csv"],
    "sage": [".tsv"],
    "spectronaut": [".tsv"],
    "wombat": [".csv"],
}


def test_shared_base_formats_are_declared_once() -> None:
    assert set(DELIMITED_BASE_FORMATS) == {".tsv", ".txt", ".csv"}
    assert DELIMITED_BASE_FORMATS[".tsv"].delimiter == "\t"
    assert DELIMITED_BASE_FORMATS[".txt"].delimiter == "\t"
    assert DELIMITED_BASE_FORMATS[".csv"].delimiter == ","
    for base in DELIMITED_BASE_FORMATS.values():
        assert base.encoding == "utf8"
        assert base.decimal_mark == "."
        assert base.thousands_marks == ()
    assert frozenset({".parquet"}) == PARQUET_EXTENSIONS


def test_each_rule_authors_only_its_reviewed_extension_hints() -> None:
    for pair in document_pairs():
        payload = json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))
        assert payload["input"]["extensions"] == EXPECTED_EXTENSIONS[pair.key]
        assert "formats" not in payload["input"]
        assert "source" not in payload["input"]


def test_only_maxquant_authors_an_exact_folder_file_name() -> None:
    authored = {
        pair.key: json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))["input"].get(
            "file_name"
        )
        for pair in document_pairs()
    }

    assert {key: name for key, name in authored.items() if name is not None} == {
        "maxquant": "evidence.txt"
    }


def test_only_spectronaut_enables_physical_format_detection() -> None:
    detected: dict[str, set[str]] = {}
    for pair in document_pairs():
        declared = json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))["input"]
        overrides = {name for name in ("delimiter", "numbers") if name in declared}
        if overrides:
            detected[pair.key] = overrides

    assert detected == {"spectronaut": {"delimiter", "numbers"}}


def test_facade_applies_shared_defaults_to_an_ordinary_tsv_rule() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "diann/v1")
    contract = pair.first_admitted_facade().working_parameters.input

    assert contract.file_name is None
    assert contract.formats == (
        DelimitedFormatContract(
            extensions=(".tsv",),
            encoding="utf8",
            quote_char='"',
            delimiter_candidates=("\t",),
            number_format_candidates=(DOT,),
        ),
    )


def test_facade_projects_parquet_without_a_text_dialect() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "diann/v2")
    contract = pair.first_admitted_facade().working_parameters.input

    assert contract.formats == (ParquetFormatContract(extensions=(".parquet",)),)


def test_spectronaut_detection_changes_only_its_projected_contract() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "spectronaut")
    contract = pair.first_admitted_facade().working_parameters.input
    (physical,) = contract.formats

    assert isinstance(physical, DelimitedFormatContract)
    assert physical.delimiter_candidates == ("\t", ";", ",")
    assert physical.number_format_candidates == (
        NumericTextFormat(decimal_mark=".", thousands_marks=(",", " ")),
        NumericTextFormat(decimal_mark=",", thousands_marks=(".", " ")),
    )
