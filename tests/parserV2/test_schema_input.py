"""The authored input schema adds vendor facts to one shared physical-format policy."""

from __future__ import annotations

import json

from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    ExcelFormatContract,
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
    "diann/v1_8": [".tsv", ".txt", ".parquet"],
    "diann/v1_7": [".tsv"],
    "diann/v2": [".parquet"],
    "fragpipe": [".tsv"],
    "i2masschroq": [".tsv", ".txt"],
    "maxquant": [".txt"],
    "msangel": [".txt", ".xlsx"],
    "peaks": [".csv"],
    "prolinestudio": [".txt", ".xlsx"],
    "quantms": [".csv"],
    "sage": [".tsv"],
    "spectronaut": [".tsv"],
    "spectronaut/v15": [".tsv"],
    "spectronaut/v21": [".tsv"],
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
        for table in payload["tables"]:
            assert table["input"]["extensions"] == EXPECTED_EXTENSIONS[pair.key]
            assert "formats" not in table["input"]
            assert "source" not in table["input"]


def test_prepared_rules_do_not_bind_levels_to_independent_files() -> None:
    authored = {
        (pair.key, level): table["input"]["file_name"]
        for pair in document_pairs()
        for table in json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))["tables"]
        if "prepare" in table and "file_name" in table["input"]
        for level in table["levels"]
    }

    assert authored == {}


def test_only_spectronaut_enables_physical_format_detection() -> None:
    detected: dict[str, set[str]] = {}
    for pair in document_pairs():
        for table in json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))["tables"]:
            declared = table["input"]
            overrides = {name for name in ("delimiter", "numbers", "encoding") if name in declared}
            if overrides:
                detected.setdefault(pair.key, set()).update(overrides)

    assert detected == {
        "spectronaut": {"delimiter", "numbers"},
        "spectronaut/v15": {"delimiter", "numbers", "encoding"},
        "spectronaut/v21": {"delimiter", "numbers"},
    }


def test_facade_applies_shared_defaults_to_each_diann_1_8_input_format() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "diann/v1_8")
    contract = pair.first_admitted_facade().working_parameters.input

    assert contract.file_name is None
    assert contract.formats == (
        DelimitedFormatContract(
            extensions=(".tsv",),
            encoding_candidates=("utf8",),
            quote_char='"',
            delimiter_candidates=("\t",),
            number_format_candidates=(DOT,),
        ),
        DelimitedFormatContract(
            extensions=(".txt",),
            encoding_candidates=("utf8",),
            quote_char='"',
            delimiter_candidates=("\t",),
            number_format_candidates=(DOT,),
        ),
        ParquetFormatContract(extensions=(".parquet",)),
    )


def test_facade_projects_parquet_without_a_text_dialect() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "diann/v2")
    contract = pair.first_admitted_facade().working_parameters.input

    assert contract.formats == (ParquetFormatContract(extensions=(".parquet",)),)


def test_facade_projects_a_named_workbook_sheet() -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "prolinestudio")
    contract = pair.first_admitted_facade().working_parameters.input

    assert contract.formats == (
        ExcelFormatContract(extensions=(".txt",), sheet_name="Quantified peptide ions"),
        ExcelFormatContract(extensions=(".xlsx",), sheet_name="Quantified peptide ions"),
    )


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
