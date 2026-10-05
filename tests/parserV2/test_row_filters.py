"""Rule predicates filter physical rows before decomposition and duplicate aggregation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from apb2.api import ConversionError, ParseRuleCompiler
from apb2.parserV2.parse_quant import delimited_input, parquet_input
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    LevelReadPlan,
    NumericTextFormat,
    SourceRowFilter,
)
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.vendor_parse_rules.schema.input import Input


def test_metamorpheus_filters_before_numeric_read_and_sums_retained_peaks(tmp_path: Path) -> None:
    path = _peaks(tmp_path)
    compiler = ParseRuleCompiler.from_software(path, "MetaMorpheus")
    parsed = compiler.compile().parse().levels["ion"]

    assert parsed.var.frame["ProForma_ion"].to_list() == ["PEPTIDEK/2"]
    assert parsed.layers["Intensity"].values.to_series().to_list() == [17.5]
    assert "input_preparation" not in parsed.uns
    assert "Full Sequences Mapped" not in parsed.var.frame.columns
    assert "Full Sequences Mapped" not in parsed.obs.frame.columns


def test_a_retained_malformed_intensity_still_fails(tmp_path: Path) -> None:
    path = _peaks(tmp_path)
    frame = pl.read_csv(path, separator="\t", infer_schema=False)
    frame.with_columns(pl.lit("n/a").alias("Peak intensity")).write_csv(path, separator="\t")

    with pytest.raises(pl.exceptions.InvalidOperationError):
        ParseRuleCompiler.from_software(path, "MetaMorpheus").compile().parse()


def test_a_missing_filter_column_rejects_the_source(tmp_path: Path) -> None:
    path = _peaks(tmp_path)
    pl.read_csv(path, separator="\t", infer_schema=False).drop("Full Sequences Mapped").write_csv(
        path, separator="\t"
    )

    with pytest.raises(ConversionError, match="no matching packaged rule"):
        ParseRuleCompiler.from_software(path, "MetaMorpheus")


@pytest.mark.parametrize("encoding", ["utf8", "windows-1252"])
def test_text_filters_combine_and_preserve_the_declared_number_format(
    tmp_path: Path, encoding: str
) -> None:
    path = tmp_path / "filtered.tsv"
    path.write_text(
        "ID\tQuantity\tMapped\tSelected\n"
        "001\t10,5\t1\tyes\n"
        "002\tn/a\t2\tyes\n"
        "003\tn/a\t1\tno\n"
        "004\tn/a\t\tyes\n",
        encoding=encoding,
    )
    contract = DelimitedFormatContract(
        extensions=(".tsv",),
        encoding_candidates=("utf8", "windows-1252"),
        quote_char='"',
        delimiter_candidates=("\t",),
        number_format_candidates=(NumericTextFormat(",", ()),),
    )
    evidence = delimited_input.detected_evidence(path, contract, lambda columns: "ID" in columns)
    if encoding == "windows-1252":
        evidence = replace(evidence, encoding="windows-1252")
    read = LevelReadPlan(
        projected_columns=("ID", "Quantity"),
        text_sources=frozenset({"ID", "Mapped", "Selected"}),
        native_numeric_sources=frozenset({"Quantity"}),
        row_filters=(SourceRowFilter("Mapped", "1"), SourceRowFilter("Selected", "yes")),
    )
    frame = delimited_input.make_delimited_reader(path, evidence, read).read().frame

    assert frame.to_dict(as_series=False) == {"ID": ["001"], "Quantity": [10.5]}


@pytest.mark.parametrize("prepared", [False, True])
def test_native_sources_filter_without_exposing_predicate_columns(
    tmp_path: Path, prepared: bool
) -> None:
    frame = pl.DataFrame({"ID": ["001", "002", "003"], "Mapped": [1, 2, None]})
    read = LevelReadPlan(
        projected_columns=("ID",),
        text_sources=frozenset(),
        native_numeric_sources=frozenset(),
        row_filters=(SourceRowFilter("Mapped", "1"),),
    )
    if prepared:
        got = PreparedInputReader(frame, read, (("ID",),)).read().frame
    else:
        path = tmp_path / "filtered.parquet"
        frame.write_parquet(path)
        got = parquet_input.make_parquet_reader(path, read).read().frame

    assert got.to_dict(as_series=False) == {"ID": ["001"]}


def test_row_filters_validate_exact_textual_comparisons() -> None:
    with pytest.raises(ValidationError):
        Input.model_validate(
            {
                "shape": "long",
                "extensions": [".tsv"],
                "row_filters": [{"source": "", "equals": "1"}],
            }
        )
    with pytest.raises(ValidationError):
        Input.model_validate(
            {"shape": "long", "extensions": [".tsv"], "row_filters": [{"source": "N", "equals": 1}]}
        )


def _peaks(tmp_path: Path) -> Path:
    path = tmp_path / "renamed.tsv"
    pl.DataFrame(
        {
            "File Name": ["run1"] * 4,
            "Base Sequence": ["PEPTIDEK", "PEPTIDEK", "SEQA|SEQB", "OTHER"],
            "Full Sequence": ["PEPTIDEK", "PEPTIDEK", "SEQA|SEQB", "OTHER"],
            "Full Sequences Mapped": ["1", "1", "2", None],
            "Peak intensity": ["10.5", "7", "n/a", "n/a"],
            "Precursor Charge": ["2"] * 4,
            "Protein Group": ["P12345"] * 4,
            "Decoy Peptide": ["False"] * 4,
        }
    ).write_csv(path, separator="\t")
    return path
