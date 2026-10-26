"""Root ``input_files``: one record per file read, checked against independent counts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2.compile import ParseRuleCompiler
from apb2.parserV2.parse_quant.io import formats
from parserV2.fixtures import DATA_DIR, input_files


def _expected(
    path: Path, frame: pl.DataFrame, compression: str, file_format: str
) -> dict[str, object]:
    return {
        "role": "table",
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compression": compression,
        "format": file_format,
        "nrows": frame.height,
        "ncols": frame.width,
        "columns": frame.columns,
    }


def test_a_gzipped_table_read_by_two_levels_is_recorded_once() -> None:
    path = DATA_DIR / "diann/v1_8/sample.txt.gz"
    parsed = (
        ParseRuleCompiler.from_software(
            path, "diann", requested_levels=("peptidoform_ion", "protein")
        )
        .compile()
        .parse()
    )

    frame = pl.read_csv(path, separator="\t", infer_schema=False)
    assert input_files(parsed.uns) == [_expected(path, frame, "gzip", "tsv")]


def test_a_parquet_table_records_its_schema_and_rows() -> None:
    path = DATA_DIR / "diann/v2/sample.parquet"
    parsed = (
        ParseRuleCompiler.from_software(path, "diann", requested_levels=("peptidoform_ion",))
        .compile()
        .parse()
    )

    assert input_files(parsed.uns) == [_expected(path, pl.read_parquet(path), "none", "parquet")]


@pytest.mark.parametrize("suffix", (".h5ad", ".h5mu", ".parquet", ".duckdb"))
def test_input_files_survive_every_result_format(tmp_path: Path, suffix: str) -> None:
    path = DATA_DIR / "diann/v2/sample.parquet"
    parsed = (
        ParseRuleCompiler.from_software(path, "diann", requested_levels=("peptidoform_ion",))
        .compile()
        .parse()
    )
    target = tmp_path / f"ion{suffix}"

    formats.write_parsed_levels(parsed, target)

    assert input_files(formats.read_parsed_levels(target).uns) == input_files(parsed.uns)
