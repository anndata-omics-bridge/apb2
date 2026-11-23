"""Optional readers are loaded only by the operations that need them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_base_api_and_missing_extras() -> None:
    """A subprocess starts with empty backend caches and can simulate a base install."""
    fixtures = Path(__file__).parent / "data"
    command = """
import sys
from pathlib import Path

from apb2.api import ConversionError, ParseRuleCompiler, ResultIOError, parse_search_parameters
from apb2.api import read_parsed_levels, write_parsed_levels
from apb2.parserV2.vendor_params.parsers.shared.errors import ParamsError

for name in ("duckdb", "fastexcel", "python_calamine"):
    assert name not in sys.modules, name
    sys.modules[name] = None
for name in ("duckdb", "parquet_reader", "parquet_writer"):
    assert f"apb2.parserV2.parse_quant.io.{name}" not in sys.modules, name

def expect_error(operation, error_type, extra):
    try:
        operation()
    except error_type as error:
        assert f"pip install 'apb2[{extra}]'" in str(error), str(error)
    else:
        raise AssertionError(f"{extra} operation succeeded without its backend")

fixtures = Path(sys.argv[1])
expect_error(lambda: read_parsed_levels(Path("result.duckdb")), ResultIOError, "duckdb")
expect_error(lambda: write_parsed_levels(None, Path("result.duckdb")), ResultIOError, "duckdb")
for vendor in ("msangel", "prolinestudio"):
    workbook = fixtures / vendor / "sample.txt"
    expect_error(
        lambda: ParseRuleCompiler.from_software(workbook, vendor),
        ConversionError,
        "excel",
    )
expect_error(
    lambda: parse_search_parameters(fixtures / "prolinestudio/sample.txt", "prolinestudio"),
    ParamsError,
    "excel",
)
"""
    subprocess.run([sys.executable, "-c", command, str(fixtures)], check=True)
