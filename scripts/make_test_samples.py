"""Create apb2's committed test artifacts from APB Studio's fixture store.

``STORE`` is the Fixture Manager's test-data directory, for example
``../apb_studio/test_data_download``; its ``downloads.csv`` lists every downloaded export. For
every packaged rule document with an admitted export there, write into ``--output`` (default:
``tests/parserV2/data/``) one folder per rule key holding:

- ``header.txt`` — the export's column names, one per line;
- ``sample.<ext>.gz`` (or ``sample.parquet``) — a stratified ~500-row sample: up to 250 rows of
  each of the first two runs for long tables, a plain head otherwise;
- ``expected.json`` — per-level observation/variable counts and layer names from a real
  ``convert_all_from_rule_config`` run over that sample.

Artifacts are append-only: an existing rule folder is left untouched unless ``--force``. The twin
script in ``extend_directflq_benchmark`` covers the rules whose exports are not ProteoBench data.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import tempfile
from pathlib import Path

import polars as pl
from cyclopts import App
from loguru import logger

from apb2.api import ConversionError
from apb2.cli.conversion import convert_all_from_rule_config
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document

app = App(name="make-test-samples", help=__doc__)

EVIDENCES = (
    SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None),
    SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False),
    SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=True),
)
DELIMITERS = (b"\t", b";", b",")
ROWS_PER_RUN = 250
MAX_ROWS = 500
SCAN_LIMIT = 400_000

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "parserV2" / "data"


def document_key(path: Path) -> str:
    """The rule key: document path segments under ``documents/``, without ``rules.json``."""
    parts = path.parts
    return "/".join(parts[parts.index("documents") + 1 : -1])


def admitted_facades(document: RuleDocument) -> list[ParseRuleFacade]:
    """One facade per declared level, under whichever evidence its gate admits."""
    facades: list[ParseRuleFacade] = []
    for level in document.levels:
        for evidence in EVIDENCES:
            try:
                facades.append(ParseRuleFacade(document, level, evidence))
                break
            except RuleNotApplicable:
                continue
    return facades


def store_exports(document: RuleDocument, root: Path) -> list[Path]:
    """Every store export of this software whose version the document admits."""
    index = root / "downloads.csv"
    if not index.exists():
        return []
    found: list[Path] = []
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["software_name"] != document.software_name or row.get("status") != "ok":
                continue
            if not re.search(document.software_version_pattern, row["software_version"]):
                continue
            path = root / row["input_file_path"]
            if path.exists():
                found.append(path)
    return found


def header_of(path: Path) -> tuple[str, ...]:
    """Column names of one export, reading only its header."""
    if path.suffix.lower() == ".parquet":
        return tuple(pl.read_parquet_schema(path))
    with path.open("rb") as handle:
        first = handle.readline()
    delimiter = _delimiter_for(first)
    return tuple(
        cell.decode("utf-8", errors="replace").strip('"')
        for cell in first.rstrip(b"\r\n").split(delimiter)
    )


def _delimiter_for(header_line: bytes) -> bytes:
    return max(DELIMITERS, key=lambda d: header_line.count(d))


def document_admits(document: RuleDocument, header: tuple[str, ...]) -> bool:
    """Whether any declared level of the document accepts this header."""
    return any(
        facade.working_parameters.accepts_header(header) for facade in admitted_facades(document)
    )


def run_columns(document: RuleDocument, header: tuple[str, ...]) -> list[int]:
    """Header indexes of the columns that can carry run identity, per the document itself."""
    sources: list[str] = []
    for facade in admitted_facades(document):
        columns = facade.working_parameters.obs
        sources += [s.source for s in columns.required_selections]
        sources += [s.source for s in columns.optional_selections]
    return [header.index(name) for name in dict.fromkeys(sources) if name in header]


def sample_text(path: Path, run_indexes: list[int]) -> tuple[bytes, int]:
    """Header plus a stratified sample: ≤250 rows of each of the first two run values."""
    kept: list[bytes] = []
    counts: dict[bytes, int] = {}
    with path.open("rb") as handle:
        header_line = handle.readline()
        delimiter = _delimiter_for(header_line)
        for scanned, line in enumerate(handle):
            if scanned >= SCAN_LIMIT or len(kept) >= MAX_ROWS:
                break
            if not line.strip():
                continue
            cells = line.rstrip(b"\r\n").split(delimiter)
            run = b"|".join(cells[i] for i in run_indexes if i < len(cells)) or b"-"
            if run not in counts and len(counts) >= 2:
                continue
            if counts.get(run, 0) >= ROWS_PER_RUN:
                continue
            counts[run] = counts.get(run, 0) + 1
            kept.append(line)
    return header_line + b"".join(kept), len(kept)


def expectations(sample: Path, rule_config: Path, params: Path | None) -> dict[str, object]:
    """Convert the sample for real and record every produced level's dimensions."""
    with tempfile.TemporaryDirectory() as scratch:
        summary = convert_all_from_rule_config(
            data=sample,
            output=Path(scratch) / "sample.h5mu",
            rule_config=rule_config,
            parameters_path=params,
            software=None,
            checks="standard",
        )
    return {
        level.level: {
            "observations": level.observation_count,
            "variables": level.variable_count,
            "layers": list(level.layer_names),
        }
        for level in summary.levels
    }


def write_artifacts(key: str, export: Path, rule_config: Path, output: Path) -> None:
    """Write header, sample, and expectations for one rule into ``output/<key>/``."""
    target = output / key
    target.mkdir(parents=True, exist_ok=True)
    header = header_of(export)
    (target / "header.txt").write_text("\n".join(header) + "\n", encoding="utf-8")

    if export.suffix.lower() == ".parquet":
        sample_name = "sample.parquet"
        pl.scan_parquet(export).head(MAX_ROWS).collect().write_parquet(target / sample_name)
        rows = min(
            MAX_ROWS, pl.scan_parquet(target / sample_name).select(pl.len()).collect().item()
        )
        sample_for_convert = target / sample_name
        scratch = None
    else:
        document = load_rule_document(rule_config)
        data, rows = sample_text(export, run_columns(document, header))
        suffix = export.suffix.lower() or ".txt"
        sample_name = f"sample{suffix}.gz"
        (target / sample_name).write_bytes(gzip.compress(data, mtime=0))
        scratch = tempfile.TemporaryDirectory()
        sample_for_convert = Path(scratch.name) / f"sample{suffix}"
        sample_for_convert.write_bytes(data)

    params = next(iter(sorted(export.parent.glob("param_0.*"))), None)
    params_copy: Path | None = None
    if params is not None:
        params_copy = target / params.name
        params_copy.write_bytes(params.read_bytes())

    try:
        levels = expectations(sample_for_convert, rule_config, params_copy)
    finally:
        if scratch is not None:
            scratch.cleanup()
    record = {
        "source": export.name,
        "sample": sample_name,
        "params": params_copy.name if params_copy is not None else None,
        "sample_rows": rows,
        "levels": levels,
    }
    (target / "expected.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{key}: {rows} rows, levels {sorted(levels)}")


@app.default
def make(store: Path, output: Path = DEFAULT_OUTPUT, *, force: bool = False) -> None:
    """Create artifacts for every packaged rule with an admitted export in STORE."""
    root = store
    made, kept, absent = 0, 0, 0
    for rule_config in PACKAGED:
        key = document_key(rule_config)
        if (output / key / "expected.json").exists() and not force:
            kept += 1
            continue
        document = load_rule_document(rule_config)
        export = next(
            (
                path
                for path in store_exports(document, root)
                if document_admits(document, header_of(path))
            ),
            None,
        )
        if export is None:
            logger.warning(f"{key}: no admitted export in {root}")
            absent += 1
            continue
        try:
            write_artifacts(key, export, rule_config, output)
            made += 1
        except ConversionError as error:
            logger.error(f"{key}: sample does not convert: {error}")
            absent += 1
    logger.info(f"done: {made} created, {kept} kept (append-only), {absent} without artifacts")


if __name__ == "__main__":
    app()
