"""Shared fixtures over the documents Parser V2 packages.

Vendor exports are not committed. Point ``APB2_TEST_DATA`` at a downloaded corpus root —
the directory holding ``raw_file_db_downloaded.csv`` and ``json_dir/`` — to exercise the
tests that read a real header. Without it those tests skip, which is the normal state of a
fresh checkout and of CI.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED as PARSER_V2_PACKAGED
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2.rule_inventory import document_key

_TEXT_DELIMITERS = ("\t", ",", ";")

SEARCH_EVIDENCES = (
    SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None),
    SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False),
    SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=True),
)
"""Enough evidence to admit every packaged level's gate at least once."""


@cache
def corpus_root() -> Path | None:
    """The downloaded vendor corpus named by ``APB2_TEST_DATA``, when it carries an index."""
    named = os.environ.get("APB2_TEST_DATA")
    if not named:
        return None
    root = Path(named).expanduser().resolve()
    return root if (root / "raw_file_db_downloaded.csv").exists() else None


def _cached_export(software_name: str, version_pattern: str) -> Path | None:
    """The first downloaded export of one software whose version the pattern admits."""
    root = corpus_root()
    if root is None:
        return None
    with (root / "raw_file_db_downloaded.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["software_name"] != software_name or row.get("status") != "ok":
                continue
            if not re.search(version_pattern, row["software_version"]):
                continue
            found = root / "json_dir" / row["input_file_path"]
            if found.exists():
                return found
    return None


def _delimited_header(path: Path, delimiters: tuple[str, ...]) -> tuple[str, ...]:
    """Read only the header row, taking the delimiter that splits it into most columns."""
    best: tuple[str, ...] = ()
    for delimiter in delimiters:
        names = tuple(
            pl.scan_csv(
                path,
                separator=delimiter,
                infer_schema_length=0,
                encoding="utf8-lossy",
                truncate_ragged_lines=True,
            )
            .collect_schema()
            .names()
        )
        if len(names) > len(best):
            best = names
    return best


def _header_of(path: Path) -> tuple[str, ...]:
    """The column names one cached export carries, without reading its rows."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return tuple(pl.read_parquet_schema(path))
    if suffix == ".csv":
        return _delimited_header(path, (",",))
    if suffix == ".tsv":
        return _delimited_header(path, ("\t",))
    if suffix == ".txt":
        return _delimited_header(path, _TEXT_DELIMITERS)
    return ()


@dataclass(frozen=True)
class PackagedDocument:
    """One vendor document Parser V2 packages, and the cached export it describes."""

    key: str
    parser_v2_path: Path

    def data_path(self) -> Path | None:
        """The cached real export of this vendor, when the corpus carries one."""
        document = load_rule_document(self.parser_v2_path)
        return _cached_export(document.software_name, document.software_version_pattern)

    def required_data_path(self) -> Path:
        """The cached real export, skipping the test when the corpus carries none."""
        found = self.data_path()
        if found is None:
            pytest.skip(f"no downloaded export for {self.key}; set APB2_TEST_DATA")
        return found

    def header(self) -> tuple[str, ...]:
        """The header of a cached real export, skipping the test when there is none.

        A test that asks for a header cannot run without one: returning an empty header
        instead would let the test proceed and fail on a fact about the corpus rather than
        about the code under test.
        """
        found = self.data_path()
        if found is None:
            pytest.skip(f"no downloaded export for {self.key}; set APB2_TEST_DATA")
        return _header_of(found)

    def first_admitted_facade(self, level: QuantificationLevel | None = None) -> ParseRuleFacade:
        """The facade for one level under whichever search-parameter evidence its gate admits."""
        document = load_rule_document(self.parser_v2_path)
        chosen = level if level is not None else document.levels[0]
        for evidence in SEARCH_EVIDENCES:
            try:
                return ParseRuleFacade(document, chosen, evidence)
            except RuleNotApplicable:
                continue
        raise AssertionError(f"no evidence admits {self.key}/{chosen}")


def document_pairs() -> tuple[PackagedDocument, ...]:
    """Every document Parser V2 packages, in packaged order."""
    return tuple(
        PackagedDocument(key=document_key(path), parser_v2_path=path) for path in PARSER_V2_PACKAGED
    )


def level_pairs() -> tuple[tuple[PackagedDocument, QuantificationLevel], ...]:
    """Every ``(document, level)`` Parser V2 declares, in packaged order."""
    return tuple(
        (document, level)
        for document in document_pairs()
        for level in load_rule_document(document.parser_v2_path).levels
    )
