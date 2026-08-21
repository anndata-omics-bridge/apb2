"""Shared fixtures for the Parser V2 suites: paired documents and real vendor headers.

The unchanged schema-0.2 package is the oracle, so every migration and parity test needs the
same two things: the pair of documents describing one vendor, and a header a real export of
that vendor actually carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from anndata_proteomics.test_data import VendorDataUnavailable, find_test_data_for_version

from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED as PARSER_V2_PACKAGED
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema_base import QuantificationLevel
from apb2.vendor_parse_rules.rules import PACKAGED as LEGACY_PACKAGED
from apb2.vendor_parse_rules.rules import load_document
from parserV2.rule_inventory import document_key

_TEXT_DELIMITERS = ("\t", ",", ";")

SEARCH_EVIDENCES = (
    SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None),
    SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False),
    SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=True),
)
"""Enough evidence to admit every packaged level's gate at least once."""


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


@dataclass(frozen=True, slots=True)
class DocumentPair:
    """One vendor's document in both generations, keyed by its path below ``documents/``."""

    key: str
    legacy_path: Path
    parser_v2_path: Path

    def data_path(self) -> Path | None:
        """The cached real export of this vendor, when the test data is available."""
        document = load_document(self.legacy_path)
        found = find_test_data_for_version(
            document.software_name, document.software_version_pattern
        )
        if isinstance(found, VendorDataUnavailable) or not found.exists():
            return None
        return found

    def header(self) -> tuple[str, ...]:
        """The header of a cached real export of this vendor, or an empty tuple."""
        found = self.data_path()
        return () if found is None else _header_of(found)

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


def document_pairs() -> tuple[DocumentPair, ...]:
    """Every packaged document, paired across the two rule generations."""
    parser_v2 = {document_key(path): path for path in PARSER_V2_PACKAGED}
    return tuple(
        DocumentPair(
            key=document_key(path),
            legacy_path=path,
            parser_v2_path=parser_v2[document_key(path)],
        )
        for path in LEGACY_PACKAGED
    )


def level_pairs() -> tuple[tuple[DocumentPair, QuantificationLevel], ...]:
    """Every ``(document pair, level)`` both generations declare, in packaged order."""
    return tuple(
        (pair, level)
        for pair in document_pairs()
        for level in load_rule_document(pair.parser_v2_path).levels
    )
