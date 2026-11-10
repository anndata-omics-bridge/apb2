"""Entries tagged with a software version: one column a vendor spells differently by version.

Names are unique ignoring case, as DuckDB's column names are, unless every entry of one name
carries a ``version``. A header then picks the entry whose source it carries; with a parameter
file, the picked entry's version must admit the software version it names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from apb2.parserV2.detect_document import RuleUnavailableError, detect_rule_documents
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedSourceEvidence,
    Folder,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.parser import ParserCollection
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.search_parameters.model import Parameters
from apb2.parserV2.vendor_parse_rules.document import RuleDocument
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.axis import SourcedColumn
from apb2.parserV2.vendor_parse_rules.schema.measurements import FactorLayer, NumericLayer
from parserV2 import synthetic
from parserV2.join_fixtures import maxquant_tables
from parserV2.synthetic import plan_snapshot

MAXQUANT = Path("src/apb2/parserV2/vendor_parse_rules/documents/maxquant_wide/rules.json")
NUMBERS = NumericTextFormat(decimal_mark=".", thousands_marks=())
_OLD = {"name": "lead", "source": "Lead", "version": "^1\\.", "required": False}
_NEW = {"name": "lead", "source": "lead", "version": "^2\\.", "required": False}


def _document(
    var: list[dict[str, Any]], layers: list[dict[str, Any]] | None = None
) -> RuleDocument:
    """A long rule keyed by ``feature``, with a test's further var entries and layers."""
    return synthetic.document(
        shape="long",
        base={
            "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
            "columns": {
                "obs": [{"name": "sample", "source": "Sample"}],
                "var": synthetic.with_flags([{"name": "feature", "source": "Feature"}, *var]),
            },
            "measurements": {
                "primary_layer": "quantity",
                "layers": [{"name": "quantity", "source": "Quantity"}, *(layers or [])],
            },
        },
        levels={"peptidoform_ion": {}},
    )


def _header(*columns: str) -> DelimitedSourceEvidence:
    return DelimitedSourceEvidence(
        columns=("Sample", "Feature", "Quantity", *columns),
        delimiter="\t",
        quote_char='"',
        encoding="utf8",
        number_format=NUMBERS,
    )


def _facade(document: RuleDocument, version: str | None = None) -> ParseRuleFacade:
    return ParseRuleFacade(document, "peptidoform_ion", synthetic.NO_EVIDENCE, version)


# ------------------------------------------------------------------------------ declaration


def test_entries_of_one_name_each_with_a_version_are_one_column() -> None:
    rule = _document([_OLD, _NEW]).declared("peptidoform_ion").declaration

    sourced = [entry for entry in rule.columns.var if isinstance(entry, SourcedColumn)]
    assert [(entry.name, entry.version) for entry in sourced if entry.name == "lead"] == [
        ("lead", "^1\\."),
        ("lead", "^2\\."),
    ]


@pytest.mark.parametrize(
    ("var", "message"),
    [
        pytest.param(
            [{**_OLD, "version": None}, {**_NEW, "version": None, "name": "Lead"}],
            r"column entry names must be unique ignoring case.*\['Lead', 'lead'\]",
            id="names-equal-ignoring-case",
        ),
        pytest.param(
            [_OLD, {**_NEW, "name": "Lead"}],
            r"unique ignoring case.*\['Lead', 'lead'\]",
            id="tagged-but-spelled-differently",
        ),
        pytest.param(
            [_OLD, {**_NEW, "version": None}],
            r"unless every entry of one name carries a version",
            id="one-untagged",
        ),
        pytest.param(
            [_OLD, {**_NEW, "type": "number"}],
            r"entries named 'lead' differ in more than source, version and note",
            id="disagreeing-type",
        ),
        pytest.param(
            [{**_OLD, "required": True}, _NEW],
            r"'lead': an entry with a version is not required",
            id="required",
        ),
        pytest.param(
            [{**_OLD, "version": "^(1"}, _NEW],
            r"version is not a valid regex",
            id="invalid-regex",
        ),
    ],
)
def test_names_equal_ignoring_case_need_a_version_on_every_entry(
    var: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _document(var).declared("peptidoform_ion")


def test_layers_follow_the_same_rule() -> None:
    old = {"name": "count", "source": "Count", "version": "^1\\.", "type": "integer"}
    new = {"name": "count", "source": "count", "version": "^2\\.", "type": "integer"}

    _document([], [old, new]).declared("peptidoform_ion")
    clashing = [{**old, "version": None}, {**new, "version": None, "name": "Count"}]
    with pytest.raises(ValidationError, match="measurement layer names must be unique"):
        _document([], clashing).declared("peptidoform_ion")


def test_a_wide_layer_carries_no_version() -> None:
    document = synthetic.wide_document(
        var_select={"feature": "Feature"},
        layers=[
            {"name": "quantity", "source": "Quantity (?P<sample>.+)", "version": "^1\\."},
            {"name": "other", "source": "Other (?P<sample>.+)"},
        ],
        primary_layer="other",
    )
    with pytest.raises(ValidationError, match="a wide layer carries no version"):
        document.declared("peptidoform_ion")


# ------------------------------------------------------------------------------- resolution


@pytest.mark.parametrize(("source", "version"), [("Lead", "^1\\."), ("lead", "^2\\.")])
def test_a_header_picks_the_entry_whose_source_it_carries(source: str, version: str) -> None:
    plan = plan_snapshot(_facade(_document([_OLD, _NEW])).resolve_source(_header(source)))

    assert plan["var"]["outputs"].count("lead") == 1
    assert "lead" not in plan["var"]["skipped"]
    assert plan["var"]["versions"] == {"lead": version}


def test_a_header_carrying_neither_spelling_has_no_such_column() -> None:
    plan = plan_snapshot(_facade(_document([_OLD, _NEW])).resolve_source(_header()))

    assert "lead" not in plan["var"]["outputs"]
    assert plan["var"]["versions"] == {}


def test_a_header_carrying_both_spellings_is_refused() -> None:
    with pytest.raises(IncompatibleSourceError, match="reads one source per name"):
        _facade(_document([_OLD, _NEW])).resolve_source(_header("Lead", "lead"))


def test_the_picked_entry_must_admit_the_parameter_files_version() -> None:
    facade = _facade(_document([_OLD, _NEW]), version="2.0")

    plan = plan_snapshot(facade.resolve_source(_header("lead")))
    assert plan["var"]["versions"] == {"lead": "^2\\."}
    with pytest.raises(IncompatibleSourceError, match=r"which software version 2\.0 does not"):
        facade.resolve_source(_header("Lead"))


def test_a_column_only_some_versions_write_has_one_tagged_entry() -> None:
    only_new = {**_NEW, "name": "library_q", "source": "Library Q"}
    facade = _facade(_document([only_new]), version="1.0")

    assert "library_q" not in plan_snapshot(facade.resolve_source(_header()))["var"]["outputs"]
    with pytest.raises(IncompatibleSourceError, match="does not write"):
        facade.resolve_source(_header("Library Q"))


def test_tagged_layers_pick_one_source_per_name() -> None:
    layers = [
        {"name": "count", "source": "Count", "version": "^1\\.", "type": "integer"},
        {"name": "count", "source": "count", "version": "^2\\.", "type": "integer"},
    ]
    plan = plan_snapshot(_facade(_document([], layers)).resolve_source(_header("count")))

    assert [entry["layer_name"] for entry in plan["layer_values"]] == ["quantity", "count"]
    assert plan["layer_versions"] == {"count": "^2\\."}


# ------------------------------------------------------------------------------- packaged


_MAXQUANT_SPELLINGS = {
    "1.5.2.8": ("Leading Proteins", "Leading Razor Protein", "MS/MS Count", "Reverse"),
    "1.5.3.30": ("Leading proteins", "Leading razor protein", "MS/MS Count", "Reverse"),
    "1.5.8.2": ("Leading proteins", "Leading razor protein", "MS/MS count", "Reverse"),
    "2.5.1.0": ("Leading proteins", "Leading razor protein", "MS/MS count", "Reverse"),
    "2.7.5.0": ("Leading proteins", "Leading razor protein", "MS/MS count", "Reverse"),
    "2.8.1.0": ("Leading proteins", "Leading razor proteins", "MS/MS count", "Decoy"),
}
"""The evidence.txt spellings of real MaxQuant releases, from ProteoBench uploads."""


def _maxquant_evidence(version: str) -> pl.DataFrame:
    """Synthetic evidence rows in one version's spelling; the reverse hit lists no protein."""
    leading, razor, count, decoy = _MAXQUANT_SPELLINGS[version]
    evidence = maxquant_tables()["evidence"].drop("Leading razor protein")
    return evidence.with_columns(
        pl.Series("Modified sequence", ["_PEPTIDE_", "_PEPTIDE_", "_EDITPEP_"]),
        pl.Series("Sequence", ["PEPTIDE", "PEPTIDE", "EDITPEP"]),
        pl.Series("Proteins", ["P;Q", "P;Q", None]),
        pl.Series(leading, ["P", "P", "REV__P"]),
        pl.Series(razor, ["P", "P", "REV__P"]),
        pl.Series(count, ["1", "2", "1"]),
        pl.Series(decoy, [None, None, "+"]),
    )


@pytest.mark.parametrize("version", list(_MAXQUANT_SPELLINGS))
def test_maxquant_reads_each_versions_spelling_under_one_name(version: str, tmp_path: Path) -> None:
    _maxquant_evidence(version).write_csv(tmp_path / "evidence.txt", separator="\t")
    detected = detect_rule_documents(
        Parameters(software_name="MaxQuant", software_version=version),
        Folder(tmp_path),
        ("peptidoform_ion",),
    )
    ion = ParserCollection(tuple(item.parser for item in detected.levels), {}).parse()
    level = ion.levels["peptidoform_ion"]
    var = level.var.frame.sort("Sequence")

    assert var["Proteins"].to_list() == ["REV__P", "P;Q"]
    assert var["Leading_proteins"].to_list() == ["REV__P", "P"]
    assert var["Leading_razor_protein"].to_list() == ["REV__P", "P"]
    assert var["apb_Decoy"].to_list() == [True, False]
    assert "MS_MS_count" in level.layers


def test_maxquant_refuses_a_spelling_its_version_never_writes(tmp_path: Path) -> None:
    _maxquant_evidence("1.5.2.8").write_csv(tmp_path / "evidence.txt", separator="\t")

    refused = r"carries 'evidence\.Leading Proteins', written by versions .*, which software version 2\.5\.1\.0 does not write"
    with pytest.raises(RuleUnavailableError, match=refused):
        detect_rule_documents(
            Parameters(software_name="MaxQuant", software_version="2.5.1.0"),
            Folder(tmp_path),
            ("peptidoform_ion",),
        )


@pytest.mark.parametrize("version", list(_MAXQUANT_SPELLINGS))
def test_maxquant_tags_admit_one_entry_per_name_for_each_known_version(version: str) -> None:
    document = load_rule_document(MAXQUANT)
    for level in document.levels:
        rule = document.declared(level).declaration
        entries = (*rule.columns.var, *rule.measurements.layers)
        tagged = {
            entry.name: entry.version
            for entry in entries
            if isinstance(entry, SourcedColumn | NumericLayer | FactorLayer)
            and entry.version is not None
        }
        admitted = [
            entry.name
            for entry in entries
            if isinstance(entry, SourcedColumn | NumericLayer | FactorLayer)
            and entry.version is not None
            and re.search(entry.version, version)
        ]
        assert sorted(admitted) == sorted(tagged), (level, version)
