"""The authored sequence graph is the executable graph, without hidden side outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Never

import polars as pl
import pytest
from pydantic import ValidationError

from apb2.parserV2.parse_quant.data.parsed import ParsedLevel
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.modifications import (
    PlainSequenceStripper,
    SequenceColumn,
    SequenceValue,
    TokenRegexStripper,
    UnknownModificationError,
)
from apb2.parserV2.parse_quant.parameters.axis import ModificationTokenPosition
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_quant.source_resolution import SourcePlanResolver
from apb2.parserV2.parser_factory import compile_level
from apb2.parserV2.vendor_params.parsers.shared.unimod import UNIMOD_REGISTRY
from apb2.parserV2.vendor_parse_rules.document import RuleDocument
from parserV2 import synthetic
from parserV2.test_facade import delimited

INLINE: dict[str, Any] = {
    "parser": "token_regex",
    "token_pattern": r"\(([^()]*)\)",
    "token_position": "after_residue",
}
STRIP: dict[str, Any] = {
    "name": "ProForma_peptide",
    "how": "stripped_sequence",
    "inputs": ["Modified_Sequence"],
    "syntax": "vendor_sequence",
}
NORMALIZE: dict[str, Any] = {
    "name": "ProForma_peptidoform",
    "how": "proforma_sequence",
    "inputs": ["Modified_Sequence"],
    "syntax": "vendor_sequence",
    "modification_map": "basic_modification_map",
}


def base(operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "axis": {"obs_keys": ["Run"], "var_keys": ["ID"]},
        "columns": {
            "obs": [{"name": "Run", "source": "run"}],
            "var": [
                {"name": "ID", "source": "id"},
                {"name": "Modified_Sequence", "source": "vendor sequence"},
                {"name": "Alternative", "source": "other sequence"},
                *deepcopy(operations),
            ],
        },
        "sequence_syntax": {"vendor_sequence": deepcopy(INLINE)},
        "modification_maps": {
            "basic_modification_map": [{"token": "ox", "accession": "UNIMOD:35"}]
        },
        "measurements": {
            "primary_layer": "Quantity",
            "duplicates": {"mode": "error"},
            "layers": [{"name": "Quantity", "source": "quantity"}],
        },
    }


def document(declaration: dict[str, Any], level: dict[str, Any] | None = None) -> RuleDocument:
    return synthetic.document(shape="long", base=declaration, levels={"ion": level or {}})


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run": ["A"],
            "id": ["1"],
            "vendor sequence": ["ACM(ox)K"],
            "other sequence": ["PEPM(ox)IDE"],
            "quantity": [1],
        }
    )


def parse(tmp_path: Path, built: RuleDocument, values: pl.DataFrame | None = None) -> ParsedLevel:
    path = tmp_path / "input.tsv"
    (frame() if values is None else values).write_csv(path, separator="\t")
    return compile_level(synthetic.facade(built), SingleFile(path), "standard").parse()


@pytest.mark.parametrize("operation", [STRIP, NORMALIZE], ids=["strip", "normalize"])
def test_changing_the_declared_input_changes_the_values_consumed(
    operation: dict[str, Any],
    tmp_path: Path,
) -> None:
    declaration = base([operation])
    before = parse(tmp_path, document(declaration)).var.frame
    declaration["columns"]["var"][-1]["inputs"] = ["Alternative"]
    after = parse(tmp_path, document(declaration)).var.frame
    name = operation["name"]
    assert before[name][0].startswith("ACM")
    assert after[name][0].startswith("PEPM")


def test_stripping_needs_neither_normalization_nor_a_map_or_unimod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_lookup(self: object, accession: str) -> Never:
        raise AssertionError(f"stripping must not resolve {accession}")

    monkeypatch.setattr(type(UNIMOD_REGISTRY), "resolve", refuse_lookup)
    declaration = base([STRIP])
    del declaration["modification_maps"]
    parsed = parse(tmp_path, document(declaration))
    assert parsed.var.frame["ProForma_peptide"].to_list() == ["ACMK"]
    assert "ProForma_peptidoform" not in parsed.var.frame
    assert "unknown_mod_tokens" not in parsed.uns


@pytest.mark.parametrize("operations", [[STRIP, NORMALIZE], [NORMALIZE, STRIP], [NORMALIZE]])
def test_independent_operations_preserve_values_and_authored_output_order(
    operations: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    parsed = parse(tmp_path, document(base(operations)))
    assert parsed.var.frame["ProForma_peptidoform"].to_list() == ["ACM[UNIMOD:35]K"]
    assert parsed.var.frame.columns[-len(operations) :] == [op["name"] for op in operations]
    if STRIP in operations:
        assert parsed.var.frame["ProForma_peptide"].to_list() == ["ACMK"]
    assert not {"stripped_sequence", "proforma_sequence", "unknown_mod_tokens"} & set(
        parsed.var.frame.columns
    )
    plan = json.loads(str(parsed.uns["plan_json"]))
    assert "modifications" not in plan
    computations = [
        *plan["var"]["key_phase"]["computers"],
        *plan["var"]["output_phase"]["computers"],
    ]
    assert all(op["inputs"] == ["Modified_Sequence"] for op in computations)


def test_aliases_are_selected_from_the_unmodified_physical_frame(tmp_path: Path) -> None:
    declaration = base([STRIP])
    declaration["columns"]["var"][1:3] = [
        {"name": "other sequence", "source": "vendor sequence"},
        {"name": "Modified_Sequence", "source": "other sequence"},
    ]
    parsed = parse(tmp_path, document(declaration))
    assert parsed.var.frame["ProForma_peptide"].to_list() == ["PEPMIDE"]


def test_sequence_operation_consumes_the_result_of_declared_coalescing(tmp_path: Path) -> None:
    declaration = base([STRIP])
    declaration["columns"]["var"][1].update(
        {
            "how": "coalesce",
            "inputs": ["Modified_Sequence", "Alternative"],
        }
    )
    values = frame().with_columns(pl.lit(None, dtype=pl.String).alias("vendor sequence"))
    parsed = parse(tmp_path, document(declaration), values)
    assert parsed.var.frame["ProForma_peptide"].to_list() == ["PEPMIDE"]


@pytest.mark.parametrize("parser", ["site_list", "embedded_site_list"])
def test_multicolumn_normalization_records_and_consumes_every_input(
    parser: str, tmp_path: Path
) -> None:
    declaration = base([NORMALIZE, STRIP])
    declaration["columns"]["var"][1]["source"] = "sequence"
    declaration["columns"]["var"].insert(3, {"name": "Mods", "source": "mods"})
    normalization = next(
        c for c in declaration["columns"]["var"] if c.get("how") == "proforma_sequence"
    )
    normalization["inputs"].append("Mods")
    syntax: dict[str, Any] = {"parser": parser, "delimiter": ";", "site_base": 1}
    values = (
        frame()
        .rename({"vendor sequence": "sequence"})
        .with_columns(pl.lit("ACMK").alias("sequence"))
    )
    if parser == "site_list":
        declaration["columns"]["var"].insert(4, {"name": "Sites", "source": "sites"})
        normalization["inputs"].append("Sites")
        values = values.with_columns(pl.lit("ox").alias("mods"), pl.lit("3").alias("sites"))
    else:
        syntax["entry_pattern"] = r"(?P<token>.+) \((?P<site>[^)]+)\)"
        values = values.with_columns(pl.lit("ox (M3)").alias("mods"))
    declaration["sequence_syntax"] = {
        "vendor_sequence": syntax,
        "bare": {"parser": "plain_sequence"},
    }
    declaration["columns"]["var"][-1]["syntax"] = "bare"
    declaration["axis"]["var_keys"] = ["ProForma_peptidoform"]
    facade = synthetic.facade(document(declaration))
    resolved = SourcePlanResolver(facade.working_parameters).resolve(
        delimited(tuple(values.columns))
    )
    expected = ("sequence", "mods", "sites") if parser == "site_list" else ("sequence", "mods")
    assert resolved.var.source.keys.raw_key_columns == expected
    with pytest.raises(IncompatibleSourceError):
        facade.resolve_source(delimited(tuple(c for c in values.columns if c != "mods")))
    parsed = parse(tmp_path, document(declaration), values)
    assert parsed.var.frame["ProForma_peptidoform"].to_list() == ["ACM[UNIMOD:35]K"]
    assert parsed.var.frame["ProForma_peptide"].to_list() == ["ACMK"]


@pytest.mark.parametrize("policy", ["preserve", "drop", "error"])
def test_diagnostics_and_errors_include_rows_discarded_after_identity_validation(
    policy: str,
    tmp_path: Path,
) -> None:
    declaration = base([NORMALIZE])
    declaration["columns"]["var"][-1]["unknown_policy"] = policy
    values = pl.concat(
        [
            frame(),
            frame().with_columns(
                pl.lit(None, dtype=pl.String).alias("id"),
                pl.lit("ACM(mystery)K").alias("vendor sequence"),
            ),
        ]
    )
    if policy == "error":
        with pytest.raises(UnknownModificationError):
            parse(tmp_path, document(declaration), values)
        return
    parsed = parse(tmp_path, document(declaration), values)
    assert parsed.var.frame.height == 1
    if policy == "preserve":
        assert parsed.uns["unknown_mod_tokens"] == ["mystery"]
    else:
        assert "unknown_mod_tokens" not in parsed.uns


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("inputs", ["physical-only"], "undeclared"),
        ("inputs", ["Modified_Sequence", "Alternative"], "requires 1 inputs"),
        ("syntax", "missing", "missing syntax"),
        ("modification_map", "missing", "missing map"),
    ],
)
def test_invalid_dependencies_and_references_fail_before_reading(
    field: str,
    value: object,
    match: str,
) -> None:
    declaration = base([NORMALIZE])
    declaration["columns"]["var"][-1][field] = value
    with pytest.raises(ValidationError, match=match):
        document(declaration).declared("ion")


def test_forward_references_and_cycles_are_not_silently_reordered() -> None:
    declaration = base([STRIP, NORMALIZE])
    declaration["columns"]["var"][-2]["inputs"] = ["ProForma_peptidoform"]
    declaration["columns"]["var"][-1]["inputs"] = ["ProForma_peptide"]
    with pytest.raises(ValidationError, match="undeclared"):
        document(declaration).declared("ion")


@pytest.mark.parametrize("operation", [STRIP, NORMALIZE])
def test_operation_rejects_incompatible_syntax(operation: dict[str, Any]) -> None:
    declaration = base([operation])
    declaration["sequence_syntax"]["vendor_sequence"] = {
        "parser": "site_list" if operation == STRIP else "plain_sequence",
    }
    with pytest.raises(ValidationError, match="syntax"):
        document(declaration).declared("ion")


def test_named_definitions_are_replaced_whole_after_base_composition(tmp_path: Path) -> None:
    declaration = base([NORMALIZE])
    level: dict[str, Any] = {
        "sequence_syntax": {
            "vendor_sequence": {"parser": "token_regex", "token_pattern": r"\[([^\]]+)\]"}
        },
        "modification_maps": {
            "basic_modification_map": [{"token": "new", "accession": "UNIMOD:35"}]
        },
    }
    built = document(declaration, level)
    effective = built.declared("ion").declaration
    assert [entry.token for entry in effective.modification_maps["basic_modification_map"]] == [
        "new"
    ]
    values = frame().with_columns(pl.lit("ACM[new]K").alias("vendor sequence"))
    assert parse(tmp_path, built, values).var.frame["ProForma_peptidoform"].to_list() == [
        "ACM[UNIMOD:35]K"
    ]
    level["sequence_syntax"]["vendor_sequence"].pop("token_pattern")
    with pytest.raises(ValidationError, match="token_pattern"):
        document(declaration, level).declared("ion")


def test_memoization_is_local_to_each_operation() -> None:
    class Counting:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def transform(self, row: tuple[str, ...], /) -> SequenceValue:
            self.calls.append(row)
            return SequenceValue(row[0])

    operation = Counting()
    first = SequenceColumn("first", ("Sequence",), operation)
    second = SequenceColumn("second", ("Sequence",), operation)
    values = (pl.Series("s", ["ACMK", "PEPTIDE", "ACMK"]),)
    assert first.compute(values).values.to_list() == ["ACMK", "PEPTIDE", "ACMK"]
    assert second.compute(values).values.to_list() == ["ACMK", "PEPTIDE", "ACMK"]
    assert operation.calls == [("ACMK",), ("PEPTIDE",), ("ACMK",), ("PEPTIDE",)]


@pytest.mark.parametrize(
    ("pattern", "position", "modified"),
    [
        (r"\(([^()]*)\)", "after_residue", "_(ac)PEPM(ox)IDE_"),
        (r"\[([^\]]+)\]", "after_residue", "_PEPM[15.9949]IDE_"),
        (r"[a-z]+", "before_residue", "PEPoxMIDE"),
    ],
)
def test_independent_stripping_handles_vendor_syntax_terminals_and_nulls(
    pattern: str,
    position: ModificationTokenPosition,
    modified: str,
) -> None:
    computer = SequenceColumn("Peptide", ("Sequence",), TokenRegexStripper(pattern, position))
    result = computer.compute((pl.Series("sequence", [modified, None, ""]),))
    assert result.values.to_list() == ["PEPMIDE", "", ""]
    assert result.unknown_mod_tokens == ()


def test_plain_stripping_preserves_site_list_residue_semantics() -> None:
    computer = SequenceColumn("Peptide", ("Sequence",), PlainSequenceStripper())
    assert computer.compute((pl.Series("sequence", ["_PEP.MIDE_", None]),)).values.to_list() == [
        "PEPMIDE",
        "",
    ]


@pytest.mark.parametrize("field", ["modification_map", "case_sensitive", "unknown_policy"])
def test_stripping_does_not_accept_normalization_only_fields(field: str) -> None:
    declaration = base([STRIP])
    declaration["columns"]["var"][-1][field] = NORMALIZE.get(field, False)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        document(declaration).declared("ion")
