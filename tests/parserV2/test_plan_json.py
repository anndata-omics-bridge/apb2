"""The resolved plan a written result carries: complete, ordered, and refusing what it cannot say.

The plan is the only record of what one source actually resolved to. The rule describes every
export its vendor may produce; after the parse, nothing but the plan can say which columns this
export provided, which dialect and notation won, which optional layers it could not supply, or
what the encoders were told. So these tests check three things: that the serialization covers
every field the plan holds, that it is stable enough to diff two runs, and that it reaches the
provenance of both outputs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import anndata
import pytest

from apb2.parserV2.compile import AnnDataOutput, ParquetOutput, ParseRuleCompiler
from apb2.parserV2.parse_quant.anndata_writer import NAMESPACE, PARSE_NAMESPACE
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.plan_json import (
    PLAN_JSON_KEY,
    as_json_value,
    resolved_plan_json,
)
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedSourceEvidence,
    NumericTextFormat,
    SingleFile,
)
from apb2.parserV2.parse_quant.parquet_writer import MANIFEST_NAME
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2 import synthetic
from parserV2.fixtures import DocumentPair, level_pairs

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())


def evidence(columns: tuple[str, ...]) -> DelimitedSourceEvidence:
    return DelimitedSourceEvidence(
        columns=columns,
        delimiter="\t",
        quote_char='"',
        encoding="utf8",
        number_format=DOT,
    )


def resolved() -> ResolvedLevelPlan:
    """One plan whose source withheld something: an optional column and an optional layer."""
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        var_optional={"Extra": "Extra"},
        layers=[
            {"name": "Quantity", "source": "Quantity"},
            {"name": "Score", "source": "Score"},
        ],
    )
    return synthetic.facade(document).resolve_source(evidence(("Sample", "Feature", "Quantity")))


def written(tmp_path: Path) -> Path:
    path = tmp_path / "report.tsv"
    path.write_text("Sample\tFeature\tQuantity\nA\tF1\t1.5\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ completeness


def test_the_serialization_covers_every_field_the_plan_holds() -> None:
    plan = resolved()

    decoded = json.loads(resolved_plan_json(plan))

    assert set(decoded) == {field.name for field in dataclasses.fields(plan)}
    assert set(decoded["read"]) == {field.name for field in dataclasses.fields(plan.read)}
    assert decoded["level"] == "ion"
    assert decoded["duplicate_mode"] == "error"


def test_the_plan_states_what_this_source_resolved_to_not_what_the_rule_permits() -> None:
    plan = resolved()

    decoded = json.loads(resolved_plan_json(plan))

    assert decoded["read"]["projected_columns"] == list(plan.read.projected_columns)
    assert [config["layer_name"] for config in decoded["raw_value_presence"]] == ["Quantity"]
    assert [config["layer_name"] for config in decoded["ann_data"]["layer_encodings"]] == [
        "Quantity"
    ]
    assert decoded["var"]["skipped"] == ["Extra"]
    assert decoded["ann_data"]["layer_contract"]["primary_layer_name"] == "Quantity"


def test_the_notation_the_source_was_read_under_survives_into_the_record() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    comma = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))
    plan = synthetic.facade(document).resolve_source(
        DelimitedSourceEvidence(
            columns=("Sample", "Feature", "Quantity"),
            delimiter=";",
            quote_char='"',
            encoding="utf8",
            number_format=comma,
        )
    )

    encoding = json.loads(resolved_plan_json(plan))["ann_data"]["layer_encodings"][0]

    assert encoding["number_format"] == {"decimal_mark": ",", "thousands_marks": ["."]}


# --------------------------------------------------------------------------------- stability


def test_a_set_becomes_a_sorted_list_so_the_record_can_be_diffed() -> None:
    plan = resolved()

    decoded = json.loads(resolved_plan_json(plan))

    assert decoded["read"]["text_sources"] == sorted(plan.read.text_sources)
    assert decoded["read"]["native_numeric_sources"] == sorted(plan.read.native_numeric_sources)
    assert isinstance(decoded["var"]["skipped"], list)


def test_two_resolutions_of_one_source_serialize_to_the_same_text() -> None:
    assert resolved_plan_json(resolved()) == resolved_plan_json(resolved())


# ---------------------------------------------------------------------------------- refusals


def test_a_value_with_no_json_form_is_refused_rather_than_stringified() -> None:
    with pytest.raises(TypeError, match="no JSON form"):
        as_json_value(object())


def test_a_key_that_is_not_text_is_refused() -> None:
    with pytest.raises(TypeError, match="expected text"):
        as_json_value({1: "one"})


# ------------------------------------------------------------------- what an output carries


def test_a_compiled_parser_stores_the_plan_beside_the_rule_it_came_from(tmp_path: Path) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    parser = ParseRuleCompiler(facade=synthetic.facade(document), output=ParquetOutput()).compile(
        SingleFile(path=written(tmp_path))
    )

    parsed = parser.parse()

    assert json.loads(str(parsed.uns[PLAN_JSON_KEY]))["level"] == "ion"
    assert "rule_json" in parsed.uns


def test_the_plan_reaches_the_parse_namespace_of_a_written_h5ad(tmp_path: Path) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    parser = ParseRuleCompiler(facade=synthetic.facade(document), output=AnnDataOutput()).compile(
        SingleFile(path=written(tmp_path))
    )
    target = tmp_path / "ion.h5ad"

    parser.convert(parser.parse(), target)

    stored = anndata.read_h5ad(target)
    plan = json.loads(stored.uns[NAMESPACE][PARSE_NAMESPACE][PLAN_JSON_KEY])
    assert plan["level"] == "ion"
    assert plan["read"]["projected_columns"] == ["Sample", "Feature", "Quantity"]


def test_the_plan_reaches_the_manifest_of_a_written_parquet_dataset(tmp_path: Path) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    parser = ParseRuleCompiler(facade=synthetic.facade(document), output=ParquetOutput()).compile(
        SingleFile(path=written(tmp_path))
    )
    target = tmp_path / "ion"

    parser.convert(parser.parse(), target)

    manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert json.loads(manifest["uns"][PLAN_JSON_KEY])["level"] == "ion"


# ------------------------------------------------------------------------- packaged coverage


@pytest.mark.parametrize(
    ("pair", "level"),
    [pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()],
)
def test_every_packaged_level_this_data_satisfies_serializes_its_plan(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    header = pair.header()
    if not header:
        pytest.skip(f"no cached export for {pair.key}")
    try:
        plan = pair.first_admitted_facade(level).resolve_source(evidence(header))
    except IncompatibleSourceError as exc:
        pytest.skip(str(exc))

    decoded = json.loads(resolved_plan_json(plan))

    assert decoded["level"] == level
    assert decoded["provenance"]["quantification_level"] == level
    assert decoded["read"]["projected_columns"] == list(plan.read.projected_columns)
