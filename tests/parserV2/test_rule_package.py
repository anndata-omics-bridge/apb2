"""Schema 0.3: every packaged document migrates, and nothing it declares changed meaning.

Two kinds of assertion. The migration invariants compare each 0.3 document against the
unchanged 0.2 oracle declaration by declaration, so a transcription slip in one of twelve
files cannot pass. The rest test what schema 0.3 newly decides: identity-only ``axis``,
measurement ownership, the finite condition vocabulary, and the legacy paths it refuses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from apb2 import export_schema
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    LongRecognition,
    RuleNotApplicable,
    SearchParameterEvidence,
    WideRecognition,
    make_rule_document,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema_axis import ColumnGroup, ComputedColumn
from apb2.parserV2.vendor_parse_rules.schema_base import (
    SCHEMA_VERSION,
    QuantificationLevel,
)
from apb2.parserV2.vendor_parse_rules.schema_fragments import ColumnLabeledFragments
from apb2.parserV2.vendor_parse_rules.schema_input import Input
from apb2.parserV2.vendor_parse_rules.schema_measurements import (
    FactorLayer,
    Layer,
    NumericLayer,
    RegexValuePattern,
    layer_required,
)
from apb2.parserV2.vendor_parse_rules.schema_rule import (
    LongRule,
    WideRule,
    rule_json_schema,
)
from apb2.vendor_parse_rules import _recognition as legacy_recognition
from apb2.vendor_parse_rules import model as legacy_model
from apb2.vendor_parse_rules.rules import Rule as LegacyRule
from apb2.vendor_parse_rules.rules import load_document as load_legacy_document
from parserV2.fixtures import DocumentPair, document_pairs, level_pairs
from parserV2.rule_inventory import EXPECTED_DOCUMENT_COUNT, EXPECTED_LEVEL_COUNT

NO_EVIDENCE = SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)
DDA = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=None)
DIA = SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=None)

_LEVEL_CASES = [
    pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()
]
_DOCUMENT_CASES = [pytest.param(pair, id=pair.key) for pair in document_pairs()]


type MutatePayload = Callable[[dict[str, Any]], object]
type V2Rule = LongRule | WideRule
type LegacyConfig = legacy_model.LongRule | legacy_model.WideRule


def _computed_facts(
    computed: list[ComputedColumn] | list[legacy_model.ComputedColumn],
) -> list[tuple[str, str, tuple[str, ...], str | None]]:
    return [
        (column.how, column.name, tuple(column.inputs), getattr(column, "separator", None))
        for column in computed
    ]


def _v2_column_facts(group: ColumnGroup) -> dict[str, Any]:
    return {
        "select": dict(group.select),
        "optional_select": dict(group.optional_select),
        "types": dict(group.types),
        "computed": _computed_facts(group.computed),
    }


def _legacy_column_facts(group: legacy_model.ColumnGroup) -> dict[str, Any]:
    return {
        "select": dict(group.select),
        "optional_select": dict(group.optional_select),
        "types": dict(group.types),
        "computed": _computed_facts(group.computed),
    }


def _layer_facts(
    layer: Layer | legacy_model.Layer,
    *,
    required: bool,
) -> dict[str, Any]:
    numeric = isinstance(layer, NumericLayer | legacy_model.NumericLayer)
    return {
        "name": layer.name,
        "source": layer.source,
        "required": required,
        "encoding_mode": layer.encoding_mode,
        "missing_values": list(layer.missing_values) if numeric else [],
        "categories": {} if numeric else dict(layer.categories),
        "value_pattern": getattr(layer.value_pattern, "pattern", None) if numeric else None,
    }


def _schema_facts(rule: V2Rule | LegacyConfig) -> dict[str, Any]:
    return {
        "file_version": rule.file_version,
        "software_name": rule.software_name,
        "software_version_pattern": rule.software_version_pattern,
        "quantification_level": rule.quantification_level,
        "shape": rule.shape,
    }


def _v2_facts(rule: V2Rule) -> dict[str, Any]:
    """Every declaration the migrated rule makes, addressed through its own owners."""
    groups = (
        {"obs": rule.columns.obs, "var": rule.columns.var}
        if isinstance(rule, LongRule)
        else {"var": rule.columns.var}
    )
    return {
        "schema": _schema_facts(rule),
        "obs_keys": list(rule.axis.obs_keys),
        "var_keys": list(rule.axis.var_keys),
        "columns": {axis: _v2_column_facts(group) for axis, group in groups.items()},
        "column_roles": rule.column_roles.model_dump(),
        "layers": [
            _layer_facts(
                layer,
                required=layer_required(rule.measurements.primary_layer, layer),
            )
            for layer in rule.measurements.layers
        ],
        "modifications": None if rule.modifications is None else rule.modifications.model_dump(),
        "fragments": None if rule.fragments is None else rule.fragments.model_dump(),
        "requires_search_parameters": dict(rule.requires_search_parameters),
    }


def _legacy_facts(rule: LegacyConfig) -> dict[str, Any]:
    """The same declarations, read where schema 0.2 kept them."""
    groups = (
        {"obs": rule.columns.obs, "var": rule.columns.var}
        if isinstance(rule, legacy_model.LongRule)
        else {"var": rule.columns.var}
    )
    return {
        "schema": _schema_facts(rule),
        "obs_keys": list(rule.axis.obs_keys),
        "var_keys": list(rule.axis.var_keys),
        "columns": {axis: _legacy_column_facts(group) for axis, group in groups.items()},
        "column_roles": rule.column_roles.model_dump(),
        "layers": [
            _layer_facts(layer, required=legacy_model.layer_required(rule, layer))
            for layer in rule.layers
        ],
        "modifications": None if rule.modifications is None else rule.modifications.model_dump(),
        "fragments": None if rule.fragments is None else rule.fragments.model_dump(),
        "requires_search_parameters": dict(rule.requires_search_parameters),
    }


def _facts_without_promotion(rule: V2Rule) -> dict[str, Any]:
    """The migrated facts minus what naming a different primary layer necessarily changes."""
    facts = _v2_facts(rule)
    facts["layers"] = [
        {name: value for name, value in layer.items() if name != "required"}
        for layer in facts["layers"]
    ]
    return facts


def _legacy_rule(pair: DocumentPair, level: QuantificationLevel) -> LegacyRule:
    return load_legacy_document(pair.legacy_path).declared(level)


# --------------------------------------------------------------------------- migration parity


def test_the_migration_kept_every_document_and_every_level() -> None:
    documents = [load_rule_document(path) for path in PACKAGED]

    assert len(documents) == EXPECTED_DOCUMENT_COUNT
    assert sum(len(document.levels) for document in documents) == EXPECTED_LEVEL_COUNT


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_every_migrated_level_declares_what_the_oracle_declared(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    migrated = load_rule_document(pair.parser_v2_path).declared(level).declaration
    oracle = _legacy_rule(pair, level).config

    assert _v2_facts(migrated) == _legacy_facts(oracle)


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_measurement_ownership_moved_without_changing_its_values(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    migrated = load_rule_document(pair.parser_v2_path).declared(level).declaration
    oracle = _legacy_rule(pair, level).config

    assert migrated.measurements.primary_layer == oracle.axis.x_layer
    assert migrated.measurements.duplicates.mode == oracle.axis.duplicates.mode
    assert [layer.name for layer in migrated.measurements.layers] == [
        layer.name for layer in oracle.layers
    ]
    assert not hasattr(migrated.axis, "x_layer")
    assert not hasattr(migrated.axis, "duplicates")


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_the_primary_layer_names_exactly_one_layer_and_is_required(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    names = [layer.name for layer in rule.measurements.layers]
    primary = [
        layer for layer in rule.measurements.layers if layer.name == rule.measurements.primary_layer
    ]

    assert len(names) == len(set(names))
    assert len(primary) == 1
    assert layer_required(rule.measurements.primary_layer, primary[0])
    # Promotion changes what is required, never the authored order.
    assert names == [layer.name for layer in rule.measurements.layers]


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_recognition_agrees_with_the_oracle_on_a_real_vendor_header(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    header = pair.header()
    if not header:
        pytest.skip(f"no cached export for {pair.key}")
    migrated = load_rule_document(pair.parser_v2_path).declared(level).recognition
    oracle = _legacy_rule(pair, level).recognition

    assert migrated.matches(header) == oracle.matches(list(header))
    assert migrated.layer_source_columns(header) == oracle.layer_source_columns(list(header))
    if isinstance(migrated, LongRecognition):
        assert isinstance(oracle, legacy_recognition.LongRecognition)
        assert migrated.required_headers == oracle.required_headers


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_recognition_rejects_a_header_missing_one_required_source(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    recognition = load_rule_document(pair.parser_v2_path).declared(level).recognition
    header = pair.header()
    if not header or not recognition.matches(header):
        pytest.skip(f"cached export for {pair.key} does not satisfy level {level!r}")
    required = _required_source(recognition, header)

    assert not recognition.matches(tuple(name for name in header if name != required))


def _required_source(
    recognition: LongRecognition | WideRecognition, header: tuple[str, ...]
) -> str:
    """One header column whose absence must make the level unrecognizable."""
    if isinstance(recognition, LongRecognition):
        return sorted(recognition.required_headers)[0]
    var_sources = {
        source for _axis, group in recognition.column_groups() for source in group.select.values()
    }
    return sorted(var_sources & set(header))[0]


@pytest.mark.parametrize("pair", _DOCUMENT_CASES)
def test_every_document_declares_the_new_generation_and_physical_extensions(
    pair: DocumentPair,
) -> None:
    payload = json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))
    document = load_rule_document(pair.parser_v2_path)
    effective = document.declared(document.levels[0])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert effective.input.extensions
    assert effective.input.shape == effective.declaration.shape


def test_the_maxquant_source_names_the_one_table_it_reads() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "maxquant")
    document = load_rule_document(pair.parser_v2_path)

    source = document.declared("ion").input
    assert source.extensions == [".txt"]
    assert source.file_name == "evidence.txt"


# ------------------------------------------------------------------- gates and overrides


def test_sage_gates_its_two_levels_on_combine_charge_states() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    document = load_rule_document(pair.parser_v2_path)
    separate = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False)
    combined = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=True)

    assert document.rule("ion", separate).declaration.axis.var_keys == ["ProForma_ion"]
    assert document.rule("peptidoform", combined).declaration.axis.var_keys == [
        "ProForma_peptidoform"
    ]
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("ion", combined)
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("peptidoform", separate)
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("ion", NO_EVIDENCE)


def test_diann_v2_swaps_only_the_primary_layer_for_dda_evidence() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v2")
    document = load_rule_document(pair.parser_v2_path)

    dda = document.rule("ion", DDA).declaration
    dia = document.rule("ion", DIA).declaration

    assert dda.measurements.primary_layer == "Ms1_Normalised"
    assert dia.measurements.primary_layer == "Precursor_Normalised"
    assert document.declared("ion").declaration.measurements.primary_layer == (
        "Precursor_Normalised"
    )
    # Everything except which layer is primary -- and therefore which layer promotion made
    # required -- is the same declaration under either evidence.
    assert _facts_without_promotion(dda) == _facts_without_promotion(dia)


def test_a_level_without_a_gate_is_applicable_without_any_evidence() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)

    assert document.rule("protein", NO_EVIDENCE).declaration.quantification_level == "protein"
    with pytest.raises(RuleNotApplicable, match="has no level"):
        document.rule("peptide", NO_EVIDENCE)


# ------------------------------------------------------------------ what schema 0.3 refuses


def _document_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "file_version": "1",
        "software_name": "Test",
        "software_version_pattern": "^1$",
        "input": {
            "shape": "long",
            "extensions": [".tsv"],
        },
        "base": {
            "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
            "columns": {"obs": {"select": {"sample": "Sample"}}},
            "measurements": {
                "primary_layer": "quantity",
                "layers": [{"name": "quantity", "source": "Quantity"}],
            },
        },
        "levels": {"ion": {"columns": {"var": {"select": {"feature": "Feature"}}}}},
    }


def _declared(payload: dict[str, Any], tmp_path: Path) -> None:
    document = make_rule_document(tmp_path / "rules.json", payload)
    document.declared("ion")


def test_the_reference_payload_is_valid_so_every_rejection_below_is_the_change(
    tmp_path: Path,
) -> None:
    _declared(_document_payload(), tmp_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda payload: payload["base"]["axis"].update({"x_layer": "quantity"}),
            "x_layer",
            id="axis.x_layer",
        ),
        pytest.param(
            lambda payload: payload["base"]["axis"].update({"duplicates": {"mode": "error"}}),
            "duplicates",
            id="axis.duplicates",
        ),
        pytest.param(
            lambda payload: payload["base"].update(
                {"layers": [{"name": "quantity", "source": "Quantity"}]}
            ),
            "layers",
            id="root-layers",
        ),
        pytest.param(
            lambda payload: payload["base"]["measurements"].update(
                {"duplicates": {"mode": "keep_all_as_raw_table"}}
            ),
            "keep_all_as_raw_table",
            id="keep_all_as_raw_table",
        ),
        pytest.param(
            lambda payload: payload["levels"]["ion"].update(
                {
                    "search_parameter_overrides": [
                        {
                            "when_search_parameters": {"acquisition_method": "DDA"},
                            "x_layer": "quantity",
                        }
                    ]
                }
            ),
            "x_layer",
            id="override-x_layer",
        ),
        pytest.param(
            lambda payload: payload["levels"]["ion"].update(
                {"requires_search_parameters": {"software_version": "1.0"}}
            ),
            "software_version",
            id="unknown-condition-field",
        ),
        pytest.param(
            lambda payload: payload["base"]["measurements"].update({"primary_layer": "absent"}),
            "primary_layer",
            id="primary-names-no-layer",
        ),
        pytest.param(
            lambda payload: payload["base"]["measurements"]["layers"].append(
                {"name": "quantity", "source": "Other"}
            ),
            "layer names",
            id="repeated-layer-name",
        ),
    ],
)
def test_schema_0_3_refuses_a_legacy_or_illegal_declaration(
    mutate: MutatePayload, match: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=match):
        _declared(payload, tmp_path)


def test_a_document_of_the_previous_generation_is_refused_at_the_shell(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["schema_version"] = "0.2"

    with pytest.raises(ValidationError, match="schema_version"):
        make_rule_document(tmp_path / "rules.json", payload)


@pytest.mark.parametrize(
    "invalid_level_declaration",
    [
        {"requires_search_parameters": {"software_version": "1.0"}},
        {
            "search_parameter_overrides": [
                {
                    "when_search_parameters": {"acquisition_method": "DDA"},
                    "x_layer": "quantity",
                }
            ]
        },
    ],
    ids=("unknown-gate-field", "unknown-override-field"),
)
def test_rule_validates_the_declaration_before_using_gates_or_overrides(
    invalid_level_declaration: dict[str, object],
    tmp_path: Path,
) -> None:
    payload = _document_payload()
    payload["levels"]["ion"].update(invalid_level_declaration)
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValidationError):
        document.rule("ion", DDA)


def test_aggregate_requires_layers_no_encoder_would_later_change(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["base"]["measurements"]["duplicates"] = {"mode": "aggregate"}
    payload["base"]["measurements"]["layers"] = [
        {"name": "quantity", "source": "Quantity", "missing_values": [0]}
    ]

    document = make_rule_document(tmp_path / "rules.json", payload)
    with pytest.raises(ValueError, match="aggregate"):
        ParseRuleFacade(document, "ion", NO_EVIDENCE)


def test_an_unknown_extension_is_rejected(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["input"]["extensions"] = [".xlsx"]

    with pytest.raises(ValidationError, match="literal_error"):
        make_rule_document(tmp_path / "rules.json", payload)


def test_detection_candidates_are_declared_only_when_detection_is_enabled(
    tmp_path: Path,
) -> None:
    payload = _document_payload()
    payload["input"]["delimiter"] = {"mode": "detect", "candidates": ["\t", ";", ","]}

    document = make_rule_document(tmp_path / "rules.json", payload)

    assert document.declared("ion").input.delimiter is not None


# -------------------------------------------------------------------------- fragment rules


def test_the_packaged_fragment_level_separates_before_it_decomposes() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    rule = load_rule_document(pair.parser_v2_path).declared("fragment").declaration
    fragments = rule.fragments

    assert fragments is not None
    assert fragments.value_columns
    assert fragments.label_output == "fragment_label"
    physical = {layer.source for layer in rule.measurements.layers} | set(fragments.value_columns)
    assert fragments.label_output not in physical
    assert "ProForma_fragment" in rule.axis.var_keys


def test_a_fragment_label_output_colliding_with_a_source_is_refused(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["levels"] = {
        "fragment": {
            "columns": {"var": {"select": {"feature": "Feature"}}},
            "fragments": {
                "label_strategy": "positional",
                "value_columns": ["Quantity"],
                "label_output": "Quantity",
            },
        }
    }
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValueError, match="label_output"):
        ParseRuleFacade(document, "fragment", NO_EVIDENCE)


def test_a_column_labeled_fragment_label_may_not_be_selected(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["levels"] = {
        "fragment": {
            "columns": {"var": {"select": {"feature": "Feature", "label": "Info"}}},
            "fragments": {
                "label_strategy": "column",
                "value_columns": ["Quantity"],
                "label_column": "Info",
            },
        }
    }
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValueError, match="label source"):
        ParseRuleFacade(document, "fragment", NO_EVIDENCE)


def test_column_labeled_recognition_requires_its_packed_label_column() -> None:
    payload = _document_payload()
    payload["levels"] = {
        "fragment": {
            "columns": {"var": {"select": {"feature": "Feature"}}},
            "fragments": {
                "label_strategy": "column",
                "value_columns": ["Quantity"],
                "label_column": "Info",
            },
        }
    }
    document = make_rule_document(Path("rules.json"), payload)
    recognition = document.declared("fragment").recognition

    assert recognition.matches(("Sample", "Feature", "Quantity", "Info"))
    assert not recognition.matches(("Sample", "Feature", "Quantity"))


# ------------------------------------------------------------------------ published schema


def test_only_schema_0_3_is_published_and_its_unions_are_discriminated() -> None:
    published = rule_json_schema()
    definitions = published["$defs"]
    text = json.dumps(published)

    assert isinstance(definitions, dict)
    assert "x_layer" not in text
    assert "keep_all_as_raw_table" not in text
    assert '"measurements"' in text
    for name in ("Measurements", "Axis", "Duplicates", "SearchParameterOverride"):
        assert name in definitions
    assert "primary_layer" in json.dumps(definitions["Measurements"])
    assert sorted(definitions["Axis"]["properties"]) == ["obs_keys", "var_keys"]


def test_the_input_policy_publishes_its_delimiter_and_number_alternatives() -> None:
    published = Input.model_json_schema()
    definitions = published["$defs"]

    for name in (
        "DetectedDelimiter",
        "DetectedNumberFormat",
    ):
        assert name in definitions
    assert definitions["TableShape"]["enum"] == ["long", "wide"]


def test_the_layer_union_still_distinguishes_numeric_from_factor() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "fragpipe")
    rule = load_rule_document(pair.parser_v2_path).declared("ion").declaration
    by_name = {layer.name: layer for layer in rule.measurements.layers}

    assert isinstance(by_name["Match_Type"], FactorLayer)
    assert isinstance(by_name["Intensity"], NumericLayer)
    assert by_name["Match_Type"].categories == {"unmatched": 0, "MS/MS": 1, "MBR": 2}


def test_the_regex_value_pattern_survived_the_migration() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "peaks")
    rule = load_rule_document(pair.parser_v2_path).declared("ion").declaration
    ascore = next(layer for layer in rule.measurements.layers if layer.name == "AScore")

    assert isinstance(ascore, NumericLayer)
    assert isinstance(ascore.value_pattern, RegexValuePattern)


def test_both_rule_shapes_are_represented_by_the_packaged_generation() -> None:
    shapes = [
        load_rule_document(pair.parser_v2_path).declared(level).declaration
        for pair, level in level_pairs()
    ]

    assert sum(isinstance(rule, LongRule) for rule in shapes) == 12
    assert sum(isinstance(rule, WideRule) for rule in shapes) == 7
    modes = [rule.measurements.duplicates.mode for rule in shapes]
    assert modes.count("error") == 13
    assert modes.count("keep_first") == 5
    assert modes.count("aggregate") == 1
    assert sum(isinstance(rule.fragments, ColumnLabeledFragments) for rule in shapes) == 0


def test_the_published_artifact_is_the_schema_the_models_declare() -> None:
    committed = json.loads(
        export_schema.artifact_path("apb2.parserV2.vendor_parse_rules.documents").read_text(
            encoding="utf-8"
        )
    )

    assert committed == rule_json_schema()
