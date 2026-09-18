"""Schema 0.7 packaged-document, composition, and validation contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    LongRecognition,
    RuleNotApplicable,
    SearchParameterEvidence,
    WideRecognition,
    make_rule_document,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.axis import computed_columns
from apb2.parserV2.vendor_parse_rules.schema.base import (
    SCHEMA_VERSION,
    QuantificationLevel,
)
from apb2.parserV2.vendor_parse_rules.schema.fragments import ColumnLabeledFragments
from apb2.parserV2.vendor_parse_rules.schema.input import Input
from apb2.parserV2.vendor_parse_rules.schema.measurements import (
    FactorLayer,
    NumericLayer,
    RegexValuePattern,
    layer_required,
)
from apb2.parserV2.vendor_parse_rules.schema.roles import ROLE_CONFIG
from apb2.parserV2.vendor_parse_rules.schema.rule import (
    LongRule,
    WideRule,
    rule_json_schema,
)
from apb2.parserV2.vendor_parse_rules.schema_artifact import artifact_path
from parserV2.fixtures import PackagedDocument, document_pairs, level_pairs
from parserV2.rule_inventory import EXPECTED_DOCUMENT_COUNT, EXPECTED_LEVEL_COUNT

NO_EVIDENCE = SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)
DDA = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=None)
DIA = SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=None)

_LEVEL_CASES = [
    pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()
]
_DOCUMENT_CASES = [pytest.param(pair, id=pair.key) for pair in document_pairs()]
_NON_PRIMARY_ABUNDANCE: dict[tuple[str, QuantificationLevel], tuple[str, ...]] = {
    ("alphapept", "ion"): (
        "MS1_Int_Sum_Apex",
        "MS1_Int_Sum_Area",
        "MS1_Int_Max_Apex",
        "MS1_Int_Max_Area",
    ),
    ("diann/v1_8", "ion"): ("Ms1_Normalised", "Precursor_Quantity", "Ms1_Area"),
    ("diann/v1_8", "protein"): ("PG_Normalised", "PG_Quantity", "Genes_MaxLFQ"),
    ("diann/v1_7", "ion"): ("Precursor_Quantity", "Ms1_Area"),
    ("diann/v1_7", "protein"): ("PG_Quantity", "Genes_MaxLFQ"),
    ("diann/v2", "ion"): ("Ms1_Normalised", "Precursor_Quantity", "Ms1_Area"),
    ("diann/v2", "protein"): ("Genes_MaxLFQ",),
    ("maxquant", "peptide"): ("LFQ_Intensity",),
    ("maxquant", "protein"): ("LFQ_Intensity", "iBAQ"),
    ("msangel", "ion"): ("Raw_Abundance",),
    ("spectronaut", "ion"): (
        "EG_ReferenceQuantity_Settings",
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
    ),
    ("spectronaut/v21", "ion"): (
        "EG_ReferenceQuantity_Settings",
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
    ),
    ("spectronaut/v15", "ion"): (
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
        "FG_MS1RawQuantity",
        "FG_MS2RawQuantity",
    ),
    ("spectronaut/v15", "fragment"): ("F_PeakHeight",),
}


type MutatePayload = Callable[[dict[str, Any]], object]
type V2Rule = LongRule | WideRule


def _without_primary_layer(rule: V2Rule) -> dict[str, Any]:
    payload = rule.model_dump(mode="json")
    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    measurements.pop("primary_layer")
    return payload


# --------------------------------------------------------------------------- migration parity


def test_entry_shaped_columns_project_roles_and_runtime_selections(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["layers"][0]["roles"] = ["abundance"]
    payload["tables"][0]["levels"]["ion"]["columns"]["var"] = [
        {
            "name": "feature",
            "source": "Feature",
            "type": "integer",
            "roles": ["protein_assignment", "fasta_accessions"],
        }
    ]

    document = make_rule_document(tmp_path / "rules.json", payload)
    working = ParseRuleFacade(document, "ion", NO_EVIDENCE).working_parameters

    assert working.var.columns.required_selections[0].logical_type == "integer"
    assert working.provenance["column_roles"] == {
        "protein_assignment": "feature",
        "fasta_accessions": "feature",
    }
    assert working.provenance["layer_roles"] == {"abundance": ["quantity"]}


def test_role_configuration_owns_the_role_vocabulary() -> None:
    assert {
        "obs": frozenset(),
        "var": frozenset({"fasta_accessions", "protein_assignment"}),
        "layer": frozenset({"abundance"}),
    } == ROLE_CONFIG


def test_entry_role_is_rejected_on_an_unconfigured_owner(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["columns"]["obs"][0]["roles"] = ["protein_assignment"]

    with pytest.raises(ValidationError, match=r"not allowed on obs"):
        _declared(payload, tmp_path)


def test_role_is_rejected_on_an_unconfigured_layer(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["layers"][0]["roles"] = ["protein_assignment"]

    with pytest.raises(ValidationError, match=r"not allowed on layer"):
        _declared(payload, tmp_path)


def test_one_role_cannot_name_two_columns(tmp_path: Path) -> None:
    payload = _document_payload()
    columns = payload["tables"][0]["levels"]["ion"]["columns"]["var"]
    columns[0]["roles"] = ["protein_assignment"]
    columns.append({"name": "protein", "source": "Protein", "roles": ["protein_assignment"]})

    with pytest.raises(ValidationError, match=r"roles must be unique on columns\.var"):
        _declared(payload, tmp_path)


def test_the_migration_kept_every_document_and_every_level() -> None:
    documents = [load_rule_document(path) for path in PACKAGED]

    assert len(documents) == EXPECTED_DOCUMENT_COUNT
    assert sum(len(document.levels) for document in documents) == EXPECTED_LEVEL_COUNT


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_measurement_ownership_is_separate_from_axis_identity(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration

    assert rule.measurements.primary_layer
    assert rule.measurements.duplicates.mode in {"error", "keep_first", "aggregate"}
    assert not hasattr(rule.axis, "x_layer")
    assert not hasattr(rule.axis, "duplicates")


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_the_primary_layer_names_exactly_one_layer_and_is_required(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    names = [layer.name for layer in rule.measurements.layers]
    primary = [
        layer for layer in rule.measurements.layers if layer.name == rule.measurements.primary_layer
    ]

    assert len(names) == len(set(names))
    assert len(primary) == 1
    assert layer_required(rule.measurements.primary_layer, primary[0])
    assert "abundance" in primary[0].roles
    # Promotion changes what is required, never the authored order.
    assert names == [layer.name for layer in rule.measurements.layers]


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_every_declared_abundance_layer_is_tagged(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    actual = {layer.name for layer in rule.measurements.layers if "abundance" in layer.roles}
    expected = {
        rule.measurements.primary_layer,
        *_NON_PRIMARY_ABUNDANCE.get((pair.key, level), ()),
    }

    assert actual == expected


def test_packaged_integer_measurements_are_exactly_the_declared_counts() -> None:
    actual = {
        (pair.key, level, layer.name)
        for pair, level in level_pairs()
        for layer in load_rule_document(pair.parser_v2_path)
        .declared(level)
        .declaration.measurements.layers
        if isinstance(layer, NumericLayer) and layer.type == "integer"
    }

    assert actual == {
        ("fragpipe", "ion", "Spectral_Count"),
        ("maxquant", "ion", "MS_MS_Count"),
        ("msangel", "ion", "PSM_Count"),
        ("prolinestudio", "ion", "PSM_Count"),
        ("spectronaut", "protein", "PG_RunEvidenceCount"),
        ("spectronaut/v21", "protein", "PG_RunEvidenceCount"),
        ("wombat", "peptidoform", "Number_Of_Psms"),
    }


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_recognition_rejects_a_header_missing_one_required_source(
    pair: PackagedDocument, level: QuantificationLevel
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
        column.source
        for _axis, group in recognition.column_groups()
        for column in group
        if column.source is not None and column.required
    }
    return sorted(var_sources & set(header))[0]


@pytest.mark.parametrize("pair", _DOCUMENT_CASES)
def test_every_document_declares_the_new_generation_and_physical_extensions(
    pair: PackagedDocument,
) -> None:
    payload = json.loads(pair.parser_v2_path.read_text(encoding="utf-8"))
    document = load_rule_document(pair.parser_v2_path)
    effective = document.declared(document.levels[0])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert effective.input.extensions
    assert effective.input.shape == effective.declaration.shape


def test_maxquant_keeps_evidence_outside_the_higher_level_prepared_table() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "maxquant")
    document = load_rule_document(pair.parser_v2_path)

    source = document.declared("ion").input
    assert source.extensions == [".txt"]
    assert source.file_name == "evidence.txt"
    assert document.table_levels == (("ion",), ("peptidoform", "peptide", "protein"))
    assert document.declared("ion").preparation is None
    for level in ("peptidoform", "peptide", "protein"):
        assert document.declared(level).preparation == "maxquant"
        assert document.declared(level).declaration.axis.obs_keys == ["Experiment"]


def test_peaks_declares_the_persisted_sample_annotation_matching_policy() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "peaks")
    document = load_rule_document(pair.parser_v2_path)
    effective = document.declared("ion").declaration
    provenance = ParseRuleFacade.from_declared_rule(
        document,
        "ion",
    ).working_parameters.provenance

    assert effective.sample_annotation is not None
    assert effective.sample_annotation.matching.mode == "fuzzy"
    assert provenance["sample_annotation_matching"] == {
        "mode": "fuzzy",
        "cutoff": 0.6,
        "margin": 0.01,
        "near_miss_limit": 3,
        "normalize": "mass_spec_basename",
    }


@pytest.mark.parametrize(
    "key",
    (
        "quantms",
        "i2masschroq",
        "prolinestudio",
        "diann/v1_7",
        "diann/v1_8",
        "diann/v2",
    ),
)
def test_file_backed_run_names_declare_exact_basename_matching(key: str) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == key)
    document = load_rule_document(pair.parser_v2_path)
    effective = document.declared(document.levels[0]).declaration
    provenance = pair.first_admitted_facade().working_parameters.provenance

    assert effective.sample_annotation is not None
    assert effective.sample_annotation.matching.mode == "exact"
    assert provenance["sample_annotation_matching"] == {
        "mode": "exact",
        "normalize": "mass_spec_basename",
    }


def test_msangel_measurements_canonicalize_the_dda_run_prefix() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "msangel")
    rule = load_rule_document(pair.parser_v2_path).declared("ion").declaration
    assert isinstance(rule, WideRule)

    for layer in rule.measurements.layers:
        column = layer.source.replace("^", "").replace("$", "")
        prefix = column.split("(?P<sample>", maxsplit=1)[0]
        name = prefix.replace("(?:DDA_)?", "DDA_") + "Condition_A_Sample_Alpha_01"
        match = re.fullmatch(layer.source, name)
        assert match is not None
        assert match.group("sample") == "Condition_A_Sample_Alpha_01"


@pytest.mark.parametrize(
    ("document_key", "level", "protein_assignment", "fasta_accessions"),
    [
        pytest.param("diann/v1_8", "ion", "Protein_Group", "Protein_Ids", id="diann-v1_8-ion"),
        pytest.param(
            "diann/v1_8", "protein", "Protein_Group", "Protein_Ids", id="diann-v1_8-protein"
        ),
        pytest.param("diann/v2", "ion", "Protein_Group", "Protein_Ids", id="diann-v2-ion"),
        pytest.param("diann/v2", "protein", "Protein_Group", "Protein_Ids", id="diann-v2-protein"),
        pytest.param(
            "spectronaut",
            "ion",
            "PG_ProteinGroups",
            "PG_ProteinAccessions",
            id="spectronaut-ion",
        ),
        pytest.param(
            "spectronaut",
            "fragment",
            "PG_ProteinGroups",
            "PG_ProteinAccessions",
            id="spectronaut-fragment",
        ),
        pytest.param(
            "spectronaut",
            "protein",
            "PG_ProteinGroups",
            "PG_ProteinAccessions",
            id="spectronaut-protein",
        ),
    ],
)
def test_protein_assignment_names_the_group_not_its_accessions(
    document_key: str,
    level: QuantificationLevel,
    protein_assignment: str,
    fasta_accessions: str,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == document_key)
    document = load_rule_document(pair.parser_v2_path)
    roles = ParseRuleFacade(document, level, NO_EVIDENCE).working_parameters.provenance[
        "column_roles"
    ]

    assert roles == {
        "protein_assignment": protein_assignment,
        "fasta_accessions": fasta_accessions,
    }


def test_spectronaut_fragment_exposes_its_parent_ion_identity() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    fragment = load_rule_document(pair.parser_v2_path).declared("fragment").declaration

    computed = {column.name: column for column in computed_columns(fragment.columns.var)}

    assert computed["ProForma_ion"].inputs == ["ProForma_peptidoform", "FG_Charge"]


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
    assert _without_primary_layer(dda) == _without_primary_layer(dia)


def test_a_level_without_a_gate_is_applicable_without_any_evidence() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    document = load_rule_document(pair.parser_v2_path)

    assert document.rule("protein", NO_EVIDENCE).declaration.quantification_level == "protein"
    with pytest.raises(RuleNotApplicable, match="has no level"):
        document.rule("peptide", NO_EVIDENCE)


# ------------------------------------------------------------------ what schema 0.7 refuses


def _document_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "file_version": "1",
        "software_name": "Test",
        "software_version_pattern": "^1$",
        "tables": [
            {
                "input": {
                    "shape": "long",
                    "extensions": [".tsv"],
                },
                "base": {
                    "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
                    "columns": {"obs": [{"name": "sample", "source": "Sample"}]},
                    "measurements": {
                        "primary_layer": "quantity",
                        "layers": [{"name": "quantity", "source": "Quantity"}],
                    },
                },
                "levels": {"ion": {"columns": {"var": [{"name": "feature", "source": "Feature"}]}}},
            }
        ],
    }


def _declared(payload: dict[str, Any], tmp_path: Path) -> None:
    document = make_rule_document(tmp_path / "rules.json", payload)
    document.declared("ion")


def test_the_reference_payload_is_valid_so_every_rejection_below_is_the_change(
    tmp_path: Path,
) -> None:
    _declared(_document_payload(), tmp_path)


def test_numeric_layer_type_defaults_to_number_and_rejects_unknown_values() -> None:
    assert NumericLayer(name="quantity", source="Quantity").type == "number"
    assert NumericLayer(name="count", source="Count", type="integer").type == "integer"

    with pytest.raises(ValidationError, match="type"):
        NumericLayer.model_validate({"name": "quantity", "source": "Quantity", "type": "float"})


def test_factor_layers_do_not_acquire_a_numeric_type() -> None:
    payload = {
        "encoding_mode": "factor",
        "name": "status",
        "source": "Status",
        "categories": {"identified": 1},
        "type": "integer",
    }

    with pytest.raises(ValidationError, match="type"):
        FactorLayer.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["axis"].update({"x_layer": "quantity"}),
            "x_layer",
            id="axis.x_layer",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["axis"].update(
                {"duplicates": {"mode": "error"}}
            ),
            "duplicates",
            id="axis.duplicates",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"].update(
                {"layers": [{"name": "quantity", "source": "Quantity"}]}
            ),
            "layers",
            id="root-layers",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["measurements"].update(
                {"duplicates": {"mode": "keep_all_as_raw_table"}}
            ),
            "keep_all_as_raw_table",
            id="keep_all_as_raw_table",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["ion"].update(
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
            lambda payload: payload["tables"][0]["levels"]["ion"].update(
                {"requires_search_parameters": {"software_version": "1.0"}}
            ),
            "software_version",
            id="unknown-condition-field",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["measurements"].update(
                {"primary_layer": "absent"}
            ),
            "primary_layer",
            id="primary-names-no-layer",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["measurements"]["layers"].append(
                {"name": "quantity", "source": "Other"}
            ),
            "layer names",
            id="repeated-layer-name",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["columns"]["obs"].append(
                {"name": "sample", "source": "Other"}
            ),
            "column entry names",
            id="repeated-column-name",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["columns"].update(
                {"obs": {"select": {"sample": "Sample"}}}
            ),
            "list_type",
            id="column-side-map",
        ),
    ],
)
def test_schema_0_4_refuses_a_legacy_or_illegal_declaration(
    mutate: MutatePayload, match: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=match):
        _declared(payload, tmp_path)


@pytest.mark.parametrize("version", ["0.3", "0.5", "0.6"])
def test_a_document_of_a_previous_generation_is_refused_at_the_shell(
    version: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    payload["schema_version"] = version

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
    payload["tables"][0]["levels"]["ion"].update(invalid_level_declaration)
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValidationError):
        document.rule("ion", DDA)


def test_aggregate_requires_layers_no_encoder_would_later_change(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["duplicates"] = {"mode": "aggregate"}
    payload["tables"][0]["base"]["measurements"]["layers"] = [
        {"name": "quantity", "source": "Quantity", "missing_values": [0]}
    ]

    document = make_rule_document(tmp_path / "rules.json", payload)
    with pytest.raises(ValueError, match="aggregate"):
        ParseRuleFacade(document, "ion", NO_EVIDENCE)


def test_an_unknown_extension_is_rejected(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["input"]["extensions"] = [".xls"]

    with pytest.raises(ValidationError, match="literal_error"):
        make_rule_document(tmp_path / "rules.json", payload)


def test_detection_candidates_are_declared_only_when_detection_is_enabled(
    tmp_path: Path,
) -> None:
    payload = _document_payload()
    payload["tables"][0]["input"]["delimiter"] = {"mode": "detect", "candidates": ["\t", ";", ","]}

    document = make_rule_document(tmp_path / "rules.json", payload)

    assert document.declared("ion").input.delimiter is not None


# -------------------------------------------------------------------------- fragment rules


def test_the_packaged_fragment_level_separates_before_it_decomposes() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
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
    payload["tables"][0]["levels"] = {
        "fragment": {
            "columns": {"var": [{"name": "feature", "source": "Feature"}]},
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
    payload["tables"][0]["levels"] = {
        "fragment": {
            "columns": {
                "var": [
                    {"name": "feature", "source": "Feature"},
                    {"name": "label", "source": "Info"},
                ]
            },
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
    payload["tables"][0]["levels"] = {
        "fragment": {
            "columns": {"var": [{"name": "feature", "source": "Feature"}]},
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


def test_only_schema_0_4_is_published_and_its_unions_are_discriminated() -> None:
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
    assert definitions["SemanticRole"]["enum"] == sorted(set().union(*ROLE_CONFIG.values()))


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

    assert sum(isinstance(rule, LongRule) for rule in shapes) == 26
    assert sum(isinstance(rule, WideRule) for rule in shapes) == 9
    modes = [rule.measurements.duplicates.mode for rule in shapes]
    assert modes.count("error") == 18
    assert modes.count("keep_first") == 16
    assert modes.count("aggregate") == 1
    assert sum(isinstance(rule.fragments, ColumnLabeledFragments) for rule in shapes) == 0


def test_the_published_artifact_is_the_schema_the_models_declare() -> None:
    committed = json.loads(artifact_path().read_text(encoding="utf-8"))

    assert committed == rule_json_schema()
