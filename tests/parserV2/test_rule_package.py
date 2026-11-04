"""Schema 0.8 packaged-document, composition, and validation contracts."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from apb2 import api
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import (
    RuleNotApplicable,
    SearchParameterEvidence,
    make_rule_document,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.axis import (
    FIXED_COLUMNS,
    Coalesce,
    ContaminantFlag,
    DecoyFlag,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    RankRows,
    SourcedColumn,
    SourcedVarColumn,
    StrippedSequence,
    computed_columns,
)
from apb2.parserV2.vendor_parse_rules.schema.base import (
    SCHEMA_VERSION,
    QuantificationLevel,
)
from apb2.parserV2.vendor_parse_rules.schema.base_modifications import (
    EmbeddedSiteListSyntax,
    SiteListSyntax,
    TokenRegexSyntax,
)
from apb2.parserV2.vendor_parse_rules.schema.fragments import (
    ColumnLabeledFragments,
    PositionalFragments,
)
from apb2.parserV2.vendor_parse_rules.schema.hierarchy import HIERARCHIES
from apb2.parserV2.vendor_parse_rules.schema.input import (
    AcquisitionFromColumns,
    Input,
    WithoutParameterFile,
)
from apb2.parserV2.vendor_parse_rules.schema.measurements import (
    FactorLayer,
    Measurements,
    NumericLayer,
    RegexValuePattern,
    RepeatedCells,
    layer_required,
)
from apb2.parserV2.vendor_parse_rules.schema.roles import ROLE_CONFIG, VAR_ROLES
from apb2.parserV2.vendor_parse_rules.schema.rule import (
    LongRule,
    WideRule,
    rule_json_schema,
    validate_rule,
)
from apb2.parserV2.vendor_parse_rules.schema_artifact import artifact_path
from parserV2 import synthetic
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
    ("alphapept", "peptidoform_ion"): (
        "ms1_int_sum_apex",
        "ms1_int_sum_area",
        "ms1_int_max_apex",
        "ms1_int_max_area",
    ),
    ("diann/v1_8", "peptidoform_ion"): ("Ms1_Normalised", "Precursor_Quantity", "Ms1_Area"),
    ("diann/v1_8", "protein"): ("PG_Normalised", "PG_Quantity", "Genes_MaxLFQ"),
    ("diann/v1_7", "peptidoform_ion"): ("Precursor_Quantity", "Ms1_Area"),
    ("diann/v1_7", "protein"): ("PG_Quantity", "PG_MaxLFQ", "Genes_MaxLFQ"),
    ("diann/v2", "peptidoform_ion"): ("Ms1_Normalised", "Precursor_Quantity", "Ms1_Area"),
    ("diann/v2", "protein"): ("Genes_MaxLFQ",),
    ("maxquant_wide", "peptide"): ("LFQ_intensity",),
    ("maxquant_wide", "protein"): ("LFQ_intensity", "iBAQ"),
    ("msangel", "peptidoform_ion"): ("raw_abundance",),
    ("msangel", "protein"): ("raw_abundance",),
    ("spectronaut", "peptidoform_ion"): (
        "EG_ReferenceQuantity_Settings",
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
    ),
    ("spectronaut/v21", "peptidoform_ion"): (
        "EG_ReferenceQuantity_Settings",
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
    ),
    ("spectronaut/v15", "peptidoform_ion"): (
        "EG_TargetQuantity_Settings",
        "EG_TotalQuantity_Settings",
        "FG_MS1RawQuantity",
        "FG_MS2RawQuantity",
    ),
    ("spectronaut/v15", "peptidoform_fragment"): ("F_PeakHeight",),
}


type MutatePayload = Callable[[dict[str, Any]], object]
type V2Rule = LongRule | WideRule


def _without_primary_layer(rule: V2Rule) -> dict[str, Any]:
    """The declaration without its primary layer and the ``required`` that follows it."""
    payload = rule.model_dump(mode="json")
    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    measurements.pop("primary_layer")
    for layer in measurements["layers"]:
        layer.pop("required")
    return payload


# --------------------------------------------------------------------------- migration parity


def test_only_a_protein_list_column_declares_a_separator() -> None:
    with pytest.raises(ValidationError, match="only a protein-list column declares a separator"):
        SourcedColumn(name="Gene", source="Gene", type="string", required=True, separator=":")


def test_entry_shaped_columns_project_roles_and_runtime_selections(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["layers"][0]["roles"] = ["abundance"]
    payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"] = [
        {
            "name": "feature",
            "source": "Feature",
            "type": "string",
            "roles": ["protein_assignment", "fasta_accessions"],
        }
    ]

    document = make_rule_document(tmp_path / "rules.json", payload)
    working = ParseRuleFacade(document, "peptidoform_ion", NO_EVIDENCE).working_parameters

    assert working.var.required_selections[0].logical_type == "protein_list"
    assert working.column_roles == {
        "protein_assignment": "feature",
        "fasta_accessions": "feature",
    }
    assert working.measurements.layers[0].roles == ("abundance",)


def test_role_configuration_owns_the_role_vocabulary() -> None:
    assert {
        "obs": frozenset(),
        "var": frozenset({"fasta_accessions", "protein_assignment"}),
        "layer": frozenset({"abundance"}),
    } == ROLE_CONFIG


def test_var_roles_are_the_var_owners_roles() -> None:
    assert ROLE_CONFIG["var"] == VAR_ROLES
    assert api.VAR_ROLES is VAR_ROLES


def test_fixed_columns_are_the_names_the_schema_fixes() -> None:
    fixed = (
        StrippedSequence,
        ProformaSequence,
        ProformaIon,
        ProformaFragment,
        DecoyFlag,
        ContaminantFlag,
    )
    names = [name for model in fixed for name in get_args(model.model_fields["name"].annotation)]
    assert tuple(names) == FIXED_COLUMNS
    assert api.FIXED_COLUMNS is FIXED_COLUMNS


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


@pytest.mark.parametrize(
    ("count", "message"),
    [(0, "declares 'apb_Decoy' 0 times"), (2, "column entry names must be unique")],
)
def test_every_rule_declares_each_vendor_marking_once(
    count: int, message: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["columns"]["var"] = [
        flag for flag in synthetic.FLAGS if flag["name"] != "apb_Decoy"
    ] + [{"name": "apb_Decoy", "how": "unmarked"}] * count

    with pytest.raises(ValidationError, match=message):
        _declared(payload, tmp_path)


@pytest.mark.parametrize(
    ("marking", "message"),
    [
        ({"inputs": ["feature"]}, "exactly one of equals or prefix"),
        ({"inputs": ["feature"], "equals": "+", "prefix": "REV__"}, "exactly one of"),
        ({"equals": "+"}, "inputs"),
        ({"inputs": ["feature"], "equals": "+", "separator": ";"}, "separator only with"),
        ({"inputs": ["feature"], "equals": "+", "roles": ["protein_assignment"]}, "at most 0"),
        ({"how": "unmarked", "inputs": ["feature"]}, "at most 0"),
        ({"how": "unmarked", "equals": "+"}, "equals"),
    ],
)
def test_a_vendor_marking_names_its_inputs_and_one_way_to_read_them(
    marking: dict[str, object], message: str, tmp_path: Path
) -> None:
    """A marking reads at least one input one way; an unmarked flag reads none."""
    payload = _document_payload()
    payload["tables"][0]["base"]["columns"]["var"] = [
        {"name": "apb_Decoy", "how": "decoy", **marking},
        {"name": "apb_Contaminant", "how": "unmarked"},
    ]

    with pytest.raises(ValidationError, match=message):
        _declared(payload, tmp_path)


@pytest.mark.parametrize(
    ("repeated_cells", "message"),
    [
        ({"mode": "sum", "sum": ["quantity"]}, "'sum' applies only to keep_best"),
        (
            {"mode": "keep_best", "by": "quantity", "best": "highest", "sum": ["quantity"]},
            "cannot also sum it",
        ),
        (
            {"mode": "keep_best", "by": "quantity", "best": "highest", "sum": ["absent"]},
            "sums undeclared layers",
        ),
    ],
)
def test_keep_best_sums_only_declared_layers_other_than_its_ranking(
    repeated_cells: dict[str, object], message: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["repeated_cells"] = repeated_cells
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises((ValidationError, ValueError), match=message):
        ParseRuleFacade(document, "peptidoform_ion", NO_EVIDENCE)


def test_one_role_cannot_name_two_columns(tmp_path: Path) -> None:
    payload = _document_payload()
    columns = payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"]
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
    assert rule.measurements.repeated_cells.mode in {"error", "same", "sum", "keep_best"}
    assert not hasattr(rule.axis, "x_layer")
    assert not hasattr(rule.axis, "repeated_cells")


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
    assert layer_required(rule.measurements, primary[0])
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


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_every_abundance_layer_treats_zero_and_negative_values_as_missing(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    unmasked = [
        layer.name
        for layer in rule.measurements.layers
        if "abundance" in layer.roles
        and not (isinstance(layer, NumericLayer) and layer.missing_at_or_below == 0.0)
    ]

    assert unmasked == []


def _numeric_layer(**fields: object) -> NumericLayer:
    """A numeric layer stating the values a table's ``defaults`` would give it."""
    return NumericLayer.model_validate(
        {"encoding_mode": "numeric", "type": "number", "required": False, **fields}
    )


def test_missing_values_combine_exact_sentinels_with_one_at_or_below_bound() -> None:
    layer = _numeric_layer(name="I", source="I", missing_values=[-1, "<= 0", 7])

    assert layer.missing_sentinels == (-1.0, 7.0)
    assert layer.missing_at_or_below == 0.0


def test_missing_values_reject_a_second_threshold() -> None:
    with pytest.raises(ValidationError, match="more than one threshold"):
        _numeric_layer(name="I", source="I", missing_values=["<=0", "<=-2.5"])


def test_missing_tokens_reject_blank_text() -> None:
    with pytest.raises(ValidationError, match="blank text"):
        _numeric_layer(name="I", source="I", missing_tokens=["-", " "])


@pytest.mark.parametrize("token", [">=0", "<0", "<=zero"])
def test_a_missing_value_text_must_be_an_at_or_below_bound(token: str) -> None:
    with pytest.raises(ValidationError):
        _numeric_layer(name="I", source="I", missing_values=[token])


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
        ("fragpipe", "peptidoform_ion", "Spectral_Count"),
        ("maxquant_wide", "peptidoform_ion", "MS_MS_count"),
        ("msangel", "peptidoform_ion", "psm_count"),
        ("msangel", "protein", "psm_count"),
        ("msangel", "protein", "peptides_count"),
        ("prolinestudio", "peptidoform_ion", "psm_count"),
        ("spectronaut", "protein", "PG_RunEvidenceCount"),
        ("spectronaut/v21", "protein", "PG_RunEvidenceCount"),
        ("wombat", "peptidoform", "number_of_psms"),
    }


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_recognition_rejects_a_header_missing_one_required_source(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    working = ParseRuleFacade.from_declared_rule(
        load_rule_document(pair.parser_v2_path), level
    ).working_parameters
    header = pair.header()
    if not header or not working.accepts_header(header):
        pytest.skip(f"cached export for {pair.key} does not satisfy level {level!r}")
    required = {
        selection.source
        for axis in (working.obs, working.var)
        for selection in axis.required_selections
    }
    assert required
    for source in required:
        assert not working.accepts_header(tuple(name for name in header if name != source))


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


def test_maxquant_prepares_evidence_apart_from_the_wide_higher_level_join() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "maxquant_wide")
    wide = load_rule_document(pair.parser_v2_path)

    ion = wide.declared("peptidoform_ion")
    assert ion.input.extensions == [".txt"]
    roles = {column.name: column.roles for column in ion.declaration.columns.var}
    assert wide.preparation("peptidoform_ion") == "maxquant_evidence"
    assert roles["Protein_IDs"] == ["protein_assignment"]
    assert roles["Proteins"] == ["fasta_accessions"]
    assert wide.table_levels == (("peptidoform_ion",), ("peptidoform", "peptide", "protein"))
    for level in ("peptidoform", "peptide", "protein"):
        assert wide.preparation(level) == "maxquant_wide"
        assert wide.declared(level).declaration.shape == "wide"
        assert wide.declared(level).declaration.axis.obs_keys == ["Experiment"]


def test_peaks_declares_the_persisted_sample_annotation_matching_policy() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "peaks")
    document = load_rule_document(pair.parser_v2_path)
    effective = document.declared("peptidoform_ion").declaration
    provenance = ParseRuleFacade.from_declared_rule(
        document,
        "peptidoform_ion",
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


@pytest.mark.parametrize("level", ["peptidoform_ion", "protein"])
def test_msangel_measurements_canonicalize_the_dda_run_prefix(level: QuantificationLevel) -> None:
    """Sources carry the preparation's sheet prefix, ``ion.`` or ``protein.``."""
    pair = next(candidate for candidate in document_pairs() if candidate.key == "msangel")
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    assert isinstance(rule, WideRule)

    for layer in rule.measurements.layers:
        column = layer.source.replace("^", "").replace("$", "").replace("\\.", ".")
        prefix = column.split("(?P<sample>", maxsplit=1)[0]
        name = prefix.replace("(?:DDA_)?", "DDA_") + "Condition_A_Sample_Alpha_01"
        match = re.fullmatch(layer.source, name)
        assert match is not None
        assert name.startswith(f"{level}.")
        assert match.group("sample") == "Condition_A_Sample_Alpha_01"


@pytest.mark.parametrize(
    ("document_key", "level", "protein_assignment", "fasta_accessions"),
    [
        pytest.param(
            "diann/v1_8", "peptidoform_ion", "Protein_Group", "Protein_Ids", id="diann-v1_8-ion"
        ),
        pytest.param(
            "diann/v1_8", "protein", "Protein_Group", "Protein_Group", id="diann-v1_8-protein"
        ),
        pytest.param(
            "diann/v2", "peptidoform_ion", "Protein_Group", "Protein_Ids", id="diann-v2-ion"
        ),
        pytest.param(
            "diann/v2", "protein", "Protein_Group", "Protein_Group", id="diann-v2-protein"
        ),
        pytest.param(
            "spectronaut",
            "peptidoform_ion",
            "PG_ProteinGroups",
            "PG_ProteinAccessions",
            id="spectronaut-ion",
        ),
        pytest.param(
            "spectronaut",
            "peptidoform_fragment",
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
        pytest.param(
            "alphadia/v2", "peptidoform_ion", "pg_name", "pg_proteins", id="alphadia-v2-ion"
        ),
        pytest.param("alphadia/v2", "protein", "pg_name", "pg_name", id="alphadia-v2-protein"),
        pytest.param(
            "maxquant_wide", "peptidoform_ion", "Protein_IDs", "Proteins", id="maxquant-wide-ion"
        ),
        pytest.param(
            "maxquant_wide", "protein", "Protein_IDs", "Protein_IDs", id="maxquant-wide-protein"
        ),
        pytest.param(
            "msangel", "peptidoform_ion", "samesets_accessions", "Proteins", id="msangel-ion"
        ),
        pytest.param("msangel", "protein", "samesets_accessions", "Proteins", id="msangel-protein"),
        pytest.param(
            "prolinestudio",
            "peptidoform_ion",
            "samesets_accessions",
            "Proteins",
            id="prolinestudio-ion",
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
    roles = ParseRuleFacade(document, level, NO_EVIDENCE).working_parameters.column_roles

    assert roles == {
        "protein_assignment": protein_assignment,
        "fasta_accessions": fasta_accessions,
    }


def test_alphadia_v1_10_genes_are_both_assignment_and_fasta_identifiers() -> None:
    # AlphaDIA 1.10 exports no accession column; ``genes`` holds UniProt entry names.
    pair = next(candidate for candidate in document_pairs() if candidate.key == "alphadia/v1_10")
    document = load_rule_document(pair.parser_v2_path)
    roles = ParseRuleFacade(
        document, "peptidoform_ion", NO_EVIDENCE
    ).working_parameters.column_roles

    assert roles == {"protein_assignment": "genes", "fasta_accessions": "genes"}


def test_spectronaut_fragment_exposes_its_parent_ion_identity() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    fragment = load_rule_document(pair.parser_v2_path).declared("peptidoform_fragment").declaration

    computed = {column.name: column for column in computed_columns(fragment.columns.var)}

    assert computed["ProForma_ion"].inputs == ["ProForma_peptidoform", "FG_Charge"]


# ------------------------------------------------------------------- gates and overrides


def test_sage_gates_its_two_levels_on_combine_charge_states() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    document = load_rule_document(pair.parser_v2_path)
    separate = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False)
    combined = SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=True)

    assert document.rule("peptidoform_ion", separate).declaration.axis.var_keys == ["ProForma_ion"]
    assert document.rule("peptidoform", combined).declaration.axis.var_keys == [
        "ProForma_peptidoform"
    ]
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("peptidoform_ion", combined)
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("peptidoform", separate)
    with pytest.raises(RuleNotApplicable, match="combine_charge_states"):
        document.rule("peptidoform_ion", NO_EVIDENCE)


def test_diann_v2_swaps_only_the_primary_layer_for_dda_evidence() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v2")
    document = load_rule_document(pair.parser_v2_path)

    dda = document.rule("peptidoform_ion", DDA).declaration
    dia = document.rule("peptidoform_ion", DIA).declaration

    assert dda.measurements.primary_layer == "Ms1_Normalised"
    assert dia.measurements.primary_layer == "Precursor_Normalised"
    required = {
        evidence: {layer.name for layer in rule.measurements.layers if layer.required}
        for evidence, rule in (("DDA", dda), ("DIA", dia))
    }
    assert "Ms1_Normalised" in required["DDA"] - required["DIA"]
    assert "Precursor_Normalised" in required["DIA"] - required["DDA"]
    assert document.declared("peptidoform_ion").declaration.measurements.primary_layer == (
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


# ------------------------------------------------------------------ what schema 0.11 refuses


def _document_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "file_version": "1",
        "software_name": "Test",
        "software_version_pattern": "^1$",
        "hierarchy": "lfq",
        "parameter_file": "required",
        "tables": [
            {
                "input": {
                    "shape": "long",
                    "extensions": [".tsv"],
                },
                "base": {
                    "defaults": copy.deepcopy(synthetic.LONG_DEFAULTS),
                    "axis": {"obs_keys": ["sample"], "var_keys": ["feature"]},
                    "columns": {
                        "obs": [{"name": "sample", "source": "Sample"}],
                        "var": [dict(flag) for flag in synthetic.FLAGS],
                    },
                    "measurements": {
                        "primary_layer": "quantity",
                        "repeated_cells": {"mode": "error"},
                        "layers": [{"name": "quantity", "source": "Quantity"}],
                    },
                },
                "levels": {
                    "peptidoform_ion": {
                        "columns": {"var": [{"name": "feature", "source": "Feature"}]}
                    }
                },
            }
        ],
    }


def _declared(payload: dict[str, Any], tmp_path: Path) -> None:
    document = make_rule_document(tmp_path / "rules.json", payload)
    document.declared("peptidoform_ion")


def _with_ranking(rank_rows: dict[str, Any], columns: list[dict[str, Any]]) -> MutatePayload:
    """Declare the level's ``rank_rows`` and append ``columns`` to its var columns."""

    def mutate(payload: dict[str, Any]) -> None:
        level = payload["tables"][0]["levels"]["peptidoform_ion"]
        level["rank_rows"] = rank_rows
        level["columns"]["var"].extend(columns)

    return mutate


def _ranked_by(by: str, ranking: list[dict[str, Any]]) -> MutatePayload:
    """Declare ``ranking`` and a var column taking its value from the row ``by`` ranks best."""
    return _with_ranking(
        {"by": by, "best": "highest"},
        [*ranking, {"name": "set", "source": "Set", "per_feature": "best_row"}],
    )


def test_the_reference_payload_is_valid_so_every_rejection_below_is_the_change(
    tmp_path: Path,
) -> None:
    _declared(_document_payload(), tmp_path)


def test_source_prefix_names_the_joined_table_of_each_source(tmp_path: Path) -> None:
    """The level's prefix reaches base and level sources; an entry's own prefix wins."""
    payload = _document_payload()
    table = payload["tables"][0]
    table["base"]["source_prefix"] = "base."
    table["levels"]["peptidoform_ion"]["source_prefix"] = "evidence."
    table["levels"]["peptidoform_ion"]["columns"]["var"].append(
        {"name": "group", "source": "Protein IDs", "source_prefix": "protein."}
    )
    document = make_rule_document(tmp_path / "rules.json", payload)

    rule = document.declared("peptidoform_ion").declaration

    assert isinstance(rule, LongRule)
    columns = [*rule.columns.obs, *rule.columns.var]
    assert {column.name: column.source for column in columns if column.source is not None} == {
        "sample": "evidence.Sample",
        "feature": "evidence.Feature",
        "group": "protein.Protein IDs",
    }
    assert rule.measurements.layers[0].source == "evidence.Quantity"
    assert "source_prefix" not in json.dumps(rule.model_dump(mode="json"))
    assert document.declared("peptidoform_ion").declaration == rule
    authored = json.loads(json.dumps(document.tables[0].levels["peptidoform_ion"]))
    assert authored["columns"]["var"][1]["source"] == "Protein IDs"


def _wide_payload(layer_source: str) -> dict[str, Any]:
    payload = _document_payload()
    table = payload["tables"][0]
    table["input"]["shape"] = "wide"
    del table["base"]["columns"]["obs"]
    del table["base"]["defaults"]["obs"]
    table["base"]["measurements"]["layers"] = [{"name": "quantity", "source": layer_source}]
    table["levels"]["peptidoform_ion"]["source_prefix"] = "peptide."
    return payload


def test_a_wide_source_takes_its_escaped_prefix_after_the_anchor(tmp_path: Path) -> None:
    payload = _wide_payload("^Intensity (?P<sample>.+)$")

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    assert rule.declaration.measurements.layers[0].source == r"^peptide\.Intensity (?P<sample>.+)$"
    feature = next(column for column in rule.declaration.columns.var if column.name == "feature")
    assert feature.source == "peptide.Feature"
    with pytest.raises(ValueError, match="starts with"):
        _declared(_wide_payload("Intensity (?P<sample>.+)$"), tmp_path)


def test_a_wide_placeholder_source_reads_as_text_around_the_sample_pattern(
    tmp_path: Path,
) -> None:
    """``{sample}`` stands for the table's ``sample_pattern``; the text matches literally."""
    payload = _wide_payload("Intensity {sample}")
    payload["tables"][0]["base"]["sample_pattern"] = "(?P<sample>.+)"
    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")
    assert rule.declaration.measurements.layers[0].source == r"^peptide\.Intensity (?P<sample>.+)$"

    payload = _wide_payload("abundance_{sample} m/z")
    payload["tables"][0]["base"]["sample_pattern"] = "(?:DDA_)?(?P<sample>.+?)"
    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    assert rule.declaration.measurements.layers[0].source == (
        r"^peptide\.abundance_(?:DDA_)?(?P<sample>.+?) m/z$"
    )
    assert "sample_pattern" not in json.dumps(rule.declaration.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("source", "pattern", "message"),
    [
        ("Intensity {sample}", ".+", r"must contain '\(\?P<sample>"),
        ("{sample} and {sample}", "(?P<sample>.+)", "names {sample} once"),
        ("Intensity {sample}", None, "declares no sample_pattern"),
        ("^Intensity (?P<sample>.+)$", "(?P<sample>.+)", "no layer writes {sample}"),
    ],
)
def test_a_wide_placeholder_needs_one_sample_group(
    source: str, pattern: str | None, message: str, tmp_path: Path
) -> None:
    payload = _wide_payload(source)
    if pattern is not None:
        payload["tables"][0]["levels"]["peptidoform_ion"]["sample_pattern"] = pattern

    with pytest.raises(ValueError, match=message):
        _declared(payload, tmp_path)


def test_a_long_table_declares_no_sample_pattern(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["sample_pattern"] = "(?P<sample>.+)"

    with pytest.raises(ValueError, match="applies to a wide table"):
        _declared(payload, tmp_path)


# ------------------------------------------------------------------------ table defaults


def test_entries_take_the_tables_defaults_and_their_own_values_win(tmp_path: Path) -> None:
    """An entry states only what differs from its table; the stored rule states everything."""
    payload = _document_payload()
    table = payload["tables"][0]
    table["base"]["defaults"]["var"]["required"] = False
    var = table["levels"]["peptidoform_ion"]["columns"]["var"]
    var[0]["required"] = True
    var.append({"name": "gene", "source": "Gene"})
    table["base"]["measurements"]["layers"].extend(
        [
            {"name": "count", "source": "Count", "type": "integer", "required": True},
            {"name": "area", "source": "Area"},
        ]
    )

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    declaration = rule.declaration
    assert isinstance(declaration, LongRule)
    columns = {column.name: column for column in declaration.columns.var}
    gene = columns["gene"]
    assert isinstance(gene, SourcedVarColumn)
    assert (gene.type, gene.required, gene.per_feature) == ("string", False, "first")
    assert columns["feature"].required
    sample = declaration.columns.obs[0]
    assert (sample.type, sample.required) == ("string", True)
    layers = declaration.measurements.layers
    assert all(isinstance(layer, NumericLayer) for layer in layers)
    assert [
        (layer.type, layer.required) for layer in layers if isinstance(layer, NumericLayer)
    ] == [("number", True), ("integer", True), ("number", False)]
    assert "defaults" not in json.dumps(declaration.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda payload: payload["tables"][0]["base"].pop("defaults"),
            "base declares no defaults",
            id="missing",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"].update(
                {"defaults": payload["tables"][0]["base"]["defaults"]}
            ),
            "defaults belong to the table's base",
            id="in-a-level",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["defaults"].pop("obs"),
            "names obs exactly for a long table",
            id="long-table-without-obs",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["defaults"]["var"].pop("per_feature"),
            r"base\.defaults\.var\.per_feature: Field required",
            id="a-field-missing",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["defaults"]["layers"].update(
                {"roles": ["abundance"]}
            ),
            r"base\.defaults\.layers\.roles: Extra inputs are not permitted",
            id="a-field-no-choice-needs",
        ),
    ],
)
def test_a_tables_defaults_are_complete_and_stated_once_in_its_base(
    mutate: MutatePayload, message: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        make_rule_document(tmp_path / "rules.json", payload)


def test_a_wide_tables_defaults_name_no_obs(tmp_path: Path) -> None:
    payload = _wide_payload("^Intensity (?P<sample>.+)$")
    payload["tables"][0]["base"]["defaults"]["obs"] = copy.deepcopy(synthetic.LONG_DEFAULTS["obs"])

    with pytest.raises(ValidationError, match="names obs exactly for a long table"):
        make_rule_document(tmp_path / "rules.json", payload)


def test_the_primary_and_ranking_layers_are_required_and_state_no_required(
    tmp_path: Path,
) -> None:
    """apb2 converts no table without them, so composition writes the only possible value."""
    payload = _document_payload()
    measurements = payload["tables"][0]["base"]["measurements"]
    measurements["repeated_cells"] = {"mode": "keep_best", "by": "score", "best": "highest"}
    measurements["layers"].extend(
        [{"name": "score", "source": "Score"}, {"name": "area", "source": "Area"}]
    )

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    layers = rule.declaration.measurements.layers
    assert [(layer.name, layer.required) for layer in layers] == [
        ("quantity", True),
        ("score", True),
        ("area", False),
    ]

    measurements["layers"][0]["required"] = True
    with pytest.raises(ValueError, match="'quantity' is the level's primary, sample or ranking"):
        _declared(payload, tmp_path)


def test_a_var_column_declares_missing_text_but_never_blank_text(tmp_path: Path) -> None:
    payload = _document_payload()
    var = payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"]
    var.append({"name": "score", "source": "Score", "type": "number", "missing_tokens": ["NaN"]})

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    score = {column.name: column for column in rule.declaration.columns.var}["score"]
    assert isinstance(score, SourcedVarColumn)
    assert score.missing_tokens == ["NaN"]
    var[-1]["missing_tokens"] = [" "]
    with pytest.raises(ValidationError, match="blank text"):
        _declared(payload, tmp_path)


def test_a_factor_layer_takes_no_type_from_the_defaults(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["layers"].append(
        {
            "name": "status",
            "source": "Status",
            "encoding_mode": "factor",
            "categories": {"identified": 1},
        }
    )

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    status = rule.declaration.measurements.layers[-1]
    assert isinstance(status, FactorLayer)
    assert status.required is False


def test_a_computed_entry_states_no_type_or_required(tmp_path: Path) -> None:
    """A computed value is text, and an own source must exist; neither is a choice."""
    payload = _document_payload()
    var = payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"]
    var.extend(
        [
            {"name": "alias", "source": "Alias"},
            {"name": "either", "how": "coalesce", "inputs": ["feature", "alias"]},
        ]
    )

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    either = rule.declaration.columns.var[-1]
    assert isinstance(either, Coalesce)
    assert (either.required, either.type) == (True, "string")
    assert {"type", "required"}.isdisjoint(either.model_dump(mode="json"))
    for key, value in (("required", True), ("type", "string")):
        var[-1] = {"name": "either", "how": "coalesce", "inputs": ["feature", "alias"], key: value}
        with pytest.raises(ValidationError, match=rf"coalesce\.{key}\n\s+Extra inputs"):
            _declared(payload, tmp_path)


def _forbidding(forbidden: list[str], *entries: dict[str, Any]) -> MutatePayload:
    """Declare the level's ``forbidden_columns`` and append ``entries`` to its var columns."""

    def mutate(payload: dict[str, Any]) -> None:
        level = payload["tables"][0]["levels"]["peptidoform_ion"]
        level["forbidden_columns"] = forbidden
        level["columns"]["var"].extend(entries)

    return mutate


def test_a_level_names_the_columns_that_rule_its_table_out(tmp_path: Path) -> None:
    payload = _document_payload()
    _forbidding(["Charge"])(payload)

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    assert rule.declaration.forbidden_columns == ["Charge"]
    assert rule.declaration.model_dump(mode="json")["forbidden_columns"] == ["Charge"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda payload: payload["tables"][0]["base"].update({"forbidden_columns": ["Charge"]}),
            "forbidden_columns belong to a level",
            id="in-the-base",
        ),
        pytest.param(
            _forbidding(["Feature"]),
            r"forbidden_columns \['Feature'\] are sources the level reads",
            id="a-var-source",
        ),
        pytest.param(
            _forbidding(["Sample"]),
            r"forbidden_columns \['Sample'\] are sources the level reads",
            id="an-obs-source",
        ),
        pytest.param(
            _forbidding(["Quantity"]),
            r"forbidden_columns \['Quantity'\] are sources the level reads",
            id="a-layer-source",
        ),
        pytest.param(
            _forbidding(["Alias"], {"name": "alias", "source": "Alias", "required": False}),
            r"forbidden_columns \['Alias'\] are sources the level reads",
            id="an-optional-source",
        ),
    ],
)
def test_forbidden_columns_belong_to_a_level_that_names_no_such_source(
    mutate: MutatePayload, message: str, tmp_path: Path
) -> None:
    payload = _document_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        _declared(payload, tmp_path)


def test_a_wide_level_may_forbid_a_column_its_layer_pattern_claims(tmp_path: Path) -> None:
    """A layer pattern claims a family of columns, so a level may forbid one; not a var source."""
    payload = _wide_payload("^Intensity (?P<sample>.+)$")
    level = payload["tables"][0]["levels"]["peptidoform_ion"]
    level["forbidden_columns"] = ["peptide.Intensity A"]

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    assert rule.declaration.forbidden_columns == ["peptide.Intensity A"]
    level["forbidden_columns"] = ["peptide.Feature"]
    with pytest.raises(ValidationError, match=r"forbidden_columns \['peptide\.Feature'\] are"):
        _declared(payload, tmp_path)


def test_a_level_cannot_forbid_its_fragments_label_column() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    rule = load_rule_document(pair.parser_v2_path).declared("peptidoform_fragment").declaration
    assert isinstance(rule, LongRule)
    assert isinstance(rule.fragments, ColumnLabeledFragments)
    label = rule.fragments.label_column
    entries = (*rule.columns.obs, *rule.columns.var, *rule.measurements.layers)
    assert label not in {entry.source for entry in entries}, "only the fragments name it"
    payload = rule.model_dump(mode="python")
    payload["forbidden_columns"] = [label]

    with pytest.raises(ValidationError, match=rf"forbidden_columns \['{re.escape(label)}'\]"):
        validate_rule(payload)


_COMPLETE_DECLARATIONS: tuple[tuple[type[BaseModel], dict[str, Any]], ...] = (
    (SourcedColumn, {"name": "gene", "source": "Gene", "type": "string", "required": True}),
    (
        SourcedVarColumn,
        {
            "name": "gene",
            "source": "Gene",
            "type": "string",
            "required": True,
            "per_feature": "first",
        },
    ),
    (
        NumericLayer,
        {
            "encoding_mode": "numeric",
            "type": "number",
            "name": "q",
            "source": "Q",
            "required": False,
        },
    ),
    (
        FactorLayer,
        {
            "encoding_mode": "factor",
            "name": "status",
            "source": "Status",
            "categories": {"identified": 1},
            "required": False,
        },
    ),
    (
        Measurements,
        {
            "primary_layer": "q",
            "repeated_cells": {"mode": "error"},
            "layers": [
                {
                    "encoding_mode": "numeric",
                    "type": "number",
                    "name": "q",
                    "source": "Q",
                    "required": False,
                }
            ],
        },
    ),
    (RepeatedCells, {"mode": "error"}),
    (RankRows, {"by": "score", "best": "highest"}),
    (
        ProformaSequence,
        {
            "how": "proforma_sequence",
            "name": "ProForma_peptidoform",
            "inputs": ["Sequence"],
            "syntax": "vendor",
            "case_sensitive": False,
            "unknown_policy": "preserve",
        },
    ),
    (
        TokenRegexSyntax,
        {
            "parser": "token_regex",
            "token_pattern": r"\((?P<residue>[^)]+)\)",
            "token_position": "after_residue",
        },
    ),
    (SiteListSyntax, {"parser": "site_list", "delimiter": ";", "site_base": 1}),
    (
        EmbeddedSiteListSyntax,
        {
            "parser": "embedded_site_list",
            "delimiter": ";",
            "entry_pattern": r"(?P<token>[^@]+)@(?P<site>\d+)",
            "site_base": 1,
        },
    ),
    (
        PositionalFragments,
        {
            "label_strategy": "positional",
            "value_columns": ["Quantity"],
            "delimiter": ";",
            "label_output": "fragment_label",
        },
    ),
    (
        ColumnLabeledFragments,
        {
            "label_strategy": "column",
            "value_columns": ["Quantity"],
            "label_column": "Info",
            "label_required": True,
            "delimiter": ";",
            "label_output": "fragment_label",
        },
    ),
    (WithoutParameterFile, {"acquisition_method": {"otherwise": "unknown"}}),
    (AcquisitionFromColumns, {"otherwise": "unknown"}),
)
"""A complete declaration of each model that holds a real choice."""

_REAL_CHOICES: dict[type[BaseModel], tuple[str, ...]] = {
    SourcedColumn: ("type", "required"),
    SourcedVarColumn: ("per_feature",),
    NumericLayer: ("encoding_mode", "type", "required"),
    FactorLayer: ("required",),
    Measurements: ("repeated_cells",),
    RepeatedCells: ("mode",),
    RankRows: ("best",),
    ProformaSequence: ("name", "case_sensitive", "unknown_policy"),
    TokenRegexSyntax: ("token_position",),
    SiteListSyntax: ("delimiter", "site_base"),
    EmbeddedSiteListSyntax: ("delimiter", "site_base"),
    PositionalFragments: ("delimiter", "label_output"),
    ColumnLabeledFragments: ("label_required", "delimiter", "label_output"),
    WithoutParameterFile: ("acquisition_method",),
    AcquisitionFromColumns: ("otherwise",),
}
"""Each model's fields with more than one valid value: a rule states them, no model default."""


@pytest.mark.parametrize(
    ("model", "complete", "choice"),
    [
        pytest.param(model, complete, choice, id=f"{model.__name__}.{choice}")
        for model, complete in _COMPLETE_DECLARATIONS
        for choice in _REAL_CHOICES[model]
    ],
)
def test_a_real_choice_has_no_model_default(
    model: type[BaseModel], complete: dict[str, Any], choice: str
) -> None:
    model.model_validate(complete)
    with pytest.raises(ValidationError) as missing:
        model.model_validate({key: value for key, value in complete.items() if key != choice})
    assert _problems(missing.value) == [("missing", (choice,))]


@pytest.mark.parametrize("choice", ["hierarchy", "parameter_file"])
def test_a_document_states_its_hierarchy_and_parameter_file(choice: str, tmp_path: Path) -> None:
    payload = _document_payload()
    del payload[choice]

    with pytest.raises(ValidationError) as missing:
        make_rule_document(tmp_path / "rules.json", payload)
    assert _problems(missing.value) == [("missing", (choice,))]


def test_no_packaged_note_ends_with_a_period() -> None:
    """A note is a fragment, as the rules write them; a trailing period marks one that is not."""
    ended: list[str] = []
    for path in PACKAGED:
        text = path.read_text(encoding="utf-8")
        rule = path.as_posix().split("/documents/")[1]
        ended += [f"{rule}: {note}" for note in re.findall(r'"note": "((?:[^"\\]|\\.)*)"', text)]
    assert ended
    assert [note for note in ended if note.endswith(".")] == []


def test_no_packaged_entry_restates_its_tables_defaults() -> None:
    """The files say each shared value once, in their table's ``defaults``."""
    restated: list[tuple[str, str, str]] = []
    for path in PACKAGED:
        document = json.loads(path.read_text(encoding="utf-8"))
        rule = path.as_posix().split("/documents/")[1]
        for table in document["tables"]:
            defaults = table["base"]["defaults"]
            for fragment in [table["base"], *table["levels"].values()]:
                entries = [
                    (axis, entry)
                    for axis in ("obs", "var")
                    for entry in fragment.get("columns", {}).get(axis, [])
                    if "how" not in entry
                ] + [
                    ("layers", layer)
                    for layer in fragment.get("measurements", {}).get("layers", [])
                ]
                restated += [
                    (rule, entry["name"], key)
                    for axis, entry in entries
                    for key, value in defaults[axis].items()
                    if entry.get(key) == value
                ]
    assert restated == []


def test_a_sequence_column_takes_the_levels_only_syntax_and_map(tmp_path: Path) -> None:
    payload = _document_payload()
    base = payload["tables"][0]["base"]
    base["sequence_syntax"] = {
        "vendor": {
            "parser": "token_regex",
            "token_pattern": r"\((?P<residue>[^)]+)\)",
            "token_position": "after_residue",
        }
    }
    base["modification_map"] = {"ox": "UNIMOD:35"}
    payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"].extend(
        [
            {"name": "Sequence", "source": "Sequence"},
            {
                "name": "ProForma_peptidoform",
                "inputs": ["Sequence"],
                "how": "proforma_sequence",
                "case_sensitive": False,
                "unknown_policy": "preserve",
            },
            {"name": "ProForma_peptide", "inputs": ["Sequence"], "how": "stripped_sequence"},
        ]
    )

    rule = make_rule_document(tmp_path / "rules.json", payload).declared("peptidoform_ion")

    columns = {column.name: column for column in rule.declaration.columns.var}
    peptidoform = columns["ProForma_peptidoform"]
    peptide = columns["ProForma_peptide"]
    assert isinstance(peptidoform, ProformaSequence)
    assert isinstance(peptide, StrippedSequence)
    assert (peptidoform.syntax, peptide.syntax) == ("vendor", "vendor")
    assert rule.declaration.modification_map == {"ox": "UNIMOD:35"}
    base["sequence_syntax"]["plain"] = {"parser": "plain_sequence"}
    with pytest.raises(ValueError, match="the level declares 2"):
        _declared(payload, tmp_path)


def test_every_column_and_layer_takes_a_note_and_a_var_column_its_per_feature(
    tmp_path: Path,
) -> None:
    payload = _document_payload()
    base = payload["tables"][0]["base"]
    base["columns"]["obs"][0]["note"] = "one sample per run"
    for flag in base["columns"]["var"]:
        flag["note"] = "the vendor marks no such rows"
    base["measurements"]["layers"][0]["note"] = "a value per run and feature"
    payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"].append(
        {
            "name": "global_q",
            "source": "Global Q",
            "type": "number",
            "required": False,
            "per_feature": "min",
            "note": "experiment-wide; placeholder 1.0 in some rows",
        }
    )

    _declared(payload, tmp_path)
    with pytest.raises(ValidationError) as missing:
        SourcedVarColumn.model_validate(
            {"name": "gene", "source": "Gene", "type": "string", "required": True}
        )
    assert [error["loc"] for error in missing.value.errors()] == [("per_feature",)]


def _problems(error: ValidationError) -> list[tuple[str, tuple[int | str, ...]]]:
    """Each problem's pydantic type and location, which a message pattern cannot tell apart."""
    return [(problem["type"], problem["loc"]) for problem in error.errors()]


def test_numeric_layer_type_is_stated_and_rejects_unknown_values() -> None:
    assert _numeric_layer(name="count", source="Count", type="integer").type == "integer"

    with pytest.raises(ValidationError) as missing:
        NumericLayer.model_validate(
            {
                "encoding_mode": "numeric",
                "name": "quantity",
                "source": "Quantity",
                "required": False,
            }
        )
    assert _problems(missing.value) == [("missing", ("type",))]
    with pytest.raises(ValidationError) as unknown:
        _numeric_layer(name="quantity", source="Quantity", type="float")
    assert _problems(unknown.value) == [("literal_error", ("type",))]


def test_factor_layers_do_not_acquire_a_numeric_type() -> None:
    payload = {
        "encoding_mode": "factor",
        "name": "status",
        "source": "Status",
        "categories": {"identified": 1},
        "required": False,
        "type": "integer",
    }

    with pytest.raises(ValidationError) as refused:
        FactorLayer.model_validate(payload)
    assert _problems(refused.value) == [("extra_forbidden", ("type",))]


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
                {"repeated_cells": {"mode": "error"}}
            ),
            "repeated_cells",
            id="axis.repeated_cells",
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
                {"repeated_cells": {"mode": "keep_all_as_raw_table"}}
            ),
            "keep_all_as_raw_table",
            id="keep_all_as_raw_table",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"].update(
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
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"].update(
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
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["columns"]["obs"][0].update(
                {"per_feature": "best_row"}
            ),
            r"per_feature\n\s+Extra inputs are not permitted",
            id="obs-per-feature",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"][
                0
            ].update({"per_feature": "best_row"}),
            "resolve no repeated rows",
            id="var-key-per-feature",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"]["columns"][
                "var"
            ].append({"name": "gene", "source": "Gene", "per_feature": "min"}),
            "numeric column",
            id="min-on-text",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"]["columns"][
                "var"
            ].append({"name": "q", "source": "Q", "type": "number", "per_feature": "sum"}),
            "'first'",
            id="var-sum",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"]["columns"][
                "var"
            ].extend(
                [
                    {"name": "q", "source": "Q", "type": "number", "per_feature": "min"},
                    {"name": "q_copy", "source": "Q", "type": "number"},
                ]
            ),
            "resolves repeated rows both",
            id="one-source-two-modes",
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["levels"]["peptidoform_ion"]["columns"][
                "var"
            ].append({"name": "set", "source": "Set", "per_feature": "best_row"}),
            "needs the level's rank_rows",
            id="best-row-without-rank-rows",
        ),
        pytest.param(
            _with_ranking(
                {"by": "score", "best": "highest"},
                [{"name": "score", "source": "Score", "type": "number"}],
            ),
            "no column is best_row",
            id="rank-rows-without-best-row",
        ),
        pytest.param(
            _with_ranking(
                {"by": "score", "best": "middle"},
                [
                    {"name": "score", "source": "Score", "type": "number"},
                    {"name": "set", "source": "Set", "per_feature": "best_row"},
                ],
            ),
            "'highest'",
            id="rank-rows-best-value",
        ),
        *(
            pytest.param(_ranked_by(by, ranking), match, id=f"rank-rows-by-{label}")
            for label, by, ranking, match in [
                ("unknown", "score", [], "a sourced var column"),
                ("key", "feature", [], "a sourced var column"),
                ("itself", "set", [], "a sourced var column"),
                ("text", "score", [{"name": "score", "source": "Score"}], "required numeric"),
                (
                    "optional",
                    "score",
                    [{"name": "score", "source": "Score", "type": "number", "required": False}],
                    "required numeric",
                ),
            ]
        ),
        pytest.param(
            lambda payload: payload["tables"][0]["base"]["measurements"]["layers"][0].update(
                {"note": ""}
            ),
            "at least 1 character",
            id="empty-note",
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


@pytest.mark.parametrize("version", ["0.3", "0.5", "0.6", "0.7", "0.8", "0.9"])
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
    payload["tables"][0]["levels"]["peptidoform_ion"].update(invalid_level_declaration)
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValidationError):
        document.rule("peptidoform_ion", DDA)


def test_numeric_reduction_requires_layers_no_encoder_would_later_change(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["base"]["measurements"]["repeated_cells"] = {"mode": "sum"}
    payload["tables"][0]["base"]["measurements"]["layers"] = [
        {"name": "quantity", "source": "Quantity", "missing_values": [0]}
    ]

    document = make_rule_document(tmp_path / "rules.json", payload)
    with pytest.raises(ValueError, match="combines only plain numeric layers"):
        ParseRuleFacade(document, "peptidoform_ion", NO_EVIDENCE)


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

    assert document.declared("peptidoform_ion").input.delimiter is not None


# -------------------------------------------------------------------------- fragment rules


def test_the_packaged_fragment_level_separates_before_it_decomposes() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    rule = load_rule_document(pair.parser_v2_path).declared("peptidoform_fragment").declaration
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
        "peptidoform_fragment": {
            "columns": {"var": [{"name": "feature", "source": "Feature"}]},
            "fragments": {
                "label_strategy": "positional",
                "value_columns": ["Quantity"],
                "delimiter": ";",
                "label_output": "Quantity",
            },
        }
    }
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValueError, match="label_output"):
        ParseRuleFacade(document, "peptidoform_fragment", NO_EVIDENCE)


def test_a_column_labeled_fragment_label_may_not_be_selected(tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["levels"] = {
        "peptidoform_fragment": {
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
                "label_required": True,
                "delimiter": ";",
                "label_output": "fragment_label",
            },
        }
    }
    document = make_rule_document(tmp_path / "rules.json", payload)

    with pytest.raises(ValueError, match="label source"):
        ParseRuleFacade(document, "peptidoform_fragment", NO_EVIDENCE)


def test_column_labeled_recognition_requires_its_packed_label_column() -> None:
    payload = _document_payload()
    payload["tables"][0]["levels"] = {
        "peptidoform_fragment": {
            "columns": {"var": [{"name": "feature", "source": "Feature"}]},
            "fragments": {
                "label_strategy": "column",
                "value_columns": ["Quantity"],
                "label_column": "Info",
                "label_required": True,
                "delimiter": ";",
                "label_output": "fragment_label",
            },
        }
    }
    document = make_rule_document(Path("rules.json"), payload)
    working = ParseRuleFacade.from_declared_rule(
        document, "peptidoform_fragment"
    ).working_parameters

    assert working.accepts_header(("Sample", "Feature", "Quantity", "Info"))
    assert not working.accepts_header(("Sample", "Feature", "Quantity"))


# ------------------------------------------------------------------------ published schema


def test_only_schema_0_4_is_published_and_its_unions_are_discriminated() -> None:
    published = rule_json_schema()
    definitions = published["$defs"]
    text = json.dumps(published)

    assert isinstance(definitions, dict)
    assert "x_layer" not in text
    assert "keep_all_as_raw_table" not in text
    assert '"measurements"' in text
    for name in ("Measurements", "Axis", "RepeatedCells", "SearchParameterOverride"):
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
    rule = load_rule_document(pair.parser_v2_path).declared("peptidoform_ion").declaration
    by_name = {layer.name: layer for layer in rule.measurements.layers}

    assert isinstance(by_name["Match_Type"], FactorLayer)
    assert isinstance(by_name["Intensity"], NumericLayer)
    assert by_name["Match_Type"].categories == {"unmatched": 0, "MS/MS": 1, "MBR": 2}


def test_the_regex_value_pattern_survived_the_migration() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "peaks")
    rule = load_rule_document(pair.parser_v2_path).declared("peptidoform_ion").declaration
    ascore = next(layer for layer in rule.measurements.layers if layer.name == "AScore")

    assert isinstance(ascore, NumericLayer)
    assert isinstance(ascore.value_pattern, RegexValuePattern)


def test_both_rule_shapes_are_represented_by_the_packaged_generation() -> None:
    shapes = [
        load_rule_document(pair.parser_v2_path).declared(level).declaration
        for pair, level in level_pairs()
    ]

    assert sum(isinstance(rule, LongRule) for rule in shapes) == 25
    assert sum(isinstance(rule, WideRule) for rule in shapes) == 14
    modes = [rule.measurements.repeated_cells.mode for rule in shapes]
    assert modes.count("error") == 21
    assert modes.count("same") == 15
    assert modes.count("keep_best") == 3
    assert modes.count("sum") == 0
    assert sum(isinstance(rule.fragments, ColumnLabeledFragments) for rule in shapes) == 1


def test_the_published_artifact_is_the_schema_the_models_declare() -> None:
    committed = json.loads(artifact_path().read_text(encoding="utf-8"))

    assert committed == rule_json_schema()


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_packaged_levels_declare_reachable_hierarchy_identities(
    pair: PackagedDocument, level: str
) -> None:
    rule = load_rule_document(pair.parser_v2_path).declared(level).declaration
    hierarchy = dict(HIERARCHIES[rule.hierarchy])
    assert level in hierarchy
    roles = {role: column.name for column in rule.columns.var for role in column.roles}
    available = {*rule.axis.var_keys, *(column.name for column in rule.columns.var)}
    coarser = list(hierarchy)[list(hierarchy).index(level) + 1 :]
    assert not coarser or any(
        roles.get(hierarchy[name], hierarchy[name]) in available for name in coarser
    )


@pytest.mark.parametrize("dtype", ["integer", "number", "boolean"])
def test_rule_var_roles_reject_non_string_types(dtype: str, tmp_path: Path) -> None:
    payload = _document_payload()
    payload["tables"][0]["levels"]["peptidoform_ion"]["columns"]["var"][0].update(
        {"type": dtype, "roles": ["protein_assignment"]}
    )
    with pytest.raises(ValidationError, match="string"):
        _declared(payload, tmp_path)
