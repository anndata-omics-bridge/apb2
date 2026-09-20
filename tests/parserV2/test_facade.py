"""``ParseRuleFacade``: one generic key walk, and one atomic resolution per source.

The point of these tests is that there is no vendor in the facade. Every packaged level goes
through the same dependency walk, and the interesting cases — a modification-derived key, a
fragment-derived key, a pruned optional chain, a permissive wide pattern — are properties of
declarations, so each is asserted on whichever document happens to declare it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

from apb2.parserV2.parse_quant.axis_columns import (
    CoalesceColumn,
    JoinNonemptyColumn,
    ProformaIonColumn,
)
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.duplicates import AggregateNumericDuplicates
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.fragments import PositionalFragmentTableSeparator
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.modifications import (
    SequenceColumn,
    SiteListNormalizer,
    TokenRegexNormalizer,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisKeyPlan,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    FactorLayerDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedSourceEvidence,
    FrameSourceEvidence,
    LevelReadPlan,
    NumericTextFormat,
    WideRawLayerPlan,
    WideRawLayerSource,
)
from apb2.parserV2.parse_quant.parser import ParseStrategy
from apb2.parserV2.parse_quant.plan_json import as_json_value
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import RuleNotApplicable, SearchParameterEvidence
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2 import synthetic
from parserV2.fixtures import PackagedDocument, document_pairs, level_pairs
from parserV2.synthetic import plan_snapshot as snapshot

NUMBERS = NumericTextFormat(decimal_mark=".", thousands_marks=())

_EVIDENCES = (
    SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None),
    SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False),
    SearchParameterEvidence(acquisition_method="DIA", combine_charge_states=True),
)

# Cached exports that do not carry their level's columns; conversion skips those unavailable
# levels for the same reason.
_INCOMPATIBLE = {
    ("spectronaut", "fragment"),
    ("spectronaut/v21", "fragment"),
    ("wombat", "ion"),
}

_LEVEL_CASES = [
    pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()
]


def delimited(columns: tuple[str, ...]) -> DelimitedSourceEvidence:
    """Header evidence for a tab-delimited, dot-decimal source."""
    return DelimitedSourceEvidence(
        columns=columns,
        delimiter="\t",
        quote_char='"',
        encoding="utf8",
        number_format=NUMBERS,
    )


def _facade_for(pair: PackagedDocument, level: QuantificationLevel) -> ParseRuleFacade:
    """The facade for one packaged level, under whichever evidence its gate admits."""
    document = load_rule_document(pair.parser_v2_path)
    for evidence in _EVIDENCES:
        try:
            return ParseRuleFacade(document, level, evidence)
        except RuleNotApplicable:
            continue
    raise AssertionError(f"no evidence admits {pair.key}/{level}")


def _contains_model(value: object, seen: set[int] | None = None) -> bool:
    """Whether a Pydantic model is reachable anywhere inside a returned value."""
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, BaseModel):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_model(getattr(value, field.name), seen) for field in dataclasses.fields(value)
        )
    slots = getattr(type(value), "__slots__", ())
    if slots:
        names = (slots,) if isinstance(slots, str) else slots
        return any(_contains_model(getattr(value, name), seen) for name in names)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_model(item, seen) for item in value)
    if isinstance(value, dict):
        return any(_contains_model(item, seen) for item in value.values())
    return False


# ------------------------------------------------------------------- every packaged level


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_every_packaged_level_projects_working_parameters(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    working = _facade_for(pair, level).working_parameters

    assert working.level == level
    assert working.obs.final_key_columns
    assert working.var.final_key_columns
    assert working.measurements.primary_layer_name in {
        layer.name for layer in working.measurements.layers
    }
    assert working.measurements.primary_layer_name in {
        layer.name for layer in working.measurements.required_layers
    }
    assert working.input.formats
    assert not _contains_model(working)


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_every_compatible_level_resolves_both_axis_key_plans(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    facade = _facade_for(pair, level)
    header = pair.header()
    if (pair.key, level) in _INCOMPATIBLE:
        with pytest.raises(IncompatibleSourceError):
            facade.resolve_source(delimited(header))
        return
    resolved = facade.resolve_source(delimited(header))

    for axis in (resolved.obs, resolved.var):
        keys = axis.keys
        assert keys.raw_key_columns
        assert keys.key_input_columns
        assert keys.final_key_columns
        assert set(keys.final_key_columns) <= set(axis.outputs)
    assert resolved.level == level
    assert not _contains_model(resolved)


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_a_delimited_plan_decides_every_projected_column_dtype(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    if (pair.key, level) in _INCOMPATIBLE:
        pytest.skip(f"cached export lacks columns for {pair.key}/{level}")
    read = _facade_for(pair, level).resolve_source(delimited(pair.header())).read

    assert read.text_sources.isdisjoint(read.native_numeric_sources)
    assert read.text_sources | read.native_numeric_sources == set(read.projected_columns)
    assert len(read.projected_columns) == len(set(read.projected_columns))


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_projected_columns_stay_in_physical_header_order(
    pair: PackagedDocument, level: QuantificationLevel
) -> None:
    if (pair.key, level) in _INCOMPATIBLE:
        pytest.skip(f"cached export lacks columns for {pair.key}/{level}")
    header = pair.header()
    projected = _facade_for(pair, level).resolve_source(delimited(header)).read.projected_columns

    assert list(projected) == [name for name in header if name in set(projected)]


# ------------------------------------------------------- the worked examples in the spec


def test_the_alphadia_wide_ion_level_resolves_exactly_as_specified() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_10/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "ion", _EVIDENCES[0])
    header = (
        "sequence",
        "mods",
        "mod_sites",
        "charge",
        "genes",
        "decoy",
        "run_A",
        "run_B",
    )

    working = facade.working_parameters
    assert working.obs.final_key_columns == ("sample",)
    assert working.var.final_key_columns == ("ProForma_ion",)
    assert working.measurements.primary_layer_name == "Intensity"
    assert working.measurements.duplicate_mode == "keep_first"

    resolved = facade.resolve_source(delimited(header))
    assert resolved.obs.keys == AxisKeyPlan(
        raw_key_columns=("sample",),
        key_input_columns=("sample",),
        final_key_columns=("sample",),
    )
    assert resolved.var.keys == AxisKeyPlan(
        raw_key_columns=("sequence", "mods", "mod_sites", "charge"),
        key_input_columns=("ProForma_peptidoform", "Charge"),
        final_key_columns=("ProForma_ion",),
    )
    # The specification's example reads the two measurement columns natively. Real exports
    # write "-", "NA", and "False" in a column a rule calls numeric, which an eager numeric
    # read cannot survive, so a measurement is read natively only where the rule sums it.
    assert resolved.read == LevelReadPlan(
        projected_columns=header,
        text_sources=frozenset(header),
        native_numeric_sources=frozenset(),
    )
    assert isinstance(resolved.decomposer, WideSourceDecomposer)
    assert resolved.decomposer == WideSourceDecomposer(
        obs=resolved.decomposer.obs,
        var=resolved.decomposer.var,
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity",
                sources=(
                    WideRawLayerSource(source_column="run_A", sample="run_A"),
                    WideRawLayerSource(source_column="run_B", sample="run_B"),
                ),
            ),
        ),
    )
    assert snapshot(resolved)["layer_values"] == [
        {
            "layer_name": "Intensity",
            "value": {"kind": "plain_numeric", "missing_values": [0.0], "type": "number"},
        }
    ]
    assert resolved.layer_validator == LayerContractValidator(
        primary_layer_name="Intensity",
        required_names=("Intensity",),
        empty_ratio=0.001,
        populated_ratio=0.5,
        strict=False,
    )


def test_the_diann_fragment_level_separates_packed_values_before_decomposing() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/diann/v1_8/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "fragment", _EVIDENCES[0])
    header = (
        "Run",
        "Modified.Sequence",
        "Precursor.Charge",
        "Protein.Group",
        "Fragment.Quant.Raw",
        "Fragment.Correlations",
    )

    resolved = facade.resolve_source(delimited(header))

    assert resolved.var.keys == AxisKeyPlan(
        raw_key_columns=("Modified.Sequence", "Precursor.Charge", "fragment_label"),
        key_input_columns=("ProForma_ion", "fragment_label"),
        final_key_columns=("ProForma_fragment",),
    )
    decomposition = resolved.decomposer
    assert isinstance(decomposition, DelimitedFragmentSourceDecomposer)
    assert decomposition.separator == PositionalFragmentTableSeparator(
        label_output="fragment_label",
        delimiter=";",
        packed_value_sources=("Fragment.Quant.Raw", "Fragment.Correlations"),
    )
    assert isinstance(decomposition.long_decomposer, LongSourceDecomposer)
    assert decomposition.long_decomposer.primary_layer_name == "Fragment_Quant_Raw"
    # The label the separator synthesizes is identity, so its tokens must survive as text.
    assert "Fragment.Quant.Raw" in resolved.read.text_sources


# ------------------------------------------------------- what the generic walk must find


def test_a_modification_derived_key_pulls_every_source_that_can_change_it() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v2/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "ion", _EVIDENCES[0])
    modifications = tuple(
        column.operation
        for column in facade.working_parameters.var.computed
        if isinstance(column, SequenceColumn) and isinstance(column.operation, SiteListNormalizer)
    )

    assert len(modifications) == 1
    config = modifications[0]
    assert isinstance(config, SiteListNormalizer)
    assert config.entries
    assert {entry.accession for entry in config.entries} == {
        "UNIMOD:1",
        "UNIMOD:4",
        "UNIMOD:35",
    }
    assert all(entry.mass_delta for entry in config.entries)


def test_a_token_regex_rule_resolves_its_accessions_at_projection() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/maxquant/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "ion", _EVIDENCES[0])
    column = next(
        column
        for column in facade.working_parameters.var.computed
        if isinstance(column, SequenceColumn) and isinstance(column.operation, TokenRegexNormalizer)
    )
    assert isinstance(column.operation, TokenRegexNormalizer)
    config = column.operation

    assert isinstance(config, TokenRegexNormalizer)
    assert column.inputs == ("Modified_Sequence",)
    assert {entry.token for entry in config.entries} >= {"ac", "ox"}
    assert all(entry.name for entry in config.entries)


def test_a_directly_selected_key_is_its_own_input() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/diann/v1_8/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "protein", _EVIDENCES[0])
    header = ("Run", "Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "PG.MaxLFQ")

    keys = facade.resolve_source(delimited(header)).var.keys

    assert keys == AxisKeyPlan(
        raw_key_columns=("Protein.Group",),
        key_input_columns=("Protein_Group",),
        final_key_columns=("Protein_Group",),
    )


def test_a_multi_column_key_keeps_every_authored_component_in_order() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/spectronaut/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "fragment", _EVIDENCES[0])
    header = (
        "R.FileName",
        "R.Condition",
        "PG.ProteinGroups",
        "PG.ProteinAccessions",
        "PEP.StrippedSequence",
        "EG.ModifiedSequence",
        "EG.PrecursorId",
        "FG.Charge",
        "F.FrgIon",
        "F.Charge",
        "F.FrgLossType",
        "F.FrgType",
        "F.FrgNum",
        "F.FrgMz",
        "F.TheoreticalMz",
        "F.PeakArea",
    )

    keys = facade.resolve_source(delimited(header)).var.keys

    assert keys.final_key_columns == (
        "EG_PrecursorId",
        "F_FrgIon",
        "F_Charge",
        "F_FrgLossType",
        "F_FrgType",
        "F_FrgNum",
    )
    assert keys.raw_key_columns == (
        "EG.PrecursorId",
        "F.FrgIon",
        "F.Charge",
        "F.FrgLossType",
        "F.FrgType",
        "F.FrgNum",
    )
    assert keys.key_input_columns == keys.final_key_columns


def test_payload_metadata_is_retained_without_becoming_identity() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_12/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "ion", _EVIDENCES[0])
    header = (
        "run",
        "sequence",
        "charge",
        "mods",
        "mod_sites",
        "genes",
        "proteins",
        "pg",
        "pg_master",
        "decoy",
        "intensity",
    )

    var_strategy = facade.resolve_source(delimited(header))
    var = var_strategy.var

    assert "genes" not in var.keys.raw_key_columns
    assert "genes" in snapshot(var_strategy)["var"]["source"]["payload_sources"]
    assert "Genes" in var.outputs


# ---------------------------------------------------------------- optional-source pruning


def test_an_absent_optional_input_narrows_a_coalesce_instead_of_blocking_it() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Primary": "Primary"},
        var_optional={"Fallback": "Fallback"},
        computed=[
            {"name": "Merged", "inputs": ["Primary", "Fallback"], "how": "coalesce"},
        ],
        var_keys=["Feature"],
    )
    facade = synthetic.facade(document)

    both_strategy = facade.resolve_source(
        delimited(("Sample", "Feature", "Primary", "Fallback", "Quantity"))
    )
    both = both_strategy.var
    only_primary_strategy = facade.resolve_source(
        delimited(("Sample", "Feature", "Primary", "Quantity"))
    )
    only_primary = only_primary_strategy.var

    assert set(snapshot(both_strategy)["var"]["skipped"]) == frozenset()
    assert both.output_phase.computers == (
        CoalesceColumn(name="Merged", inputs=("Primary", "Fallback")),
    )
    assert set(snapshot(only_primary_strategy)["var"]["skipped"]) == frozenset({"Fallback"})
    assert only_primary.output_phase.computers == (
        CoalesceColumn(name="Merged", inputs=("Primary",)),
    )
    assert "Fallback" not in only_primary.outputs
    assert "Merged" in only_primary.outputs


def test_pruning_removes_exactly_the_chain_a_missing_optional_blocks() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Charge": "Charge"},
        var_optional={"Extra": "Extra"},
        var_types={"Charge": "integer"},
        computed=[
            {
                "name": "Joined",
                "inputs": ["Feature", "Charge"],
                "how": "join_nonempty",
                "separator": "-",
            },
            {
                "name": "Downstream",
                "inputs": ["Joined", "Extra"],
                "how": "join_nonempty",
                "separator": "+",
            },
        ],
        var_keys=["Feature"],
    )
    facade = synthetic.facade(document)

    var_strategy = facade.resolve_source(delimited(("Sample", "Feature", "Charge", "Quantity")))
    var = var_strategy.var

    # ``join_nonempty`` keeps the inputs it has, so only the absent name is skipped.
    assert set(snapshot(var_strategy)["var"]["skipped"]) == frozenset({"Extra"})
    assert var.output_phase.computers == (
        JoinNonemptyColumn(name="Joined", inputs=("Feature", "Charge"), separator="-"),
        JoinNonemptyColumn(name="Downstream", inputs=("Joined",), separator="+"),
    )


def test_a_blocked_sequence_operation_removes_itself_and_its_consumers() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        var_optional={"Charge": "Charge"},
        var_types={"Charge": "integer"},
        computed=[
            {"name": "Ion", "inputs": ["Feature", "Charge"], "how": "coalesce"},
            {
                "name": "Label",
                "inputs": ["Ion", "Feature"],
                "how": "join_nonempty",
                "separator": "/",
            },
        ],
        var_keys=["Feature"],
    )
    facade = synthetic.facade(document)

    var_strategy = facade.resolve_source(delimited(("Sample", "Feature", "Quantity")))
    var = var_strategy.var

    assert set(snapshot(var_strategy)["var"]["skipped"]) == frozenset({"Charge"})
    assert var.output_phase.computers[0].inputs == ("Feature",)


def test_a_missing_dependency_of_a_final_key_makes_the_level_incompatible() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Charge": "Charge"},
        var_types={"Charge": "integer"},
        var_keys=["Feature"],
    )
    facade = synthetic.facade(document)

    with pytest.raises(IncompatibleSourceError, match="Charge"):
        facade.resolve_source(delimited(("Sample", "Feature", "Quantity")))


@pytest.mark.parametrize("how", ["coalesce", "join_nonempty"])
def test_combining_operations_with_no_surviving_inputs_are_skipped(how: str) -> None:
    document = synthetic.long_document(
        obs_select={"Sample": "Sample"},
        var_select={"Feature": "Feature"},
        var_optional={"First": "First", "Second": "Second"},
        computed=[
            {
                "name": "Merged",
                "inputs": ["First", "Second"],
                "how": how,
                **({"separator": "|"} if how == "join_nonempty" else {}),
            }
        ],
    )
    var_strategy = synthetic.facade(document).resolve_source(
        delimited(("Sample", "Feature", "Quantity"))
    )
    var = var_strategy.var

    assert set(snapshot(var_strategy)["var"]["skipped"]) == {"First", "Second", "Merged"}
    assert var.output_phase.computers == ()
    assert "Merged" not in var.outputs


def test_a_non_injective_coalesce_is_planned_and_left_for_the_parser_to_catch() -> None:
    """Injectivity is a property of the data, so resolution plans it either way."""
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"First": "First", "Second": "Second"},
        computed=[{"name": "Key", "inputs": ["First", "Second"], "how": "coalesce"}],
        var_keys=["Key"],
    )
    facade = synthetic.facade(document)

    var_strategy = facade.resolve_source(delimited(("Sample", "First", "Second", "Quantity")))
    var = var_strategy.var

    assert var.keys == AxisKeyPlan(
        raw_key_columns=("First", "Second"),
        key_input_columns=("First", "Second"),
        final_key_columns=("Key",),
    )
    assert var.key_phase.computers[0].name == "Key"
    assert var.output_phase.computers == ()


def test_the_key_phase_holds_identity_and_the_output_phase_holds_the_rest() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Charge": "Charge", "Gene": "Gene"},
        var_types={"Charge": "integer"},
        computed=[
            {
                "name": "Key",
                "inputs": ["Feature", "Charge"],
                "how": "join_nonempty",
                "separator": "/",
            },
            {
                "name": "Note",
                "inputs": ["Gene", "Feature"],
                "how": "join_nonempty",
                "separator": ",",
            },
        ],
        var_keys=["Key"],
    )
    facade = synthetic.facade(document)

    var_strategy = facade.resolve_source(
        delimited(("Sample", "Feature", "Charge", "Gene", "Quantity"))
    )
    var = var_strategy.var

    assert [selection.name for selection in var.key_phase.selections] == ["Feature", "Charge"]
    assert [computer.name for computer in var.key_phase.computers] == ["Key"]
    assert [selection.name for selection in var.output_phase.selections] == ["Gene"]
    assert [computer.name for computer in var.output_phase.computers] == ["Note"]


def test_a_computed_column_may_widen_the_selection_of_its_own_name() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Proteins": "Proteins"},
        var_optional={"Leading": "Leading"},
        computed=[{"name": "Proteins", "inputs": ["Proteins", "Leading"], "how": "coalesce"}],
        var_keys=["Feature"],
    )
    facade = synthetic.facade(document)

    var_strategy = facade.resolve_source(
        delimited(("Sample", "Feature", "Proteins", "Leading", "Quantity"))
    )
    var = var_strategy.var

    assert var.output_phase.computers[0].inputs == ("Proteins", "Leading")
    assert "Proteins" in snapshot(var_strategy)["var"]["source"]["payload_sources"]
    assert "Leading" in snapshot(var_strategy)["var"]["source"]["payload_sources"]


# ------------------------------------------------------------------------- wide resolution


def test_wide_layers_resolve_to_concrete_columns_in_header_order() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"},
            {"name": "Count", "source": r"^(?P<sample>.+) Count$"},
        ],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)
    header = ("Feature", "B Intensity", "A Intensity", "A Count", "B Count")

    resolved = facade.resolve_source(delimited(header))
    decomposition = resolved.decomposer
    assert isinstance(decomposition, WideSourceDecomposer)
    plans = {plan.name: plan for plan in decomposition.layer_plans}

    assert [source.sample for source in plans["Intensity"].sources] == ["B", "A"]
    assert [source.source_column for source in plans["Count"].sources] == ["A Count", "B Count"]
    assert resolved.obs.keys.raw_key_columns == ("sample",)


def test_a_permissive_wide_pattern_never_claims_an_accounted_for_column() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature", "Gene": "Gene"},
        layers=[{"name": "Intensity", "source": r"^(?P<sample>.+)$"}],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(delimited(("Feature", "Gene", "run_1", "run_2")))
    decomposition = resolved.decomposer
    assert isinstance(decomposition, WideSourceDecomposer)

    assert [source.sample for source in decomposition.layer_plans[0].sources] == [
        "run_1",
        "run_2",
    ]


def test_an_optional_wide_layer_without_aligned_samples_is_omitted() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"},
            {"name": "Count", "source": r"^(?P<sample>.+) Count$"},
        ],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(delimited(("Feature", "A Intensity", "Z Count")))
    decomposition = resolved.decomposer
    assert isinstance(decomposition, WideSourceDecomposer)

    assert [plan.name for plan in decomposition.layer_plans] == ["Intensity"]
    assert list(resolved.layer_parsers) == ["Intensity"]


def test_a_required_wide_layer_matching_only_other_samples_stays_as_an_empty_layer() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"},
            {"name": "Count", "source": r"^(?P<sample>.+) Count$", "required": True},
        ],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(delimited(("Feature", "A Intensity", "Z Count")))
    decomposition = resolved.decomposer
    assert isinstance(decomposition, WideSourceDecomposer)
    plans = {plan.name: plan for plan in decomposition.layer_plans}

    assert plans["Count"].sources == ()
    assert isinstance(resolved.layer_validator, LayerContractValidator)
    assert resolved.layer_validator.required_names == ("Intensity", "Count")


def test_a_required_wide_layer_with_no_match_at_all_is_incompatible() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"},
            {"name": "Count", "source": r"^(?P<sample>.+) Count$", "required": True},
        ],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)

    with pytest.raises(IncompatibleSourceError, match="Count"):
        facade.resolve_source(delimited(("Feature", "A Intensity")))


def test_a_primary_wide_layer_with_no_match_is_incompatible() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[{"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"}],
        primary_layer="Intensity",
    )
    facade = synthetic.facade(document)

    with pytest.raises(IncompatibleSourceError, match="primary layer"):
        facade.resolve_source(delimited(("Feature", "A Count")))


# ------------------------------------------------------------------------ long resolution


def test_an_optional_long_layer_absent_from_the_header_is_omitted() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Quantity", "source": "Quantity", "roles": ["abundance"]},
            {"name": "Other", "source": "Other", "roles": ["abundance"]},
        ],
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(delimited(("Sample", "Feature", "Quantity")))
    decomposition = resolved.decomposer
    assert isinstance(decomposition, LongSourceDecomposer)

    assert [source.name for source in decomposition.layer_sources] == ["Quantity"]
    assert isinstance(resolved.layer_validator, LayerContractValidator)
    assert resolved.layer_validator.required_names == ("Quantity",)
    assert resolved.provenance["layer_roles"] == {"abundance": ["Quantity"]}


def test_a_required_long_layer_absent_from_the_header_is_incompatible() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Quantity", "source": "Quantity"},
            {"name": "Score", "source": "Score", "required": True},
        ],
    )
    facade = synthetic.facade(document)

    with pytest.raises(IncompatibleSourceError, match="Score"):
        facade.resolve_source(delimited(("Sample", "Feature", "Quantity")))


def test_layers_keep_their_authored_order_across_the_required_split() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "First", "source": "First"},
            {"name": "Quantity", "source": "Quantity"},
            {"name": "Last", "source": "Last", "required": True},
        ],
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(delimited(("Sample", "Feature", "First", "Quantity", "Last")))

    assert list(resolved.layer_parsers) == [
        "First",
        "Quantity",
        "Last",
    ]


# ------------------------------------------------------------ retained layer declarations


def test_resolution_reuses_each_retained_layer_declaration_without_copying_it() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        layers=[
            {"name": "Quantity", "source": "Quantity", "missing_values": [0]},
            {"name": "Plain", "source": "Plain"},
            {
                "name": "Structured",
                "source": "Structured",
                "value_pattern": {"mode": "regex", "pattern": r":(-?\d+(?:\.\d+)?)"},
            },
            {
                "name": "Kind",
                "source": "Kind",
                "encoding_mode": "factor",
                "categories": {"a": 0, "b": 1},
            },
        ],
    )
    facade = synthetic.facade(document)

    resolved = facade.resolve_source(
        delimited(("Sample", "Feature", "Quantity", "Plain", "Structured", "Kind"))
    )
    values = {
        config["layer_name"]: config["value"] for config in snapshot(resolved)["layer_values"]
    }
    for layer in facade.working_parameters.measurements.layers:
        assert values[layer.name] == as_json_value(layer.value)
    assert values["Quantity"] == as_json_value(PlainNumericLayerDeclaration(missing_values=(0.0,)))
    assert values["Plain"] == as_json_value(PlainNumericLayerDeclaration(missing_values=()))
    assert values["Structured"] == as_json_value(
        RegexNumericLayerDeclaration(missing_values=(), pattern=r":(-?\d+(?:\.\d+)?)")
    )
    assert values["Kind"] == as_json_value(FactorLayerDeclaration(categories=(("a", 0), ("b", 1))))
    # Every measurement keeps its tokens; only an aggregating rule reads them as numbers.
    assert {"Kind", "Structured", "Plain", "Quantity"} <= resolved.read.text_sources
    assert resolved.read.native_numeric_sources == frozenset()


def test_a_grouped_number_notation_leaves_every_value_as_text() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
    )
    facade = synthetic.facade(document)
    grouped = dataclasses.replace(
        delimited(("Sample", "Feature", "Quantity")),
        number_format=NumericTextFormat(decimal_mark=",", thousands_marks=(".",)),
    )

    resolved = facade.resolve_source(grouped)
    read = resolved.read

    assert snapshot(resolved)["number_format"] == as_json_value(grouped.number_format)
    assert read.native_numeric_sources == frozenset()
    assert read.text_sources == set(read.projected_columns)


# -------------------------------------------------------------------------- other evidence


def test_parquet_evidence_keeps_its_physical_schema() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
    )
    facade = synthetic.facade(document)
    evidence = FrameSourceEvidence(
        columns=("Sample", "Feature", "Quantity", "Unused"),
        dtypes=(
            ("Sample", pl.String()),
            ("Feature", pl.String()),
            ("Quantity", pl.Float64()),
            ("Unused", pl.Int64()),
        ),
    )

    read = facade.resolve_source(evidence).read

    assert read.projected_columns == ("Sample", "Feature", "Quantity")
    assert read.text_sources == frozenset()
    assert read.native_numeric_sources == frozenset()


def test_an_aggregate_rule_needs_layer_values_this_source_delivers_as_numbers() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        duplicates="aggregate",
    )
    facade = synthetic.facade(document)
    grouped = dataclasses.replace(
        delimited(("Sample", "Feature", "Quantity")),
        number_format=NumericTextFormat(decimal_mark=",", thousands_marks=(".",)),
    )

    resolved = facade.resolve_source(delimited(("Sample", "Feature", "Quantity")))
    assert isinstance(resolved.duplicates, AggregateNumericDuplicates)
    # Summing needs numbers, so an aggregating rule is the one case that reads them eagerly.
    assert resolved.read.native_numeric_sources == frozenset({"Quantity"})
    with pytest.raises(IncompatibleSourceError, match="native numeric"):
        facade.resolve_source(grouped)


def test_a_missing_modification_source_makes_the_level_incompatible() -> None:
    path = Path("src/apb2/parserV2/vendor_parse_rules/documents/diann/v1_8/rules.json")
    facade = ParseRuleFacade(load_rule_document(path), "ion", _EVIDENCES[0])
    header = (
        "Run",
        "Stripped.Sequence",
        "Precursor.Charge",
        "Precursor.Id",
        "Protein.Group",
        "Protein.Ids",
        "Protein.Names",
        "Genes",
        "Precursor.Normalised",
    )

    with pytest.raises(IncompatibleSourceError, match=r"Modified\.Sequence"):
        facade.resolve_source(delimited(header))


def test_the_facade_exposes_only_resolution_and_header_only_construction() -> None:
    public = {
        name
        for name in dir(ParseRuleFacade)
        if not name.startswith("_") and callable(getattr(ParseRuleFacade, name, None))
    }

    assert public == {"from_declared_rule", "resolve_source"}
    assert isinstance(ParseRuleFacade.working_parameters, property)
    assert ParseRuleFacade.__slots__ == ("_configuration",)


def test_the_facade_retains_no_rule_model_after_construction() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    facade = ParseRuleFacade(load_rule_document(pair.parser_v2_path), "ion", _EVIDENCES[0])

    assert not _contains_model(facade.working_parameters)
    assert not _contains_model(facade.resolve_source(delimited(pair.header())))
    assert vars(ParseRuleFacade).get("__slots__") == ("_configuration",)


def test_a_resolved_plan_carries_every_field_the_compiler_destructures() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "alphapept")
    facade = ParseRuleFacade(load_rule_document(pair.parser_v2_path), "ion", _EVIDENCES[0])

    resolved = facade.resolve_source(delimited(pair.header()))

    assert {field.name for field in dataclasses.fields(ParseStrategy)} == {
        "level",
        "read",
        "decomposer",
        "obs",
        "var",
        "duplicates",
        "layer_parsers",
        "layer_validator",
        "provenance",
    }
    assert all(getattr(resolved, field) is not None for field in ("read", "decomposer"))
    assert isinstance(resolved.provenance["rule_json"], str)


def test_a_proforma_ion_computer_reads_the_peptidoform_and_the_typed_charge() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    facade = ParseRuleFacade(
        load_rule_document(pair.parser_v2_path),
        "ion",
        SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=False),
    )

    var_strategy = facade.resolve_source(delimited(pair.header()))
    var = var_strategy.var
    computers = {computer.name: computer for computer in var.key_phase.computers}

    assert computers["ProForma_ion"] == ProformaIonColumn(
        name="ProForma_ion",
        inputs=("ProForma_peptidoform", "Charge"),
    )
    assert computers["ProForma_peptidoform"].inputs == ("Peptide",)
