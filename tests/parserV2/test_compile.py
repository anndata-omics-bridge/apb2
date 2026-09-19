"""The composition root: every tag consumed once, and nothing carries one afterwards.

What these tests are for is the claim the whole architecture rests on — that after compilation
no object knows what vendor, level, layout, encoding, duplicate mode, or output format it came
from. So they check the registries for coverage, the constructed graph for tags, and
the compiler objects for the ordering and skipping behaviour a multi-level caller relies on.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, get_args

import pytest

from apb2.parserV2.compile import ExplicitRuleCompiler
from apb2.parserV2.parse_quant.axis_columns import (
    BooleanAxisCoercer,
    CoalesceColumn,
    IntegerAxisCoercer,
    JoinNonemptyColumn,
    NumberAxisCoercer,
    ProformaFragmentColumn,
    ProformaIonColumn,
    StringAxisCoercer,
)
from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
    NullOnlyRawValuePresence,
    PlainNumericRawValuePresence,
    RegexNumericRawValuePresence,
)
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.modifications import (
    SequenceColumn,
    SiteListNormalizer,
    TokenRegexNormalizer,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisKeyPlan,
    AxisLogicalType,
    AxisSourcePlan,
    CoalesceColumnConfig,
    JoinNonemptyColumnConfig,
    PlainSequenceSyntaxConfig,
    ProformaFragmentColumnConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
    TokenRegexModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    FactorLayerDeclaration,
    LayerContractConfig,
    LayerValueConfig,
    LayerValueDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentSeparationConfig,
    DelimitedFragmentDecompositionConfig,
    InputSource,
    LongDecompositionConfig,
    LongRawLayerSource,
    NumericTextFormat,
    PositionalFragmentSeparationConfig,
    SingleFile,
    WideDecompositionConfig,
    WideRawLayerPlan,
    WideRawLayerSource,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.parser_factory import (
    compile_level,
    duplicate_policy_for,
    make_axis_coercer,
    make_column_computer,
    make_fragment_table_separator,
    make_layer_operations,
    make_layer_validator,
    make_sequence_normalizer,
    make_source_decomposer,
)
from apb2.parserV2.vendor_parse_rules.document import make_rule_document
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, SCHEMA_VERSION
from parserV2 import synthetic
from parserV2.fixtures import PackagedDocument, document_pairs, level_pairs

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
DOT_NUMBERS = NumberNotation(decimal_mark=".", thousands_marks=())
AXIS = AxisSourcePlan(
    keys=AxisKeyPlan(raw_key_columns=("a",), key_input_columns=("A",), final_key_columns=("A",)),
    payload_sources=(),
)


def entries() -> tuple[str, ...]:
    return ("token",)


# ------------------------------------------------------------------------ registry coverage


@pytest.mark.parametrize("logical_type", get_args(AxisLogicalType.__value__))
def test_every_declared_logical_type_names_one_coercer(logical_type: AxisLogicalType) -> None:
    coercer = make_axis_coercer(logical_type, DOT)

    assert type(coercer) in {
        StringAxisCoercer,
        IntegerAxisCoercer,
        NumberAxisCoercer,
        BooleanAxisCoercer,
    }
    assert not hasattr(coercer, "logical_type")
    if isinstance(coercer, (IntegerAxisCoercer, NumberAxisCoercer)):
        assert coercer.notation == DOT_NUMBERS


def test_the_four_coercers_are_four_different_implementations() -> None:
    selected = {
        type(make_axis_coercer(logical, DOT)) for logical in get_args(AxisLogicalType.__value__)
    }

    assert len(selected) == 4


@pytest.mark.parametrize("mode", get_args(DuplicateMode.__value__))
def test_every_executable_duplicate_mode_names_one_policy(mode: DuplicateMode) -> None:
    policy = duplicate_policy_for(mode)

    assert type(policy) in {ErrorOnDuplicates, KeepFirstDuplicate, AggregateNumericDuplicates}
    assert not hasattr(policy, "mode")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (CoalesceColumnConfig(kind="coalesce", name="C", inputs=("a", "b")), CoalesceColumn),
        (
            JoinNonemptyColumnConfig(
                kind="join_nonempty", name="J", inputs=("a", "b"), separator=","
            ),
            JoinNonemptyColumn,
        ),
        (
            StrippedSequenceColumnConfig(
                kind="stripped_sequence",
                name="S",
                inputs=("Sequence",),
                syntax=PlainSequenceSyntaxConfig(kind="plain_sequence"),
            ),
            SequenceColumn,
        ),
        (
            ProformaSequenceColumnConfig(
                kind="proforma_sequence",
                name="P",
                inputs=("Modified_Sequence",),
                normalization=TokenRegexModificationConfig(
                    kind="token_regex",
                    token_pattern=r"\(([^()]*)\)",
                    token_position="after_residue",
                    case_sensitive=False,
                    unknown_policy="preserve",
                    entries=(),
                ),
            ),
            SequenceColumn,
        ),
        (
            ProformaIonColumnConfig(kind="proforma_ion", name="I", inputs=("P", "Z")),
            ProformaIonColumn,
        ),
        (
            ProformaFragmentColumnConfig(kind="proforma_fragment", name="F", inputs=("I", "L")),
            ProformaFragmentColumn,
        ),
    ],
    ids=lambda value: getattr(value, "kind", getattr(value, "__name__", "")),
)
def test_every_computed_column_declaration_names_one_computer(
    config: object, expected: type
) -> None:
    computer = make_column_computer(config)  # pyright: ignore[reportArgumentType]

    assert isinstance(computer, expected)
    assert not hasattr(computer, "kind")


@pytest.mark.parametrize(
    ("value", "expected_presence", "expected_parser"),
    [
        (
            PlainNumericLayerDeclaration(missing_values=()),
            NullOnlyRawValuePresence,
            PlainNumericLayerParser,
        ),
        (
            PlainNumericLayerDeclaration(missing_values=(0.0,)),
            PlainNumericRawValuePresence,
            PlainNumericLayerParser,
        ),
        (
            RegexNumericLayerDeclaration(missing_values=(0.0,), pattern=r"(\d+)"),
            RegexNumericRawValuePresence,
            RegexNumericLayerParser,
        ),
        (
            RegexNumericLayerDeclaration(missing_values=(), pattern=r"(\d+)"),
            RegexNumericRawValuePresence,
            RegexNumericLayerParser,
        ),
        (
            FactorLayerDeclaration(categories=(("a", 0),)),
            NullOnlyRawValuePresence,
            FactorLayerParser,
        ),
    ],
    ids=lambda value: getattr(value, "kind", getattr(value, "__name__", "")),
)
def test_one_declaration_selects_both_tagless_layer_operations(
    value: LayerValueDeclaration, expected_presence: type, expected_parser: type
) -> None:
    config = LayerValueConfig(layer_name="L", value=value)
    presence, parser = make_layer_operations(config, DOT)

    assert config.value is value
    assert isinstance(presence, expected_presence)
    assert isinstance(parser, expected_parser)
    assert not hasattr(presence, "kind")
    assert not hasattr(presence, "layer_name")
    assert not hasattr(parser, "kind")


@pytest.mark.parametrize(
    "value",
    [
        PlainNumericLayerDeclaration(missing_values=(1000.0,), type="integer"),
        RegexNumericLayerDeclaration(
            missing_values=(1000.0,), pattern=r"value=(\S+)", type="integer"
        ),
    ],
)
def test_both_numeric_operations_share_the_resolved_notation(
    value: PlainNumericLayerDeclaration | RegexNumericLayerDeclaration,
) -> None:
    numbers = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))

    presence, parser = make_layer_operations(LayerValueConfig("Count", value), numbers)

    assert isinstance(presence, PlainNumericRawValuePresence | RegexNumericRawValuePresence)
    assert isinstance(parser, PlainNumericLayerParser | RegexNumericLayerParser)
    assert presence.number_format is parser.number_format
    assert parser.number_format == NumberNotation(decimal_mark=",", thousands_marks=(".",))
    assert presence.missing_values == parser.missing_values == (1000.0,)
    assert parser.numeric_type == "integer"
    assert parser.layer_name == "Count"
    if isinstance(value, RegexNumericLayerDeclaration):
        assert isinstance(presence, RegexNumericRawValuePresence)
        assert isinstance(parser, RegexNumericLayerParser)
        assert presence.pattern == parser.pattern == value.pattern


def test_every_modification_declaration_names_one_normalizer() -> None:
    site_list = SiteListModificationConfig(
        kind="site_list",
        delimiter=";",
        site_base=1,
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(),
    )
    token_regex = TokenRegexModificationConfig(
        kind="token_regex",
        token_pattern=r"\(([^()]*)\)",
        token_position="after_residue",
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(),
    )

    from_site_list = make_sequence_normalizer(site_list)
    from_token_regex = make_sequence_normalizer(token_regex)

    assert isinstance(from_site_list, SiteListNormalizer)
    assert isinstance(from_token_regex, TokenRegexNormalizer)
    assert not hasattr(from_site_list, "sources")
    assert not hasattr(from_token_regex, "sources")
    assert not hasattr(from_site_list.rules, "kind")
    assert not hasattr(from_token_regex.rules, "kind")


def test_every_separation_declaration_names_one_separator() -> None:
    positional = make_fragment_table_separator(
        PositionalFragmentSeparationConfig(
            kind="positional",
            label_output="fragment_label",
            delimiter=";",
            packed_value_sources=("Quant",),
        )
    )
    labelled = make_fragment_table_separator(
        ColumnLabeledFragmentSeparationConfig(
            kind="column",
            label_source="Info",
            label_output="fragment_label",
            delimiter=";",
            packed_value_sources=("Quant",),
        )
    )

    assert isinstance(positional, PositionalFragmentTableSeparator)
    assert isinstance(labelled, ColumnLabeledFragmentTableSeparator)
    assert not hasattr(positional, "kind")
    assert not hasattr(labelled, "kind")


def test_every_physical_shape_names_one_decomposer() -> None:
    long_config = LongDecompositionConfig(
        kind="long",
        primary_layer_name="Intensity",
        layer_sources=(LongRawLayerSource(name="Intensity", source_column="intensity"),),
    )
    wide_config = WideDecompositionConfig(
        kind="wide",
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity",
                sources=(WideRawLayerSource(source_column="A", sample="A"),),
            ),
        ),
    )
    fragment_config = DelimitedFragmentDecompositionConfig(
        kind="delimited_fragment",
        separator=PositionalFragmentSeparationConfig(
            kind="positional",
            label_output="fragment_label",
            delimiter=";",
            packed_value_sources=("Quant",),
        ),
        long=long_config,
    )

    from_long = make_source_decomposer(long_config, AXIS, AXIS)
    from_wide = make_source_decomposer(wide_config, AXIS, AXIS)
    from_fragment = make_source_decomposer(fragment_config, AXIS, AXIS)

    assert isinstance(from_long, LongSourceDecomposer)
    assert isinstance(from_wide, WideSourceDecomposer)
    assert isinstance(from_fragment, DelimitedFragmentSourceDecomposer)
    assert isinstance(from_fragment.long_decomposer, LongSourceDecomposer)
    for decomposer in (from_long, from_wide, from_fragment):
        assert not hasattr(decomposer, "kind")
        assert not hasattr(decomposer, "config")


def test_checks_configure_a_separate_layer_set_validator() -> None:
    config = LayerContractConfig(
        primary_layer_name="Quantity",
        required_names=("Quantity",),
        empty_ratio=0.001,
        populated_ratio=0.5,
    )

    standard = make_layer_validator(config, "standard")
    strict = make_layer_validator(config, "strict")

    assert isinstance(standard, LayerContractValidator)
    assert isinstance(strict, LayerContractValidator)
    assert standard.strict is False
    assert strict.strict is True


# --------------------------------------------------------------------------- one compilation


def written(tmp_path: Path, header: tuple[str, ...], *rows: tuple[str, ...]) -> Path:
    path = tmp_path / "report.tsv"
    lines = ["\t".join(header), *("\t".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_one_level_compiles_into_a_complete_parser(tmp_path: Path) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    path = written(tmp_path, ("Sample", "Feature", "Quantity"), ("A", "F1", "1.5"))
    facade = synthetic.facade(document)

    parser = compile_level(facade, SingleFile(path=path), "standard")
    parsed = parser.parse()

    assert isinstance(parser, Parser)
    assert parser.level == "ion"
    assert parsed.obs.frame.to_dicts() == [{"sample": "A"}]
    assert parsed.layers["Quantity"].values.get_column("obs_0").to_list() == [1.5]


def test_compilation_injects_the_detected_number_notation_into_axis_coercers(
    tmp_path: Path,
) -> None:
    document = make_rule_document(
        tmp_path / "rules.json",
        {
            "schema_version": SCHEMA_VERSION,
            "file_version": "1",
            "software_name": "Localized",
            "software_version_pattern": "^1$",
            "tables": [
                {
                    "input": {
                        "shape": "long",
                        "extensions": [".tsv"],
                        "numbers": {
                            "mode": "detect",
                            "decimal_candidates": [".", ","],
                            "thousands_candidates": [",", ".", " "],
                        },
                    },
                    "base": {
                        "axis": {"obs_keys": ["sample"], "var_keys": ["Feature"]},
                        "columns": {
                            "obs": [{"name": "sample", "source": "Sample"}],
                            "var": [
                                {"name": "Feature", "source": "Feature"},
                                {"name": "Score", "source": "Score", "type": "number"},
                            ],
                        },
                        "measurements": {
                            "primary_layer": "Quantity",
                            "duplicates": {"mode": "error"},
                            "layers": [{"name": "Quantity", "source": "Quantity"}],
                        },
                    },
                    "levels": {"ion": {}},
                }
            ],
        },
    )
    path = written(
        tmp_path,
        ("Sample", "Feature", "Score", "Quantity"),
        ("A", "F1", "23,451117", "10,5"),
    )

    parser = compile_level(synthetic.facade(document), SingleFile(path=path), "standard")
    parsed = parser.parse()

    assert parsed.var.frame.get_column("Score").to_list() == [23.451117]


def test_compilation_resolves_the_source_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original = ParseRuleFacade.resolve_source

    def counting(self: ParseRuleFacade, evidence: object) -> object:
        calls.append("resolve_source")
        return original(self, evidence)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(ParseRuleFacade, "resolve_source", counting)
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    path = written(tmp_path, ("Sample", "Feature", "Quantity"), ("A", "F1", "1.5"))

    compile_level(synthetic.facade(document), SingleFile(path=path), "standard")

    assert calls == ["resolve_source"]


def test_a_compiled_parser_holds_no_registry_and_no_output_declaration(
    tmp_path: Path,
) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    path = written(tmp_path, ("Sample", "Feature", "Quantity"), ("A", "F1", "1.5"))

    parser = compile_level(synthetic.facade(document), SingleFile(path=path), "standard")

    held = {name: getattr(parser, name) for name in Parser.__slots__}
    assert not any(isinstance(value, dict) and "kind" in value for value in held.values())
    assert not hasattr(parser, "_output")
    assert not hasattr(parser, "_registry")
    assert not hasattr(parser, "_facade")


# ------------------------------------------------------------------------- several levels


def test_several_levels_return_one_collection_in_canonical_order() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    parser = ExplicitRuleCompiler(
        document,
        source,
        ("protein", "ion"),
        synthetic.NO_EVIDENCE,
        checks="standard",
    ).compile()
    parsed = parser.parse()

    assert list(parsed.levels) == ["ion", "protein"]
    assert LEVELS.index("ion") < LEVELS.index("protein")


def test_each_detection_selection_is_compiled_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    compiled: list[str] = []
    from apb2.parserV2 import compile as compilation

    original_compile = compilation.compile_level

    def record_compile(
        facade: ParseRuleFacade,
        selected_source: InputSource,
        checks: Literal["standard", "strict"],
    ) -> Parser:
        compiled.append(facade.working_parameters.level)
        return original_compile(facade, selected_source, checks)

    monkeypatch.setattr(compilation, "compile_level", record_compile)

    ExplicitRuleCompiler(
        document,
        source,
        ("ion", "protein"),
        synthetic.NO_EVIDENCE,
        checks="standard",
    ).compile()

    assert compiled == ["ion", "protein"]


def test_collection_parser_has_no_persistence_api() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    parser = ExplicitRuleCompiler(
        document,
        source,
        ("ion", "protein"),
        synthetic.NO_EVIDENCE,
        checks="standard",
    ).compile()

    assert not hasattr(parser, "convert")
    assert parser.parse().uns == {}


def test_an_incompatible_level_does_not_poison_the_compatible_ones() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    parser = ExplicitRuleCompiler(
        document,
        source,
        document.levels,
        synthetic.NO_EVIDENCE,
        checks="standard",
    ).compile()

    # The cached export carries no fragment columns; the other two levels still compile.
    assert list(parser.parse().levels) == ["ion", "protein"]


def test_an_empty_selection_fails_explicitly() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    with pytest.raises(ValueError, match="at least one quantification level"):
        ExplicitRuleCompiler(
            load_rule_document(pair.parser_v2_path),
            SingleFile(path=pair.required_data_path()),
            (),
            synthetic.NO_EVIDENCE,
            checks="standard",
        )


def test_a_gated_level_is_skipped_without_evidence_that_admits_it() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    combined = dataclasses.replace(synthetic.NO_EVIDENCE, combine_charge_states=True)
    parser = ExplicitRuleCompiler(
        document,
        source,
        document.levels,
        combined,
        checks="standard",
    ).compile()

    assert list(parser.parse().levels) == ["peptidoform"]


def test_duplicate_selections_fail_before_compilation() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    document = load_rule_document(pair.parser_v2_path)
    source = SingleFile(path=pair.required_data_path())
    with pytest.raises(ValueError, match="duplicate quantification levels"):
        ExplicitRuleCompiler(
            document,
            source,
            ("ion", "ion"),
            synthetic.NO_EVIDENCE,
            checks="standard",
        )


# ---------------------------------------------------------------------- the header predicate


def test_the_header_predicate_asks_only_for_what_the_level_cannot_do_without() -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature"},
        var_optional={"Extra": "Extra"},
        layers=[
            {"name": "Quantity", "source": "Quantity"},
            {"name": "Score", "source": "Score"},
        ],
    )
    working = synthetic.facade(document).working_parameters

    assert working.accepts_header(("Sample", "Feature", "Quantity"))
    assert working.accepts_header(("Sample", "Feature", "Quantity", "Extra"))
    assert not working.accepts_header(("Sample", "Quantity"))
    assert not working.accepts_header(("Sample", "Feature"))


def test_a_wide_level_asks_whether_anything_matches_its_layer_pattern() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[{"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"}],
        primary_layer="Intensity",
    )
    working = synthetic.facade(document).working_parameters

    assert working.accepts_header(("Feature", "A Intensity"))
    assert not working.accepts_header(("Feature", "A Count"))


@pytest.mark.parametrize(
    ("pair", "level"),
    [pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()],
)
def test_every_packaged_level_accepts_a_header_built_from_its_own_requirements(
    pair: PackagedDocument, level: str
) -> None:
    facade = pair.first_admitted_facade(level)  # pyright: ignore[reportArgumentType]
    working = facade.working_parameters
    header = pair.header()
    if not header:
        pytest.skip(f"no cached export for {pair.key}")

    # Whether this particular export satisfies the level is source resolution's answer; the
    # predicate must at least agree with it about the required columns being present.
    exact = {
        selection.source
        for axis in (working.obs, working.var)
        for selection in axis.columns.required_selections
    }
    accepts = working.accepts_header(header)
    assert accepts == (exact <= set(header) and accepts)
