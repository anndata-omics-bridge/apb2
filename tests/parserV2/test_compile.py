"""The composition root: each tag selects behavior once; settings retain provenance.

What these tests are for is the claim the whole architecture rests on — that after compilation
no algorithm dispatches on vendor, level, layout, encoding, duplicate mode, or output format.
So they check registry coverage, runtime behavior, canonical settings identity, and
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
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
)
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    PlainSequenceStripper,
    SequenceColumn,
    SiteListNormalizer,
    TokenRegexNormalizer,
)
from apb2.parserV2.parse_quant.operations import (
    duplicate_policy_for,
    make_axis_coercer,
    make_layer_parser,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisKeyPlan,
    AxisLogicalType,
    AxisSourcePlan,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    FactorLayerDeclaration,
    LayerValueDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedSourceEvidence,
    InputSource,
    NumericTextFormat,
    SingleFile,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.value_parsing import (
    FactorLayerParser,
    PlainNumericLayerParser,
    RegexNumericLayerParser,
)
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.parser_factory import compile_level
from apb2.parserV2.vendor_parse_rules.document import make_rule_document
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.axis import (
    Coalesce,
    ComputedColumn,
    JoinNonempty,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
)
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, SCHEMA_VERSION
from parserV2 import synthetic
from parserV2.fixtures import PackagedDocument, document_pairs, level_pairs

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
DOT_NUMBERS = NumericTextFormat(decimal_mark=".", thousands_marks=())
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
        (Coalesce(how="coalesce", name="C", inputs=["a", "b"]), CoalesceColumn),
        (
            JoinNonempty(how="join_nonempty", name="J", inputs=["a", "b"], separator=","),
            JoinNonemptyColumn,
        ),
        (
            StrippedSequence(
                how="stripped_sequence",
                inputs=["Sequence"],
                syntax="plain",
            ),
            PlainSequenceStripper,
        ),
        (
            ProformaSequence(
                how="proforma_sequence",
                inputs=["Modified_Sequence"],
                syntax="tokens",
                modification_map="basic",
            ),
            SequenceColumn,
        ),
        (
            ProformaIon(how="proforma_ion", inputs=["P", "Z"]),
            ProformaIonColumn,
        ),
        (
            ProformaFragment(how="proforma_fragment", inputs=["I", "L"]),
            ProformaFragmentColumn,
        ),
    ],
    ids=lambda value: getattr(value, "how", getattr(value, "__name__", "")),
)
def test_every_computed_column_declaration_names_one_computer(
    config: ComputedColumn, expected: type
) -> None:
    document = synthetic.document(
        shape="long",
        base={
            "axis": {"obs_keys": ["sample"], "var_keys": [config.name]},
            "columns": {
                "obs": [{"name": "sample", "source": "Run"}],
                "var": [
                    {"name": name, "source": name, "type": "integer" if name == "Z" else "string"}
                    for name in config.inputs
                ]
                + [config.model_dump(mode="json")],
            },
            "measurements": {
                "primary_layer": "Quantity",
                "layers": [{"name": "Quantity", "source": "Quantity"}],
            },
            "sequence_syntax": {
                "plain": {"parser": "plain_sequence"},
                "tokens": {"parser": "token_regex", "token_pattern": r"\(([^()]*)\)"},
            },
            "modification_maps": {"basic": [{"token": "ox", "accession": "UNIMOD:35"}]},
        },
        levels={"fragment": {}},
    )
    facade = ParseRuleFacade(document, "fragment", synthetic.NO_EVIDENCE)
    computer = facade.working_parameters.var.computed[0]

    assert isinstance(computer, expected)
    assert computer.inputs == tuple(config.inputs)
    assert computer.name == config.name
    assert not hasattr(computer, "kind")
    strategy = facade.resolve_source(
        DelimitedSourceEvidence(("Run", "Quantity", *config.inputs), "\t", '"', "utf8", DOT)
    )
    assert strategy.var.key_phase.computers[0] is computer
    assert (
        synthetic.plan_snapshot(strategy)["var"]["key_phase"]["computers"][0]["kind"] == config.how
    )


@pytest.mark.parametrize(
    ("value", "expected_parser"),
    [
        (
            PlainNumericLayerDeclaration(missing_values=()),
            PlainNumericLayerParser,
        ),
        (
            PlainNumericLayerDeclaration(missing_values=(0.0,)),
            PlainNumericLayerParser,
        ),
        (
            RegexNumericLayerDeclaration(missing_values=(0.0,), pattern=r"(\d+)"),
            RegexNumericLayerParser,
        ),
        (
            RegexNumericLayerDeclaration(missing_values=(), pattern=r"(\d+)"),
            RegexNumericLayerParser,
        ),
        (
            FactorLayerDeclaration(categories=(("a", 0),)),
            FactorLayerParser,
        ),
    ],
    ids=lambda value: getattr(value, "kind", getattr(value, "__name__", "")),
)
def test_one_declaration_selects_one_layer_parser(
    value: LayerValueDeclaration, expected_parser: type
) -> None:
    parser = make_layer_parser("L", value, DOT)

    assert isinstance(parser, expected_parser)
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
def test_numeric_parser_retains_the_resolved_notation(
    value: PlainNumericLayerDeclaration | RegexNumericLayerDeclaration,
) -> None:
    numbers = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))

    parser = make_layer_parser("Count", value, numbers)

    assert isinstance(parser, PlainNumericLayerParser | RegexNumericLayerParser)
    assert parser.number_format is numbers
    assert parser.missing_values == (1000.0,)
    assert parser.numeric_type == "integer"
    assert parser.layer_name == "Count"
    if isinstance(value, RegexNumericLayerDeclaration):
        assert isinstance(parser, RegexNumericLayerParser)
        assert parser.pattern == value.pattern


def test_normalizers_own_their_settings_without_configuration_wrappers() -> None:
    site_list = SiteListNormalizer(
        delimiter=";",
        site_base=1,
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(),
    )
    token_regex = TokenRegexNormalizer(
        token_pattern=r"\(([^()]*)\)",
        token_position="after_residue",
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(),
    )
    embedded = EmbeddedSiteListNormalizer(
        delimiter=";",
        entry_pattern=r"^(?P<token>.+?)\s+\((?P<site>[^)]+)\)$",
        site_base=1,
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(),
    )

    for normalizer in (site_list, token_regex, embedded):
        assert not hasattr(normalizer, "rules")
        assert not hasattr(normalizer, "sources")
        assert not hasattr(normalizer, "kind")


@pytest.mark.parametrize("label_strategy", ["positional", "column"])
def test_source_resolution_constructs_the_separator_and_long_decomposer(
    label_strategy: str,
) -> None:
    document = synthetic.document(
        shape="long",
        base={
            "axis": {"obs_keys": ["sample"], "var_keys": ["Feature"]},
            "columns": {
                "obs": [{"name": "sample", "source": "Sample"}],
                "var": [{"name": "Feature", "source": "Feature"}],
            },
            "measurements": {
                "primary_layer": "Quantity",
                "layers": [{"name": "Quantity", "source": "Quantity"}],
            },
        },
        levels={
            "fragment": {
                "fragments": {
                    "label_strategy": label_strategy,
                    "value_columns": ["Quantity"],
                    **({"label_column": "Info"} if label_strategy == "column" else {}),
                }
            }
        },
    )
    strategy = synthetic.facade(document, "fragment").resolve_source(
        DelimitedSourceEvidence(("Sample", "Feature", "Quantity", "Info"), "\t", '"', "utf8", DOT)
    )
    decomposer = strategy.decomposer
    assert isinstance(decomposer, DelimitedFragmentSourceDecomposer)
    assert isinstance(decomposer.long_decomposer, LongSourceDecomposer)
    expected = {
        "positional": PositionalFragmentTableSeparator,
        "column": ColumnLabeledFragmentTableSeparator,
    }[label_strategy]
    assert isinstance(decomposer.separator, expected)
    assert decomposer.separator.packed_value_sources == ("Quantity",)
    assert not hasattr(decomposer, "config")
    assert not hasattr(decomposer.separator, "kind")


@pytest.mark.parametrize("shape", ["long", "wide"])
def test_source_resolution_constructs_the_physical_decomposer(shape: str) -> None:
    if shape == "wide":
        document = synthetic.wide_document(
            var_select={"Feature": "Feature"},
            layers=[{"name": "Quantity", "source": "^(?P<sample>A)$"}],
            primary_layer="Quantity",
        )
        header = ("Feature", "A")
        expected = WideSourceDecomposer
    else:
        document = synthetic.long_document(
            obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
        )
        header = ("Sample", "Feature", "Quantity")
        expected = LongSourceDecomposer
    strategy = synthetic.facade(document).resolve_source(
        DelimitedSourceEvidence(header, "\t", '"', "utf8", DOT)
    )
    assert isinstance(strategy.decomposer, expected)
    assert not hasattr(strategy.decomposer, "kind")
    assert not hasattr(strategy.decomposer, "config")


def test_checks_configure_a_separate_layer_set_validator() -> None:
    facade = synthetic.facade(
        synthetic.long_document(obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"})
    )
    evidence = DelimitedSourceEvidence(("Sample", "Feature", "Quantity"), "\t", '"', "utf8", DOT)
    standard = facade.resolve_source(evidence, checks="standard").layer_validator
    strict = facade.resolve_source(evidence, checks="strict").layer_validator
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

    def counting(
        self: ParseRuleFacade,
        evidence: object,
        *,
        checks: Literal["standard", "strict"] = "standard",
    ) -> object:
        calls.append("resolve_source")
        return original(self, evidence, checks=checks)  # pyright: ignore[reportArgumentType]

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
    from apb2.parserV2 import detect_document as compilation

    original_compile = compilation.compile_level

    def record_compile(
        facade: ParseRuleFacade,
        selected_source: InputSource,
        checks: Literal["standard", "strict"],
    ) -> Parser:
        compiled.append(facade.working_parameters.level)
        return original_compile(facade, selected_source, checks)

    monkeypatch.setattr(compilation, "compile_level", record_compile)

    compiler = ExplicitRuleCompiler(
        document,
        source,
        ("ion", "protein"),
        synthetic.NO_EVIDENCE,
        checks="standard",
    )
    compiler.compile()
    compiler.compile()

    assert compiled.count("ion") == compiled.count("protein") == 1


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
        for selection in axis.required_selections
    }
    accepts = working.accepts_header(header)
    assert accepts == (exact <= set(header) and accepts)
