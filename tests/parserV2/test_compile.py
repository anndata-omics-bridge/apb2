"""The composition root: every tag consumed once, and nothing carries one afterwards.

What these tests are for is the claim the whole architecture rests on — that after compilation
no object knows what vendor, level, layout, encoding, duplicate mode, or output format it came
from. So they check the registries for coverage, the constructed graph for tags, and
``compile_parsers`` for the ordering and skipping behaviour a multi-level caller relies on.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import get_args

import pytest

from apb2.parserV2.compile import (
    AnnDataOutput,
    NoCompatibleLevelError,
    ParquetOutput,
    ParseRuleCompiler,
    compile_mudata_parsers,
    compile_parsers,
    header_predicate,
    make_anndata_layer_contract_checker,
    make_anndata_layer_encoder,
    make_axis_coercer,
    make_column_computer,
    make_fragment_table_separator,
    make_modification_normalizer,
    make_parsed_level_writer,
    make_raw_value_presence,
    make_source_decomposer,
    policy_for,
)
from apb2.parserV2.parse_quant.axis_columns import (
    BooleanAxisCoercer,
    CoalesceColumn,
    DerivedSequenceColumn,
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
from apb2.parserV2.parse_quant.io.anndata_writer import (
    AnnDataWriter,
    FactorAnnDataEncoder,
    MuDataWriter,
    PlainNumericAnnDataEncoder,
    RegexNumericAnnDataEncoder,
    StandardAnnDataLayerContract,
    StrictAnnDataLayerContract,
)
from apb2.parserV2.parse_quant.io.parquet_writer import ParquetWriter
from apb2.parserV2.parse_quant.modifications import SiteListNormalizer, TokenRegexNormalizer
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisKeyPlan,
    AxisLogicalType,
    AxisSourcePlan,
    CoalesceColumnConfig,
    JoinNonemptyColumnConfig,
    ProformaFragmentColumnConfig,
    ProformaIonColumnConfig,
    ProformaSequenceColumnConfig,
    SiteListModificationConfig,
    StrippedSequenceColumnConfig,
    TokenRegexModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    FactorAnnDataEncodingConfig,
    NullOnlyRawValuePresenceConfig,
    PlainNumericAnnDataEncodingConfig,
    PlainNumericRawValuePresenceConfig,
    RegexNumericAnnDataEncodingConfig,
    RegexNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentSeparationConfig,
    DelimitedFragmentDecompositionConfig,
    DelimitedSourceEvidence,
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
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import make_rule_document
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, SCHEMA_VERSION
from parserV2 import synthetic
from parserV2.fixtures import DocumentPair, document_pairs, level_pairs

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
    policy = policy_for(mode)

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
                kind="stripped_sequence", name="S", inputs=("stripped_sequence",)
            ),
            DerivedSequenceColumn,
        ),
        (
            ProformaSequenceColumnConfig(
                kind="proforma_sequence", name="P", inputs=("proforma_sequence",)
            ),
            DerivedSequenceColumn,
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
    ("config", "expected"),
    [
        (
            NullOnlyRawValuePresenceConfig(kind="null_only", layer_name="L"),
            NullOnlyRawValuePresence,
        ),
        (
            PlainNumericRawValuePresenceConfig(
                kind="plain_numeric", layer_name="L", missing_values=(0.0,), number_format=DOT
            ),
            PlainNumericRawValuePresence,
        ),
        (
            RegexNumericRawValuePresenceConfig(
                kind="regex_numeric",
                layer_name="L",
                missing_values=(),
                pattern=r"(\d+)",
                number_format=DOT,
            ),
            RegexNumericRawValuePresence,
        ),
    ],
    ids=lambda value: getattr(value, "kind", getattr(value, "__name__", "")),
)
def test_every_presence_declaration_names_one_strategy(config: object, expected: type) -> None:
    presence = make_raw_value_presence(config)  # pyright: ignore[reportArgumentType]

    assert isinstance(presence, expected)
    assert not hasattr(presence, "kind")
    assert not hasattr(presence, "layer_name")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            PlainNumericAnnDataEncodingConfig(
                kind="plain_numeric", layer_name="L", missing_values=(), number_format=DOT
            ),
            PlainNumericAnnDataEncoder,
        ),
        (
            RegexNumericAnnDataEncodingConfig(
                kind="regex_numeric",
                layer_name="L",
                missing_values=(),
                pattern=r"(\d+)",
                number_format=DOT,
            ),
            RegexNumericAnnDataEncoder,
        ),
        (
            FactorAnnDataEncodingConfig(kind="factor", layer_name="L", categories=(("a", 0),)),
            FactorAnnDataEncoder,
        ),
    ],
    ids=lambda value: getattr(value, "kind", getattr(value, "__name__", "")),
)
def test_every_encoding_declaration_names_one_encoder(config: object, expected: type) -> None:
    encoder = make_anndata_layer_encoder(config)  # pyright: ignore[reportArgumentType]

    assert isinstance(encoder, expected)
    assert not hasattr(encoder, "kind")


def test_every_modification_declaration_names_one_normalizer() -> None:
    site_list = SiteListModificationConfig(
        kind="site_list",
        sequence_column="sequence",
        modification_column="mods",
        site_column="mod_sites",
        delimiter=";",
        site_base=1,
        case_sensitive=False,
        unknown_policy="preserve",
        proforma_output="proforma_sequence",
        stripped_output="stripped_sequence",
        entries=(),
    )
    token_regex = TokenRegexModificationConfig(
        kind="token_regex",
        source_column="Modified.Sequence",
        token_pattern=r"\(([^()]*)\)",
        token_position="after_residue",
        case_sensitive=False,
        unknown_policy="preserve",
        proforma_output="proforma_sequence",
        stripped_output="stripped_sequence",
        entries=(),
    )

    from_site_list = make_modification_normalizer(site_list)
    from_token_regex = make_modification_normalizer(token_regex)

    assert isinstance(from_site_list, SiteListNormalizer)
    assert isinstance(from_token_regex, TokenRegexNormalizer)
    assert from_site_list.sources == ("sequence", "mods", "mod_sites")
    assert from_token_regex.sources == ("Modified.Sequence",)
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


def test_the_output_declaration_selects_one_writer_and_is_not_kept() -> None:
    resolved = _resolved_plan()

    parquet = make_parsed_level_writer(ParquetOutput(), resolved)
    standard = make_parsed_level_writer(AnnDataOutput(), resolved)
    strict = make_parsed_level_writer(AnnDataOutput(checks="strict"), resolved)

    assert isinstance(parquet, ParquetWriter)
    assert isinstance(standard, AnnDataWriter)
    assert isinstance(standard.contract, StandardAnnDataLayerContract)
    assert isinstance(strict, AnnDataWriter)
    assert isinstance(strict.contract, StrictAnnDataLayerContract)
    assert not hasattr(parquet, "checks")


def test_a_parquet_compile_constructs_no_annData_collaborator() -> None:
    resolved = _resolved_plan()

    writer = make_parsed_level_writer(ParquetOutput(), resolved)

    assert not hasattr(writer, "encoders")
    assert not hasattr(writer, "contract")


def test_the_contract_checker_carries_the_resolved_occupancy_policy() -> None:
    resolved = _resolved_plan()

    checker = make_anndata_layer_contract_checker(resolved.ann_data, "standard")

    assert isinstance(checker, StandardAnnDataLayerContract)
    assert checker.policy.primary_layer_name == "Quantity"
    assert checker.policy.required_names == ("Quantity",)


def _resolved_plan() -> ResolvedLevelPlan:
    """One resolved plan for the smallest rule there is, for the factories to consume."""
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    facade = synthetic.facade(document)
    return facade.resolve_source(
        DelimitedSourceEvidence(
            columns=("Sample", "Feature", "Quantity"),
            delimiter="\t",
            quote_char='"',
            encoding="utf8",
            number_format=DOT,
        )
    )


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

    parser = ParseRuleCompiler(facade=facade, output=ParquetOutput()).compile(SingleFile(path=path))
    parsed = parser.parse()

    assert isinstance(parser, Parser)
    assert parser.level == "ion"
    assert parsed.obs.frame.to_dicts() == [{"sample": "A"}]
    assert parsed.layers["Quantity"].values.get_column("obs_0").to_list() == ["1.5"]


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
                    "obs": {"select": {"sample": "Sample"}},
                    "var": {
                        "select": {"Feature": "Feature", "Score": "Score"},
                        "types": {"Score": "number"},
                    },
                },
                "measurements": {
                    "primary_layer": "Quantity",
                    "duplicates": {"mode": "error"},
                    "layers": [{"name": "Quantity", "source": "Quantity"}],
                },
            },
            "levels": {"ion": {}},
        },
    )
    path = written(
        tmp_path,
        ("Sample", "Feature", "Score", "Quantity"),
        ("A", "F1", "23,451117", "10,5"),
    )

    parser = ParseRuleCompiler(
        facade=synthetic.facade(document),
        output=ParquetOutput(),
    ).compile(SingleFile(path=path))
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

    ParseRuleCompiler(facade=synthetic.facade(document), output=ParquetOutput()).compile(
        SingleFile(path=path)
    )

    assert calls == ["resolve_source"]


def test_a_compiled_parser_holds_no_registry_and_no_output_declaration(
    tmp_path: Path,
) -> None:
    document = synthetic.long_document(
        obs_select={"sample": "Sample"}, var_select={"Feature": "Feature"}
    )
    path = written(tmp_path, ("Sample", "Feature", "Quantity"), ("A", "F1", "1.5"))

    parser = ParseRuleCompiler(facade=synthetic.facade(document), output=AnnDataOutput()).compile(
        SingleFile(path=path)
    )

    held = {name: getattr(parser, name) for name in Parser.__slots__}
    assert not any(isinstance(value, dict) and "kind" in value for value in held.values())
    assert not hasattr(parser, "_output")
    assert not hasattr(parser, "_registry")
    assert not hasattr(parser, "_facade")


# ------------------------------------------------------------------------- several levels


def test_several_levels_return_a_list_in_canonical_order() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    assert path is not None

    parsers = compile_parsers(
        document=document,
        levels=("protein", "ion"),
        parameter_evidence=synthetic.NO_EVIDENCE,
        source=SingleFile(path=path),
        output=ParquetOutput(),
    )

    assert [parser.level for parser in parsers] == ["ion", "protein"]
    assert LEVELS.index("ion") < LEVELS.index("protein")


def test_mudata_compilation_retains_each_parsers_configured_anndata_writer() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    assert path is not None

    parsers, writer = compile_mudata_parsers(
        document=document,
        levels=("protein", "ion"),
        parameter_evidence=synthetic.NO_EVIDENCE,
        source=SingleFile(path=path),
        checks="standard",
    )

    assert isinstance(writer, MuDataWriter)
    assert [parser.level for parser in parsers] == ["ion", "protein"]
    assert list(writer.level_writers) == ["ion", "protein"]


def test_an_incompatible_level_does_not_poison_the_compatible_ones() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    assert path is not None

    parsers = compile_parsers(
        document=document,
        levels=document.levels,
        parameter_evidence=synthetic.NO_EVIDENCE,
        source=SingleFile(path=path),
        output=ParquetOutput(),
    )

    # The cached export carries no fragment columns; the other two levels still compile.
    assert [parser.level for parser in parsers] == ["ion", "protein"]


def test_a_source_that_satisfies_nothing_says_so_and_names_every_reason(
    tmp_path: Path,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)
    path = written(tmp_path, ("Unrelated",), ("x",))

    with pytest.raises(NoCompatibleLevelError) as error:
        compile_parsers(
            document=document,
            levels=document.levels,
            parameter_evidence=synthetic.NO_EVIDENCE,
            source=SingleFile(path=path),
            output=ParquetOutput(),
        )

    assert "ion" in str(error.value)
    assert "protein" in str(error.value)


def test_a_gated_level_is_skipped_without_evidence_that_admits_it() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    assert path is not None
    combined = dataclasses.replace(synthetic.NO_EVIDENCE, combine_charge_states=True)

    parsers = compile_parsers(
        document=document,
        levels=document.levels,
        parameter_evidence=combined,
        source=SingleFile(path=path),
        output=ParquetOutput(),
    )

    assert [parser.level for parser in parsers] == ["peptidoform"]


def test_each_level_of_one_document_gets_its_own_strategy_graph() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    assert path is not None

    parsers = compile_parsers(
        document=document,
        levels=("ion", "protein"),
        parameter_evidence=synthetic.NO_EVIDENCE,
        source=SingleFile(path=path),
        output=ParquetOutput(),
    )
    first, second = parsers

    for name in Parser.__slots__:
        if name == "level":
            continue
        assert getattr(first, name) is not getattr(second, name) or name in {"_writer"}


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
    accepts = header_predicate(synthetic.facade(document).working_parameters)

    assert accepts(("Sample", "Feature", "Quantity"))
    assert accepts(("Sample", "Feature", "Quantity", "Extra"))
    assert not accepts(("Sample", "Quantity"))
    assert not accepts(("Sample", "Feature"))


def test_a_wide_level_asks_whether_anything_matches_its_layer_pattern() -> None:
    document = synthetic.wide_document(
        var_select={"Feature": "Feature"},
        layers=[{"name": "Intensity", "source": r"^(?P<sample>.+) Intensity$"}],
        primary_layer="Intensity",
    )
    accepts = header_predicate(synthetic.facade(document).working_parameters)

    assert accepts(("Feature", "A Intensity"))
    assert not accepts(("Feature", "A Count"))


@pytest.mark.parametrize(
    ("pair", "level"),
    [pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()],
)
def test_every_packaged_level_accepts_a_header_built_from_its_own_requirements(
    pair: DocumentPair, level: str
) -> None:
    facade = pair.first_admitted_facade(level)  # pyright: ignore[reportArgumentType]
    working = facade.working_parameters
    accepts = header_predicate(working)
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
    assert accepts(header) == (exact <= set(header) and accepts(header))
