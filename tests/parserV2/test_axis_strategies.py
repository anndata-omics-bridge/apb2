"""Axis leaf algorithms: one coercion, one computed column, one normalized sequence.

These run on the small axis frames, so the interesting properties are per-value: what a
declared logical type accepts, what it refuses and how loudly, and that the modification
algorithm produces exactly what the unchanged implementation produced on real vendor data.
"""

from __future__ import annotations

import polars as pl
import pytest
from anndata_proteomics.modifications import apply_rules as oracle_normalize
from anndata_proteomics.modifications.model import ModifiedSequence as OracleModifiedSequence

from apb2.parserV2 import compile as composition
from apb2.parserV2.compile import make_modification_normalizer
from apb2.parserV2.parse_quant.axis_columns import (
    AxisCoercionError,
    BooleanAxisCoercer,
    CoalesceColumn,
    ColumnComputationError,
    DerivedSequenceColumn,
    IntegerAxisCoercer,
    JoinNonemptyColumn,
    NumberAxisCoercer,
    ProformaFragmentColumn,
    ProformaIonColumn,
    StringAxisCoercer,
)
from apb2.parserV2.parse_quant.contracts import (
    AxisValueCoercer,
    ColumnComputer,
    ModificationNormalizer,
)
from apb2.parserV2.parse_quant.modifications import (
    ModificationOccurrence,
    PackedSiteMismatchError,
    SiteListNormalizer,
    SiteListRules,
    TokenRegexNormalizer,
    TokenRegexRules,
    UnknownModificationError,
    normalize_site_list,
    normalize_token_regex,
    render_proforma,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    ModificationMapEntry,
    ModificationTokenPosition,
    SiteListModificationConfig,
    TokenRegexModificationConfig,
    UnknownModificationPolicy,
)
from apb2.parserV2.parse_quant.parameters.source import LevelReadPlan, SingleFile
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from parserV2.fixtures import DocumentPair, document_pairs

OXIDATION = ModificationMapEntry(
    token="ox",
    name="Oxidation",
    accession="UNIMOD:35",
    target=("M",),
    position="Anywhere",
    mass_delta=15.994915,
)
# The canonical registry record: Acetyl sits on the N-terminus, so both its target and
# its position say so. A token matches only when both agree with where it was found.
ACETYL = ModificationMapEntry(
    token="ac",
    name="Acetyl",
    accession="UNIMOD:1",
    target=("N-term",),
    position="N-term",
    mass_delta=42.010565,
)


def token_rules(
    *,
    pattern: str = r"\(([^()]*)\)",
    position: ModificationTokenPosition = "after_residue",
    policy: UnknownModificationPolicy = "preserve",
    entries: tuple[ModificationMapEntry, ...] = (OXIDATION, ACETYL),
) -> TokenRegexRules:
    """The rules the pure algorithm reads: how tokens are written and what they mean."""
    return TokenRegexRules(
        token_pattern=pattern,
        token_position=position,
        case_sensitive=False,
        unknown_policy=policy,
        entries=entries,
    )


def site_rules(
    *, site_base: int = 1, policy: UnknownModificationPolicy = "preserve"
) -> SiteListRules:
    return SiteListRules(
        delimiter=";",
        site_base=site_base,
        case_sensitive=False,
        unknown_policy=policy,
        entries=(
            ModificationMapEntry(
                token="Oxidation@M",
                name="Oxidation",
                accession="UNIMOD:35",
                target=("M",),
                position="Anywhere",
                mass_delta=15.994915,
            ),
        ),
    )


def token_regex(
    *,
    pattern: str = r"\(([^()]*)\)",
    position: ModificationTokenPosition = "after_residue",
    policy: UnknownModificationPolicy = "preserve",
    entries: tuple[ModificationMapEntry, ...] = (OXIDATION, ACETYL),
) -> TokenRegexModificationConfig:
    """The declaration the compiler reads, from which those rules are built."""
    return TokenRegexModificationConfig(
        kind="token_regex",
        source_column="Modified.Sequence",
        token_pattern=pattern,
        token_position=position,
        case_sensitive=False,
        unknown_policy=policy,
        proforma_output="proforma_sequence",
        stripped_output="stripped_sequence",
        entries=entries,
    )


def site_list(
    *,
    site_base: int = 1,
    policy: UnknownModificationPolicy = "preserve",
) -> SiteListModificationConfig:
    return SiteListModificationConfig(
        kind="site_list",
        sequence_column="sequence",
        modification_column="mods",
        site_column="mod_sites",
        delimiter=";",
        site_base=site_base,
        case_sensitive=False,
        unknown_policy=policy,
        proforma_output="proforma_sequence",
        stripped_output="stripped_sequence",
        entries=(
            ModificationMapEntry(
                token="Oxidation@M",
                name="Oxidation",
                accession="UNIMOD:35",
                target=("M",),
                position="Anywhere",
                mass_delta=15.994915,
            ),
        ),
    )


# ----------------------------------------------------------------------------- coercion


def test_string_coercion_keeps_the_exact_vendor_token() -> None:
    values = pl.Series("Charge", ["01", "1", None, ""])

    coerced = StringAxisCoercer().coerce(values, name="Charge", source="charge")

    assert coerced.to_list() == ["01", "1", None, ""]
    assert coerced.len() == values.len()


def test_integer_coercion_reads_integers_and_keeps_nulls_missing() -> None:
    values = pl.Series("Charge", ["2", "03", None])

    coerced = IntegerAxisCoercer().coerce(values, name="Charge", source="charge")

    assert coerced.to_list() == [2, 3, None]
    assert coerced.dtype == pl.Int64


@pytest.mark.parametrize("token", ["2.5", "abc", "1e400", ""])
def test_integer_coercion_names_the_column_the_source_and_the_tokens(token: str) -> None:
    values = pl.Series("Charge", [token, "2"])

    with pytest.raises(AxisCoercionError) as error:
        IntegerAxisCoercer().coerce(values, name="Charge", source="charge")

    assert "Charge" in str(error.value)
    assert "charge" in str(error.value)
    assert repr(token) in str(error.value) or token in str(error.value)


def test_number_coercion_rejects_non_finite_and_unreadable_tokens() -> None:
    assert NumberAxisCoercer().coerce(
        pl.Series("Mass", ["1.5", None]), name="Mass", source="mass"
    ).to_list() == [1.5, None]
    for token in ("nan", "inf", "abc"):
        with pytest.raises(AxisCoercionError):
            NumberAxisCoercer().coerce(pl.Series("Mass", [token]), name="Mass", source="mass")


def test_boolean_coercion_accepts_only_the_canonical_spellings() -> None:
    values = pl.Series("Decoy", ["True", "false", "1", "0.0", None])

    coerced = BooleanAxisCoercer().coerce(values, name="Decoy", source="decoy")

    assert coerced.to_list() == [True, False, True, False, None]
    with pytest.raises(AxisCoercionError, match="Decoy"):
        BooleanAxisCoercer().coerce(pl.Series("Decoy", ["yes"]), name="Decoy", source="decoy")


def test_a_native_boolean_column_survives_its_declared_type() -> None:
    values = pl.Series("Decoy", [True, False, None])

    assert BooleanAxisCoercer().coerce(values, name="Decoy", source="decoy").to_list() == [
        True,
        False,
        None,
    ]


def test_a_coercion_error_lists_at_most_five_distinct_examples() -> None:
    values = pl.Series("Charge", [f"bad{index % 8}" for index in range(40)])

    with pytest.raises(AxisCoercionError) as error:
        IntegerAxisCoercer().coerce(values, name="Charge", source="charge")

    assert str(error.value).count("bad") == 5
    assert "40 invalid" in str(error.value)


def test_every_coercer_satisfies_the_parser_owned_contract() -> None:
    coercers: tuple[AxisValueCoercer, ...] = (
        StringAxisCoercer(),
        IntegerAxisCoercer(),
        NumberAxisCoercer(),
        BooleanAxisCoercer(),
    )

    for coercer in coercers:
        result = coercer.coerce(pl.Series("x", [None, None]), name="x", source="x")
        assert result.len() == 2


# --------------------------------------------------------------------- computed columns


def test_coalesce_takes_the_first_non_null_in_declaration_order() -> None:
    first = pl.Series("a", ["p", None, None])
    second = pl.Series("b", ["q", "q", None])

    computed = CoalesceColumn(name="Merged", inputs=("a", "b")).compute((first, second))

    assert computed.to_list() == ["p", "q", None]


def test_join_nonempty_skips_nulls_and_empty_strings_alike() -> None:
    first = pl.Series("a", ["p", "", None, ""])
    second = pl.Series("b", ["q", "q", "q", None])

    computed = JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=",").compute(
        (first, second)
    )

    assert computed.to_list() == ["p,q", "q", "q", None]


def test_a_derived_sequence_column_exposes_what_normalization_produced() -> None:
    derived = pl.Series("proforma_sequence", ["PEPM[UNIMOD:35]IDE"])

    computed = DerivedSequenceColumn(
        name="ProForma_peptidoform", inputs=("proforma_sequence",)
    ).compute((derived,))

    assert computed.to_list() == ["PEPM[UNIMOD:35]IDE"]


def test_a_proforma_ion_needs_a_present_positive_charge() -> None:
    sequences = pl.Series("s", ["PEPTIDE", "OTHER"])
    charges = pl.Series("c", [2, 3], dtype=pl.Int64)

    computed = ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
        (sequences, charges)
    )

    assert computed.to_list() == ["PEPTIDE/2", "OTHER/3"]
    with pytest.raises(ColumnComputationError, match="missing charge"):
        ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
            (sequences, pl.Series("c", [2, None], dtype=pl.Int64))
        )
    with pytest.raises(ColumnComputationError, match="positive"):
        ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
            (sequences, pl.Series("c", [2, 0], dtype=pl.Int64))
        )


def test_a_proforma_fragment_joins_an_ion_and_a_label() -> None:
    ions = pl.Series("i", ["PEPTIDE/2", None])
    labels = pl.Series("l", ["frag_0", "frag_1"])

    computed = ProformaFragmentColumn(name="ProForma_fragment", inputs=("i", "l")).compute(
        (ions, labels)
    )

    # A missing ion leaves a missing fragment key, which axis preparation then drops; the
    # legacy implementation rendered the string "nan/frag_1" instead.
    assert computed.to_list() == ["PEPTIDE/2/frag_0", None]


@pytest.mark.parametrize(
    "computer",
    [
        CoalesceColumn(name="C", inputs=("a", "b")),
        JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=","),
        DerivedSequenceColumn(name="D", inputs=("a",)),
        ProformaIonColumn(name="I", inputs=("a", "b")),
        ProformaFragmentColumn(name="F", inputs=("a", "b")),
    ],
    ids=lambda computer: type(computer).__name__,
)
def test_a_computer_refuses_a_series_tuple_that_is_not_its_inputs(
    computer: ColumnComputer,
) -> None:
    with pytest.raises(ColumnComputationError, match="declares inputs"):
        computer.compute((pl.Series("a", ["x"]), pl.Series("b", ["y"]), pl.Series("c", ["z"])))


@pytest.mark.parametrize(
    "computer",
    [
        CoalesceColumn(name="C", inputs=("a", "b")),
        JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=","),
        DerivedSequenceColumn(name="D", inputs=("a",)),
    ],
    ids=lambda computer: type(computer).__name__,
)
def test_a_computer_preserves_its_input_length_and_row_order(
    computer: ColumnComputer,
) -> None:
    height = 6
    columns = tuple(
        pl.Series(name, [f"{name}{index}" for index in range(height)]) for name in computer.inputs
    )

    result = computer.compute(columns)

    assert result.len() == height
    assert result.to_list()[0] != result.to_list()[1]


# ------------------------------------------------------------------------ modifications


def test_an_inline_token_becomes_a_localized_proforma_modification() -> None:
    result = normalize_token_regex("PEPM(ox)IDE", token_rules())

    assert result.stripped_sequence == "PEPMIDE"
    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


def test_a_terminal_token_renders_before_the_sequence() -> None:
    result = normalize_token_regex("_(ac)PEPTIDE_", token_rules())

    assert result.stripped_sequence == "PEPTIDE"
    assert result.proforma_sequence == "[UNIMOD:1]-PEPTIDE"


def test_a_before_residue_vendor_attaches_the_token_to_what_follows() -> None:
    rules = token_rules(
        pattern="[a-z]+",
        position="before_residue",
        entries=(
            ModificationMapEntry(
                token="ox",
                name="Oxidation",
                accession="UNIMOD:35",
                target=("M",),
                position="Anywhere",
                mass_delta=15.994915,
            ),
        ),
    )

    result = normalize_token_regex("PEPoxMIDE", rules)

    assert result.stripped_sequence == "PEPMIDE"
    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


def test_a_numeric_token_matches_on_mass_target_and_position() -> None:
    rules = token_rules(pattern=r"\[([^\]]+)\]")

    result = normalize_token_regex("PEPM[15.9949]IDE", rules)

    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


@pytest.mark.parametrize(
    ("policy", "expected", "unknown_tokens"),
    [
        ("preserve", "PEPM[weird]IDE", ("weird",)),
        ("drop", "PEPMIDE", ()),
    ],
)
def test_an_unknown_token_follows_the_declared_policy(
    policy: UnknownModificationPolicy,
    expected: str,
    unknown_tokens: tuple[str, ...],
) -> None:
    rules = token_rules(policy=policy)

    result = normalize_token_regex("PEPM(weird)IDE", rules)

    assert result.proforma_sequence == expected
    assert result.unknown_tokens == unknown_tokens


def test_an_unknown_token_can_be_declared_an_error() -> None:
    with pytest.raises(UnknownModificationError, match="weird"):
        normalize_token_regex("PEPM(weird)IDE", token_rules(policy="error"))


def test_parallel_site_lists_are_paired_index_wise() -> None:
    result = normalize_site_list("PEPMIDE", "Oxidation@M", "4", site_rules())

    assert result.stripped_sequence == "PEPMIDE"
    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


def test_a_preserved_unknown_site_list_token_is_returned_for_reporting() -> None:
    result = normalize_site_list("PEPMIDE", "Mystery@M", "4", site_rules())

    assert result.proforma_sequence == "PEPM[Mystery@M]IDE"
    assert result.unknown_tokens == ("Mystery@M",)


def test_site_zero_is_the_n_terminus_whatever_the_site_base_is() -> None:
    for base in (0, 1):
        result = normalize_site_list("PEPMIDE", "Oxidation@M", "0", site_rules(site_base=base))
        assert result.proforma_sequence.startswith("[UNIMOD:35]-")


def test_a_site_list_of_mismatched_length_is_a_vendor_file_defect() -> None:
    with pytest.raises(PackedSiteMismatchError, match="length mismatch"):
        normalize_site_list("PEPMIDE", "Oxidation@M;Oxidation@M", "4", site_rules())
    with pytest.raises(PackedSiteMismatchError, match="non-integer"):
        normalize_site_list("PEPMIDE", "Oxidation@M", "x", site_rules())


def test_an_empty_modification_list_leaves_the_bare_sequence() -> None:
    result = normalize_site_list("PEPMIDE", "", "", site_rules())

    assert result.proforma_sequence == "PEPMIDE"


def test_two_modifications_on_one_residue_concatenate() -> None:
    rendered = render_proforma(
        "PEPMIDE",
        (
            ModificationOccurrence(
                name="Oxidation",
                accession="UNIMOD:35",
                position="Anywhere",
                target_residue="M",
                sequence_index=3,
                source_token="ox",
            ),
            ModificationOccurrence(
                name="Acetyl",
                accession="UNIMOD:1",
                position="Anywhere",
                target_residue="M",
                sequence_index=3,
                source_token="ac",
            ),
        ),
        {},
    )

    assert rendered == "PEPM[UNIMOD:35][UNIMOD:1]IDE"


def test_a_normalizer_returns_its_declared_derived_columns_and_keeps_row_order() -> None:
    normalizer = make_modification_normalizer(token_regex())
    sequences = pl.Series(
        "Modified.Sequence",
        ["PEPM(ox)IDE", "PEPM(weird)IDE", "PEPM(ox)IDE", None],
    )

    derived = normalizer.normalize((sequences,))

    assert set(derived) == {
        "proforma_sequence",
        "stripped_sequence",
        "unknown_mod_tokens",
    }
    assert derived["proforma_sequence"].to_list() == [
        "PEPM[UNIMOD:35]IDE",
        "PEPM[weird]IDE",
        "PEPM[UNIMOD:35]IDE",
        "",
    ]
    assert derived["unknown_mod_tokens"].to_list() == [[], ["weird"], [], []]
    assert all(series.len() == sequences.len() for series in derived.values())


def test_a_site_list_normalizer_declares_its_three_sources_in_order() -> None:
    normalizer = make_modification_normalizer(site_list())

    assert isinstance(normalizer, SiteListNormalizer)
    assert normalizer.sources == ("sequence", "mods", "mod_sites")
    derived = normalizer.normalize(
        (
            pl.Series("sequence", ["PEPMIDE"]),
            pl.Series("mods", ["Oxidation@M"]),
            pl.Series("mod_sites", ["4"]),
        )
    )
    assert derived["proforma_sequence"].to_list() == ["PEPM[UNIMOD:35]IDE"]


def test_both_normalizers_satisfy_the_parser_owned_contract() -> None:
    normalizers: tuple[ModificationNormalizer, ...] = (
        make_modification_normalizer(token_regex()),
        make_modification_normalizer(site_list()),
    )

    assert isinstance(normalizers[0], TokenRegexNormalizer)
    for normalizer in normalizers:
        assert normalizer.sources


# ------------------------------------------------------- parity with the unchanged domain

_PARITY_ROWS = 2000
_PARITY_CASES = [pytest.param(pair, id=pair.key) for pair in document_pairs()]


@pytest.mark.parametrize("pair", _PARITY_CASES)
def test_normalization_matches_the_unchanged_implementation_on_real_sequences(
    pair: DocumentPair,
) -> None:
    document = load_rule_document(pair.parser_v2_path)
    level = document.levels[0]
    facade = pair.first_admitted_facade(level)
    configs = facade.working_parameters.modifications
    path = pair.data_path()
    if not configs or path is None:
        pytest.skip(f"{pair.key} declares no consumed modifications, or has no cached export")
    config = configs[0]
    normalizer = make_modification_normalizer(config)
    if set(normalizer.sources) - set(pair.header()):
        pytest.skip(f"cached export for {pair.key} lacks the modification sources")

    source = SingleFile(path=path)
    bound = composition.bind_source(source, facade.working_parameters.input)
    evidence = composition.source_evidence(source, bound, document.matches)
    frame = (
        composition.make_reader(
            bound,
            evidence,
            LevelReadPlan(
                projected_columns=normalizer.sources,
                text_sources=frozenset(normalizer.sources),
                native_numeric_sources=frozenset(),
            ),
        )
        .read()
        .frame.head(_PARITY_ROWS)
    )
    columns = tuple(frame.get_column(name) for name in normalizer.sources)
    derived = normalizer.normalize(columns)

    expected = _oracle_results(config, columns)
    assert derived[config.proforma_output].to_list() == [
        result.proforma_sequence for result in expected
    ]
    assert derived[config.stripped_output].to_list() == [
        result.stripped_sequence for result in expected
    ]


def _oracle_results(
    config: SiteListModificationConfig | TokenRegexModificationConfig,
    columns: tuple[pl.Series, ...],
) -> list[OracleModifiedSequence]:
    """Run the external APB normalizer over the same resolved configuration and values."""
    entries = tuple(
        oracle_normalize.MapEntry(
            token=entry.token,
            name=entry.name,
            accession=entry.accession,
            target=entry.target,
            position=entry.position,
            mass_delta=entry.mass_delta,
        )
        for entry in config.entries
    )
    texts = [
        [value if value is not None else "" for value in column.cast(pl.String).to_list()]
        for column in columns
    ]
    if isinstance(config, SiteListModificationConfig):
        settings = oracle_normalize.SiteListRule(
            delimiter=config.delimiter,
            site_base=config.site_base,
            case_sensitive=config.case_sensitive,
            unknown_policy=config.unknown_policy,
            entries=entries,
        )
        return [
            oracle_normalize.apply_site_list(sequence, mods, sites, settings)
            for sequence, mods, sites in zip(*texts, strict=True)
        ]
    token_settings = oracle_normalize.ModificationRule(
        token_pattern=config.token_pattern,
        token_position=config.token_position,
        case_sensitive=config.case_sensitive,
        unknown_policy=config.unknown_policy,
        entries=entries,
    )
    return [oracle_normalize.apply_rule(value, token_settings) for value in texts[0]]
