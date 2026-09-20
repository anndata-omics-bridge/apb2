"""Axis leaf algorithms: one coercion, one computed column, one normalized sequence.

These run on the small axis frames, so the interesting properties are per-value: what a
declared logical type accepts, what it refuses and how loudly, and that the modification
algorithm produces exactly what the unchanged implementation produced on real vendor data.
"""

from __future__ import annotations

import polars as pl
import pytest

from apb2.parserV2.parse_quant.axis_columns import (
    AxisCoercionError,
    BooleanAxisCoercer,
    CoalesceColumn,
    ColumnComputationError,
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
)
from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    ModificationOccurrence,
    PackedSiteMismatchError,
    PlainSequenceStripper,
    SequenceColumn,
    SequenceOperation,
    SiteListNormalizer,
    TokenRegexNormalizer,
    TokenRegexStripper,
    UnknownModificationError,
    normalize_embedded_site_list,
    normalize_site_list,
    normalize_token_regex,
    render_proforma,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    EmbeddedSiteListModificationConfig,
    ModificationMapEntry,
    ModificationTokenPosition,
    SiteListModificationConfig,
    TokenRegexModificationConfig,
    UnknownModificationPolicy,
)

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

DOT_NUMBERS = NumberNotation(decimal_mark=".", thousands_marks=())
COMMA_NUMBERS = NumberNotation(decimal_mark=",", thousands_marks=(".", " "))


def token_regex(
    *,
    pattern: str = r"\(([^()]*)\)",
    position: ModificationTokenPosition = "after_residue",
    policy: UnknownModificationPolicy = "preserve",
    entries: tuple[ModificationMapEntry, ...] = (OXIDATION, ACETYL),
) -> TokenRegexModificationConfig:
    """The same settings consumed by the compiler and sequence algorithm."""
    return TokenRegexModificationConfig(
        kind="token_regex",
        token_pattern=pattern,
        token_position=position,
        case_sensitive=False,
        unknown_policy=policy,
        entries=entries,
    )


def site_list(
    *,
    site_base: int = 1,
    policy: UnknownModificationPolicy = "preserve",
) -> SiteListModificationConfig:
    return SiteListModificationConfig(
        kind="site_list",
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


def embedded_site_list() -> EmbeddedSiteListModificationConfig:
    return EmbeddedSiteListModificationConfig(
        kind="embedded_site_list",
        delimiter=";",
        entry_pattern=r"^(?P<token>.+?)\s+\((?P<site>[^)]+)\)$",
        site_base=1,
        case_sensitive=False,
        unknown_policy="preserve",
        entries=(
            ModificationMapEntry(
                token="Oxidation",
                name="Oxidation",
                accession="UNIMOD:35",
                target=("M",),
                position="Anywhere",
                mass_delta=15.994915,
            ),
            ModificationMapEntry(
                token="Acetyl",
                name="Acetyl",
                accession="UNIMOD:1",
                target=("N-term",),
                position="N-term",
                mass_delta=42.010565,
            ),
        ),
    )


# ----------------------------------------------------------------------------- coercion


def coerce(coercer: AxisValueCoercer, values: pl.Series, *, name: str, source: str) -> pl.Series:
    frame = values.rename(source).to_frame()
    return frame.select(coercer.coerce(frame, name=name, source=source)).to_series()


def test_string_coercion_keeps_the_exact_vendor_token() -> None:
    values = pl.Series("Charge", ["01", "1", None, ""])

    coerced = coerce(StringAxisCoercer(), values, name="Charge", source="charge")

    assert coerced.to_list() == ["01", "1", None, ""]
    assert coerced.len() == values.len()


def test_integer_coercion_reads_integers_and_keeps_nulls_missing() -> None:
    values = pl.Series("Charge", ["2", "03", None])

    coerced = coerce(IntegerAxisCoercer(DOT_NUMBERS), values, name="Charge", source="charge")

    assert coerced.to_list() == [2, 3, None]
    assert coerced.dtype == pl.Int64


@pytest.mark.parametrize("token", ["2.5", "abc", "1e400", ""])
def test_integer_coercion_names_the_column_the_source_and_the_tokens(token: str) -> None:
    values = pl.Series("Charge", [token, "2"])

    with pytest.raises(AxisCoercionError) as error:
        coerce(IntegerAxisCoercer(DOT_NUMBERS), values, name="Charge", source="charge")

    assert "Charge" in str(error.value)
    assert "charge" in str(error.value)
    assert repr(token) in str(error.value) or token in str(error.value)


def test_number_coercion_rejects_non_finite_and_unreadable_tokens() -> None:
    assert coerce(
        NumberAxisCoercer(DOT_NUMBERS), pl.Series("Mass", ["1.5", None]), name="Mass", source="mass"
    ).to_list() == [1.5, None]
    for token in ("nan", "inf", "abc"):
        with pytest.raises(AxisCoercionError):
            coerce(
                NumberAxisCoercer(DOT_NUMBERS),
                pl.Series("Mass", [token]),
                name="Mass",
                source="mass",
            )


def test_numeric_axis_coercers_honor_the_resolved_number_notation() -> None:
    numbers = coerce(
        NumberAxisCoercer(COMMA_NUMBERS),
        pl.Series("Score", ["23,451117", "100.000.000,5", "1 234,25"]),
        name="Score",
        source="score",
    )
    integers = coerce(
        IntegerAxisCoercer(COMMA_NUMBERS),
        pl.Series("Charge", ["2,0", "1.000,0"]),
        name="Charge",
        source="charge",
    )

    assert numbers.to_list() == [23.451117, 100_000_000.5, 1234.25]
    assert integers.to_list() == [2, 1000]


def test_boolean_coercion_accepts_only_the_canonical_spellings() -> None:
    values = pl.Series("Decoy", ["True", "false", "1", "0.0", None])

    coerced = coerce(BooleanAxisCoercer(), values, name="Decoy", source="decoy")

    assert coerced.to_list() == [True, False, True, False, None]
    with pytest.raises(AxisCoercionError, match="Decoy"):
        coerce(BooleanAxisCoercer(), pl.Series("Decoy", ["yes"]), name="Decoy", source="decoy")


def test_a_native_boolean_column_survives_its_declared_type() -> None:
    values = pl.Series("Decoy", [True, False, None])

    assert coerce(BooleanAxisCoercer(), values, name="Decoy", source="decoy").to_list() == [
        True,
        False,
        None,
    ]


def test_a_coercion_error_lists_at_most_five_distinct_examples() -> None:
    values = pl.Series("Charge", [f"bad{index % 8}" for index in range(40)])

    with pytest.raises(AxisCoercionError) as error:
        coerce(IntegerAxisCoercer(DOT_NUMBERS), values, name="Charge", source="charge")

    assert str(error.value).count("bad") == 5
    assert "40 invalid" in str(error.value)


def test_every_coercer_satisfies_the_parser_owned_contract() -> None:
    coercers: tuple[AxisValueCoercer, ...] = (
        StringAxisCoercer(),
        IntegerAxisCoercer(DOT_NUMBERS),
        NumberAxisCoercer(DOT_NUMBERS),
        BooleanAxisCoercer(),
    )

    for coercer in coercers:
        result = coerce(coercer, pl.Series("x", [None, None]), name="x", source="x")
        assert result.len() == 2


# --------------------------------------------------------------------- computed columns


def test_coalesce_takes_the_first_non_null_in_declaration_order() -> None:
    first = pl.Series("a", ["p", None, None])
    second = pl.Series("b", ["q", "q", None])

    computed, tokens = CoalesceColumn(name="Merged", inputs=("a", "b")).compute(
        pl.DataFrame([first, second])
    )

    assert computed["Merged"].to_list() == ["p", "q", None]
    assert tokens == ()


def test_join_nonempty_skips_nulls_and_empty_strings_alike() -> None:
    first = pl.Series("a", ["p", "", None, ""])
    second = pl.Series("b", ["q", "q", "q", None])

    computed, _ = JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=",").compute(
        pl.DataFrame([first, second])
    )

    assert computed["J"].to_list() == ["p,q", "q", "q", None]


def test_a_stripping_column_consumes_the_sequence_directly() -> None:
    sequence = pl.Series("Modified_Sequence", ["PEPM(ox)IDE"])
    computer = SequenceColumn(
        name="ProForma_peptide",
        inputs=("Modified_Sequence",),
        operation=TokenRegexStripper(r"\(([^()]*)\)", "after_residue"),
    )
    result, tokens = computer.compute(sequence.to_frame())
    assert result[computer.name].to_list() == ["PEPMIDE"]
    assert tokens == ()


def test_a_proforma_ion_needs_a_present_positive_charge() -> None:
    sequences = pl.Series("s", ["PEPTIDE", "OTHER"])
    charges = pl.Series("c", [2, 3], dtype=pl.Int64)

    computed, _ = ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
        pl.DataFrame([sequences, charges])
    )

    assert computed["ProForma_ion"].to_list() == ["PEPTIDE/2", "OTHER/3"]
    with pytest.raises(ColumnComputationError, match="missing charge"):
        ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
            pl.DataFrame([sequences, pl.Series("c", [2, None], dtype=pl.Int64)])
        )
    with pytest.raises(ColumnComputationError, match="positive"):
        ProformaIonColumn(name="ProForma_ion", inputs=("s", "c")).compute(
            pl.DataFrame([sequences, pl.Series("c", [2, 0], dtype=pl.Int64)])
        )


def test_a_proforma_fragment_joins_an_ion_and_a_label() -> None:
    ions = pl.Series("i", ["PEPTIDE/2", None])
    labels = pl.Series("l", ["frag_0", "frag_1"])

    computed, _ = ProformaFragmentColumn(name="ProForma_fragment", inputs=("i", "l")).compute(
        pl.DataFrame([ions, labels])
    )

    # A missing ion leaves a missing fragment key, which axis preparation then drops; the
    # legacy implementation rendered the string "nan/frag_1" instead.
    assert computed["ProForma_fragment"].to_list() == ["PEPTIDE/2/frag_0", None]


@pytest.mark.parametrize(
    "computer",
    [
        CoalesceColumn(name="C", inputs=("a", "b")),
        JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=","),
        PlainSequenceStripper(name="D", inputs=("a",)),
        ProformaIonColumn(name="I", inputs=("a", "b")),
        ProformaFragmentColumn(name="F", inputs=("a", "b")),
    ],
    ids=lambda computer: type(computer).__name__,
)
def test_a_computer_requires_its_named_inputs(
    computer: ColumnComputer,
) -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        computer.compute(pl.DataFrame({"unrelated": ["x"]}))


@pytest.mark.parametrize(
    "computer",
    [
        CoalesceColumn(name="C", inputs=("a", "b")),
        JoinNonemptyColumn(name="J", inputs=("a", "b"), separator=","),
        PlainSequenceStripper(name="D", inputs=("a",)),
    ],
    ids=lambda computer: type(computer).__name__,
)
def test_a_computer_preserves_its_input_length_and_row_order(
    computer: ColumnComputer,
) -> None:
    height = 6
    columns = tuple(
        pl.Series(name, [name + "A" * index for index in range(height)]) for name in computer.inputs
    )

    result, _ = computer.compute(pl.DataFrame(columns))

    assert result.height == height
    assert result[computer.name][0] != result[computer.name][1]


# ------------------------------------------------------------------------ modifications


@pytest.mark.parametrize("sequences", [[], [None, None]])
def test_sequence_mapping_handles_empty_and_all_null_frames(sequences: list[str | None]) -> None:
    frame = pl.DataFrame({"Sequence": sequences}, schema={"Sequence": pl.String})
    computer = SequenceColumn(
        "Peptide", ("Sequence",), TokenRegexStripper(r"\(([^()]*)\)", "after_residue")
    )
    result, tokens = computer.compute(frame)
    assert result["Peptide"].to_list() == [""] * len(sequences)
    assert result.schema["Peptide"] == pl.String
    assert tokens == ()


def test_sequence_mapping_preserves_other_columns_and_first_seen_diagnostics() -> None:
    frame = pl.DataFrame(
        {"Sequence": ["M(second)", "M(first)", "M(second)", "M(third)"], "_result": [4, 3, 2, 1]}
    )
    computer = SequenceColumn("Sequence", ("Sequence",), TokenRegexNormalizer(token_regex()))
    result, tokens = computer.compute(frame)
    assert result.to_dict(as_series=False) == {
        "Sequence": ["M-[second]", "M-[first]", "M-[second]", "M-[third]"],
        "_result": [4, 3, 2, 1],
    }
    assert tokens == ("second", "first", "third")


def test_site_mapping_distinguishes_all_inputs_and_preserves_order() -> None:
    frame = pl.DataFrame(
        {
            "Sequence": ["MM", "MM", "MM", "MM"],
            "Mods": ["Oxidation@M"] * 4,
            "Sites": ["2", "1", "2", "1"],
        }
    )
    computer = SequenceColumn("P", ("Sequence", "Mods", "Sites"), SiteListNormalizer(site_list()))
    result, tokens = computer.compute(frame)
    assert result["P"].to_list() == [
        "MM[UNIMOD:35]",
        "M[UNIMOD:35]M",
        "MM[UNIMOD:35]",
        "M[UNIMOD:35]M",
    ]
    assert result.select(frame.columns).equals(frame)
    assert tokens == ()


def test_plain_stripping_keeps_unicode_letters_and_handles_empty_frames() -> None:
    computer = PlainSequenceStripper("Sequence", ("Sequence",))
    frame = pl.DataFrame({"Sequence": ["_Aaβ中1²\u2160ⓐ\u0301-", None, ""]})
    result, tokens = computer.compute(frame)
    assert result["Sequence"].to_list() == ["Aaβ中", "", ""]
    assert tokens == ()
    assert computer.compute(frame.clear())[0].equals(frame.clear())


def test_an_inline_token_becomes_a_localized_proforma_modification() -> None:
    result = normalize_token_regex("PEPM(ox)IDE", token_regex())

    assert result.stripped_sequence == "PEPMIDE"
    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


def test_a_terminal_token_renders_before_the_sequence() -> None:
    result = normalize_token_regex("_(ac)PEPTIDE_", token_regex())

    assert result.stripped_sequence == "PEPTIDE"
    assert result.proforma_sequence == "[UNIMOD:1]-PEPTIDE"


def test_a_before_residue_vendor_attaches_the_token_to_what_follows() -> None:
    rules = token_regex(
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
    rules = token_regex(pattern=r"\[([^\]]+)\]")

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
    rules = token_regex(policy=policy)

    result = normalize_token_regex("PEPM(weird)IDE", rules)

    assert result.proforma_sequence == expected
    assert result.unknown_tokens == unknown_tokens


def test_an_unknown_token_can_be_declared_an_error() -> None:
    with pytest.raises(UnknownModificationError, match="weird"):
        normalize_token_regex("PEPM(weird)IDE", token_regex(policy="error"))


def test_parallel_site_lists_are_paired_index_wise() -> None:
    result = normalize_site_list("PEPMIDE", "Oxidation@M", "4", site_list())

    assert result.stripped_sequence == "PEPMIDE"
    assert result.proforma_sequence == "PEPM[UNIMOD:35]IDE"


def test_a_preserved_unknown_site_list_token_is_returned_for_reporting() -> None:
    result = normalize_site_list("PEPMIDE", "Mystery@M", "4", site_list())

    assert result.proforma_sequence == "PEPM[Mystery@M]IDE"
    assert result.unknown_tokens == ("Mystery@M",)


def test_site_zero_is_the_n_terminus_whatever_the_site_base_is() -> None:
    for base in (0, 1):
        result = normalize_site_list("PEPMIDE", "Oxidation@M", "0", site_list(site_base=base))
        assert result.proforma_sequence.startswith("[UNIMOD:35]-")


def test_a_site_list_of_mismatched_length_is_a_vendor_file_defect() -> None:
    with pytest.raises(PackedSiteMismatchError, match="length mismatch"):
        normalize_site_list("PEPMIDE", "Oxidation@M;Oxidation@M", "4", site_list())
    with pytest.raises(PackedSiteMismatchError, match="non-integer"):
        normalize_site_list("PEPMIDE", "Oxidation@M", "x", site_list())


def test_an_empty_modification_list_leaves_the_bare_sequence() -> None:
    result = normalize_site_list("PEPMIDE", "", "", site_list())

    assert result.proforma_sequence == "PEPMIDE"


def test_embedded_sites_localize_residue_and_terminal_modifications() -> None:
    result = normalize_embedded_site_list(
        "PEPMIDE", "Acetyl (Protein N-term); Oxidation (M4)", embedded_site_list()
    )

    assert result.proforma_sequence == "[UNIMOD:1]-PEPM[UNIMOD:35]IDE"


def test_an_embedded_site_must_point_to_the_declared_residue() -> None:
    with pytest.raises(PackedSiteMismatchError, match="points to"):
        normalize_embedded_site_list("PEPMIDE", "Oxidation (M3)", embedded_site_list())


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


def test_normalization_returns_one_column_and_explicit_diagnostics_in_row_order() -> None:
    computer = SequenceColumn(
        name="ProForma_peptidoform",
        inputs=("Modified_Sequence",),
        operation=TokenRegexNormalizer(token_regex()),
    )
    sequences = pl.Series("vendor sequence", ["PEPM(ox)IDE", "PEPM(weird)IDE", "PEPM(ox)IDE", None])
    result, tokens = computer.compute(sequences.rename("Modified_Sequence").to_frame())
    assert result[computer.name].to_list() == [
        "PEPM[UNIMOD:35]IDE",
        "PEPM[weird]IDE",
        "PEPM[UNIMOD:35]IDE",
        "",
    ]
    assert tokens == ("weird",)
    assert result.height == sequences.len()


def test_a_site_list_normalizer_consumes_its_three_inputs_in_order() -> None:
    normalizer = SiteListNormalizer(site_list())
    assert isinstance(normalizer, SiteListNormalizer)
    computer = SequenceColumn("ProForma_peptidoform", ("Sequence", "Mods", "Sites"), normalizer)
    result, _ = computer.compute(
        pl.DataFrame({"Sequence": ["PEPMIDE"], "Mods": ["Oxidation@M"], "Sites": ["4"]})
    )
    assert result[computer.name].to_list() == ["PEPM[UNIMOD:35]IDE"]


def test_an_embedded_site_normalizer_consumes_its_two_inputs_in_order() -> None:
    normalizer = EmbeddedSiteListNormalizer(embedded_site_list())
    assert isinstance(normalizer, EmbeddedSiteListNormalizer)
    result = normalizer.transform(("PEPMIDE", "Oxidation (M4)"))
    assert result.value == "PEPM[UNIMOD:35]IDE"


def test_all_normalizers_satisfy_the_sequence_column_owned_contract() -> None:
    normalizers: tuple[SequenceOperation, ...] = (
        TokenRegexNormalizer(token_regex()),
        SiteListNormalizer(site_list()),
        EmbeddedSiteListNormalizer(embedded_site_list()),
    )
    assert isinstance(normalizers[0], TokenRegexNormalizer)
    for normalizer, row in zip(
        normalizers, [("PEPMIDE",), ("PEPMIDE", "", ""), ("PEPMIDE", "")], strict=True
    ):
        assert normalizer.transform(row).value == "PEPMIDE"
