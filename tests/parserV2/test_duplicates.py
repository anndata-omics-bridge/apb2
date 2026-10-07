"""Presence and duplicate resolution: only what a raw cell claims, never what it means.

The distinction these tests protect: presence may inspect a token but never converts it, and
a policy may select or add scalars but never compares final keys. A token that cannot be read
stays present, so a later encoding failure is not silently resolved away here.
"""

from __future__ import annotations

from typing import get_args

import polars as pl
import pytest

from apb2.parserV2.parse_quant.contracts import DuplicatePolicy, RawValuePresence
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable
from apb2.parserV2.parse_quant.data.raw import RawLayerTable
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    AggregateTypeError,
    DuplicateCellError,
    ErrorOnDuplicates,
    KeepBestDuplicate,
    KeepFirstDuplicate,
)
from apb2.parserV2.parse_quant.operations import (
    NUMERIC_DUPLICATE_MODES,
    duplicate_policy,
    duplicate_policy_for,
    make_layer_parser,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    DuplicateRanking,
    FactorLayerDeclaration,
    LayerValueDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
    WorkingMeasurements,
)
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat
from apb2.parserV2.vendor_parse_rules.schema.measurements import Duplicates

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
GROUPED = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))

NULL_ONLY = make_layer_parser("L", PlainNumericLayerDeclaration(missing_values=()), DOT)
ZERO_SENTINEL = make_layer_parser("L", PlainNumericLayerDeclaration(missing_values=(0.0,)), DOT)
ASCORE = make_layer_parser(
    "L",
    RegexNumericLayerDeclaration(missing_values=(0.0,), pattern=r":(-?\d+(?:\.\d+)?)"),
    DOT,
)
NONPOSITIVE = make_layer_parser(
    "L", PlainNumericLayerDeclaration(missing_values=(), missing_at_or_below=0.0), DOT
)

# A PEP 695 alias holds its literal union in ``__value__``.
MODES: tuple[DuplicateMode, ...] = get_args(DuplicateMode.__value__)
# keep_best is built from a ranking layer, so it is tested on its own below.
STATELESS: tuple[DuplicateMode, ...] = tuple(mode for mode in MODES if mode != "keep_best")
SUM = duplicate_policy_for("sum")
MAX = duplicate_policy_for("max")


def layer(frame: pl.DataFrame, *, keys: tuple[str, ...] = ("Feature",)) -> RawLayerTable:
    return RawLayerTable(layer_name="L", raw_var_key_columns=keys, values=frame)


def presence_mask(strategy: RawValuePresence, values: pl.Series) -> pl.Series:
    """Evaluate one presence expression as the duplicate resolver does."""
    return (
        values.to_frame()
        .select(strategy.present(pl.col(values.name), values.dtype).alias(values.name))
        .to_series()
    )


# ------------------------------------------------------------------------------- presence


def test_null_only_presence_asks_nothing_of_the_value_itself() -> None:
    values = pl.Series("obs_0", [1.0, 0.0, None])

    mask = presence_mask(NULL_ONLY, values)

    assert mask.to_list() == [True, True, False]
    assert mask.dtype == pl.Boolean
    assert mask.null_count() == 0


@pytest.mark.parametrize(
    "value",
    [
        PlainNumericLayerDeclaration(missing_values=()),
        FactorLayerDeclaration(categories=(("", 0),)),
    ],
)
def test_no_sentinel_and_factor_declarations_keep_blank_text_present(
    value: LayerValueDeclaration,
) -> None:
    presence = make_layer_parser("L", value, DOT)

    assert presence_mask(presence, pl.Series("obs_0", ["", "  ", None])).to_list() == [
        True,
        True,
        False,
    ]


def test_a_declared_sentinel_claims_nothing_without_replacing_the_value() -> None:
    values = pl.Series("obs_0", [12.0, 0.0, None])

    assert presence_mask(ZERO_SENTINEL, values).to_list() == [True, False, False]
    # The strategy returned a mask; the value it was asked about is untouched.
    assert values.to_list() == [12.0, 0.0, None]


def test_an_at_or_below_bound_claims_neither_zero_nor_negative_values() -> None:
    values = pl.Series("obs_0", [12.0, 0.0, -3.5, None])

    assert presence_mask(NONPOSITIVE, values).to_list() == [True, False, False, False]


def test_an_at_or_below_bound_reads_the_number_inside_a_structured_token() -> None:
    parser = make_layer_parser(
        "L",
        RegexNumericLayerDeclaration(
            missing_values=(), pattern=r":(-?\d+(?:\.\d+)?)", missing_at_or_below=0.0
        ),
        DOT,
    )
    values = pl.Series("obs_0", ["a:4.5", "b:0", "c:-1"])

    assert presence_mask(parser, values).to_list() == [True, False, False]


def test_an_at_or_below_bound_leaves_only_positive_final_values() -> None:
    final = FinalLayerTable(
        layer_name="L",
        values=(pl.DataFrame({"Feature": ["F1", "F2", "F3"], "obs_0": [5.0, 0.0, -1.0]})).drop(
            ("Feature",), strict=False
        ),
        semantic_roles=("abundance",),
    )

    parsed = NONPOSITIVE.parse(final)

    assert parsed.values.get_column("obs_0").to_list() == [5.0, None, None]


def test_blank_text_is_the_written_spelling_of_a_missing_number() -> None:
    values = pl.Series("obs_0", ["12", "", "   ", None])

    assert presence_mask(ZERO_SENTINEL, values).to_list() == [True, False, False, False]


def test_a_nonblank_token_that_cannot_be_read_stays_present() -> None:
    values = pl.Series("obs_0", ["12", "not a number"])

    # Keep-first must not be able to hide this; the AnnData encoder is where it fails.
    assert presence_mask(ZERO_SENTINEL, values).to_list() == [True, True]


def test_a_localized_sentinel_is_recognized_under_its_own_notation() -> None:
    presence = make_layer_parser(
        "L",
        PlainNumericLayerDeclaration(missing_values=(0.0, 1000.0)),
        GROUPED,
    )
    values = pl.Series("obs_0", ["1.000", "1.000,5", "0", None])

    assert presence_mask(presence, values).to_list() == [False, True, False, False]


def test_regex_presence_reads_the_number_inside_a_structured_token() -> None:
    values = pl.Series("obs_0", ["site:1.5", "site:0", "", None, "unstructured"])

    assert presence_mask(ASCORE, values).to_list() == [True, False, False, False, True]


def test_every_presence_strategy_returns_a_mask_of_its_input_shape() -> None:
    strategies: tuple[RawValuePresence, ...] = (NULL_ONLY, ZERO_SENTINEL, ASCORE)
    values = pl.Series("obs_0", ["1", None, "2", ""])

    for strategy in strategies:
        mask = presence_mask(strategy, values)
        assert mask.len() == values.len()
        assert mask.dtype == pl.Boolean
        assert mask.null_count() == 0


# ------------------------------------------------------------------------------- policies


def test_two_claiming_values_in_one_cell_are_an_error_when_the_rule_says_so() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [1.0, 2.0]}))

    with pytest.raises(DuplicateCellError, match="more than once"):
        ErrorOnDuplicates().resolve(repeated, NULL_ONLY)


def test_one_claiming_value_beside_a_sentinel_is_not_a_duplicate() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [0.0, 2.0]}))

    resolved = ErrorOnDuplicates().resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": 2.0}]


def test_keep_first_selects_independently_in_every_observation_column() -> None:
    repeated = layer(
        pl.DataFrame(
            {
                "Feature": ["F1", "F1", "F2"],
                "obs_0": [None, 12.0, 5.0],
                "obs_1": [3.0, 4.0, None],
            }
        )
    )

    resolved = KeepFirstDuplicate().resolve(repeated, NULL_ONLY)

    assert resolved.values.to_dicts() == [
        {"Feature": "F1", "obs_0": 12.0, "obs_1": 3.0},
        {"Feature": "F2", "obs_0": 5.0, "obs_1": None},
    ]


def test_keep_first_skips_a_sentinel_and_keeps_the_real_value_unencoded() -> None:
    """AlphaDIA writes 0 for "not measured"; the value kept is the vendor's own scalar."""
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["0", "12.5"]}))

    resolved = KeepFirstDuplicate().resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": "12.5"}]
    assert resolved.values.schema["obs_0"] == pl.String


def test_an_unreadable_token_reaches_the_result_instead_of_being_resolved_away() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["broken", "12.5"]}))

    resolved = KeepFirstDuplicate().resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": "broken"}]


def test_an_unknown_factor_label_also_stays_present() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["surprise", "MBR"]}))

    with pytest.raises(DuplicateCellError):
        ErrorOnDuplicates().resolve(repeated, NULL_ONLY)


def test_numeric_aggregate_sums_only_the_claiming_values() -> None:
    repeated = layer(
        pl.DataFrame({"Feature": ["F1", "F1", "F1", "F2"], "obs_0": [1.0, 0.0, 2.0, 7.0]})
    )

    resolved = SUM.resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [
        {"Feature": "F1", "obs_0": 3.0},
        {"Feature": "F2", "obs_0": 7.0},
    ]


def test_numeric_aggregate_drops_nonpositive_values_before_summing() -> None:
    repeated = layer(
        pl.DataFrame({"Feature": ["F1", "F1", "F1", "F2"], "obs_0": [1.0, -2.0, 0.0, -7.0]})
    )

    resolved = SUM.resolve(repeated, NONPOSITIVE)

    assert resolved.values.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0},
        {"Feature": "F2", "obs_0": None},
    ]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
@pytest.mark.parametrize("missing", [0.0, None, float("nan")])
def test_a_cell_with_nothing_present_stays_null_instead_of_becoming_zero(
    policy: DuplicatePolicy,
    missing: float | None,
) -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [missing, None]}))

    resolved = policy.resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": None}]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_numeric_aggregate_refuses_values_that_are_not_numbers(policy: DuplicatePolicy) -> None:
    text = layer(pl.DataFrame({"Feature": ["F1"], "obs_0": ["MBR"]}))

    with pytest.raises(AggregateTypeError, match="needs numeric"):
        policy.resolve(text, NULL_ONLY)


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_numeric_aggregate_accepts_a_layer_that_resolved_to_no_values_at_all(
    policy: DuplicatePolicy,
) -> None:
    empty = layer(
        pl.DataFrame({"Feature": ["F1"], "obs_0": [None]}, schema_overrides={"obs_0": pl.Null})
    )

    resolved = policy.resolve(empty, NULL_ONLY)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": None}]


def test_numeric_max_never_selects_a_value_presence_masked() -> None:
    repeated = layer(
        pl.DataFrame({"Feature": ["F1", "F1", "F2", "F2"], "obs_0": [1.0, 3.0, -7.0, -1.0]})
    )

    resolved = MAX.resolve(repeated, NONPOSITIVE)

    assert resolved.values.to_dicts() == [
        {"Feature": "F1", "obs_0": 3.0},
        {"Feature": "F2", "obs_0": None},
    ]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_nan_is_absent_before_a_numeric_reduction(policy: DuplicatePolicy) -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [float("nan"), 1.0]}))

    resolved = policy.resolve(repeated, NULL_ONLY)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": 1.0}]


def test_numeric_max_stays_above_a_negative_threshold() -> None:
    """Values above -1 can sum to -1 or less; their max cannot, so parsing keeps it."""
    above_minus_one = make_layer_parser(
        "L", PlainNumericLayerDeclaration(missing_values=(), missing_at_or_below=-1.0), DOT
    )
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [-0.5, -0.8]}))

    resolved = MAX.resolve(repeated, above_minus_one)
    final = FinalLayerTable(
        layer_name="L", values=resolved.values.drop("Feature"), semantic_roles=("abundance",)
    )

    assert above_minus_one.parse(final).values.get_column("obs_0").to_list() == [-0.5]


def test_numeric_max_keeps_an_integer_layer_integer() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [3, 5]}))

    resolved = MAX.resolve(repeated, NULL_ONLY)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": 5}]
    assert resolved.values.schema["obs_0"] == pl.Int64


@pytest.mark.parametrize(
    "policy", [duplicate_policy_for(mode) for mode in STATELESS], ids=STATELESS
)
@pytest.mark.parametrize("empty", [False, True])
def test_every_policy_keeps_the_keys_the_group_order_and_the_layer_name(
    policy: DuplicatePolicy,
    empty: bool,
) -> None:
    values = layer(
        pl.DataFrame(
            {
                "Feature": ["F2", "F1", "F3"],
                "Charge": ["3", "2", "1"],
                "obs_0": [1.0, 2.0, 3.0],
            }
        ),
        keys=("Feature", "Charge"),
    )
    if empty:
        values.values = values.values.head(0)

    resolved = policy.resolve(values, NULL_ONLY)

    assert resolved.layer_name == "L"
    assert resolved.raw_var_key_columns == ("Feature", "Charge")
    assert resolved.values.columns == ["Feature", "Charge", "obs_0"]
    assert resolved.values.get_column("Feature").to_list() == ([] if empty else ["F2", "F1", "F3"])


@pytest.mark.parametrize(
    "policy", [duplicate_policy_for(mode) for mode in STATELESS], ids=STATELESS
)
def test_a_multi_column_raw_key_groups_as_one_identity(policy: DuplicatePolicy) -> None:
    values = layer(
        pl.DataFrame(
            {
                "Feature": ["F1", "F1", "F1"],
                "Charge": ["2", "3", "2"],
                "obs_0": [1.0, 2.0, None],
            }
        ),
        keys=("Feature", "Charge"),
    )

    resolved = policy.resolve(values, NULL_ONLY)

    assert resolved.values.height == 2


def test_grouping_treats_two_null_key_components_as_the_same_identity() -> None:
    values = layer(
        pl.DataFrame(
            {"Feature": ["F1", None, None], "obs_0": [1.0, 2.0, None]},
        )
    )

    resolved = KeepFirstDuplicate().resolve(values, NULL_ONLY)

    assert resolved.values.height == 2
    assert resolved.values.get_column("obs_0").to_list() == [1.0, 2.0]


def test_a_layer_with_no_observation_columns_resolves_to_its_keys() -> None:
    values = layer(pl.DataFrame({"Feature": ["F1", "F1"]}))

    resolved = KeepFirstDuplicate().resolve(values, NULL_ONLY)

    assert resolved.values.to_dicts() == [{"Feature": "F1"}]
    assert ErrorOnDuplicates().resolve(values, NULL_ONLY).values.height == 1
    assert SUM.resolve(values, NULL_ONLY).values.height == 1
    assert MAX.resolve(values, NULL_ONLY).values.height == 1


@pytest.mark.parametrize(
    "policy", [duplicate_policy_for(mode) for mode in STATELESS], ids=STATELESS
)
@pytest.mark.parametrize(
    "names",
    [
        ("Feature", "value", "present", "_present_0"),
        ("_duplicate", "_duplicate_", "_duplicate__", "value"),
    ],
)
def test_internal_names_do_not_collide_with_keys_or_measurement_columns(
    policy: DuplicatePolicy,
    names: tuple[str, str, str, str],
) -> None:
    key, first, second, third = names
    values = layer(
        pl.DataFrame(
            {
                key: ["F2", "F1", "F2"],
                first: [None, 2.5, 3.0],
                second: [4, None, None],
                third: [None, None, 8.0],
            }
        ),
        keys=(key,),
    )
    resolved = policy.resolve(values, NULL_ONLY).values
    assert resolved.schema == values.values.schema
    assert resolved.to_dicts() == [
        {key: "F2", first: 3.0, second: 4, third: 8.0},
        {key: "F1", first: 2.5, second: None, third: None},
    ]


def test_the_declared_mode_selects_one_stateless_policy() -> None:
    """The removed legacy mode is not a value this selector can be given."""
    assert isinstance(duplicate_policy_for("error"), ErrorOnDuplicates)
    assert isinstance(duplicate_policy_for("keep_first"), KeepFirstDuplicate)
    assert isinstance(duplicate_policy_for("sum"), AggregateNumericDuplicates)
    assert isinstance(duplicate_policy_for("max"), AggregateNumericDuplicates)
    assert set(MODES) == {"error", "keep_first", "sum", "max", "keep_best"}


def test_the_numeric_modes_are_exactly_those_whose_policy_reduces_numbers() -> None:
    reducing = {
        mode
        for mode in STATELESS
        if isinstance(duplicate_policy_for(mode), AggregateNumericDuplicates)
    }

    assert reducing == NUMERIC_DUPLICATE_MODES


def test_no_duplicate_policy_retains_its_discriminator() -> None:
    for value in (duplicate_policy_for(mode) for mode in STATELESS):
        assert not hasattr(value, "kind")
        assert not hasattr(value, "mode")
        assert not hasattr(value, "layer_name")


# ------------------------------------------------------------------------------- keep_best


def _ranked_pair() -> tuple[RawLayerTable, RawLayerTable]:
    """An intensity and a score layer from one pivot: F1 has two PSMs in obs_0."""
    intensity = RawLayerTable(
        layer_name="Intensity",
        raw_var_key_columns=("Feature",),
        values=pl.DataFrame(
            {
                "Feature": ["F1", "F1", "F2"],
                "obs_0": [100.0, 200.0, 10.0],
                "obs_1": [50.0, None, 20.0],
            }
        ),
    )
    score = RawLayerTable(
        layer_name="Score",
        raw_var_key_columns=("Feature",),
        values=pl.DataFrame(
            {
                "Feature": ["F1", "F1", "F2"],
                "obs_0": ["1.5", "7.0", "2.0"],
                "obs_1": ["3.0", None, "1.0"],
            }
        ),
    )
    return intensity, score


@pytest.mark.parametrize(("highest", "kept"), [(True, 200.0), (False, 100.0)])
def test_keep_best_takes_every_layer_from_the_best_ranked_row(highest: bool, kept: float) -> None:
    intensity, score = _ranked_pair()
    policy = KeepBestDuplicate(by="Score", highest=highest).ranked(score, NULL_ONLY)

    values = policy.resolve(intensity, NULL_ONLY).values.to_dicts()
    scores = policy.resolve(score, NULL_ONLY).values.to_dicts()

    assert values == [
        {"Feature": "F1", "obs_0": kept, "obs_1": 50.0},
        {"Feature": "F2", "obs_0": 10.0, "obs_1": 20.0},
    ]
    assert scores[0]["obs_0"] == ("7.0" if highest else "1.5"), "the score of the kept PSM"


def test_keep_best_prefers_a_ranked_row_and_keeps_file_order_on_ties() -> None:
    intensity = layer(pl.DataFrame({"Feature": ["F1", "F1", "F1"], "obs_0": [1.0, 2.0, 3.0]}))
    score = layer(pl.DataFrame({"Feature": ["F1", "F1", "F1"], "obs_0": [None, 4.0, 4.0]}))

    policy = KeepBestDuplicate(by="L", highest=False).ranked(score, NULL_ONLY)

    assert policy.resolve(intensity, NULL_ONLY).values.to_dicts() == [
        {"Feature": "F1", "obs_0": 2.0}
    ]


def test_keep_best_refuses_a_ranking_layer_that_is_not_numbers() -> None:
    score = layer(pl.DataFrame({"Feature": ["F1"], "obs_0": ["high"]}))

    with pytest.raises(AggregateTypeError, match="not plain numbers"):
        KeepBestDuplicate(by="L", highest=True).ranked(score, NULL_ONLY)


def test_keep_best_refuses_a_layer_that_does_not_repeat_the_ranking_rows() -> None:
    intensity, score = _ranked_pair()
    policy = KeepBestDuplicate(by="Score", highest=True).ranked(score, NULL_ONLY)
    shorter = layer(intensity.values.head(2))

    with pytest.raises(ConversionError, match="does not repeat the rows"):
        policy.resolve(shorter, NULL_ONLY)


def test_keep_best_is_built_from_the_declared_ranking_only() -> None:
    declared = WorkingMeasurements(
        primary_layer_name="Intensity",
        duplicate_mode="keep_best",
        layers=(),
        required_names=frozenset(),
        duplicate_ranking=DuplicateRanking(layer="Score", highest=True),
    )

    assert duplicate_policy(declared) == KeepBestDuplicate(by="Score", highest=True)
    with pytest.raises(ValueError, match="ranking layer"):
        duplicate_policy_for("keep_best")


def test_a_rule_names_its_ranking_layer_exactly_for_keep_best() -> None:
    assert Duplicates(mode="keep_best", by="Score").best == "highest"
    for invalid in (
        {"mode": "keep_best"},
        {"mode": "max", "by": "Score"},
        {"mode": "max", "best": "lowest"},
    ):
        with pytest.raises(ValueError, match="keep_best"):
            Duplicates.model_validate(invalid)
