"""Presence and duplicate resolution: only what a raw cell claims, never what it means.

The distinction these tests protect: presence may inspect a token but never converts it, and
a policy may select or add scalars but never compares final keys. A token that cannot be read
stays present, so a later encoding failure is not silently resolved away here.

Layers are written here as small wide frames for readability; ``resolved`` places them as long
rows, nulls what presence rejects, and resolves them as the parser does: the rows alone in
their cell pass through and the policy receives the repeated ones.
"""

from __future__ import annotations

from dataclasses import replace
from typing import get_args

import polars as pl
import pytest

from apb2.parserV2.parse_quant.contracts import DuplicatePolicy, LayerValueParser
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.layer_columns import ObservationGrid
from apb2.parserV2.parse_quant.data.raw import OBS_POS, VALUE, VAR_POS, LongLayer
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


def placed(
    frame: pl.DataFrame,
    presence: LayerValueParser,
    *,
    keys: tuple[str, ...] = ("Feature",),
    name: str = "L",
) -> LongLayer:
    """A wide test frame as placed long rows, column by column, with absent scalars nulled."""
    features = frame.select(keys).unique(maintain_order=True)
    var_pos = (
        frame.select(keys)
        .join(
            features.with_row_index(VAR_POS),
            on=list(keys),
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        .get_column(VAR_POS)
    )
    columns = [column for column in frame.columns if column not in keys]
    rows = pl.concat(
        [
            pl.DataFrame(
                [
                    var_pos,
                    pl.repeat(index, frame.height, dtype=pl.UInt32, eager=True).alias(OBS_POS),
                    frame.get_column(column).alias(VALUE),
                ]
            )
            for index, column in enumerate(columns)
        ],
        how="vertical_relaxed",
    )
    present = rows.select(presence.present(pl.col(VALUE), rows.schema[VALUE])).to_series()
    rows = rows.with_columns(pl.when(present).then(pl.col(VALUE)).alias(VALUE))
    return LongLayer(name, rows, features)


def resolved(policy: DuplicatePolicy, layer: LongLayer) -> LongLayer:
    """Resolve as the parser does: single cells pass, the policy sees the repeated ones."""
    once, repeated = layer.split()
    result = policy.resolve(repeated)
    if not result.rows.height:
        return once
    return replace(once, rows=pl.concat([once.rows, result.rows], how="vertical_relaxed"))


def wide(layer: LongLayer, *labels: str) -> pl.DataFrame:
    """The resolved cells back in the wide test shape: keys first, then one column per label."""
    grid = ObservationGrid(layer.var_keys.height, labels)
    rows = layer.rows
    values = grid.widen(rows.get_column(VAR_POS), rows.get_column(OBS_POS), rows[VALUE], None)
    return pl.concat([layer.var_keys, values], how="horizontal")


def resolve_wide(
    policy: DuplicatePolicy, frame: pl.DataFrame, presence: LayerValueParser
) -> pl.DataFrame:
    labels = [column for column in frame.columns if column != "Feature"]
    return wide(resolved(policy, placed(frame, presence)), *labels)


def presence_mask(strategy: LayerValueParser, values: pl.Series) -> pl.Series:
    """Evaluate one presence expression as the parser does before resolution."""
    return (
        values.to_frame()
        .select(strategy.present(pl.col(values.name), values.dtype).alias(values.name))
        .to_series()
    )


def parse_wide(parser: LayerValueParser, frame: pl.DataFrame) -> pl.DataFrame:
    """Parse one wide test column through the long cells the parser would hand it."""
    layer = placed(frame.with_row_index("Feature"), parser, keys=("Feature",))
    return parser.parse(layer, ObservationGrid(frame.height, tuple(frame.columns)))[0].wide()


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
    parsed = parse_wide(NONPOSITIVE, pl.DataFrame({"obs_0": [5.0, 0.0, -1.0]}))

    assert parsed.get_column("obs_0").to_list() == [5.0, None, None]


def test_a_nonblank_token_that_cannot_be_read_stays_present() -> None:
    values = pl.Series("obs_0", ["12", "not a number"])

    # Keep-first must not be able to hide this; the AnnData encoder is where it fails.
    assert presence_mask(ZERO_SENTINEL, values).to_list() == [True, True]


def test_a_declared_missing_token_claims_nothing() -> None:
    dashes = make_layer_parser(
        "L", PlainNumericLayerDeclaration(missing_values=(), missing_tokens=("-",)), DOT
    )
    values = pl.Series("obs_0", ["12", "-", "not a number", None])

    assert presence_mask(dashes, values).to_list() == [True, False, True, False]


def test_a_localized_sentinel_is_recognized_under_its_own_notation() -> None:
    presence = make_layer_parser(
        "L",
        PlainNumericLayerDeclaration(missing_values=(0.0, 1000.0)),
        GROUPED,
    )
    values = pl.Series("obs_0", ["1.000", "1.000,5", "0", None])

    assert presence_mask(presence, values).to_list() == [False, True, False, False]


def test_regex_presence_reads_the_number_inside_a_structured_token() -> None:
    values = pl.Series("obs_0", ["site:1.5", "site:0", None, None, "unstructured"])

    assert presence_mask(ASCORE, values).to_list() == [True, False, False, False, True]


def test_every_presence_strategy_returns_a_mask_of_its_input_shape() -> None:
    strategies: tuple[LayerValueParser, ...] = (NULL_ONLY, ZERO_SENTINEL, ASCORE)
    values = pl.Series("obs_0", ["1", None, "2", ""])

    for strategy in strategies:
        mask = presence_mask(strategy, values)
        assert mask.len() == values.len()
        assert mask.dtype == pl.Boolean
        assert mask.null_count() == 0


# ------------------------------------------------------------------------------- policies


def test_two_claiming_values_in_one_cell_are_an_error_when_the_rule_says_so() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [1.0, 2.0]})

    with pytest.raises(
        DuplicateCellError, match=r"1 raw key\(s\).*examples: \[\{'Feature': 'F1'\}\]"
    ):
        resolve_wide(ErrorOnDuplicates(), repeated, NULL_ONLY)


def test_one_claiming_value_beside_a_sentinel_is_not_a_duplicate() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [0.0, 2.0]})

    result = resolve_wide(ErrorOnDuplicates(), repeated, ZERO_SENTINEL)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": 2.0}]


def test_keep_first_selects_independently_in_every_observation_column() -> None:
    repeated = pl.DataFrame(
        {
            "Feature": ["F1", "F1", "F2"],
            "obs_0": [None, 12.0, 5.0],
            "obs_1": [3.0, 4.0, None],
        }
    )

    result = resolve_wide(KeepFirstDuplicate(), repeated, NULL_ONLY)

    assert result.to_dicts() == [
        {"Feature": "F1", "obs_0": 12.0, "obs_1": 3.0},
        {"Feature": "F2", "obs_0": 5.0, "obs_1": None},
    ]


def test_keep_first_skips_a_sentinel_and_keeps_the_real_value_unencoded() -> None:
    """AlphaDIA writes 0 for "not measured"; the value kept is the vendor's own scalar."""
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["0", "12.5"]})

    result = resolve_wide(KeepFirstDuplicate(), repeated, ZERO_SENTINEL)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": "12.5"}]
    assert result.schema["obs_0"] == pl.String


def test_an_unreadable_token_reaches_the_result_instead_of_being_resolved_away() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["broken", "12.5"]})

    result = resolve_wide(KeepFirstDuplicate(), repeated, ZERO_SENTINEL)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": "broken"}]


def test_an_unknown_factor_label_also_stays_present() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": ["surprise", "MBR"]})

    with pytest.raises(DuplicateCellError):
        resolve_wide(ErrorOnDuplicates(), repeated, NULL_ONLY)


def test_numeric_aggregate_sums_only_the_claiming_values() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1", "F1", "F2"], "obs_0": [1.0, 0.0, 2.0, 7.0]})

    result = resolve_wide(SUM, repeated, ZERO_SENTINEL)

    assert result.to_dicts() == [
        {"Feature": "F1", "obs_0": 3.0},
        {"Feature": "F2", "obs_0": 7.0},
    ]


def test_numeric_aggregate_drops_nonpositive_values_before_summing() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1", "F1", "F2"], "obs_0": [1.0, -2.0, 0.0, -7.0]})

    result = resolve_wide(SUM, repeated, NONPOSITIVE)

    assert result.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0},
        {"Feature": "F2", "obs_0": None},
    ]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
@pytest.mark.parametrize("missing", [0.0, None, float("nan")])
def test_a_cell_with_nothing_present_stays_null_instead_of_becoming_zero(
    policy: DuplicatePolicy,
    missing: float | None,
) -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [missing, None]})

    result = resolve_wide(policy, repeated, ZERO_SENTINEL)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": None}]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_numeric_aggregate_refuses_values_that_are_not_numbers(policy: DuplicatePolicy) -> None:
    text = pl.DataFrame({"Feature": ["F1"], "obs_0": ["MBR"]})

    with pytest.raises(AggregateTypeError, match="needs numeric"):
        resolve_wide(policy, text, NULL_ONLY)


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_numeric_aggregate_accepts_a_layer_that_resolved_to_no_values_at_all(
    policy: DuplicatePolicy,
) -> None:
    empty = pl.DataFrame(
        {"Feature": ["F1", "F1"], "obs_0": [None, None]}, schema_overrides={"obs_0": pl.Null}
    )

    result = resolve_wide(policy, empty, NULL_ONLY)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": None}]


def test_numeric_max_never_selects_a_value_presence_masked() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1", "F2", "F2"], "obs_0": [1.0, 3.0, -7.0, -1.0]})

    result = resolve_wide(MAX, repeated, NONPOSITIVE)

    assert result.to_dicts() == [
        {"Feature": "F1", "obs_0": 3.0},
        {"Feature": "F2", "obs_0": None},
    ]


@pytest.mark.parametrize("policy", [SUM, MAX], ids=["sum", "max"])
def test_nan_is_absent_before_a_numeric_reduction(policy: DuplicatePolicy) -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [float("nan"), 1.0]})

    result = resolve_wide(policy, repeated, NULL_ONLY)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": 1.0}]


def test_numeric_max_stays_above_a_negative_threshold() -> None:
    """Values above -1 can sum to -1 or less; their max cannot, so parsing keeps it."""
    above_minus_one = make_layer_parser(
        "L", PlainNumericLayerDeclaration(missing_values=(), missing_at_or_below=-1.0), DOT
    )
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [-0.5, -0.8]})

    result = resolve_wide(MAX, repeated, above_minus_one).drop("Feature")

    assert parse_wide(above_minus_one, result).get_column("obs_0").to_list() == [-0.5]


def test_numeric_max_keeps_an_integer_layer_integer() -> None:
    repeated = pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [3, 5]})

    result = resolve_wide(MAX, repeated, NULL_ONLY)

    assert result.to_dicts() == [{"Feature": "F1", "obs_0": 5}]
    assert result.schema["obs_0"] == pl.Int64


@pytest.mark.parametrize(
    "policy", [duplicate_policy_for(mode) for mode in STATELESS], ids=STATELESS
)
@pytest.mark.parametrize("empty", [False, True])
def test_every_policy_keeps_the_layer_name_and_one_row_per_cell_in_first_order(
    policy: DuplicatePolicy,
    empty: bool,
) -> None:
    rows = pl.DataFrame(
        {VAR_POS: [2, 0, 2, 1], OBS_POS: [0, 0, 0, 1], VALUE: [1.0, None, None, 3.0]},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64},
    )
    layer = LongLayer("L", rows.head(0) if empty else rows)

    result = policy.resolve(layer)

    assert result.layer_name == "L"
    assert result.rows.columns == [VAR_POS, OBS_POS, VALUE]
    assert result.rows.select(VAR_POS, OBS_POS).rows() == (
        [] if empty else [(2, 0), (0, 0), (1, 1)]
    )


def test_split_sets_apart_the_rows_that_repeat_a_cell() -> None:
    rows = pl.DataFrame(
        {VAR_POS: [0, 1, 0, 2], OBS_POS: [0, 0, 0, 0], VALUE: ["a", "b", "c", "d"]},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.String},
    )

    once, repeated = LongLayer("L", rows).split()

    assert once.rows.get_column(VALUE).to_list() == ["b", "d"]
    assert repeated.rows.get_column(VALUE).to_list() == ["a", "c"]
    unique_once, nothing = LongLayer("L", once.rows).split()
    assert unique_once.rows is once.rows, "a layer without repeats is passed on as it is"
    assert nothing.rows.height == 0
    assert nothing.rows.schema == once.rows.schema


def test_the_declared_mode_selects_one_stateless_policy() -> None:
    """The removed legacy mode is not a value this selector can be given."""
    assert isinstance(duplicate_policy_for("error"), ErrorOnDuplicates)
    assert isinstance(duplicate_policy_for("keep_first"), KeepFirstDuplicate)
    assert isinstance(duplicate_policy_for("sum"), AggregateNumericDuplicates)
    assert isinstance(duplicate_policy_for("min"), AggregateNumericDuplicates)
    assert isinstance(duplicate_policy_for("max"), AggregateNumericDuplicates)
    assert set(MODES) == {"error", "keep_first", "sum", "min", "max", "keep_best"}


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


INTENSITY = pl.DataFrame(
    {
        "Feature": ["F1", "F1", "F2"],
        "obs_0": [100.0, 200.0, 10.0],
        "obs_1": [50.0, None, 20.0],
    }
)
SCORE = pl.DataFrame(
    {
        "Feature": ["F1", "F1", "F2"],
        "obs_0": ["1.5", "7.0", "2.0"],
        "obs_1": ["3.0", None, "1.0"],
    }
)


def _ranked_pair() -> tuple[LongLayer, LongLayer]:
    """An intensity and a score layer from one long table: F1 has two PSMs in obs_0."""
    return (
        placed(INTENSITY, NULL_ONLY, name="Intensity"),
        placed(SCORE, NULL_ONLY, name="Score"),
    )


@pytest.mark.parametrize(("highest", "kept"), [(True, 200.0), (False, 100.0)])
def test_keep_best_takes_every_layer_from_the_best_ranked_row(highest: bool, kept: float) -> None:
    intensity, score = _ranked_pair()
    policy = KeepBestDuplicate(by="Score", highest=highest).ranked(score)

    values = wide(resolved(policy, intensity), "obs_0", "obs_1").to_dicts()
    scores = wide(resolved(policy, score), "obs_0", "obs_1").to_dicts()

    assert values == [
        {"Feature": "F1", "obs_0": kept, "obs_1": 50.0},
        {"Feature": "F2", "obs_0": 10.0, "obs_1": 20.0},
    ]
    assert scores[0]["obs_0"] == ("7.0" if highest else "1.5"), "the score of the kept PSM"


def test_keep_best_sums_its_summed_layers_and_takes_the_rest_from_the_best_row() -> None:
    intensity, score = _ranked_pair()
    policy = KeepBestDuplicate(by="Score", highest=False, summed=frozenset({"Intensity"})).ranked(
        score
    )

    assert wide(resolved(policy, intensity), "obs_0", "obs_1").to_dicts() == [
        {"Feature": "F1", "obs_0": 300.0, "obs_1": 50.0},
        {"Feature": "F2", "obs_0": 10.0, "obs_1": 20.0},
    ], "F1's two PSMs in obs_0 add up"
    assert wide(resolved(policy, score), "obs_0", "obs_1").to_dicts()[0]["obs_0"] == "1.5"


def test_keep_best_prefers_a_ranked_row_and_keeps_file_order_on_ties() -> None:
    intensity = placed(pl.DataFrame({"Feature": ["F1"] * 3, "obs_0": [1.0, 2.0, 3.0]}), NULL_ONLY)
    score = placed(pl.DataFrame({"Feature": ["F1"] * 3, "obs_0": [None, 4.0, 4.0]}), NULL_ONLY)

    policy = KeepBestDuplicate(by="L", highest=False).ranked(score)

    assert wide(resolved(policy, intensity), "obs_0").to_dicts() == [
        {"Feature": "F1", "obs_0": 2.0}
    ]


def test_keep_best_refuses_a_ranking_layer_that_is_not_numbers() -> None:
    score = placed(pl.DataFrame({"Feature": ["F1"], "obs_0": ["high"]}), NULL_ONLY)

    with pytest.raises(AggregateTypeError, match="not plain numbers"):
        KeepBestDuplicate(by="L", highest=True).ranked(score)


def test_keep_best_reads_every_ranking_value_even_in_a_cell_of_its_own() -> None:
    """A bad ranking value fails whether or not its cell repeats; F2 is alone in obs_0."""
    score = placed(
        pl.DataFrame({"Feature": ["F1", "F1", "F2"], "obs_0": ["1", "2", "high"]}), NULL_ONLY
    )

    with pytest.raises(AggregateTypeError, match="not plain numbers"):
        KeepBestDuplicate(by="L", highest=True).ranked(score)


def test_keep_best_refuses_a_layer_that_does_not_repeat_the_ranking_rows() -> None:
    _intensity, score = _ranked_pair()
    policy = KeepBestDuplicate(by="Score", highest=True).ranked(score)
    shorter = placed(INTENSITY.slice(1), NULL_ONLY, name="Intensity")

    with pytest.raises(ConversionError, match="does not repeat the rows"):
        resolved(policy, shorter)


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
