"""Presence and duplicate resolution: only what a raw cell claims, never what it means.

The distinction these tests protect: presence may inspect a token but never converts it, and
a policy may select or add scalars but never compares final keys. A token that cannot be read
stays present, so a later encoding failure is not silently resolved away here.
"""

from __future__ import annotations

from typing import get_args

import polars as pl
import pytest

from apb2.parserV2.compile import make_raw_value_presence, policy_for
from apb2.parserV2.parse_quant.contracts import DuplicatePolicy, RawValuePresence
from apb2.parserV2.parse_quant.data.raw import RawLayerTable
from apb2.parserV2.parse_quant.duplicates import (
    AggregateNumericDuplicates,
    AggregateTypeError,
    DuplicateCellError,
    ErrorOnDuplicates,
    KeepFirstDuplicate,
    NullOnlyRawValuePresence,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    NullOnlyRawValuePresenceConfig,
    PlainNumericRawValuePresenceConfig,
    RegexNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
GROUPED = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))

NULL_ONLY = make_raw_value_presence(
    NullOnlyRawValuePresenceConfig(kind="null_only", layer_name="L")
)
ZERO_SENTINEL = make_raw_value_presence(
    PlainNumericRawValuePresenceConfig(
        kind="plain_numeric", layer_name="L", missing_values=(0.0,), number_format=DOT
    )
)
ASCORE = make_raw_value_presence(
    RegexNumericRawValuePresenceConfig(
        kind="regex_numeric",
        layer_name="L",
        missing_values=(0.0,),
        pattern=r":(-?\d+(?:\.\d+)?)",
        number_format=DOT,
    )
)


def layer(frame: pl.DataFrame, *, keys: tuple[str, ...] = ("Feature",)) -> RawLayerTable:
    return RawLayerTable(layer_name="L", raw_var_key_columns=keys, values=frame)


# ------------------------------------------------------------------------------- presence


def test_null_only_presence_asks_nothing_of_the_value_itself() -> None:
    values = pl.Series("obs_0", [1.0, 0.0, None])

    mask = NULL_ONLY.present(values)

    assert mask.to_list() == [True, True, False]
    assert mask.dtype == pl.Boolean
    assert mask.null_count() == 0


def test_a_declared_sentinel_claims_nothing_without_replacing_the_value() -> None:
    values = pl.Series("obs_0", [12.0, 0.0, None])

    assert ZERO_SENTINEL.present(values).to_list() == [True, False, False]
    # The strategy returned a mask; the value it was asked about is untouched.
    assert values.to_list() == [12.0, 0.0, None]


def test_blank_text_is_the_written_spelling_of_a_missing_number() -> None:
    values = pl.Series("obs_0", ["12", "", "   ", None])

    assert ZERO_SENTINEL.present(values).to_list() == [True, False, False, False]


def test_a_nonblank_token_that_cannot_be_read_stays_present() -> None:
    values = pl.Series("obs_0", ["12", "not a number"])

    # Keep-first must not be able to hide this; the AnnData encoder is where it fails.
    assert ZERO_SENTINEL.present(values).to_list() == [True, True]


def test_a_localized_sentinel_is_recognized_under_its_own_notation() -> None:
    presence = make_raw_value_presence(
        PlainNumericRawValuePresenceConfig(
            kind="plain_numeric",
            layer_name="L",
            missing_values=(0.0, 1000.0),
            number_format=GROUPED,
        )
    )
    values = pl.Series("obs_0", ["1.000", "1.000,5", "0", None])

    assert presence.present(values).to_list() == [False, True, False, False]


def test_regex_presence_reads_the_number_inside_a_structured_token() -> None:
    values = pl.Series("obs_0", ["site:1.5", "site:0", "", None, "unstructured"])

    assert ASCORE.present(values).to_list() == [True, False, False, False, True]


def test_every_presence_strategy_returns_a_mask_of_its_input_shape() -> None:
    strategies: tuple[RawValuePresence, ...] = (NULL_ONLY, ZERO_SENTINEL, ASCORE)
    values = pl.Series("obs_0", ["1", None, "2", ""])

    for strategy in strategies:
        mask = strategy.present(values)
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

    resolved = AggregateNumericDuplicates().resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [
        {"Feature": "F1", "obs_0": 3.0},
        {"Feature": "F2", "obs_0": 7.0},
    ]


def test_a_cell_with_nothing_present_stays_null_instead_of_becoming_zero() -> None:
    repeated = layer(pl.DataFrame({"Feature": ["F1", "F1"], "obs_0": [0.0, None]}))

    resolved = AggregateNumericDuplicates().resolve(repeated, ZERO_SENTINEL)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": None}]


def test_numeric_aggregate_refuses_values_that_are_not_numbers() -> None:
    text = layer(pl.DataFrame({"Feature": ["F1"], "obs_0": ["MBR"]}))

    with pytest.raises(AggregateTypeError, match="needs numeric"):
        AggregateNumericDuplicates().resolve(text, NULL_ONLY)


def test_numeric_aggregate_accepts_a_layer_that_resolved_to_no_values_at_all() -> None:
    empty = layer(
        pl.DataFrame({"Feature": ["F1"], "obs_0": [None]}, schema_overrides={"obs_0": pl.Null})
    )

    resolved = AggregateNumericDuplicates().resolve(empty, NULL_ONLY)

    assert resolved.values.to_dicts() == [{"Feature": "F1", "obs_0": None}]


@pytest.mark.parametrize(
    "policy",
    [ErrorOnDuplicates(), KeepFirstDuplicate(), AggregateNumericDuplicates()],
    ids=lambda policy: type(policy).__name__,
)
def test_every_policy_keeps_the_keys_the_group_order_and_the_layer_name(
    policy: DuplicatePolicy,
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

    resolved = policy.resolve(values, NULL_ONLY)

    assert resolved.layer_name == "L"
    assert resolved.raw_var_key_columns == ("Feature", "Charge")
    assert resolved.values.columns == ["Feature", "Charge", "obs_0"]
    assert resolved.values.get_column("Feature").to_list() == ["F2", "F1", "F3"]


@pytest.mark.parametrize(
    "policy",
    [ErrorOnDuplicates(), KeepFirstDuplicate(), AggregateNumericDuplicates()],
    ids=lambda policy: type(policy).__name__,
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


def test_the_declared_mode_selects_one_stateless_policy() -> None:
    """The removed legacy mode is not a value this selector can be given."""
    assert isinstance(policy_for("error"), ErrorOnDuplicates)
    assert isinstance(policy_for("keep_first"), KeepFirstDuplicate)
    assert isinstance(policy_for("aggregate"), AggregateNumericDuplicates)
    # A PEP 695 alias holds its literal union in ``__value__``.
    assert set(get_args(DuplicateMode.__value__)) == {"error", "keep_first", "aggregate"}


def test_no_policy_or_presence_strategy_retains_its_discriminator() -> None:
    for value in (
        ErrorOnDuplicates(),
        KeepFirstDuplicate(),
        AggregateNumericDuplicates(),
        NullOnlyRawValuePresence(),
    ):
        assert not hasattr(value, "kind")
        assert not hasattr(value, "mode")
        assert not hasattr(value, "layer_name")
