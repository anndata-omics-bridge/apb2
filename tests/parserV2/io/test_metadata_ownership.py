"""Generic H5AD namespace composition has no tool-specific ownership rules."""

from copy import deepcopy

import pytest

from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.metadata import compose_metadata, split_metadata


@pytest.mark.parametrize(
    ("root", "level"),
    [
        ({}, {}),
        ({"extension": {}}, {"extension": {"value": 6}}),
        ({"extension": {"value": 6}}, {"extension": {}}),
        ({"extension": {"empty": {}}}, {"extension": {"empty": {}}}),
        (
            {"extension": {"z": {}, "b": 2, "a": None, "items": []}},
            {"extension": {"a/b": "not a path", "": True}},
        ),
        (
            {"tool": {"provenance": {"operation": {"source": "input.tsv"}}}},
            {"tool": {"operation": {"count": 6, "diagnostics": [1, None, "x"]}}},
        ),
    ],
)
def test_disjoint_metadata_round_trip(
    root: dict[str, JsonValue], level: dict[str, JsonValue]
) -> None:
    before = deepcopy((root, level))
    combined, ownership = compose_metadata(root, level)
    assert split_metadata(combined, ownership) == before
    assert (root, level) == before
    assert set(ownership) == {"root", "level"}
    assert split_metadata(deepcopy(combined), deepcopy(ownership)) == before


@pytest.mark.parametrize("other", [1, 2, {}, {"child": 1}, None])
def test_overlapping_leaves_never_choose_precedence(other: JsonValue) -> None:
    with pytest.raises(InvalidResultError, match=r"conflicting APB metadata.*tool.*value"):
        compose_metadata({"tool": {"value": 1}}, {"tool": {"value": other}})


@pytest.mark.parametrize(
    "paths",
    [[], [["missing"]], [["x"], ["x"]], [["tool"]], [["tool", "n"]]],
)
def test_ownership_cannot_drop_duplicate_or_invent_values(paths: list[JsonValue]) -> None:
    with pytest.raises(InvalidResultError):
        split_metadata(
            {"x": 1, "tool": {"n": 2}},
            {
                "root": {"values": paths, "empty_objects": []},
                "level": {"values": [], "empty_objects": [[]]},
            },
        )
