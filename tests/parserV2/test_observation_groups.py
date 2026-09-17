"""Different measurement resolutions never become a misleading global observation union."""

from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.observation_groups import group_observations


def _level(frame: pl.DataFrame, key: str) -> ParsedLevel:
    values = pl.DataFrame(
        {"feature": ["P"], **{f"obs_{i}": [i + 100] for i in range(frame.height)}}
    )
    return ParsedLevel(
        obs=ObsFinal(frame, (key,)),
        var=VarFinal(pl.DataFrame({"feature": ["P"]}), ("feature",)),
        primary_layer_name="Intensity",
        uns={},
        layers={"Intensity": FinalLayerTable("Intensity", ("feature",), values)},
        obsm={},
        varm={},
        obsp={},
        varp={},
    )


def _parsed(experiments: list[str | None], samples: list[str]) -> ParsedLevels:
    return ParsedLevels(
        levels={
            "ion": _level(
                pl.DataFrame({"Raw_File": ["raw_z", "raw_a"], "Experiment": experiments}),
                "Raw_File",
            ),
            "protein": _level(pl.DataFrame({"Experiment": samples}), "Experiment"),
        },
        uns={},
    )


def test_bijective_alignment_preserves_measurement_order_and_original_metadata() -> None:
    parsed = _parsed(["B", "A"], ["A", "B"])
    original = parsed.levels["protein"]
    (result,) = group_observations(parsed)
    protein = result.levels["protein"]
    assert protein.obs.key_columns == ("Raw_File",)
    assert protein.obs.frame["Raw_File"].to_list() == ["raw_a", "raw_z"]
    assert protein.obs.frame["Experiment"].to_list() == ["A", "B"]
    assert protein.layers["Intensity"] is original.layers["Intensity"]
    assert original.obs.key_columns == ("Experiment",)
    assert original.obs.frame.columns == ["Experiment"]
    assert protein.uns["observation_keys_original"] == ["Experiment"]
    relationships = result.uns["observation_relationships"]
    assert isinstance(relationships, str)
    assert json.loads(relationships)[0]["aligned"] is True


@pytest.mark.parametrize(
    ("experiments", "samples"),
    [(["A", "A"], ["A"]), (["A", None], ["A"]), (["A", "B"], ["C"]), (["A", "B"], ["A", "C"])],
    ids=["fractionated", "incomplete", "disjoint", "partially-mapped"],
)
def test_nonbijective_or_incomplete_mapping_stays_separate(
    experiments: list[str | None], samples: list[str]
) -> None:
    parsed = _parsed(experiments, samples)
    groups = group_observations(parsed)
    assert [list(group.levels) for group in groups] == [["ion"], ["protein"]]
    for group in groups:
        for name, level in group.levels.items():
            assert level is parsed.levels[name]


def test_missing_relationship_cannot_be_inferred_from_matching_labels() -> None:
    parsed = _parsed(["A", "B"], ["raw_z", "raw_a"])
    ion = parsed.levels["ion"]
    ion.obs.frame = ion.obs.frame.drop("Experiment")
    assert len(group_observations(parsed)) == 2


def test_authored_metadata_is_never_overwritten_by_alignment() -> None:
    parsed = _parsed(["A", "B"], ["A", "B"])
    protein = parsed.levels["protein"]
    protein.obs.frame = protein.obs.frame.with_columns(pl.lit("different").alias("Raw_File"))
    original = protein.obs.frame.clone()
    assert len(group_observations(parsed)) == 2
    assert_frame_equal(protein.obs.frame, original)


def test_single_identity_preserves_original_result() -> None:
    parsed = _parsed(["A", "B"], ["A", "B"])
    del parsed.levels["ion"]
    assert group_observations(parsed) == (parsed,)
