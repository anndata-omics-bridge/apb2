"""Different measurement resolutions never become a misleading global observation union."""

from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from parserV2.fixtures import parse_section


def _level(frame: pl.DataFrame, key: str) -> ParsedLevel:
    values = pl.DataFrame(
        {"feature": ["P"], **{f"obs_{i}": [i + 100] for i in range(frame.height)}}
    )
    return ParsedLevel(
        obs=ObsFinal(frame, (key,)),
        var=VarFinal(pl.DataFrame({"feature": ["P"]}), ("feature",)),
        primary_layer_name="Intensity",
        uns={},
        layers={
            "Intensity": FinalLayerTable.from_wide(
                "Intensity",
                (values).drop(("feature",), strict=False),
                semantic_roles=("abundance",),
            )
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
    )


def _parsed(experiments: list[str | None], samples: list[str]) -> ParsedLevels:
    return ParsedLevels(
        levels={
            "peptidoform_ion": _level(
                pl.DataFrame({"Raw_File": ["raw_z", "raw_a"], "Experiment": experiments}),
                "Raw_File",
            ),
            "protein": _level(pl.DataFrame({"Experiment": samples}), "Experiment"),
        },
        uns={},
    )


def test_bijective_alignment_preserves_cells_on_one_shared_axis() -> None:
    parsed = _parsed(["B", "A"], ["A", "B"])
    original = parsed.levels["protein"]
    (result,) = parsed.observation_groups()
    protein = result.levels["protein"]
    assert protein.obs.key_columns == ("Raw_File",)
    assert protein.obs.frame["Raw_File"].to_list() == ["raw_z", "raw_a"]
    assert protein.obs.frame["Experiment"].to_list() == ["B", "A"]
    assert protein.layers["Intensity"].wide().rows() == [(101, 100)]
    assert original.obs.key_columns == ("Experiment",)
    assert original.obs.frame.columns == ["Experiment"]
    assert parse_section(protein.uns, "result")["observation_keys_original"] == ["Experiment"]
    relationships = parse_section(result.uns, "result")["observation_relationships"]
    assert isinstance(relationships, str)
    assert json.loads(relationships)[0]["aligned"] is True


def test_alignment_fills_an_empty_observation_axis_without_losing_features() -> None:
    parsed = _parsed(["B", "A"], [])
    parsed.levels["protein"].obs.frame = pl.DataFrame(schema={"Experiment": pl.String})
    (result,) = parsed.observation_groups()
    protein = result.levels["protein"]
    assert protein.obs.frame["Raw_File"].to_list() == ["raw_z", "raw_a"]
    assert protein.layers["Intensity"].wide().rows() == [(None, None)]
    assert parsed.levels["protein"].obs.frame.is_empty()


def test_duplicate_observation_keys_are_rejected_before_alignment() -> None:
    parsed = _parsed(["B", "A"], ["B", "A", "B"])
    with pytest.raises(ValueError, match="duplicate observation keys"):
        parsed.observation_groups()
    assert parsed.levels["protein"].obs.frame["Experiment"].to_list() == ["B", "A", "B"]


@pytest.mark.parametrize(
    ("experiments", "samples"),
    [(["A", "A"], ["A"]), (["A", None], ["A"]), (["A", "B"], ["C"]), (["A", "B"], ["A", "C"])],
    ids=["fractionated", "incomplete", "disjoint", "partially-mapped"],
)
def test_nonbijective_or_incomplete_mapping_stays_separate(
    experiments: list[str | None], samples: list[str]
) -> None:
    parsed = _parsed(experiments, samples)
    groups = parsed.observation_groups()
    assert [list(group.levels) for group in groups] == [["peptidoform_ion"], ["protein"]]
    for group in groups:
        for name, level in group.levels.items():
            assert level is parsed.levels[name]


def test_separate_groups_keep_only_the_links_within_them() -> None:
    members = pl.DataFrame({"member": ["M1"]})
    parsed = (
        _parsed(["A", "A"], ["A"])
        .with_annotation_table("members", members, ("member",))
        .with_feature_relation(
            "membership",
            pl.DataFrame(
                {
                    "source": ["peptidoform_ion", "members"],
                    "row": [0, 0],
                    "target": ["protein", "peptidoform_ion"],
                    "column": [0, 0],
                    "value": [1.0, 1.0],
                }
            ),
        )
    )

    ion, protein = parsed.observation_groups()

    links = ion.feature_relations["membership"].coordinates
    assert links.select("source", "target").rows() == [("members", "peptidoform_ion")]
    assert "membership" not in protein.feature_relations, "no link lies within the protein group"


def test_every_separate_group_keeps_the_root_parse_provenance() -> None:
    parsed = _parsed(["A", "A"], ["A"])
    provenance: dict[str, JsonValue] = {"search_parameters": {"file_name": "search.params"}}
    parsed.uns["provenance"] = provenance
    groups = parsed.observation_groups()
    assert len(groups) == 2
    assert all(group.uns["provenance"] == provenance for group in groups)


def test_missing_relationship_cannot_be_inferred_from_matching_labels() -> None:
    parsed = _parsed(["A", "B"], ["raw_z", "raw_a"])
    ion = parsed.levels["peptidoform_ion"]
    ion.obs.frame = ion.obs.frame.drop("Experiment")
    assert len(parsed.observation_groups()) == 2


def test_authored_metadata_is_never_overwritten_by_alignment() -> None:
    parsed = _parsed(["A", "B"], ["A", "B"])
    protein = parsed.levels["protein"]
    protein.obs.frame = protein.obs.frame.with_columns(pl.lit("different").alias("Raw_File"))
    original = protein.obs.frame.clone()
    assert len(parsed.observation_groups()) == 2
    assert_frame_equal(protein.obs.frame, original)


def test_single_identity_preserves_original_result() -> None:
    parsed = _parsed(["A", "B"], ["A", "B"])
    del parsed.levels["peptidoform_ion"]
    assert parsed.observation_groups() == (parsed,)
