"""Building and querying parsed results through the public classes only."""

from __future__ import annotations

import polars as pl
import pytest

import apb2.api as public_api
from apb2.api import FinalLayerTable, ParsedLevel, ParsedLevels
from apb2.parserV2.parse_quant.data.parsed import (
    CategoricalLayerSemantics,
    QuantitativeLayerSemantics,
)


def _level() -> ParsedLevel:
    return ParsedLevel.build(
        obs=pl.DataFrame({"sample": ["A", "B"]}),
        obs_keys=("sample",),
        var=pl.DataFrame({"peptide": ["PEPA", "PEPB", "PEPC"], "protein": ["P1", "P1", "P2"]}),
        var_keys=("peptide",),
        var_roles={"protein_assignment": "protein"},
        primary_layer="Intensity",
        abundance={
            "Intensity": pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, None, 6.0]}),
            "Raw": pl.DataFrame({"x": [1.5, 2.5, 3.5], "y": [4.5, 5.5, 6.5]}),
        },
        auxiliary={"Count": pl.DataFrame({"A": [1, 2, 3], "B": [0, 0, 1]})},
    )


def test_the_public_api_is_exactly_the_approved_names() -> None:
    assert sorted(public_api.__all__) == [
        "AnnotationCompiler",
        "AnnotationError",
        "AnnotationResult",
        "ConversionError",
        "FinalLayerTable",
        "JsonValue",
        "LevelHierarchy",
        "LevelParseTimings",
        "Parameters",
        "ParseRuleCompiler",
        "ParsedLevel",
        "ParsedLevels",
        "QuantificationLevel",
        "ResultIOError",
        "RuleVariant",
        "SdrfSource",
        "SoftwareRules",
        "UnsJsonCodec",
        "canonical_modification_names",
        "get_rules",
        "packaged_rule_declarations",
        "parse_search_parameters",
        "read_parsed_levels",
        "sidecar_path",
        "write_parsed_levels",
    ]
    assert public_api.canonical_modification_names()["UNIMOD:35"] == "Oxidation"


def test_build_names_value_columns_by_observation_position() -> None:
    level = _level()

    intensity = level.layers["Intensity"]
    assert intensity.wide().columns == ["obs_0", "obs_1"]
    assert intensity.semantic_roles == ("abundance",)
    assert level.var.roles == {"protein_assignment": "protein"}
    assert level.layers["Count"].semantics == QuantitativeLayerSemantics("integer")
    assert level.layers["Count"].role.accepts_primary_layer() is False


def test_build_rejects_an_inconsistent_level() -> None:
    obs = pl.DataFrame({"sample": ["A"]})
    var = pl.DataFrame({"peptide": ["PEPA"]})
    one = {"Intensity": pl.DataFrame({"A": [1.0]})}
    count = {"Count": pl.DataFrame({"A": [1]})}

    with pytest.raises(ValueError, match="not an abundance layer"):
        ParsedLevel.build(obs, ("sample",), var, ("peptide",), {}, "Count", one, count)
    with pytest.raises(ValueError, match="absent"):
        ParsedLevel.build(obs, ("sample",), var, ("missing",), {}, "Intensity", one)
    with pytest.raises(ValueError, match="shape"):
        ParsedLevel.build(obs.vstack(obs), ("sample",), var, ("peptide",), {}, "Intensity", one)


def test_with_layers_returns_a_new_level_and_refuses_existing_names() -> None:
    level = _level()

    extended = level.with_layers(
        abundance={"Summed": pl.DataFrame({"s": [1.0, 1.0, 1.0], "t": [2.0, 2.0, 2.0]})},
        varm={"diagnostics": pl.DataFrame({"score": [0.1, 0.2, 0.3]})},
        metadata={"provenance": ["summed"]},
    )

    assert extended.layers["Summed"].wide().columns == ["obs_0", "obs_1"]
    assert "Summed" not in level.layers
    assert extended.metadata == {"provenance": ["summed"]}
    with pytest.raises(ValueError, match="already has"):
        extended.with_layers(auxiliary={"Summed": pl.DataFrame({"s": [1, 1, 1], "t": [1, 1, 1]})})


def test_export_uns_is_built_empty_or_as_given_and_copied_with_the_level() -> None:
    assert _level().export_uns == {}
    level = ParsedLevel.build(
        obs=pl.DataFrame({"sample": ["A"]}),
        obs_keys=("sample",),
        var=pl.DataFrame({"peptide": ["PEPA"]}),
        var_keys=("peptide",),
        var_roles={},
        primary_layer="Intensity",
        abundance={"Intensity": pl.DataFrame({"A": [1.0]})},
        export_uns={"search_engine": "dia-nn", "acquisition": None},
    )

    extended = level.with_layers(auxiliary={"Count": pl.DataFrame({"A": [1]})})
    extended.export_uns["added"] = True

    assert level.export_uns == {"search_engine": "dia-nn", "acquisition": None}
    assert extended.export_uns["search_engine"] == "dia-nn"


def test_a_result_copy_owns_its_export_uns() -> None:
    parsed = ParsedLevels(levels={"protein": _level()}, uns={}, export_uns={"status": "draft"})

    annotated = parsed.with_annotation_table(
        "members", pl.DataFrame({"member": ["P1"]}), ("member",)
    )
    annotated.export_uns["status"] = "final"

    assert parsed.export_uns == {"status": "draft"}


def test_abundance_layers_selects_all_or_validates_names() -> None:
    level = _level()

    assert level.abundance_layers() == ("Intensity", "Raw")
    assert level.abundance_layers(("Raw",)) == ("Raw",)
    with pytest.raises(ValueError, match="does not carry the abundance role"):
        level.abundance_layers(("Count",))
    with pytest.raises(ValueError, match="has no layer 'Missing'"):
        level.abundance_layers(("Missing",))
    with pytest.raises(ValueError, match="no layer carrying"):
        level.abundance_layers(())


def test_layer_values_are_numbers_or_decoded_labels() -> None:
    level = _level()
    flags = FinalLayerTable.from_wide(
        "Flag",
        pl.DataFrame({"obs_0": [0, 1, -1], "obs_1": [1, 1, 0]}),
        semantics=CategoricalLayerSemantics(categories=(("low", 0), ("high", 1))),
    )

    assert level.layers["Intensity"].decoded_values().equals(level.layers["Intensity"].wide())
    assert flags.decoded_values().to_dict(as_series=False) == {
        "obs_0": ["low", "high", None],
        "obs_1": ["high", "high", "low"],
    }
    with pytest.raises(ValueError, match="categorical"):
        flags.quantitative_values()


def test_annotation_tables_and_relations_attach_to_a_new_result() -> None:
    parsed = ParsedLevels(levels={"protein": _level()}, uns={})

    annotated = parsed.with_annotation_table(
        "members", pl.DataFrame({"member": ["P1", "P2"]}), ("member",), {"producer": "test"}
    ).with_feature_relation(
        "membership",
        pl.DataFrame(
            {"source": ["members"] * 2, "row": [0, 1], "target": ["protein"] * 2, "column": [0, 2]}
        ),
    )

    assert annotated.annotation_tables["members"].metadata == {"producer": "test"}
    assert annotated.feature_relations["membership"].coordinates.get_column("row").to_list() == [
        0,
        1,
    ]
    assert parsed.annotation_tables == {}
    with pytest.raises(ValueError, match="no level or annotation table 'absent'"):
        parsed.with_feature_relation(
            "r", pl.DataFrame({"source": ["absent"], "target": ["protein"]})
        )
    with pytest.raises(ValueError, match="no level or annotation table None"):
        parsed.with_feature_relation(
            "r",
            pl.DataFrame(
                {"source": [None], "target": ["protein"]},
                schema=dict.fromkeys(("source", "target"), pl.String),
            ),
        )
    with pytest.raises(ValueError, match="lacks key columns"):
        parsed.with_annotation_table("t", pl.DataFrame({"a": [1]}), ("b",))


# ------------------------------------------------------------------- long layers, wide on demand


@pytest.mark.parametrize(
    ("values", "semantics"),
    [
        (
            pl.DataFrame({"a": [1.0, None, float("nan")], "b": [None, None, 2.5]}),
            QuantitativeLayerSemantics(),
        ),
        (
            pl.DataFrame(
                {"a": [1, None, 3], "b": [None, 5, None]}, schema={"a": pl.Int64, "b": pl.Int64}
            ),
            QuantitativeLayerSemantics("integer"),
        ),
        (
            pl.DataFrame(
                {"a": [1, -1, 2], "b": [-1, None, 1]}, schema={"a": pl.Int64, "b": pl.Int64}
            ),
            CategoricalLayerSemantics(categories=(("found", 1), ("transferred", 2))),
        ),
        (pl.DataFrame(schema={"a": pl.Float64, "b": pl.Float64}), QuantitativeLayerSemantics()),
        (pl.DataFrame(), QuantitativeLayerSemantics()),
    ],
    ids=["number", "integer", "categorical", "no-var", "no-obs"],
)
def test_a_layer_stores_its_cells_and_widens_back_to_the_same_frame(
    values: pl.DataFrame, semantics: QuantitativeLayerSemantics | CategoricalLayerSemantics
) -> None:
    layer = FinalLayerTable.from_wide("L", values, semantics=semantics)

    assert layer.wide().equals(values, null_equal=True)
    assert layer.wide().schema == values.schema
    assert layer.height == values.height
    assert layer.labels == tuple(values.columns)


def test_only_cells_that_do_not_hold_the_fill_are_stored() -> None:
    numbers = FinalLayerTable.from_wide("L", pl.DataFrame({"a": [1.0, None], "b": [None, 2.0]}))
    codes = FinalLayerTable.from_wide(
        "S",
        pl.DataFrame({"a": [1, -1], "b": [-1, None]}),
        semantics=CategoricalLayerSemantics(categories=(("found", 1),)),
    )

    assert numbers.cells.rows() == [(0, 0, 1.0), (1, 1, 2.0)]
    assert codes.cells.rows() == [(0, 0, 1), (1, 1, None)], "a null code is kept, -1 is the fill"
