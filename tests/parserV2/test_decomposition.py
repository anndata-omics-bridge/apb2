"""Decomposition: three physical shapes, one raw contract, repeated cells intact.

What must hold whatever the layout was: a raw axis has one stable-first row per complete raw
key, a raw layer is wide with its var keys first and its value columns positionally aligned to
the obs frame, and nothing has been decided about repeated cells yet.
"""

from __future__ import annotations

import polars as pl
import pytest

from apb2.parserV2.compile import (
    make_fragment_table_separator,
    make_source_decomposer,
)
from apb2.parserV2.parse_quant.data.raw import DecomposedDataRaw
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.decomposition import DelimitedFragmentSourceDecomposer
from apb2.parserV2.parse_quant.fragments import PackedLengthError
from apb2.parserV2.parse_quant.layer_labels import observation_labels
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan, AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentSeparationConfig,
    DelimitedFragmentDecompositionConfig,
    LongDecompositionConfig,
    LongRawLayerSource,
    PositionalFragmentSeparationConfig,
    WideDecompositionConfig,
    WideRawLayerPlan,
    WideRawLayerSource,
)


def axis(raw: tuple[str, ...], payload: tuple[str, ...] = ()) -> AxisSourcePlan:
    return AxisSourcePlan(
        keys=AxisKeyPlan(raw_key_columns=raw, key_input_columns=raw, final_key_columns=raw),
        payload_sources=payload,
    )


def long_config(*sources: tuple[str, str], primary: str) -> LongDecompositionConfig:
    return LongDecompositionConfig(
        kind="long",
        primary_layer_name=primary,
        layer_sources=tuple(
            LongRawLayerSource(name=name, source_column=column) for name, column in sources
        ),
    )


def layers_of(raw: DecomposedDataRaw) -> dict[str, pl.DataFrame]:
    return {layer.layer_name: layer.values for layer in raw.layers.values}


# ------------------------------------------------------------------------- one raw contract


def test_equivalent_long_and_wide_inputs_produce_the_same_raw_invariant() -> None:
    long_table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "A", "B", "B"],
                "Feature": ["F1", "F2", "F1", "F2"],
                "intensity": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )
    wide_table = LevelSourceTable(
        frame=pl.DataFrame({"Feature": ["F1", "F2"], "A": [1.0, 2.0], "B": [3.0, 4.0]})
    )

    from_long = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("Feature",)),
    ).decompose(long_table)
    from_wide = make_source_decomposer(
        WideDecompositionConfig(
            kind="wide",
            primary_layer_name="Intensity",
            layer_plans=(
                WideRawLayerPlan(
                    name="Intensity",
                    sources=(
                        WideRawLayerSource(source_column="A", sample="A"),
                        WideRawLayerSource(source_column="B", sample="B"),
                    ),
                ),
            ),
        ),
        axis(("run",)),
        axis(("Feature",)),
    ).decompose(wide_table)

    assert from_long.obs.frame.get_column("run").to_list() == ["A", "B"]
    assert from_wide.obs.frame.get_column("run").to_list() == ["A", "B"]
    assert from_long.var.frame.to_dicts() == from_wide.var.frame.to_dicts()
    # The wide file lists F1's samples across columns, the long file across rows; both mean
    # the same layer.
    assert layers_of(from_long)["Intensity"].to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0},
        {"Feature": "F2", "obs_0": 2.0, "obs_1": 4.0},
    ]
    assert layers_of(from_wide)["Intensity"].to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0},
        {"Feature": "F2", "obs_0": 2.0, "obs_1": 4.0},
    ]


def test_a_layer_puts_its_var_keys_first_and_the_obs_columns_in_axis_order() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["B", "A"],
                "seq": ["P", "P"],
                "charge": ["2", "2"],
                "intensity": [1.0, 2.0],
            }
        )
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("seq", "charge")),
    ).decompose(table)
    values = layers_of(raw)["Intensity"]

    assert raw.obs.frame.get_column("run").to_list() == ["B", "A"]
    assert values.columns == ["seq", "charge", "obs_0", "obs_1"]
    assert values.to_dicts() == [{"seq": "P", "charge": "2", "obs_0": 1.0, "obs_1": 2.0}]


def test_a_raw_axis_keeps_the_first_payload_it_saw_for_one_key() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "A"],
                "Feature": ["F1", "F1"],
                "Gene": ["first", "second"],
                "intensity": [1.0, 2.0],
            }
        )
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("Feature",), ("Gene",)),
    ).decompose(table)

    assert raw.var.frame.to_dicts() == [{"Feature": "F1", "Gene": "first"}]


def test_a_missing_raw_key_component_is_an_identity_of_its_own() -> None:
    """A declared ``coalesce`` may still make a valid final key out of it.

    So decomposition keeps the row: "complete raw-key tuple" means the whole tuple, not a
    tuple without nulls. Whether the identity survives is the final key's question, and axis
    preparation removes the row and its layer cells when it does not.
    """
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", None, "B"],
                "Feature": ["F1", "F1", None],
                "intensity": [1.0, 2.0, 3.0],
            }
        )
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("Feature",)),
    ).decompose(table)

    assert raw.obs.frame.get_column("run").to_list() == ["A", None, "B"]
    assert raw.var.frame.get_column("Feature").to_list() == ["F1", None]
    assert layers_of(raw)["Intensity"].to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 2.0, "obs_2": None},
        {"Feature": None, "obs_0": None, "obs_1": None, "obs_2": 3.0},
    ]


def test_a_multi_column_obs_key_becomes_several_columns_of_one_raw_axis() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "A"],
                "fraction": ["1", "2"],
                "Feature": ["F1", "F1"],
                "intensity": [1.0, 2.0],
            }
        )
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run", "fraction")),
        axis(("Feature",)),
    ).decompose(table)

    assert raw.obs.raw_key_columns == ("run", "fraction")
    assert raw.obs.frame.height == 2
    assert layers_of(raw)["Intensity"].to_dicts() == [{"Feature": "F1", "obs_0": 1.0, "obs_1": 2.0}]


# ------------------------------------------------------------------------- repeated cells


def test_a_repeated_long_cell_becomes_a_repeated_var_row() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "B", "A", "C"],
                "Feature": ["F1", "F1", "F1", "F1"],
                "intensity": [10.0, 20.0, 11.0, 30.0],
            }
        )
    )

    values = layers_of(
        make_source_decomposer(
            long_config(("Intensity", "intensity"), primary="Intensity"),
            axis(("run",)),
            axis(("Feature",)),
        ).decompose(table)
    )["Intensity"]

    assert values.to_dicts() == [
        {"Feature": "F1", "obs_0": 10.0, "obs_1": 20.0, "obs_2": 30.0},
        {"Feature": "F1", "obs_0": 11.0, "obs_1": None, "obs_2": None},
    ]


def test_two_wide_columns_claiming_one_sample_become_repeated_rows() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {"Feature": ["F1", "F2"], "A one": [1.0, 2.0], "A two": [9.0, None], "B": [3.0, 4.0]}
        )
    )
    config = WideDecompositionConfig(
        kind="wide",
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity",
                sources=(
                    WideRawLayerSource(source_column="A one", sample="A"),
                    WideRawLayerSource(source_column="B", sample="B"),
                    WideRawLayerSource(source_column="A two", sample="A"),
                ),
            ),
        ),
    )

    values = layers_of(
        make_source_decomposer(config, axis(("sample",)), axis(("Feature",))).decompose(table)
    )["Intensity"]

    assert values.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0},
        {"Feature": "F1", "obs_0": 9.0, "obs_1": None},
        {"Feature": "F2", "obs_0": 2.0, "obs_1": 4.0},
        {"Feature": "F2", "obs_0": None, "obs_1": None},
    ]


def test_a_required_wide_layer_with_no_aligned_column_stays_aligned_and_empty() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Feature": ["F1"], "A": [1.0]}))
    config = WideDecompositionConfig(
        kind="wide",
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity",
                sources=(WideRawLayerSource(source_column="A", sample="A"),),
            ),
            WideRawLayerPlan(name="Count", sources=()),
        ),
    )

    raw = make_source_decomposer(config, axis(("sample",)), axis(("Feature",))).decompose(table)
    count = layers_of(raw)["Count"]

    assert count.columns == ["Feature", "obs_0"]
    assert count.to_dicts() == [{"Feature": "F1", "obs_0": None}]
    assert [layer.layer_name for layer in raw.layers.values] == ["Intensity", "Count"]


def test_the_wide_observation_axis_comes_from_the_primary_layer_alone() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"Feature": ["F1"], "A Int": [1.0], "Z Count": [2.0]})
    )
    config = WideDecompositionConfig(
        kind="wide",
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity",
                sources=(WideRawLayerSource(source_column="A Int", sample="A"),),
            ),
            WideRawLayerPlan(
                name="Count",
                sources=(WideRawLayerSource(source_column="Z Count", sample="Z"),),
            ),
        ),
    )

    raw = make_source_decomposer(config, axis(("sample",)), axis(("Feature",))).decompose(table)

    assert raw.obs.frame.get_column("sample").to_list() == ["A"]
    # A non-primary sample token does not expand the observation axis, so the layer that
    # matched it aligns to nothing.
    assert layers_of(raw)["Count"].to_dicts() == [{"Feature": "F1", "obs_0": None}]


# ---------------------------------------------------------------------- packed fragments


def positional(*packed: str, delimiter: str = ";") -> PositionalFragmentSeparationConfig:
    return PositionalFragmentSeparationConfig(
        kind="positional",
        label_output="fragment_label",
        delimiter=delimiter,
        packed_value_sources=packed,
    )


def test_positional_separation_labels_each_scalar_by_its_index() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"Seq": ["P"], "Quant": ["1200;900;450"], "Corr": ["0.9;0.8;0.7"]})
    )

    separated = make_fragment_table_separator(positional("Quant", "Corr")).separate(table)

    assert separated.frame.to_dicts() == [
        {"Seq": "P", "Quant": "1200", "Corr": "0.9", "fragment_label": "frag_0"},
        {"Seq": "P", "Quant": "900", "Corr": "0.8", "fragment_label": "frag_1"},
        {"Seq": "P", "Quant": "450", "Corr": "0.7", "fragment_label": "frag_2"},
    ]


def test_a_trailing_terminator_is_not_an_extra_fragment() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["10;20;"]}))

    separated = make_fragment_table_separator(positional("Quant")).separate(table)

    assert separated.frame.get_column("Quant").to_list() == ["10", "20"]


def test_whitespace_is_trimmed_around_the_cell_and_around_each_token() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["  10 ; 20  "]}))

    separated = make_fragment_table_separator(positional("Quant")).separate(table)

    assert separated.frame.get_column("Quant").to_list() == ["10", "20"]


def test_an_interior_empty_token_stays_an_empty_scalar_at_its_position() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["10;;30"]}))

    separated = make_fragment_table_separator(positional("Quant")).separate(table)

    assert separated.frame.get_column("Quant").to_list() == ["10", "", "30"]
    assert separated.frame.get_column("fragment_label").to_list() == [
        "frag_0",
        "frag_1",
        "frag_2",
    ]


@pytest.mark.parametrize("cell", [None, "", "   ", ";", ";;"])
def test_a_row_with_no_tokens_contributes_no_scalar_row(cell: str | None) -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": [cell, "5"]}, strict=False))

    separated = make_fragment_table_separator(positional("Quant")).separate(table)

    assert separated.frame.get_column("Quant").to_list() == ["5"]


def test_parallel_packed_cells_of_different_length_are_a_vendor_defect() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["1;2"], "Corr": ["1;2;3"]}))

    with pytest.raises(PackedLengthError, match="different numbers"):
        make_fragment_table_separator(positional("Quant", "Corr")).separate(table)


def test_column_labelled_separation_takes_the_token_before_the_first_slash() -> None:
    config = ColumnLabeledFragmentSeparationConfig(
        kind="column",
        label_source="Info",
        label_output="fragment_label",
        delimiter=";",
        packed_value_sources=("Quant",),
    )
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {"Seq": ["P"], "Info": ["b4-unknown^1/327.16;y3^1/400.2;"], "Quant": ["10;20;"]}
        )
    )

    separated = make_fragment_table_separator(config).separate(table)

    assert separated.frame.columns == ["Seq", "Quant", "fragment_label"]
    assert separated.frame.get_column("fragment_label").to_list() == ["b4-unknown^1", "y3^1"]


def test_the_fragment_path_separates_first_and_then_delegates_once() -> None:
    calls: list[str] = []

    class Separator:
        def separate(self, table: LevelSourceTable, /) -> LevelSourceTable:
            calls.append("separate")
            return table

    class Long:
        def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
            calls.append("decompose")
            raise NotImplementedError

    composed = DelimitedFragmentSourceDecomposer(separator=Separator(), long_decomposer=Long())

    with pytest.raises(NotImplementedError):
        composed.decompose(LevelSourceTable(frame=pl.DataFrame({"a": [1]})))

    assert calls == ["separate", "decompose"]


def test_the_fragment_decomposer_reuses_the_ordinary_long_implementation() -> None:
    config = DelimitedFragmentDecompositionConfig(
        kind="delimited_fragment",
        separator=positional("Quant"),
        long=long_config(("Fragment_Quant", "Quant"), primary="Fragment_Quant"),
    )
    obs, var = axis(("Run",)), axis(("Seq", "fragment_label"))
    composed = make_source_decomposer(config, obs, var)
    table = LevelSourceTable(
        frame=pl.DataFrame({"Run": ["A", "B"], "Seq": ["P", "P"], "Quant": ["10;20", "30;40"]})
    )

    raw = composed.decompose(table)

    assert isinstance(composed, DelimitedFragmentSourceDecomposer)
    assert type(composed.long_decomposer) is type(make_source_decomposer(config.long, obs, var))
    assert raw.var.frame.to_dicts() == [
        {"Seq": "P", "fragment_label": "frag_0"},
        {"Seq": "P", "fragment_label": "frag_1"},
    ]
    assert layers_of(raw)["Fragment_Quant"].to_dicts() == [
        {"Seq": "P", "fragment_label": "frag_0", "obs_0": "10", "obs_1": "30"},
        {"Seq": "P", "fragment_label": "frag_1", "obs_0": "20", "obs_1": "40"},
    ]


# -------------------------------------------------------------------------- storage labels


def test_positional_labels_step_aside_for_a_vendor_column_of_the_same_name() -> None:
    assert observation_labels(2, reserved=()) == ("obs_0", "obs_1")
    assert observation_labels(2, reserved=("obs_0",)) == ("obs__0", "obs__1")
    assert observation_labels(0, reserved=("obs_0",)) == ()


def test_a_layer_value_column_never_collides_with_a_var_key_column() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"run": ["A"], "obs_0": ["F1"], "intensity": [1.0]})
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("obs_0",)),
    ).decompose(table)

    assert layers_of(raw)["Intensity"].columns == ["obs_0", "obs__0"]


# ------------------------------------------------------------------- nothing else survives


def test_no_returned_value_carries_a_counter_a_coordinate_or_a_matrix() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"run": ["A", "A"], "Feature": ["F1", "F1"], "intensity": [1.0, 2.0]})
    )

    raw = make_source_decomposer(
        long_config(("Intensity", "intensity"), primary="Intensity"),
        axis(("run",)),
        axis(("Feature",)),
    ).decompose(table)

    internal = {"_occurrence", "_var_slot", "_obs_slot"}
    assert not internal & set(raw.obs.frame.columns)
    assert not internal & set(raw.var.frame.columns)
    for layer in raw.layers.values:
        assert not internal & set(layer.values.columns)
        assert isinstance(layer.values, pl.DataFrame)
