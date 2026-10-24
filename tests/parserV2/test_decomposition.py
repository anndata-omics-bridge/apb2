"""Decomposition: three physical shapes, one raw contract, repeated cells intact.

What must hold whatever the layout was: a raw axis has one stable-first row per complete raw
key, a layer is long rows naming their raw axis rows beside the value as read, and nothing has
been decided about repeated cells yet.
"""

from __future__ import annotations

import polars as pl
import pytest

from apb2.parserV2.parse_quant.data.layer_columns import ObservationGrid, observation_labels
from apb2.parserV2.parse_quant.data.raw import (
    DISAGREEING_VALUES,
    OBS_POS,
    OBS_ROW,
    VALUE,
    VAR_POS,
    VAR_ROW,
    FinalPositions,
    RawLevel,
)
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.duplicates import (
    ExtremeFeatureValue,
    FirstFeatureValue,
    KeepFirstDuplicate,
    RankedFeatureValue,
)
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PackedLengthError,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan, AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.source import (
    LevelSourceTable,
    LongRawLayerSource,
    NumericTextFormat,
    WideRawLayerPlan,
    WideRawLayerSource,
)


def axis(raw: tuple[str, ...], payload: tuple[str, ...] = ()) -> AxisSourcePlan:
    return AxisSourcePlan(
        keys=AxisKeyPlan(raw_key_columns=raw, key_input_columns=raw, final_key_columns=raw),
        payload_sources=payload,
    )


def first_values(var: AxisSourcePlan) -> tuple[FirstFeatureValue, ...]:
    """The default policy, a feature's first nonblank value, for every var payload column."""
    return tuple(FirstFeatureValue(source) for source in var.payload_sources)


def long_decomposer(
    *sources: tuple[str, str], primary: str, obs: AxisSourcePlan, var: AxisSourcePlan
) -> LongSourceDecomposer:
    return LongSourceDecomposer(
        obs=obs,
        var=var,
        var_values=first_values(var),
        primary_layer_name=primary,
        layer_sources=tuple(
            LongRawLayerSource(name=name, source_column=column) for name, column in sources
        ),
    )


def cells_of(raw: RawLevel, name: str) -> list[tuple[object, ...]]:
    """One layer's rows as (raw var keys…, raw obs keys…, value), in physical order."""
    layer = next(layer for layer in raw.layers if layer.layer_name == name)
    var_keys = raw.var.frame.select(raw.var.raw_key_columns)
    obs_keys = raw.obs.frame.select(raw.obs.raw_key_columns)
    return [
        (*var_keys.row(var_row), *obs_keys.row(obs_row), value)
        for var_row, obs_row, value in layer.rows.iter_rows()
    ]


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

    from_long = long_decomposer(
        ("Intensity", "intensity"), primary="Intensity", obs=axis(("run",)), var=axis(("Feature",))
    ).decompose(long_table)
    from_wide = WideSourceDecomposer(
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
        obs=axis(("run",)),
        var=axis(("Feature",)),
        var_values=(),
    ).decompose(wide_table)

    assert from_long.obs.frame.get_column("run").to_list() == ["A", "B"]
    assert from_wide.obs.frame.get_column("run").to_list() == ["A", "B"]
    assert from_long.var.frame.to_dicts() == from_wide.var.frame.to_dicts()
    # The wide file lists F1's samples across columns, the long file across rows; both mean
    # the same layer.
    cells = [("F1", "A", 1.0), ("F2", "A", 2.0), ("F1", "B", 3.0), ("F2", "B", 4.0)]
    assert cells_of(from_long, "Intensity") == cells
    assert cells_of(from_wide, "Intensity") == cells


def test_a_layer_names_its_axis_rows_and_the_obs_axis_keeps_first_seen_order() -> None:
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

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("seq", "charge")),
    ).decompose(table)
    (layer,) = raw.layers

    assert raw.obs.frame.get_column("run").to_list() == ["B", "A"]
    assert layer.rows.columns == [VAR_ROW, OBS_ROW, VALUE]
    assert layer.rows.rows() == [(0, 0, 1.0), (0, 1, 2.0)]
    assert cells_of(raw, "Intensity") == [("P", "2", "B", 1.0), ("P", "2", "A", 2.0)]


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

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("Feature",), ("Gene",)),
    ).decompose(table)

    assert raw.var.frame.to_dicts() == [{"Feature": "F1", "Gene": "first"}]
    assert {source: frame.to_dicts() for source, frame in raw.var.disagreements.items()} == {
        "Gene": [{"Feature": "F1", DISAGREEING_VALUES: ["first", "second"]}]
    }


def test_keep_first_takes_the_first_nonblank_value_and_agreeing_rows_are_no_finding() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "B", "A", "B"],
                "Feature": ["F1", "F1", "F2", "F2"],
                "Gene": [None, "G1", " ", "G2"],
                "Lib": ["0.01", "0.01", "0.02", None],
                "intensity": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("Feature",), ("Gene", "Lib")),
    ).decompose(table)

    assert raw.var.frame.to_dicts() == [
        {"Feature": "F1", "Gene": "G1", "Lib": "0.01"},
        {"Feature": "F2", "Gene": "G2", "Lib": "0.02"},
    ]
    assert raw.var.disagreements == {}


@pytest.mark.parametrize(("largest", "kept"), [(False, ["6,51e-4", "0,5"]), (True, ["1,0", "0,5"])])
def test_an_extreme_value_keeps_the_vendor_text_of_the_smallest_or_largest_number(
    largest: bool, kept: list[str]
) -> None:
    """DIA-NN writes 1.0 into some rows of a protein group; ``min`` keeps the group's value."""
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "B", "C", "A"],
                "Group": ["P1", "P1", "P1", "P2"],
                "Global": ["6,51e-4", "1,0", None, "0,5"],
                "intensity": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )
    var = axis(("Group",), ("Global",))
    comma = NumericTextFormat(decimal_mark=",", thousands_marks=())

    raw = LongSourceDecomposer(
        obs=axis(("run",)),
        var=var,
        var_values=(ExtremeFeatureValue("Global", largest=largest, notation=comma),),
        primary_layer_name="Intensity",
        layer_sources=(LongRawLayerSource(name="Intensity", source_column="intensity"),),
    ).decompose(table)

    assert raw.var.frame.get_column("Global").to_list() == kept
    assert raw.var.disagreements == {}, "the rule expects the rows to disagree"


def test_ranked_columns_come_from_the_best_scoring_row_whatever_the_row_order() -> None:
    """Proline lists a shared peptide once per protein set; the set's columns stay together.

    The best row's absent subset stays absent, a tie keeps file order, and a ``NaN`` score
    ranks as absent, for ``max`` too.
    """
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "A", "A", "A", "A"],
                "Ion": ["I1", "I1", "I1", "I2", "I2"],
                "Sameset": ["FLNB", "FLNA", "FLNC", "P2", "P4"],
                "Subset": ["FLNB2", None, "FLNC2", "P3", None],
                "Score": ["4894,2", "6585,4", "NaN", "10", "10"],
                "intensity": [1.0, 1.0, 1.0, 2.0, 2.0],
            }
        )
    )
    comma = NumericTextFormat(decimal_mark=",", thousands_marks=())

    raw = LongSourceDecomposer(
        obs=axis(("run",)),
        var=axis(("Ion",), ("Sameset", "Subset", "Score")),
        var_values=(
            RankedFeatureValue("Sameset", "Score", highest=True, notation=comma),
            RankedFeatureValue("Subset", "Score", highest=True, notation=comma),
            ExtremeFeatureValue("Score", largest=True, notation=comma),
        ),
        primary_layer_name="Intensity",
        layer_sources=(LongRawLayerSource(name="Intensity", source_column="intensity"),),
    ).decompose(table)

    assert raw.var.frame.to_dicts() == [
        {"Ion": "I1", "Sameset": "FLNA", "Subset": None, "Score": "6585,4"},
        {"Ion": "I2", "Sameset": "P2", "Subset": "P3", "Score": "10"},
    ]
    assert raw.var.disagreements == {}, "a declared ranking is no disagreement"


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

    raw = long_decomposer(
        ("Intensity", "intensity"), primary="Intensity", obs=axis(("run",)), var=axis(("Feature",))
    ).decompose(table)

    assert raw.obs.frame.get_column("run").to_list() == ["A", None, "B"]
    assert raw.var.frame.get_column("Feature").to_list() == ["F1", None]
    assert cells_of(raw, "Intensity") == [("F1", "A", 1.0), ("F1", None, 2.0), (None, "B", 3.0)]


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

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run", "fraction")),
        var=axis(("Feature",)),
    ).decompose(table)

    assert raw.obs.raw_key_columns == ("run", "fraction")
    assert raw.obs.frame.height == 2
    assert cells_of(raw, "Intensity") == [("F1", "A", "1", 1.0), ("F1", "A", "2", 2.0)]


def test_long_layers_share_their_row_numbers_and_keep_the_column_as_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_pivot(*_args: object, **_kwargs: object) -> pl.DataFrame:
        raise AssertionError("decomposition widens nothing")

    monkeypatch.setattr(pl.DataFrame, "pivot", no_pivot)
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "B"],
                "Feature": ["F1", "F1"],
                "intensity": [1.0, 2.0],
                "status": ["found", "transferred"],
                "empty": pl.Series([None, None], dtype=pl.Float64),
            }
        )
    )

    raw = long_decomposer(
        ("Intensity", "intensity"),
        ("Intensity_Copy", "intensity"),
        ("Status", "status"),
        ("Empty", "empty"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("Feature",)),
    ).decompose(table)
    layers = {layer.layer_name: layer.rows for layer in raw.layers}

    assert layers["Intensity"].equals(layers["Intensity_Copy"])
    assert all(
        rows.select(VAR_ROW, OBS_ROW).equals(layers["Status"].select(VAR_ROW, OBS_ROW))
        for rows in layers.values()
    )
    assert layers["Status"].get_column(VALUE).to_list() == ["found", "transferred"]
    assert layers["Empty"].schema == {VAR_ROW: pl.UInt32, OBS_ROW: pl.UInt32, VALUE: pl.Float64}


# ------------------------------------------------------------------------- repeated cells


def test_a_repeated_long_cell_becomes_a_second_row_naming_the_same_cell() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "B", "A", "C"],
                "Feature": ["F1", "F1", "F1", "F1"],
                "intensity": [10.0, 20.0, 11.0, 30.0],
            }
        )
    )

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("Feature",)),
    ).decompose(table)

    assert cells_of(raw, "Intensity") == [
        ("F1", "A", 10.0),
        ("F1", "B", 20.0),
        ("F1", "A", 11.0),
        ("F1", "C", 30.0),
    ]


def test_two_wide_columns_claiming_one_sample_repeat_its_cells_header_first() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {"Feature": ["F1", "F2"], "A one": [1.0, 2.0], "A two": [9.0, None], "B": [3.0, 4.0]}
        )
    )
    config = WideSourceDecomposer(
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
        obs=axis(("sample",)),
        var=axis(("Feature",)),
        var_values=(),
    )

    assert cells_of(config.decompose(table), "Intensity") == [
        ("F1", "A", 1.0),
        ("F2", "A", 2.0),
        ("F1", "B", 3.0),
        ("F2", "B", 4.0),
        ("F1", "A", 9.0),
        ("F2", "A", None),
    ]


def test_a_required_wide_layer_with_no_aligned_column_is_kept_without_rows() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Feature": ["F1"], "A": [1.0]}))
    config = WideSourceDecomposer(
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity", sources=(WideRawLayerSource(source_column="A", sample="A"),)
            ),
            WideRawLayerPlan(name="Count", sources=()),
        ),
        obs=axis(("sample",)),
        var=axis(("Feature",)),
        var_values=(),
    )

    raw = config.decompose(table)

    assert cells_of(raw, "Count") == []
    assert [layer.layer_name for layer in raw.layers] == ["Intensity", "Count"]


@pytest.mark.parametrize("repetitions", [0, 1, 20])
@pytest.mark.parametrize("duplicate_headers", [False, True])
def test_wide_keep_first_preserves_file_order_for_equal_keys(
    repetitions: int, duplicate_headers: bool
) -> None:
    table = LevelSourceTable(
        pl.DataFrame(
            {
                "Feature": ["F1", "F2", None] * repetitions,
                "A": list(range(1, repetitions * 3 + 1)),
                "A again": [999] * (repetitions * 3),
            },
            schema={"Feature": pl.String, "A": pl.Int64, "A again": pl.Int64},
        )
    )
    sources = (WideRawLayerSource("A", "A"),)
    if duplicate_headers:
        sources += (WideRawLayerSource("A again", "A"),)
    raw = WideSourceDecomposer(
        primary_layer_name="Intensity",
        layer_plans=(WideRawLayerPlan("Intensity", sources),),
        obs=axis(("sample",)),
        var=axis(("Feature",)),
        var_values=(),
    ).decompose(table)
    positions = FinalPositions(
        var=pl.int_range(raw.var.frame.height, dtype=pl.UInt32, eager=True),
        obs=pl.int_range(raw.obs.frame.height, dtype=pl.UInt32, eager=True),
        var_keys=raw.var.frame,
        grid=ObservationGrid(raw.var.frame.height, ("obs_0",)),
    )
    result = KeepFirstDuplicate().resolve(positions.place(raw.layers[0])).rows
    assert raw.var.frame.get_column("Feature").to_list() == (
        ["F1", "F2", None] if repetitions else []
    )
    assert result.sort(VAR_POS).select(VAR_POS, OBS_POS, VALUE).rows() == (
        [(0, 0, 1), (1, 0, 2), (2, 0, 3)] if repetitions else []
    )


def test_the_wide_observation_axis_comes_from_the_primary_layer_alone() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"Feature": ["F1"], "A Int": [1.0], "Z Count": [2.0]})
    )
    config = WideSourceDecomposer(
        primary_layer_name="Intensity",
        layer_plans=(
            WideRawLayerPlan(
                name="Intensity", sources=(WideRawLayerSource(source_column="A Int", sample="A"),)
            ),
            WideRawLayerPlan(
                name="Count", sources=(WideRawLayerSource(source_column="Z Count", sample="Z"),)
            ),
        ),
        obs=axis(("sample",)),
        var=axis(("Feature",)),
        var_values=(),
    )

    raw = config.decompose(table)

    assert raw.obs.frame.get_column("sample").to_list() == ["A"]
    # A non-primary sample token does not expand the observation axis, so the layer that
    # matched it has no rows.
    assert cells_of(raw, "Count") == []


# ---------------------------------------------------------------------- packed fragments


def positional(*packed: str, delimiter: str = ";") -> PositionalFragmentTableSeparator:
    return PositionalFragmentTableSeparator(
        label_output="fragment_label", delimiter=delimiter, packed_value_sources=packed
    )


def separated(
    separator: PositionalFragmentTableSeparator | ColumnLabeledFragmentTableSeparator,
    table: LevelSourceTable,
) -> pl.DataFrame:
    """Both separation steps, with nothing between them."""
    return separator.exploded(separator.listed(table.frame))


def test_positional_separation_labels_each_scalar_by_its_index() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"Seq": ["P"], "Quant": ["1200;900;450"], "Corr": ["0.9;0.8;0.7"]})
    )

    rows = separated(positional("Quant", "Corr"), table)

    assert rows.to_dicts() == [
        {"Seq": "P", "Quant": "1200", "Corr": "0.9", "fragment_label": "frag_0"},
        {"Seq": "P", "Quant": "900", "Corr": "0.8", "fragment_label": "frag_1"},
        {"Seq": "P", "Quant": "450", "Corr": "0.7", "fragment_label": "frag_2"},
    ]


def test_a_trailing_terminator_is_not_an_extra_fragment() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["10;20;"]}))

    rows = separated(positional("Quant"), table)

    assert rows.get_column("Quant").to_list() == ["10", "20"]


def test_whitespace_is_trimmed_around_the_cell_and_around_each_token() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["  10 ; 20  "]}))

    rows = separated(positional("Quant"), table)

    assert rows.get_column("Quant").to_list() == ["10", "20"]


def test_an_interior_empty_token_is_a_missing_scalar_at_its_position() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["10;;30"]}))

    rows = separated(positional("Quant"), table)

    assert rows.get_column("Quant").to_list() == ["10", None, "30"]
    assert rows.get_column("fragment_label").to_list() == [
        "frag_0",
        "frag_1",
        "frag_2",
    ]


@pytest.mark.parametrize("cell", [None, "", "   ", ";", ";;"])
def test_a_row_with_no_tokens_contributes_no_scalar_row(cell: str | None) -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": [cell, "5"]}, strict=False))

    rows = separated(positional("Quant"), table)

    assert rows.get_column("Quant").to_list() == ["5"]


def test_parallel_packed_cells_of_different_length_are_a_vendor_defect() -> None:
    table = LevelSourceTable(frame=pl.DataFrame({"Quant": ["1;2"], "Corr": ["1;2;3"]}))

    with pytest.raises(PackedLengthError, match="different numbers"):
        positional("Quant", "Corr").listed(table.frame)


def test_column_labelled_separation_takes_the_token_before_the_first_slash() -> None:
    config = ColumnLabeledFragmentTableSeparator(
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

    rows = separated(config, table)

    assert rows.columns == ["Seq", "Quant", "fragment_label"]
    assert rows.get_column("fragment_label").to_list() == ["b4-unknown^1", "y3^1"]


def test_fragments_get_their_row_numbers_before_the_explode() -> None:
    """The explode multiplies integers, the payload and the packed values, never key text."""
    exploded_columns: list[str] = []
    inner = positional("Quant")

    class Recording:
        label_output = inner.label_output
        packed_sources = inner.packed_sources

        def listed(self, frame: pl.DataFrame, /) -> pl.DataFrame:
            return inner.listed(frame)

        def exploded(self, frame: pl.DataFrame, /) -> pl.DataFrame:
            exploded_columns.extend(frame.columns)
            return inner.exploded(frame)

    obs, var = axis(("Run",)), axis(("Seq", "fragment_label"), ("Gene",))
    composed = DelimitedFragmentSourceDecomposer(
        separator=Recording(),
        long_decomposer=long_decomposer(
            ("Fragment_Quant", "Quant"), primary="Fragment_Quant", obs=obs, var=var
        ),
    )
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {"Run": ["A", "B"], "Seq": ["P", "P"], "Gene": ["G", "G"], "Quant": ["1;2", "3;4"]}
        )
    )

    raw = composed.decompose(table)

    assert not {"Run", "Seq"} & set(exploded_columns)
    assert {"Gene", "Quant"} <= set(exploded_columns)
    assert raw.var.frame.to_dicts() == [
        {"Seq": "P", "fragment_label": "frag_0", "Gene": "G"},
        {"Seq": "P", "fragment_label": "frag_1", "Gene": "G"},
    ]


def test_the_fragment_decomposer_reuses_the_ordinary_long_implementation() -> None:
    obs, var = axis(("Run",)), axis(("Seq", "fragment_label"))
    composed = DelimitedFragmentSourceDecomposer(
        separator=positional("Quant"),
        long_decomposer=long_decomposer(
            ("Fragment_Quant", "Quant"), primary="Fragment_Quant", obs=obs, var=var
        ),
    )
    table = LevelSourceTable(
        frame=pl.DataFrame({"Run": ["A", "B"], "Seq": ["P", "P"], "Quant": ["10;20", "30;40"]})
    )

    raw = composed.decompose(table)

    assert isinstance(composed, DelimitedFragmentSourceDecomposer)
    assert type(composed.long_decomposer) is LongSourceDecomposer
    assert raw.var.frame.to_dicts() == [
        {"Seq": "P", "fragment_label": "frag_0"},
        {"Seq": "P", "fragment_label": "frag_1"},
    ]
    assert cells_of(raw, "Fragment_Quant") == [
        ("P", "frag_0", "A", "10"),
        ("P", "frag_1", "A", "20"),
        ("P", "frag_0", "B", "30"),
        ("P", "frag_1", "B", "40"),
    ]


# -------------------------------------------------------------------------- storage labels


def test_positional_labels_step_aside_for_a_vendor_column_of_the_same_name() -> None:
    assert observation_labels(2, reserved=()) == ("obs_0", "obs_1")
    assert observation_labels(2, reserved=("obs_0",)) == ("obs__0", "obs__1")
    assert observation_labels(0, reserved=("obs_0",)) == ()


# ------------------------------------------------------------------- nothing else survives


def test_a_layer_carries_only_its_two_row_numbers_and_the_value() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame({"run": ["A", "A"], "Feature": ["F1", "F1"], "intensity": [1.0, 2.0]})
    )

    raw = long_decomposer(
        ("Intensity", "intensity"), primary="Intensity", obs=axis(("run",)), var=axis(("Feature",))
    ).decompose(table)

    assert raw.obs.frame.columns == ["run"]
    assert raw.var.frame.columns == ["Feature"]
    for layer in raw.layers:
        assert layer.rows.schema == {VAR_ROW: pl.UInt32, OBS_ROW: pl.UInt32, VALUE: pl.Float64}


def test_a_composite_var_key_names_one_raw_var_row() -> None:
    table = LevelSourceTable(
        frame=pl.DataFrame(
            {
                "run": ["A", "A", "B"],
                "Feature": ["F1", "F1", "F1"],
                "Charge": ["2", "3", "2"],
                "intensity": [1.0, 2.0, 3.0],
            }
        )
    )

    raw = long_decomposer(
        ("Intensity", "intensity"),
        primary="Intensity",
        obs=axis(("run",)),
        var=axis(("Feature", "Charge")),
    ).decompose(table)

    assert raw.var.frame.rows() == [("F1", "2"), ("F1", "3")]
    assert raw.layers[0].rows.get_column(VAR_ROW).to_list() == [0, 1, 0]
