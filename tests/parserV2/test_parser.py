"""The parser algorithm: call order, identity, validity, and layer alignment.

The tests are deliberately built from fakes where order is the property under test, and from
real configured strategies where the result is. What they protect is the separation the
architecture is for: a repeated cell is the duplicate policy's question, a collapsed identity
is an error, an incomplete identity is a filter, and none of the three is allowed to be
mistaken for another.
"""

from __future__ import annotations

from dataclasses import replace
from typing import get_args

import polars as pl
import pytest

from apb2.parserV2.parse_quant.axis_columns import (
    CoalesceColumn,
    IntegerAxisCoercer,
    ProformaIonColumn,
    StringAxisCoercer,
)
from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    ColumnComputer,
    LayerSetValidator,
    LayerValueParser,
    RawValuePresence,
    SelectedAxisColumn,
)
from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, ParsedLevel
from apb2.parserV2.parse_quant.data.raw import (
    DecomposedDataRaw,
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.decomposition import LongSourceDecomposer
from apb2.parserV2.parse_quant.duplicates import DuplicateCellError, KeepBestDuplicate
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.operations import (
    duplicate_policy_for,
    make_layer_parser,
)
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan, AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    PlainNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import (
    LevelReadPlan,
    LongRawLayerSource,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.parser import (
    CanonicalKeyCollisionError,
    Parser,
    ParseStrategy,
)

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
DOT_NUMBERS = NumericTextFormat(decimal_mark=".", thousands_marks=())


def numeric_layer_parser(name: str) -> LayerValueParser:
    return make_layer_parser(name, PlainNumericLayerDeclaration(missing_values=()), DOT)


def layer_validator(primary: str) -> LayerSetValidator:
    return LayerContractValidator(
        primary_layer_name=primary,
        required_names=(primary,),
        empty_ratio=0.001,
        populated_ratio=0.5,
        strict=False,
    )


def axis_source(
    raw: tuple[str, ...],
    inputs: tuple[str, ...],
    final: tuple[str, ...],
    payload: tuple[str, ...] = (),
) -> AxisSourcePlan:
    return AxisSourcePlan(
        keys=AxisKeyPlan(raw_key_columns=raw, key_input_columns=inputs, final_key_columns=final),
        payload_sources=payload,
    )


def phase(
    selections: tuple[SelectedAxisColumn, ...] = (),
    computers: tuple[ColumnComputer, ...] = (),
) -> AxisPhaseRuntimePlan:
    return AxisPhaseRuntimePlan(selections=selections, computers=computers)


def selected(name: str, source: str, *, integer: bool = False) -> SelectedAxisColumn:
    return SelectedAxisColumn(
        name=name,
        source=source,
        coercer=IntegerAxisCoercer(DOT_NUMBERS) if integer else StringAxisCoercer(),
    )


def parser_for(
    frame: pl.DataFrame,
    *,
    obs_plan: AxisRuntimePlan,
    var_plan: AxisRuntimePlan,
    obs: AxisSourcePlan,
    var: AxisSourcePlan,
    layers: tuple[tuple[str, str], ...] = (("Intensity", "intensity"),),
    duplicates: DuplicateMode = "error",
    rank_by: str | None = None,
) -> Parser:
    class Reader:
        def read(self) -> LevelSourceTable:
            return LevelSourceTable(frame=frame)

    config = LongSourceDecomposer(
        primary_layer_name=layers[0][0],
        layer_sources=tuple(
            (LongRawLayerSource(name=name, source_column=column) for name, column in layers)
        ),
        obs=obs,
        var=var,
    )
    return Parser(
        input_reader=Reader(),
        strategy=ParseStrategy(
            level="ion",
            decomposer=config,
            obs=obs_plan,
            var=var_plan,
            duplicates=(
                KeepBestDuplicate(by=rank_by or layers[0][0], highest=True)
                if duplicates == "keep_best"
                else duplicate_policy_for(duplicates)
            ),
            layer_parsers={name: numeric_layer_parser(name) for name, _source in layers},
            layer_validator=layer_validator(layers[0][0]),
            provenance={
                "hierarchy": "lfq",
                "software_name": "Synthetic",
                "quantification_level": "ion",
            },
            read=LevelReadPlan((), frozenset(), frozenset()),
        ),
    )


SIMPLE_OBS = axis_source(("run",), ("Run",), ("Run",))
SIMPLE_VAR = axis_source(("feature",), ("Feature",), ("Feature",))
SIMPLE_OBS_PLAN = AxisRuntimePlan(
    keys=SIMPLE_OBS.keys,
    key_phase=phase((selected("Run", "run"),)),
    output_phase=phase(),
    outputs=("Run",),
)
SIMPLE_VAR_PLAN = AxisRuntimePlan(
    keys=SIMPLE_VAR.keys,
    key_phase=phase((selected("Feature", "feature"),)),
    output_phase=phase(),
    outputs=("Feature",),
)
SIMPLE_FRAME = pl.DataFrame(
    {
        "run": ["A", "A", "B", "B"],
        "feature": ["F1", "F2", "F1", "F2"],
        "intensity": [1.0, 2.0, 3.0, 4.0],
    }
)


# ----------------------------------------------------------------------------- call order


def test_parse_runs_its_collaborators_in_the_documented_order() -> None:
    calls: list[str] = []
    raw = DecomposedDataRaw(
        obs=ObsRaw(frame=pl.DataFrame({"run": ["A"]}), raw_key_columns=("run",)),
        var=VarRaw(frame=pl.DataFrame({"feature": ["F1"]}), raw_key_columns=("feature",)),
        layers=LayersRaw(
            primary_layer_name="Intensity",
            values=(
                RawLayerTable(
                    layer_name="Intensity",
                    raw_var_key_columns=("feature",),
                    values=pl.DataFrame({"feature": ["F1"], "obs_0": [1.0]}),
                ),
            ),
        ),
    )

    class Reader:
        def read(self) -> LevelSourceTable:
            calls.append("read")
            return LevelSourceTable(frame=pl.DataFrame({"a": [1]}))

    class Decomposer:
        def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
            calls.append("decompose")
            return raw

    class Normalizer:
        name = "Feature"
        inputs: tuple[str, ...] = ("Feature",)

        def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
            calls.append("normalize")
            return frame, ("Mystery@M", "Mystery@M", "Other@C")

    class Values:
        def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
            del dtype
            calls.append("present")
            return values.is_not_null()

        def parse(self, layer: FinalLayerTable, /) -> FinalLayerTable:
            calls.append("parse_values")
            return numeric_layer_parser("Intensity").parse(layer)

    class Policy:
        def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
            calls.append("resolve")
            presence.present(pl.col("obs_0"), layer.values.schema["obs_0"])
            return layer

    parser = Parser(
        input_reader=Reader(),
        strategy=ParseStrategy(
            level="ion",
            decomposer=Decomposer(),
            obs=SIMPLE_OBS_PLAN,
            var=replace(
                SIMPLE_VAR_PLAN,
                key_phase=phase(SIMPLE_VAR_PLAN.key_phase.selections, (Normalizer(),)),
            ),
            duplicates=Policy(),
            layer_parsers={"Intensity": Values()},
            layer_validator=layer_validator("Intensity"),
            provenance={},
            read=LevelReadPlan((), frozenset(), frozenset()),
        ),
    )

    parsed = parser.parse()

    assert calls == [
        "read",
        "decompose",
        "normalize",
        "resolve",
        "present",
        "parse_values",
    ]
    assert parsed.primary_layer_name == "Intensity"
    assert parsed.uns["unknown_mod_tokens"] == ["Mystery@M", "Other@C"]


# ------------------------------------------------------------------------------- identity


def test_a_simple_parse_produces_both_axes_and_one_aligned_layer() -> None:
    parsed = parser_for(
        SIMPLE_FRAME,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
    ).parse()

    assert parsed.obs.frame.to_dicts() == [{"Run": "A"}, {"Run": "B"}]
    assert parsed.var.frame.to_dicts() == [{"Feature": "F1"}, {"Feature": "F2"}]
    assert parsed.layers["Intensity"].values.width == parsed.obs.frame.height
    assert parsed.layers["Intensity"].values.to_dicts() == [
        {"obs_0": 1.0, "obs_1": 3.0},
        {"obs_0": 2.0, "obs_1": 4.0},
    ]


def test_a_computed_key_is_materialized_before_identity_is_checked() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "seq": ["PEP", "OTH"],
            "z": ["2", "3"],
            "gene": ["G1", "G2"],
            "intensity": [1.0, 2.0],
        }
    )
    var = axis_source(("seq", "z"), ("Sequence", "Charge"), ("ProForma_ion",), ("gene",))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (selected("Sequence", "seq"), selected("Charge", "z", integer=True)),
            (ProformaIonColumn(name="ProForma_ion", inputs=("Sequence", "Charge")),),
        ),
        output_phase=phase((selected("Gene", "gene"),)),
        outputs=("ProForma_ion", "Sequence", "Charge", "Gene"),
    )

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=var_plan,
        obs=SIMPLE_OBS,
        var=var,
    ).parse()

    assert parsed.var.key_columns == ("ProForma_ion",)
    assert parsed.var.frame.columns == ["ProForma_ion", "Sequence", "Charge", "Gene"]
    assert parsed.var.frame.get_column("ProForma_ion").to_list() == ["PEP/2", "OTH/3"]
    assert parsed.layers["Intensity"].values.rows() == [(1.0,), (2.0,)]


def test_two_raw_identities_collapsing_into_one_final_key_are_reported() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "first": ["K", None],
            "second": [None, "K"],
            "intensity": [1.0, 2.0],
        }
    )
    var = axis_source(("first", "second"), ("First", "Second"), ("Key",))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (selected("First", "first"), selected("Second", "second")),
            (CoalesceColumn(name="Key", inputs=("First", "Second")),),
        ),
        output_phase=phase(),
        outputs=("Key",),
    )

    with pytest.raises(CanonicalKeyCollisionError, match="more than one raw identity"):
        parser_for(
            frame,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=var_plan,
            obs=SIMPLE_OBS,
            var=var,
        ).parse()


def test_an_injective_coalesce_of_the_same_shape_parses_normally() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "first": ["K1", None],
            "second": [None, "K2"],
            "intensity": [1.0, 2.0],
        }
    )
    var = axis_source(("first", "second"), ("First", "Second"), ("Key",))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (selected("First", "first"), selected("Second", "second")),
            (CoalesceColumn(name="Key", inputs=("First", "Second")),),
        ),
        output_phase=phase(),
        outputs=("Key",),
    )

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=var_plan,
        obs=SIMPLE_OBS,
        var=var,
    ).parse()

    assert parsed.var.frame.get_column("Key").to_list() == ["K1", "K2"]


@pytest.mark.parametrize("mode", get_args(DuplicateMode.__value__))
@pytest.mark.parametrize("key_name", ["Key", "first"])
def test_a_canonical_collision_is_reported_under_every_duplicate_policy(
    mode: DuplicateMode,
    key_name: str,
) -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "first": ["K", None],
            "second": [None, "K"],
            "intensity": [1.0, 2.0],
        }
    )
    var = axis_source(("first", "second"), ("First", "Second"), (key_name,))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (selected("First", "first"), selected("Second", "second")),
            (CoalesceColumn(name=key_name, inputs=("First", "Second")),),
        ),
        output_phase=phase(),
        outputs=(key_name,),
    )

    with pytest.raises(CanonicalKeyCollisionError, match="1 value") as error:
        parser_for(
            frame,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=var_plan,
            obs=SIMPLE_OBS,
            var=var,
            duplicates=mode,
        ).parse()
    assert f"'final': {{'{key_name}': 'K'}}" in str(error.value)
    assert "'raw': {'first': 'K', 'second': None}" in str(error.value)
    assert "'raw': {'first': None, 'second': 'K'}" in str(error.value)


def test_a_repeated_raw_key_reaches_the_duplicate_policy_instead() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "feature": ["F1", "F1"],
            "intensity": [1.0, 2.0],
        }
    )

    with pytest.raises(DuplicateCellError):
        parser_for(
            frame,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=SIMPLE_VAR_PLAN,
            obs=SIMPLE_OBS,
            var=SIMPLE_VAR,
        ).parse()

    kept = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
        duplicates="keep_first",
    ).parse()
    assert kept.layers["Intensity"].values.to_dicts() == [{"obs_0": 1.0}]


def test_keep_best_keeps_one_psm_per_cell_across_layers() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A", "B", "A"],
            "feature": ["F1", "F1", "F1", "F2"],
            "intensity": [1.0, 2.0, 3.0, 4.0],
            "score": [9.0, 5.0, 1.0, 2.0],
        }
    )

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
        layers=(("Intensity", "intensity"), ("Score", "score")),
        duplicates="keep_best",
        rank_by="Score",
    ).parse()

    assert parsed.layers["Intensity"].values.to_dicts() == [
        {"obs_0": 1.0, "obs_1": 3.0},
        {"obs_0": 4.0, "obs_1": None},
    ], "F1 in A keeps the PSM scored 9, not the first or the larger intensity"
    assert parsed.layers["Score"].values.to_dicts() == [
        {"obs_0": 9.0, "obs_1": 1.0},
        {"obs_0": 2.0, "obs_1": None},
    ]


def test_a_nan_key_is_the_same_absence_as_a_null_key() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "mass": ["1.5", "nan"],
            "intensity": [1.0, 2.0],
        }
    )
    var = axis_source(("mass",), ("Mass",), ("Mass",))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (
                SelectedAxisColumn(
                    name="Mass",
                    source="mass",
                    coercer=_LenientNumber(),
                ),
            )
        ),
        output_phase=phase(),
        outputs=("Mass",),
    )

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=var_plan,
        obs=SIMPLE_OBS,
        var=var,
    ).parse()

    assert parsed.var.frame.get_column("Mass").to_list() == [1.5]
    assert parsed.layers["Intensity"].values.to_dicts() == [{"obs_0": 1.0}]


class _LenientNumber:
    """A number coercion that admits NaN, so the parser's own normalization is visible."""

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        return pl.col(source).cast(pl.Float64, strict=False).alias(name)


# ------------------------------------------------------------------------------- validity


def test_an_incomplete_final_key_removes_its_axis_row_and_its_layer_cells() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A", "B"],
            "feature": ["F1", "", "F1"],
            "intensity": [1.0, 2.0, 3.0],
        }
    )
    var = axis_source(("feature",), ("Feature",), ("Feature",))
    var_plan = AxisRuntimePlan(
        keys=var.keys,
        key_phase=phase(
            (SelectedAxisColumn(name="Feature", source="feature", coercer=_BlankToNull()),)
        ),
        output_phase=phase(),
        outputs=("Feature",),
    )

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=var_plan,
        obs=SIMPLE_OBS,
        var=var,
    ).parse()

    assert parsed.var.frame.to_dicts() == [{"Feature": "F1"}]
    assert parsed.layers["Intensity"].values.to_dicts() == [{"obs_0": 1.0, "obs_1": 3.0}]


class _BlankToNull:
    """A string coercion that reads a blank cell as the absence it is."""

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        text = pl.col(source).cast(pl.String)
        return pl.when(text.is_null() | (text == "")).then(None).otherwise(text).alias(name)


def test_an_observation_whose_key_is_incomplete_loses_its_value_column() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "", "B"],
            "feature": ["F1", "F1", "F1"],
            "intensity": [1.0, 2.0, 3.0],
        }
    )
    obs = axis_source(("run",), ("Run",), ("Run",))
    obs_plan = AxisRuntimePlan(
        keys=obs.keys,
        key_phase=phase((SelectedAxisColumn(name="Run", source="run", coercer=_BlankToNull()),)),
        output_phase=phase(),
        outputs=("Run",),
    )

    parsed = parser_for(
        frame,
        obs_plan=obs_plan,
        var_plan=SIMPLE_VAR_PLAN,
        obs=obs,
        var=SIMPLE_VAR,
    ).parse()

    assert parsed.obs.frame.to_dicts() == [{"Run": "A"}, {"Run": "B"}]
    assert parsed.layers["Intensity"].values.rows() == [(1.0, 3.0)]
    assert parsed.layers["Intensity"].values.to_dicts() == [{"obs_0": 1.0, "obs_1": 3.0}]


# ---------------------------------------------------------------------- layers and results


def test_a_final_variable_a_layer_never_measured_becomes_a_row_of_nulls() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "feature": ["F1", "F2"],
            "intensity": [1.0, 2.0],
            "score": [0.5, None],
        }
    )
    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
        layers=(("Intensity", "intensity"), ("Score", "score")),
    ).parse()

    assert list(parsed.layers) == ["Intensity", "Score"]
    assert parsed.layers["Score"].values.to_dicts() == [
        {"obs_0": 0.5},
        {"obs_0": None},
    ]


def test_a_composite_observation_identity_stays_in_the_axis_not_in_a_column_name() -> None:
    frame = pl.DataFrame(
        {
            "run": ["A", "A"],
            "fraction": ["1", "2"],
            "feature": ["F1", "F1"],
            "intensity": [1.0, 2.0],
        }
    )
    obs = axis_source(("run", "fraction"), ("Run", "Fraction"), ("Run", "Fraction"))
    obs_plan = AxisRuntimePlan(
        keys=obs.keys,
        key_phase=phase((selected("Run", "run"), selected("Fraction", "fraction"))),
        output_phase=phase(),
        outputs=("Run", "Fraction"),
    )

    parsed = parser_for(
        frame,
        obs_plan=obs_plan,
        var_plan=SIMPLE_VAR_PLAN,
        obs=obs,
        var=SIMPLE_VAR,
    ).parse()

    assert parsed.obs.key_columns == ("Run", "Fraction")
    assert parsed.obs.frame.to_dicts() == [
        {"Run": "A", "Fraction": "1"},
        {"Run": "A", "Fraction": "2"},
    ]
    # The value columns are positions, so nothing concatenated "A" and "1" into a name.
    assert parsed.layers["Intensity"].values.rows() == [(1.0, 2.0)]


def test_a_parsed_level_is_a_direct_composition_and_keeps_no_key_map() -> None:
    parsed = parser_for(
        SIMPLE_FRAME,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
    ).parse()

    assert set(ParsedLevel.__slots__) == {
        "obs",
        "var",
        "primary_layer_name",
        "uns",
        "layers",
        "obsm",
        "varm",
        "obsp",
        "varp",
        "metadata",
    }
    assert parsed.uns == {
        "software_name": "Synthetic",
        "quantification_level": "ion",
    }
    assert isinstance(parsed.obs.frame, pl.DataFrame)
    assert isinstance(parsed.layers["Intensity"].values, pl.DataFrame)


def test_no_parse_result_holds_a_matrix_a_pandas_index_or_an_encoded_layer() -> None:
    parsed = parser_for(
        SIMPLE_FRAME,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
    ).parse()

    for frame in (
        parsed.obs.frame,
        parsed.var.frame,
        *(layer.values for layer in parsed.layers.values()),
    ):
        assert isinstance(frame, pl.DataFrame)
        assert type(frame).__module__.startswith("polars")
    assert not hasattr(parsed, "X")
    assert "X" not in parsed.layers


def test_the_parser_holds_only_configured_behaviour() -> None:
    parser = parser_for(
        SIMPLE_FRAME,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
    )

    assert set(Parser.__slots__) == {"input_reader", "strategy"}
    assert set(ParseStrategy.__slots__) == {
        "level",
        "read",
        "decomposer",
        "obs",
        "var",
        "duplicates",
        "layer_parsers",
        "layer_validator",
        "provenance",
    }
    assert parser.level == "ion"
    assert not hasattr(parser, "_output")
    assert not hasattr(parser, "_rule")
    assert not hasattr(parser, "_registry")


# ---------------------------------------------------------------------- collaborator shape


def test_a_coercion_that_changes_the_row_count_fails_at_the_boundary() -> None:
    class Shrinking:
        def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
            return pl.col(source).head(1).alias(name)

    var_plan = AxisRuntimePlan(
        keys=SIMPLE_VAR.keys,
        key_phase=phase(
            (SelectedAxisColumn(name="Feature", source="feature", coercer=Shrinking()),)
        ),
        output_phase=phase(),
        outputs=("Feature",),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        parser_for(
            SIMPLE_FRAME,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=var_plan,
            obs=SIMPLE_OBS,
            var=SIMPLE_VAR,
        ).parse()


def test_a_computed_column_of_the_wrong_length_fails_at_the_boundary() -> None:
    class Shrinking:
        name = "Feature"
        inputs: tuple[str, ...] = ("Feature",)

        def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
            return frame.with_columns(pl.col(self.inputs[0]).head(1).alias(self.name)), ()

    parser = Parser(
        input_reader=_ReaderOf(SIMPLE_FRAME),
        strategy=ParseStrategy(
            level="ion",
            decomposer=LongSourceDecomposer(
                primary_layer_name="Intensity",
                layer_sources=(LongRawLayerSource(name="Intensity", source_column="intensity"),),
                obs=SIMPLE_OBS,
                var=SIMPLE_VAR,
            ),
            obs=SIMPLE_OBS_PLAN,
            var=replace(
                SIMPLE_VAR_PLAN,
                key_phase=phase(SIMPLE_VAR_PLAN.key_phase.selections, (Shrinking(),)),
            ),
            duplicates=duplicate_policy_for("error"),
            layer_parsers={"Intensity": numeric_layer_parser("Intensity")},
            layer_validator=layer_validator("Intensity"),
            provenance={},
            read=LevelReadPlan((), frozenset(), frozenset()),
        ),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        parser.parse()


class _ReaderOf:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def read(self) -> LevelSourceTable:
        return LevelSourceTable(frame=self._frame)
