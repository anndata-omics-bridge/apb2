"""The parser algorithm: call order, identity, validity, and layer alignment.

The tests are deliberately built from fakes where order is the property under test, and from
real configured strategies where the result is. What they protect is the separation the
architecture is for: a repeated cell is the duplicate policy's question, a collapsed identity
is an error, an incomplete identity is a filter, and none of the three is allowed to be
mistaken for another.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2.compile import (
    make_raw_value_presence,
    make_source_decomposer,
    policy_for,
)
from apb2.parserV2.parse_quant.columns import (
    CoalesceColumn,
    IntegerAxisCoercer,
    ProformaIonColumn,
    StringAxisCoercer,
)
from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    ColumnComputer,
    ParsedLevelWriter,
    RawValuePresence,
    SelectedAxisColumn,
)
from apb2.parserV2.parse_quant.data.parsed import ObsFinal, ParsedLevel, VarFinal
from apb2.parserV2.parse_quant.data.raw import (
    DecomposedDataRaw,
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.duplicates import DuplicateCellError
from apb2.parserV2.parse_quant.parameters.axis import AxisKeyPlan, AxisSourcePlan
from apb2.parserV2.parse_quant.parameters.measurements import (
    DuplicateMode,
    NullOnlyRawValuePresenceConfig,
    PlainNumericRawValuePresenceConfig,
)
from apb2.parserV2.parse_quant.parameters.source import (
    LongDecompositionConfig,
    LongRawLayerSource,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.parser import (
    AxisShapeError,
    CanonicalKeyCollisionError,
    Parser,
)

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
NULL_ONLY = make_raw_value_presence(
    NullOnlyRawValuePresenceConfig(kind="null_only", layer_name="Intensity")
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
        coercer=IntegerAxisCoercer() if integer else StringAxisCoercer(),
    )


class Writer:
    """A writer that records what it was handed, and complains if parse called it."""

    def __init__(self) -> None:
        self.written: list[tuple[ParsedLevel, Path]] = []

    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        self.written.append((parsed, target))


def parser_for(
    frame: pl.DataFrame,
    *,
    obs_plan: AxisRuntimePlan,
    var_plan: AxisRuntimePlan,
    obs: AxisSourcePlan,
    var: AxisSourcePlan,
    layers: tuple[tuple[str, str], ...] = (("Intensity", "intensity"),),
    duplicates: DuplicateMode = "error",
    presence: Mapping[str, RawValuePresence] | None = None,
    writer: ParsedLevelWriter | None = None,
) -> Parser:
    class Reader:
        def read(self) -> LevelSourceTable:
            return LevelSourceTable(frame=frame)

    config = LongDecompositionConfig(
        kind="long",
        primary_layer_name=layers[0][0],
        layer_sources=tuple(
            LongRawLayerSource(name=name, source_column=column) for name, column in layers
        ),
    )
    return Parser(
        level="ion",
        input_reader=Reader(),
        decomposer=make_source_decomposer(config, obs, var),
        obs_plan=obs_plan,
        var_plan=var_plan,
        modification_normalizers=(),
        duplicates=policy_for(duplicates),
        raw_value_presence=presence or {name: NULL_ONLY for name, _ in layers},
        writer=writer or Writer(),
        provenance={"software_name": "Synthetic", "quantification_level": "ion"},
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
        sources: tuple[str, ...] = ("feature",)

        def normalize(self, columns: tuple[pl.Series, ...], /) -> dict[str, pl.Series]:
            calls.append("normalize")
            return {}

    class Presence:
        def present(self, values: pl.Series, /) -> pl.Series:
            calls.append("present")
            return values.is_not_null()

    class Policy:
        def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
            calls.append("resolve")
            presence.present(layer.values.get_column("obs_0"))
            return layer

    parser = Parser(
        level="ion",
        input_reader=Reader(),
        decomposer=Decomposer(),
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        modification_normalizers=(Normalizer(),),
        duplicates=Policy(),
        raw_value_presence={"Intensity": Presence()},
        writer=Writer(),
        provenance={},
    )

    parsed = parser.parse()

    assert calls == ["read", "decompose", "normalize", "resolve", "present"]
    assert parsed.primary_layer_name == "Intensity"


def test_convert_writes_the_result_it_is_given_and_parses_nothing(tmp_path: Path) -> None:
    calls: list[str] = []

    class Reader:
        def read(self) -> LevelSourceTable:
            calls.append("read")
            raise AssertionError("convert must not read")

    class Decomposer:
        def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
            calls.append("decompose")
            raise AssertionError("convert must not decompose")

    writer = Writer()
    parser = Parser(
        level="ion",
        input_reader=Reader(),
        decomposer=Decomposer(),
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        modification_normalizers=(),
        duplicates=policy_for("error"),
        raw_value_presence={},
        writer=writer,
        provenance={},
    )
    parsed = ParsedLevel(
        obs=ObsFinal(frame=pl.DataFrame({"Run": ["A"]}), key_columns=("Run",)),
        var=VarFinal(frame=pl.DataFrame({"Feature": ["F1"]}), key_columns=("Feature",)),
        primary_layer_name="Intensity",
        uns={},
        layers={},
    )

    parser.convert(parsed, tmp_path / "out")

    assert calls == []
    assert writer.written == [(parsed, tmp_path / "out")]


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
    assert parsed.layers["Intensity"].var_key_columns == ("Feature",)
    assert parsed.layers["Intensity"].values.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0},
        {"Feature": "F2", "obs_0": 2.0, "obs_1": 4.0},
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
    assert parsed.layers["Intensity"].values.get_column("ProForma_ion").to_list() == [
        "PEP/2",
        "OTH/3",
    ]


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


@pytest.mark.parametrize("mode", ["error", "keep_first", "aggregate"])
def test_a_canonical_collision_is_reported_under_every_duplicate_policy(
    mode: DuplicateMode,
) -> None:
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

    with pytest.raises(CanonicalKeyCollisionError):
        parser_for(
            frame,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=var_plan,
            obs=SIMPLE_OBS,
            var=var,
            duplicates=mode,
        ).parse()


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
    assert kept.layers["Intensity"].values.to_dicts() == [{"Feature": "F1", "obs_0": 1.0}]


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
    assert parsed.layers["Intensity"].values.to_dicts() == [{"Mass": 1.5, "obs_0": 1.0}]


class _LenientNumber:
    """A number coercion that admits NaN, so the parser's own normalization is visible."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        del name, source
        return values.cast(pl.Float64, strict=False)


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
    assert parsed.layers["Intensity"].values.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0}
    ]


class _BlankToNull:
    """A string coercion that reads a blank cell as the absence it is."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        del name, source
        text = values.cast(pl.String)
        return pl.select(
            pl.when(text.is_null() | (text == "")).then(None).otherwise(text)
        ).to_series()


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
    assert parsed.layers["Intensity"].values.columns == ["Feature", "obs_0", "obs_1"]
    assert parsed.layers["Intensity"].values.to_dicts() == [
        {"Feature": "F1", "obs_0": 1.0, "obs_1": 3.0}
    ]


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
    presence = {
        "Intensity": NULL_ONLY,
        "Score": make_raw_value_presence(
            PlainNumericRawValuePresenceConfig(
                kind="plain_numeric", layer_name="Score", missing_values=(), number_format=DOT
            )
        ),
    }

    parsed = parser_for(
        frame,
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        obs=SIMPLE_OBS,
        var=SIMPLE_VAR,
        layers=(("Intensity", "intensity"), ("Score", "score")),
        presence=presence,
    ).parse()

    assert list(parsed.layers) == ["Intensity", "Score"]
    assert parsed.layers["Score"].values.to_dicts() == [
        {"Feature": "F1", "obs_0": 0.5},
        {"Feature": "F2", "obs_0": None},
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
    assert parsed.layers["Intensity"].values.columns == ["Feature", "obs_0", "obs_1"]


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
    }
    assert parsed.uns == {"software_name": "Synthetic", "quantification_level": "ion"}
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

    assert set(Parser.__slots__) == {
        "level",
        "_input",
        "_decomposer",
        "_obs_plan",
        "_var_plan",
        "_modification_normalizers",
        "_duplicates",
        "_raw_value_presence",
        "_writer",
        "_provenance",
    }
    assert parser.level == "ion"
    assert not hasattr(parser, "_output")
    assert not hasattr(parser, "_rule")
    assert not hasattr(parser, "_registry")


# ---------------------------------------------------------------------- collaborator shape


def test_a_coercion_that_changes_the_row_count_fails_at_the_boundary() -> None:
    class Shrinking:
        def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
            del name, source
            return values.head(1)

    var_plan = AxisRuntimePlan(
        keys=SIMPLE_VAR.keys,
        key_phase=phase(
            (SelectedAxisColumn(name="Feature", source="feature", coercer=Shrinking()),)
        ),
        output_phase=phase(),
        outputs=("Feature",),
    )

    with pytest.raises(AxisShapeError, match="row order"):
        parser_for(
            SIMPLE_FRAME,
            obs_plan=SIMPLE_OBS_PLAN,
            var_plan=var_plan,
            obs=SIMPLE_OBS,
            var=SIMPLE_VAR,
        ).parse()


def test_a_derived_column_of_the_wrong_length_fails_at_the_boundary() -> None:
    class Shrinking:
        sources: tuple[str, ...] = ("feature",)

        def normalize(self, columns: tuple[pl.Series, ...], /) -> dict[str, pl.Series]:
            return {"proforma_sequence": columns[0].head(1)}

    parser = Parser(
        level="ion",
        input_reader=_ReaderOf(SIMPLE_FRAME),
        decomposer=make_source_decomposer(
            LongDecompositionConfig(
                kind="long",
                primary_layer_name="Intensity",
                layer_sources=(LongRawLayerSource(name="Intensity", source_column="intensity"),),
            ),
            SIMPLE_OBS,
            SIMPLE_VAR,
        ),
        obs_plan=SIMPLE_OBS_PLAN,
        var_plan=SIMPLE_VAR_PLAN,
        modification_normalizers=(Shrinking(),),
        duplicates=policy_for("error"),
        raw_value_presence={"Intensity": NULL_ONLY},
        writer=Writer(),
        provenance={},
    )

    with pytest.raises(AxisShapeError, match="derived column"):
        parser.parse()


class _ReaderOf:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def read(self) -> LevelSourceTable:
        return LevelSourceTable(frame=self._frame)
