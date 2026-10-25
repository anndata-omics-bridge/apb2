"""The parsing-owned vocabulary: pipeline data, parameters, and client-owned contracts.

One cohesive module rather than one file per class: what is worth asserting about these
values is that identity is explicit, that a raw and a final layer are two unrelated states,
and that the runtime plans accept the collaborators the compiler will inject.
"""

from __future__ import annotations

import polars as pl
import pytest

from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    AxisValueCoercer,
    BoundInputReader,
    ColumnComputer,
    DuplicatePolicy,
    FragmentTableSeparator,
    SelectedAxisColumn,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.data.layer_columns import ObservationGrid
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    LevelHierarchy,
    ObsFinal,
    ParsedLevel,
    VarFinal,
)
from apb2.parserV2.parse_quant.data.raw import (
    OBS_POS,
    OBS_ROW,
    VALUE,
    VAR_POS,
    VAR_ROW,
    FinalPositions,
    LongLayer,
    ObsRaw,
    RawLevel,
    VarRaw,
)
from apb2.parserV2.parse_quant.operations import (
    RetainedFields,
    WorkingAxisConfiguration,
    WorkingParseConfiguration,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnSelection,
    AxisKeyPlan,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    PlainNumericLayerDeclaration,
    WorkingMeasurementLayer,
    WorkingMeasurements,
)
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentLayout,
    DelimitedFormatContract,
    DelimitedSourceEvidence,
    InputContract,
    LevelReadPlan,
    LevelSourceTable,
    LongSourceLayout,
    NumericTextFormat,
    PositionalFragmentLayout,
    SourceLayoutDeclaration,
    WideRawLayerPlan,
    WideRawLayerSource,
    WideSourceLayout,
)

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())


# ------------------------------------------------------------------------------ pipeline data


def test_a_single_column_identity_names_its_own_columns() -> None:
    obs = ObsRaw(frame=pl.DataFrame({"sample": ["A", "B"]}), raw_key_columns=("sample",))

    assert obs.raw_key_columns == ("sample",)
    assert list(obs.frame.columns) == ["sample"]


def test_a_multi_column_identity_states_every_key_in_authored_order() -> None:
    var = VarRaw(
        frame=pl.DataFrame(
            {
                "sequence": ["PEPMIDE", "OTHER"],
                "mods": ["Oxidation@M", None],
                "charge": ["2", "3"],
                "genes": ["GENE1", "GENE2"],
            }
        ),
        raw_key_columns=("sequence", "mods", "charge"),
    )

    assert var.raw_key_columns == ("sequence", "mods", "charge")
    # Payload metadata rides along without becoming identity.
    assert "genes" in var.frame.columns
    assert "genes" not in var.raw_key_columns


def test_a_long_layer_keeps_repeated_cells_and_splits_them_off_once_placed() -> None:
    rows = pl.DataFrame(
        {VAR_POS: [0, 0, 1], OBS_POS: [0, 0, 0], VALUE: [100.0, 110.0, 50.0]},
        schema={VAR_POS: pl.UInt32, OBS_POS: pl.UInt32, VALUE: pl.Float64},
    )

    once, repeated = LongLayer(layer_name="Intensity", rows=rows).split()

    # Repeated raw cells survive decomposition for the duplicate policy to resolve.
    assert repeated.rows.get_column(VALUE).to_list() == [100.0, 110.0]
    assert once.rows.get_column(VALUE).to_list() == [50.0]


def test_final_positions_drop_the_rows_of_axis_rows_that_did_not_survive() -> None:
    positions = FinalPositions(
        var=pl.Series([0, None, 1], dtype=pl.UInt32),
        obs=pl.Series([None, 0], dtype=pl.UInt32),
        var_keys=pl.DataFrame({"feature": ["F1", "F3"]}),
        grid=ObservationGrid(height=2, labels=("obs_0",)),
    )
    layer = LongLayer(
        "Intensity",
        pl.DataFrame(
            {VAR_ROW: [0, 1, 2, 2], OBS_ROW: [1, 1, 1, 0], VALUE: [1.0, 2.0, 3.0, 4.0]},
            schema={VAR_ROW: pl.UInt32, OBS_ROW: pl.UInt32, VALUE: pl.Float64},
        ),
    )

    placed = positions.place(layer)

    assert placed.rows.rows() == [(0, 0, 1.0), (1, 0, 3.0)]
    assert placed.var_keys is positions.var_keys


def test_decomposition_reduces_every_physical_shape_to_one_raw_contract() -> None:
    raw = RawLevel(
        obs=ObsRaw(frame=pl.DataFrame({"sample": ["A"]}), raw_key_columns=("sample",)),
        var=VarRaw(frame=pl.DataFrame({"feature": ["F"]}), raw_key_columns=("feature",)),
        primary_layer_name="Intensity",
        layers=(
            LongLayer(
                "Intensity",
                pl.DataFrame({VAR_ROW: [0], OBS_ROW: [0], VALUE: [1.0]}),
            ),
        ),
    )

    assert raw.primary_layer_name == "Intensity"
    assert tuple(layer.layer_name for layer in raw.layers) == ("Intensity",)


def test_raw_and_final_layer_tables_share_no_mode_bearing_abstraction() -> None:
    raw = LongLayer("Intensity", pl.DataFrame({VAR_ROW: [0], OBS_ROW: [0], VALUE: [1.0]}))
    final = FinalLayerTable.from_wide(
        layer_name="Intensity",
        values=(pl.DataFrame({"ProForma_ion": ["F/2"], "A": [1.0]})).drop(
            ("ProForma_ion",), strict=False
        ),
        semantic_roles=("abundance",),
    )

    assert type(raw).__mro__[1:] == (object,)
    assert type(final).__mro__[1:] == (object,)
    assert not hasattr(raw, "kind")
    assert not hasattr(final, "is_final")


def test_a_parsed_level_composes_final_values_and_nothing_new() -> None:
    parsed = ParsedLevel(
        obs=ObsFinal(frame=pl.DataFrame({"sample": ["A", "B"]}), key_columns=("sample",)),
        var=VarFinal(
            frame=pl.DataFrame({"ProForma_ion": ["F/2"], "genes": ["GENE1"]}),
            key_columns=("ProForma_ion",),
        ),
        primary_layer_name="Intensity",
        uns={"hierarchy": "lfq", "software_name": "AlphaDIA", "quantification_level": "ion"},
        layers={
            "Intensity": FinalLayerTable.from_wide(
                layer_name="Intensity",
                values=(pl.DataFrame({"ProForma_ion": ["F/2"], "A": [1.0], "B": [2.0]})).drop(
                    ("ProForma_ion",), strict=False
                ),
                semantic_roles=("abundance",),
            )
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
    )

    assert set(vars(ParsedLevel).get("__slots__", ())) == {
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
        "export_uns",
    }
    assert isinstance(parsed.uns, dict)
    assert isinstance(parsed.layers, dict)
    assert parsed.primary_layer_name in parsed.layers


def test_pipeline_values_carry_no_matrices_and_results_carry_no_row_numbers() -> None:
    """Layers are long rows of integer positions until widened, last; no record is a matrix,
    and a parsed result keeps no row numbers or positions of its own."""
    for record in (LevelSourceTable, ObsRaw, VarRaw, LongLayer, RawLevel, FinalPositions):
        assert not {"X", "matrix", "index", "join_map"} & set(vars(record).get("__slots__", ()))
    for record in (ObsFinal, VarFinal, FinalLayerTable, ParsedLevel):
        slots = set(vars(record).get("__slots__", ()))
        assert not {"X", "matrix", "index", "join_map", "rows", "positions"} & slots


# --------------------------------------------------------------------------------- parameters


def test_a_working_configuration_derives_presence_and_canonical_values() -> None:
    layer = WorkingMeasurementLayer(
        name="Intensity",
        source="precursor.intensity",
        value=PlainNumericLayerDeclaration(missing_values=(0.0,)),
    )
    working = WorkingParseConfiguration(
        level="ion",
        software_name="AlphaDIA",
        hierarchy=LevelHierarchy("lfq", (("ion", "feature"),)),
        input=InputContract(
            file_name=None,
            formats=(
                DelimitedFormatContract(
                    extensions=(".tsv",),
                    encoding_candidates=("utf8",),
                    quote_char='"',
                    delimiter_candidates=("\t",),
                    number_format_candidates=(DOT,),
                ),
            ),
        ),
        source_layout=LongSourceLayout(),
        obs=WorkingAxisConfiguration(
            final_key_columns=("sample",),
            required_selections=(
                AxisColumnSelection(name="sample", source="run", logical_type="string"),
            ),
            optional_selections=(),
            computed=(),
            declared_order=("sample",),
        ),
        var=WorkingAxisConfiguration(
            final_key_columns=("ProForma_ion",),
            required_selections=(),
            optional_selections=(),
            computed=(),
            declared_order=(),
        ),
        measurements=WorkingMeasurements(
            primary_layer_name="Intensity",
            duplicate_mode="keep_first",
            layers=(layer,),
            required_names=frozenset({"Intensity"}),
        ),
        column_roles={},
        provenance={},
        retained=RetainedFields(
            obs=frozenset({"sample"}), var=frozenset({"ProForma_ion"}), layers=frozenset()
        ),
    )

    assert working.measurements.required_layers[0] is layer
    assert layer.value == PlainNumericLayerDeclaration(missing_values=(0.0,))
    assert working.measurements.required_layers == working.measurements.layers
    # Optionality is a separate collection, never a flag on the record.
    assert not hasattr(working.obs.required_selections[0], "required")


@pytest.mark.parametrize(
    ("layout", "source", "header", "packed", "synthesized"),
    [
        (LongSourceLayout(), "Quantity", ("Quantity",), (), ()),
        (
            WideSourceLayout(sample_layer_name="Quantity"),
            r"^(?P<sample>.+) Quantity$",
            ("A Quantity",),
            (),
            (),
        ),
        (
            PositionalFragmentLayout(
                delimiter=";",
                label_output="fragment_label",
                packed_value_sources=("Quantity",),
            ),
            "Quantity",
            ("Quantity",),
            ("Quantity",),
            ("fragment_label",),
        ),
        (
            ColumnLabeledFragmentLayout(
                label_source="Info",
                delimiter=";",
                label_output="fragment_label",
                packed_value_sources=("Quantity",),
            ),
            "Quantity",
            ("Info", "Quantity"),
            ("Info", "Quantity"),
            ("fragment_label",),
        ),
    ],
)
def test_source_layouts_answer_their_own_header_questions(
    layout: SourceLayoutDeclaration,
    source: str,
    header: tuple[str, ...],
    packed: tuple[str, ...],
    synthesized: tuple[str, ...],
) -> None:
    assert layout.has_layer_source(source, header)
    assert layout.packed_sources() == packed
    assert layout.synthesized_var_columns() == synthesized


def test_working_measurements_derives_all_views_from_one_ordered_collection() -> None:
    first = WorkingMeasurementLayer(
        name="First",
        source="first",
        value=PlainNumericLayerDeclaration(missing_values=()),
    )
    primary = WorkingMeasurementLayer(
        name="Quantity",
        source="quantity",
        value=PlainNumericLayerDeclaration(missing_values=()),
    )
    last = WorkingMeasurementLayer(
        name="Last",
        source="last",
        value=PlainNumericLayerDeclaration(missing_values=()),
    )

    measurements = WorkingMeasurements(
        primary_layer_name="Quantity",
        duplicate_mode="error",
        layers=(first, primary, last),
        required_names=frozenset({"Quantity", "Last"}),
    )

    assert measurements.layers == (first, primary, last)
    assert measurements.required_layers == (primary, last)


@pytest.mark.parametrize(
    ("layers", "primary", "message"),
    [
        ([], "Quantity", "at least 1 item"),
        (
            [{"name": "Quantity", "source": "first"}, {"name": "Quantity", "source": "second"}],
            "Quantity",
            "must be unique",
        ),
        ([{"name": "Quantity", "source": "quantity"}], "Unknown", "matches no layer"),
    ],
)
def test_invalid_measurements_are_rejected_at_the_authored_boundary(
    layers: list[dict[str, str]],
    primary: str,
    message: str,
) -> None:
    from parserV2 import synthetic

    document = synthetic.document(
        shape="long",
        base={
            "axis": {"obs_keys": ["sample"], "var_keys": ["Feature"]},
            "columns": {
                "obs": [{"name": "sample", "source": "Run"}],
                "var": synthetic.with_flags([{"name": "Feature", "source": "Feature"}]),
            },
            "measurements": {"primary_layer": primary, "layers": layers},
        },
        levels={"ion": {}},
    )
    with pytest.raises(ValueError, match=message):
        synthetic.facade(document)


@pytest.mark.parametrize(
    ("shape", "sample_layer", "message"),
    [
        ("wide", "Unknown", "sample_layer='Unknown' matches no layer"),
        ("long", "Quantity", "sample_layer is valid only for wide rules"),
    ],
)
def test_invalid_sample_layers_are_rejected_at_the_authored_boundary(
    shape: str,
    sample_layer: str,
    message: str,
) -> None:
    from parserV2 import synthetic

    columns: dict[str, object] = {"var": [{"name": "Feature", "source": "Feature"}]}
    source = r"^(?P<sample>.+) Quantity$"
    if shape == "long":
        columns["obs"] = [{"name": "sample", "source": "Run"}]
        source = "Quantity"
    document = synthetic.document(
        shape=shape,
        base={
            "axis": {"obs_keys": ["sample"], "var_keys": ["Feature"]},
            "columns": columns,
            "measurements": {
                "primary_layer": "Quantity",
                "sample_layer": sample_layer,
                "layers": [{"name": "Quantity", "source": source}],
            },
        },
        levels={"ion": {}},
    )
    with pytest.raises(ValueError, match=message):
        synthetic.facade(document)


def test_primary_layer_is_required_even_without_an_authored_required_flag() -> None:
    from parserV2 import synthetic

    facade = synthetic.facade(
        synthetic.long_document(
            obs_select={"sample": "run"},
            var_select={"Feature": "feature"},
            layers=[{"name": "Quantity", "source": "Quantity", "required": False}],
        )
    )
    measurements = facade.working_parameters.measurements
    assert measurements.required_names == {"Quantity"}
    assert measurements.required_layers == measurements.layers


def test_source_compilation_returns_one_executable_strategy_not_a_resolved_graph() -> None:
    from apb2.parserV2.parse_quant.parser import ParseStrategy
    from parserV2 import synthetic

    facade = synthetic.facade(
        synthetic.long_document(obs_select={"sample": "run"}, var_select={"Feature": "feature"})
    )
    strategy = facade.resolve_source(
        DelimitedSourceEvidence(("run", "feature", "Quantity"), "\t", '"', "utf8", DOT)
    )
    assert isinstance(strategy, ParseStrategy)
    read = strategy.read
    assert read.text_sources.isdisjoint(read.native_numeric_sources)
    assert read.text_sources | read.native_numeric_sources == set(read.projected_columns)
    assert strategy.var.keys.final_key_columns == ("Feature",)
    assert callable(strategy.decompose) and callable(strategy.parse_decomposed)
    assert not hasattr(strategy, "layer_values")
    assert not hasattr(strategy, "decomposition")


def test_wide_layer_plans_keep_their_resolved_header_order() -> None:
    plan = WideRawLayerPlan(
        name="Intensity",
        sources=(WideRawLayerSource("run_A", "run_A"), WideRawLayerSource("run_B", "run_B")),
    )
    assert tuple(source.sample for source in plan.sources) == ("run_A", "run_B")


def test_delimited_evidence_preserves_the_physical_header_order() -> None:
    evidence = DelimitedSourceEvidence(
        columns=("b", "a", "c"),
        delimiter="\t",
        quote_char='"',
        encoding="utf8",
        number_format=DOT,
    )

    assert evidence.columns == ("b", "a", "c")


# ---------------------------------------------------------------------- structural conformance


class _Reader:
    def read(self) -> LevelSourceTable:
        return LevelSourceTable(frame=pl.DataFrame({"a": [1]}))


class _Decomposer:
    def decompose(self, table: LevelSourceTable, /) -> RawLevel:
        del table
        raise NotImplementedError


class _Separator:
    label_output = "fragment_label"
    packed_sources: tuple[str, ...] = ("a",)

    def listed(self, frame: pl.DataFrame, /) -> pl.DataFrame:
        return frame

    def exploded(self, frame: pl.DataFrame, /) -> pl.DataFrame:
        return frame


class _Coercer:
    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        return pl.col(source).alias(name)


class _Computer:
    name = "ProForma_ion"
    inputs: tuple[str, ...] = ("ProForma_peptidoform", "Charge")

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
        return frame.with_columns(pl.col(self.inputs[0]).alias(self.name)), ()


class _Presence:
    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        del dtype
        return values.is_not_null()


class _Policy:
    def resolve(self, layer: LongLayer, /) -> LongLayer:
        return layer


def test_the_intended_collaborators_satisfy_their_client_owned_contracts() -> None:
    reader: BoundInputReader = _Reader()
    decomposer: SourceDecomposer = _Decomposer()
    separator: FragmentTableSeparator = _Separator()
    coercer: AxisValueCoercer = _Coercer()
    computer: ColumnComputer = _Computer()
    presence = _Presence()
    policy: DuplicatePolicy = _Policy()

    assert reader.read().frame.height == 1
    assert separator.exploded(separator.listed(pl.DataFrame({"a": [1]}))).height == 1
    frame = pl.DataFrame({"x": [1]})
    assert frame.select(coercer.coerce(frame, name="x", source="x")).to_series().to_list() == [1]
    assert computer.inputs == ("ProForma_peptidoform", "Charge")
    presence_values = pl.Series("x", [1.0, None])
    assert presence_values.to_frame().select(
        presence.present(pl.col("x"), presence_values.dtype).alias("x")
    ).to_series().to_list() == [True, False]
    assert decomposer is not None
    assert policy is not None


def test_a_runtime_axis_plan_holds_configured_behaviour_and_no_discriminator() -> None:
    plan = AxisRuntimePlan(
        keys=AxisKeyPlan(
            raw_key_columns=("charge",),
            key_input_columns=("Charge",),
            final_key_columns=("Charge",),
        ),
        key_phase=AxisPhaseRuntimePlan(
            selections=(SelectedAxisColumn(name="Charge", source="charge", coercer=_Coercer()),),
            computers=(),
        ),
        output_phase=AxisPhaseRuntimePlan(selections=(), computers=(_Computer(),)),
        outputs=("Charge", "ProForma_ion"),
    )

    assert not hasattr(plan, "skipped")
    assert not hasattr(plan.key_phase.selections[0], "required")
    assert not hasattr(plan.key_phase.selections[0], "logical_type")


def test_a_shared_read_keeps_a_column_categorical_only_where_every_level_does() -> None:
    ion = LevelReadPlan(
        ("run", "seq", "q"), frozenset({"run", "seq", "q"}), frozenset(), frozenset({"run", "seq"})
    )
    protein = LevelReadPlan(
        ("run", "seq"), frozenset({"run", "seq"}), frozenset(), frozenset({"run"})
    )

    union = LevelReadPlan.union([ion, protein])

    assert union.categorical_sources == frozenset({"run"})
    assert union.text_sources == frozenset({"run", "seq", "q"})


def test_a_shared_read_converts_a_column_only_where_every_level_does() -> None:
    ion = LevelReadPlan(
        ("q", "pg"), frozenset({"q", "pg"}), frozenset(), converted_sources=frozenset({"q", "pg"})
    )
    protein = LevelReadPlan(("pg",), frozenset({"pg"}), frozenset())

    union = LevelReadPlan.union([ion, protein])

    assert union.converted_sources == frozenset({"q"})
    assert union.text_sources == frozenset({"q", "pg"})


def test_a_blank_category_is_as_absent_as_blank_text() -> None:
    frame = pl.DataFrame(
        {
            "key": pl.Series(["A", " ", None, "B", "\t"], dtype=pl.Categorical),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    table = LevelSourceTable(frame)

    assert table.frame.get_column("key").to_list() == ["A", None, None, "B", None]
    assert table.frame.schema["key"] == pl.Categorical()
