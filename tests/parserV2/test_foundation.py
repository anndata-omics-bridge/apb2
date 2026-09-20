"""The parsing-owned vocabulary: pipeline data, parameters, and client-owned contracts.

One cohesive module rather than one file per class: what is worth asserting about these
values is that identity is explicit, that a raw and a final layer are two unrelated states,
and that the runtime plans accept the collaborators the compiler will inject.
"""

from __future__ import annotations

from pathlib import Path

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
    ParsedLevelWriter,
    RawValuePresence,
    SelectedAxisColumn,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.data.computed import ColumnComputation
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    VarFinal,
)
from apb2.parserV2.parse_quant.data.raw import (
    DecomposedDataRaw,
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    RawToFinalKeyMap,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.operations import (
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


def test_a_raw_layer_puts_its_var_keys_first_and_obs_values_after_them() -> None:
    layer = RawLayerTable(
        layer_name="Intensity",
        raw_var_key_columns=("sequence", "charge"),
        values=pl.DataFrame(
            {
                "sequence": ["PEPMIDE", "PEPMIDE", "OTHER"],
                "charge": ["2", "2", "3"],
                "A": [100.0, 110.0, 50.0],
                "B": [120.0, None, 60.0],
            }
        ),
    )

    keys = len(layer.raw_var_key_columns)
    assert layer.values.columns[:keys] == ["sequence", "charge"]
    # Repeated raw cells survive decomposition for the duplicate policy to resolve.
    assert layer.values.height == 3


def test_a_key_map_relates_equal_length_raw_and_final_frames() -> None:
    mapping = RawToFinalKeyMap(
        raw_keys=pl.DataFrame({"sequence": ["PEPMIDE", "OTHER"], "charge": ["2", "3"]}),
        final_keys=pl.DataFrame({"ProForma_ion": ["PEPMIDE/2", "OTHER/3"]}),
    )

    assert mapping.raw_keys.height == mapping.final_keys.height


def test_decomposition_reduces_every_physical_shape_to_one_raw_contract() -> None:
    raw = DecomposedDataRaw(
        obs=ObsRaw(frame=pl.DataFrame({"sample": ["A"]}), raw_key_columns=("sample",)),
        var=VarRaw(frame=pl.DataFrame({"feature": ["F"]}), raw_key_columns=("feature",)),
        layers=LayersRaw(
            primary_layer_name="Intensity",
            values=(
                RawLayerTable(
                    layer_name="Intensity",
                    raw_var_key_columns=("feature",),
                    values=pl.DataFrame({"feature": ["F"], "A": [1.0]}),
                ),
            ),
        ),
    )

    assert raw.layers.primary_layer_name == "Intensity"
    assert tuple(layer.layer_name for layer in raw.layers.values) == ("Intensity",)


def test_raw_and_final_layer_tables_share_no_mode_bearing_abstraction() -> None:
    raw = RawLayerTable(
        layer_name="Intensity",
        raw_var_key_columns=("feature",),
        values=pl.DataFrame({"feature": ["F"], "A": [1.0]}),
    )
    final = FinalLayerTable(
        layer_name="Intensity",
        var_key_columns=("ProForma_ion",),
        values=pl.DataFrame({"ProForma_ion": ["F/2"], "A": [1.0]}),
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
        uns={"software_name": "AlphaDIA", "quantification_level": "ion"},
        layers={
            "Intensity": FinalLayerTable(
                layer_name="Intensity",
                var_key_columns=("ProForma_ion",),
                values=pl.DataFrame({"ProForma_ion": ["F/2"], "A": [1.0], "B": [2.0]}),
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
    }
    assert isinstance(parsed.uns, dict)
    assert isinstance(parsed.layers, dict)
    assert parsed.primary_layer_name in parsed.layers


def test_pipeline_values_carry_no_matrices_indexes_or_temporary_identities() -> None:
    forbidden = {"X", "codes", "index", "matrix", "obs_codes", "var_codes", "join_map"}
    for record in (
        LevelSourceTable,
        ObsRaw,
        VarRaw,
        RawLayerTable,
        LayersRaw,
        DecomposedDataRaw,
        RawToFinalKeyMap,
        ObsFinal,
        VarFinal,
        FinalLayerTable,
        ParsedLevel,
    ):
        assert not forbidden & set(vars(record).get("__slots__", ()))


# --------------------------------------------------------------------------------- parameters


def test_a_working_configuration_derives_presence_and_canonical_values() -> None:
    layer = WorkingMeasurementLayer(
        name="Intensity",
        source="precursor.intensity",
        value=PlainNumericLayerDeclaration(missing_values=(0.0,)),
    )
    working = WorkingParseConfiguration(
        level="ion",
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
        provenance={"software_name": "AlphaDIA"},
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
            WideSourceLayout(),
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
                "var": [{"name": "Feature", "source": "Feature"}],
            },
            "measurements": {"primary_layer": primary, "layers": layers},
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
    assert callable(strategy.parse)
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
    def decompose(self, table: LevelSourceTable, /) -> DecomposedDataRaw:
        del table
        raise NotImplementedError


class _Separator:
    def separate(self, table: LevelSourceTable, /) -> LevelSourceTable:
        return table


class _Coercer:
    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        del name, source
        return values


class _Computer:
    name = "ProForma_ion"
    inputs: tuple[str, ...] = ("ProForma_peptidoform", "Charge")

    def compute(self, columns: tuple[pl.Series, ...], /) -> ColumnComputation:
        return ColumnComputation(columns[0])


class _Presence:
    def present(self, values: pl.Expr, dtype: pl.DataType, /) -> pl.Expr:
        del dtype
        return values.is_not_null()


class _Policy:
    def resolve(self, layer: RawLayerTable, presence: RawValuePresence, /) -> RawLayerTable:
        del presence
        return layer


class _Writer:
    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        del parsed, target


def test_the_intended_collaborators_satisfy_their_client_owned_contracts() -> None:
    reader: BoundInputReader = _Reader()
    decomposer: SourceDecomposer = _Decomposer()
    separator: FragmentTableSeparator = _Separator()
    coercer: AxisValueCoercer = _Coercer()
    computer: ColumnComputer = _Computer()
    presence: RawValuePresence = _Presence()
    policy: DuplicatePolicy = _Policy()
    writer: ParsedLevelWriter = _Writer()

    assert reader.read().frame.height == 1
    assert separator.separate(LevelSourceTable(frame=pl.DataFrame({"a": [1]}))).frame.height == 1
    assert coercer.coerce(pl.Series("x", [1]), name="x", source="x").to_list() == [1]
    assert computer.inputs == ("ProForma_peptidoform", "Charge")
    presence_values = pl.Series("x", [1.0, None])
    assert presence_values.to_frame().select(
        presence.present(pl.col("x"), presence_values.dtype).alias("x")
    ).to_series().to_list() == [True, False]
    assert decomposer is not None
    assert policy is not None
    assert writer is not None


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
