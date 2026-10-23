"""Uninformative columns and layers are left out of a level; a finding says what they held."""

from __future__ import annotations

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    VarFinal,
)
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedSourceEvidence,
    LevelSourceTable,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.uninformative import UninformativeFields
from parserV2 import synthetic

PRUNER = UninformativeFields(
    retained_obs=frozenset({"Run"}),
    retained_var=frozenset({"Precursor", "Protein", "ProForma_ion", "Decoy", "apb_Decoy"}),
    retained_layers=frozenset({"Intensity"}),
)


def obs(**columns: list[object]) -> ObsFinal:
    return ObsFinal(frame=pl.DataFrame({"Run": ["A", "B", "C"], **columns}), key_columns=("Run",))


def var(**columns: list[object]) -> VarFinal:
    return VarFinal(
        frame=pl.DataFrame({"Precursor": ["P1", "P2"], **columns}), key_columns=("Precursor",)
    )


def layer(name: str, rows: list[list[float | None]]) -> FinalLayerTable:
    """One layer with two features and three runs, given row by row."""
    columns = list(zip(*rows, strict=True))
    return FinalLayerTable(
        name, pl.DataFrame({f"c{index}": list(column) for index, column in enumerate(columns)})
    )


INTENSITY = layer("Intensity", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_a_single_valued_column_and_layer_are_left_out_with_their_value() -> None:
    pruned = PRUNER.prune(
        obs(Instrument=["Astral", "Astral", "Astral"], Order=[1, 2, 3]),
        var(Lib_Q_Value=[0.0, 0.0], Global_Q_Value=[0.01, 0.02]),
        {
            "Intensity": INTENSITY,
            "IM": layer("IM", [[0.0, 0.0, None], [0.0, 0.0, 0.0]]),
            "RT": layer("RT", [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
        },
        {},
    )

    assert pruned.obs.frame.columns == ["Run", "Order"]
    assert pruned.var.frame.columns == ["Precursor", "Global_Q_Value"]
    assert list(pruned.layers) == ["Intensity", "RT"]
    assert pruned.single_valued == {
        "obs": {"Instrument": {"value": "Astral"}},
        "var": {"Lib_Q_Value": {"value": 0.0}},
        "layers": {"IM": {"value": 0.0}},
    }
    assert pruned.effectively_empty == {"obs": {}, "var": {}, "layers": {}}


def test_one_filled_cell_is_a_measurement_not_a_single_value() -> None:
    """A one-run result keeps its obs columns; a lone value is no finding."""
    one_run = ObsFinal(pl.DataFrame({"Run": ["A"], "File": ["a.raw"]}), ("Run",))

    pruned = PRUNER.prune(one_run, var(Genes=["G1", None]), {"Intensity": INTENSITY}, {})

    assert pruned.obs.frame.columns == ["Run", "File"]
    assert pruned.var.frame.columns == ["Precursor", "Genes"]
    assert pruned.single_valued == {"obs": {}, "var": {}, "layers": {}}


def test_an_empty_column_or_layer_is_left_out_with_its_filled_count() -> None:
    pruned = PRUNER.prune(
        obs(),
        var(Genes=[None, None], Sites=[None, None]),
        {"Intensity": INTENSITY, "Channel_Q": layer("Channel_Q", [[None] * 3, [None] * 3])},
        {},
    )

    assert pruned.var.frame.columns == ["Precursor"]
    assert list(pruned.layers) == ["Intensity"]
    assert pruned.effectively_empty == {
        "obs": {},
        "var": {"Genes": {"filled": 0}, "Sites": {"filled": 0}},
        "layers": {"Channel_Q": {"filled": 0}},
    }


def test_an_effectively_empty_layer_keeps_the_validators_verdict_and_a_summary() -> None:
    verdict: dict[str, JsonValue] = {"occupancy": 2 / 6, "reference_layers": ["Intensity"]}
    sparse = layer("MS1", [[1.0, None, None], [3.0, None, None]])

    pruned = PRUNER.prune(obs(), var(), {"Intensity": INTENSITY, "MS1": sparse}, {"MS1": verdict})

    assert list(pruned.layers) == ["Intensity"]
    assert pruned.effectively_empty["layers"] == {
        "MS1": {
            **verdict,
            "filled": 2,
            "summary": {
                "min": 1.0,
                "first_quartile": 1.5,
                "median": 2.0,
                "mean": 2.0,
                "third_quartile": 2.5,
                "max": 3.0,
            },
        }
    }


def test_retained_fields_stay_whatever_they_hold() -> None:
    """Keys, role and computed columns, the flags' inputs and the primary layer stay."""
    constant = layer("Intensity", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    pruned = PRUNER.prune(
        ObsFinal(pl.DataFrame({"Run": ["A", "A"]}), ("Run",)),
        var(
            Protein=["Q1", "Q1"],
            ProForma_ion=["PEPK/2", "PEPK/2"],
            Decoy=[False, False],
            apb_Decoy=[False, False],
        ),
        {"Intensity": constant},
        {},
    )

    assert pruned.var.frame.columns == [
        "Precursor",
        "Protein",
        "ProForma_ion",
        "Decoy",
        "apb_Decoy",
    ]
    assert list(pruned.layers) == ["Intensity"]
    assert pruned.single_valued == {"obs": {}, "var": {}, "layers": {}}


def test_a_single_valued_factor_layer_reports_its_label() -> None:
    status = FinalLayerTable(
        "Status",
        pl.DataFrame({"c0": [1, 1], "c1": [1, -1], "c2": [1, 1]}),
        role=AuxiliaryLayerRole(),
        semantics=CategoricalLayerSemantics(categories=(("identified", 1), ("matched", 2))),
    )

    pruned = PRUNER.prune(obs(), var(), {"Intensity": INTENSITY, "Status": status}, {})

    assert pruned.single_valued["layers"] == {"Status": {"value": "identified"}}


def test_a_parsed_level_reports_what_it_left_out() -> None:
    """Through the facade: the flag's input stays, the constant column goes, with a finding."""
    document = synthetic.long_document(
        obs_select={"sample": "Sample"},
        var_select={"Feature": "Feature", "Decoy": "Decoy", "Organism": "Organism"},
        computed=[{"name": "apb_Decoy", "inputs": ["Decoy"], "how": "decoy", "equals": "true"}],
        var_keys=["Feature"],
    )
    strategy = synthetic.facade(document).resolve_source(
        DelimitedSourceEvidence(
            columns=("Sample", "Feature", "Decoy", "Organism", "Quantity"),
            delimiter="\t",
            quote_char='"',
            encoding="utf8",
            number_format=NumericTextFormat(decimal_mark=".", thousands_marks=()),
        )
    )
    frame = pl.DataFrame(
        {
            "Sample": ["A", "B", "A"],
            "Feature": ["P1", "P1", "P2"],
            "Decoy": ["false", "false", "false"],
            "Organism": ["Homo sapiens"] * 3,
            "Quantity": ["1", "2", "3"],
        }
    )

    parsed = strategy.parse(LevelSourceTable(frame=frame))

    assert "Decoy" in parsed.var.frame.columns
    assert "Organism" not in parsed.var.frame.columns
    result = parsed.uns["result"]
    assert isinstance(result, dict)
    assert result["uninformative_fields"] == {
        "single_valued": {"obs": {}, "var": {"Organism": {"value": "Homo sapiens"}}, "layers": {}},
        "effectively_empty": {"obs": {}, "var": {}, "layers": {}},
    }
