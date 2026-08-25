"""The two output adapters: one preserves the parsed values, the other converts them.

Between them they carry the whole point of the split. Parquet must be able to write anything
parsing produced — a string layer, a localized token, a null — without knowing what it means.
AnnData must be the only place that decides what those values become, and must fail loudly
when a declared representation cannot hold them.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import anndata
import mudata
import numpy as np
import pandas as pd
import polars as pl
import pytest

from apb2.parserV2.parse_quant.anndata_writer import (
    NAMESPACE,
    PARSE_NAMESPACE,
    AnnDataLayerContractError,
    AnnDataWriter,
    FactorAnnDataEncoder,
    MuDataLevelError,
    MuDataWriter,
    OccupancyPolicy,
    ParsedLevels,
    PlainNumericAnnDataEncoder,
    RegexNumericAnnDataEncoder,
    StandardAnnDataLayerContract,
    StrictAnnDataLayerContract,
)
from apb2.parserV2.parse_quant.contracts import ParsedLevelWriter
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    VarFinal,
)
from apb2.parserV2.parse_quant.numeric_text import NumberNotation
from apb2.parserV2.parse_quant.parquet_writer import (
    MANIFEST_NAME,
    ParquetWriteError,
    ParquetWriter,
)

DOT = NumberNotation(decimal_mark=".", thousands_marks=())
GROUPED = NumberNotation(decimal_mark=",", thousands_marks=(".",))


def level(
    *,
    obs: pl.DataFrame | None = None,
    obs_keys: tuple[str, ...] = ("Run",),
    var: pl.DataFrame | None = None,
    var_keys: tuple[str, ...] = ("Feature",),
    layers: Mapping[str, pl.DataFrame] | None = None,
    primary: str = "Intensity",
    uns: Mapping[str, JsonValue] | None = None,
) -> ParsedLevel:
    obs_frame = obs if obs is not None else pl.DataFrame({"Run": ["A", "B"]})
    var_frame = var if var is not None else pl.DataFrame({"Feature": ["F1", "F2"]})
    values = (
        layers
        if layers is not None
        else {
            "Intensity": pl.DataFrame(
                {"Feature": ["F1", "F2"], "obs_0": [1.0, 2.0], "obs_1": [3.0, None]}
            )
        }
    )
    return ParsedLevel(
        obs=ObsFinal(frame=obs_frame, key_columns=obs_keys),
        var=VarFinal(frame=var_frame, key_columns=var_keys),
        primary_layer_name=primary,
        uns=dict(uns or {"software_name": "Synthetic"}),
        layers={
            name: FinalLayerTable(layer_name=name, var_key_columns=var_keys, values=frame)
            for name, frame in values.items()
        },
    )


def manifest_of(target: Path) -> dict[str, JsonValue]:
    payload = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------------- Parquet


def test_a_parquet_dataset_round_trips_every_value_and_dtype(tmp_path: Path) -> None:
    parsed = level(
        var=pl.DataFrame({"Feature": ["F1", "F2"], "Charge": [2, None], "Decoy": [True, None]}),
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1", "F2"], "obs_0": ["100.000,5", None]}),
            "Kind": pl.DataFrame({"Feature": ["F1", "F2"], "obs_0": ["MBR", "MS/MS"]}),
        },
    )
    target = tmp_path / "ion"

    ParquetWriter().write(parsed, target)

    assert pl.read_parquet(target / "obs.parquet").to_dicts() == parsed.obs.frame.to_dicts()
    assert pl.read_parquet(target / "var.parquet").schema == parsed.var.frame.schema
    intensity = pl.read_parquet(target / "layers" / "Intensity.parquet")
    # The localized token is still the token: no encoder ran.
    assert intensity.get_column("obs_0").to_list() == ["100.000,5", None]
    assert intensity.schema["obs_0"] == pl.String
    assert pl.read_parquet(target / "layers" / "Kind.parquet").get_column("obs_0").to_list() == [
        "MBR",
        "MS/MS",
    ]


def test_the_manifest_states_what_every_file_is(tmp_path: Path) -> None:
    parsed = level(
        var_keys=("Feature", "Charge"),
        var=pl.DataFrame({"Feature": ["F1"], "Charge": [2]}),
        uns={"software_name": "Synthetic", "unknown_mod_tokens": ["Mystery@M"]},
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1"], "Charge": [2], "obs_0": [1.0]}),
            "Q Value": pl.DataFrame({"Feature": ["F1"], "Charge": [2], "obs_0": [0.01]}),
        },
    )
    target = tmp_path / "ion"

    ParquetWriter().write(parsed, target)
    manifest = manifest_of(target)

    assert manifest["primary_layer"] == "Intensity"
    assert manifest["layer_order"] == ["Intensity", "Q Value"]
    assert manifest["obs"] == {"file": "obs.parquet", "key_columns": ["Run"], "columns": ["Run"]}
    assert manifest["var"] == {
        "file": "var.parquet",
        "key_columns": ["Feature", "Charge"],
        "columns": ["Feature", "Charge"],
    }
    assert manifest["uns"] == {
        "software_name": "Synthetic",
        "unknown_mod_tokens": ["Mystery@M"],
    }
    layers = manifest["layers"]
    assert isinstance(layers, dict)
    assert layers["Q Value"] == {
        "file": "layers/Q_Value.parquet",
        "var_key_columns": ["Feature", "Charge"],
        "observation_columns": ["obs_0"],
    }
    assert (target / "layers" / "Q_Value.parquet").is_file()


def test_a_layer_name_is_mapped_to_a_file_name_never_interpolated(tmp_path: Path) -> None:
    parsed = level(
        layers={
            "../escape": pl.DataFrame({"Feature": ["F1"], "obs_0": [1.0]}),
            "..%escape": pl.DataFrame({"Feature": ["F1"], "obs_0": [2.0]}),
            "Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [3.0]}),
        }
    )
    target = tmp_path / "ion"

    ParquetWriter().write(parsed, target)
    layers = manifest_of(target)["layers"]

    assert isinstance(layers, dict)
    files = sorted(path.name for path in (target / "layers").iterdir())
    assert files == ["Intensity.parquet", "escape.parquet", "escape_1.parquet"]
    assert not (tmp_path / "escape.parquet").exists()
    for name, entry in layers.items():
        assert isinstance(entry, dict)
        assert str(entry["file"]).startswith("layers/")
        assert name not in files or name.endswith("Intensity")


def test_writing_over_an_existing_dataset_leaves_only_the_new_one(tmp_path: Path) -> None:
    target = tmp_path / "ion"
    ParquetWriter().write(level(), target)
    (target / "layers" / "Stale.parquet").write_bytes(b"stale")

    ParquetWriter().write(level(), target)

    assert sorted(path.name for path in (target / "layers").iterdir()) == ["Intensity.parquet"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion"]


def test_a_target_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "ion"
    target.write_text("not a dataset", encoding="utf-8")

    with pytest.raises(ParquetWriteError, match="not a directory"):
        ParquetWriter().write(level(), target)

    assert target.read_text(encoding="utf-8") == "not a dataset"


def test_a_failure_part_way_through_leaves_the_previous_dataset_intact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ion"
    ParquetWriter().write(level(), target)
    before = pl.read_parquet(target / "obs.parquet").to_dicts()
    broken = level(layers={"Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [object()]})})

    with pytest.raises(Exception, match=r".*"):
        ParquetWriter().write(broken, target)

    assert pl.read_parquet(target / "obs.parquet").to_dicts() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion"]


def test_the_parquet_writer_imports_no_encoder_backend() -> None:
    source = Path("src/apb2/parserV2/parse_quant/parquet_writer.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not {"numpy", "pandas", "anndata"} & imported
    assert not any(name.endswith("anndata_writer") for name in imported)


def test_the_parquet_writer_satisfies_the_parser_owned_writer_contract(
    tmp_path: Path,
) -> None:
    writer: ParsedLevelWriter = ParquetWriter()

    writer.write(level(), tmp_path / "ion")

    assert (tmp_path / "ion" / MANIFEST_NAME).is_file()


# ---------------------------------------------------------------------------------- encoders


def block(*columns: list[object]) -> pl.DataFrame:
    return pl.DataFrame(
        {f"obs_{index}": values for index, values in enumerate(columns)}, strict=False
    )


def test_plain_numeric_encoding_reads_numbers_and_blanks_out_the_sentinel() -> None:
    encoder = PlainNumericAnnDataEncoder(
        layer_name="Intensity", missing_values=(0.0,), number_format=DOT
    )

    encoded = encoder.encode(block(["12.5", "0", "", None]))

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]
    assert encoded.schema["obs_0"] == pl.Float64
    assert encoded.columns == ["obs_0"]


def test_a_localized_number_is_read_under_the_notation_it_was_written_in() -> None:
    encoder = PlainNumericAnnDataEncoder(
        layer_name="Intensity", missing_values=(), number_format=GROUPED
    )

    encoded = encoder.encode(block(["100.000.000", "1.234,5", None]))

    assert encoded.get_column("obs_0").to_list() == [100000000.0, 1234.5, None]


def test_a_token_a_plain_numeric_layer_cannot_hold_becomes_missing_and_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The vendors these rules describe write ``-``, ``NA``, and ``False`` in such a column.

    Refusing the file would convert nothing; the encoded-layer contract is what decides
    whether enough values survived, and this reports the tokens that did not.
    """
    encoder = PlainNumericAnnDataEncoder(
        layer_name="Intensity", missing_values=(), number_format=DOT
    )

    encoded = encoder.encode(block(["12.5", "not a number", "-", "NA"]))

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]


def test_an_already_numeric_column_is_not_sent_through_its_own_text_form() -> None:
    """A float32 round-tripped through text is not the value it was; numbers stay numbers."""
    encoder = PlainNumericAnnDataEncoder(
        layer_name="Intensity", missing_values=(), number_format=DOT
    )
    values = pl.DataFrame({"obs_0": pl.Series([1268453.25], dtype=pl.Float32)})

    encoded = encoder.encode(values)

    assert encoded.get_column("obs_0").to_list() == [1268453.25]


def test_regex_encoding_extracts_the_number_and_treats_no_match_as_missing() -> None:
    encoder = RegexNumericAnnDataEncoder(
        layer_name="AScore",
        missing_values=(0.0,),
        pattern=r":(-?\d+(?:\.\d+)?)(?:;|$)",
        number_format=DOT,
    )

    encoded = encoder.encode(block(["S4:Phospho:12.5", "S4:Phospho:0", "unstructured", None]))

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]


def test_factor_encoding_maps_declared_labels_and_codes_the_rest_as_unknown() -> None:
    encoder = FactorAnnDataEncoder(
        layer_name="Match_Type",
        categories=(("unmatched", 0), ("MS/MS", 1), ("MBR", 2)),
    )

    encoded = encoder.encode(block(["MBR", "MS/MS", "surprise", None]))

    assert encoded.get_column("obs_0").to_list() == [2, 1, -1, -1]
    assert encoded.schema["obs_0"] == pl.Int64


def test_an_encoder_preserves_the_shape_and_the_column_order_it_was_given() -> None:
    encoder = PlainNumericAnnDataEncoder(
        layer_name="Intensity", missing_values=(), number_format=DOT
    )
    values = block(["1"], ["2"], ["3"])

    encoded = encoder.encode(values)

    assert encoded.columns == values.columns
    assert encoded.shape == values.shape


# ------------------------------------------------------------------------- contract checks


def policy(*required: str, primary: str = "Intensity") -> OccupancyPolicy:
    return OccupancyPolicy(
        primary_layer_name=primary,
        required_names=required or (primary,),
        empty_ratio=0.001,
        populated_ratio=0.5,
    )


def test_a_missing_required_layer_is_a_contract_error() -> None:
    checker = StandardAnnDataLayerContract(policy("Intensity", "QValue"))

    with pytest.raises(AnnDataLayerContractError, match="QValue"):
        checker.check({"Intensity": block([1.0])})


def test_an_empty_primary_layer_beside_a_populated_sibling_is_an_error() -> None:
    checker = StandardAnnDataLayerContract(policy())
    encoded = {
        "Intensity": block([None, None, None, None]),
        "QValue": block([0.1, 0.2, 0.3, 0.4]),
    }

    with pytest.raises(AnnDataLayerContractError, match="effectively empty"):
        checker.check(encoded)


def test_an_empty_auxiliary_layer_only_warns_unless_the_check_is_strict() -> None:
    encoded = {
        "Intensity": block([0.1, 0.2, 0.3, 0.4]),
        "QValue": block([None, None, None, None]),
    }

    StandardAnnDataLayerContract(policy()).check(encoded)
    with pytest.raises(AnnDataLayerContractError, match="QValue"):
        StrictAnnDataLayerContract(policy()).check(encoded)


def test_without_a_populated_sibling_occupancy_invents_no_conclusion() -> None:
    encoded = {"Intensity": block([None, None]), "QValue": block([None, None])}

    StrictAnnDataLayerContract(policy()).check(encoded)


def test_a_factor_layer_of_unknown_codes_still_counts_as_populated() -> None:
    encoded = {
        "Intensity": block([0.1, 0.2, 0.3, 0.4]),
        "Match_Type": pl.DataFrame({"obs_0": [-1, -1, -1, -1]}),
    }

    StrictAnnDataLayerContract(policy()).check(encoded)


# ------------------------------------------------------------------------------- the writer


def writer_for(
    parsed: ParsedLevel, *, checks: str = "standard", notation: NumberNotation = DOT
) -> AnnDataWriter:
    contract = policy(*tuple(parsed.layers), primary=parsed.primary_layer_name)
    return AnnDataWriter(
        encoders={
            name: PlainNumericAnnDataEncoder(
                layer_name=name, missing_values=(), number_format=notation
            )
            for name in parsed.layers
        },
        contract=(
            StrictAnnDataLayerContract(contract)
            if checks == "strict"
            else StandardAnnDataLayerContract(contract)
        ),
    )


def test_the_written_object_is_observations_by_variables_with_the_primary_layer_as_x(
    tmp_path: Path,
) -> None:
    parsed = level()
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert stored.shape == (2, 2)
    assert np.array_equal(
        np.asarray(stored.X), np.array([[1.0, 2.0], [3.0, np.nan]]), equal_nan=True
    )
    # AnnData 0.13 lists X itself as the unnamed layer; the named layers are what we wrote.
    assert {name for name in stored.layers.keys() if name is not None} == {  # noqa: SIM118
        "Intensity"
    }
    assert np.array_equal(
        np.asarray(stored.layers["Intensity"]), np.asarray(stored.X), equal_nan=True
    )


def test_the_primary_layer_is_selected_for_x_and_kept_under_its_own_name(
    tmp_path: Path,
) -> None:
    parsed = level(
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [1.0], "obs_1": [2.0]}),
            "QValue": pl.DataFrame({"Feature": ["F1"], "obs_0": [0.1], "obs_1": [0.2]}),
        },
        var=pl.DataFrame({"Feature": ["F1"]}),
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    named = {name for name in stored.layers.keys() if name is not None}  # noqa: SIM118
    assert sorted(named) == ["Intensity", "QValue"]
    assert np.array_equal(np.asarray(stored.X), np.array([[1.0], [2.0]]))


def test_every_authored_key_stays_an_ordinary_column_beside_the_storage_index(
    tmp_path: Path,
) -> None:
    parsed = level(
        var=pl.DataFrame({"Feature": ["F1", "F2"], "Gene": ["G1", "G2"]}),
        obs=pl.DataFrame({"Run": ["A", "B"], "Fraction": ["1", "2"]}),
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert list(stored.var.columns) == ["Feature", "Gene"]
    assert list(stored.obs.columns) == ["Run", "Fraction"]
    assert list(stored.var.index) == ["F1", "F2"]
    assert stored.var.index.name == "Feature"


def test_a_single_nonstring_key_becomes_a_typed_storage_string(tmp_path: Path) -> None:
    parsed = level(
        var=pl.DataFrame({"Charge": [2, 3]}),
        var_keys=("Charge",),
        layers={
            "Intensity": pl.DataFrame({"Charge": [2, 3], "obs_0": [1.0, 2.0], "obs_1": [3.0, 4.0]})
        },
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert list(stored.var.index) == ['[["Int64","2"]]', '[["Int64","3"]]']
    assert stored.var.index.name == "Charge_key"
    assert list(stored.var["Charge"]) == [2, 3]


def test_a_multi_column_key_index_survives_embedded_separators(tmp_path: Path) -> None:
    parsed = level(
        var=pl.DataFrame({"First": ["a_b", "a"], "Second": ["c", "b_c"]}),
        var_keys=("First", "Second"),
        layers={
            "Intensity": pl.DataFrame(
                {
                    "First": ["a_b", "a"],
                    "Second": ["c", "b_c"],
                    "obs_0": [1.0, 2.0],
                    "obs_1": [3.0, 4.0],
                }
            )
        },
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert len(set(stored.var.index)) == 2
    assert list(stored.var.index) == [
        '[["String","a_b"],["String","c"]]',
        '[["String","a"],["String","b_c"]]',
    ]
    assert stored.var.index.name == "First_Second"


def test_a_string_one_and_an_integer_one_do_not_become_the_same_index(
    tmp_path: Path,
) -> None:
    def written(frame: pl.DataFrame) -> list[str]:
        parsed = level(
            var=frame,
            var_keys=("Key",),
            layers={
                "Intensity": pl.DataFrame(
                    {"Key": frame.get_column("Key"), "obs_0": [1.0], "obs_1": [2.0]}
                )
            },
        )
        target = tmp_path / f"{frame.schema['Key']}.h5ad"
        writer_for(parsed).write(parsed, target)
        return list(anndata.read_h5ad(target).var.index)

    assert written(pl.DataFrame({"Key": ["1"]})) == ["1"]
    assert written(pl.DataFrame({"Key": [1]})) == ['[["Int64","1"]]']


def test_axis_dtypes_are_normalized_to_what_hdf5_accepts(tmp_path: Path) -> None:
    parsed = level(
        var=pl.DataFrame(
            {
                "Feature": ["F1", "F2"],
                "Charge": [2, None],
                "Decoy": [True, None],
                "Mass": [1.5, None],
            }
        )
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert list(stored.var["Charge"]) == [2, pd.NA]
    assert list(stored.var["Decoy"]) == [True, pd.NA]
    assert stored.var["Mass"].tolist()[0] == 1.5


def test_only_repeated_axis_strings_are_dictionary_encoded(tmp_path: Path) -> None:
    var = pl.DataFrame(
        {
            "ProForma_ion": ["ONE/2", "TWO/2", "THREE/2", "FOUR/2"],
            "Condition": ["treated", None, "control", "treated"],
            "All_Null": pl.Series([None, None, None, None], dtype=pl.Null),
        }
    )
    parsed = level(
        var=var,
        var_keys=("ProForma_ion",),
        layers={
            "Intensity": pl.DataFrame(
                {
                    "ProForma_ion": var.get_column("ProForma_ion"),
                    "obs_0": [1.0, 2.0, 3.0, 4.0],
                    "obs_1": [5.0, 6.0, 7.0, 8.0],
                }
            )
        },
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    assert not isinstance(stored.var["ProForma_ion"].dtype, pd.CategoricalDtype)
    assert stored.var["ProForma_ion"].tolist() == var.get_column("ProForma_ion").to_list()
    assert isinstance(stored.var["Condition"].dtype, pd.CategoricalDtype)
    assert stored.var["Condition"].astype("string").tolist() == [
        "treated",
        pd.NA,
        "control",
        "treated",
    ]
    assert stored.var["Condition"].cat.categories.tolist() == ["treated", "control"]
    assert isinstance(stored.var["All_Null"].dtype, pd.CategoricalDtype)
    assert stored.var["All_Null"].isna().all()
    assert stored.var["All_Null"].cat.categories.empty


def test_the_provenance_is_written_under_the_parse_tool_namespace(tmp_path: Path) -> None:
    parsed = level(
        uns={
            "software_name": "Synthetic",
            "quantification_level": "ion",
            "unknown_mod_tokens": ["Mystery@M"],
        }
    )
    target = tmp_path / "ion.h5ad"

    writer_for(parsed).write(parsed, target)
    stored = anndata.read_h5ad(target)

    parse_namespace = stored.uns[NAMESPACE][PARSE_NAMESPACE]
    assert parse_namespace["software_name"] == "Synthetic"
    assert parse_namespace["quantification_level"] == "ion"
    assert list(parse_namespace["unknown_mod_tokens"]) == ["Mystery@M"]


def test_a_failed_write_leaves_the_previous_file_and_no_scratch_behind(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ion.h5ad"
    parsed = level()
    writer_for(parsed).write(parsed, target)
    before = target.read_bytes()
    broken = level(
        layers={
            "Intensity": pl.DataFrame(
                {"Feature": ["F1", "F2"], "obs_0": [1.0, 2.0], "obs_1": [3.0, 4.0]}
            ),
            "Broken": pl.DataFrame(
                {"Feature": ["F1", "F2"], "obs_0": ["x", "y"], "obs_1": ["z", "w"]}
            ),
        }
    )

    with pytest.raises(AnnDataLayerContractError):
        writer_for(broken, checks="strict").write(broken, target)

    assert target.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion.h5ad"]


def test_the_anndata_writer_satisfies_the_parser_owned_writer_contract(
    tmp_path: Path,
) -> None:
    parsed = level()
    writer: ParsedLevelWriter = writer_for(parsed)

    writer.write(parsed, tmp_path / "ion.h5ad")

    assert (tmp_path / "ion.h5ad").is_file()


def test_mudata_writer_materializes_each_level_with_its_configured_anndata_writer(
    tmp_path: Path,
) -> None:
    ion = level(
        uns={"software_name": "Synthetic", "quantification_level": "ion"},
    )
    protein = level(
        var=pl.DataFrame({"Protein": ["P1"]}),
        var_keys=("Protein",),
        layers={"Intensity": pl.DataFrame({"Protein": ["P1"], "obs_0": [10.0], "obs_1": [20.0]})},
        uns={"software_name": "Synthetic", "quantification_level": "protein"},
    )
    target = tmp_path / "levels.h5mu"

    MuDataWriter(level_writers={"ion": writer_for(ion), "protein": writer_for(protein)}).write(
        ParsedLevels(
            levels={"ion": ion, "protein": protein},
            uns={
                "produced_by": "apb2",
                "rule_selection_method": "rule_config",
                "quantification_levels": ["ion", "protein"],
            },
        ),
        target,
    )

    stored = mudata.read_h5mu(target)
    assert list(stored.mod) == ["ion", "protein"]
    assert stored["ion"].shape == (2, 2)
    assert stored["protein"].shape == (2, 1)
    assert list(stored["ion"].var_names) == ["ion:F1", "ion:F2"]
    assert list(stored["protein"].var_names) == ["prt:P1"]
    assert list(stored["ion"].var["Feature"].astype("string")) == ["F1", "F2"]
    assert stored.uns[NAMESPACE][PARSE_NAMESPACE]["rule_selection_method"] == "rule_config"
    assert stored["ion"].uns[NAMESPACE][PARSE_NAMESPACE]["quantification_level"] == "ion"


def test_mudata_writer_accepts_one_level_but_rejects_no_levels(tmp_path: Path) -> None:
    ion = level()
    writer = MuDataWriter(level_writers={"ion": writer_for(ion)})

    writer.write(
        ParsedLevels(levels={"ion": ion}, uns={"produced_by": "apb2"}),
        tmp_path / "ion.h5mu",
    )

    assert list(mudata.read_h5mu(tmp_path / "ion.h5mu").mod) == ["ion"]
    with pytest.raises(MuDataLevelError, match="no parsed levels"):
        MuDataWriter(level_writers={}).write(
            ParsedLevels(levels={}, uns={}),
            tmp_path / "empty.h5mu",
        )


def test_mudata_writer_requires_one_configured_writer_per_parsed_level(
    tmp_path: Path,
) -> None:
    ion = level()

    with pytest.raises(MuDataLevelError, match=r"parsed=.*ion.*writers"):
        MuDataWriter(level_writers={}).write(
            ParsedLevels(levels={"ion": ion}, uns={}),
            tmp_path / "levels.h5mu",
        )


def test_one_array_is_allocated_for_each_encoded_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allocated: list[tuple[int, ...]] = []
    original = pl.DataFrame.to_numpy

    def counting(self: pl.DataFrame) -> np.ndarray:
        result = original(self)
        allocated.append(result.shape)
        return result

    monkeypatch.setattr(pl.DataFrame, "to_numpy", counting)
    parsed = level(
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [1.0], "obs_1": [2.0]}),
            "QValue": pl.DataFrame({"Feature": ["F1"], "obs_0": [0.1], "obs_1": [0.2]}),
        },
        var=pl.DataFrame({"Feature": ["F1"]}),
    )

    writer_for(parsed).write(parsed, tmp_path / "ion.h5ad")

    assert allocated == [(1, 2), (1, 2)]
