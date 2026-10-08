"""Canonical layer parsing and structural result-writer laws."""

from __future__ import annotations

import ast
import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import anndata
import mudata
import numpy as np
import pandas as pd
import polars as pl
import pytest

from apb2.parserV2.parse_quant.contracts import LayerValueParser
from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    MeasurementLayerRole,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.errors import LayerContractError, LayerValueError
from apb2.parserV2.parse_quant.io.anndata_writer import (
    AnnDataWriter,
    H5adWriter,
    MuDataLevelError,
    MuDataWriter,
)
from apb2.parserV2.parse_quant.io.errors import (
    InvalidResultError,
)
from apb2.parserV2.parse_quant.io.metadata import (
    NAMESPACE,
    PARSE_NAMESPACE,
    STORAGE_NAMESPACE,
)
from apb2.parserV2.parse_quant.io.parquet_reader import ParquetReader
from apb2.parserV2.parse_quant.io.parquet_writer import (
    MANIFEST_NAME,
    ParquetLevelsWriter,
)
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.operations import make_layer_parser
from apb2.parserV2.parse_quant.parameters.measurements import (
    FactorLayerDeclaration,
    PlainNumericLayerDeclaration,
    RegexNumericLayerDeclaration,
)
from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
GROUPED = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))


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
    metadata: dict[str, JsonValue] = {"provenance": {"software_name": "Synthetic"}}
    metadata.update(uns or {})
    return ParsedLevel(
        obs=ObsFinal(frame=obs_frame, key_columns=obs_keys),
        var=VarFinal(frame=var_frame, key_columns=var_keys),
        primary_layer_name=primary,
        uns=metadata,
        layers={
            name: FinalLayerTable(
                layer_name=name,
                values=(frame).drop(var_keys, strict=False),
                semantic_roles=("abundance",),
            )
            for name, frame in values.items()
        },
        obsm={},
        varm={},
        obsp={},
        varp={},
    )


def one(parsed: ParsedLevel) -> ParsedLevels:
    """Wrap one level as the one-level result the writers persist."""
    return ParsedLevels(levels={"ion": parsed}, uns={})


def manifest_of(target: Path) -> dict[str, JsonValue]:
    payload = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_a_final_layer_is_a_measurement_unless_its_role_is_explicit() -> None:
    intensity = level().layers["Intensity"]

    assert isinstance(intensity.role, MeasurementLayerRole)


# ---------------------------------------------------------------------------------- Parquet


def test_a_parquet_dataset_round_trips_every_value_and_dtype(tmp_path: Path) -> None:
    parsed = level(
        obs=pl.DataFrame({"Run": ["A"]}),
        var=pl.DataFrame({"Feature": ["F1", "F2"], "Charge": [2, None], "Decoy": [True, None]}),
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1", "F2"], "obs_0": [100000.5, None]}),
            "Kind": pl.DataFrame({"Feature": ["F1", "F2"], "obs_0": [2, 1]}),
        },
    )
    parsed.layers["Kind"].role = AuxiliaryLayerRole()
    parsed.layers["Kind"].semantics = CategoricalLayerSemantics(
        categories=(("MS/MS", 1), ("MBR", 2))
    )
    target = tmp_path / "ion"

    ParquetLevelsWriter().write(one(parsed), target)
    restored = ParquetReader().read(target).levels["ion"]

    assert restored.obs.frame.equals(parsed.obs.frame)
    assert restored.var.frame.schema == parsed.var.frame.schema
    intensity = restored.layers["Intensity"].values
    assert intensity.get_column("obs_0").to_list() == [100000.5, None]
    assert intensity.schema["obs_0"] == pl.Float64
    assert restored.layers["Kind"].values.get_column("obs_0").to_list() == [
        2,
        1,
    ]
    assert restored.layers["Kind"].semantics == parsed.layers["Kind"].semantics


def test_the_manifest_states_what_every_file_is(tmp_path: Path) -> None:
    parsed = level(
        obs=pl.DataFrame({"Run": ["A"]}),
        var_keys=("Feature", "Charge"),
        var=pl.DataFrame({"Feature": ["F1"], "Charge": [2]}),
        uns={"result": {"unknown_mod_tokens": ["Mystery@M"]}},
        layers={
            "Intensity": pl.DataFrame({"Feature": ["F1"], "Charge": [2], "obs_0": [1.0]}),
            "Q Value": pl.DataFrame({"Feature": ["F1"], "Charge": [2], "obs_0": [0.01]}),
        },
    )
    target = tmp_path / "ion"

    ParquetLevelsWriter().write(one(parsed), target)
    manifest = manifest_of(target)

    assert manifest["format_version"] == "7"
    assert manifest["level_order"] == ["ion"]
    levels = manifest["levels"]
    assert isinstance(levels, dict)
    ion = levels["ion"]
    assert isinstance(ion, dict)
    assert ion["primary_layer"] == "Intensity"
    assert ion["layer_order"] == ["Intensity", "Q Value"]
    apb = ion["apb"]
    assert isinstance(apb, dict)
    assert apb["parse"] == {
        "provenance": {"software_name": "Synthetic"},
        "result": {"unknown_mod_tokens": ["Mystery@M"]},
    }
    layers = ion["layers"]
    assert isinstance(layers, dict)
    assert layers["Q Value"] == {
        "file": "Q_Value.parquet",
        "columns": ["obs_0"],
        "schema": [{"name": "Float64"}],
        "role": "measurement",
        "semantics": {"kind": "quantitative", "logical_type": "number"},
    }
    assert (target / "levels" / "ion" / "layers" / "Q_Value.parquet").is_file()


def test_a_layer_name_is_mapped_to_a_file_name_never_interpolated(tmp_path: Path) -> None:
    parsed = level(
        obs=pl.DataFrame({"Run": ["A"]}),
        var=pl.DataFrame({"Feature": ["F1"]}),
        layers={
            "../escape": pl.DataFrame({"Feature": ["F1"], "obs_0": [1.0]}),
            "..%escape": pl.DataFrame({"Feature": ["F1"], "obs_0": [2.0]}),
            "Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [3.0]}),
        },
    )
    target = tmp_path / "ion"

    ParquetLevelsWriter().write(one(parsed), target)
    manifest = manifest_of(target)
    levels = manifest["levels"]
    assert isinstance(levels, dict)
    ion = levels["ion"]
    assert isinstance(ion, dict)
    layers = ion["layers"]

    assert isinstance(layers, dict)
    layer_directory = target / "levels" / "ion" / "layers"
    files = sorted(path.name for path in layer_directory.iterdir())
    assert files == ["Intensity.parquet", "escape.parquet", "escape_1.parquet"]
    assert not (tmp_path / "escape.parquet").exists()
    for name, entry in layers.items():
        assert isinstance(entry, dict)
        assert Path(str(entry["file"])).name == str(entry["file"])
        assert name not in files or name.endswith("Intensity")


def test_writing_over_an_existing_dataset_leaves_only_the_new_one(tmp_path: Path) -> None:
    target = tmp_path / "ion"
    ParquetLevelsWriter().write(one(level()), target)
    layer_directory = target / "levels" / "ion" / "layers"
    (layer_directory / "Stale.parquet").write_bytes(b"stale")

    ParquetLevelsWriter().write(one(level()), target)

    assert sorted(path.name for path in layer_directory.iterdir()) == ["Intensity.parquet"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion"]


def test_a_target_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "ion"
    target.write_text("not a dataset", encoding="utf-8")

    with pytest.raises(InvalidResultError, match="not a directory"):
        ParquetLevelsWriter().write(one(level()), target)

    assert target.read_text(encoding="utf-8") == "not a dataset"


def test_a_failure_part_way_through_leaves_the_previous_dataset_intact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ion"
    ParquetLevelsWriter().write(one(level()), target)
    before = (target / MANIFEST_NAME).read_bytes()
    broken = level(
        obs=pl.DataFrame({"Run": ["A"]}),
        var=pl.DataFrame({"Feature": ["F1"]}),
        layers={"Intensity": pl.DataFrame({"Feature": ["F1"], "obs_0": [object()]})},
    )

    with pytest.raises(Exception, match=r".*"):
        ParquetLevelsWriter().write(one(broken), target)

    assert (target / MANIFEST_NAME).read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion"]


def test_the_parquet_writer_imports_no_encoder_backend() -> None:
    source = Path("src/apb2/parserV2/parse_quant/io/parquet_writer.py")
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


# ---------------------------------------------------------------------------------- encoders


def block(*columns: list[object]) -> pl.DataFrame:
    return pl.DataFrame(
        {f"obs_{index}": values for index, values in enumerate(columns)}, strict=False
    )


def canonical_values(
    parser: LayerValueParser,
    values: pl.DataFrame,
) -> pl.DataFrame:
    layer = FinalLayerTable(
        layer_name="L", values=(values).drop((), strict=False), semantic_roles=("abundance",)
    )
    return parser.parse(layer)[0].values


def test_plain_numeric_encoding_reads_numbers_and_blanks_out_the_sentinel() -> None:
    encoder = make_layer_parser(
        "Intensity", PlainNumericLayerDeclaration(missing_values=(0.0,)), DOT
    )

    encoded = canonical_values(encoder, block(["12.5", "0", "", None]))

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]
    assert encoded.schema["obs_0"] == pl.Float64
    assert encoded.columns == ["obs_0"]


def test_a_localized_number_is_read_under_the_notation_it_was_written_in() -> None:
    encoder = make_layer_parser(
        "Intensity", PlainNumericLayerDeclaration(missing_values=()), GROUPED
    )

    encoded = canonical_values(encoder, block(["100.000.000", "1.234,5", None]))

    assert encoded.get_column("obs_0").to_list() == [100000000.0, 1234.5, None]


def test_a_token_a_plain_numeric_layer_cannot_hold_becomes_missing_and_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The vendors these rules describe write ``-``, ``NA``, and ``False`` in such a column.

    Refusing the file would convert nothing; the encoded-layer contract is what decides
    whether enough values survived, and this reports the tokens that did not.
    """
    encoder = make_layer_parser("Intensity", PlainNumericLayerDeclaration(missing_values=()), DOT)

    encoded = canonical_values(encoder, block(["12.5", "not a number", "-", "NA"]))

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]


def test_numeric_diagnostics_count_all_cells_and_tokens_before_bounding_examples() -> None:
    parser = make_layer_parser(
        "Intensity", PlainNumericLayerDeclaration(missing_values=(0.0,)), DOT
    )
    invalid = [f"bad{index}" for index in range(7)]
    values = pl.DataFrame(
        {
            "obs_0": [*invalid, "bad0", "", None, "0", "1.5"],
            "obs_1": [*invalid, "bad7", "   ", None, "0", "2.5"],
        }
    )
    parsed, evidence = parser.parse(FinalLayerTable(layer_name="Intensity", values=values))
    assert evidence == {
        "cell_count": 16,
        "distinct_token_count": 8,
        "examples": ["bad0", "bad1", "bad2", "bad3", "bad4"],
    }
    assert parsed.values.get_column("obs_0").to_list() == [*([None] * 11), 1.5]


@pytest.mark.parametrize(
    "declaration",
    [
        PlainNumericLayerDeclaration(missing_values=(), missing_tokens=("-", "NA")),
        RegexNumericLayerDeclaration(
            missing_values=(), pattern=r"^(-?\d+(?:\.\d+)?)$", missing_tokens=("-", "NA")
        ),
    ],
)
def test_a_declared_missing_token_is_missing_and_not_reported_as_unreadable(
    declaration: PlainNumericLayerDeclaration | RegexNumericLayerDeclaration,
) -> None:
    parser = make_layer_parser("Intensity", declaration, DOT)

    parsed, evidence = parser.parse(
        FinalLayerTable(layer_name="Intensity", values=block(["12.5", "-", " NA ", "n/a"]))
    )

    assert parsed.values.get_column("obs_0").to_list() == [12.5, None, None, None]
    assert evidence == {"cell_count": 1, "distinct_token_count": 1, "examples": ["n/a"]}


def test_regex_numeric_diagnostics_count_nonblank_failed_captures() -> None:
    parser = make_layer_parser(
        "AScore",
        RegexNumericLayerDeclaration(
            missing_values=(0.0,),
            pattern=r":(-?\d+(?:\.\d+)?)(?:;|$)",
        ),
        DOT,
    )
    parsed, evidence = parser.parse(
        FinalLayerTable(
            layer_name="AScore",
            values=block(
                ["x:12.5", "x:0", "unstructured", "unstructured", "wrong", "", "  ", None]
            ),
        )
    )
    assert evidence == {
        "cell_count": 3,
        "distinct_token_count": 2,
        "examples": ["unstructured", "wrong"],
    }
    assert parsed.values.get_column("obs_0").to_list() == [12.5, *([None] * 7)]


def test_an_already_numeric_column_is_not_sent_through_its_own_text_form() -> None:
    """A float32 round-tripped through text is not the value it was; numbers stay numbers."""
    encoder = make_layer_parser("Intensity", PlainNumericLayerDeclaration(missing_values=()), DOT)
    values = pl.DataFrame({"obs_0": pl.Series([1268453.25], dtype=pl.Float32)})

    encoded = canonical_values(encoder, values)

    assert encoded.get_column("obs_0").to_list() == [1268453.25]


def test_regex_encoding_extracts_the_number_and_treats_no_match_as_missing() -> None:
    encoder = make_layer_parser(
        "AScore",
        RegexNumericLayerDeclaration(missing_values=(0.0,), pattern=r":(-?\d+(?:\.\d+)?)(?:;|$)"),
        DOT,
    )

    encoded = canonical_values(
        encoder, block(["S4:Phospho:12.5", "S4:Phospho:0", "unstructured", None])
    )

    assert encoded.get_column("obs_0").to_list() == [12.5, None, None, None]


def test_integer_encoding_accepts_whole_values_null_nan_and_unreadable_tokens() -> None:
    encoder = make_layer_parser(
        "MS_MS_Count", PlainNumericLayerDeclaration(missing_values=(), type="integer"), DOT
    )

    encoded = canonical_values(encoder, block([1, 2.0, None, float("nan"), "unreadable"]))
    values = encoded.get_column("obs_0").to_list()

    assert values[:4] == [1, 2, None, None]
    assert values[4] is None
    assert encoded.schema["obs_0"] == pl.Int64


@pytest.mark.parametrize("invalid", [1.5, float("inf"), float("-inf")])
def test_integer_encoding_rejects_fractional_and_infinite_values(invalid: float) -> None:
    encoder = make_layer_parser(
        "MS_MS_Count", PlainNumericLayerDeclaration(missing_values=(), type="integer"), DOT
    )

    with pytest.raises(
        LayerValueError,
        match=r"integer layer 'MS_MS_Count'.*examples=",
    ):
        canonical_values(encoder, block([1.0, invalid, None]))


def test_integer_validation_bounds_reported_examples() -> None:
    encoder = make_layer_parser(
        "Spectral_Count",
        RegexNumericLayerDeclaration(missing_values=(), pattern=r"value=(\S+)", type="integer"),
        DOT,
    )

    with pytest.raises(LayerValueError) as error:
        canonical_values(encoder, block([f"value={value + 0.5}" for value in range(10)]))

    assert str(error.value).count(".5") == 5


def test_factor_encoding_maps_declared_labels_and_codes_the_rest_as_unknown() -> None:
    encoder = make_layer_parser(
        "Match_Type",
        FactorLayerDeclaration(categories=(("unmatched", 0), ("MS/MS", 1), ("MBR", 2))),
        DOT,
    )

    encoded = canonical_values(encoder, block(["MBR", "MS/MS", "surprise", None]))

    assert encoded.get_column("obs_0").to_list() == [2, 1, -1, -1]
    assert encoded.schema["obs_0"] == pl.Int64


def test_an_encoder_preserves_the_shape_and_the_column_order_it_was_given() -> None:
    encoder = make_layer_parser("Intensity", PlainNumericLayerDeclaration(missing_values=()), DOT)
    values = block(["1"], ["2"], ["3"])

    encoded = canonical_values(encoder, values)

    assert encoded.columns == values.columns
    assert encoded.shape == values.shape


# ------------------------------------------------------------------------- contract checks


def contract(*required: str, primary: str = "Intensity") -> LayerContractValidator:
    return LayerContractValidator(
        primary_layer_name=primary,
        required_names=required or (primary,),
        empty_ratio=0.001,
        populated_ratio=0.5,
        strict=False,
    )


def validate_layers(
    values: Mapping[str, pl.DataFrame],
    *,
    checks: Literal["standard", "strict"] = "standard",
    auxiliary: tuple[str, ...] = (),
    config: LayerContractValidator | None = None,
) -> None:
    layers = {
        name: FinalLayerTable(
            layer_name=name,
            values=(frame).drop((), strict=False),
            role=AuxiliaryLayerRole() if name in auxiliary else MeasurementLayerRole(),
        )
        for name, frame in values.items()
    }
    selected = config or contract()
    LayerContractValidator(
        primary_layer_name=selected.primary_layer_name,
        required_names=selected.required_names,
        empty_ratio=selected.empty_ratio,
        populated_ratio=selected.populated_ratio,
        strict=checks == "strict",
    ).validate(layers)


def test_layer_contract_errors_belong_to_the_parse_error_hierarchy() -> None:
    assert issubclass(LayerContractError, ValueError)


def test_a_missing_required_auxiliary_layer_is_still_a_contract_error() -> None:
    encoded = {"Intensity": block([1.0])}

    with pytest.raises(LayerContractError, match="ObservationCount"):
        validate_layers(encoded, config=contract("Intensity", "ObservationCount"))


def test_an_empty_primary_layer_beside_a_populated_sibling_is_an_error() -> None:
    encoded = {
        "Intensity": block([None, None, None, None]),
        "QValue": block([0.1, 0.2, 0.3, 0.4]),
    }

    with pytest.raises(LayerContractError, match="effectively empty"):
        validate_layers(encoded)


def test_an_empty_nonprimary_measurement_only_warns_unless_the_check_is_strict() -> None:
    encoded = {
        "Intensity": block([0.1, 0.2, 0.3, 0.4]),
        "QValue": block([None, None, None, None]),
    }

    validate_layers(encoded)
    with pytest.raises(LayerContractError, match="QValue"):
        validate_layers(encoded, checks="strict")


def test_nonfatal_empty_layer_diagnostics_record_occupancy_and_reference() -> None:
    layers = {
        "Intensity": FinalLayerTable(layer_name="Intensity", values=block([1.0, 2.0])),
        "QValue": FinalLayerTable(layer_name="QValue", values=block([None, None])),
    }
    assert contract().validate(layers) == {
        "QValue": {
            "occupancy": 0.0,
            "empty_ratio": 0.001,
            "populated_ratio": 0.5,
            "reference_layers": ["Intensity"],
        }
    }


@pytest.mark.parametrize("checks", ["standard", "strict"])
def test_a_populated_auxiliary_layer_does_not_make_an_empty_primary_suspicious(
    checks: Literal["standard", "strict"],
) -> None:
    encoded = {
        "Intensity": block([None, None, None, None]),
        "ObservationCount": block([2.0, 3.0, 4.0, 5.0]),
    }

    validate_layers(encoded, checks=checks, auxiliary=("ObservationCount",))


@pytest.mark.parametrize("checks", ["standard", "strict"])
def test_an_empty_auxiliary_layer_is_not_an_occupancy_failure(
    checks: Literal["standard", "strict"],
) -> None:
    encoded = {
        "Intensity": block([0.1, 0.2, 0.3, 0.4]),
        "ObservationCount": block([None, None, None, None]),
    }

    validate_layers(encoded, checks=checks, auxiliary=("ObservationCount",))


def test_without_a_populated_sibling_occupancy_invents_no_conclusion() -> None:
    encoded = {"Intensity": block([None, None]), "QValue": block([None, None])}

    validate_layers(encoded, checks="strict")


def test_a_factor_layer_of_unknown_codes_still_counts_as_populated() -> None:
    encoded = {
        "Intensity": block([0.1, 0.2, 0.3, 0.4]),
        "Match_Type": pl.DataFrame({"obs_0": [-1, -1, -1, -1]}),
    }

    validate_layers(encoded, checks="strict")


# ------------------------------------------------------------------------------- the writer


def write_h5ad(parsed: ParsedLevel, target: Path) -> None:
    H5adWriter().write(one(parsed), target)


def test_an_auxiliary_layer_cannot_be_the_primary_matrix(tmp_path: Path) -> None:
    parsed = level()
    parsed.layers["Intensity"].role = AuxiliaryLayerRole()
    target = tmp_path / "ion.h5ad"

    with pytest.raises(InvalidResultError, match=r"primary layer.*is auxiliary"):
        write_h5ad(parsed, target)

    assert not target.exists()


@pytest.mark.parametrize("checks", ["standard", "strict"])
def test_writer_excludes_an_auxiliary_layer_from_occupancy_comparisons(
    checks: str,
    tmp_path: Path,
) -> None:
    parsed = level(
        layers={
            "Intensity": pl.DataFrame(
                {
                    "Feature": ["F1", "F2"],
                    "obs_0": [None, None],
                    "obs_1": [None, None],
                }
            ),
            "ObservationCount": pl.DataFrame(
                {
                    "Feature": ["F1", "F2"],
                    "obs_0": [2.0, 3.0],
                    "obs_1": [4.0, 5.0],
                }
            ),
        }
    )
    parsed.layers["ObservationCount"].role = AuxiliaryLayerRole()
    target = tmp_path / f"{checks}.h5ad"

    write_h5ad(parsed, target)

    assert target.is_file()


@pytest.mark.parametrize("checks", ["standard", "strict"])
def test_writer_does_not_repeat_parse_time_occupancy_checks(
    checks: str,
    tmp_path: Path,
) -> None:
    parsed = level(
        layers={
            "Intensity": pl.DataFrame(
                {
                    "Feature": ["F1", "F2"],
                    "obs_0": [None, None],
                    "obs_1": [None, None],
                }
            ),
            "QValue": pl.DataFrame(
                {
                    "Feature": ["F1", "F2"],
                    "obs_0": [0.1, 0.2],
                    "obs_1": [0.3, 0.4],
                }
            ),
        }
    )
    target = tmp_path / f"{checks}.h5ad"

    write_h5ad(parsed, target)

    assert target.exists()


def test_the_parser_owned_anndata_writer_validates_layer_key_alignment(tmp_path: Path) -> None:
    parsed = level()
    parsed.layers["Intensity"].values = parsed.layers["Intensity"].values.head(1)
    target = tmp_path / "ion.h5ad"

    with pytest.raises(InvalidResultError, match="rows"):
        write_h5ad(parsed, target)

    assert not target.exists()


def test_the_written_object_is_observations_by_variables_with_the_primary_layer_as_x(
    tmp_path: Path,
) -> None:
    parsed = level()
    target = tmp_path / "ion.h5ad"

    write_h5ad(parsed, target)
    stored = anndata.read_h5ad(target)

    assert stored.shape == (2, 2)
    assert np.array_equal(
        np.asarray(stored.X), np.array([[1.0, 2.0], [3.0, np.nan]]), equal_nan=True
    )
    assert {name for name in stored.layers.keys() if name is not None} == set()  # noqa: SIM118


def test_the_primary_layer_is_stored_only_in_x(
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

    write_h5ad(parsed, target)
    stored = anndata.read_h5ad(target)

    named = {name for name in stored.layers.keys() if name is not None}  # noqa: SIM118
    assert sorted(named) == ["QValue"]
    assert np.array_equal(np.asarray(stored.X), np.array([[1.0], [2.0]]))


def test_h5ad_namespaces_have_one_scientific_owner_and_one_storage_descriptor(
    tmp_path: Path,
) -> None:
    parsed = level()
    target = tmp_path / "ion.h5ad"

    write_h5ad(parsed, target)
    stored = anndata.read_h5ad(target)

    assert set(stored.uns[NAMESPACE]) == {PARSE_NAMESPACE, STORAGE_NAMESPACE}
    apb = stored.uns["ion"][NAMESPACE]
    assert set(apb) == {PARSE_NAMESPACE, "roles", STORAGE_NAMESPACE}
    storage = json.loads(apb[STORAGE_NAMESPACE])
    assert storage["layers"] == [
        {
            "location": "X",
            "name": "Intensity",
            "role": "measurement",
            "semantics": {"kind": "quantitative", "logical_type": "number"},
            "value_columns": ["obs_0", "obs_1"],
        }
    ]
    assert "column_roles" not in storage
    assert "layer_roles" not in storage


def test_every_authored_key_stays_an_ordinary_column_beside_the_storage_index(
    tmp_path: Path,
) -> None:
    parsed = level(
        var=pl.DataFrame({"Feature": ["F1", "F2"], "Gene": ["G1", "G2"]}),
        obs=pl.DataFrame({"Run": ["A", "B"], "Fraction": ["1", "2"]}),
    )
    target = tmp_path / "ion.h5ad"

    write_h5ad(parsed, target)
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

    write_h5ad(parsed, target)
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

    write_h5ad(parsed, target)
    stored = anndata.read_h5ad(target)

    assert len(set(stored.var.index)) == 2
    assert list(stored.var.index) == [
        '[["String","a_b"],["String","c"]]',
        '[["String","a"],["String","b_c"]]',
    ]
    assert stored.var.index.name == "First_Second"


def test_bulk_axis_conversion_preserves_scalar_spelling_and_json_escaping() -> None:
    frame = pl.DataFrame(
        {
            "string": ['a"\\\n\t雪', "other"],
            "bool": [True, False],
            "float": [1e-5, -0.0],
            "int": [2, -3],
            "datetime": [
                dt.datetime(2020, 1, 2, 3, 4, 5),
                dt.datetime(2020, 1, 2, 3, 4, 5, 123000),
            ],
            "null": [None, None],
        }
    )
    keys = frame.select(pl.exclude("null"))
    parsed = level(
        var=frame,
        var_keys=tuple(keys.columns),
        layers={"Intensity": keys.with_columns(obs_0=pl.lit(1.0), obs_1=pl.lit(2.0))},
    )
    with pytest.warns(pl.exceptions.PolarsInefficientMapWarning):
        converted = AnnDataWriter().to_anndata_for_level(parsed, "ion").var
    expected = [
        json.dumps(
            [
                [str(dtype), None if value is None else str(value)]
                for dtype, value in zip(keys.dtypes, row, strict=True)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in keys.iter_rows()
    ]
    assert converted.index.tolist() == expected
    assert converted["datetime"].tolist() == ["2020-01-02 03:04:05", "2020-01-02 03:04:05.123000"]
    assert converted["bool"].dtype == pd.BooleanDtype()
    assert converted["int"].dtype == pd.Int64Dtype()


def test_bulk_axis_conversion_preserves_empty_typed_columns() -> None:
    frame = pl.DataFrame(
        schema={"key": pl.String, "flag": pl.Boolean, "count": pl.Int32, "value": pl.Float32}
    )
    parsed = level(
        var=frame,
        var_keys=("key",),
        layers={
            "Intensity": frame.select("key").with_columns(obs_0=pl.lit(1.0), obs_1=pl.lit(2.0))
        },
    )
    converted = AnnDataWriter().to_anndata_for_level(parsed, "ion").var
    assert isinstance(converted, pd.DataFrame)
    assert converted.empty
    assert converted.dtypes.astype(str).tolist() == ["string", "boolean", "Int64", "float64"]


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
        write_h5ad(parsed, target)
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

    write_h5ad(parsed, target)
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

    write_h5ad(parsed, target)
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
    parsed = level(uns={"result": {"unknown_mod_tokens": ["Mystery@M"]}})
    target = tmp_path / "ion.h5ad"

    write_h5ad(parsed, target)
    stored = anndata.read_h5ad(target)

    parse_namespace = stored.uns["ion"][NAMESPACE][PARSE_NAMESPACE]
    assert parse_namespace["provenance"]["software_name"] == "Synthetic"
    assert list(parse_namespace["result"]["unknown_mod_tokens"]) == ["Mystery@M"]


def test_a_failed_write_leaves_the_previous_file_and_no_scratch_behind(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ion.h5ad"
    parsed = level()
    write_h5ad(parsed, target)
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

    with pytest.raises(InvalidResultError, match="not numeric"):
        write_h5ad(broken, target)

    assert target.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ion.h5ad"]


def test_mudata_writer_materializes_each_canonical_level(
    tmp_path: Path,
) -> None:
    ion = level(
        uns={"hierarchy": "lfq", "software_name": "Synthetic", "quantification_level": "ion"},
    )
    protein = level(
        var=pl.DataFrame({"Protein": ["P1"]}),
        var_keys=("Protein",),
        layers={"Intensity": pl.DataFrame({"Protein": ["P1"], "obs_0": [10.0], "obs_1": [20.0]})},
        uns={"hierarchy": "lfq", "software_name": "Synthetic", "quantification_level": "protein"},
    )
    target = tmp_path / "levels.h5mu"

    MuDataWriter().write(
        ParsedLevels(
            levels={"ion": ion, "protein": protein},
            uns={
                "produced_by": "apb2",
                "rule_selection_method": "rule_config",
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
    writer = MuDataWriter()

    writer.write(
        ParsedLevels(levels={"ion": ion}, uns={"produced_by": "apb2"}),
        tmp_path / "ion.h5mu",
    )

    assert list(mudata.read_h5mu(tmp_path / "ion.h5mu").mod) == ["ion"]
    with pytest.raises(MuDataLevelError, match="no parsed levels"):
        MuDataWriter().write(
            ParsedLevels(levels={}, uns={}),
            tmp_path / "empty.h5mu",
        )


def test_mudata_writer_is_not_configured_per_level(
    tmp_path: Path,
) -> None:
    del tmp_path

    assert not hasattr(MuDataWriter(), "level_writers")


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

    write_h5ad(parsed, tmp_path / "ion.h5ad")

    assert allocated == [(1, 2), (1, 2)]
