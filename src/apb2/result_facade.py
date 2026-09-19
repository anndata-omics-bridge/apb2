"""Public storage-neutral APB result and result-I/O boundary."""

from apb2.parserV2.parse_quant.data.layer_columns import observation_labels
from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FeatureRelation,
    FinalLayerTable,
    JsonValue,
    MeasurementLayerRole,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
    QuantitativeLayerSemantics,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.anndata_writer import quantitative_layer_values
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels
from apb2.parserV2.parse_quant.io.json_representation import (
    project_result,
    sidecar_path,
    write_result_representation,
)


def get_provenance(parsed: ParsedLevels, /) -> dict[str, JsonValue]:
    """Return existing shared and per-level provenance without interpreting it."""
    return {
        "shared": dict(parsed.uns),
        "levels": {name: dict(level.uns) for name, level in parsed.levels.items()},
    }


__all__ = [
    "AnnotationTable",
    "AuxiliaryLayerRole",
    "CategoricalLayerSemantics",
    "FeatureRelation",
    "FinalLayerTable",
    "JsonValue",
    "MeasurementLayerRole",
    "ObsFinal",
    "ParsedLevel",
    "ParsedLevelName",
    "ParsedLevels",
    "QuantitativeLayerSemantics",
    "VarFinal",
    "get_provenance",
    "observation_labels",
    "project_result",
    "quantitative_layer_values",
    "read_parsed_levels",
    "sidecar_path",
    "write_parsed_levels",
    "write_result_representation",
]
