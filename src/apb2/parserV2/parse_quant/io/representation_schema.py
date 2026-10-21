"""The JSON Schema of the format-version-5 result representation written beside every result.

The models mirror the dictionaries ``json_representation`` builds; they only validate. Structural
objects forbid unknown keys. Tool metadata (``apb`` parts and annotation ``metadata``) stays open,
except that producer records follow the summary and details contract of ``RecordContract``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal, NotRequired

from loguru import logger
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, with_config

# typing_extensions for PEP 728 ``extra_items``, which ``typing`` gains only in Python 3.15.
from typing_extensions import TypedDict

from apb2.parserV2.parse_quant.data.parsed import JsonScalar, JsonValue
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.search_parameters.model import Parameters

SCHEMA_PACKAGE = "apb2.parserV2.parse_quant.io"
SCHEMA_TITLE = "APB result representation, format version 5"

_STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
_OPEN = ConfigDict(strict=True, allow_inf_nan=False)

type Count = Annotated[int, Field(ge=0)]
type Name = Annotated[str, Field(min_length=1)]
type NonObject = JsonScalar | list[JsonValue]


@with_config(_STRICT)
class CheckedEntry(TypedDict):
    """A summary entry that was checked: fine or needing attention."""

    name: Name
    label: str
    value: JsonScalar
    unit: str
    status: Literal["ok", "attention"]
    layer: NotRequired[str]


@with_config(_STRICT)
class UncheckedEntry(TypedDict):
    """A summary entry that was not checked; it has no value."""

    name: Name
    label: str
    value: None
    unit: str
    status: Literal["not_checked"]
    layer: NotRequired[str]


type SummaryEntry = Annotated[CheckedEntry | UncheckedEntry, Field(discriminator="status")]


@with_config(_STRICT)
class DetailReference(TypedDict):
    """Where a record's detailed result is stored in the scientific artifact."""

    slot: Literal["varm", "layers", "annotation_tables", "feature_relations"]
    name: Name


@with_config(_OPEN)
class NamedRecord(TypedDict, extra_items=JsonValue):
    """A record one level below its producer, such as ``catalog.identification_confidence``."""

    summary: NotRequired[list[SummaryEntry]]
    details: NotRequired[list[DetailReference]]


@with_config(_OPEN)
class ProducerRecord(TypedDict, extra_items=NamedRecord | NonObject):
    """One producer's record: provenance, results, summary entries and detail references."""

    schema_version: NotRequired[JsonValue]
    provenance: NotRequired[JsonValue]
    result: NotRequired[JsonValue]
    summary: NotRequired[list[SummaryEntry]]
    details: NotRequired[list[DetailReference]]


@with_config(_OPEN)
class ApbPart(TypedDict, extra_items=ProducerRecord | NonObject):
    """A level ``apb`` part: APB2's own entries and one record per producer."""

    hierarchy: NotRequired[dict[str, JsonValue]]
    roles: NotRequired[dict[str, JsonValue]]
    layer_descriptors: NotRequired[dict[str, JsonValue]]
    annotation_tables: NotRequired[dict[str, JsonValue]]
    feature_relations: NotRequired[dict[str, JsonValue]]


@with_config(_STRICT)
class SearchParametersRecord(TypedDict):
    """The parameter file a conversion read: base name, SHA-256 of its bytes, typed record."""

    file_name: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    parameters: Parameters


@with_config(_OPEN)
class RootParseProvenance(TypedDict, extra_items=JsonValue):
    """Conversion-wide parse provenance; absent keys mean the input was not supplied."""

    search_parameters: NotRequired[SearchParametersRecord]


@with_config(_OPEN)
class RootParseRecord(TypedDict, extra_items=NamedRecord | NonObject):
    """The root ``parse`` record: conversion-wide provenance and observation grouping."""

    schema_version: NotRequired[JsonValue]
    provenance: NotRequired[RootParseProvenance]
    result: NotRequired[JsonValue]
    summary: NotRequired[list[SummaryEntry]]
    details: NotRequired[list[DetailReference]]


@with_config(_OPEN)
class RootApbPart(TypedDict, extra_items=ProducerRecord | NonObject):
    """The root ``apb`` part: a level part's entries plus the typed root ``parse`` record."""

    parse: NotRequired[RootParseRecord]
    hierarchy: NotRequired[dict[str, JsonValue]]
    roles: NotRequired[dict[str, JsonValue]]
    layer_descriptors: NotRequired[dict[str, JsonValue]]
    annotation_tables: NotRequired[dict[str, JsonValue]]
    feature_relations: NotRequired[dict[str, JsonValue]]


@with_config(_STRICT)
class ApbRoot(TypedDict):
    """The container's own ``apb`` part."""

    apb: RootApbPart


@with_config(_STRICT)
class Artifact(TypedDict):
    """The scientific artifact this document describes."""

    name: str
    physical_format: Literal["h5ad", "h5mu", "parquet", "duckdb"]
    size_bytes: Count


@with_config(_STRICT)
class Column(TypedDict):
    """One table column: its name, null count and Polars dtype."""

    name: str
    null_count: Count
    dtype: str


@with_config(_STRICT)
class Table(TypedDict):
    """A table's schema without its rows."""

    row_count: Count
    key_columns: list[str]
    columns: list[Column]


@with_config(_STRICT)
class NamedTable(Table):
    """An aligned ``obsm``, ``varm``, ``obsp`` or ``varp`` table."""

    name: str


@with_config(_STRICT)
class Shape(TypedDict):
    """Observation and variable counts."""

    observations: Count
    variables: Count


@with_config(_STRICT)
class Observation(TypedDict):
    """One observation identity, referred to by index from layer summaries."""

    index: Count
    key: dict[str, JsonScalar]
    label: str


@with_config(_STRICT)
class Observations(TypedDict):
    """At most 100 observation identities."""

    total_count: Count
    emitted_count: Count
    truncated: bool
    items: list[Observation]


@with_config(_STRICT)
class Statistics(TypedDict):
    """Counts, moments, extrema and quartiles of the finite values; null when none exist."""

    total_count: Count
    finite_count: Count
    null_count: Count
    nan_count: Count
    positive_infinity_count: Count
    negative_infinity_count: Count
    zero_count: Count
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    first_quartile: float | None
    median: float | None
    third_quartile: float | None
    quartile_interpolation: Literal["linear"]
    quartile_method: Literal["linear_exact", "linear_deterministic_grid_sample"]
    quartile_sample_count: Count
    quartile_sample_limit: Count


@with_config(_STRICT)
class ObservationStatistics(TypedDict):
    """The statistics of one observation's values."""

    observation_index: Count
    statistics: Statistics


@with_config(_STRICT)
class ObservationSummaries(TypedDict):
    """Per-observation statistics for at most 100 observations."""

    total_count: Count
    emitted_count: Count
    truncated: bool
    items: list[ObservationStatistics]


@with_config(_STRICT)
class LayerBase(TypedDict):
    """What every layer records: role, primary status, storage slot, shape, unit and scale."""

    name: str
    role: Literal["measurement", "auxiliary"]
    primary: bool
    storage_slot: Literal["X", "layers"]
    shape: Shape
    unit: JsonScalar
    scale: JsonScalar


@with_config(_STRICT)
class QuantitativeLayer(LayerBase):
    """A numeric layer: logical type, physical dtype and bounded statistics."""

    value_kind: Literal["quantitative"]
    type: Literal["number", "integer"]
    dtype: str
    statistics: Statistics
    observation_summaries: ObservationSummaries


@with_config(_STRICT)
class CategoryCounts(TypedDict):
    """Cells holding a declared category, and the rest."""

    total_count: Count
    known_count: Count
    missing_or_unknown_count: Count


@with_config(_STRICT)
class CategoricalLayer(LayerBase):
    """A factor layer: category count and known versus missing cells, never statistics."""

    value_kind: Literal["categorical"]
    dtype: Literal["categorical"]
    category_count: Count
    counts: CategoryCounts


type Layer = Annotated[QuantitativeLayer | CategoricalLayer, Field(discriminator="value_kind")]


@with_config(_STRICT)
class Aligned(TypedDict):
    """The schemas of the aligned tables."""

    obsm: list[NamedTable]
    varm: list[NamedTable]
    obsp: list[NamedTable]
    varp: list[NamedTable]


@with_config(_STRICT)
class Level(TypedDict):
    """One quantification level: axes, layers, aligned tables and its ``apb`` part."""

    name: str
    dimensions: Shape
    primary_layer: str
    observations: Observations
    obs: Table
    var: Table
    layers: list[Layer]
    aligned: Aligned
    apb: ApbPart


@with_config(_STRICT)
class AnnotationTable(Table):
    """An annotation table's schema and metadata."""

    name: str
    metadata: dict[str, JsonValue]


@with_config(_STRICT)
class FeatureRelation(TypedDict):
    """A relation from an annotation table to a level's features."""

    name: str
    annotation_table: str
    target_level: str
    coordinates: Table
    metadata: dict[str, JsonValue]


@with_config(_STRICT)
class RepresentationDocument(TypedDict):
    """The JSON document written beside an APB result as ``<artifact>.apb.json``."""

    format: Literal["apb2-result-representation"]
    format_version: Literal["5"]
    artifact: Artifact | None
    root: ApbRoot | None
    levels: list[Level]
    annotation_tables: list[AnnotationTable]
    feature_relations: list[FeatureRelation]


_REPRESENTATION_ADAPTER: TypeAdapter[RepresentationDocument] = TypeAdapter(RepresentationDocument)


def validate_representation(document: Mapping[str, JsonValue]) -> None:
    """Refuse a document that does not match the format-version-5 schema."""
    try:
        _REPRESENTATION_ADAPTER.validate_python(document)
    except ValidationError as error:
        raise InvalidResultError(f"representation does not match its schema: {error}") from error


def representation_json_schema() -> dict[str, object]:
    """Return the JSON Schema of the format-version-5 result representation."""
    schema = _REPRESENTATION_ADAPTER.json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **schema,
        "title": SCHEMA_TITLE,
    }


def representation_schema_path() -> Path:
    """Where APB2 publishes ``representation.schema.json``."""
    return Path(str(resources.files(SCHEMA_PACKAGE))) / "_schema" / "representation.schema.json"


def write_representation_schema() -> Path:
    """Regenerate the published representation schema and return its path."""
    output = representation_schema_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(representation_json_schema(), indent=2) + "\n")
    logger.info("wrote {}", output)
    return output
