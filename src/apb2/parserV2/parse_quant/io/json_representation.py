"""Compact, deterministic scientific representation of an APB2 result.

The document is a view of ``ParsedLevels``, not another storage format. It records schemas,
dimensions, provenance, and bounded quantitative summaries while deliberately excluding axis
rows and matrix values.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath
from typing import cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    AnnotationTable,
    FeatureRelation,
    JsonScalar,
    JsonValue,
    ParsedLevel,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.io.anndata_writer import represent_layer_values
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.layer_representation import OBSERVATION_SUMMARY_LIMIT
from apb2.parserV2.parse_quant.io.metadata import (
    collection_shared_scope,
    column_descriptions,
    compose_metadata,
    level_scope,
    shared_scope,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels

FORMAT = "apb2-result-representation"
FORMAT_VERSION = "4"
SIDECAR_SUFFIX = ".apb.json"
_EMBEDDED_JSON_FIELDS = frozenset(
    {
        "aggregate",
        "plan_json",
        "rule_json",
    }
)


def sidecar_path(artifact: Path, /) -> Path:
    """Return the representation path adjacent to one scientific artifact."""
    return artifact.with_name(f"{artifact.name}{SIDECAR_SUFFIX}")


def project_result(
    parsed: ParsedLevels,
    artifact: Path | None = None,
    /,
) -> dict[str, JsonValue]:
    """Project a storage-neutral APB2 result into the version-4 JSON document.

    ``artifact`` is optional so in-memory clients can inspect and validate the scientific
    representation before selecting a physical format or output path. Sidecar publication
    always supplies it.
    """
    validate_parsed_levels(parsed)
    physical_format = artifact.suffix.lower() if artifact is not None else ""
    root = (
        collection_shared_scope(parsed)
        if physical_format == ".h5mu"
        else shared_scope(parsed.uns, parsed.metadata)
    )
    scopes = {name: level_scope(level) for name, level in parsed.levels.items()}
    root_document: JsonValue = {"apb": _portable(root)}
    if physical_format == ".h5ad":
        if len(scopes) != 1 or parsed.annotation_tables or parsed.feature_relations:
            raise InvalidResultError("H5AD representation requires one standalone level")
        name = next(iter(scopes))
        scopes[name], _ = compose_metadata(root, scopes[name])
        root_document = None
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "artifact": _artifact_descriptor(artifact),
        "root": root_document,
        "levels": [
            _level(name, parsed.levels[name], scopes[name])
            for name in LEVEL_ORDER
            if name in parsed.levels
        ],
        "annotation_tables": [
            _annotation_table(name, table) for name, table in parsed.annotation_tables.items()
        ],
        "feature_relations": [
            _feature_relation(name, relation) for name, relation in parsed.feature_relations.items()
        ],
    }


def write_result_representation(parsed: ParsedLevels, artifact: Path, /) -> Path:
    """Atomically write the compact JSON sidecar after its scientific artifact exists.

    The scientific artifact is intentionally not rolled back if projection or publication
    fails. Any previous sidecar is invalidated before projection so it cannot describe a
    newly replaced artifact after a failed representation write.
    """
    destination = sidecar_path(artifact)
    destination.unlink(missing_ok=True)
    document = project_result(parsed, artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_result_with_representation(
    parsed: ParsedLevels,
    artifact: Path,
    write_artifact: Callable[[], None],
    /,
) -> Path:
    """Write one scientific artifact, then publish its matching representation.

    Artifact writers remain responsible for their own atomic persistence. This composition
    guarantees that a sidecar is not invalidated until that persistence has succeeded.
    """
    write_artifact()
    return write_result_representation(parsed, artifact)


def _level(
    name: str, parsed: ParsedLevel, metadata: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    emitted_observations = min(parsed.obs.frame.height, OBSERVATION_SUMMARY_LIMIT)
    observations = _observations(parsed, emitted_observations)
    layers: list[JsonValue] = []
    for layer_name, layer in parsed.layers.items():
        representation = represent_layer_values(
            parsed,
            layer_name,
            observation_limit=emitted_observations,
        )
        layer_document: dict[str, JsonValue] = {
            "name": layer_name,
            "role": layer.role.persisted_name(),
            "primary": layer_name == parsed.primary_layer_name,
            "storage_slot": "X" if layer_name == parsed.primary_layer_name else "layers",
            "shape": {
                "observations": parsed.obs.frame.height,
                "variables": parsed.var.frame.height,
            },
            "unit": _layer_descriptor_value(parsed, layer_name, "unit"),
            "scale": _layer_descriptor_value(parsed, layer_name, "scale"),
        }
        layer_document.update(representation)
        layers.append(layer_document)
    return {
        "name": name,
        "dimensions": {
            "observations": parsed.obs.frame.height,
            "variables": parsed.var.frame.height,
        },
        "primary_layer": parsed.primary_layer_name,
        "observations": observations,
        "obs": _table(parsed.obs.frame, parsed.obs.key_columns),
        "var": _table(parsed.var.frame, parsed.var.key_columns),
        "layers": layers,
        "aligned": {
            "obsm": _named_tables(parsed.obsm),
            "varm": _named_tables(parsed.varm),
            "obsp": _named_tables(parsed.obsp),
            "varp": _named_tables(parsed.varp),
        },
        "apb": _portable(dict(metadata)),
    }


def _table(frame: pl.DataFrame, key_columns: tuple[str, ...] = ()) -> dict[str, JsonValue]:
    return {
        "row_count": frame.height,
        "key_columns": list(key_columns),
        "columns": cast(list[JsonValue], column_descriptions(frame)),
    }


def _named_tables(frames: Mapping[str, pl.DataFrame]) -> list[JsonValue]:
    return [{"name": name, **_table(frame)} for name, frame in frames.items()]


def _annotation_table(name: str, table: AnnotationTable) -> dict[str, JsonValue]:
    return {
        "name": name,
        **_table(table.frame, table.key_columns),
        "metadata": _portable(table.metadata),
    }


def _feature_relation(name: str, relation: FeatureRelation) -> dict[str, JsonValue]:
    return {
        "name": name,
        "annotation_table": relation.annotation_table,
        "target_level": relation.target_level,
        "coordinates": _table(relation.coordinates),
        "metadata": _portable(relation.metadata),
    }


def _observations(parsed: ParsedLevel, limit: int) -> dict[str, JsonValue]:
    keys = parsed.obs.frame.select(list(parsed.obs.key_columns)).head(limit).iter_rows(named=True)
    items: list[JsonValue] = []
    for index, key in enumerate(keys):
        scalar_key = {name: _json_scalar(value) for name, value in key.items()}
        portable_key: dict[str, JsonValue] = dict(scalar_key)
        items.append(
            {
                "index": index,
                "key": portable_key,
                "label": _observation_label(scalar_key),
            }
        )
    return {
        "total_count": parsed.obs.frame.height,
        "emitted_count": len(items),
        "truncated": len(items) < parsed.obs.frame.height,
        "items": items,
    }


def _observation_label(key: Mapping[str, JsonScalar]) -> str:
    if len(key) == 1:
        value = next(iter(key.values()))
        return "null" if value is None else str(value)
    return " · ".join(f"{name}={value}" for name, value in key.items())


def _layer_descriptor_value(parsed: ParsedLevel, layer_name: str, field: str) -> JsonScalar:
    """Read optional scientific terms from the level's extension metadata convention."""
    descriptors = parsed.metadata.get("layer_descriptors")
    if not isinstance(descriptors, dict):
        return None
    descriptor = descriptors.get(layer_name)
    if not isinstance(descriptor, dict):
        return None
    value = descriptor.get(field)
    return (
        _json_scalar(value)
        if isinstance(value, bool | int | float | str) or value is None
        else None
    )


def _portable(value: JsonValue, field_name: str | None = None) -> JsonValue:
    if field_name in _EMBEDDED_JSON_FIELDS and isinstance(value, str):
        structured = _json_container(value)
        if structured is not None:
            return _portable(structured)
    if isinstance(value, dict):
        return {name: _portable(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_portable(item, field_name) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and _is_path_field(field_name) and _is_absolute_path(value):
        return _path_name(value)
    return value


def _json_container(value: str) -> dict[str, JsonValue] | list[JsonValue] | None:
    """Decode one known embedded JSON container while retaining scalar or invalid text."""
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict | list):
        return None
    return cast(dict[str, JsonValue] | list[JsonValue], decoded)


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _is_path_field(field_name: str | None) -> bool:
    if field_name is None:
        return False
    normalized = field_name.casefold()
    return normalized in {"path", "paths"} or normalized.endswith(("_path", "_paths"))


def _path_name(value: str) -> str:
    return PureWindowsPath(value).name if PureWindowsPath(value).is_absolute() else Path(value).name


def _json_scalar(value: object) -> JsonScalar:
    if value is None or isinstance(value, bool | int | float | str):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
    raise FileNotFoundError(path)


def _artifact_descriptor(artifact: Path | None) -> JsonValue:
    if artifact is None:
        return None
    return {
        "name": artifact.name,
        "physical_format": artifact.suffix.removeprefix(".").lower(),
        "size_bytes": _path_size(artifact),
    }
