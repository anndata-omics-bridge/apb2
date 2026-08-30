"""Lossless APB2 result persistence as a versioned Parquet directory dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    FinalLayerTable,
    JsonValue,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.metadata import (
    PARQUET_FORMAT,
    PARQUET_FORMAT_VERSION,
    PARQUET_LEVELS_DIRECTORY,
    PARQUET_MANIFEST_NAME,
    safe_names,
    table_metadata,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels

FORMAT = PARQUET_FORMAT
FORMAT_VERSION = PARQUET_FORMAT_VERSION
MANIFEST_NAME = PARQUET_MANIFEST_NAME
LEVELS_DIRECTORY = PARQUET_LEVELS_DIRECTORY


@dataclass(frozen=True, slots=True)
class ParquetWriter:
    """Parser-owned single-level writer using the collection format on disk."""

    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        level = _level_name(parsed)
        ParquetLevelsWriter().write(ParsedLevels(levels={level: parsed}, uns={}), target)


@dataclass(frozen=True, slots=True)
class ParquetLevelsWriter:
    """Write one or more parsed levels without translating any Polars scalar."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        level_directories = safe_names(parsed.levels, prefix="level", suffix="")
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch_text:
            scratch = Path(scratch_text)
            staged = scratch / target.name
            (staged / LEVELS_DIRECTORY).mkdir(parents=True)
            level_metadata: dict[str, JsonValue] = {}
            for name, level in parsed.levels.items():
                directory = level_directories[name]
                level_metadata[name] = _write_level(
                    level,
                    staged / LEVELS_DIRECTORY / directory,
                    directory,
                )
            manifest: dict[str, JsonValue] = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "level_order": list(parsed.levels),
                "levels": level_metadata,
                "uns": dict(parsed.uns),
            }
            (staged / MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _replace_directory(staged, target, scratch)


def _write_level(parsed: ParsedLevel, directory: Path, physical_name: str) -> dict[str, JsonValue]:
    directory.mkdir()
    obs = table_metadata(parsed.obs.frame, "obs.parquet")
    var = table_metadata(parsed.var.frame, "var.parquet")
    parsed.obs.frame.write_parquet(directory / "obs.parquet")
    parsed.var.frame.write_parquet(directory / "var.parquet")
    return {
        "directory": physical_name,
        "primary_layer": parsed.primary_layer_name,
        "obs": {**obs, "key_columns": list(parsed.obs.key_columns)},
        "var": {**var, "key_columns": list(parsed.var.key_columns)},
        "layer_order": list(parsed.layers),
        "layers": _write_layers(parsed.layers, directory / "layers"),
        "obsm_order": list(parsed.obsm),
        "obsm": _write_named_frames(parsed.obsm, directory / "obsm"),
        "varm_order": list(parsed.varm),
        "varm": _write_named_frames(parsed.varm, directory / "varm"),
        "obsp_order": list(parsed.obsp),
        "obsp": _write_named_frames(parsed.obsp, directory / "obsp"),
        "varp_order": list(parsed.varp),
        "varp": _write_named_frames(parsed.varp, directory / "varp"),
        "uns": dict(parsed.uns),
    }


def _write_layers(layers: Mapping[str, FinalLayerTable], directory: Path) -> dict[str, JsonValue]:
    directory.mkdir()
    names = safe_names(layers, prefix="layer", suffix=".parquet")
    result: dict[str, JsonValue] = {}
    for name, layer in layers.items():
        layer.values.write_parquet(directory / names[name])
        result[name] = {
            **table_metadata(layer.values, names[name]),
            "var_key_columns": list(layer.var_key_columns),
            "role": layer.role.persisted_name(),
        }
    return result


def _write_named_frames(
    frames: Mapping[str, pl.DataFrame],
    directory: Path,
) -> dict[str, JsonValue]:
    directory.mkdir()
    names = safe_names(frames, prefix="table", suffix=".parquet")
    result: dict[str, JsonValue] = {}
    for name, frame in frames.items():
        frame.write_parquet(directory / names[name])
        result[name] = table_metadata(frame, names[name])
    return result


def _level_name(parsed: ParsedLevel) -> ParsedLevelName:
    value = parsed.uns.get("quantification_level")
    if not isinstance(value, str) or value not in LEVEL_ORDER:
        raise InvalidResultError(
            "a parser-owned Parquet write requires uns['quantification_level']"
        )
    return value


def _replace_directory(staged: Path, target: Path, scratch: Path) -> None:
    if target.exists() and not target.is_dir():
        raise InvalidResultError(
            f"{target} exists and is not a directory; an APB2 Parquet result is a directory"
        )
    previous = scratch / f"{target.name}.previous"
    if target.exists():
        logger.info(f"replacing the existing Parquet result at {target}")
        target.replace(previous)
    try:
        staged.replace(target)
    except OSError:
        if previous.exists():
            previous.replace(target)
        raise
