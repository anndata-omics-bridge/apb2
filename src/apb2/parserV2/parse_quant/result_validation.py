"""Storage-neutral invariants every physical result writer enforces before staging."""

from __future__ import annotations

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import ParsedLevel, ParsedLevels
from apb2.parserV2.parse_quant.errors import InvalidResultError

_PAIRWISE_COLUMNS = ("row", "column", "value")


def validate_parsed_levels(parsed: ParsedLevels, /) -> None:
    """Validate collection, axis, layer, aligned-frame, and coordinate invariants."""
    if not parsed.levels:
        raise InvalidResultError("a persisted result must contain at least one level")
    for name, level in parsed.levels.items():
        _validate_level(name, level)


def _validate_level(name: str, parsed: ParsedLevel) -> None:
    if parsed.primary_layer_name not in parsed.layers:
        raise InvalidResultError(
            f"level {name!r} has no primary layer {parsed.primary_layer_name!r}"
        )
    _require_columns(name, "obs keys", parsed.obs.frame, parsed.obs.key_columns)
    _require_columns(name, "var keys", parsed.var.frame, parsed.var.key_columns)
    for layer_name, layer in parsed.layers.items():
        if layer.layer_name != layer_name:
            raise InvalidResultError(
                f"level {name!r} stores layer {layer_name!r} with internal name "
                f"{layer.layer_name!r}"
            )
        _require_columns(name, f"layer {layer_name!r} keys", layer.values, layer.var_key_columns)
        if layer.values.height != parsed.var.frame.height:
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} has {layer.values.height} rows; "
                f"var has {parsed.var.frame.height}"
            )
        value_count = layer.values.width - len(layer.var_key_columns)
        if value_count != parsed.obs.frame.height:
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} has {value_count} observation "
                f"columns; obs has {parsed.obs.frame.height} rows"
            )
    _validate_aligned(name, "obsm", parsed.obsm, parsed.obs.frame.height)
    _validate_aligned(name, "varm", parsed.varm, parsed.var.frame.height)
    _validate_pairwise(name, "obsp", parsed.obsp, parsed.obs.frame.height)
    _validate_pairwise(name, "varp", parsed.varp, parsed.var.frame.height)


def _require_columns(
    level: str,
    role: str,
    frame: pl.DataFrame,
    required: tuple[str, ...],
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InvalidResultError(f"level {level!r} {role} are missing column(s) {missing}")


def _validate_aligned(
    level: str,
    slot: str,
    frames: dict[str, pl.DataFrame],
    expected_rows: int,
) -> None:
    for name, frame in frames.items():
        if frame.height != expected_rows:
            raise InvalidResultError(
                f"level {level!r} {slot}[{name!r}] has {frame.height} rows; "
                f"the axis has {expected_rows}"
            )


def _validate_pairwise(
    level: str,
    slot: str,
    frames: dict[str, pl.DataFrame],
    axis_size: int,
) -> None:
    for name, frame in frames.items():
        if tuple(frame.columns) != _PAIRWISE_COLUMNS:
            raise InvalidResultError(
                f"level {level!r} {slot}[{name!r}] must have exactly "
                f"{list(_PAIRWISE_COLUMNS)}, got {frame.columns}"
            )
        if frame.select(pl.struct("row", "column").is_duplicated().any()).item():
            raise InvalidResultError(
                f"level {level!r} {slot}[{name!r}] repeats a matrix coordinate"
            )
        positions = frame.select("row", "column").unpivot().get_column("value")
        invalid = positions.is_null() | (positions < 0) | (positions >= axis_size)
        if invalid.any():
            raise InvalidResultError(
                f"level {level!r} {slot}[{name!r}] has a coordinate outside [0, {axis_size})"
            )
