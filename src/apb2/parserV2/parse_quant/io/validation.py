"""Storage-neutral invariants every physical result writer enforces before staging."""

from __future__ import annotations

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import ParsedLevel, ParsedLevels
from apb2.parserV2.parse_quant.io.errors import InvalidResultError

_PAIRWISE_COLUMNS = ("row", "column", "value")


def validate_parsed_levels(parsed: ParsedLevels, /) -> None:
    """Validate collection, axis, layer, aligned-frame, and coordinate invariants."""
    if not parsed.levels:
        raise InvalidResultError("a persisted result must contain at least one level")
    for name, level in parsed.levels.items():
        validate_parsed_level(name, level)


def validate_parsed_level(name: str, parsed: ParsedLevel, /) -> None:
    """Validate one level before a parser-owned writer projects or persists it."""
    if parsed.primary_layer_name not in parsed.layers:
        raise InvalidResultError(
            f"level {name!r} has no primary layer {parsed.primary_layer_name!r}"
        )
    if not parsed.layers[parsed.primary_layer_name].role.accepts_primary_layer():
        raise InvalidResultError(
            f"level {name!r} primary layer {parsed.primary_layer_name!r} is auxiliary"
        )
    _validate_axis_keys(name, "obs", parsed.obs.frame, parsed.obs.key_columns)
    _validate_axis_keys(name, "var", parsed.var.frame, parsed.var.key_columns)
    for layer_name, layer in parsed.layers.items():
        if layer.layer_name != layer_name:
            raise InvalidResultError(
                f"level {name!r} stores layer {layer_name!r} with internal name "
                f"{layer.layer_name!r}"
            )
        if layer.var_key_columns != parsed.var.key_columns:
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} declares var keys "
                f"{list(layer.var_key_columns)}; var declares {list(parsed.var.key_columns)}"
            )
        leading_columns = tuple(layer.values.columns[: len(layer.var_key_columns)])
        if leading_columns != layer.var_key_columns:
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} must begin with var keys "
                f"{list(layer.var_key_columns)}, got {list(leading_columns)}"
            )
        if layer.values.height != parsed.var.frame.height:
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} has {layer.values.height} rows; "
                f"var has {parsed.var.frame.height}"
            )
        layer_keys = layer.values.select(list(layer.var_key_columns))
        var_keys = parsed.var.frame.select(list(parsed.var.key_columns))
        if not layer_keys.equals(var_keys):
            raise InvalidResultError(
                f"level {name!r} layer {layer_name!r} var keys do not match var row-for-row"
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


def _validate_axis_keys(
    level: str,
    role: str,
    frame: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> None:
    if not key_columns:
        raise InvalidResultError(f"level {level!r} {role} has no authored key columns")
    _require_columns(level, f"{role} keys", frame, key_columns)
    keys = frame.select(list(key_columns))
    missing = [
        pl.col(name).is_null() | (pl.col(name).is_nan() if dtype.is_float() else pl.lit(False))
        for name, dtype in keys.schema.items()
    ]
    if keys.select(pl.any_horizontal(missing).any()).item():
        raise InvalidResultError(f"level {level!r} {role} contains an incomplete key")
    if keys.is_duplicated().any():
        raise InvalidResultError(f"level {level!r} {role} contains a duplicate key")


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
