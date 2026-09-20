"""Storage-neutral invariants every physical result writer enforces before staging."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    CategoricalLayerSemantics,
    FeatureRelation,
    ParsedLevel,
    ParsedLevels,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError

_PAIRWISE_COLUMNS = ("row", "column", "value")


def validate_parsed_levels(parsed: ParsedLevels, /) -> None:
    """Validate collection, axis, layer, aligned-frame, and coordinate invariants."""
    if not parsed.levels:
        raise InvalidResultError("a persisted result must contain at least one level")
    for name, level in parsed.levels.items():
        validate_parsed_level(name, level)
    for name, table in parsed.annotation_tables.items():
        if name in parsed.levels:
            raise InvalidResultError(f"annotation table {name!r} collides with a level")
        _validate_annotation_table(name, table)
    for name, relation in parsed.feature_relations.items():
        _validate_feature_relation(name, relation, parsed)


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
        _validate_layer_values(
            name,
            layer_name,
            layer.values.select(pl.exclude(layer.var_key_columns)),
            layer.semantics,
        )
    _validate_aligned(name, "obsm", parsed.obsm, parsed.obs.frame.height)
    _validate_aligned(name, "varm", parsed.varm, parsed.var.frame.height)
    _validate_pairwise(name, "obsp", parsed.obsp, parsed.obs.frame.height)
    _validate_pairwise(name, "varp", parsed.varp, parsed.var.frame.height)


def _validate_layer_values(
    level: str,
    layer: str,
    values: pl.DataFrame,
    semantics: QuantitativeLayerSemantics | CategoricalLayerSemantics,
    /,
) -> None:
    if isinstance(semantics, QuantitativeLayerSemantics):
        nonnumeric = values.select(~(cs.numeric() | cs.by_dtype(pl.Null))).columns
        if nonnumeric:
            raise InvalidResultError(
                f"level {level!r} quantitative layer {layer!r} is not numeric in "
                f"column(s) {nonnumeric}"
            )
        if (
            semantics.logical_type == "integer"
            and values.select(
                pl.any_horizontal(
                    pl.lit(False), (cs.float().fill_nan(None) != cs.float().floor()).any()
                )
            ).item()
        ):
            raise InvalidResultError(
                f"level {level!r} integer layer {layer!r} contains fractional values"
            )
        return
    noninteger = values.select(~(cs.integer() | cs.by_dtype(pl.Null))).columns
    if noninteger:
        raise InvalidResultError(
            f"level {level!r} categorical layer {layer!r} is not integer-coded in "
            f"column(s) {noninteger}"
        )
    valid = [code for _label, code in semantics.categories]
    if semantics.missing_code in valid:
        raise InvalidResultError(
            f"level {level!r} categorical layer {layer!r} reuses its missing code"
        )
    if values.select(
        pl.any_horizontal(pl.lit(False), (~pl.all().is_in([*valid, semantics.missing_code])).any())
    ).item():
        raise InvalidResultError(
            f"level {level!r} categorical layer {layer!r} contains undeclared codes"
        )


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
    if keys.fill_nan(None).select(pl.any_horizontal(pl.all().is_null()).any()).item():
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
        positions = pl.col("row", "column")
        invalid = positions.is_null() | (positions < 0) | (positions >= axis_size)
        if frame.select(pl.any_horizontal(invalid.any())).item():
            raise InvalidResultError(
                f"level {level!r} {slot}[{name!r}] has a coordinate outside [0, {axis_size})"
            )


def _validate_annotation_table(name: str, table: AnnotationTable) -> None:
    _validate_axis_keys(name, "annotation table", table.frame, table.key_columns)


def _validate_feature_relation(
    name: str,
    relation: FeatureRelation,
    parsed: ParsedLevels,
) -> None:
    try:
        source_size = parsed.annotation_tables[relation.annotation_table].frame.height
    except KeyError as error:
        raise InvalidResultError(
            f"feature relation {name!r} references unavailable annotation table "
            f"{relation.annotation_table!r}"
        ) from error
    try:
        target_size = parsed.levels[relation.target_level].var.frame.height
    except KeyError as error:
        raise InvalidResultError(
            f"feature relation {name!r} references unavailable target level "
            f"{relation.target_level!r}"
        ) from error
    frame = relation.coordinates
    if tuple(frame.columns) != _PAIRWISE_COLUMNS:
        raise InvalidResultError(
            f"feature relation {name!r} must have exactly {list(_PAIRWISE_COLUMNS)}, "
            f"got {frame.columns}"
        )
    if frame.select(pl.struct("row", "column").is_duplicated().any()).item():
        raise InvalidResultError(f"feature relation {name!r} repeats a matrix coordinate")
    if frame.select(cs.by_name("row", "column") & cs.integer()).width != 2:
        raise InvalidResultError(f"feature relation {name!r} coordinates must be integers")
    source, target = pl.col("row"), pl.col("column")
    invalid_source = source.is_null() | (source < 0) | (source >= source_size)
    invalid_target = target.is_null() | (target < 0) | (target >= target_size)
    if frame.select(pl.any_horizontal(invalid_source, invalid_target).any()).item():
        raise InvalidResultError(
            f"feature relation {name!r} has a coordinate outside its source or target axis"
        )
