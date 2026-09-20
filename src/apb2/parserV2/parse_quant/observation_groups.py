"""Separate observation resolutions, aligning only explicit bijective metadata mappings.

Layer columns are positional. Relabelling observation keys without changing row order
therefore preserves every quantitative cell, including missing values.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)


def group_observations(parsed: ParsedLevels) -> tuple[ParsedLevels, ...]:
    """Return independently writable groups with compatible observation identities.

    Canonical level order determines the preferred observation identity. Different key
    declarations are joined only through complete, non-null, one-to-one metadata; labels
    themselves are never guessed. Observed relationships remain parse provenance in every
    returned result, including when fractionation prevents alignment.
    """
    groups: dict[tuple[str, ...], dict[ParsedLevelName, ParsedLevel]] = {}
    for name in LEVEL_ORDER:
        if name in parsed.levels:
            level = parsed.levels[name]
            groups.setdefault(level.obs.key_columns, {})[name] = level
    if len(groups) <= 1:
        return (parsed,)
    relationships: list[JsonValue] = []
    for keys, levels in tuple(groups.items()):
        if keys not in groups:
            continue
        for other_keys, other_levels in tuple(groups.items()):
            if keys == other_keys:
                continue
            mapping = _relationship(levels, keys, other_keys)
            if mapping is None:
                continue
            aligned = _align(other_levels, mapping, keys, other_keys)
            relationships.append(
                {
                    "from_keys": list(keys),
                    "to_keys": list(other_keys),
                    "rows": cast("JsonValue", json.loads(mapping.write_json())),
                    "aligned": bool(aligned),
                }
            )
            if aligned:
                levels.update(aligned)
                del groups[other_keys]
    return tuple(
        replace(
            parsed,
            levels={name: levels[name] for name in LEVEL_ORDER if name in levels},
            uns={
                **parsed.uns,
                "observation_keys": list(keys),
                "observation_relationships": json.dumps(relationships),
            },
            feature_relations={
                name: relation
                for name, relation in parsed.feature_relations.items()
                if relation.target_level in levels
            },
        )
        for keys, levels in groups.items()
    )


def _relationship(
    levels: dict[ParsedLevelName, ParsedLevel],
    keys: tuple[str, ...],
    other_keys: tuple[str, ...],
) -> pl.DataFrame | None:
    columns = tuple(dict.fromkeys((*keys, *other_keys)))
    frames = [
        level.obs.frame.select(columns)
        for level in levels.values()
        if set(columns) <= set(level.obs.frame.columns)
    ]
    if not frames or any(frame.schema != frames[0].schema for frame in frames):
        return None
    return pl.concat(frames).unique(maintain_order=True)


def _align(
    levels: dict[ParsedLevelName, ParsedLevel],
    mapping: pl.DataFrame,
    keys: tuple[str, ...],
    other_keys: tuple[str, ...],
) -> dict[ParsedLevelName, ParsedLevel]:
    if (
        mapping.is_empty()
        or any(mapping.null_count().row(0))
        or mapping.select(keys).is_duplicated().any()
        or mapping.select(other_keys).is_duplicated().any()
    ):
        return {}
    aligned: dict[ParsedLevelName, ParsedLevel] = {}
    additions = tuple(key for key in keys if key not in other_keys)
    for name, level in levels.items():
        frame = level.obs.frame
        if any(frame.schema[key] != mapping.schema[key] for key in other_keys):
            return {}
        # Do not overwrite an already authored metadata column with a different value.
        if set(additions) & set(frame.columns):
            return {}
        joined = frame.join(
            mapping, on=other_keys, how="inner", maintain_order="left", validate="m:1"
        )
        if joined.height != frame.height:
            return {}
        aligned[name] = replace(
            level,
            obs=ObsFinal(frame=joined, key_columns=keys),
            uns={**level.uns, "observation_keys_original": list(other_keys)},
        )
    return aligned
