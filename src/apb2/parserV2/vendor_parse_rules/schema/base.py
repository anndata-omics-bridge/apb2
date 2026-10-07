"""Shared scalar declarations for the Parser V2 rules.json schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from apb2.parserV2.vendor_parse_rules.schema.hierarchy import HIERARCHIES

type TableShape = Literal["long", "wide"]
type SchemaVersion = Literal["0.8"]
type QuantificationLevel = str
type AxisColumnType = Literal["string", "integer", "number", "boolean"]
type DuplicateMode = Literal["error", "sum", "max", "keep_first", "keep_best"]
type TokenPosition = Literal[
    "before_residue", "after_residue", "n_term", "c_term", "embedded", "unknown"
]
type UnknownPolicy = Literal["preserve", "drop", "error"]

SCHEMA_VERSION: SchemaVersion = "0.8"

LEVELS: tuple[QuantificationLevel, ...] = tuple(
    dict.fromkeys(level for identities in HIERARCHIES.values() for level, _identity in identities)
)


class ModelBase(BaseModel):
    """Strict base for project-authored rule declarations."""

    model_config = ConfigDict(extra="forbid")
