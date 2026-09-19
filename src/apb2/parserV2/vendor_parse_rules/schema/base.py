"""Shared scalar declarations for the Parser V2 rules.json schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

type TableShape = Literal["long", "wide"]
type SchemaVersion = Literal["0.8"]
type QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
type AxisColumnType = Literal["string", "integer", "number", "boolean"]
type DuplicateMode = Literal["error", "aggregate", "keep_first"]
type TokenPosition = Literal[
    "before_residue", "after_residue", "n_term", "c_term", "embedded", "unknown"
]
type UnknownPolicy = Literal["preserve", "drop", "error"]

SCHEMA_VERSION: SchemaVersion = "0.8"

LEVELS: tuple[QuantificationLevel, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)


class ModelBase(BaseModel):
    """Strict base for project-authored rule declarations."""

    model_config = ConfigDict(extra="forbid")
