"""Shared scalar declarations for the Parser V2 rules.json schema."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from apb2.parserV2.vendor_parse_rules.schema.hierarchy import HIERARCHIES

type TableShape = Literal["long", "wide"]
type SchemaVersion = Literal["0.11"]
type QuantificationLevel = str
type AxisColumnType = Literal["string", "integer", "number", "boolean"]
type DuplicateMode = Literal["error", "same", "sum", "keep_best"]
type PerFeature = Literal["first", "min", "max", "best_row"]
type TokenPosition = Literal[
    "before_residue", "after_residue", "n_term", "c_term", "embedded", "unknown"
]
type UnknownPolicy = Literal["preserve", "drop", "error"]

SCHEMA_VERSION: SchemaVersion = "0.11"

LEVELS: tuple[QuantificationLevel, ...] = tuple(
    dict.fromkeys(level for identities in HIERARCHIES.values() for level, _identity in identities)
)


class ModelBase(BaseModel):
    """Strict base for project-authored rule declarations."""

    model_config = ConfigDict(extra="forbid")


type Note = Annotated[
    str | None,
    Field(
        min_length=1,
        description="Why the entry is placed or read as it is, for the rule's readers; "
        "parsing ignores it",
    ),
]


def _compiled_version(pattern: str | None) -> str | None:
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"version is not a valid regex: {error}") from error
    return pattern


type VersionPattern = Annotated[
    str | None,
    Field(
        min_length=1,
        description="The software versions whose files carry the entry's source, a regex "
        "searched in the version as software_version_pattern is; entries of one name, each "
        "with a version, read one column the vendor spells differently across versions",
    ),
    AfterValidator(_compiled_version),
]
