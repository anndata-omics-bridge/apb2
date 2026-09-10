"""Semantic rule roles and their configured declaration owners."""

from __future__ import annotations

from importlib import resources
from typing import Annotated, Literal

from pydantic import TypeAdapter, WithJsonSchema

type RoleOwner = Literal["obs", "var", "layer"]

ROLE_CONFIG = TypeAdapter(dict[RoleOwner, frozenset[str]]).validate_json(
    resources.files("apb2.parserV2.vendor_parse_rules.schema")
    .joinpath("role_policy.json")
    .read_bytes()
)
"""The semantic roles permitted on each declaration owner."""

_ROLE_NAMES = sorted({role for roles in ROLE_CONFIG.values() for role in roles})
type SemanticRole = Annotated[str, WithJsonSchema({"type": "string", "enum": _ROLE_NAMES})]
