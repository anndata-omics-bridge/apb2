"""Plain level names and JSON values used by compiler settings and provenance."""

from __future__ import annotations

from typing import Literal

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
"""The parsing-owned level vocabulary, structurally equal to the rule package's own."""

LEVELS: tuple[QuantificationLevel, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)
"""Canonical level order preserved by compiler selection."""
