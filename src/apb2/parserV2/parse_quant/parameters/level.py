"""Plain level names and JSON values used by compiler settings and provenance."""

from __future__ import annotations

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type QuantificationLevel = str
"""Open quantification level names carried by compiled settings."""
