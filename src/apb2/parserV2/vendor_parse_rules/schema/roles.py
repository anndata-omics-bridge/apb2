"""Semantic rule roles and the declaration kinds permitted to own them."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal


class SemanticRole(StrEnum):
    """A semantic name that downstream consumers use to find declared data."""

    PROTEIN_ASSIGNMENT = "protein_assignment"
    FASTA_ACCESSIONS = "fasta_accessions"


type RoleOwner = Literal["obs", "var", "layer"]

ROLE_CONFIG: Final[dict[SemanticRole, frozenset[RoleOwner]]] = {
    SemanticRole.PROTEIN_ASSIGNMENT: frozenset({"var"}),
    SemanticRole.FASTA_ACCESSIONS: frozenset({"var"}),
}
"""The declaration kinds on which each semantic role is valid."""


def role_is_allowed(role: SemanticRole, owner: RoleOwner) -> bool:
    """Whether ``owner`` may carry ``role`` under the central role configuration."""
    return owner in ROLE_CONFIG[role]
