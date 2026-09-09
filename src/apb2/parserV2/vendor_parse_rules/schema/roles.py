"""Semantic rule roles and the declaration kinds permitted to own them."""

from __future__ import annotations

from typing import Literal

type SemanticRole = Literal["protein_assignment", "fasta_accessions"]
type RoleOwner = Literal["obs", "var", "layer"]

ROLE_CONFIG: dict[SemanticRole, frozenset[RoleOwner]] = {
    "protein_assignment": frozenset({"var"}),
    "fasta_accessions": frozenset({"var"}),
}
"""The declaration kinds on which each semantic role is valid."""
