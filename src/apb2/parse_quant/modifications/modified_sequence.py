"""What normalizing one vendor sequence produces: the peptide and its modifications.

Both classes are results the normalizer builds, never storage read from anywhere — nothing
validates or dumps them — so they are plain classes, not pydantic models. One
``ModifiedSequence`` is built per *distinct* source value, which is also why per-field
validation has no business here: an accession reaching an occurrence has already been
resolved through the Unimod registry when the applier was constructed.
"""

from __future__ import annotations


class ModificationOccurrence:
    """One localized modification on a peptide.

    ``sequence_index`` is 0-based into the stripped sequence and absent for a terminal or
    unlocalized modification; ``position`` (``"Anywhere"``, ``"N-term"``, ``"C-term"``) is
    what the ProForma renderer groups by.
    """

    def __init__(
        self,
        *,
        name: str,
        accession: str | None = None,
        target_residue: str | None = None,
        sequence_index: int | None = None,
        position: str | None = None,
        mass_delta: float | None = None,
        source_token: str | None = None,
    ) -> None:
        if sequence_index is not None and sequence_index < 0:
            raise ValueError(f"sequence_index must be non-negative; got {sequence_index}")
        self.name = name
        self.accession = accession
        self.target_residue = target_residue
        self.sequence_index = sequence_index
        self.position = position
        self.mass_delta = mass_delta
        self.source_token = source_token


class ModifiedSequence:
    """A modified peptide as observed in one quantification result row."""

    def __init__(
        self,
        *,
        stripped_sequence: str,
        proforma_sequence: str,
        occurrences: tuple[ModificationOccurrence, ...] = (),
        source_sequence: str | None = None,
        unknown_tokens: tuple[str, ...] = (),
    ) -> None:
        self.stripped_sequence = stripped_sequence
        self.proforma_sequence = proforma_sequence
        self.occurrences = occurrences
        self.source_sequence = source_sequence
        self.unknown_tokens = unknown_tokens
