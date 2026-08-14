"""ProForma sequence rendering from modification occurrences."""

from __future__ import annotations

from apb2.modifications.model import ModificationOccurrence


def render_proforma(
    stripped: str,
    occurrences: list[ModificationOccurrence],
    unknown_tokens: dict[int, str],
) -> str:
    """Build a ProForma 2.0 string from a stripped sequence + modifications.

    Parameters
    ----------
    stripped
        Unmodified amino acid sequence.
    occurrences
        Localized modifications. ``sequence_index`` is 0-based into
        ``stripped``; ``position`` may be ``"N-term"`` / ``"C-term"`` for
        terminal modifications (then ``sequence_index`` is ignored).
    unknown_tokens
        Mapping ``{sequence_index: original_vendor_token}`` for
        unresolved tokens. Index ``-1`` denotes N-term, ``len(stripped)``
        denotes C-term. Pass an empty mapping when every token was resolved.

    Notes
    -----
    Mods at the same residue are concatenated (``M[Oxidation][Acetyl]``).
    Preferred label per occurrence: accession when present
    (``[UNIMOD:35]``), else name (``[Oxidation]``).
    """
    nterm, cterm, by_residue = _group_occurrences(occurrences)
    _add_unknown_tokens(
        unknown_tokens,
        sequence_length=len(stripped),
        nterm=nterm,
        cterm=cterm,
        by_residue=by_residue,
    )

    out: list[str] = []
    if nterm:
        out.append("[" + "][".join(nterm) + "]-")
    for i, residue in enumerate(stripped):
        out.append(residue)
        if i in by_residue:
            out.append("[" + "][".join(by_residue[i]) + "]")
    if cterm:
        out.append("-[" + "][".join(cterm) + "]")
    return "".join(out)


def _group_occurrences(
    occurrences: list[ModificationOccurrence],
) -> tuple[list[str], list[str], dict[int, list[str]]]:
    nterm: list[str] = []
    cterm: list[str] = []
    by_residue: dict[int, list[str]] = {}
    for occurrence in occurrences:
        tag = occurrence.accession or occurrence.name
        if occurrence.position == "N-term":
            nterm.append(tag)
        elif occurrence.position == "C-term":
            cterm.append(tag)
        elif occurrence.sequence_index is not None:
            by_residue.setdefault(occurrence.sequence_index, []).append(tag)
    return nterm, cterm, by_residue


def _add_unknown_tokens(
    unknown_tokens: dict[int, str],
    *,
    sequence_length: int,
    nterm: list[str],
    cterm: list[str],
    by_residue: dict[int, list[str]],
) -> None:
    for index, token in unknown_tokens.items():
        if index == -1:
            nterm.append(token)
        elif index == sequence_length:
            cterm.append(token)
        else:
            by_residue.setdefault(index, []).append(token)
