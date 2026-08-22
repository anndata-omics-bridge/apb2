"""Normalize one vendor's modified sequences into ProForma, on a small axis frame.

Two layers, and the boundary between them is the point of the module. Underneath is the pure
sequence algorithm: take one vendor string — ``"PEPM[15.9949]TIDE"``, ``"_(ac)PEPTIDEM(ox)_"``,
or a bare sequence beside parallel name and site columns — and produce the localized
occurrences and their ProForma rendering. On top are the two normalizers the parser injects,
which select a column, run that algorithm once per *distinct* value, and hand back the derived
series.

Memoization is not an optimization detail, it is why normalization belongs on the var axis:
normalizing is a pure function of the source values, so a column with fifty thousand distinct
sequences tokenizes fifty thousand times whatever the measurement count is.

Map lookup uses the tuple ``(mass_delta, target, position)``, not mass alone, so Acetyl on a
protein N-terminus and Acetyl on a lysine stay distinguishable at the same mass. The entries
arrive already resolved: rule projection turned each accession into its canonical identity, so
nothing here reads a registry.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from apb2.parserV2.parse_quant.parameters.axis import (
    ModificationMapEntry,
    ModificationTokenPosition,
    UnknownModificationPolicy,
)

_MASS_TOLERANCE = 1e-3
_TERM_MARKERS = {"_", "-", "."}
_TERMINUS_TARGETS = {
    "N-term",
    "C-term",
    "Protein N-term",
    "Protein C-term",
    "Peptide N-term",
    "Peptide C-term",
}
_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_INTEGER_SITE = re.compile(r"^[+-]?\d+$")


class UnknownModificationError(ValueError):
    """A vendor token matched no declared modification and the rule refuses to guess."""


class PackedSiteMismatchError(ValueError):
    """A vendor row pairs a different number of modification names and sites."""


# --------------------------------------------------------------------- what normalizing yields


@dataclass(frozen=True, slots=True)
class ModificationOccurrence:
    """One localized modification on a peptide.

    ``sequence_index`` is 0-based into the stripped sequence and absent for a terminal or
    unlocalized modification; ``position`` is what the ProForma renderer groups by.
    """

    name: str
    accession: str
    position: str
    target_residue: str
    sequence_index: int
    source_token: str


@dataclass(frozen=True, slots=True)
class ModifiedSequence:
    """A modified peptide as observed in one quantification result row."""

    stripped_sequence: str
    proforma_sequence: str
    unknown_tokens: tuple[str, ...]


# ------------------------------------------------------------------------ ProForma rendering


_NO_INDEX = -2
"""Not a sequence position: a terminal or unlocalized occurrence has none."""


def _grouped_labels(
    occurrences: Sequence[ModificationOccurrence],
    unknown_tokens: dict[int, str],
    sequence_length: int,
) -> tuple[list[str], list[str], dict[int, list[str]]]:
    """Sort every label into the three places ProForma can put one.

    Resolved occurrences carry an accession or a name; an unresolved token carries itself,
    which is what the ``preserve`` policy means — the vendor's own spelling stays visible in
    the sequence instead of disappearing.
    """
    nterm: list[str] = []
    cterm: list[str] = []
    by_residue: dict[int, list[str]] = {}
    for occurrence in occurrences:
        tag = occurrence.accession or occurrence.name
        if occurrence.position == "N-term":
            nterm.append(tag)
        elif occurrence.position == "C-term":
            cterm.append(tag)
        elif occurrence.sequence_index != _NO_INDEX:
            by_residue.setdefault(occurrence.sequence_index, []).append(tag)
    for index, token in unknown_tokens.items():
        if index == -1:
            nterm.append(token)
        elif index == sequence_length:
            cterm.append(token)
        else:
            by_residue.setdefault(index, []).append(token)
    return nterm, cterm, by_residue


def render_proforma(
    stripped: str,
    occurrences: Sequence[ModificationOccurrence],
    unknown_tokens: dict[int, str],
) -> str:
    """Build a ProForma 2.0 string from a stripped sequence and its modifications.

    Modifications on one residue concatenate (``M[Oxidation][Acetyl]``). The preferred label
    is the accession when there is one, the name otherwise. ``unknown_tokens`` maps a
    sequence index to the original vendor token, with ``-1`` for the N-terminus and
    ``len(stripped)`` for the C-terminus.
    """
    nterm, cterm, by_residue = _grouped_labels(occurrences, unknown_tokens, len(stripped))
    out: list[str] = []
    if nterm:
        out.append("[" + "][".join(nterm) + "]-")
    for index, residue in enumerate(stripped):
        out.append(residue)
        if index in by_residue:
            out.append("[" + "][".join(by_residue[index]) + "]")
    if cterm:
        out.append("-[" + "][".join(cterm) + "]")
    return "".join(out)


# ---------------------------------------------------------------------------- where a token sits


@dataclass(frozen=True, slots=True)
class AdjacentResidue:
    """A location has an adjacent amino-acid residue."""

    value: str

    def carries(self, target: str) -> bool:
        """Whether a modification allowed on ``target`` could sit at this location."""
        return target == self.value


@dataclass(frozen=True, slots=True)
class NoAdjacentResidue:
    """A location has no adjacent amino-acid residue."""

    def carries(self, target: str) -> bool:
        """Nothing residue-specific can sit where there is no residue."""
        del target
        return False


@dataclass(frozen=True, slots=True)
class ResidueLocation:
    """A modification localized to one sequence residue."""

    sequence_index: int
    residue: str

    def target_position(self) -> str:
        return "Anywhere"

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return AdjacentResidue(self.residue)

    def occurrence(self, entry: ModificationMapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position="Anywhere",
            target_residue=self.residue,
            sequence_index=self.sequence_index,
            source_token=raw_token,
        )

    def record_unknown_token(
        self, unknown_tokens: dict[int, str], raw_token: str, sequence_length: int
    ) -> None:
        del sequence_length
        unknown_tokens[self.sequence_index] = raw_token


@dataclass(frozen=True, slots=True)
class TerminalLocation:
    """A terminal modification beside a concrete sequence residue."""

    position: Literal["N-term", "C-term"]
    adjacent_residue: str

    def target_position(self) -> str:
        return self.position

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return AdjacentResidue(self.adjacent_residue)

    def occurrence(self, entry: ModificationMapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position=self.position,
            target_residue=self.adjacent_residue,
            sequence_index=_NO_INDEX,
            source_token=raw_token,
        )

    def record_unknown_token(
        self, unknown_tokens: dict[int, str], raw_token: str, sequence_length: int
    ) -> None:
        unknown_tokens[_terminal_index(self.position, sequence_length)] = raw_token


@dataclass(frozen=True, slots=True)
class TerminalOnlyLocation:
    """A terminal modification with no residue association."""

    position: Literal["N-term", "C-term"]

    def target_position(self) -> str:
        return self.position

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return NoAdjacentResidue()

    def occurrence(self, entry: ModificationMapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position=self.position,
            target_residue="",
            sequence_index=_NO_INDEX,
            source_token=raw_token,
        )

    def record_unknown_token(
        self, unknown_tokens: dict[int, str], raw_token: str, sequence_length: int
    ) -> None:
        unknown_tokens[_terminal_index(self.position, sequence_length)] = raw_token


@dataclass(frozen=True, slots=True)
class UnlocalizedLocation:
    """A non-terminal token that cannot be attached to a residue."""

    def target_position(self) -> str:
        return "Anywhere"

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return NoAdjacentResidue()

    def occurrence(self, entry: ModificationMapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position="Anywhere",
            target_residue="",
            sequence_index=_NO_INDEX,
            source_token=raw_token,
        )

    def record_unknown_token(
        self, unknown_tokens: dict[int, str], raw_token: str, sequence_length: int
    ) -> None:
        """Record nothing: an unlocalized token has no index to attach to.

        The identity arm, stated rather than left as a chain ending in a bare ``elif``.
        """
        del unknown_tokens, raw_token, sequence_length


type ModificationLocation = (
    ResidueLocation | TerminalLocation | TerminalOnlyLocation | UnlocalizedLocation
)


def _terminal_index(position: Literal["N-term", "C-term"], sequence_length: int) -> int:
    """The index a terminal token occupies: before, or just past, the sequence."""
    return -1 if position == "N-term" else sequence_length


# ------------------------------------------------------------------------- matching one token


def _target_matches(entry_target: tuple[str, ...], location: ModificationLocation) -> bool:
    """Whether an entry's allowed targets are compatible with a token's context."""
    if not entry_target:
        return True
    position = location.target_position()
    adjacent = location.adjacent()
    for target in entry_target:
        if target in _TERMINUS_TARGETS:
            if target.endswith(position) or target == position:
                return True
        elif adjacent.carries(target):
            return True
    return False


def _entry_fits(entry: ModificationMapEntry, location: ModificationLocation) -> bool:
    return entry.position == location.target_position() and _target_matches(entry.target, location)


def _parsed_mass(raw: str) -> float | None:
    """The mass shift a token denotes, or nothing when it is not numeric."""
    cleaned = raw.strip().lstrip("+")
    return float(cleaned) if _NUMERIC_TOKEN.fullmatch(cleaned) else None


def _matched_entry(
    entries: tuple[ModificationMapEntry, ...],
    raw_token: str,
    location: ModificationLocation,
    *,
    case_sensitive: bool,
) -> ModificationMapEntry | None:
    """Pick the map entry one vendor token denotes at one location.

    A numeric token is matched on ``(mass_delta, target, position)``; anything else on its
    exact spelling. Nothing matched is an outcome, which the unknown-token policy decides
    what to do about.
    """
    mass = _parsed_mass(raw_token)
    if mass is not None:
        for entry in entries:
            if _entry_fits(entry, location) and math.isclose(
                entry.mass_delta, mass, abs_tol=_MASS_TOLERANCE
            ):
                return entry
    compared = raw_token if case_sensitive else raw_token.lower()
    for entry in entries:
        if not _entry_fits(entry, location):
            continue
        spelling = entry.token if case_sensitive else entry.token.lower()
        if spelling == compared:
            return entry
    return None


def _apply_unknown_policy(
    policy: UnknownModificationPolicy,
    raw_token: str,
    location: ModificationLocation,
    stripped_length: int,
    unknown_tokens: dict[int, str],
    unknown_token_list: list[str],
) -> None:
    """Apply the declared unknown-token policy to one unmatched token."""
    if policy == "error":
        raise UnknownModificationError(f"unknown modification token: {raw_token!r}")
    if policy == "drop":
        return
    unknown_token_list.append(raw_token)
    location.record_unknown_token(unknown_tokens, raw_token, stripped_length)


# -------------------------------------------------------------------------- inline token regex


@dataclass(frozen=True, slots=True)
class _PendingToken:
    raw_token: str
    location: ModificationLocation


@dataclass(frozen=True, slots=True)
class _TokenPlacement:
    """Where one token sits, and which residues were consumed while placing it."""

    location: ModificationLocation
    consumed_residues: tuple[str, ...]
    next_cursor: int


def _strip_terminal_markers(sequence: str) -> str:
    """Drop leading and trailing terminal markers (``_``, ``-``, ``.``)."""
    start, end = 0, len(sequence)
    while start < end and sequence[start] in _TERM_MARKERS:
        start += 1
    while end > start and sequence[end - 1] in _TERM_MARKERS:
        end -= 1
    return sequence[start:end]


def _place_token(
    sequence: str,
    match: re.Match[str],
    token_position: str,
    residues: list[str],
) -> _TokenPlacement:
    """Locate one regex match relative to the residues parsed before it."""
    if not residues and match.start() == 0:
        adjacent = sequence[match.end() : match.end() + 1]
        location: ModificationLocation = (
            TerminalLocation("N-term", adjacent) if adjacent else TerminalOnlyLocation("N-term")
        )
        return _TokenPlacement(location, (), match.end())
    if match.end() == len(sequence) and token_position != "before_residue":
        location = (
            TerminalLocation("C-term", residues[-1]) if residues else TerminalOnlyLocation("C-term")
        )
        return _TokenPlacement(location, (), match.end())
    if token_position == "before_residue":
        following = sequence[match.end() : match.end() + 1]
        if following.isalpha():
            return _TokenPlacement(
                ResidueLocation(len(residues), following), (following,), match.end() + 1
            )
    location = (
        ResidueLocation(len(residues) - 1, residues[-1]) if residues else UnlocalizedLocation()
    )
    return _TokenPlacement(location, (), match.end())


def _tokenize(
    sequence: str, pattern: re.Pattern[str], token_position: str
) -> tuple[list[str], list[_PendingToken]]:
    """Walk the regex matches, building the stripped residues and the placed tokens."""
    residues: list[str] = []
    pending: list[_PendingToken] = []
    cursor = 0
    for match in pattern.finditer(sequence):
        residues.extend(
            character for character in sequence[cursor : match.start()] if character.isalpha()
        )
        groups = [group for group in match.groups() if group is not None]
        raw_token = groups[0] if groups else match.group(0)
        placement = _place_token(sequence, match, token_position, residues)
        residues.extend(placement.consumed_residues)
        pending.append(_PendingToken(raw_token, placement.location))
        cursor = placement.next_cursor
    residues.extend(character for character in sequence[cursor:] if character.isalpha())
    return residues, pending


@dataclass(frozen=True, slots=True)
class TokenRegexRules:
    """How one vendor writes inline modification tokens, and what they resolve to."""

    token_pattern: str
    token_position: ModificationTokenPosition
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    entries: tuple[ModificationMapEntry, ...]


@dataclass(frozen=True, slots=True)
class SiteListRules:
    """How one vendor writes parallel modification-name and site columns."""

    delimiter: str
    site_base: int
    case_sensitive: bool
    unknown_policy: UnknownModificationPolicy
    entries: tuple[ModificationMapEntry, ...]


def normalize_token_regex(modified_sequence: str, config: TokenRegexRules) -> ModifiedSequence:
    """Normalize one inline-token sequence: strip, tokenize, resolve, render."""
    pattern = re.compile(config.token_pattern)
    sequence = _strip_terminal_markers(modified_sequence)
    residues, pending = _tokenize(sequence, pattern, config.token_position)
    stripped = "".join(residues)
    occurrences: list[ModificationOccurrence] = []
    unknown_tokens: dict[int, str] = {}
    unknown_token_list: list[str] = []
    for token in pending:
        entry = _matched_entry(
            config.entries,
            token.raw_token,
            token.location,
            case_sensitive=config.case_sensitive,
        )
        if entry is not None:
            occurrences.append(token.location.occurrence(entry, token.raw_token))
            continue
        _apply_unknown_policy(
            config.unknown_policy,
            token.raw_token,
            token.location,
            len(stripped),
            unknown_tokens,
            unknown_token_list,
        )
    return ModifiedSequence(
        stripped_sequence=stripped,
        proforma_sequence=render_proforma(stripped, occurrences, unknown_tokens),
        unknown_tokens=tuple(unknown_token_list),
    )


# ------------------------------------------------------------------- parallel name/site lists


def _site_location(site: int, config: SiteListRules, stripped: str) -> ModificationLocation:
    """Map one vendor site value to a ProForma position and a 0-based residue index.

    Site ``0`` is the N-terminus by convention, independent of ``site_base``: a 1-based
    vendor writing a protein N-terminal modification has no residue to point at. A site past
    the sequence end is reported as C-terminal.
    """
    if site == 0:
        return TerminalOnlyLocation("N-term")
    index = site - config.site_base
    if index >= len(stripped):
        return TerminalOnlyLocation("C-term")
    return ResidueLocation(index, stripped[index])


def normalize_site_list(
    sequence: str,
    modifications: str,
    sites: str,
    config: SiteListRules,
) -> ModifiedSequence:
    """Normalize a bare sequence plus its parallel modification and site columns."""
    stripped = "".join(character for character in sequence if character.isalpha())
    tokens = [token for token in modifications.split(config.delimiter) if token]
    raw_sites = [site for site in sites.split(config.delimiter) if site]
    if len(tokens) != len(raw_sites):
        raise PackedSiteMismatchError(
            f"modification/site length mismatch for sequence {sequence!r}: "
            f"{len(tokens)} token(s) {tokens} vs {len(raw_sites)} site(s) {raw_sites}"
        )
    by_token = {
        (entry.token if config.case_sensitive else entry.token.lower()): entry
        for entry in config.entries
    }
    occurrences: list[ModificationOccurrence] = []
    unknown_tokens: dict[int, str] = {}
    unknown_token_list: list[str] = []
    for raw_token, raw_site in zip(tokens, raw_sites, strict=True):
        if not _INTEGER_SITE.fullmatch(raw_site.strip()):
            raise PackedSiteMismatchError(
                f"non-integer modification site {raw_site!r} for sequence {sequence!r}"
            )
        location = _site_location(int(raw_site), config, stripped)
        key = raw_token if config.case_sensitive else raw_token.lower()
        entry = by_token.get(key)
        if entry is not None:
            occurrences.append(location.occurrence(entry, raw_token))
            continue
        _apply_unknown_policy(
            config.unknown_policy,
            raw_token,
            location,
            len(stripped),
            unknown_tokens,
            unknown_token_list,
        )
    return ModifiedSequence(
        stripped_sequence=stripped,
        proforma_sequence=render_proforma(stripped, occurrences, unknown_tokens),
        unknown_tokens=tuple(unknown_token_list),
    )


# --------------------------------------------------------------- the normalizers Parser injects


def _normalize_once_per_distinct[K: Hashable](
    keys: Iterable[K], normalize: Callable[[K], ModifiedSequence]
) -> list[ModifiedSequence]:
    """Normalize each distinct key once and replay the result per row."""
    memo: dict[K, ModifiedSequence] = {}
    results: list[ModifiedSequence] = []
    for key in keys:
        cached = memo.get(key)
        if cached is None:
            cached = normalize(key)
            memo[key] = cached
        results.append(cached)
    return results


def _derived(
    results: list[ModifiedSequence], proforma_output: str, stripped_output: str
) -> dict[str, pl.Series]:
    """The normalized sequence columns and unresolved vendor tokens."""
    return {
        proforma_output: pl.Series(
            proforma_output, [result.proforma_sequence for result in results], dtype=pl.String
        ),
        stripped_output: pl.Series(
            stripped_output, [result.stripped_sequence for result in results], dtype=pl.String
        ),
        "unknown_mod_tokens": pl.Series(
            "unknown_mod_tokens",
            [list(result.unknown_tokens) for result in results],
            dtype=pl.List(pl.String),
        ),
    }


@dataclass(frozen=True, slots=True)
class TokenRegexNormalizer:
    """Normalize inline modification tokens read from one sequence column."""

    rules: TokenRegexRules
    sources: tuple[str, ...]
    proforma_output: str
    stripped_output: str

    def normalize(self, columns: tuple[pl.Series, ...], /) -> dict[str, pl.Series]:
        (sequences,) = columns
        results = _normalize_once_per_distinct(
            (value or "" for value in sequences.cast(pl.String).to_list()),
            lambda sequence: normalize_token_regex(sequence, self.rules),
        )
        return _derived(results, self.proforma_output, self.stripped_output)


@dataclass(frozen=True, slots=True)
class SiteListNormalizer:
    """Normalize a bare sequence beside its parallel modification and site columns."""

    rules: SiteListRules
    sources: tuple[str, ...]
    proforma_output: str
    stripped_output: str

    def normalize(self, columns: tuple[pl.Series, ...], /) -> dict[str, pl.Series]:
        sequences, modifications, sites = columns
        rows = zip(
            (value or "" for value in sequences.cast(pl.String).to_list()),
            (value or "" for value in modifications.cast(pl.String).to_list()),
            (value or "" for value in sites.cast(pl.String).to_list()),
            strict=True,
        )
        results = _normalize_once_per_distinct(
            rows, lambda key: normalize_site_list(*key, self.rules)
        )
        return _derived(results, self.proforma_output, self.stripped_output)
