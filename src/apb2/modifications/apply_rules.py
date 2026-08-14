"""Token extraction and mapping from vendor modified sequences.

Takes a vendor-specific modified sequence (e.g. ``"PEPM[15.9949]TIDE"`` or
``"_(ac)PEPTIDEM(ox)_"``) plus a modification rule (regex + map entries)
and produces a :class:`ModifiedSequence` with localized
:class:`ModificationOccurrence`\\s and a ProForma rendering.

Map lookup uses the tuple ``(mass_delta, target, position)`` per the plan,
not mass alone — so e.g. Acetyl-Nterm and Acetyl-K with the same mass
remain distinguishable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from apb2.modifications.model import (
    ModificationOccurrence,
    ModifiedSequence,
)
from apb2.modifications.proforma import render_proforma


@dataclass(frozen=True)
class MapEntry:
    """One ``modifications.map`` entry from a parsing-rule JSON document."""

    token: str
    name: str
    accession: str
    target: tuple[str, ...]
    position: str
    mass_delta: float


@dataclass(frozen=True)
class ModificationRule:
    """Parsed ``[modifications]`` section."""

    token_pattern: str
    token_position: str  # "before_residue" | "after_residue"
    case_sensitive: bool = False
    unknown_policy: str = "preserve"  # "preserve" | "drop" | "error"
    entries: tuple[MapEntry, ...] = ()


@dataclass(frozen=True)
class SiteListRule:
    """Parsed ``parser="site_list"`` ``[modifications]`` section."""

    delimiter: str = ";"
    site_base: int = 1
    case_sensitive: bool = False
    unknown_policy: str = "preserve"  # "preserve" | "drop" | "error"
    entries: tuple[MapEntry, ...] = ()


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


@dataclass(frozen=True)
class ResidueLocation:
    """A modification localized to one sequence residue."""

    sequence_index: int
    residue: str

    def target_position(self) -> str:
        return "Anywhere"

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return AdjacentResidue(self.residue)

    def occurrence(self, entry: MapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            target_residue=self.residue,
            sequence_index=self.sequence_index,
            position="Anywhere",
            mass_delta=entry.mass_delta,
            source_token=raw_token,
        )

    def record_unknown_token(
        self,
        unknown_tokens: dict[int, str],
        raw_token: str,
        sequence_length: int,
    ) -> None:
        unknown_tokens[self.sequence_index] = raw_token


@dataclass(frozen=True)
class TerminalLocation:
    """A terminal modification beside a concrete sequence residue."""

    position: Literal["N-term", "C-term"]
    adjacent_residue: str

    def target_position(self) -> str:
        return self.position

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return AdjacentResidue(self.adjacent_residue)

    def occurrence(self, entry: MapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            target_residue=self.adjacent_residue,
            position=self.position,
            mass_delta=entry.mass_delta,
            source_token=raw_token,
        )

    def record_unknown_token(
        self,
        unknown_tokens: dict[int, str],
        raw_token: str,
        sequence_length: int,
    ) -> None:
        unknown_tokens[_terminal_token_index(self.position, sequence_length)] = raw_token


@dataclass(frozen=True)
class TerminalOnlyLocation:
    """A terminal modification with no residue association."""

    position: Literal["N-term", "C-term"]

    def target_position(self) -> str:
        return self.position

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return NoAdjacentResidue()

    def occurrence(self, entry: MapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position=self.position,
            mass_delta=entry.mass_delta,
            source_token=raw_token,
        )

    def record_unknown_token(
        self,
        unknown_tokens: dict[int, str],
        raw_token: str,
        sequence_length: int,
    ) -> None:
        unknown_tokens[_terminal_token_index(self.position, sequence_length)] = raw_token


@dataclass(frozen=True)
class UnlocalizedLocation:
    """A non-terminal token that cannot be attached to a residue."""

    def target_position(self) -> str:
        return "Anywhere"

    def adjacent(self) -> AdjacentResidue | NoAdjacentResidue:
        return NoAdjacentResidue()

    def occurrence(self, entry: MapEntry, raw_token: str) -> ModificationOccurrence:
        return ModificationOccurrence(
            name=entry.name,
            accession=entry.accession,
            position="Anywhere",
            mass_delta=entry.mass_delta,
            source_token=raw_token,
        )

    def record_unknown_token(
        self,
        unknown_tokens: dict[int, str],
        raw_token: str,
        sequence_length: int,
    ) -> None:
        """Record nothing: an unlocalized token has no index to attach to.

        The identity arm. It is what let the old chain end in a bare ``elif`` with no
        ``else``, which reads as an oversight rather than as the decision it is.
        """
        return


@dataclass(frozen=True)
class ParsedMass:
    """A vendor token parsed as a numeric mass shift."""

    value: float


@dataclass(frozen=True)
class NonNumericToken:
    """A vendor token is not a numeric mass shift."""

    token: str


@dataclass(frozen=True)
class MatchedMapEntry:
    """A vendor token matched one modification-map entry."""

    entry: MapEntry


@dataclass(frozen=True)
class UnmatchedMapEntry:
    """No modification-map entry matched a vendor token and location."""

    token: str


ModificationLocation = (
    ResidueLocation | TerminalLocation | TerminalOnlyLocation | UnlocalizedLocation
)


def _terminal_token_index(position: Literal["N-term", "C-term"], sequence_length: int) -> int:
    """Index a terminal token occupies in the unknown-token map: before, or past, the sequence."""
    return -1 if position == "N-term" else sequence_length


_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_INTEGER_SITE = re.compile(r"^[+-]?\d+$")


def _target_matches(entry_target: tuple[str, ...], location: ModificationLocation) -> bool:
    """Decide whether an entry's allowed ``target``s are compatible with a token's context.

    ``entry_target`` is the tuple of residues/termini the modification may sit on.
    Terminal targets (e.g. ``"N-term"``) match a corresponding terminal
    location; residue targets (``"M"``, ``"C"``, …) match its adjacent amino
    acid. An empty target tuple matches anything.
    """
    if not entry_target:
        return True
    position = location.target_position()
    adjacent_residue = location.adjacent()
    for target in entry_target:
        if target in _TERMINUS_TARGETS:
            if target.endswith(position) or target == position:
                return True
        elif isinstance(adjacent_residue, AdjacentResidue) and target == adjacent_residue.value:
            return True
    return False


@dataclass(frozen=True)
class AdjacentResidue:
    """A location has an adjacent amino-acid residue."""

    value: str


@dataclass(frozen=True)
class NoAdjacentResidue:
    """A location has no adjacent amino-acid residue."""


def _parse_mass(raw: str) -> ParsedMass | NonNumericToken:
    cleaned = raw.strip().lstrip("+")
    if _NUMERIC_TOKEN.fullmatch(cleaned):
        return ParsedMass(float(cleaned))
    return NonNumericToken(raw)


def _match_entry(
    entries: Iterable[MapEntry],
    raw_token: str,
    location: ModificationLocation,
    case_sensitive: bool,
) -> MatchedMapEntry | UnmatchedMapEntry:
    """Pick the best map entry for a vendor token.

    For numeric tokens the lookup key is the tuple
    ``(mass_delta, target, position)``. For non-numeric tokens (e.g.
    ``"ox"``, ``"ac"``) the exact token string is used as the fallback.
    """
    candidates = tuple(entries)
    parsed_mass = _parse_mass(raw_token)
    if isinstance(parsed_mass, ParsedMass):
        for entry in candidates:
            if _entry_context_matches(entry, location) and math.isclose(
                entry.mass_delta,
                parsed_mass.value,
                abs_tol=_MASS_TOLERANCE,
            ):
                return MatchedMapEntry(entry)

    cmp_token = raw_token if case_sensitive else raw_token.lower()
    for entry in candidates:
        if not _entry_context_matches(entry, location):
            continue
        entry_token = entry.token if case_sensitive else entry.token.lower()
        if entry_token == cmp_token:
            return MatchedMapEntry(entry)
    return UnmatchedMapEntry(raw_token)


def _entry_context_matches(
    entry: MapEntry,
    location: ModificationLocation,
) -> bool:
    position = location.target_position()
    return entry.position == position and _target_matches(
        entry.target,
        location,
    )


@dataclass
class _PendingToken:
    raw_token: str
    location: ModificationLocation


@dataclass(frozen=True)
class _TokenPlacement:
    """Location plus sequence characters consumed while placing one token."""

    location: ModificationLocation
    consumed_residues: tuple[str, ...]
    next_cursor: int


def _strip_terminal_markers(seq: str) -> str:
    """Drop leading/trailing terminal markers (``_``, ``-``, ``.``)."""
    while seq and seq[0] in _TERM_MARKERS:
        seq = seq[1:]
    while seq and seq[-1] in _TERM_MARKERS:
        seq = seq[:-1]
    return seq


def _place_token(
    seq: str,
    match: re.Match[str],
    token_position: str,
    stripped_chars: list[str],
) -> _TokenPlacement:
    """Locate one regex token relative to the residues parsed before it."""
    if not stripped_chars and match.start() == 0:
        adjacent = seq[match.end() : match.end() + 1]
        location: ModificationLocation = (
            TerminalLocation("N-term", adjacent) if adjacent else TerminalOnlyLocation("N-term")
        )
        return _TokenPlacement(location, (), match.end())

    if match.end() == len(seq) and token_position != "before_residue":
        location = (
            TerminalLocation("C-term", stripped_chars[-1])
            if stripped_chars
            else TerminalOnlyLocation("C-term")
        )
        return _TokenPlacement(location, (), match.end())

    if token_position == "before_residue":
        next_residue = seq[match.end() : match.end() + 1]
        if next_residue.isalpha():
            location = ResidueLocation(len(stripped_chars), next_residue)
            return _TokenPlacement(location, (next_residue,), match.end() + 1)

    location = (
        ResidueLocation(len(stripped_chars) - 1, stripped_chars[-1])
        if stripped_chars
        else UnlocalizedLocation()
    )
    return _TokenPlacement(location, (), match.end())


def _tokenize(
    seq: str, pattern: re.Pattern[str], token_position: str
) -> tuple[list[str], list[_PendingToken]]:
    """Walk regex matches, building the stripped residue sequence and the
    position-classified pending tokens (N-term / C-term / before-residue / Anywhere).
    """
    stripped_chars: list[str] = []
    pending: list[_PendingToken] = []
    cursor = 0

    for match in pattern.finditer(seq):
        for ch in seq[cursor : match.start()]:
            if ch.isalpha():
                stripped_chars.append(ch)

        groups = [g for g in match.groups() if g is not None]
        raw_token = groups[0] if groups else match.group(0)

        placement = _place_token(seq, match, token_position, stripped_chars)
        stripped_chars.extend(placement.consumed_residues)
        pending.append(_PendingToken(raw_token, placement.location))
        cursor = placement.next_cursor

    for ch in seq[cursor:]:
        if ch.isalpha():
            stripped_chars.append(ch)

    return stripped_chars, pending


def _resolve_tokens(
    pending: list[_PendingToken], rule: ModificationRule, stripped: str
) -> tuple[list[ModificationOccurrence], dict[int, str], list[str]]:
    """Match pending tokens to map entries; apply ``unknown_policy`` and record
    terminal/residue indices for unresolved tokens.
    """
    occurrences: list[ModificationOccurrence] = []
    unknown_tokens: dict[int, str] = {}
    unknown_list: list[str] = []

    for tok in pending:
        matched = _match_entry(
            rule.entries,
            tok.raw_token,
            tok.location,
            rule.case_sensitive,
        )
        if isinstance(matched, MatchedMapEntry):
            occurrences.append(tok.location.occurrence(matched.entry, tok.raw_token))
            continue
        _apply_unknown_policy(
            rule.unknown_policy,
            tok.raw_token,
            tok.location,
            len(stripped),
            unknown_tokens,
            unknown_list,
        )

    return occurrences, unknown_tokens, unknown_list


def _apply_unknown_policy(
    unknown_policy: str,
    raw_token: str,
    location: ModificationLocation,
    stripped_length: int,
    unknown_tokens: dict[int, str],
    unknown_list: list[str],
) -> None:
    """Apply one rule's unknown-token policy to one unmatched token."""
    if unknown_policy == "error":
        raise ValueError(f"unknown modification token: {raw_token!r}")
    if unknown_policy == "drop":
        return
    unknown_list.append(raw_token)
    location.record_unknown_token(unknown_tokens, raw_token, stripped_length)


def _modified_sequence(
    stripped: str,
    occurrences: list[ModificationOccurrence],
    unknown_tokens: dict[int, str],
    unknown_list: list[str],
    source_sequence: str,
) -> ModifiedSequence:
    """Render the ProForma string and assemble the one result both parsers return."""
    proforma = render_proforma(stripped, occurrences, unknown_tokens=unknown_tokens)
    return ModifiedSequence(
        stripped_sequence=stripped,
        proforma_sequence=proforma,
        occurrences=occurrences,
        source_sequence=source_sequence,
        unknown_tokens=unknown_list,
    )


def apply_rule(modified_sequence: str, rule: ModificationRule) -> ModifiedSequence:
    """Normalize a vendor modified sequence via ``rule``: strip → tokenize → resolve → render."""
    pattern = re.compile(rule.token_pattern)
    seq = _strip_terminal_markers(modified_sequence)
    stripped_chars, pending = _tokenize(seq, pattern, rule.token_position)
    stripped = "".join(stripped_chars)
    occurrences, unknown_tokens, unknown_list = _resolve_tokens(pending, rule, stripped)
    return _modified_sequence(
        stripped, occurrences, unknown_tokens, unknown_list, modified_sequence
    )


def _site_location(site: int, rule: SiteListRule, stripped: str) -> ModificationLocation:
    """Map a vendor site value to a ProForma position and 0-based residue index.

    Site ``0`` is the N-terminus by convention, independent of ``site_base``: a
    1-based vendor writing a protein N-terminal modification has no residue to point
    at and uses ``0``. Sites past the sequence end are reported as C-terminal.
    """
    if site == 0:
        return TerminalOnlyLocation("N-term")
    index = site - rule.site_base
    if index >= len(stripped):
        return TerminalOnlyLocation("C-term")
    return ResidueLocation(index, stripped[index])


def apply_site_list(
    sequence: str, modifications: str, sites: str, rule: SiteListRule
) -> ModifiedSequence:
    """Normalize a bare sequence plus parallel modification/site columns.

    ``modifications`` and ``sites`` are paired index-wise after splitting on
    ``rule.delimiter``; a length mismatch is a vendor-file defect and raises.
    """
    stripped = "".join(ch for ch in sequence if ch.isalpha())
    tokens = [t for t in modifications.split(rule.delimiter) if t] if modifications else []
    raw_sites = [s for s in sites.split(rule.delimiter) if s] if sites else []

    if len(tokens) != len(raw_sites):
        raise ValueError(
            f"modification/site length mismatch for sequence {sequence!r}: "
            f"{len(tokens)} token(s) {tokens} vs {len(raw_sites)} site(s) {raw_sites}"
        )

    by_token = {
        (entry.token if rule.case_sensitive else entry.token.lower()): entry
        for entry in rule.entries
    }

    occurrences: list[ModificationOccurrence] = []
    unknown_tokens: dict[int, str] = {}
    unknown_list: list[str] = []

    for raw_token, raw_site in zip(tokens, raw_sites, strict=True):
        if not _INTEGER_SITE.fullmatch(raw_site.strip()):
            raise ValueError(
                f"non-integer modification site {raw_site!r} for sequence {sequence!r}"
            )
        site = int(raw_site)

        location = _site_location(site, rule, stripped)
        token_key = raw_token if rule.case_sensitive else raw_token.lower()

        if token_key in by_token:
            occurrences.append(location.occurrence(by_token[token_key], raw_token))
            continue
        _apply_unknown_policy(
            rule.unknown_policy, raw_token, location, len(stripped), unknown_tokens, unknown_list
        )

    return _modified_sequence(stripped, occurrences, unknown_tokens, unknown_list, sequence)
