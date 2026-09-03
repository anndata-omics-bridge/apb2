"""Delimited text: resolve one file's dialect, then read exactly one level's projection.

Detection is not an unrestricted guess. A candidate delimiter is viable only when the header
it exposes satisfies the rule being constructed for, which is how the rule's required sources
enter here — as a predicate over headers, never as a rule. Several viable candidates are
ambiguous and reported as such; the caller binds an explicit dialect instead of having one
chosen for it.

Reading happens once, with every projected column's dtype already decided: lexical sources
stay text so a token like ``01`` cannot collapse before the canonicalization check, and plain
numeric layers are parsed natively. Nothing is left to inference.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

import polars as pl

from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFile,
    DelimitedFormatContract,
    DelimitedSourceEvidence,
    LevelReadPlan,
    NumericTextFormat,
    TextEncoding,
)

type HeaderPredicate = Callable[[tuple[str, ...]], bool]
"""Whether one inspected header satisfies the level being constructed for."""

_SCAN_LINES = 500
_GROUP_WIDTH = 3


def detected_evidence(
    path: Path,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
) -> DelimitedSourceEvidence:
    """Resolve this file's dialect within the candidates the rule permits.

    Raises ``IncompatibleSourceError`` when no candidate exposes a usable header and
    ``AmbiguousDialectError`` when more than one does.
    """
    encoding = _resolved_encoding(path, contract)
    delimiter, header = _detected_header(path, contract, accepts, encoding)
    return DelimitedSourceEvidence(
        columns=header,
        delimiter=delimiter,
        quote_char=contract.quote_char,
        encoding=encoding,
        number_format=_resolved_number_format(path, delimiter, contract),
    )


def detected_header_evidence(
    path: Path,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
) -> DelimitedSourceEvidence:
    """Resolve only header metadata for rule recognition, without inspecting data rows."""
    encoding = _resolved_encoding(path, contract)
    delimiter, header = _detected_header(path, contract, accepts, encoding)
    candidates = contract.number_format_candidates
    number_format = next(
        (candidate for candidate in candidates if candidate.decimal_mark != delimiter),
        candidates[0],
    )
    return DelimitedSourceEvidence(
        columns=header,
        delimiter=delimiter,
        quote_char=contract.quote_char,
        encoding=encoding,
        number_format=number_format,
    )


def _detected_header(
    path: Path,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
    encoding: TextEncoding,
) -> tuple[str, tuple[str, ...]]:
    """Return the unique declared delimiter and header accepted by one level."""
    viable = [
        (delimiter, header)
        for delimiter, header in (
            (delimiter, _header_of(path, delimiter, contract, encoding))
            for delimiter in contract.delimiter_candidates
        )
        if accepts(header)
    ]
    if not viable:
        raise IncompatibleSourceError(
            f"{path} exposes no usable header under any declared delimiter "
            f"{list(contract.delimiter_candidates)}"
        )
    if len(viable) > 1:
        raise AmbiguousDialectError(
            f"{path} exposes a usable header under several declared delimiters "
            f"{[delimiter for delimiter, _header in viable]}; bind an explicit dialect instead"
        )
    return viable[0]


def stated_evidence(
    source: DelimitedFile,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
) -> DelimitedSourceEvidence:
    """Accept a caller-stated dialect, once the rule permits it and the header satisfies it."""
    if source.delimiter not in contract.delimiter_candidates:
        raise IncompatibleSourceError(
            f"{source.path}: delimiter {source.delimiter!r} is not among the declared "
            f"candidates {list(contract.delimiter_candidates)}"
        )
    if source.encoding not in contract.encoding_candidates:
        raise IncompatibleSourceError(
            f"{source.path}: encoding {source.encoding!r} is not among the declared "
            f"candidates {list(contract.encoding_candidates)}"
        )
    if source.numbers not in contract.number_format_candidates:
        raise IncompatibleSourceError(
            f"{source.path}: number format {source.numbers} is not among the declared "
            f"candidates {list(contract.number_format_candidates)}"
        )
    header = _header_of(source.path, source.delimiter, contract, source.encoding)
    if not accepts(header):
        raise IncompatibleSourceError(
            f"{source.path} does not carry the columns this level requires under the stated "
            f"dialect {source.delimiter!r}"
        )
    return DelimitedSourceEvidence(
        columns=header,
        delimiter=source.delimiter,
        quote_char=source.quote_char,
        encoding=source.encoding,
        number_format=source.numbers,
    )


def _header_of(
    path: Path,
    delimiter: str,
    contract: DelimitedFormatContract,
    encoding: TextEncoding,
) -> tuple[str, ...]:
    """Read only the column names, so an unusable candidate costs one row, not one table."""
    schema = (
        pl.scan_csv(
            path,
            separator=delimiter,
            quote_char=contract.quote_char,
            encoding=_scan_encoding(encoding),
            infer_schema_length=0,
        )
        .collect_schema()
        .names()
    )
    return tuple(schema)


def _scan_encoding(encoding: TextEncoding) -> Literal["utf8", "utf8-lossy"]:
    """The encoding a streaming scan can honor.

    ``pl.scan_csv`` streams only UTF-8; any other resolved encoding is honored by the eager
    read in :meth:`DelimitedInputReader.read`. For header inspection a lossy scan is exact
    whenever the column names themselves are UTF-8, which every packaged rule requires by
    authoring ASCII source names.
    """
    return encoding if encoding in ("utf8", "utf8-lossy") else "utf8-lossy"


def _resolved_number_format(
    path: Path, delimiter: str, contract: DelimitedFormatContract
) -> NumericTextFormat:
    """Decide which declared notation this file writes its numbers in.

    A comma cannot be both the field separator and the decimal mark, so that candidate is
    excluded before anything is read. When more than one candidate survives, the file's own
    values decide — and if they support two readings, that is ambiguity, not a coin toss.
    """
    candidates = contract.number_format_candidates
    if len(candidates) == 1:
        return candidates[0]
    usable = tuple(candidate for candidate in candidates if candidate.decimal_mark != delimiter)
    if not usable:
        raise IncompatibleSourceError(
            f"{path}: every declared decimal mark is also the field separator {delimiter!r}"
        )
    if len(usable) == 1:
        return usable[0]
    observed = _decimal_marks_in_use(
        path, delimiter, {candidate.decimal_mark for candidate in usable}
    )
    if len(observed) > 1:
        raise AmbiguousDialectError(
            f"{path} contains fields readable as decimals under several declared marks "
            f"{sorted(observed)}; bind an explicit dialect instead"
        )
    if not observed:
        return usable[0]
    mark = observed.pop()
    return next(candidate for candidate in usable if candidate.decimal_mark == mark)


def _decimal_marks_in_use(
    path: Path,
    delimiter: str,
    marks: set[str],
) -> set[str]:
    """Which candidate marks this file actually writes fractions with.

    Only the shape of the number distinguishes the two readings of ``1,234``: a thousands
    separator always groups exactly three digits, so a field whose mark is followed by three
    digits is ambiguous and is never counted as evidence.
    """
    patterns = {
        mark: re.compile(rf"^-?\d+{re.escape(mark)}(\d+)$") for mark in marks if mark != delimiter
    }
    observed: set[str] = set()
    # Tolerant decoding: this scan looks only for digits and punctuation, and must not turn a
    # file the reader can still parse into a decode failure before the reader sees it.
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        handle.readline()
        for line in islice(handle, _SCAN_LINES):
            for field in line.rstrip("\n").split(delimiter):
                for mark, pattern in patterns.items():
                    match = pattern.match(field)
                    if match is not None and len(match.group(1)) != _GROUP_WIDTH:
                        observed.add(mark)
    return observed


_ENCODING_PROBE_BYTES = 8 * 1024 * 1024


def _resolved_encoding(path: Path, contract: DelimitedFormatContract) -> TextEncoding:
    """Decide which declared encoding this file is written in.

    Encoding candidates are a preference order, not an ambiguity to arbitrate: nearly every
    byte stream decodes under a single-byte fallback, so the first candidate that survives a
    bounded probe of the file wins. The probe is bounded the way number-format detection is —
    corruption past it still fails loudly at read time under the chosen encoding, and never
    silently switches the interpretation of bytes the probe did see.
    """
    candidates = contract.encoding_candidates
    if len(candidates) == 1:
        return candidates[0]
    with path.open("rb") as handle:
        probe = handle.read(_ENCODING_PROBE_BYTES)
    for candidate in candidates:
        if _probe_decodes(probe, candidate):
            return candidate
    raise IncompatibleSourceError(
        f"{path}: no declared encoding candidate {list(candidates)} decodes the first "
        f"{len(probe)} bytes"
    )


def _probe_decodes(probe: bytes, encoding: TextEncoding) -> bool:
    """Whether one candidate decodes the probe, tolerating a codepoint cut at its edge."""
    if encoding == "utf8-lossy":
        return True
    codec = "utf-8" if encoding == "utf8" else "cp1252"
    try:
        codecs.getincrementaldecoder(codec)().decode(probe, final=False)
    except UnicodeDecodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class DelimitedInputReader:
    """One delimited file, its resolved dialect, and one level's exact projection."""

    path: Path
    evidence: DelimitedSourceEvidence
    plan: LevelReadPlan

    def read(self) -> LevelSourceTable:
        """Read the projected columns, each with the dtype resolution already decided.

        A UTF-8 source streams through ``scan_csv``. Any other resolved encoding goes through
        the eager reader, which transcodes the file before parsing — the only path polars
        offers for it — so a non-UTF-8 file costs one extra pass, and a UTF-8 file costs
        nothing new.
        """
        overrides: dict[str, pl.DataType] = {name: pl.String() for name in self.plan.text_sources}
        overrides.update({name: pl.Float64() for name in self.plan.native_numeric_sources})
        encoding = self.evidence.encoding
        if encoding not in ("utf8", "utf8-lossy"):
            frame = pl.read_csv(
                self.path,
                separator=self.evidence.delimiter,
                quote_char=self.evidence.quote_char,
                encoding=encoding,
                columns=list(self.plan.projected_columns),
                schema_overrides=overrides,
                decimal_comma=self.evidence.number_format.decimal_mark == ",",
            ).select(list(self.plan.projected_columns))
            return LevelSourceTable(frame=frame)
        frame = (
            pl.scan_csv(
                self.path,
                separator=self.evidence.delimiter,
                quote_char=self.evidence.quote_char,
                encoding=encoding,
                schema_overrides=overrides,
                decimal_comma=self.evidence.number_format.decimal_mark == ",",
            )
            .select(list(self.plan.projected_columns))
            .collect()
        )
        return LevelSourceTable(frame=frame)


def make_delimited_reader(
    path: Path, evidence: DelimitedSourceEvidence, plan: LevelReadPlan
) -> DelimitedInputReader:
    """Construct the reader one resolved delimited source and level plan describe."""
    return DelimitedInputReader(path=path, evidence=evidence, plan=plan)
