"""Delimited text: resolve one file's dialect, then read exactly one level's projection.

Detection is not an unrestricted guess. A candidate delimiter is viable only when the header
it exposes satisfies the rule being constructed for, which is how the rule's required sources
enter here — as a predicate over headers, never as a rule. Several viable candidates are
ambiguous and reported as such; none is chosen silently.

Reading happens once, with every projected column's dtype already decided: lexical sources
stay text so a token like ``01`` cannot collapse before the canonicalization check, summed
layers are read as numbers, and plain numeric layers are converted from their text inside the
same scan. Nothing is left to inference.
"""

from __future__ import annotations

import codecs
import os
import re
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.numeric_text import as_numbers
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    DelimitedSourceEvidence,
    InputFile,
    InputTable,
    LevelReadPlan,
    LevelSourceTable,
    NumericTextFormat,
    TextEncoding,
)
from apb2.parserV2.parse_quant.parameters.vendor_file import VendorFile

type HeaderPredicate = Callable[[tuple[str, ...]], bool]
"""Whether one inspected header satisfies the level being constructed for."""

_SCAN_ROWS = 500
_GROUP_WIDTH = 3


def detected_evidence(
    file: VendorFile,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
) -> DelimitedSourceEvidence:
    """Resolve this file's dialect within the candidates the rule permits.

    Raises ``IncompatibleSourceError`` when no candidate exposes a usable header and
    ``AmbiguousDialectError`` when more than one does.
    """
    encoding = _resolved_encoding(file, contract)
    delimiter, header = _detected_header(file.path, contract, accepts, encoding)
    return DelimitedSourceEvidence(
        columns=header,
        delimiter=delimiter,
        quote_char=contract.quote_char,
        encoding=encoding,
        number_format=_resolved_number_format(file.path, delimiter, contract),
    )


def detected_header(
    file: VendorFile,
    contract: DelimitedFormatContract,
    accepts: HeaderPredicate,
) -> tuple[str, ...]:
    """Resolve only the accepted header, for rule recognition, without inspecting data rows."""
    return _detected_header(file.path, contract, accepts, _resolved_encoding(file, contract))[1]


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
            f"{[delimiter for delimiter, _header in viable]}"
        )
    return viable[0]


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
    # Bound logical records, respecting quoted fields. Lossy UTF-8 is sufficient for this
    # digit/punctuation probe even when the final reader uses a single-byte encoding.
    # Three trailing digits remain ambiguous with thousands grouping, not decimal evidence.
    observed = (
        pl.scan_csv(
            path,
            separator=delimiter,
            quote_char=contract.quote_char,
            encoding="utf8-lossy",
            infer_schema=False,
            n_rows=_SCAN_ROWS,
        )
        .select(
            pl.any_horizontal(
                pl.all().str.extract(rf"^-?\d+{re.escape(mark)}(\d+)$", 1).str.len_chars()
                != _GROUP_WIDTH
            )
            .any()
            .alias(mark)
            for mark in {candidate.decimal_mark for candidate in usable}
        )
        .collect()
        .row(0, named=True)
    )
    marks = {mark for mark, present in observed.items() if present}
    if len(marks) > 1:
        raise AmbiguousDialectError(
            f"{path} contains fields readable as decimals under several declared marks "
            f"{sorted(marks)}"
        )
    if not marks:
        return usable[0]
    return next(candidate for candidate in usable if candidate.decimal_mark in marks)


_ENCODING_PROBE_BYTES = 8 * 1024 * 1024
_TRANSCODE_CHARS = 16 * 1024 * 1024


def _resolved_encoding(file: VendorFile, contract: DelimitedFormatContract) -> TextEncoding:
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
    with file.open() as handle:
        probe = handle.read(_ENCODING_PROBE_BYTES)
    for candidate in candidates:
        if _probe_decodes(probe, candidate):
            return candidate
    raise IncompatibleSourceError(
        f"{file.path}: no declared encoding candidate {list(candidates)} decodes the first "
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

    file: VendorFile
    evidence: DelimitedSourceEvidence
    plan: LevelReadPlan

    def read(self) -> LevelSourceTable:
        """Read the projected columns, each with the dtype resolution already decided.

        Converted columns are read as text and turned into numbers in the same streaming
        plan, so their text exists one chunk at a time. A converted column holding a value
        that needs its text — a token no number reads, or the text ``NaN`` — is read again as
        text, so it reaches the parser exactly as an unconverted layer does. A native numeric
        column whose layer declares missing tokens is read as text and cast here, the tokens
        as null.
        """
        columns = self.plan.projected_columns
        converted = tuple(name for name in columns if name in self.plan.converted_sources)
        flags = _flag_names(len(converted), self.evidence.columns)
        notation = self.evidence.number_format
        numbers = {name: as_numbers(pl.col(name), pl.String(), notation) for name in converted}
        numbers |= {
            name: self.plan.tokened_number(name, notation.decimal_mark)
            for name in self.plan.missing_tokens
        }
        with self._utf8_source() as (path, encoding):
            frame = self._read(path, encoding, columns, converted, flags, numbers)
        file_format = _FORMATS.get(self.evidence.delimiter, "delimited")
        table = InputTable(
            InputFile(self.file.path, "table", file_format), self.evidence.columns, frame.height
        )
        return LevelSourceTable(frame, (table,))

    def _read(
        self,
        path: Path,
        encoding: Literal["utf8", "utf8-lossy"],
        columns: tuple[str, ...],
        converted: tuple[str, ...],
        flags: tuple[str, ...],
        numbers: dict[str, pl.Expr],
    ) -> pl.DataFrame:
        """Both passes of :meth:`read` over one UTF-8 source."""
        frame = (
            self._scan(path, encoding, columns)
            .select(
                *(numbers[name].alias(name) if name in numbers else name for name in columns),
                *(
                    (
                        pl.col(name).str.contains(r"\S")
                        & (numbers[name].is_null() | numbers[name].is_nan())
                    ).alias(flag)
                    for name, flag in zip(converted, flags, strict=True)
                ),
            )
            .collect(engine="streaming")
        )
        needs_text = tuple(
            name
            for name, flag in zip(converted, flags, strict=True)
            if frame.get_column(flag).any()
        )
        frame = frame.drop(flags)
        if needs_text:
            frame = frame.with_columns(
                self._scan(path, encoding, needs_text).collect().get_columns()
            )
        return frame

    @contextmanager
    def _utf8_source(self) -> Generator[tuple[Path, Literal["utf8", "utf8-lossy"]]]:
        """The file as UTF-8 that ``scan_csv`` can stream, and the encoding to scan it with.

        A UTF-8 source is scanned where it lies, gzipped or not. Any other resolved encoding
        is transcoded to a temporary UTF-8 copy, chunk by chunk, and the copy is removed after
        the read. Polars streams only UTF-8; its eager reader decodes a whole file in memory,
        which took 44 GB for an 11 GB Spectronaut report whose every row holds one
        windows-1252 dash.
        """
        encoding = self.evidence.encoding
        if encoding in ("utf8", "utf8-lossy"):
            yield self.file.path, encoding
            return
        handle, name = tempfile.mkstemp(prefix="apb2-utf8-", suffix=".txt")
        os.close(handle)
        path = Path(name)
        try:
            with (
                self.file.open_text(encoding) as source,
                path.open("w", encoding="utf-8", newline="") as target,
            ):
                while chunk := source.read(_TRANSCODE_CHARS):
                    target.write(chunk)
            yield path, "utf8"
        finally:
            path.unlink(missing_ok=True)

    def _scan(
        self, path: Path, encoding: Literal["utf8", "utf8-lossy"], columns: tuple[str, ...]
    ) -> pl.LazyFrame:
        """The given columns of a UTF-8 source, each with its read dtype, converted as text."""
        overrides: dict[str, pl.DataType] = {name: pl.String() for name in self.plan.text_sources}
        overrides.update({name: pl.Categorical() for name in self.plan.categorical_sources})
        overrides.update(
            {
                name: pl.String() if name in self.plan.missing_tokens else pl.Float64()
                for name in self.plan.native_numeric_sources
            }
        )
        return pl.scan_csv(
            path,
            separator=self.evidence.delimiter,
            quote_char=self.evidence.quote_char,
            encoding=encoding,
            schema_overrides=overrides,
            decimal_comma=self.evidence.number_format.decimal_mark == ",",
        ).select(columns)


def _flag_names(count: int, header: tuple[str, ...]) -> tuple[str, ...]:
    """Names for the per-column conversion flags that no header column holds."""
    prefix = "_needs_text_"
    while any(name.startswith(prefix) for name in header):
        prefix = f"_{prefix}"
    return tuple(f"{prefix}{index}" for index in range(count))


_FORMATS = {"\t": "tsv", ",": "csv"}
