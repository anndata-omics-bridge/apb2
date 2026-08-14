"""Bind one typed source to one concrete table for one rule.

The single composition-root dispatch over the ``InputSource`` union lives here; past
``bind_source`` every binding is concrete, and delimiter plus decimal notation are facts,
not policies. Detection is not an unrestricted guess: a candidate delimiter is viable only
when the header it exposes satisfies the rule's required sources, zero viable candidates is
an incompatibility, and several viable candidates is an ambiguity the caller resolves by
binding an explicit ``DelimitedFile`` dialect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apb2.dialect import DelimitedDialect, UngroupedNumbers
from apb2.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.input.plan import ReadPlan
from apb2.input.readers import DelimitedTableReader, ParquetTableReader
from apb2.input.tabular import DelimitedText, Parquet, UnknownFormat
from apb2.sources import (
    DelimitedFile,
    FileRoles,
    Folder,
    InputSource,
    SingleFile,
)
from apb2.vendor_parse_rules.model import LongRule, WideRule

_PARQUET_SUFFIX = ".parquet"
_DELIMITER_CANDIDATES: dict[str, tuple[str, ...]] = {
    ".csv": (",",),
    ".tsv": ("\t",),
    ".txt": ("\t", ",", ";"),
}


@dataclass(frozen=True, slots=True)
class BoundDelimited:
    """One delimited file with its completely resolved dialect."""

    path: Path
    dialect: DelimitedDialect

    def header(self) -> list[str]:
        return DelimitedText(self.path, self.dialect.delimiter).columns()

    def make_reader(self, plan: ReadPlan) -> DelimitedTableReader:
        return DelimitedTableReader(self.path, self.dialect, plan)


@dataclass(frozen=True, slots=True)
class BoundParquet:
    """One Parquet file; its physical schema replaces textual dialect resolution."""

    path: Path

    def header(self) -> list[str]:
        return Parquet(self.path).columns()

    def make_reader(self, plan: ReadPlan) -> ParquetTableReader:
        return ParquetTableReader(self.path, plan)


type BoundTable = BoundDelimited | BoundParquet


def bind_source(source: InputSource, rule: LongRule | WideRule) -> BoundTable:
    """Resolve one typed source into one concrete bound table for ``rule``."""
    match source:
        case DelimitedFile(path=path, dialect=dialect):
            return BoundDelimited(path, dialect)
        case SingleFile(path=path):
            return _bind_single_file(path, rule)
        case Folder() | FileRoles():
            raise IncompatibleSourceError(
                f"rule {rule.software_name!r} level {rule.quantification_level!r} reads one "
                "table; folder and file-role sources need a file-set rule, and no packaged "
                "rule declares one yet (plan stage 7)"
            )


def _bind_single_file(path: Path, rule: LongRule | WideRule) -> BoundTable:
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return BoundParquet(path)
    candidates = _DELIMITER_CANDIDATES.get(suffix)
    if candidates is None:
        known = sorted((_PARQUET_SUFFIX, *_DELIMITER_CANDIDATES))
        raise UnknownFormat(f"unsupported extension {path.suffix!r} for {path}; known: {known}")
    viable = [
        delimiter
        for delimiter in candidates
        if rule.matches_headers(DelimitedText(path, delimiter).columns())
    ]
    if not viable:
        raise IncompatibleSourceError(
            f"{path} does not expose the columns required by {rule.software_name!r} level "
            f"{rule.quantification_level!r} under any candidate delimiter {candidates!r}"
        )
    if len(viable) > 1:
        raise AmbiguousDialectError(
            f"{path} satisfies {rule.software_name!r} level "
            f"{rule.quantification_level!r} under several delimiters {viable!r}; bind an "
            "explicit DelimitedFile dialect instead"
        )
    delimiter = viable[0]
    decimal = DelimitedText(path, delimiter).decimal_separator()
    return BoundDelimited(path, DelimitedDialect(delimiter, UngroupedNumbers(decimal)))
