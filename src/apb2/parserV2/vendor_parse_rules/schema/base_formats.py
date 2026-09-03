"""Shared physical-format defaults and explicit detection overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase

type SupportedExtension = Literal[".tsv", ".txt", ".csv", ".parquet"]
type TextEncoding = Literal["utf8", "utf8-lossy", "windows-1252"]
type DecimalMark = Literal[".", ","]
type SingleCharacter = Annotated[str, Field(min_length=1, max_length=1)]


@dataclass(frozen=True, slots=True)
class BaseDelimitedFormat:
    """Ordinary interpretation of one delimited extension."""

    delimiter: str
    encoding: TextEncoding = "utf8"
    quote_char: str = '"'
    decimal_mark: DecimalMark = "."
    thousands_marks: tuple[str, ...] = ()


DELIMITED_BASE_FORMATS: dict[SupportedExtension, BaseDelimitedFormat] = {
    ".tsv": BaseDelimitedFormat(delimiter="\t"),
    ".txt": BaseDelimitedFormat(delimiter="\t"),
    ".csv": BaseDelimitedFormat(delimiter=","),
}
PARQUET_EXTENSIONS: frozenset[SupportedExtension] = frozenset({".parquet"})


class DetectedDelimiter(ModelBase):
    """The bounded delimiters an exceptional vendor export may use."""

    mode: Literal["detect"]
    candidates: list[SingleCharacter] = Field(min_length=1)


class DetectedNumberFormat(ModelBase):
    """The bounded decimal and grouping marks an exceptional export may use."""

    mode: Literal["detect"]
    decimal_candidates: list[DecimalMark] = Field(min_length=1)
    thousands_candidates: list[SingleCharacter] = Field(default_factory=list)


class DetectedEncoding(ModelBase):
    """The bounded text encodings an exceptional vendor export may use, in preference order.

    The first candidate whose decoding survives a bounded probe of the file wins, so ``utf8``
    should come first: any file it accepts is read exactly, and a fallback such as
    ``windows-1252`` is consulted only for bytes UTF-8 rejects.
    """

    mode: Literal["detect"]
    candidates: list[TextEncoding] = Field(min_length=1)
