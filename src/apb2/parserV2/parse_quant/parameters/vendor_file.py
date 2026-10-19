"""A vendor table file as apb2 reads it: stored as written, or gzipped.

Compression is a fact about one file, decided once from its name by :func:`make_vendor_file`.
Consumers ask the file for its logical name and suffix, open it for bytes or text, or hand its
path to Polars, which decompresses gzipped tables itself.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import TextIO

GZIP_SUFFIX = ".gz"


@dataclass(frozen=True, slots=True)
class PlainFile:
    """A vendor table stored as the producing tool wrote it."""

    path: Path

    @property
    def name(self) -> str:
        """The file name the producing tool wrote."""
        return self.path.name

    @property
    def suffix(self) -> str:
        """The lower-case table suffix, such as ``.tsv``."""
        return self.path.suffix.lower()

    def open(self) -> BufferedIOBase:
        """Open the stored bytes."""
        return self.path.open("rb")

    def open_text(self, encoding: str) -> TextIO:
        """Open the stored text, leaving line endings to the reader."""
        return self.path.open(encoding=encoding, newline="")


@dataclass(frozen=True, slots=True)
class GzipFile:
    """A gzipped vendor table, such as ``report.tsv.gz``, read as ``report.tsv``."""

    path: Path

    @property
    def name(self) -> str:
        """The file name the producing tool wrote, without ``.gz``."""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """The lower-case table suffix under ``.gz``, such as ``.tsv``."""
        return Path(self.path.stem).suffix.lower()

    def open(self) -> BufferedIOBase:
        """Open the decompressed bytes."""
        return gzip.open(self.path, "rb")

    def open_text(self, encoding: str) -> TextIO:
        """Open the decompressed text, leaving line endings to the reader."""
        return gzip.open(self.path, "rt", encoding=encoding, newline="")


type VendorFile = PlainFile | GzipFile


def make_vendor_file(path: Path) -> VendorFile:
    """Choose how one vendor file is read, from its ``.gz`` suffix."""
    return GzipFile(path) if path.suffix.lower() == GZIP_SUFFIX else PlainFile(path)
