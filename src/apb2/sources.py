"""Typed source values binding concrete resources at construction.

The source is a required argument of ``make_parser``/``make_parsers``; each variant owns
the resolution of concrete tables for a rule-declared layout (plan stage 3), so
construction never branches on which variant it received.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from apb2.dialect import DelimitedDialect


@dataclass(frozen=True, slots=True)
class SingleFile:
    """One vendor report file; its physical dialect is resolved during construction."""

    path: Path


@dataclass(frozen=True, slots=True)
class DelimitedFile:
    """One delimited vendor report with an explicitly bound dialect.

    The escape hatch for files whose dialect detection fails or is ambiguous: the bound
    dialect is still validated against the file and rule contract during construction.
    """

    path: Path
    dialect: DelimitedDialect


@dataclass(frozen=True, slots=True)
class Folder:
    """A folder satisfying a file-set rule through the rule's declared filenames."""

    root: Path


@dataclass(frozen=True, slots=True)
class FileRoles:
    """Explicit mapping from rule-declared table roles to concrete files.

    Per-role values compose: a role bound as ``DelimitedFile`` carries its own explicit
    dialect, so distinct files in one logical source may use distinct notations.
    """

    tables: Mapping[str, SingleFile | DelimitedFile]


type InputSource = SingleFile | DelimitedFile | Folder | FileRoles
