"""Construction-time errors of parser V2."""

from __future__ import annotations


class IncompatibleSourceError(ValueError):
    """The bound source cannot satisfy the rule's declared contract."""


class NoCompatibleLevelError(ValueError):
    """None of the supplied rules can be constructed from the bound source."""


class AmbiguousDialectError(ValueError):
    """Several candidate dialects satisfy the rule; bind an explicit ``DelimitedFile``."""
