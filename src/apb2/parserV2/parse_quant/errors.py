"""Errors two or more parse-owned boundaries raise about one physical source.

Only errors with several raisers live here. An error belonging to one operation stays beside
that operation, so catching it cannot accidentally catch someone else's failure: packed-length,
duplicate-cell, aggregate-type, canonical-collision, encoding, contract, and writer errors are
all declared by the strategy that raises them.
"""

from __future__ import annotations


class IncompatibleSourceError(ValueError):
    """This source cannot satisfy the rule's declared format, columns, layers, or keys.

    The skip contract: ``compile_parsers`` catches this to move to the next quantification
    level, so anything meaning "not this source for this level" must be this class.
    """


class AmbiguousDialectError(ValueError):
    """Several allowed physical interpretations satisfy the same rule.

    Never resolved by guessing. The caller binds an explicit dialect instead.
    """


class ResultIOError(ValueError):
    """A path or persisted result cannot satisfy the APB2 result-I/O contract."""


class UnsupportedResultFormatError(ResultIOError):
    """A path suffix does not name one of APB2's result formats."""


class InvalidResultError(ResultIOError):
    """A result value or persisted result violates its declared format contract."""
