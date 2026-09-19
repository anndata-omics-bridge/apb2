"""Errors two or more parse-owned operations raise about one physical source.

Only errors with several raisers live here. An error belonging to one operation stays beside
that operation, so catching it cannot accidentally catch someone else's failure. Result-I/O
errors belong to the ``io`` child rather than this parent module.
"""

from __future__ import annotations


class IncompatibleSourceError(ValueError):
    """This source cannot satisfy the rule's declared format, columns, layers, or keys.

    Detection catches this while resolving level selections, so anything meaning "not this
    source for this level" must be this class.
    """


class AmbiguousDialectError(ValueError):
    """Several allowed physical interpretations satisfy the same rule.

    Never resolved by guessing. The caller binds an explicit dialect instead.
    """


class LayerValueError(ValueError):
    """A parsed layer cannot satisfy its declared canonical value semantics."""


class LayerContractError(ValueError):
    """Canonical measurement layers violate their required occupancy contract."""
