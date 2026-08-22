"""The one naming convention for a wide layer's observation value columns.

A layer table's value columns are *positions*, not identities: their order aligns them with
the rows of the corresponding observation frame, and semantic identity stays in that frame
plus its explicit key tuple. A raw obs key may be composite, may repeat as text, or may
collide with a var-key column name, so it cannot be a column name at all.

Three producers must agree on this convention — both decomposers and the parser's final
alignment — which is why it lives in one module instead of three.
"""

from __future__ import annotations

from collections.abc import Iterable

_OBSERVATION_PREFIX = "obs"
_PRESENCE_PREFIX = "_present"
_SEPARATOR = "_"


class StorageLabelError(ValueError):
    """No positional label prefix is free of the key columns it must sit beside."""


def observation_labels(count: int, reserved: Iterable[str]) -> tuple[str, ...]:
    """``count`` observation value-column labels, none colliding with a reserved name."""
    return _positional(_OBSERVATION_PREFIX, count, reserved)


def presence_labels(count: int, reserved: Iterable[str]) -> tuple[str, ...]:
    """``count`` labels for the Boolean presence columns one duplicate resolution needs.

    A different convention from the value labels because they live in the same frame for the
    length of one grouping and must not be mistaken for values.
    """
    return _positional(_PRESENCE_PREFIX, count, reserved)


def _positional(prefix: str, count: int, reserved: Iterable[str]) -> tuple[str, ...]:
    """``count`` labels under one prefix, none of which collides with a reserved name.

    The prefix grows an underscore until the whole block is free, so a vendor column
    genuinely named ``obs_0`` shifts the labels rather than silently shadowing one.
    """
    taken = set(reserved)
    for _attempt in range(len(taken) + 1):
        labels = tuple(f"{prefix}{_SEPARATOR}{index}" for index in range(count))
        if not taken.intersection(labels):
            return labels
        prefix += _SEPARATOR
    raise StorageLabelError(
        f"cannot place {count} positional {prefix!r} labels beside {sorted(taken)[:10]}"
    )
