"""The wide shape of a final layer: its observation column labels and how cells fill it.

A layer table's value columns are *positions*, not identities: their order aligns them with
the rows of the observation frame, and semantic identity stays in that frame plus its explicit
key tuple. A raw obs key may be composite, may repeat as text, or may collide with a var-key
column name, so it cannot be a column name at all.

Layers stay long until they are widened, last; ``ObservationGrid`` is the one place that turns
positioned cells into the wide frame, so every producer of a wide layer agrees on it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.data.errors import ConversionError

_OBSERVATION_PREFIX = "obs"
_SEPARATOR = "_"


class StorageLabelError(ConversionError):
    """No positional label prefix is free of the key columns it must sit beside."""


def observation_labels(count: int, reserved: Iterable[str]) -> tuple[str, ...]:
    """Create observation value-column labels that avoid reserved names.

    Args:
        count: Number of positional labels to create.
        reserved: Names that the complete generated block must avoid.

    Returns:
        Exactly ``count`` labels in observation order.

    Raises:
        StorageLabelError: No collision-free prefix can be found.
    """
    return _positional(_OBSERVATION_PREFIX, count, reserved)


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


@dataclass(frozen=True, slots=True)
class ObservationGrid:
    """The wide frame one level's layers fill: ``height`` var rows, one column per label."""

    height: int
    labels: tuple[str, ...]

    def widen(
        self, var_pos: pl.Series, obs_pos: pl.Series, values: pl.Series, fill: int | None
    ) -> pl.DataFrame:
        """Place each value at its (var, obs) position; every other cell holds ``fill``.

        One buffer of the layer's size is filled and its columns are slices of it, so a cell's
        null stays null and its NaN stays NaN, in the values' own dtype.
        """
        cells = pl.repeat(fill, self.height * len(self.labels), dtype=values.dtype, eager=True)
        if values.dtype != pl.Null:  # a Null column holds nothing to place, and cannot scatter
            cells.scatter(obs_pos.cast(pl.UInt64) * self.height + var_pos.cast(pl.UInt64), values)
        return pl.DataFrame(
            [
                cells.slice(index * self.height, self.height).alias(label)
                for index, label in enumerate(self.labels)
            ]
        )
