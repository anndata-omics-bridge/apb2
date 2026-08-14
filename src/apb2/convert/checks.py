"""Conversion contract checks over the materialized quantitative layers.

These are *contract* checks, not QC aggregates: they report how many values each declared
layer actually carries and fail when a layer is empty in a way that means the conversion
lost data rather than the experiment lacking it. Nothing here is persisted, and no matrix
is densified — a sparse layer is measured by its stored non-zeros.

Report-level missingness, distributions, and CVs stay out of conversion by design; see
``../TODO/TODO_pmultiqc_support.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from loguru import logger

from apb2.convert._matrix_types import QuantMatrix, is_sparse_matrix

# A layer holding under this share of its cells is "effectively empty".
_EMPTY_RATIO = 0.001
# ...but only counts as lost data when a sibling layer from the same file is this full,
# which is what separates a parse failure from a genuinely sparse experiment: in
# single-cell acquisitions every layer is sparse together.
_POPULATED_RATIO = 0.5


class LayerContractError(ValueError):
    """A converted layer carries too few values to be a usable quantitative layer."""


@dataclass(frozen=True, slots=True)
class LayerOccupancy:
    """How many cells of one declared layer carry a value."""

    name: str
    present: int
    total: int

    @property
    def ratio(self) -> float:
        """Share of cells carrying a value; ``0.0`` for an empty layer."""
        return self.present / self.total if self.total else 0.0

    def describe(self) -> str:
        """Render the occupancy for a log line."""
        return f"{self.name}: {self.present}/{self.total} ({self.ratio:.2%})"


def layer_occupancies(layers: Mapping[str, QuantMatrix]) -> list[LayerOccupancy]:
    """Measure every layer's occupancy without densifying a sparse matrix."""
    return [
        LayerOccupancy(name=name, present=_present_count(matrix), total=_cell_count(matrix))
        for name, matrix in layers.items()
    ]


def check_layer_occupancy(
    layers: Mapping[str, QuantMatrix],
    *,
    x_layer: str,
    strict: bool = False,
) -> list[LayerOccupancy]:
    """Report per-layer occupancy and reject a conversion that lost its quantities.

    An effectively empty layer beside a populated sibling means the vendor column was
    read but its values did not survive parsing — a mis-detected decimal separator or an
    unhandled sentinel, not an empty experiment. That is an error for ``x_layer``, whose
    emptiness makes the whole object unusable, and a warning for the rest. ``strict``
    promotes those warnings to errors.
    """
    occupancies = layer_occupancies(layers)
    for occupancy in occupancies:
        logger.debug(f"layer occupancy {occupancy.describe()}")
    populated = [item.name for item in occupancies if item.ratio >= _POPULATED_RATIO]
    suspect = [
        item for item in occupancies if item.ratio < _EMPTY_RATIO and populated != [item.name]
    ]
    if not suspect or not populated:
        return occupancies

    reference = ", ".join(populated[:3])
    for occupancy in suspect:
        message = (
            f"layer {occupancy.name!r} is effectively empty ({occupancy.describe()}) "
            f"while {reference} is populated — the source column was read but its values "
            f"did not parse; check the vendor number format and missing-value sentinels"
        )
        if occupancy.name == x_layer:
            raise LayerContractError(message)
        if strict:
            raise LayerContractError(message)
        logger.warning(message)
    return occupancies


def _present_count(matrix: QuantMatrix) -> int:
    """Count cells carrying a usable value, reading sparse structure rather than values.

    ``coerce_numeric`` normalizes every layer to ``float64``, so a missing cell is NaN and
    a sparse layer's stored ``data`` is all that has to be inspected.
    """
    if is_sparse_matrix(matrix):
        data = np.asarray(matrix.data)
        return int(np.count_nonzero(np.isfinite(data) & (data != 0)))
    return int(np.count_nonzero(np.isfinite(np.asarray(matrix))))


def _cell_count(matrix: QuantMatrix) -> int:
    """Total addressable cells of a layer's run x feature axes."""
    rows, columns = matrix.shape
    return int(rows) * int(columns)
