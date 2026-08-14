"""Shared dataclass for converter outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type DenseLayerMatrix = NDArray[np.float64]
"""One converted (obs x var) quantitative layer."""

type CategoryCodes = NDArray[np.intp]
"""Integer category codes addressing one converted axis."""

type ValidKeyMask = NDArray[np.bool_]
"""Rows whose observation and feature keys are both present."""


@dataclass
class ConversionPieces:
    """Backend-neutral axes and matrices produced from one vendor table."""

    X: DenseLayerMatrix
    obs: pd.DataFrame
    var: pd.DataFrame
    layers: dict[str, DenseLayerMatrix] = field(default_factory=dict)
