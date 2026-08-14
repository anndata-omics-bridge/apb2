"""Backend-neutral result of one quantification-level parse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from apb2.serialization import JsonValue

type DenseLayerMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ParsedData:
    """One AnnData-shaped conversion result, independent of any storage backend.

    ``uns`` carries the complete, JSON-typed output provenance, so persisting this value
    is a pure function of the value itself (see ``parser_v2/output/``, plan stage 6).
    """

    X: DenseLayerMatrix
    obs: pd.DataFrame
    var: pd.DataFrame
    uns: Mapping[str, JsonValue]
    layers: Mapping[str, DenseLayerMatrix]
