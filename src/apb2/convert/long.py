"""Convert a long-format DataFrame into AnnData pieces.

Each layer is built by scattering the long values into a dense (obs x var) matrix via
integer category codes, rather than ``DataFrame.pivot_table``. pivot_table materialises a
huge transient for high-cardinality var axes (the fragment level fans one report out to
millions of rows x hundreds of thousands of features and peaks at many GB); the scatter is
O(nnz + obs·var) and matches pivot_table's semantics exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apb2.convert._axis import build_axis_frame, build_index
from apb2.convert._pieces import (
    ConversionPieces,
    DenseLayerMatrix,
)
from apb2.convert.duplicates import DuplicatePolicy
from apb2.convert.layers import LayerPlan
from apb2.convert.numeric import warn_if_all_missing

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LongConversion:
    """Everything needed to convert one long table, with no shape flag left to read.

    There is no ``input_shape`` field and no guard rejecting a wide rule: choosing this class
    is what answering that question means. The ``raise ... called with ... rule`` checks the
    flag used to need are type errors now, caught before the program runs.
    """

    obs_keys: tuple[str, ...]
    var_keys: tuple[str, ...]
    obs_columns: tuple[str, ...]
    var_columns: tuple[str, ...]
    layers: tuple[LayerPlan, ...]
    x_layer: str
    duplicates: DuplicatePolicy

    def convert(self, df: pd.DataFrame) -> ConversionPieces:
        """Convert a long DataFrame to backend-neutral pieces."""
        obs_keys = list(self.obs_keys)
        var_keys = list(self.var_keys)
        self.duplicates.reject_duplicate_keys(df, [*obs_keys, *var_keys])

        obs_df = build_axis_frame(df, obs_keys, list(self.obs_columns))
        var_df = build_axis_frame(df, var_keys, list(self.var_columns))

        # Map every input row to its position in the obs/var axes. build_axis_frame keeps the
        # first occurrence per key, so the Categorical codes index directly into obs_df/var_df.
        obs_codes = pd.Categorical(build_index(df, obs_keys), categories=obs_df.index).codes
        var_codes = pd.Categorical(build_index(df, var_keys), categories=var_df.index).codes
        key_ok = df[obs_keys + var_keys].notna().all(axis=1).to_numpy()

        n_obs, n_var = len(obs_df), len(var_df)

        layers: dict[str, DenseLayerMatrix] = {}
        for layer in self.layers:
            if layer.source not in df.columns:
                if not layer.required:
                    logger.info(
                        "skipping optional layer %r: source column %r absent from input",
                        layer.name,
                        layer.source,
                    )
                    continue
                raise KeyError(
                    f"required layer {layer.name!r} source column {layer.source!r} "
                    f"is missing from the input"
                )
            values = layer.coerce(df[layer.source])
            layers[layer.name] = self.duplicates.scatter(
                obs_codes,
                var_codes,
                np.asarray(values, dtype="float64"),
                key_ok,
                n_obs,
                n_var,
            )
            warn_if_all_missing(layers[layer.name], layer.name)

        return ConversionPieces(
            X=layers[self.x_layer],
            obs=obs_df,
            var=var_df,
            layers=layers,
        )
