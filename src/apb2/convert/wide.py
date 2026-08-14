"""Convert a wide-format DataFrame into AnnData pieces using a LongRule | WideRule."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apb2.convert._axis import build_axis_frame, build_index
from apb2.convert._pieces import ConversionPieces, DenseLayerMatrix
from apb2.convert.duplicates import DuplicatePolicy
from apb2.convert.layers import LayerPlan
from apb2.convert.numeric import (
    warn_if_all_missing,
)

logger = logging.getLogger(__name__)


def _matching_columns(headers: list[str], pattern: str) -> list[tuple[str, str]]:
    """Return [(column, sample_token), ...] for columns matching `pattern`."""
    compiled = re.compile(pattern)
    out: list[tuple[str, str]] = []
    for h in headers:
        m = compiled.match(h)
        if m is None:
            continue
        out.append((h, m.group("sample")))
    return out


def _gather_layer_matrix(
    df: pd.DataFrame,
    layer: LayerPlan,
    headers: list[str],
    sample_order: list[str],
    var_index: pd.Index,
    var_keys: list[str],
    duplicates: DuplicatePolicy,
) -> DenseLayerMatrix:
    """Build (n_obs x n_var) matrix for a single wide layer."""
    matches = _matching_columns(headers, layer.source)
    sample_to_columns: dict[str, list[str]] = {}
    for column, sample in matches:
        sample_to_columns.setdefault(sample, []).append(column)

    n_obs = len(sample_order)
    n_var = len(var_index)
    matrix = np.full((n_obs, n_var), np.nan, dtype="float64")
    feature_index = build_index(df, var_keys)

    for i, sample in enumerate(sample_order):
        columns = sample_to_columns.get(sample, [])
        if not columns:
            continue
        duplicates.reject_multiple_columns(layer.name, sample, columns)
        values = [layer.coerce(df[column]) for column in columns]
        series = duplicates.combine_columns(pd.concat(values, axis=1))
        series.index = feature_index
        series = duplicates.combine_by_index(series)
        matrix[i, :] = series.reindex(var_index).to_numpy(dtype="float64")
    return matrix


@dataclass(frozen=True, slots=True)
class WideConversion:
    """Everything needed to convert one wide table, with no shape flag left to read.

    ``Layer.source`` finally has one meaning here. In the document it is a ``str`` that means
    an exact column name for long rules and a sample regex for wide ones — one field, two
    types, disambiguated by a flag on another object. A ``WideConversion``'s sources are
    always the regex kind.
    """

    var_keys: tuple[str, ...]
    var_columns: tuple[str, ...]
    layers: tuple[LayerPlan, ...]
    x_layer: str
    duplicates: DuplicatePolicy
    obs_outputs: tuple[str, ...]
    declared_columns: frozenset[str]
    software_name: str

    def convert(self, df: pd.DataFrame) -> ConversionPieces:
        """Convert a wide DataFrame to backend-neutral pieces."""
        self.duplicates.reject_duplicate_keys(df, list(self.var_keys))

        # Modifications and column materialization both run before this point, so APB's
        # derived columns and the rule's renamed `select` outputs are on the frame by now.
        # Neither is a vendor sample column, so keep them away from the layer patterns.
        headers = [column for column in df.columns if column not in self.declared_columns]

        # The x-layer defines the observation axis. Optional auxiliary layers may expose
        # summary columns or malformed tokens; those must not expand the run axis.
        x_layer = next(layer for layer in self.layers if layer.name == self.x_layer)
        sample_order = list(
            dict.fromkeys(sample for _, sample in _matching_columns(headers, x_layer.source))
        )
        sample_set = set(sample_order)

        if not sample_order:
            raise ValueError(
                f"no columns matched any layer pattern for rule {self.software_name!r}; "
                f"layers: {[layer.source for layer in self.layers]}"
            )

        var_df = build_axis_frame(df, list(self.var_keys), list(self.var_columns))

        layers: dict[str, DenseLayerMatrix] = {}
        for layer in self.layers:
            layer_matches = _matching_columns(headers, layer.source)
            extra_samples = list(
                dict.fromkeys(sample for _, sample in layer_matches if sample not in sample_set)
            )
            if extra_samples:
                logger.warning(
                    "ignoring layer %r sample token(s) outside x-layer axis: %s",
                    layer.name,
                    extra_samples,
                )
            axis_matches = [
                (column, sample) for column, sample in layer_matches if sample in sample_set
            ]
            if not layer.required and not axis_matches:
                logger.info(
                    "skipping optional layer %r: no x-layer samples matched %r",
                    layer.name,
                    layer.source,
                )
                continue
            layers[layer.name] = _gather_layer_matrix(
                df,
                layer,
                headers,
                sample_order,
                var_df.index,
                list(self.var_keys),
                self.duplicates,
            )
            warn_if_all_missing(layers[layer.name], layer.name)

        obs_names = list(sample_order)
        obs_df = pd.DataFrame(
            {name: list(obs_names) for name in self.obs_outputs},
            index=pd.Index(obs_names, name="sample"),
        )
        return ConversionPieces(
            X=layers[self.x_layer],
            obs=obs_df,
            var=var_df,
            layers=layers,
        )
