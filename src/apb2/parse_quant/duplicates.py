"""The ``axis.duplicates`` block: how several contributions to one cell become one value.

Wide and long tables mean different things by "duplicate": the wide converter finds
several *columns* claiming one sample, the long converter finds repeated key *rows*.
That is why a policy answers several combining questions rather than one — the two
converters do not share a representation. ``configure_parse.policy_for`` reads the declared
``mode`` once and hands the conversion the policy it names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type DenseLayerMatrix = NDArray[np.float64]
type CategoryCodes = NDArray[np.intp]
type ValidKeyMask = NDArray[np.bool_]


def _first_across_columns(columns: pd.DataFrame) -> pd.Series:
    """Row-wise first non-null across several columns claiming one sample."""
    return columns.bfill(axis=1).iloc[:, 0]


def _first_by_index(series: pd.Series) -> pd.Series:
    """First non-null value per repeated feature-index entry, in row order."""
    return series.groupby(level=0, sort=False).first()


def _scatter_first(
    obs_codes: CategoryCodes,
    var_codes: CategoryCodes,
    values: DenseLayerMatrix,
    key_ok: ValidKeyMask,
    n_obs: int,
    n_var: int,
) -> DenseLayerMatrix:
    """Keep the first non-null value in row order; a cell with no rows stays NaN."""
    matrix = np.full((n_obs, n_var), np.nan, dtype="float64")
    keep = key_ok & ~np.isnan(values)
    oc, vc, vv = obs_codes[keep], var_codes[keep], values[keep]
    flat_cells = oc.astype(np.int64, copy=False) * n_var + vc
    _, first = np.unique(flat_cells, return_index=True)
    matrix[oc[first], vc[first]] = vv[first]
    return matrix


class ErrorOnDuplicates:
    """Repeated keys are a rule error, so combining is never permitted.

    Once the rejections below have passed there is at most one contribution per cell, so
    the combining methods take it. They are reachable and correct, not dead arms: "take
    the only one" is what keeping the first means when there is exactly one.
    """

    def reject_duplicate_keys(self, df: pd.DataFrame, keys: list[str]) -> None:
        valid = df[keys].notna().all(axis=1)
        duplicated = df.loc[valid, keys].duplicated(keep=False)
        if not duplicated.any():
            return
        examples = (
            df.loc[valid, keys].loc[duplicated].drop_duplicates().head(5).to_dict(orient="records")
        )
        raise ValueError(
            "duplicate observation-feature keys are not allowed when "
            f"axis.duplicates.mode='error'; examples: {examples}"
        )

    def reject_multiple_columns(self, layer_name: str, sample: str, columns: list[str]) -> None:
        if len(columns) <= 1:
            return
        raise ValueError(
            "duplicate observation-feature keys are not allowed when "
            f"axis.duplicates.mode='error'; layer {layer_name!r} has multiple "
            f"columns for sample {sample!r}: {columns}"
        )

    def combine_columns(self, columns: pd.DataFrame) -> pd.Series:
        return _first_across_columns(columns)

    def combine_by_index(self, series: pd.Series) -> pd.Series:
        return _first_by_index(series)

    def scatter(
        self,
        obs_codes: CategoryCodes,
        var_codes: CategoryCodes,
        values: DenseLayerMatrix,
        key_ok: ValidKeyMask,
        n_obs: int,
        n_var: int,
    ) -> DenseLayerMatrix:
        return _scatter_first(obs_codes, var_codes, values, key_ok, n_obs, n_var)


class KeepFirstDuplicate:
    """Repeated keys are permitted; the first non-null contribution wins."""

    def reject_duplicate_keys(self, df: pd.DataFrame, keys: list[str]) -> None:
        return

    def reject_multiple_columns(self, layer_name: str, sample: str, columns: list[str]) -> None:
        return

    def combine_columns(self, columns: pd.DataFrame) -> pd.Series:
        return _first_across_columns(columns)

    def combine_by_index(self, series: pd.Series) -> pd.Series:
        return _first_by_index(series)

    def scatter(
        self,
        obs_codes: CategoryCodes,
        var_codes: CategoryCodes,
        values: DenseLayerMatrix,
        key_ok: ValidKeyMask,
        n_obs: int,
        n_var: int,
    ) -> DenseLayerMatrix:
        return _scatter_first(obs_codes, var_codes, values, key_ok, n_obs, n_var)


class SumDuplicates:
    """Repeated keys are permitted; their non-null contributions are summed."""

    def reject_duplicate_keys(self, df: pd.DataFrame, keys: list[str]) -> None:
        return

    def reject_multiple_columns(self, layer_name: str, sample: str, columns: list[str]) -> None:
        return

    def combine_columns(self, columns: pd.DataFrame) -> pd.Series:
        return columns.sum(axis=1)

    def combine_by_index(self, series: pd.Series) -> pd.Series:
        return series.groupby(level=0, sort=False).sum()

    def scatter(
        self,
        obs_codes: CategoryCodes,
        var_codes: CategoryCodes,
        values: DenseLayerMatrix,
        key_ok: ValidKeyMask,
        n_obs: int,
        n_var: int,
    ) -> DenseLayerMatrix:
        """Sum non-null values.

        Mirrors ``GroupBy.sum``: a cell that has rows but only null values is 0.0, while
        a cell with no rows at all stays NaN.
        """
        matrix = np.full((n_obs, n_var), np.nan, dtype="float64")
        finite = key_ok & ~np.isnan(values)
        totals = np.zeros((n_obs, n_var), dtype="float64")
        np.add.at(totals, (obs_codes[finite], var_codes[finite]), values[finite])
        present = np.zeros((n_obs, n_var), dtype=bool)
        present[obs_codes[key_ok], var_codes[key_ok]] = True
        matrix[present] = totals[present]
        return matrix


type DuplicatePolicy = ErrorOnDuplicates | KeepFirstDuplicate | SumDuplicates
