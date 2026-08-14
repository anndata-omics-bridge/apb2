"""How several contributions to one observation-feature cell become one value.

``axis.duplicates.mode`` selects this, and before this module the selection was re-made at
six places across the two converters: a mode-to-aggfunc translation, that aggfunc branched
on again inside the scatter, two near-identical pre-pass rejections, and three more
branches inside one loop in the wide converter. Each site had to agree with the others
about what a mode meant, and one of them did not — the wide converter never checked
``keep_all_as_raw_table`` at all, so an unimplemented mode silently behaved as keep-first.

Wide and long tables mean different things by "duplicate": the wide converter finds several
*columns* claiming one sample, the long converter finds repeated key *rows*. That is why a
policy answers three combining questions rather than one — the two converters do not share
a representation, which is a separate matter from the one this module settles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apb2.convert._pieces import (
    CategoryCodes,
    DenseLayerMatrix,
    ValidKeyMask,
)
from apb2.vendor_parse_rules.model import DuplicateMode, Duplicates


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


@dataclass(frozen=True, slots=True)
class ErrorOnDuplicates:
    """Repeated keys are a rule error, so combining is never permitted.

    Once the rejections below have passed there is at most one contribution per cell, so
    the combining methods take it. They are reachable and correct, not dead arms: "take the
    only one" is what keeping the first means when there is exactly one.
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
        return columns.bfill(axis=1).iloc[:, 0]

    def combine_by_index(self, series: pd.Series) -> pd.Series:
        return series.groupby(level=0, sort=False).first()

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


@dataclass(frozen=True, slots=True)
class KeepFirstDuplicate:
    """Repeated keys are permitted; the first non-null contribution wins."""

    def reject_duplicate_keys(self, df: pd.DataFrame, keys: list[str]) -> None:
        return

    def reject_multiple_columns(self, layer_name: str, sample: str, columns: list[str]) -> None:
        return

    def combine_columns(self, columns: pd.DataFrame) -> pd.Series:
        return columns.bfill(axis=1).iloc[:, 0]

    def combine_by_index(self, series: pd.Series) -> pd.Series:
        return series.groupby(level=0, sort=False).first()

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


@dataclass(frozen=True, slots=True)
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

        Mirrors ``GroupBy.sum``: a cell that has rows but only null values is 0.0, while a
        cell with no rows at all stays NaN.
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

_BY_MODE: dict[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(),
    "keep_first": KeepFirstDuplicate(),
    "aggregate": SumDuplicates(),
    # "keep_all_as_raw_table" is absent on purpose: the schema accepts the mode and no
    # policy implements it, so policy_for raises once, naming it, before any conversion
    # work starts.
}


def policy_for(duplicates: Duplicates) -> DuplicatePolicy:
    """Select the policy a rule's duplicate mode names.

    Raises NotImplementedError for a mode the schema permits but no policy implements.
    """
    policy = _BY_MODE.get(duplicates.mode)
    if policy is None:
        raise NotImplementedError(f"duplicates.mode={duplicates.mode!r} is not yet supported")
    return policy
