"""The shape block: convert one key-prepared flat table into matrices and axis frames.

``conversion_for(rule, strict=…)`` is the single composition-root dispatch over the rule
shape. Past it the shape does not exist: a ``LongConversion`` scatters long rows into
dense matrices via integer category codes (pivot_table materialises a huge transient for
high-cardinality var axes; the scatter is O(nnz + obs·var) with identical semantics), a
``WideConversion`` reads its observation axis out of the layer regex captures.

The axis frames a conversion returns are *raw*: they carry the prepared key columns plus
every vendor source column the declared columns will be materialized from — that
materialization runs afterwards, on these deduplicated frames (``columns.finish``), not
on the flat table.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from apb2.parse_quant.duplicates import DuplicatePolicy, policy_for
from apb2.parse_quant.layers import LayerPlan, warn_if_all_missing
from apb2.parse_quant.modifications import modification_sources
from apb2.parse_quant.result import ParsedData
from apb2.vendor_parse_rules.model import (
    ColumnGroup,
    LongRule,
    WideRule,
    group_names,
    modification_outputs,
)

logger = logging.getLogger(__name__)

KEY_SEPARATOR = "_"

# A layer holding under this share of its cells is "effectively empty" — but only counts
# as lost data when a sibling layer from the same file is populated, which is what
# separates a parse failure from a genuinely sparse experiment.
_EMPTY_RATIO = 0.001
_POPULATED_RATIO = 0.5


class LayerContractError(ValueError):
    """A converted layer carries too few values to be a usable quantitative layer."""


def _build_index(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Build a string index from one or more key columns, vectorised."""
    if len(keys) == 1:
        return df[keys[0]].astype("string")
    joined = df[keys[0]].astype("string")
    for key in keys[1:]:
        joined = joined + KEY_SEPARATOR + df[key].astype("string")
    return joined


def _build_axis_frame(df: pd.DataFrame, keys: list[str], carry: list[str]) -> pd.DataFrame:
    """Take the first occurrence per key tuple, carrying the columns present in ``df``.

    ``carry`` names the prepared keys plus every raw source the axis's declared columns
    are materialized from later; sources this export does not carry (skipped
    ``optional_select``) are simply absent and drop out here. Keys head the carry list,
    so the frame is one column-projection plus one dedup — nothing else to copy.
    """
    present = [column for column in carry if column in df.columns]
    out = df[present].drop_duplicates(subset=keys)
    out.index = pd.Index(_build_index(out, keys), name=KEY_SEPARATOR.join(keys))
    return out


def _carry_columns(keys: list[str], group: ColumnGroup, extras: tuple[str, ...]) -> list[str]:
    """Everything an axis frame must take off the flat table, in stable order.

    Declared names cover the already-prepared key-closure columns; raw sources cover
    everything ``columns.finish`` materializes afterwards. Absent names (skipped
    optionals, columns not yet materialized) drop out at the present-filter.
    """
    return list(
        dict.fromkeys(
            [
                *keys,
                *group_names(group),
                *group.select.values(),
                *group.optional_select.values(),
                *extras,
            ]
        )
    )


def _var_extras(rule: LongRule | WideRule) -> tuple[str, ...]:
    """Raw modification sources/outputs and the fragment label the var frame may need."""
    extras: list[str] = []
    if rule.modifications is not None:
        extras.extend(modification_sources(rule.modifications))
        extras.extend(sorted(modification_outputs(rule.modifications)))
    if rule.fragments is not None:
        extras.append(rule.fragments.label_output)
    return tuple(extras)


def _check_layer_occupancy(
    layers: dict[str, NDArray[np.float64]], *, x_layer: str, strict: bool
) -> None:
    """Reject a conversion that lost its quantities.

    An effectively empty layer beside a populated sibling means the vendor column was
    read but its values did not survive parsing — a mis-detected decimal separator or an
    unhandled sentinel, not an empty experiment. That is an error for ``x_layer``, whose
    emptiness makes the whole object unusable, and a warning for the rest; ``strict``
    promotes those warnings to errors.
    """
    ratios = {
        name: (float(np.count_nonzero(np.isfinite(matrix))) / matrix.size if matrix.size else 0.0)
        for name, matrix in layers.items()
    }
    populated = [name for name, ratio in ratios.items() if ratio >= _POPULATED_RATIO]
    suspects = [name for name, ratio in ratios.items() if ratio < _EMPTY_RATIO]
    if not suspects or not populated:
        return
    reference = ", ".join(populated[:3])
    for name in suspects:
        message = (
            f"layer {name!r} is effectively empty ({ratios[name]:.2%}) while {reference} is "
            "populated — the source column was read but its values did not parse; check the "
            "vendor number format and missing-value sentinels"
        )
        if name == x_layer or strict:
            raise LayerContractError(message)
        logger.warning(message)


class LongConversion:
    """Convert one long table: one row per (observation, feature)."""

    def __init__(self, rule: LongRule, *, strict: bool) -> None:
        self.obs_keys = list(rule.axis.obs_keys)
        self.var_keys = list(rule.axis.var_keys)
        self.obs_carry = _carry_columns(self.obs_keys, rule.columns.obs, ())
        self.var_carry = _carry_columns(self.var_keys, rule.columns.var, _var_extras(rule))
        self.layers = tuple(LayerPlan(rule, layer) for layer in rule.layers)
        self.x_layer = rule.axis.x_layer
        self.duplicates: DuplicatePolicy = policy_for(rule.axis.duplicates)
        self.strict = strict

    def parse(self, df: pd.DataFrame) -> ParsedData:
        """Scatter the long values into dense (obs x var) matrices via category codes."""
        self.duplicates.reject_duplicate_keys(df, [*self.obs_keys, *self.var_keys])

        obs_df = _build_axis_frame(df, self.obs_keys, self.obs_carry)
        var_df = _build_axis_frame(df, self.var_keys, self.var_carry)

        # Map every input row to its position in the obs/var axes. _build_axis_frame keeps
        # the first occurrence per key, so the Categorical codes index into obs_df/var_df.
        obs_codes = pd.Categorical(_build_index(df, self.obs_keys), categories=obs_df.index).codes
        var_codes = pd.Categorical(_build_index(df, self.var_keys), categories=var_df.index).codes
        key_ok = df[self.obs_keys + self.var_keys].notna().all(axis=1).to_numpy()

        n_obs, n_var = len(obs_df), len(var_df)
        layers: dict[str, NDArray[np.float64]] = {}
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

        _check_layer_occupancy(layers, x_layer=self.x_layer, strict=self.strict)
        return ParsedData(X=layers[self.x_layer], obs=obs_df, var=var_df, uns={}, layers=layers)


class WideConversion:
    """Convert one wide table: one row per feature, observations in matrix headers."""

    def __init__(self, rule: WideRule, *, strict: bool) -> None:
        self.var_keys = list(rule.axis.var_keys)
        self.var_carry = _carry_columns(self.var_keys, rule.columns.var, _var_extras(rule))
        self.layers = tuple(LayerPlan(rule, layer) for layer in rule.layers)
        self.x_layer = rule.axis.x_layer
        self.duplicates: DuplicatePolicy = policy_for(rule.axis.duplicates)
        self.obs_outputs = tuple(rule.axis.obs_keys)
        self.software_name = rule.software_name
        # Everything the frame can hold besides vendor sample columns: the carry set plus
        # the synthesized modification columns. A rule whose sample pattern cannot anchor
        # on a suffix (AlphaDIA's run columns are bare run names) must not match any of
        # them as extra samples.
        self.excluded = frozenset(self.var_carry) | {"stripped_sequence", "unknown_mod_tokens"}
        self.strict = strict

    def parse(self, df: pd.DataFrame) -> ParsedData:
        """Gather each layer's sample-captured columns into dense (obs x var) matrices."""
        self.duplicates.reject_duplicate_keys(df, self.var_keys)

        # The frame carries raw vendor columns plus the prepared key columns; none of the
        # accounted-for names may be mistaken for a vendor sample column.
        headers = [column for column in df.columns if column not in self.excluded]

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

        var_df = _build_axis_frame(df, self.var_keys, self.var_carry)

        layers: dict[str, NDArray[np.float64]] = {}
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
            layers[layer.name] = self._gather_layer_matrix(df, layer, headers, sample_order, var_df)
            warn_if_all_missing(layers[layer.name], layer.name)

        obs_names = list(sample_order)
        obs_df = pd.DataFrame(
            {name: list(obs_names) for name in self.obs_outputs},
            index=pd.Index(obs_names, name="sample"),
        )
        _check_layer_occupancy(layers, x_layer=self.x_layer, strict=self.strict)
        return ParsedData(X=layers[self.x_layer], obs=obs_df, var=var_df, uns={}, layers=layers)

    def _gather_layer_matrix(
        self,
        df: pd.DataFrame,
        layer: LayerPlan,
        headers: list[str],
        sample_order: list[str],
        var_df: pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Build the (n_obs x n_var) matrix for a single wide layer."""
        sample_to_columns: dict[str, list[str]] = {}
        for column, sample in _matching_columns(headers, layer.source):
            sample_to_columns.setdefault(sample, []).append(column)

        matrix = np.full((len(sample_order), len(var_df)), np.nan, dtype="float64")
        feature_index = _build_index(df, self.var_keys)
        for i, sample in enumerate(sample_order):
            columns = sample_to_columns.get(sample, [])
            if not columns:
                continue
            self.duplicates.reject_multiple_columns(layer.name, sample, columns)
            values = [layer.coerce(df[column]) for column in columns]
            series = self.duplicates.combine_columns(pd.concat(values, axis=1))
            series.index = feature_index
            series = self.duplicates.combine_by_index(series)
            matrix[i, :] = series.reindex(var_df.index).to_numpy(dtype="float64")
        return matrix


def _matching_columns(headers: list[str], pattern: str) -> list[tuple[str, str]]:
    """Return [(column, sample_token), ...] for columns matching ``pattern``."""
    compiled = re.compile(pattern)
    out: list[tuple[str, str]] = []
    for header in headers:
        match = compiled.match(header)
        if match is None:
            continue
        out.append((header, match.group("sample")))
    return out


type Conversion = LongConversion | WideConversion


def conversion_for(rule: LongRule | WideRule, *, strict: bool) -> Conversion:
    """Read a rule's shape once, and return the conversion it names."""
    if isinstance(rule, LongRule):
        return LongConversion(rule, strict=strict)
    return WideConversion(rule, strict=strict)
