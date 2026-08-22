"""One fully configured quantification level, and the algorithm it runs.

The point of this class is that you can read it. Every collaborator is already configured, so
``parse`` is the sequence of operations and nothing else: read, decompose, prepare each axis
on its own small frame, then reindex each layer onto the axes that survived. No step asks what
vendor, level, layout, encoding, duplicate mode, or output format it is dealing with.

Two things are decided here and nowhere else. Identity: raw keys become authored final keys on
the small axis frames, and two distinct raw identities that collapse into one valid final
identity are an information loss, reported rather than resolved. And validity: a row whose
final key is incomplete cannot enter an axis, so the temporary key map keeps it just long
enough to remove the layer cells that pointed at it.

``convert`` writes a result it is handed. It never calls ``parse``, so a second read is
impossible unless a caller asks for one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import polars as pl

from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    BoundInputReader,
    DuplicatePolicy,
    ModificationNormalizer,
    ParsedLevelWriter,
    RawValuePresence,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.data.layer_columns import observation_labels
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    VarFinal,
)
from apb2.parserV2.parse_quant.data.raw import (
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    RawToFinalKeyMap,
    VarRaw,
)
from apb2.parserV2.parse_quant.parameters.working import QuantificationLevel

_EXAMPLE_LIMIT = 5
_UNKNOWN_MOD_TOKENS = "unknown_mod_tokens"


class CanonicalKeyCollisionError(ValueError):
    """Distinct raw identities materialized into one valid final identity.

    Never a duplicate: a duplicate is several values for one cell, while this is two cells
    that can no longer be told apart. Reported under every duplicate policy, because no
    policy is allowed to hide it.
    """


class AxisShapeError(ValueError):
    """One axis collaborator returned a series that does not line up with its input."""


class Parser:
    """One quantification level's completed strategy graph."""

    __slots__ = (
        "_decomposer",
        "_duplicates",
        "_input",
        "_modification_normalizers",
        "_obs_plan",
        "_provenance",
        "_raw_value_presence",
        "_var_plan",
        "_writer",
        "level",
    )

    def __init__(
        self,
        *,
        level: QuantificationLevel,
        input_reader: BoundInputReader,
        decomposer: SourceDecomposer,
        obs_plan: AxisRuntimePlan,
        var_plan: AxisRuntimePlan,
        modification_normalizers: tuple[ModificationNormalizer, ...],
        duplicates: DuplicatePolicy,
        raw_value_presence: Mapping[str, RawValuePresence],
        writer: ParsedLevelWriter,
        provenance: Mapping[str, JsonValue],
    ) -> None:
        self.level = level
        self._input = input_reader
        self._decomposer = decomposer
        self._obs_plan = obs_plan
        self._var_plan = var_plan
        self._modification_normalizers = modification_normalizers
        self._duplicates = duplicates
        self._raw_value_presence = dict(raw_value_presence)
        self._writer = writer
        self._provenance = dict(provenance)

    def parse(self) -> ParsedLevel:
        """Read one bound source and return one parsed level."""
        source = self._input.read()
        raw = self._decomposer.decompose(source)

        obs, obs_map = self._prepare_obs(raw.obs)
        var, var_map, unknown_mod_tokens = self._prepare_var(raw.var)
        layers = self._prepare_layers(raw.layers, obs_map, var_map)
        uns = dict(self._provenance)
        if unknown_mod_tokens:
            uns[_UNKNOWN_MOD_TOKENS] = list(unknown_mod_tokens)

        return ParsedLevel(
            obs=obs,
            var=var,
            primary_layer_name=raw.layers.primary_layer_name,
            uns=uns,
            layers=layers,
        )

    def convert(self, parsed: ParsedLevel, target: Path, /) -> None:
        """Write a result the caller already has. This never parses anything."""
        self._writer.write(parsed, target)

    # ------------------------------------------------------------------------ the two axes

    def _prepare_obs(self, raw: ObsRaw) -> tuple[ObsFinal, RawToFinalKeyMap]:
        frame, mapping = self._prepare_axis(
            raw.frame,
            raw.raw_key_columns,
            {},
            self._obs_plan,
        )
        return ObsFinal(frame=frame, key_columns=self._obs_plan.keys.final_key_columns), mapping

    def _prepare_var(self, raw: VarRaw) -> tuple[VarFinal, RawToFinalKeyMap, tuple[str, ...]]:
        derived = self._normalize_modification_columns(
            raw.frame,
            self._modification_normalizers,
        )
        unknown_mod_tokens = self._distinct_unknown_modification_tokens(derived)
        frame, mapping = self._prepare_axis(
            raw.frame,
            raw.raw_key_columns,
            derived,
            self._var_plan,
        )
        return (
            VarFinal(frame=frame, key_columns=self._var_plan.keys.final_key_columns),
            mapping,
            unknown_mod_tokens,
        )

    @staticmethod
    def _prepare_axis(
        raw: pl.DataFrame,
        raw_key_columns: tuple[str, ...],
        derived: Mapping[str, pl.Series],
        plan: AxisRuntimePlan,
    ) -> tuple[pl.DataFrame, RawToFinalKeyMap]:
        """One staged algorithm for both axes: identity first, then public metadata.

        The raw axis already holds one stable-first row per raw key, so nothing here calls
        ``unique`` on the final keys: a repeated valid final key means two raw identities
        collapsed, which is an error rather than a deduplication.
        """
        working = Parser._add_derived_columns(raw, derived)
        working = Parser._materialize_axis_columns(working, plan.key_phase)

        mapping = RawToFinalKeyMap(
            # Read from the frame as it arrived: a declared column may carry the name of the
            # physical column it was selected from, and materializing it would then replace
            # the raw values this map exists to hold.
            raw_keys=raw.select(list(raw_key_columns)),
            final_keys=Parser._normalized_keys(working.select(list(plan.keys.final_key_columns))),
        )
        Parser._require_injective_key_mapping(mapping)

        valid = Parser._valid_final_key_rows(mapping.final_keys)
        final_rows = working.filter(valid)
        final_rows = Parser._materialize_axis_columns(final_rows, plan.output_phase)
        return (
            Parser._finalize_axis_frame(final_rows, outputs=plan.outputs),
            mapping,
        )

    @staticmethod
    def _add_derived_columns(frame: pl.DataFrame, derived: Mapping[str, pl.Series]) -> pl.DataFrame:
        """Put the modification-derived columns on the axis frame, under their own names."""
        if not derived:
            return frame
        for name, values in derived.items():
            if values.len() != frame.height:
                raise AxisShapeError(
                    f"derived column {name!r} returned {values.len()} row(s) for "
                    f"{frame.height} axis row(s)"
                )
        return frame.with_columns([values.alias(name) for name, values in derived.items()])

    @staticmethod
    def _normalize_modification_columns(
        frame: pl.DataFrame,
        normalizers: tuple[ModificationNormalizer, ...],
    ) -> dict[str, pl.Series]:
        """Hand each normalizer exactly the series it declared, and merge what comes back."""
        derived: dict[str, pl.Series] = {}
        for normalizer in normalizers:
            columns = tuple(frame.get_column(name) for name in normalizer.sources)
            derived.update(normalizer.normalize(columns))
        return derived

    @staticmethod
    def _distinct_unknown_modification_tokens(
        derived: Mapping[str, pl.Series],
    ) -> tuple[str, ...]:
        """Collect unresolved vendor tokens once, in first-observed order."""
        values = derived.get(_UNKNOWN_MOD_TOKENS)
        if values is None:
            return ()
        rows = cast(list[list[str] | None], values.to_list())
        return tuple(dict.fromkeys(token for row in rows if row for token in row))

    @staticmethod
    def _materialize_axis_columns(
        frame: pl.DataFrame,
        phase: AxisPhaseRuntimePlan,
        /,
    ) -> pl.DataFrame:
        """Run one phase's configured operations in order: selections, then computations."""
        result = frame
        for selected in phase.selections:
            values = result.get_column(selected.source)
            coerced = selected.coercer.coerce(
                values,
                name=selected.name,
                source=selected.source,
            )
            result = result.with_columns(
                Parser._same_shape(coerced, result.height, selected.name).alias(selected.name)
            )
        for computer in phase.computers:
            inputs = tuple(result.get_column(name) for name in computer.inputs)
            computed = computer.compute(inputs)
            result = result.with_columns(
                Parser._same_shape(computed, result.height, computer.name).alias(computer.name)
            )
        return result

    @staticmethod
    def _same_shape(values: pl.Series, height: int, name: str) -> pl.Series:
        """Every axis operation returns its input's length and row order, or it is a defect."""
        if values.len() != height:
            raise AxisShapeError(
                f"axis operation for {name!r} returned {values.len()} row(s) for {height} "
                "input row(s); length and row order must be preserved"
            )
        return values

    @staticmethod
    def _normalized_keys(keys: pl.DataFrame) -> pl.DataFrame:
        """Make ``NaN`` and null the same absence before any key is compared.

        A key column's dtype is exactly what its declared logical type coerced it to, so
        reading the dtype here is reading that declaration, not sniffing the data.
        """
        return keys.with_columns(
            [
                pl.when(pl.col(name).is_nan()).then(None).otherwise(pl.col(name)).alias(name)
                for name, dtype in keys.schema.items()
                if dtype.is_float()
            ]
        )

    @staticmethod
    def _valid_final_key_rows(final_keys: pl.DataFrame) -> pl.Series:
        """Which rows have every component of their authored final key."""
        if not final_keys.columns:
            return pl.Series("_valid", [True] * final_keys.height, dtype=pl.Boolean)
        return final_keys.select(
            pl.all_horizontal([pl.col(name).is_not_null() for name in final_keys.columns]).alias(
                "_valid"
            )
        ).to_series()

    @staticmethod
    def _require_injective_key_mapping(mapping: RawToFinalKeyMap) -> None:
        """Distinct raw identities must stay distinct once they are canonicalized.

        Both halves are copied into positionally named columns first, because a raw key and a
        final key may well share a name — a rule may select ``Charge`` from a column called
        ``Charge`` — and the comparison has to keep them apart.
        """
        valid = Parser._valid_final_key_rows(mapping.final_keys)
        final = mapping.final_keys.filter(valid)
        raw = mapping.raw_keys.filter(valid)
        final_columns = [f"final_{index}" for index in range(final.width)]
        raw_columns = [f"raw_{index}" for index in range(raw.width)]
        pairs = pl.DataFrame(
            [
                *(
                    final.get_column(name).rename(label)
                    for name, label in zip(final.columns, final_columns, strict=True)
                ),
                *(
                    raw.get_column(name).rename(label)
                    for name, label in zip(raw.columns, raw_columns, strict=True)
                ),
            ]
        ).unique(maintain_order=True)
        collisions = (
            pairs.group_by(final_columns, maintain_order=True).len().filter(pl.col("len") > 1)
        )
        if not collisions.height:
            return
        evidence = (
            collisions.head(_EXAMPLE_LIMIT)
            .join(pairs, on=final_columns, how="left")
            .drop("len")
            .rename(
                dict(
                    zip(
                        [*final_columns, *raw_columns],
                        [*mapping.final_keys.columns, *mapping.raw_keys.columns],
                        strict=True,
                    )
                )
            )
        )
        raise CanonicalKeyCollisionError(
            f"{collisions.height} value(s) of the final key "
            f"{list(mapping.final_keys.columns)} were produced by more than one raw identity; "
            f"the raw evidence behind the first of them is: {evidence.to_dicts()}"
        )

    @staticmethod
    def _finalize_axis_frame(frame: pl.DataFrame, *, outputs: tuple[str, ...]) -> pl.DataFrame:
        """Project the axis to its retained declared columns, in authored order."""
        return frame.select(list(outputs))

    # --------------------------------------------------------------------------- the layers

    def _prepare_layers(
        self,
        raw: LayersRaw,
        obs_map: RawToFinalKeyMap,
        var_map: RawToFinalKeyMap,
    ) -> dict[str, FinalLayerTable]:
        layers: dict[str, FinalLayerTable] = {}
        for layer in raw.values:
            mappable = self._retain_mappable_layer(layer, obs_map, var_map)
            resolved = self._duplicates.resolve(
                mappable,
                self._raw_value_presence[layer.layer_name],
            )
            layers[layer.layer_name] = self._align_layer_keys(
                resolved,
                obs_map,
                var_map,
            )
        return layers

    @staticmethod
    def _retain_mappable_layer(
        layer: RawLayerTable,
        obs: RawToFinalKeyMap,
        var: RawToFinalKeyMap,
        /,
    ) -> RawLayerTable:
        """Drop the cells that point at an identity the axes could not keep.

        Fixed validity filtering, not a policy: the duplicate policy is then asked only
        about cells that can actually enter the result, while still grouping by raw keys.
        """
        keys = list(layer.raw_var_key_columns)
        value_columns = layer.values.columns[len(keys) :]
        kept_columns = [
            label
            for label, usable in zip(
                value_columns, Parser._valid_final_key_rows(obs.final_keys), strict=True
            )
            if usable
        ]
        usable_var = var.raw_keys.filter(Parser._valid_final_key_rows(var.final_keys))
        rows = layer.values.join(
            usable_var.unique(maintain_order=True),
            on=keys,
            how="semi",
            nulls_equal=True,
            maintain_order="left",
        )
        return RawLayerTable(
            layer_name=layer.layer_name,
            raw_var_key_columns=layer.raw_var_key_columns,
            values=rows.select([*keys, *kept_columns]),
        )

    @staticmethod
    def _align_layer_keys(
        layer: RawLayerTable,
        obs: RawToFinalKeyMap,
        var: RawToFinalKeyMap,
        /,
    ) -> FinalLayerTable:
        """Reindex one resolved layer onto the final axes; the only producer of a final layer.

        The valid variable map in final order is the left spine, so a final variable this
        layer never measured becomes a row of nulls rather than a missing row.
        """
        keys = list(layer.raw_var_key_columns)
        var_valid = Parser._valid_final_key_rows(var.final_keys)
        final_keys = var.final_keys.filter(var_valid)
        spine = var.raw_keys.filter(var_valid)
        joined = spine.join(
            layer.values, on=keys, how="left", nulls_equal=True, maintain_order="left"
        )
        value_columns = layer.values.columns[len(keys) :]
        labels = observation_labels(len(value_columns), reserved=final_keys.columns)
        values = pl.DataFrame(
            [
                *(final_keys.get_column(name) for name in final_keys.columns),
                *(
                    joined.get_column(column).rename(label)
                    for column, label in zip(value_columns, labels, strict=True)
                ),
            ]
        )
        return FinalLayerTable(
            layer_name=layer.layer_name,
            var_key_columns=tuple(final_keys.columns),
            values=values,
        )
