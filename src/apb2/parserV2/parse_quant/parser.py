"""One fully configured quantification level, and the algorithm it runs.

The point of this class is that you can read it. Every collaborator is already configured, so
``parse`` is the sequence of operations and nothing else: read, decompose, prepare each axis
on its own small frame, then reindex each layer onto the axes that survived. No step asks what
vendor, level, layout, value form, duplicate mode, or output format it is dealing with.

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
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    BoundInputReader,
    DuplicatePolicy,
    LayerSetValidator,
    LayerValueParser,
    ParsedLevelWriter,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.data.layer_columns import observation_labels
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.data.raw import (
    LayersRaw,
    ObsRaw,
    RawLayerTable,
    RawToFinalKeyMap,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.parameters.level import QuantificationLevel
from apb2.parserV2.parse_quant.parameters.source import LevelReadPlan

_EXAMPLE_LIMIT = 5
_UNKNOWN_MOD_TOKENS = "unknown_mod_tokens"


class CanonicalKeyCollisionError(ValueError):
    """Distinct raw identities materialized into one valid final identity.

    Never a duplicate: a duplicate is several values for one cell, while this is two cells
    that can no longer be told apart. Reported under every duplicate policy, because no
    policy is allowed to hide it.
    """


class ParserCollection:
    """Compiled level parsers that produce one canonical ``ParsedLevels`` value."""

    __slots__ = ("_parsers",)

    def __init__(self, parsers: tuple[Parser, ...], /) -> None:
        if not parsers:
            raise ValueError("a parser collection requires at least one level parser")
        levels = tuple(parser.level for parser in parsers)
        duplicates = sorted(level for level in set(levels) if levels.count(level) > 1)
        if duplicates:
            raise ValueError(f"duplicate parser levels: {duplicates}")
        self._parsers = parsers

    def parse(self) -> ParsedLevels:
        """Parse every compiled level once and assemble the canonical collection."""
        return ParsedLevels(
            levels={parser.level: parser.parse() for parser in self._parsers},
            uns={},
        )


@dataclass(frozen=True, slots=True)
class Parser:
    """Bind physical IO without duplicating the executable strategy."""

    input_reader: BoundInputReader
    strategy: ParseStrategy
    writer: ParsedLevelWriter

    @property
    def level(self) -> QuantificationLevel:
        """The quantification level this bound parser produces."""
        return self.strategy.level

    def parse(self) -> ParsedLevel:
        """Read the bound input once, then execute the compiled strategy."""
        return self.strategy.parse(self.input_reader.read())

    def convert(self, parsed: ParsedLevel, target: Path, /) -> None:
        """Write a supplied result without reading or parsing again."""
        self.writer.write(parsed, target)


@dataclass(frozen=True, slots=True)
class ParseStrategy:
    """One executable graph, compiled directly from authored declarations and evidence."""

    level: QuantificationLevel
    read: LevelReadPlan
    decomposer: SourceDecomposer
    obs: AxisRuntimePlan
    var: AxisRuntimePlan
    duplicates: DuplicatePolicy
    layer_parsers: Mapping[str, LayerValueParser]
    layer_validator: LayerSetValidator
    provenance: dict[str, JsonValue]

    def parse(self, source: LevelSourceTable) -> ParsedLevel:
        """Execute the shared parsing pipeline on an already-read source table."""
        raw = self.decomposer.decompose(source)

        obs, obs_map = self._prepare_obs(raw.obs)
        var, var_map, unknown_mod_tokens = self._prepare_var(raw.var)
        layers = self._prepare_layers(raw.layers, obs_map, var_map)
        self.layer_validator.validate(layers)
        uns = dict(self.provenance)
        if unknown_mod_tokens:
            uns[_UNKNOWN_MOD_TOKENS] = list(unknown_mod_tokens)

        return ParsedLevel(
            obs=obs,
            var=var,
            primary_layer_name=raw.layers.primary_layer_name,
            uns=uns,
            layers=layers,
            obsm={},
            varm={},
            obsp={},
            varp={},
        )

    # ------------------------------------------------------------------------ the two axes

    def _prepare_obs(self, raw: ObsRaw) -> tuple[ObsFinal, RawToFinalKeyMap]:
        frame, mapping, _diagnostics = self._prepare_axis(raw.frame, raw.raw_key_columns, self.obs)
        return ObsFinal(frame=frame, key_columns=self.obs.keys.final_key_columns), mapping

    def _prepare_var(self, raw: VarRaw) -> tuple[VarFinal, RawToFinalKeyMap, tuple[str, ...]]:
        frame, mapping, unknown_mod_tokens = self._prepare_axis(
            raw.frame, raw.raw_key_columns, self.var
        )
        return (
            VarFinal(frame=frame, key_columns=self.var.keys.final_key_columns),
            mapping,
            unknown_mod_tokens,
        )

    @staticmethod
    def _prepare_axis(
        raw: pl.DataFrame,
        raw_key_columns: tuple[str, ...],
        plan: AxisRuntimePlan,
    ) -> tuple[pl.DataFrame, RawToFinalKeyMap, tuple[str, ...]]:
        """One staged algorithm for both axes: identity first, then public metadata.

        The raw axis already holds one stable-first row per raw key, so nothing here calls
        ``unique`` on the final keys: a repeated valid final key means two raw identities
        collapsed, which is an error rather than a deduplication.
        """
        working, early_tokens = ParseStrategy._materialize_axis_columns(raw, raw, plan.key_phase)

        mapping = RawToFinalKeyMap(
            # Read from the frame as it arrived: a declared column may carry the name of the
            # physical column it was selected from, and materializing it would then replace
            # the raw values this map exists to hold.
            raw_keys=raw.select(list(raw_key_columns)),
            final_keys=ParseStrategy._normalized_keys(
                working.select(list(plan.keys.final_key_columns))
            ),
        )
        ParseStrategy._require_injective_key_mapping(mapping)

        valid = ParseStrategy._valid_final_key_rows(mapping.final_keys)
        final_rows, output_tokens = ParseStrategy._materialize_axis_columns(
            working.filter(valid), raw.filter(valid), plan.output_phase
        )
        return (
            ParseStrategy._finalize_axis_frame(final_rows, outputs=plan.outputs),
            mapping,
            tuple(dict.fromkeys((*early_tokens, *output_tokens))),
        )

    @staticmethod
    def _materialize_axis_columns(
        frame: pl.DataFrame,
        physical: pl.DataFrame,
        phase: AxisPhaseRuntimePlan,
        /,
    ) -> tuple[pl.DataFrame, tuple[str, ...]]:
        """Select from immutable physical values, then compute through logical inputs."""
        result = frame
        if phase.selections:
            selected = physical.select(
                column.coercer.coerce(physical, name=column.name, source=column.source)
                for column in phase.selections
            )
            result = result.with_columns(selected)
        unknown: dict[str, None] = {}
        for computer in phase.computers:
            result, tokens = computer.compute(result)
            unknown.update(dict.fromkeys(tokens))
        return result, tuple(unknown)

    @staticmethod
    def _normalized_keys(keys: pl.DataFrame) -> pl.DataFrame:
        """Make ``NaN`` and null the same absence before any key is compared.

        A key column's dtype is exactly what its declared logical type coerced it to, so
        reading the dtype here is reading that declaration, not sniffing the data.
        """
        return keys.fill_nan(None)

    @staticmethod
    def _valid_final_key_rows(final_keys: pl.DataFrame) -> pl.Series:
        """Which rows have every component of their authored final key."""
        if not final_keys.columns:
            return final_keys.select(pl.repeat(True, pl.len()).alias("_valid")).to_series()
        return final_keys.select(
            pl.all_horizontal(pl.all().is_not_null()).alias("_valid")
        ).to_series()

    @staticmethod
    def _require_injective_key_mapping(mapping: RawToFinalKeyMap) -> None:
        """Distinct raw identities must stay distinct once they are canonicalized.

        Decomposition already made raw keys unique. Repeated valid final keys therefore
        identify collisions directly; nested evidence keeps overlapping column names apart.
        """
        duplicated = (
            ParseStrategy._valid_final_key_rows(mapping.final_keys)
            & mapping.final_keys.is_duplicated()
        )
        if not duplicated.any():
            return
        evidence = pl.DataFrame(
            {"final": mapping.final_keys.to_struct(), "raw": mapping.raw_keys.to_struct()}
        ).filter(duplicated)
        raise CanonicalKeyCollisionError(
            f"{evidence.n_unique(subset='final')} value(s) of the final key "
            f"{list(mapping.final_keys.columns)} were produced by more than one raw identity; "
            f"examples of final keys and their raw evidence: {evidence.head(_EXAMPLE_LIMIT).to_dicts()}"
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
            parser = self.layer_parsers[layer.layer_name]
            mappable = self._retain_mappable_layer(layer, obs_map, var_map)
            resolved = self.duplicates.resolve(mappable, parser)
            aligned = self._align_layer_keys(
                resolved,
                obs_map,
                var_map,
            )
            layers[layer.layer_name] = parser.parse(aligned)
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
        kept_columns = (
            pl.DataFrame(
                {
                    "label": value_columns,
                    "valid": ParseStrategy._valid_final_key_rows(obs.final_keys),
                }
            )
            .filter(pl.col("valid"))
            .select("label")
            .to_series()
            .to_list()
        )
        usable_var = var.raw_keys.filter(ParseStrategy._valid_final_key_rows(var.final_keys))
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
        var_valid = ParseStrategy._valid_final_key_rows(var.final_keys)
        final_keys = var.final_keys.filter(var_valid)
        spine = var.raw_keys.filter(var_valid)
        joined = spine.join(
            layer.values, on=keys, how="left", nulls_equal=True, maintain_order="left"
        )
        value_columns = layer.values.columns[len(keys) :]
        labels = observation_labels(len(value_columns), reserved=final_keys.columns)
        values = final_keys.hstack(
            joined.select(value_columns).rename(dict(zip(value_columns, labels, strict=True)))
        )
        return FinalLayerTable(
            layer_name=layer.layer_name,
            var_key_columns=tuple(final_keys.columns),
            values=values,
        )
