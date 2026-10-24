"""One fully configured quantification level, and the algorithm it runs.

The point of this class is that you can read it. Every collaborator is already configured, so
``parse`` is the sequence of operations and nothing else: decompose the source it is handed,
prepare each axis on its own small frame, place each layer's long rows at the final positions
that survived, resolve repeated cells, parse and widen one layer at a time, then validate the
layers and leave out the fields that carry no information. No step asks what vendor, level,
layout, value form, duplicate mode, or output format it is dealing with.

Two things are decided here and nowhere else. Identity: raw keys become authored final keys on
the small axis frames, and two distinct raw identities that collapse into one valid final
identity are an information loss, reported rather than resolved. And validity: a row whose
final key is incomplete cannot enter an axis, so it gets no final position, and the layer rows
that pointed at it leave when the layer is placed.

Parsing never writes: ``parse`` returns the levels, and only a result writer persists them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import polars as pl

from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    BoundInputReader,
    DuplicatePolicy,
    FieldPruner,
    LayerSetValidator,
    LayerValueParser,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.conversion_record import ConversionEvidence
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.layer_columns import (
    OBS_POS,
    VALUE,
    VAR_POS,
    ObservationGrid,
    observation_labels,
)
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    LevelHierarchy,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.data.raw import (
    DISAGREEING_VALUES,
    OBS_ROW,
    VAR_ROW,
    FinalPositions,
    LongLayer,
    ObsRaw,
    RawLevel,
    VarRaw,
)
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.duplicates import KeepBestDuplicate
from apb2.parserV2.parse_quant.parameters.level import QuantificationLevel
from apb2.parserV2.parse_quant.parameters.source import (
    InputFileRecord,
    InputTable,
    LevelReadPlan,
    LevelSourceTable,
)

_EXAMPLE_LIMIT = 5


class CanonicalKeyCollisionError(ConversionError):
    """Distinct raw identities materialized into one valid final identity.

    Never a duplicate: a duplicate is several values for one cell, while this is two cells
    that can no longer be told apart. Reported under every duplicate policy, because no
    policy is allowed to hide it.
    """


@dataclass(frozen=True, slots=True)
class LevelParseTimings:
    """Wall time spent reading and parsing one selected quantification level."""

    level: QuantificationLevel
    read_seconds: float
    parse_seconds: float


class ParserCollection:
    """Compiled level parsers that produce one canonical ``ParsedLevels`` value.

    ``provenance`` is the root parse provenance, such as the search-parameter record; ``read``
    records the files read before parsing, such as the parameter file. Parsing adds every input
    table, once per file, to the root ``input_files``.
    """

    __slots__ = ("_hierarchy", "_parsers", "_provenance", "_read")

    def __init__(
        self,
        parsers: tuple[Parser, ...],
        provenance: Mapping[str, JsonValue],
        read: tuple[InputFileRecord, ...] = (),
        /,
    ) -> None:
        self._provenance = dict(provenance)
        self._read = read
        if not parsers:
            raise ValueError("a parser collection requires at least one level parser")
        levels = tuple(parser.level for parser in parsers)
        duplicates = sorted(level for level in set(levels) if levels.count(level) > 1)
        if duplicates:
            raise ValueError(f"duplicate parser levels: {duplicates}")
        self._hierarchy = parsers[0].strategy.hierarchy
        if any(parser.strategy.hierarchy != self._hierarchy for parser in parsers):
            raise ValueError("parser levels must declare one shared hierarchy")
        order = self._hierarchy.level_names
        self._parsers = tuple(sorted(parsers, key=lambda parser: order.index(parser.level)))

    def parse(self) -> ParsedLevels:
        """Parse every compiled level once and assemble the canonical collection."""
        return self.parse_with_timings()[0]

    def parse_with_timings(self) -> tuple[ParsedLevels, tuple[LevelParseTimings, ...]]:
        """Parse once while measuring each bound read and strategy execution."""
        levels: dict[QuantificationLevel, ParsedLevel] = {}
        timings: list[LevelParseTimings] = []
        tables: dict[Path, InputTable] = {}
        for parser in self._parsers:
            parsed_level, timing, read = parser.parse_with_timings()
            levels[parser.level] = parsed_level
            timings.append(timing)
            for table in read:
                tables.setdefault(table.file.path, table)
        input_files = [*self._read, *(table.record() for table in tables.values())]
        uns: dict[str, JsonValue] = {
            "provenance": {**self._provenance, "input_files": cast(JsonValue, input_files)}
        }
        parsed = ParsedLevels(levels=levels, uns=uns, hierarchy=self._hierarchy)
        return parsed, tuple(timings)


@dataclass(frozen=True, slots=True)
class Parser:
    """Bind physical input without duplicating the executable strategy."""

    input_reader: BoundInputReader
    strategy: ParseStrategy

    @property
    def level(self) -> QuantificationLevel:
        """The quantification level this bound parser produces."""
        return self.strategy.level

    def parse(self) -> ParsedLevel:
        """Read the bound input once, then execute the compiled strategy."""
        return self.parse_with_timings()[0]

    def parse_with_timings(
        self,
    ) -> tuple[ParsedLevel, LevelParseTimings, tuple[InputTable, ...]]:
        """Read and parse once, reporting the two wall times and the tables read.

        The read table is let go as soon as it is decomposed: its key and payload columns are
        then held by nothing, and each layer's column only until that layer is parsed.
        """
        started = perf_counter()
        with logged_step("read", level=self.level):
            source = self.input_reader.read()
        read_seconds = perf_counter() - started
        tables = source.input_tables
        started = perf_counter()
        raw = self.strategy.decompose(source)
        del source
        parsed = self.strategy.parse_decomposed(raw)
        timing = LevelParseTimings(self.level, read_seconds, perf_counter() - started)
        return parsed, timing, tables


@dataclass(frozen=True, slots=True)
class SemanticRoles:
    """The var column each role names, and the layers each role names, as the rule declares."""

    columns: Mapping[str, str]
    layers: Mapping[str, tuple[str, ...]]

    def assign(self, var: VarFinal, layers: Mapping[str, FinalLayerTable]) -> None:
        """Give the surviving var columns and layers their roles."""
        var.roles = {
            role: column for role, column in self.columns.items() if column in var.frame.columns
        }
        for name, layer in layers.items():
            layer.semantic_roles = tuple(
                role for role, names in self.layers.items() if name in names
            )


@dataclass(frozen=True, slots=True)
class ParseStrategy:
    """One executable graph, compiled directly from authored declarations and evidence."""

    level: QuantificationLevel
    read: LevelReadPlan
    decomposer: SourceDecomposer
    obs: AxisRuntimePlan
    var: AxisRuntimePlan
    duplicates: DuplicatePolicy | KeepBestDuplicate
    layer_parsers: Mapping[str, LayerValueParser]
    layer_validator: LayerSetValidator
    pruner: FieldPruner
    hierarchy: LevelHierarchy
    roles: SemanticRoles
    provenance: Mapping[str, JsonValue]

    def parse(self, source: LevelSourceTable) -> ParsedLevel:
        """Execute the shared parsing pipeline on an already-read source table."""
        return self.parse_decomposed(self.decompose(source))

    def decompose(self, source: LevelSourceTable) -> RawLevel:
        """Reduce the read table to raw axes and long layers, the first step of ``parse``."""
        with logged_step("parse.decompose", level=self.level):
            return self.decomposer.decompose(source)

    def parse_decomposed(self, raw: RawLevel) -> ParsedLevel:
        """Every step of ``parse`` after decomposition; it consumes ``raw``'s layers."""
        with logged_step("parse.obs", level=self.level):
            obs, obs_positions, _obs_keys = self._prepare_obs(raw.obs)
        with logged_step("parse.var", level=self.level):
            var, var_positions, var_keys, unknown_mod_tokens = self._prepare_var(raw.var)
        positions = FinalPositions(
            var=var_positions,
            obs=obs_positions,
            var_keys=var_keys,
            grid=ObservationGrid(
                height=var.frame.height,
                labels=observation_labels(
                    obs.frame.height, reserved=self.var.keys.final_key_columns
                ),
            ),
        )
        with logged_step("parse.layers", level=self.level):
            pending = list(raw.layers)
            raw.layers = ()
            layers, unreadable_numeric = self._prepare_layers(pending, positions)
        with logged_step("parse.validate", level=self.level):
            effectively_empty = self.layer_validator.validate(layers)
            pruned = self.pruner.prune(obs, var, layers, effectively_empty or {})
        obs, var, layers = pruned.obs, pruned.var, pruned.layers
        self.roles.assign(var, layers)
        evidence = ConversionEvidence(
            unknown_mod_tokens,
            unreadable_numeric,
            pruned.single_valued,
            pruned.effectively_empty,
            self._var_disagreements(raw.var, var_keys),
        )

        return ParsedLevel(
            obs=obs,
            var=var,
            primary_layer_name=raw.primary_layer_name,
            uns=evidence.record(self.provenance),
            layers=layers,
            obsm={},
            varm={},
            obsp={},
            varp={},
        )

    # ------------------------------------------------------------------------ the two axes

    def _prepare_obs(self, raw: ObsRaw) -> tuple[ObsFinal, pl.Series, pl.DataFrame]:
        frame, positions, keys, _diagnostics = self._prepare_axis(
            raw.frame, raw.raw_key_columns, self.obs
        )
        return ObsFinal(frame=frame, key_columns=self.obs.keys.final_key_columns), positions, keys

    def _prepare_var(
        self, raw: VarRaw
    ) -> tuple[VarFinal, pl.Series, pl.DataFrame, tuple[str, ...]]:
        frame, positions, keys, unknown_mod_tokens = self._prepare_axis(
            raw.frame, raw.raw_key_columns, self.var
        )
        return (
            VarFinal(frame=frame, key_columns=self.var.keys.final_key_columns),
            positions,
            keys,
            unknown_mod_tokens,
        )

    @staticmethod
    def _prepare_axis(
        raw: pl.DataFrame,
        raw_key_columns: tuple[str, ...],
        plan: AxisRuntimePlan,
    ) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, tuple[str, ...]]:
        """One staged algorithm for both axes: identity first, then public metadata.

        The raw axis already holds one stable-first row per raw key, so nothing here calls
        ``unique`` on the final keys: a repeated valid final key means two raw identities
        collapsed, which is an error rather than a deduplication.

        Returns the final axis, each raw row's final position (null where the row's final key
        is incomplete), and the raw keys of the rows that stayed, in final order.
        """
        working, early_tokens = ParseStrategy._materialize_axis_columns(raw, raw, plan.key_phase)

        # Read from the frame as it arrived: a declared column may carry the name of the
        # physical column it was selected from, and materializing it would then replace the
        # raw values the identity check reports.
        raw_keys = raw.select(list(raw_key_columns))
        # A number coerced from text such as ``nan`` is as absent as null.
        final_keys = working.select(list(plan.keys.final_key_columns)).fill_nan(None)
        ParseStrategy._require_injective_keys(raw_keys, final_keys)

        valid = ParseStrategy._valid_final_key_rows(final_keys)
        final_rows, output_tokens = ParseStrategy._materialize_axis_columns(
            working.filter(valid), raw.filter(valid), plan.output_phase
        )
        positions = valid.cast(pl.UInt32).cum_sum() - 1
        return (
            final_rows.select(list(plan.outputs)),
            pl.select(pl.when(valid).then(positions)).to_series(),
            raw_keys.filter(valid),
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
    def _var_disagreements(raw: VarRaw, kept: pl.DataFrame) -> dict[str, JsonValue]:
        """Per vendor column, the features in the result whose rows disagree, with examples."""
        keys = list(raw.raw_key_columns)
        found: dict[str, JsonValue] = {}
        for source, disagreeing in raw.disagreements.items():
            remaining = disagreeing.join(kept, on=keys, how="semi", nulls_equal=True)
            if remaining.height:
                found[source] = {
                    "features": remaining.height,
                    "examples": [
                        {"key": [row[key] for key in keys], "values": row[DISAGREEING_VALUES]}
                        for row in remaining.head(_EXAMPLE_LIMIT).iter_rows(named=True)
                    ],
                }
        return found

    @staticmethod
    def _valid_final_key_rows(final_keys: pl.DataFrame) -> pl.Series:
        """Which rows have every component of their authored final key."""
        if not final_keys.columns:
            return final_keys.select(pl.repeat(True, pl.len()).alias("_valid")).to_series()
        return final_keys.select(
            pl.all_horizontal(pl.all().is_not_null()).alias("_valid")
        ).to_series()

    @staticmethod
    def _require_injective_keys(raw_keys: pl.DataFrame, final_keys: pl.DataFrame) -> None:
        """Distinct raw identities must stay distinct once they are canonicalized.

        Decomposition already made raw keys unique. Repeated valid final keys therefore
        identify collisions directly; nested evidence keeps overlapping column names apart.
        """
        duplicated = ParseStrategy._valid_final_key_rows(final_keys) & final_keys.is_duplicated()
        if not duplicated.any():
            return
        evidence = pl.DataFrame(
            {"final": final_keys.to_struct(), "raw": raw_keys.to_struct()}
        ).filter(duplicated)
        raise CanonicalKeyCollisionError(
            f"{evidence.n_unique(subset='final')} value(s) of the final key "
            f"{list(final_keys.columns)} were produced by more than one raw identity; "
            f"examples of final keys and their raw evidence: {evidence.head(_EXAMPLE_LIMIT).to_dicts()}"
        )

    # --------------------------------------------------------------------------- the layers

    def _prepare_layers(
        self,
        pending: list[LongLayer],
        positions: FinalPositions,
    ) -> tuple[dict[str, FinalLayerTable], dict[str, JsonValue]]:
        """Parse the layers in authored order, letting each raw layer go once it is parsed."""
        layers: dict[str, FinalLayerTable] = {}
        diagnostics: dict[str, JsonValue] = {}
        duplicates = self._ranked_duplicates(pending, positions)
        previous: tuple[pl.DataFrame, LongLayer] | None = None
        while pending:
            layer = pending.pop(0)
            values = self.layer_parsers[layer.layer_name]
            # Layers read from one table point at the same raw rows; placing them alike lets
            # every parsed layer share one copy of the position columns.
            raw_rows = layer.rows.select(VAR_ROW, OBS_ROW)
            like = previous[1] if previous and previous[0].equals(raw_rows) else None
            placed = positions.place(layer, like)
            previous = (raw_rows, LongLayer(layer.layer_name, placed.rows.select(VAR_POS, OBS_POS)))
            once, repeated = self._claimed(placed, values).split()
            resolved = duplicates.resolve(repeated)
            cells = (
                replace(once, rows=pl.concat([once.rows, resolved.rows], how="vertical_relaxed"))
                if resolved.rows.height
                else once
            )
            parsed, evidence = values.parse(cells, positions.grid)
            layers[layer.layer_name] = parsed
            if evidence:
                diagnostics[layer.layer_name] = evidence
        return layers, diagnostics

    def _ranked_duplicates(
        self, raw: Sequence[LongLayer], positions: FinalPositions
    ) -> DuplicatePolicy:
        """keep_best ranks once, on its ``by`` layer; every other policy needs no ranking."""
        if not isinstance(self.duplicates, KeepBestDuplicate):
            return self.duplicates
        ranking = next((layer for layer in raw if layer.layer_name == self.duplicates.by), None)
        if ranking is None:
            raise ConversionError(
                f"keep_best ranks by layer {self.duplicates.by!r}, which this source lacks"
            )
        return self.duplicates.ranked(
            self._claimed(positions.place(ranking), self.layer_parsers[ranking.layer_name])
        )

    @staticmethod
    def _claimed(placed: LongLayer, values: LayerValueParser, /) -> LongLayer:
        """The placed layer with every scalar that claims no cell nulled.

        Placing is fixed validity filtering, not a policy: rows pointing at an axis row that
        did not survive have left, so the duplicate policy is asked only about cells that can
        enter the result. Presence is the layer's own question; the raw scalar is kept
        unconverted.
        """
        present = placed.rows.select(
            values.present(pl.col(VALUE), placed.rows.schema[VALUE])
        ).to_series()
        if present.all():
            return placed
        value = pl.col(VALUE)
        return replace(
            placed, rows=placed.rows.with_columns(pl.when(present).then(value).alias(VALUE))
        )
