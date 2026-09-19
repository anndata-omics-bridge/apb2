"""Resolve authored runtime declarations against physical source evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnSelection,
    AxisKeyPlan,
    AxisMaterializationConfig,
    AxisSourcePlan,
    CoalesceColumnConfig,
    ComputedColumnConfig,
    JoinNonemptyColumnConfig,
    ProformaSequenceColumnConfig,
    ResolvedAxisColumnPlan,
    WorkingAxisConfiguration,
)
from apb2.parserV2.parse_quant.parameters.level import (
    JsonValue,
    ResolvedLevelPlan,
    WorkingParseConfiguration,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    LayerContractConfig,
    WorkingMeasurementLayer,
)
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentLayout,
    ColumnLabeledFragmentSeparationConfig,
    DecompositionConfig,
    DelimitedFragmentDecompositionConfig,
    DelimitedSourceEvidence,
    ExcelSourceEvidence,
    FragmentSeparationConfig,
    FrameSourceEvidence,
    LevelReadPlan,
    LongDecompositionConfig,
    LongRawLayerSource,
    LongSourceLayout,
    NumericTextFormat,
    PositionalFragmentLayout,
    PositionalFragmentSeparationConfig,
    SourceEvidence,
    WideDecompositionConfig,
    WideRawLayerPlan,
    WideRawLayerSource,
    WideSourceLayout,
)

_EMPTY_RATIO = 0.001
_POPULATED_RATIO = 0.5
_SAMPLE_GROUP = "sample"

type RawSources = Mapping[str, tuple[str, ...]]
"""Which physical columns each declared logical name is ultimately derived from."""


@dataclass(frozen=True, slots=True)
class _ResolvedLayers:
    """The layers one source can actually provide, and how their values are laid out.

    Exactly one of ``long_sources`` and ``wide_plans`` is populated: they are the two shapes
    a decomposition configuration is built from, and the layout decided which before this
    value existed.
    """

    retained: tuple[WorkingMeasurementLayer, ...]
    required_names: tuple[str, ...]
    long_sources: tuple[LongRawLayerSource, ...]
    wide_plans: tuple[WideRawLayerPlan, ...]
    source_columns: frozenset[str]
    plain_numeric_columns: frozenset[str]


@dataclass(frozen=True, slots=True)
class SourcePlanResolver:
    """Bind one working configuration without depending on vendor-rule storage."""

    _configuration: WorkingParseConfiguration

    # -------------------------------------------------------------------- source resolution

    def resolve(self, evidence: SourceEvidence) -> ResolvedLevelPlan:
        """Bind the working parameters to one observed source, once and completely.

        Raises ``IncompatibleSourceError`` when this source cannot satisfy the level: a
        required column, layer, or final key that its header does not provide.
        """
        working = self._configuration
        present = frozenset(evidence.columns)
        var = self._resolve_axis(working.var, present, self._var_synthesized())
        layers = self._resolve_layers(evidence.columns, present, self._accounted(var))
        obs = self._resolve_axis(working.obs, present, self._obs_synthesized())
        read = self._read_plan(evidence, obs, var, layers)
        self._require_aggregatable(evidence, read, layers)
        numbers = self._resolved_numbers(evidence)
        return ResolvedLevelPlan(
            level=working.level,
            number_format=numbers,
            read=read,
            decomposition=self._decomposition(layers),
            obs=obs,
            var=var,
            duplicate_mode=working.measurements.duplicate_mode,
            raw_value_presence=tuple(
                layer.raw_presence_config(numbers) for layer in layers.retained
            ),
            layer_values=tuple(layer.canonical_value_config(numbers) for layer in layers.retained),
            layer_contract=LayerContractConfig(
                primary_layer_name=working.measurements.primary_layer_name,
                required_names=layers.required_names,
                empty_ratio=_EMPTY_RATIO,
                populated_ratio=_POPULATED_RATIO,
            ),
            provenance={**working.provenance, "layer_roles": self._layer_roles(layers.retained)},
        )

    def _var_synthesized(self) -> tuple[str, ...]:
        """Names the fragment separator creates as real columns of the scalar-long table."""
        return self._configuration.source_layout.synthesized_var_columns()

    def _obs_synthesized(self) -> tuple[str, ...]:
        """A wide observation axis comes from header captures, not from selected columns."""
        if isinstance(self._configuration.source_layout, WideSourceLayout):
            return self._configuration.obs.final_key_columns
        return ()

    def _accounted(self, var: ResolvedAxisColumnPlan) -> frozenset[str]:
        """Names a permissive wide layer pattern must not mistake for a sample column."""
        declaration = self._configuration.var.columns
        return frozenset(
            {
                *declaration.declared_order,
                *(selection.source for selection in declaration.required_selections),
                *(selection.source for selection in declaration.optional_selections),
                *var.source.keys.raw_key_columns,
                *var.source.payload_sources,
            }
        )

    # ---------------------------------------------------------------------------- axis plans

    def _resolve_axis(
        self,
        axis: WorkingAxisConfiguration,
        present: frozenset[str],
        synthesized: tuple[str, ...],
    ) -> ResolvedAxisColumnPlan:
        """Resolve one axis: prune what this source cannot provide, then plan both phases."""
        declaration = axis.columns
        missing = [
            selection.source
            for selection in declaration.required_selections
            if selection.source not in present
        ]
        if missing:
            raise IncompatibleSourceError(
                f"{self._label()} selects column(s) {missing} that this source does not carry"
            )
        selections = (
            *declaration.required_selections,
            *(
                selection
                for selection in declaration.optional_selections
                if selection.source in present
            ),
        )
        skipped = {
            selection.name
            for selection in declaration.optional_selections
            if selection.source not in present
        }
        raw_sources, computers = self._materializable(
            declaration.computed, selections, synthesized, skipped
        )
        keys = self._axis_key_plan(axis.final_key_columns, computers, raw_sources, skipped)
        # Diagnostics and unknown-token errors cover the unfiltered raw axis, including
        # metadata-only normalizations. This changes scheduling, not identity membership.
        early_outputs = tuple(
            computer.name
            for computer in computers
            if isinstance(computer, ProformaSequenceColumnConfig)
        )
        closure = self._dependency_closure((*axis.final_key_columns, *early_outputs), computers)
        self._require_output_phase_keeps_identity(axis.final_key_columns, closure, selections)
        return ResolvedAxisColumnPlan(
            source=AxisSourcePlan(
                keys=keys,
                payload_sources=self._ordered_unique(
                    column
                    for column in (*(selection.source for selection in selections),)
                    if column not in set(keys.raw_key_columns)
                ),
            ),
            key_phase=self._phase(selections, computers, closure, inside=True),
            output_phase=self._phase(selections, computers, closure, inside=False),
            outputs=tuple(name for name in declaration.declared_order if name not in skipped),
            skipped=frozenset(skipped),
        )

    def _require_output_phase_keeps_identity(
        self,
        final_keys: tuple[str, ...],
        closure: frozenset[str],
        selections: tuple[AxisColumnSelection, ...],
    ) -> None:
        """Metadata is materialized after the collision check, so it may not rewrite a key.

        A selection or computation naming a final key must be inside the identity closure;
        outside it, it would run after the check that just proved the keys are distinct.
        """
        offenders = sorted(
            selection.name
            for selection in selections
            if selection.name in set(final_keys) and selection.name not in closure
        )
        if offenders:
            raise ValueError(
                f"{self._label()} materializes the axis key(s) {offenders} outside the "
                "identity closure, which would overwrite them after validation"
            )

    def _materializable(
        self,
        declared: tuple[ComputedColumnConfig, ...],
        selections: tuple[AxisColumnSelection, ...],
        synthesized: tuple[str, ...],
        skipped: set[str],
    ) -> tuple[RawSources, tuple[ComputedColumnConfig, ...]]:
        """Walk the declarations in order, binding each name to its physical closure.

        Reading the environment before rebinding is what lets a computed column consume a
        selected column of its own name — the coalesce that widens ``Proteins`` reads the
        selected ``Proteins`` and then becomes it — without the walk chasing its own tail.
        """
        raw_sources: dict[str, tuple[str, ...]] = {name: (name,) for name in synthesized}
        for selection in selections:
            raw_sources[selection.name] = (selection.source,)
        retained: list[ComputedColumnConfig] = []
        for computer in declared:
            resolved = self._prune_inputs(computer, raw_sources)
            if resolved is None:
                skipped.add(computer.name)
                continue
            retained.append(resolved)
            raw_sources[resolved.name] = self._ordered_unique(
                column for name in resolved.inputs for column in raw_sources[name]
            )
        return raw_sources, tuple(retained)

    def _axis_key_plan(
        self,
        final_keys: tuple[str, ...],
        computers: tuple[ComputedColumnConfig, ...],
        raw_sources: RawSources,
        skipped: set[str],
    ) -> AxisKeyPlan:
        """Derive the three identity column sets from the authored keys alone.

        Generic by construction: the walk asks each key how it is materialized and follows
        that answer, so no vendor or level appears anywhere in it.
        """
        by_name = {computer.name: computer for computer in computers}
        inputs: list[str] = []
        raw: list[str] = []
        for key in final_keys:
            if key in skipped or key not in raw_sources:
                raise IncompatibleSourceError(
                    f"{self._label()} cannot materialize the axis key {key!r} from this source"
                )
            computer = by_name.get(key)
            inputs.extend(computer.inputs if computer is not None else (key,))
            raw.extend(raw_sources[key])
        return AxisKeyPlan(
            raw_key_columns=self._ordered_unique(raw),
            key_input_columns=self._ordered_unique(inputs),
            final_key_columns=final_keys,
        )

    # -------------------------------------------------------------------------------- layers

    def _resolve_layers(
        self,
        columns: tuple[str, ...],
        present: frozenset[str],
        accounted: frozenset[str],
    ) -> _ResolvedLayers:
        """Resolve every declared measurement against this header, by physical layout."""
        layout = self._configuration.source_layout
        if isinstance(layout, WideSourceLayout):
            return self._resolve_wide_layers(columns, accounted)
        return self._resolve_long_layers(present)

    def _resolve_long_layers(self, present: frozenset[str]) -> _ResolvedLayers:
        """Every long layer names one exact column; a packed one is split before reading it."""
        measurements = self._configuration.measurements
        required = {layer.name for layer in measurements.required_layers}
        missing = [
            layer.source for layer in measurements.required_layers if layer.source not in present
        ]
        if missing:
            raise IncompatibleSourceError(
                f"{self._label()} requires layer source column(s) {missing} that this source "
                "does not carry"
            )
        retained = tuple(
            layer for layer in measurements.authored_layers() if layer.source in present
        )
        return _ResolvedLayers(
            retained=retained,
            required_names=tuple(layer.name for layer in retained if layer.name in required),
            long_sources=tuple(
                LongRawLayerSource(name=layer.name, source_column=layer.source)
                for layer in retained
            ),
            wide_plans=(),
            source_columns=frozenset(layer.source for layer in retained),
            plain_numeric_columns=frozenset(
                layer.source for layer in retained if layer.supports_native_numeric_read()
            ),
        )

    def _resolve_wide_layers(
        self, columns: tuple[str, ...], accounted: frozenset[str]
    ) -> _ResolvedLayers:
        """Expand each layer's header regex, then align every layer to the primary samples.

        The primary layer defines the observation axis. A permissive pattern must not turn
        an accounted-for column into an extra sample, and a layer that matched only tokens
        outside that axis is not evidence of more observations.
        """
        measurements = self._configuration.measurements
        candidates = tuple(name for name in columns if name not in accounted)
        matches = {
            layer.name: self._match_samples(candidates, layer.source)
            for layer in measurements.authored_layers()
        }
        primary = measurements.primary_layer_name
        samples = self._ordered_unique(sample for _column, sample in matches[primary])
        if not samples:
            raise IncompatibleSourceError(
                f"{self._label()} matched no observation column for its primary layer {primary!r}"
            )
        required = {layer.name for layer in measurements.required_layers}
        retained: list[WorkingMeasurementLayer] = []
        plans: list[WideRawLayerPlan] = []
        for layer in measurements.authored_layers():
            aligned = tuple(
                WideRawLayerSource(source_column=column, sample=sample)
                for column, sample in matches[layer.name]
                if sample in set(samples)
            )
            if layer.name in required and not matches[layer.name]:
                raise IncompatibleSourceError(
                    f"{self._label()} matched no column for its required layer "
                    f"{layer.name!r} pattern {layer.source!r}"
                )
            if layer.name not in required and not aligned:
                continue
            retained.append(layer)
            plans.append(WideRawLayerPlan(name=layer.name, sources=aligned))
        return _ResolvedLayers(
            retained=tuple(retained),
            required_names=tuple(layer.name for layer in retained if layer.name in required),
            long_sources=(),
            wide_plans=tuple(plans),
            source_columns=frozenset(
                source.source_column for plan in plans for source in plan.sources
            ),
            plain_numeric_columns=frozenset(
                source.source_column
                for layer, plan in zip(retained, plans, strict=True)
                if layer.supports_native_numeric_read()
                for source in plan.sources
            ),
        )

    # -------------------------------------------------------------------- read and structure

    def _read_plan(
        self,
        evidence: SourceEvidence,
        obs: ResolvedAxisColumnPlan,
        var: ResolvedAxisColumnPlan,
        layers: _ResolvedLayers,
    ) -> LevelReadPlan:
        """Project exactly this level's transitive source closure, and decide every dtype.

        A measurement column is read natively only when the rule sums its values, because
        summing needs numbers and the schema already checked that those layers are plain. Every
        other measurement stays text: parsing it as a float is a semantic transformation owned
        by the parser, and real exports write ``-``, ``NA``, or ``False`` in a column a rule
        calls numeric — which an eager numeric read cannot survive.
        """
        lexical = frozenset(
            {
                *obs.source.keys.raw_key_columns,
                *obs.source.payload_sources,
                *var.source.keys.raw_key_columns,
                *var.source.payload_sources,
                *self._configuration.source_layout.packed_sources(),
            }
        )
        needed = lexical | layers.source_columns
        projected = tuple(name for name in evidence.columns if name in needed)
        if isinstance(evidence, FrameSourceEvidence):
            # Parquet carries its own schema; overriding it would discard physical types.
            return LevelReadPlan(
                projected_columns=projected,
                text_sources=frozenset(),
                native_numeric_sources=frozenset(),
            )
        native = (
            frozenset()
            if evidence.number_format.thousands_marks
            or self._configuration.measurements.duplicate_mode != "aggregate"
            else frozenset(
                column
                for column in projected
                if column in layers.plain_numeric_columns and column not in lexical
            )
        )
        return LevelReadPlan(
            projected_columns=projected,
            text_sources=frozenset(projected) - native,
            native_numeric_sources=native,
        )

    def _decomposition(self, layers: _ResolvedLayers) -> DecompositionConfig:
        """Name the one physical shape this level's table has."""
        layout = self._configuration.source_layout
        primary = self._configuration.measurements.primary_layer_name
        if isinstance(layout, WideSourceLayout):
            return WideDecompositionConfig(
                kind="wide", primary_layer_name=primary, layer_plans=layers.wide_plans
            )
        long = LongDecompositionConfig(
            kind="long", primary_layer_name=primary, layer_sources=layers.long_sources
        )
        if isinstance(layout, LongSourceLayout):
            return long
        return DelimitedFragmentDecompositionConfig(
            kind="delimited_fragment",
            separator=self._separator(layout, layers),
            long=long,
        )

    def _separator(
        self,
        layout: PositionalFragmentLayout | ColumnLabeledFragmentLayout,
        layers: _ResolvedLayers,
    ) -> FragmentSeparationConfig:
        """Keep the retained packed sources in authored order; drop what is absent."""
        packed = tuple(
            column for column in layout.packed_value_sources if column in layers.source_columns
        )
        if not packed:
            raise IncompatibleSourceError(
                f"{self._label()} carries none of the packed fragment columns "
                f"{list(layout.packed_value_sources)}"
            )
        if isinstance(layout, ColumnLabeledFragmentLayout):
            return ColumnLabeledFragmentSeparationConfig(
                kind="column",
                label_source=layout.label_source,
                label_output=layout.label_output,
                delimiter=layout.delimiter,
                packed_value_sources=packed,
            )
        return PositionalFragmentSeparationConfig(
            kind="positional",
            label_output=layout.label_output,
            delimiter=layout.delimiter,
            packed_value_sources=packed,
        )

    def _require_aggregatable(
        self, evidence: SourceEvidence, read: LevelReadPlan, layers: _ResolvedLayers
    ) -> None:
        """Reject an aggregate rule whose values this source cannot deliver as numbers.

        Checked here rather than at runtime because it is a property of the rule and the
        source together, and the alternative is discovering it after reading a large table.
        """
        if self._configuration.measurements.duplicate_mode != "aggregate":
            return
        if isinstance(evidence, FrameSourceEvidence):
            numeric = frozenset(name for name, dtype in evidence.dtypes if dtype.is_numeric())
            offenders = sorted(layers.source_columns - numeric)
        else:
            offenders = sorted(layers.source_columns - read.native_numeric_sources)
        if offenders:
            raise IncompatibleSourceError(
                f"{self._label()} aggregates duplicate cells, which requires native numeric "
                f"layer values; these resolve to text: {offenders}"
            )

    def _label(self) -> str:
        """How this level names itself in an error message."""
        working = self._configuration
        software = working.provenance.get("software_name", "rule")
        return f"{software!r} level {working.level!r}"

    @staticmethod
    def _layer_roles(
        layers: Sequence[WorkingMeasurementLayer],
    ) -> dict[str, JsonValue]:
        roles = sorted({role for layer in layers for role in layer.roles})
        projected: dict[str, JsonValue] = {
            role: [layer.name for layer in layers if role in layer.roles] for role in roles
        }
        return projected

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _match_samples(candidates: tuple[str, ...], pattern: str) -> tuple[tuple[str, str], ...]:
        """Every candidate column this layer pattern claims, with the sample it captured."""
        compiled = re.compile(pattern)
        matched: list[tuple[str, str]] = []
        for name in candidates:
            match = compiled.match(name)
            if match is not None:
                matched.append((name, match.group(_SAMPLE_GROUP)))
        return tuple(matched)

    @staticmethod
    def _prune_inputs(
        computer: ComputedColumnConfig, available: RawSources
    ) -> ComputedColumnConfig | None:
        """Drop the inputs this source cannot provide, or report the computation as blocked.

        Only the two combining operations survive a missing input: coalescing or joining the
        columns that are present is the operation the rule asked for. Everything else needs
        every input it declared.
        """
        if isinstance(computer, CoalesceColumnConfig | JoinNonemptyColumnConfig):
            kept = tuple(name for name in computer.inputs if name in available)
            if not kept:
                return None
            if isinstance(computer, CoalesceColumnConfig):
                return CoalesceColumnConfig(kind="coalesce", name=computer.name, inputs=kept)
            return JoinNonemptyColumnConfig(
                kind="join_nonempty",
                name=computer.name,
                inputs=kept,
                separator=computer.separator,
            )
        if any(name not in available for name in computer.inputs):
            return None
        return computer

    @staticmethod
    def _dependency_closure(
        final_keys: tuple[str, ...], computers: tuple[ComputedColumnConfig, ...]
    ) -> frozenset[str]:
        """Every declared name that must be materialized before identity can be checked."""
        by_name = {computer.name: computer for computer in computers}
        closure = set(final_keys)
        pending = list(final_keys)
        while pending:
            computer = by_name.get(pending.pop())
            if computer is None:
                continue
            for name in computer.inputs:
                if name not in closure:
                    closure.add(name)
                    pending.append(name)
        return frozenset(closure)

    @staticmethod
    def _phase(
        selections: tuple[AxisColumnSelection, ...],
        computers: tuple[ComputedColumnConfig, ...],
        closure: frozenset[str],
        *,
        inside: bool,
    ) -> AxisMaterializationConfig:
        """Split the declarations at the identity closure, keeping declaration order in each."""
        return AxisMaterializationConfig(
            selections=tuple(
                selection for selection in selections if (selection.name in closure) is inside
            ),
            computers=tuple(
                computer for computer in computers if (computer.name in closure) is inside
            ),
        )

    @staticmethod
    def _resolved_numbers(evidence: SourceEvidence) -> NumericTextFormat:
        """The notation a retained token must be read with; Parquet values are already numbers."""
        if isinstance(evidence, DelimitedSourceEvidence | ExcelSourceEvidence):
            return evidence.number_format
        return NumericTextFormat(decimal_mark=".", thousands_marks=())
