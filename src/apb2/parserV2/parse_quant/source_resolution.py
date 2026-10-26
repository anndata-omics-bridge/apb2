"""Resolve authored runtime declarations against physical source evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from apb2.parserV2.parse_quant.axis_columns import (
    CoalesceColumn,
    JoinNonemptyColumn,
    MarkerColumn,
)
from apb2.parserV2.parse_quant.contracts import (
    AxisPhaseRuntimePlan,
    AxisRuntimePlan,
    FeatureValuePolicy,
    FragmentTableSeparator,
    SelectedAxisColumn,
    SourceDecomposer,
)
from apb2.parserV2.parse_quant.decomposition import (
    DelimitedFragmentSourceDecomposer,
    LongSourceDecomposer,
    WideSourceDecomposer,
)
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.fragments import (
    ColumnLabeledFragmentTableSeparator,
    PositionalFragmentTableSeparator,
)
from apb2.parserV2.parse_quant.layer_validation import LayerContractValidator
from apb2.parserV2.parse_quant.modifications import (
    EmbeddedSiteListNormalizer,
    SequenceColumn,
    SiteListNormalizer,
    TokenRegexNormalizer,
)
from apb2.parserV2.parse_quant.operations import (
    NUMERIC_DUPLICATE_MODES,
    ComputedOperation,
    WorkingAxisConfiguration,
    WorkingParseConfiguration,
    duplicate_policy,
    feature_value_policy,
    make_axis_coercer,
    make_layer_parser,
)
from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnSelection,
    AxisKeyPlan,
    AxisSourcePlan,
    FeatureDuplicates,
    FeatureRanking,
)
from apb2.parserV2.parse_quant.parameters.measurements import (
    PlainNumericLayerDeclaration,
    WorkingMeasurementLayer,
)
from apb2.parserV2.parse_quant.parameters.source import (
    ColumnLabeledFragmentLayout,
    DelimitedSourceEvidence,
    FrameSourceEvidence,
    LevelReadPlan,
    LongRawLayerSource,
    LongSourceLayout,
    NumericTextFormat,
    PositionalFragmentLayout,
    SourceEvidence,
    WideRawLayerPlan,
    WideRawLayerSource,
    WideSourceLayout,
)
from apb2.parserV2.parse_quant.parser import ParseStrategy, SemanticRoles
from apb2.parserV2.parse_quant.plan_json import PLAN_JSON_KEY, resolved_plan_json
from apb2.parserV2.parse_quant.uninformative import UninformativeFields

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

    def sources_of(self, names: Iterable[str]) -> frozenset[str]:
        """The physical columns holding the named layers' values."""
        named = frozenset(names)
        return frozenset(
            {source.source_column for source in self.long_sources if source.name in named}
            | {
                source.source_column
                for plan in self.wide_plans
                if plan.name in named
                for source in plan.sources
            }
        )


@dataclass(frozen=True, slots=True)
class SourcePlanResolver:
    """Bind one working configuration without depending on vendor-rule storage."""

    _configuration: WorkingParseConfiguration

    # -------------------------------------------------------------------- source resolution

    def resolve(
        self, evidence: SourceEvidence, *, checks: Literal["standard", "strict"] = "standard"
    ) -> ParseStrategy:
        """Compile source-dependent decisions directly into the executable strategy."""
        working = self._configuration
        present = frozenset(evidence.columns)
        numbers = evidence.number_format
        var_source, var, var_json = self._resolve_axis(
            working.var, present, self._var_synthesized(), numbers
        )
        layers = self._resolve_layers(evidence.columns, present, self._accounted(var_source))
        obs_source, obs, obs_json = self._resolve_axis(
            working.obs, present, self._obs_synthesized(), numbers
        )
        var_modes = self._var_duplicates(working.var, var_source)
        read = self._read_plan(evidence, obs_source, var_source, layers, var_modes)
        self._require_aggregatable(evidence, read, layers)
        var_values = tuple(
            feature_value_policy(source, mode, numbers) for source, mode in var_modes.items()
        )
        decomposer, decomposition_json = self._decomposition(
            layers, obs_source, var_source, var_values, present, numbers
        )
        validator = LayerContractValidator(
            primary_layer_name=working.measurements.primary_layer_name,
            required_names=layers.required_names,
            empty_ratio=_EMPTY_RATIO,
            populated_ratio=_POPULATED_RATIO,
            strict=checks == "strict",
        )
        snapshot = resolved_plan_json(
            {
                "level": working.level,
                "number_format": numbers,
                "read": read,
                "decomposition": decomposition_json,
                "obs": obs_json,
                "var": var_json,
                "duplicate_mode": working.measurements.duplicate_mode,
                "duplicate_ranking": working.measurements.duplicate_ranking,
                "var_duplicates": var_modes,
                "layer_values": [
                    {"layer_name": layer.name, "value": layer.value} for layer in layers.retained
                ],
                "layer_contract": {
                    "primary_layer_name": validator.primary_layer_name,
                    "required_names": validator.required_names,
                    "empty_ratio": validator.empty_ratio,
                    "populated_ratio": validator.populated_ratio,
                },
            }
        )
        return ParseStrategy(
            level=working.level,
            read=read,
            decomposer=decomposer,
            obs=obs,
            var=var,
            duplicates=duplicate_policy(working.measurements),
            layer_parsers={
                layer.name: make_layer_parser(layer.name, layer.value, numbers)
                for layer in layers.retained
            },
            layer_validator=validator,
            pruner=UninformativeFields(
                retained_obs=working.retained.obs,
                retained_var=working.retained.var,
                retained_layers=working.retained.layers,
            ),
            hierarchy=working.hierarchy,
            roles=SemanticRoles(working.column_roles, self._layer_roles(layers.retained)),
            provenance={**working.provenance, PLAN_JSON_KEY: snapshot},
        )

    def _var_synthesized(self) -> tuple[str, ...]:
        """Names the fragment separator creates as real columns of the scalar-long table."""
        return self._configuration.source_layout.synthesized_var_columns()

    def _obs_synthesized(self) -> tuple[str, ...]:
        """A wide observation axis comes from header captures, not from selected columns."""
        if isinstance(self._configuration.source_layout, WideSourceLayout):
            return self._configuration.obs.final_key_columns
        return ()

    @staticmethod
    def _var_duplicates(
        declaration: WorkingAxisConfiguration, var: AxisSourcePlan
    ) -> dict[str, FeatureDuplicates]:
        """The duplicate mode of each var payload source, in payload order."""
        modes: dict[str, FeatureDuplicates] = {
            selection.source: selection.duplicates
            for selection in (*declaration.optional_selections, *declaration.required_selections)
        }
        return {source: modes[source] for source in var.payload_sources}

    def _accounted(self, var: AxisSourcePlan) -> frozenset[str]:
        """Names a permissive wide layer pattern must not mistake for a sample column."""
        declaration = self._configuration.var
        return frozenset(
            {
                *declaration.declared_order,
                *(selection.source for selection in declaration.required_selections),
                *(selection.source for selection in declaration.optional_selections),
                *var.keys.raw_key_columns,
                *var.payload_sources,
            }
        )

    # ---------------------------------------------------------------------------- axis plans

    def _resolve_axis(
        self,
        axis: WorkingAxisConfiguration,
        present: frozenset[str],
        synthesized: tuple[str, ...],
        numbers: NumericTextFormat,
    ) -> tuple[AxisSourcePlan, AxisRuntimePlan, dict[str, object]]:
        """Resolve one axis: prune what this source cannot provide, then plan both phases."""
        declaration = axis
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
            if isinstance(computer, SequenceColumn)
            and isinstance(
                computer.operation,
                TokenRegexNormalizer | SiteListNormalizer | EmbeddedSiteListNormalizer,
            )
        )
        closure = self._dependency_closure((*axis.final_key_columns, *early_outputs), computers)
        source = AxisSourcePlan(
            keys=keys,
            payload_sources=self._ordered_unique(
                selection.source
                for selection in selections
                if selection.source not in set(keys.raw_key_columns)
            ),
        )
        key_phase, key_json = self._phase(selections, computers, closure, numbers, inside=True)
        output_phase, output_json = self._phase(
            selections, computers, closure, numbers, inside=False
        )
        outputs = tuple(name for name in declaration.declared_order if name not in skipped)
        return (
            source,
            AxisRuntimePlan(keys, key_phase, output_phase, outputs),
            {
                "source": source,
                "key_phase": key_json,
                "output_phase": output_json,
                "outputs": outputs,
                "skipped": frozenset(skipped),
            },
        )

    def _materializable(
        self,
        declared: tuple[ComputedOperation, ...],
        selections: tuple[AxisColumnSelection, ...],
        synthesized: tuple[str, ...],
        skipped: set[str],
    ) -> tuple[RawSources, tuple[ComputedOperation, ...]]:
        """Walk the declarations in order, binding each name to its physical closure.

        Reading the environment before rebinding is what lets a computed column consume a
        selected column of its own name — the coalesce that widens ``Proteins`` reads the
        selected ``Proteins`` and then becomes it — without the walk chasing its own tail.
        """
        raw_sources: dict[str, tuple[str, ...]] = {name: (name,) for name in synthesized}
        for selection in selections:
            raw_sources[selection.name] = (selection.source,)
        retained: list[ComputedOperation] = []
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
        computers: tuple[ComputedOperation, ...],
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
            return self._resolve_wide_layers(columns, accounted, layout.sample_layer_name)
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
        retained = tuple(layer for layer in measurements.layers if layer.source in present)
        return _ResolvedLayers(
            retained=retained,
            required_names=tuple(layer.name for layer in retained if layer.name in required),
            long_sources=tuple(
                LongRawLayerSource(name=layer.name, source_column=layer.source)
                for layer in retained
            ),
            wide_plans=(),
            source_columns=frozenset(layer.source for layer in retained),
        )

    def _resolve_wide_layers(
        self, columns: tuple[str, ...], accounted: frozenset[str], sample_layer: str
    ) -> _ResolvedLayers:
        """Expand each layer's header regex, then align every layer to the sample layer.

        The sample layer's captures define the observation axis. A permissive pattern must not
        turn an accounted-for column into an extra sample, and a layer that matched only tokens
        outside that axis is not evidence of more observations.
        """
        measurements = self._configuration.measurements
        candidates = tuple(name for name in columns if name not in accounted)
        matches = {
            layer.name: self._match_samples(candidates, layer.source)
            for layer in measurements.layers
        }
        samples = self._ordered_unique(sample for _column, sample in matches[sample_layer])
        if not samples:
            raise IncompatibleSourceError(
                f"{self._label()} matched no observation column for its sample layer "
                f"{sample_layer!r}"
            )
        required = {layer.name for layer in measurements.required_layers}
        retained: list[WorkingMeasurementLayer] = []
        plans: list[WideRawLayerPlan] = []
        for layer in measurements.layers:
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
        )

    # -------------------------------------------------------------------- read and structure

    def _read_plan(
        self,
        evidence: SourceEvidence,
        obs: AxisSourcePlan,
        var: AxisSourcePlan,
        layers: _ResolvedLayers,
        var_modes: Mapping[str, FeatureDuplicates],
    ) -> LevelReadPlan:
        """Project exactly this level's transitive source closure, and decide every dtype.

        A measurement column is read natively only when the rule sums its values, because
        summing needs numbers and the schema already checked that those layers are plain. Every
        other measurement is text: real exports write ``-`` or ``NA`` in a column a rule calls
        numeric, which an eager numeric read cannot survive. A delimited file still converts
        the text of plain numeric layers while reading it (``converted_sources``), and keeps a
        column as text where a value needs it, so the parser sees what text would have given.

        Key and payload text is read as ``pl.Categorical``: a run path repeated on every row
        costs four bytes there. Packed and layer columns stay text, and so do the sources of
        var columns resolved by number (``min``, ``max``, ``keep_best``) and their ranking
        columns, because those cast every row.
        """
        lexical = frozenset(
            {
                *obs.keys.raw_key_columns,
                *obs.payload_sources,
                *var.keys.raw_key_columns,
                *var.payload_sources,
                *self._configuration.source_layout.packed_sources(),
            }
        )
        needed = lexical | layers.source_columns
        projected = tuple(name for name in evidence.columns if name in needed)
        reduced = self._reduced_sources(layers)
        ranked = {
            column
            for source, mode in var_modes.items()
            if mode != "keep_first"
            for column in (source, *((mode.by,) if isinstance(mode, FeatureRanking) else ()))
        }
        keyed = (
            lexical
            - frozenset(self._configuration.source_layout.packed_sources())
            - layers.source_columns
            - ranked
        )
        if isinstance(evidence, FrameSourceEvidence):
            # Parquet and prepared frames keep their own types. Only text layers that a
            # number-reducing rule must add up are read as numbers, as delimited input does.
            text = frozenset(name for name, dtype in evidence.dtypes if dtype == pl.String)
            return LevelReadPlan(
                projected_columns=projected,
                text_sources=frozenset(),
                native_numeric_sources=(reduced - lexical) & text,
                categorical_sources=keyed & text,
            )
        native = (
            frozenset()
            if evidence.number_format.thousands_marks
            # The facade already rejects number-reducing rules with non-plain numeric layers.
            else reduced - lexical
        )
        text = frozenset(projected) - native
        converted = (
            self._plain_numeric_sources(layers) & (text - lexical)
            if isinstance(evidence, DelimitedSourceEvidence)
            else frozenset[str]()
        )
        return LevelReadPlan(
            projected_columns=projected,
            text_sources=text,
            native_numeric_sources=native,
            categorical_sources=keyed & text,
            converted_sources=converted,
        )

    @staticmethod
    def _plain_numeric_sources(layers: _ResolvedLayers) -> frozenset[str]:
        """The columns of plain numeric layers without missing tokens: readable as numbers."""
        return layers.sources_of(
            layer.name
            for layer in layers.retained
            if isinstance(layer.value, PlainNumericLayerDeclaration)
            and not layer.value.missing_tokens
        )

    def _decomposition(
        self,
        layers: _ResolvedLayers,
        obs: AxisSourcePlan,
        var: AxisSourcePlan,
        var_values: tuple[FeatureValuePolicy, ...],
        present: frozenset[str],
        numbers: NumericTextFormat,
    ) -> tuple[SourceDecomposer, dict[str, object]]:
        """Construct the physical-shape strategy and record its source decisions."""
        layout = self._configuration.source_layout
        primary = self._configuration.measurements.primary_layer_name
        if isinstance(layout, WideSourceLayout):
            return WideSourceDecomposer(primary, layers.wide_plans, obs, var, var_values), {
                "kind": "wide",
                "primary_layer_name": primary,
                "layer_plans": layers.wide_plans,
            }
        long = LongSourceDecomposer(primary, layers.long_sources, obs, var, var_values)
        long_json: dict[str, object] = {
            "kind": "long",
            "primary_layer_name": primary,
            "layer_sources": layers.long_sources,
        }
        if isinstance(layout, LongSourceLayout):
            return long, long_json
        separator, separator_json = self._separator(layout, layers, present, numbers)
        return DelimitedFragmentSourceDecomposer(separator, long), {
            "kind": "delimited_fragment",
            "separator": separator_json,
            "long": long_json,
        }

    def _separator(
        self,
        layout: PositionalFragmentLayout | ColumnLabeledFragmentLayout,
        layers: _ResolvedLayers,
        present: frozenset[str],
        numbers: NumericTextFormat,
    ) -> tuple[FragmentTableSeparator, dict[str, object]]:
        """Keep the retained packed sources in authored order; drop what is absent.

        A packed column of a plain numeric layer is read as numbers inside its lists
        (``converted_sources``), as a delimited reader converts such a scalar column.
        """
        packed = tuple(
            column for column in layout.packed_value_sources if column in layers.source_columns
        )
        if not packed:
            raise IncompatibleSourceError(
                f"{self._label()} carries none of the packed fragment columns "
                f"{list(layout.packed_value_sources)}"
            )
        converted = self._plain_numeric_sources(layers) & frozenset(packed)
        # An optional label column the source lacks leaves the scalars labelled by position.
        if isinstance(layout, ColumnLabeledFragmentLayout) and layout.label_source in present:
            return ColumnLabeledFragmentTableSeparator(
                layout.label_source,
                layout.label_output,
                layout.delimiter,
                packed,
                converted,
                numbers,
            ), {
                "kind": "column",
                "label_source": layout.label_source,
                "label_output": layout.label_output,
                "delimiter": layout.delimiter,
                "packed_value_sources": packed,
                "converted_sources": sorted(converted),
            }
        return PositionalFragmentTableSeparator(
            layout.label_output, layout.delimiter, packed, converted, numbers
        ), {
            "kind": "positional",
            "label_output": layout.label_output,
            "delimiter": layout.delimiter,
            "packed_value_sources": packed,
            "converted_sources": sorted(converted),
        }

    def _require_aggregatable(
        self, evidence: SourceEvidence, read: LevelReadPlan, layers: _ResolvedLayers
    ) -> None:
        """Reject a number-reducing rule whose values this source cannot deliver as numbers.

        Checked here rather than at runtime because it is a property of the rule and the
        source together, and the alternative is discovering it after reading a large table.
        """
        mode = self._configuration.measurements.duplicate_mode
        reduced = self._reduced_sources(layers)
        if not reduced:
            return
        if isinstance(evidence, FrameSourceEvidence):
            numeric = frozenset(name for name, dtype in evidence.dtypes if dtype.is_numeric())
            offenders = sorted(reduced - numeric - read.native_numeric_sources)
        else:
            offenders = sorted(reduced - read.native_numeric_sources)
        if offenders:
            raise IncompatibleSourceError(
                f"{self._label()} takes the {mode} of duplicate cells, which requires native numeric "
                f"layer values; these resolve to text: {offenders}"
            )

    def _reduced_sources(self, layers: _ResolvedLayers) -> frozenset[str]:
        """The layer sources whose repeated cells are added up, so they must be numbers.

        Every layer's under ``sum`` and ``max``; under ``keep_best``, only the layers it sums.
        """
        measurements = self._configuration.measurements
        if measurements.duplicate_mode in NUMERIC_DUPLICATE_MODES:
            return layers.source_columns
        ranking = measurements.duplicate_ranking
        return layers.sources_of(ranking.summed if ranking is not None else ())

    def _label(self) -> str:
        """How this level names itself in an error message."""
        working = self._configuration
        return f"{working.software_name!r} level {working.level!r}"

    @staticmethod
    def _layer_roles(layers: Sequence[WorkingMeasurementLayer]) -> dict[str, tuple[str, ...]]:
        roles = sorted({role for layer in layers for role in layer.roles})
        return {
            role: tuple(layer.name for layer in layers if role in layer.roles) for role in roles
        }

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
        computer: ComputedOperation, available: RawSources
    ) -> ComputedOperation | None:
        """Drop the inputs this source cannot provide, or report the computation as blocked.

        Only the two combining operations survive a missing input: coalescing or joining the
        columns that are present is the operation the rule asked for. A marker keeps what is
        present and, with nothing present, marks no row, so it is never missing. Everything
        else needs every input it declared.
        """
        if isinstance(computer, MarkerColumn):
            kept = tuple(name for name in computer.inputs if name in available)
            return computer if kept == computer.inputs else replace(computer, inputs=kept)
        if isinstance(computer, CoalesceColumn | JoinNonemptyColumn):
            kept = tuple(name for name in computer.inputs if name in available)
            if not kept:
                return None
            return computer if kept == computer.inputs else replace(computer, inputs=kept)
        if any(name not in available for name in computer.inputs):
            return None
        return computer

    @staticmethod
    def _dependency_closure(
        final_keys: tuple[str, ...], computers: tuple[ComputedOperation, ...]
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
        computers: tuple[ComputedOperation, ...],
        closure: frozenset[str],
        numbers: NumericTextFormat,
        *,
        inside: bool,
    ) -> tuple[AxisPhaseRuntimePlan, dict[str, object]]:
        """Compile each phase once; declarations survive only in the JSON snapshot."""
        selections = tuple(s for s in selections if (s.name in closure) is inside)
        computers = tuple(c for c in computers if (c.name in closure) is inside)
        return AxisPhaseRuntimePlan(
            selections=tuple(
                SelectedAxisColumn(
                    s.name, s.source, make_axis_coercer(s.logical_type, numbers, s.separator)
                )
                for s in selections
            ),
            computers=computers,
        ), {"selections": selections, "computers": computers}
