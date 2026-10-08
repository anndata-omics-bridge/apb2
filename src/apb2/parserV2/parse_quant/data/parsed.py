"""The result of one parse: final axes, canonical layers, and their composition.

Layer values have already been converted to their declared numeric or categorical meaning.
Writers only map these storage-neutral values into a physical backend, which is why this
module knows nothing about matrices, pandas indexes, or AnnData.

``JsonScalar`` and ``JsonValue`` are declared here rather than imported: provenance crosses
this boundary as data, and a shared parent module holding the alias would force this child
to import upward. The identical shape in ``parameters/level.py`` is the input side of the
same value, not duplicated behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import observation_labels

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type ParsedLevelName = str
type NumericLayerType = Literal["number", "integer"]


@dataclass(slots=True)
class ObsFinal:
    """The public observation axis: authored keys plus retained output metadata."""

    frame: pl.DataFrame
    # pl.DataFrame({"sample": ["A", "B", "C"]})

    key_columns: tuple[str, ...]
    # ("sample",)


@dataclass(slots=True)
class VarFinal:
    """The public variable axis: authored keys, retained metadata, and column roles."""

    frame: pl.DataFrame
    # pl.DataFrame({
    #     "ProForma_ion": ["PEPM[UNIMOD:35]IDE/2", "OTHER/3"],
    #     "genes": ["GENE1", "GENE2"],
    # })

    key_columns: tuple[str, ...]
    # ("ProForma_ion",)

    roles: dict[str, str] = field(default_factory=dict)
    # {"protein_assignment": "Protein_Group"}


@dataclass(frozen=True, slots=True)
class LevelHierarchy:
    """Ordered level identities, expressed as var columns or semantic role names."""

    name: str
    identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = self.level_names
        if not self.name or not names or len(names) != len(set(names)):
            raise ValueError("a hierarchy requires a name and unique ordered levels")
        if any(not level or not identity for level, identity in self.identities):
            raise ValueError("hierarchy levels and identities must be nonempty")

    @property
    def level_names(self) -> tuple[str, ...]:
        """Return levels from fine to coarse."""
        return tuple(level for level, _identity in self.identities)

    def key(self, level: str, var: VarFinal) -> str:
        """Resolve one identity against this axis's typed roles."""
        identity = dict(self.identities)[level]
        return var.roles.get(identity, identity)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the self-contained persisted hierarchy."""
        return {"name": self.name, "identities": [list(pair) for pair in self.identities]}

    @classmethod
    def from_json(cls, value: JsonValue) -> LevelHierarchy:
        """Validate a persisted hierarchy without consulting packaged configurations."""
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("hierarchy must have a text name")
        pairs = value.get("identities")
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, str) for item in pair)
            for pair in pairs
        ):
            raise ValueError("hierarchy identities must be ordered pairs of strings")
        return cls(
            cast(str, value["name"]),
            tuple((pair[0], pair[1]) for pair in cast(list[list[str]], pairs)),
        )


class MeasurementLayerRole:
    """A quantitative layer that participates in matrix occupancy checks."""

    __slots__ = ()

    def occupancy_candidates(
        self,
        layer_name: str,
        encoded_values: pl.DataFrame,
        /,
    ) -> dict[str, pl.DataFrame]:
        """Contribute this measurement to the occupancy comparison."""
        return {layer_name: encoded_values}

    def accepts_primary_layer(self) -> bool:
        """A measurement may define the primary quantitative matrix."""
        return True

    def persisted_name(self) -> Literal["measurement"]:
        """Return the stable storage name for this role."""
        return "measurement"


class AuxiliaryLayerRole:
    """A numeric diagnostic layer that is exempt from matrix occupancy checks."""

    __slots__ = ()

    def occupancy_candidates(
        self,
        layer_name: str,
        encoded_values: pl.DataFrame,
        /,
    ) -> dict[str, pl.DataFrame]:
        """Exclude this auxiliary matrix from the occupancy comparison."""
        del layer_name, encoded_values
        return {}

    def accepts_primary_layer(self) -> bool:
        """An auxiliary matrix cannot define the primary quantitative matrix."""
        return False

    def persisted_name(self) -> Literal["auxiliary"]:
        """Return the stable storage name for this role."""
        return "auxiliary"


type FinalLayerRole = MeasurementLayerRole | AuxiliaryLayerRole


@dataclass(frozen=True, slots=True)
class QuantitativeLayerSemantics:
    """A canonical numeric layer and its declared scientific number type."""

    logical_type: NumericLayerType = "number"

    def numeric(self, layer_name: str, values: pl.DataFrame) -> pl.DataFrame:
        """Numbers are already the layer's values."""
        del layer_name
        return values

    def decode(self, values: pl.DataFrame) -> pl.DataFrame:
        """Numbers need no decoding."""
        return values


@dataclass(frozen=True, slots=True)
class CategoricalLayerSemantics:
    """Canonical category codes together with their stable label mapping."""

    categories: tuple[tuple[str, int], ...]
    missing_code: int = -1

    def __post_init__(self) -> None:
        if self.missing_code in {code for _label, code in self.categories}:
            raise ValueError(f"categorical missing code {self.missing_code} is reserved")

    def numeric(self, layer_name: str, values: pl.DataFrame) -> pl.DataFrame:
        """Category codes are not numbers."""
        del values
        raise ValueError(f"layer {layer_name!r} is categorical, not quantitative")

    def decode(self, values: pl.DataFrame) -> pl.DataFrame:
        """Replace codes with their labels; the missing code becomes null."""
        labels = {code: label for label, code in self.categories}
        return values.select(pl.all().replace_strict(labels, default=None, return_dtype=pl.String))


type FinalLayerSemantics = QuantitativeLayerSemantics | CategoricalLayerSemantics


@dataclass(slots=True)
class FinalLayerTable:
    """One matrix: one row per var row and one column per obs row, by position."""

    layer_name: str
    # "Intensity"

    values: pl.DataFrame
    # pl.DataFrame({
    #     "A": [100.0, 50.0],
    #     "B": [120.0, 60.0],
    #     "C": [90.0, 70.0],
    # })

    role: FinalLayerRole = field(default_factory=MeasurementLayerRole)
    # MeasurementLayerRole()

    semantic_roles: tuple[str, ...] = ()
    # ("abundance",)

    semantics: FinalLayerSemantics = field(default_factory=QuantitativeLayerSemantics)
    # QuantitativeLayerSemantics(logical_type="number")

    def quantitative_values(self) -> pl.DataFrame:
        """Return the numbers: one row per var row and one column per obs row.

        Raises:
            ValueError: The layer is categorical.
        """
        return self.semantics.numeric(self.layer_name, self.values)

    def decoded_values(self) -> pl.DataFrame:
        """Return the values with category codes replaced by their labels; numbers unchanged."""
        return self.semantics.decode(self.values)


@dataclass(slots=True)
class ParsedLevel:
    """One parsed quantification level; it introduces no identity of its own."""

    obs: ObsFinal
    # ObsFinal(frame=obs_final_frame, key_columns=("sample",))

    var: VarFinal
    # VarFinal(frame=var_final_frame, key_columns=("ProForma_ion",))

    primary_layer_name: str
    # "Intensity"

    uns: dict[str, JsonValue]
    # The parse record, as stored: {"provenance": {"rule_json": "..."}, "result": {...},
    #                               "summary": [{"name": "unreadable_cells", ...}]}

    layers: dict[str, FinalLayerTable]
    # {"Intensity": intensity_final, "QValue": q_value_final}

    obsm: dict[str, pl.DataFrame]
    # {"sample_covariates": pl.DataFrame({"batch": ["A", "B", "A"]})}

    varm: dict[str, pl.DataFrame]
    # {"protein_scores": pl.DataFrame({"score": [0.91, 0.73]})}

    obsp: dict[str, pl.DataFrame]
    # {"sample_graph": pl.DataFrame({"row": [0, 1], "column": [1, 0], "value": [0.8, 0.8]})}

    varp: dict[str, pl.DataFrame]
    # {"similarity": pl.DataFrame({"row": [0], "column": [1], "value": [0.6]})}

    metadata: dict[str, JsonValue] = field(default_factory=dict)
    # Other producers' records: {"prolfquapp": {"result": {"annotation": {...}}, "summary": []}}

    @classmethod
    def build(
        cls,
        obs: pl.DataFrame,
        obs_keys: Sequence[str],
        var: pl.DataFrame,
        var_keys: Sequence[str],
        var_roles: Mapping[str, str],
        primary_layer: str,
        abundance: Mapping[str, pl.DataFrame],
        auxiliary: Mapping[str, pl.DataFrame] | None = None,
        uns: Mapping[str, JsonValue] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        obsm: Mapping[str, pl.DataFrame] | None = None,
        varm: Mapping[str, pl.DataFrame] | None = None,
        obsp: Mapping[str, pl.DataFrame] | None = None,
        varp: Mapping[str, pl.DataFrame] | None = None,
    ) -> ParsedLevel:
        """Build a level from plain frames.

        Every layer is one frame with a row per ``var`` row and a column per ``obs`` row, by
        position; its column names are replaced. Abundance layers are measurements carrying the
        ``abundance`` role; auxiliary layers are diagnostics, integer when all their columns are.

        Raises:
            ValueError: A key or role column is absent, a layer's shape does not match the axes,
                a name is both abundance and auxiliary, or ``primary_layer`` is not abundance.
        """
        absent = (
            (set(obs_keys) - set(obs.columns))
            | (set(var_keys) - set(var.columns))
            | (set(var_roles.values()) - set(var.columns))
        )
        if absent:
            raise ValueError(f"axis columns {sorted(absent)} are absent")
        if primary_layer not in abundance:
            raise ValueError(f"primary layer {primary_layer!r} is not an abundance layer")
        labels = observation_labels(obs.height, reserved=var.columns)
        return cls(
            obs=ObsFinal(frame=obs, key_columns=tuple(obs_keys)),
            var=VarFinal(frame=var, key_columns=tuple(var_keys), roles=dict(var_roles)),
            primary_layer_name=primary_layer,
            uns=dict(uns or {}),
            layers=_new_layers(abundance, auxiliary or {}, labels, var.height),
            obsm=dict(obsm or {}),
            varm=dict(varm or {}),
            obsp=dict(obsp or {}),
            varp=dict(varp or {}),
            metadata=dict(metadata or {}),
        )

    def with_layers(
        self,
        abundance: Mapping[str, pl.DataFrame] | None = None,
        auxiliary: Mapping[str, pl.DataFrame] | None = None,
        varm: Mapping[str, pl.DataFrame] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> ParsedLevel:
        """Return a new level with layers and ``varm`` tables added and metadata sections set.

        Layers follow :meth:`build`; ``metadata`` replaces whole top-level sections.

        Raises:
            ValueError: A layer or ``varm`` name already exists, or a layer's shape does not
                match the axes.
        """
        added_varm = dict(varm or {})
        labels = tuple(self.layers[self.primary_layer_name].values.columns)
        added = _new_layers(abundance or {}, auxiliary or {}, labels, self.var.frame.height)
        clashes = (set(added) & set(self.layers)) | (set(added_varm) & set(self.varm))
        if clashes:
            raise ValueError(f"level already has {sorted(clashes)}")
        return replace(
            self,
            uns=dict(self.uns),
            layers={**self.layers, **added},
            obsm=dict(self.obsm),
            varm={**self.varm, **added_varm},
            obsp=dict(self.obsp),
            varp=dict(self.varp),
            metadata={**self.metadata, **(metadata or {})},
        )

    def abundance_layers(self, names: Sequence[str] | None = None) -> tuple[str, ...]:
        """Return every abundance layer in authored order, or ``names`` validated as abundance.

        Raises:
            ValueError: Nothing is selected, a name is absent, or a layer is not a measurement
                carrying the ``abundance`` role.
        """
        selected = (
            tuple(
                name for name, layer in self.layers.items() if "abundance" in layer.semantic_roles
            )
            if names is None
            else tuple(names)
        )
        if not selected:
            raise ValueError("level has no layer carrying the abundance role")
        for name in selected:
            if name not in self.layers:
                raise ValueError(f"level has no layer {name!r}")
            layer = self.layers[name]
            if "abundance" not in layer.semantic_roles or not layer.role.accepts_primary_layer():
                raise ValueError(f"layer {name!r} does not carry the abundance role")
        return selected


def _new_layers(
    abundance: Mapping[str, pl.DataFrame],
    auxiliary: Mapping[str, pl.DataFrame],
    labels: tuple[str, ...],
    rows: int,
) -> dict[str, FinalLayerTable]:
    """Measurement layers with the abundance role, then integer-or-number diagnostics."""
    both = set(abundance) & set(auxiliary)
    if both:
        raise ValueError(f"layers {sorted(both)} are both abundance and auxiliary")
    layers = {
        name: FinalLayerTable(
            name, _positional(name, values, labels, rows), semantic_roles=("abundance",)
        )
        for name, values in abundance.items()
    }
    for name, values in auxiliary.items():
        frame = _positional(name, values, labels, rows)
        integer = all(dtype.is_integer() for dtype in frame.dtypes)
        layers[name] = FinalLayerTable(
            name,
            frame,
            role=AuxiliaryLayerRole(),
            semantics=QuantitativeLayerSemantics("integer" if integer else "number"),
        )
    return layers


def _positional(
    name: str, values: pl.DataFrame, labels: tuple[str, ...], rows: int
) -> pl.DataFrame:
    """Name a layer's value columns by observation position after checking its shape."""
    if values.shape != (rows, len(labels)):
        raise ValueError(
            f"layer {name!r} has shape {values.shape}; the axes need {(rows, len(labels))}"
        )
    return values.select(pl.nth(index).alias(label) for index, label in enumerate(labels))


@dataclass(slots=True)
class AnnotationTable:
    """A keyed feature table without a quantitative observation matrix."""

    frame: pl.DataFrame
    key_columns: tuple[str, ...]
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureRelation:
    """Directed coordinates from an annotation table to a level's variable axis."""

    annotation_table: str
    target_level: ParsedLevelName
    coordinates: pl.DataFrame
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedLevels:
    """One or more parsed levels on one shared observation axis."""

    levels: dict[ParsedLevelName, ParsedLevel]
    # {"ion": ion_parsed_level, "protein": protein_parsed_level}

    uns: dict[str, JsonValue]
    # Empty after vendor parsing; later collection-level operations may add provenance.

    metadata: dict[str, JsonValue] = field(default_factory=dict)
    # Post-parse sections persisted beside, never inside, APB's parse provenance.

    annotation_tables: dict[str, AnnotationTable] = field(default_factory=dict)
    # {"protein_group_members": AnnotationTable(...)}

    feature_relations: dict[str, FeatureRelation] = field(default_factory=dict)
    # {"protein_group_membership": FeatureRelation(...)}

    hierarchy: LevelHierarchy | None = None

    def with_annotation_table(
        self,
        name: str,
        frame: pl.DataFrame,
        key_columns: Sequence[str],
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> ParsedLevels:
        """Return a new result with one keyed feature table added.

        Raises:
            ValueError: The table name exists, or a key column is absent from ``frame``.
        """
        if name in self.annotation_tables:
            raise ValueError(f"result already has annotation table {name!r}")
        absent = set(key_columns) - set(frame.columns)
        if absent:
            raise ValueError(f"annotation table {name!r} lacks key columns {sorted(absent)}")
        table = AnnotationTable(
            frame=frame, key_columns=tuple(key_columns), metadata=dict(metadata or {})
        )
        return self._copy({**self.annotation_tables, name: table}, self.feature_relations)

    def with_feature_relation(
        self,
        name: str,
        table: str,
        target_level: str,
        coordinates: pl.DataFrame,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> ParsedLevels:
        """Return a new result relating one annotation table to one level's variable axis.

        Raises:
            ValueError: The relation name exists, or the table or level is absent.
        """
        if name in self.feature_relations:
            raise ValueError(f"result already has feature relation {name!r}")
        if table not in self.annotation_tables:
            raise ValueError(f"result has no annotation table {table!r}")
        if target_level not in self.levels:
            raise ValueError(f"result has no level {target_level!r}")
        relation = FeatureRelation(
            annotation_table=table,
            target_level=target_level,
            coordinates=coordinates,
            metadata=dict(metadata or {}),
        )
        return self._copy(self.annotation_tables, {**self.feature_relations, name: relation})

    def observation_groups(self) -> tuple[ParsedLevels, ...]:
        """Return independently writable groups with compatible observation identities.

        Canonical level order determines the preferred observation identity. Different key
        declarations are joined only through complete, non-null, one-to-one metadata; labels
        themselves are never guessed. Observed relationships remain parse provenance in every
        returned result, including when fractionation prevents alignment. Layer columns are
        positional, so relabelling observation keys without changing row order preserves
        every quantitative cell, including missing values.

        Raises:
            ValueError: A level has duplicate observation keys.
        """
        groups: dict[tuple[str, ...], dict[ParsedLevelName, ParsedLevel]] = {}
        for name, level in self.levels.items():
            if level.obs.frame.select(level.obs.key_columns).is_duplicated().any():
                raise ValueError(f"level {name!r} has duplicate observation keys")
            groups.setdefault(level.obs.key_columns, {})[name] = level
        if len(groups) <= 1:
            return (replace(self, levels=_shared_axis(self.levels)),)
        relationships: list[JsonValue] = []
        for keys, levels in tuple(groups.items()):
            if keys not in groups:
                continue
            for other_keys, other_levels in tuple(groups.items()):
                if keys == other_keys:
                    continue
                mapping = _relationship(levels, keys, other_keys)
                if mapping is None:
                    continue
                aligned = _align(other_levels, mapping, keys, other_keys)
                relationships.append(
                    {
                        "from_keys": list(keys),
                        "to_keys": list(other_keys),
                        "rows": cast("JsonValue", json.loads(mapping.write_json())),
                        "aligned": bool(aligned),
                    }
                )
                if aligned:
                    levels.update(aligned)
                    del groups[other_keys]
        return tuple(
            replace(
                self,
                levels=_shared_axis({name: levels[name] for name in self.levels if name in levels}),
                uns=_with_result(
                    self.uns,
                    {
                        "observation_keys": list(keys),
                        "observation_relationships": json.dumps(relationships),
                    },
                ),
                feature_relations={
                    name: relation
                    for name, relation in self.feature_relations.items()
                    if relation.target_level in levels
                },
            )
            for keys, levels in groups.items()
        )

    def _copy(
        self,
        annotation_tables: dict[str, AnnotationTable],
        feature_relations: dict[str, FeatureRelation],
    ) -> ParsedLevels:
        """A new result sharing level objects, with its own containers."""
        return replace(
            self,
            levels=dict(self.levels),
            uns=dict(self.uns),
            metadata=dict(self.metadata),
            annotation_tables=dict(annotation_tables),
            feature_relations=dict(feature_relations),
        )


def _relationship(
    levels: dict[ParsedLevelName, ParsedLevel],
    keys: tuple[str, ...],
    other_keys: tuple[str, ...],
) -> pl.DataFrame | None:
    columns = tuple(dict.fromkeys((*keys, *other_keys)))
    frames = [
        level.obs.frame.select(columns)
        for level in levels.values()
        if set(columns) <= set(level.obs.frame.columns)
    ]
    if not frames or any(frame.schema != frames[0].schema for frame in frames):
        return None
    return pl.concat(frames).unique(maintain_order=True)


def _align(
    levels: dict[ParsedLevelName, ParsedLevel],
    mapping: pl.DataFrame,
    keys: tuple[str, ...],
    other_keys: tuple[str, ...],
) -> dict[ParsedLevelName, ParsedLevel]:
    if (
        mapping.is_empty()
        or any(mapping.null_count().row(0))
        or mapping.select(keys).is_duplicated().any()
        or mapping.select(other_keys).is_duplicated().any()
    ):
        return {}
    aligned: dict[ParsedLevelName, ParsedLevel] = {}
    additions = tuple(key for key in keys if key not in other_keys)
    for name, level in levels.items():
        frame = level.obs.frame
        if any(frame.schema[key] != mapping.schema[key] for key in other_keys):
            return {}
        # Do not overwrite an already authored metadata column with a different value.
        if set(additions) & set(frame.columns):
            return {}
        joined = frame.join(
            mapping, on=other_keys, how="inner", maintain_order="left", validate="m:1"
        )
        if joined.height != frame.height:
            return {}
        aligned[name] = replace(
            level,
            obs=ObsFinal(frame=joined, key_columns=keys),
            uns=_with_result(level.uns, {"observation_keys_original": list(other_keys)}),
        )
    return aligned


def _with_result(
    record: dict[str, JsonValue], values: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """A copy of one ``parse`` record with values added to its ``result``."""
    result = record.get("result", {})
    if not isinstance(result, dict):
        raise TypeError("the parse record's result is not an object")
    return {**record, "result": {**result, **values}}


def _shared_axis(levels: dict[str, ParsedLevel]) -> dict[str, ParsedLevel]:
    """Align declared observation keys to their ordered union, retaining missing cells."""
    keys = next(iter(levels.values())).obs.key_columns
    spine = pl.concat([level.obs.frame.select(keys) for level in levels.values()]).unique(
        maintain_order=True
    )
    if all(level.obs.frame.select(keys).equals(spine) for level in levels.values()):
        return levels
    result: dict[str, ParsedLevel] = {}
    position = "__apb_observation_position"
    for name, level in levels.items():
        indexed = level.obs.frame.select(keys).with_row_index(position)
        mapping = (
            spine.join(indexed, on=keys, how="left", maintain_order="left")
            .get_column(position)
            .to_list()
        )
        layers = {}
        for layer_name, layer in level.layers.items():
            labels = observation_labels(spine.height, level.var.frame.columns)
            values = pl.DataFrame(
                [
                    layer.values.get_column(layer.values.columns[index]).rename(label)
                    if index is not None
                    else pl.Series(
                        label,
                        [None] * level.var.frame.height,
                        dtype=next(iter(layer.values.dtypes), pl.Null),
                    )
                    for label, index in zip(labels, mapping, strict=True)
                ]
            )
            layers[layer_name] = replace(layer, values=values)
        row_mapping = {old: new for new, old in enumerate(mapping) if old is not None}
        result[name] = replace(
            level,
            obs=ObsFinal(
                spine.join(level.obs.frame, on=keys, how="left", maintain_order="left"), keys
            ),
            layers=layers,
            obsm={
                slot: spine.join(
                    level.obs.frame.select(keys).hstack(frame),
                    on=keys,
                    how="left",
                    maintain_order="left",
                ).drop(keys)
                for slot, frame in level.obsm.items()
            },
            obsp={
                slot: frame.with_columns(pl.col("row", "column").replace_strict(row_mapping))
                for slot, frame in level.obsp.items()
            },
        )
    return result
