"""The result of one parse: final axes, final wide layers, and their composition.

Layer values are still Polars scalars exactly as the vendor wrote them. Encoding them for a
backend is the writer's business, which is why this module knows nothing about matrices,
pandas indexes, or AnnData.

``JsonScalar`` and ``JsonValue`` are declared here rather than imported: provenance crosses
this boundary as data, and a shared parent module holding the alias would force this child
to import upward. The identical shape in ``parameters/working.py`` is the input side of the
same value, not duplicated behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type ParsedLevelName = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]

LEVEL_ORDER: tuple[ParsedLevelName, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)


@dataclass(slots=True)
class ObsFinal:
    """The public observation axis: authored keys plus retained output metadata."""

    frame: pl.DataFrame
    # pl.DataFrame({"sample": ["A", "B", "C"]})

    key_columns: tuple[str, ...]
    # ("sample",)


@dataclass(slots=True)
class VarFinal:
    """The public variable axis: authored keys plus retained output metadata."""

    frame: pl.DataFrame
    # pl.DataFrame({
    #     "ProForma_ion": ["PEPM[UNIMOD:35]IDE/2", "OTHER/3"],
    #     "genes": ["GENE1", "GENE2"],
    # })

    key_columns: tuple[str, ...]
    # ("ProForma_ion",)


@dataclass(frozen=True, slots=True)
class MeasurementLayerRole:
    """A quantitative layer that participates in matrix occupancy checks."""

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


@dataclass(frozen=True, slots=True)
class AuxiliaryLayerRole:
    """A numeric diagnostic layer that is exempt from matrix occupancy checks."""

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


@dataclass(slots=True)
class FinalLayerTable:
    """One matrix layer aligned to the final axes, its raw scalars still unencoded."""

    layer_name: str
    # "Intensity"

    var_key_columns: tuple[str, ...]
    # ("ProForma_ion",)

    values: pl.DataFrame
    # pl.DataFrame({
    #     "ProForma_ion": ["PEPM[UNIMOD:35]IDE/2", "OTHER/3"],
    #     "A": [100.0, 50.0],
    #     "B": [120.0, 60.0],
    #     "C": [90.0, 70.0],
    # })

    role: FinalLayerRole = field(default_factory=MeasurementLayerRole)
    # MeasurementLayerRole()


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
    # {"software_name": "AlphaDIA", "quantification_level": "ion"}

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


@dataclass(slots=True)
class ParsedLevels:
    """One or more parsed quantification levels and their shared provenance."""

    levels: dict[ParsedLevelName, ParsedLevel]
    # {"ion": ion_parsed_level, "protein": protein_parsed_level}

    uns: dict[str, JsonValue]
    # {"produced_by": "apb2", "rule_selection_method": "software_version"}
