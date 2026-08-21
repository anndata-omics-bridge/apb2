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

from dataclasses import dataclass

import polars as pl

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


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


@dataclass(slots=True)
class FinalLayerTable:
    """One measurement aligned to the final axes, its raw scalars still unencoded."""

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
