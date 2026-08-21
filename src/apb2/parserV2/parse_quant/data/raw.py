"""Decomposed physical shape: small raw axes, wide raw layers, and the temporary key map.

These are the states between one physical table and one parsed level. Identity is explicit:
every axis and every layer states the ordered raw-key columns that distinguish its rows, so
no consumer has to rediscover which columns matter. A layer's remaining columns align by
*position* with the rows of the corresponding raw obs frame — their names are collision-free
storage labels, never observation identity.

``RawToFinalKeyMap`` exists only while an axis is being prepared and a layer aligned. It is
discarded before ``ParsedLevel`` is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class ObsRaw:
    """One stable-first row per complete raw obs-key tuple."""

    frame: pl.DataFrame
    # pl.DataFrame({"sample": ["A", "B", "C"]})

    raw_key_columns: tuple[str, ...]
    # ("sample",)


@dataclass(slots=True)
class VarRaw:
    """One stable-first row per complete raw var-key tuple, with its payload metadata."""

    frame: pl.DataFrame
    # pl.DataFrame({
    #     "sequence": ["PEPMIDE", "OTHER"],
    #     "mods": ["Oxidation@M", None],
    #     "mod_sites": ["4", None],
    #     "charge": ["2", "3"],
    #     "genes": ["GENE1", "GENE2"],
    # })

    raw_key_columns: tuple[str, ...]
    # ("sequence", "mods", "mod_sites", "charge")


@dataclass(slots=True)
class RawLayerTable:
    """One measurement, wide: raw var-key columns first, then one column per raw obs row.

    Repeated raw cells survive as repeated var rows; resolving them is the duplicate
    policy's question, not the decomposer's.
    """

    layer_name: str
    # "Intensity"

    raw_var_key_columns: tuple[str, ...]
    # ("sequence", "mods", "mod_sites", "charge")

    values: pl.DataFrame
    # pl.DataFrame({
    #     "sequence": ["PEPMIDE", "PEPMIDE", "OTHER"],
    #     "mods": ["Oxidation@M", "Oxidation@M", None],
    #     "mod_sites": ["4", "4", None],
    #     "charge": ["2", "2", "3"],
    #     "A": [100.0, 110.0, 50.0],
    #     "B": [120.0, None, 60.0],
    #     "C": [None, 90.0, 70.0],
    # })


@dataclass(slots=True)
class LayersRaw:
    """Every retained raw layer in authored order, and which of them is primary."""

    primary_layer_name: str
    # "Intensity"

    values: tuple[RawLayerTable, ...]
    # (intensity_raw, q_value_raw)


@dataclass(slots=True)
class DecomposedDataRaw:
    """What every physical shape reduces to: two small raw axes and wide raw layers."""

    obs: ObsRaw
    # ObsRaw(frame=obs_frame, raw_key_columns=("sample",))

    var: VarRaw
    # VarRaw(
    #     frame=var_frame,
    #     raw_key_columns=("sequence", "mods", "mod_sites", "charge"),
    # )

    layers: LayersRaw
    # LayersRaw(
    #     primary_layer_name="Intensity",
    #     values=(intensity_raw, q_value_raw),
    # )


@dataclass(slots=True)
class RawToFinalKeyMap:
    """One temporary row relation: equal-length raw-key and final-key frames."""

    raw_keys: pl.DataFrame
    # pl.DataFrame({
    #     "sequence": ["PEPMIDE", "OTHER"],
    #     "mods": ["Oxidation@M", None],
    #     "mod_sites": ["4", None],
    #     "charge": ["2", "3"],
    # })

    final_keys: pl.DataFrame
    # pl.DataFrame({
    #     "ProForma_ion": ["PEPM[UNIMOD:35]IDE/2", "OTHER/3"],
    # })
