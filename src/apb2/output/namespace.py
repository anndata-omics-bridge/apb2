"""Typed access to the APB-owned AnnData ``uns`` namespace.

The namespace key stays ``anndata_proteomics`` while legacy apb is the parity oracle:
the same fixture must produce the same ``.h5ad``, key for key.
"""

from __future__ import annotations

from collections.abc import Mapping

from anndata import AnnData

from apb2.serialization import JsonObject, JsonValue, to_json_compatible

NAMESPACE = "anndata_proteomics"


def read_namespace(target: AnnData) -> JsonObject:
    """Read and decode the complete namespace as a recursive JSON object."""
    return parse_namespace(target.uns.get(NAMESPACE))


def parse_namespace(stored: object) -> JsonObject:
    """Validate and decode one raw namespace value at a storage boundary."""
    if stored is None:
        return {}
    if not isinstance(stored, Mapping):
        raise TypeError(f"uns[{NAMESPACE!r}] must be a mapping")
    for key in stored:
        if not isinstance(key, str):
            raise TypeError(f"uns[{NAMESPACE!r}] keys must be strings")
    decoded = to_json_compatible(stored)
    if not isinstance(decoded, dict):
        raise TypeError(f"uns[{NAMESPACE!r}] must decode to a JSON object")
    return decoded


def write_namespace(target: AnnData, namespace: JsonObject) -> None:
    """Replace the namespace with an already validated payload."""
    target.uns[NAMESPACE] = namespace


def update_namespace(target: AnnData, updates: Mapping[str, JsonValue]) -> None:
    """Merge keys into the namespace, leaving untouched keys in place."""
    namespace = read_namespace(target)
    namespace.update(updates)
    write_namespace(target, namespace)
