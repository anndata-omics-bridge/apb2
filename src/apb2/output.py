"""Persist one backend-neutral parsed result as AnnData, under the apb2 uns namespace.

The namespace key stays ``anndata_proteomics`` while legacy apb is the parity oracle:
the same fixture must produce the same ``.h5ad``, key for key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from anndata import AnnData

from apb2.result import ParsedData
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


def as_anndata(parsed: ParsedData) -> AnnData:
    """Construct the in-memory AnnData for one parsed result.

    A pure function of ``parsed``: the complete provenance already rides in
    ``parsed.uns`` and is stored under the apb2 namespace. Composition roots that must
    attach further provenance (selection method, search parameters) do so on the
    returned object before persisting it.
    """
    adata = AnnData(X=parsed.X, obs=parsed.obs, var=parsed.var, layers=dict(parsed.layers))
    write_namespace(adata, dict(parsed.uns))
    return adata


def to_anndata(parsed: ParsedData, target: Path) -> None:
    """Write ``parsed`` to ``target`` atomically as ``.h5ad``."""
    write_atomically(target, as_anndata(parsed).write_h5ad)


def write_atomically(target: Path, writer: Callable[[Path], None]) -> None:
    """Write beside the destination and replace it only after a complete write.

    The scratch directory cleans itself up on failure, so an interrupted write never
    leaves a partial file beside the target.
    """
    with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as folder:
        scratch = Path(folder) / target.name
        writer(scratch)
        scratch.replace(target)
