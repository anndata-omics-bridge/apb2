"""Persist one backend-neutral parsed result as one AnnData file."""

from __future__ import annotations

from pathlib import Path

from anndata import AnnData

from apb2.output.namespace import write_namespace
from apb2.result import ParsedData


def as_anndata(parsed: ParsedData) -> AnnData:
    """Construct the in-memory AnnData for one parsed result.

    A pure function of ``parsed``: the complete provenance already rides in
    ``parsed.uns`` and is stored under the APB namespace. Composition roots that must
    attach further provenance (selection method, search parameters) do so on the
    returned object before persisting it.
    """
    adata = AnnData(
        X=parsed.X,
        obs=parsed.obs,
        var=parsed.var,
        layers=dict(parsed.layers),
    )
    write_namespace(adata, dict(parsed.uns))
    return adata


def to_anndata(parsed: ParsedData, target: Path) -> None:
    """Write ``parsed`` to ``target`` atomically as ``.h5ad``."""
    adata = as_anndata(parsed)
    scratch = target.with_name(target.name + ".tmp")
    adata.write_h5ad(scratch)
    scratch.replace(target)
