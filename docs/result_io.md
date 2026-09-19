# Read and write parsed results

APB2 result I/O operates on one storage-neutral `ParsedLevels` value. Every format crossing reads
that value and passes it to another writer; there is no DuckDB-to-Parquet or other backend shortcut.

The [APB metadata specification](metadata_specification.md) is the normative contract for namespace ownership, H5AD composition, tool operations and persisted versions. This page describes result-I/O usage and physical fidelity.

## Reformat from the CLI

```bash
apb2 reformat SOURCE TARGET
```

Both formats are inferred from their suffixes:

```bash
apb2 reformat results.h5mu results.parquet
apb2 reformat results.parquet results.duckdb
apb2 reformat results.duckdb results.h5mu
```

The command performs storage conversion only. It does not load rules, read vendor parameter files,
parse vendor tables, annotate FASTA data, or run protein inference.

## Explicit Python API

Format selection is explicit in the primary API:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.io.formats import ResultFormat, reader_for, writer_for

source = Path("results.parquet")
target = Path("results.duckdb")

parsed = reader_for(ResultFormat.PARQUET).read(source)
writer_for(ResultFormat.DUCKDB).write(parsed, target)
```

`read()` returns `ParsedLevels`. `write()` persists the supplied value and returns `None`.

Low-level adapters write only their physical format. The path-inferred public writer below composes physical persistence with the compact JSON representation sidecar.

## Path-inferred conveniences

Use the convenience functions when the paths already carry the format:

```python
from pathlib import Path

from apb2.result_facade import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("results.duckdb"))
write_parsed_levels(parsed, Path("results.h5mu"))
```

The programmatic equivalent of the CLI command is:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.io.formats import reformat

reformat(Path("results.parquet"), Path("results.duckdb"))
```

Both `write_parsed_levels()` and `reformat()` automatically publish an adjacent `<artifact>.apb.json` document after the scientific artifact succeeds. The `apb2 convert` and `apb2 annotate` workflows use the same sidecar lifecycle.

## Compact JSON representation

The sidecar is a versioned scientific view with `format: "apb2-result-representation"` and `format_version: "4"`. It is derived from `ParsedLevels`, so h5ad, h5mu, Parquet, and DuckDB results expose the same semantic sections. `project_result(parsed)` also produces the document before a physical artifact exists; its `artifact` member is then `null`.

It records the artifact basename, physical format and byte size; root and per-level tool metadata; axis schemas, key columns and null counts; structural layer roles, semantic column and layer roles, primary status and shapes; aligned-slot schemas; annotation-table schemas; and feature-relation structure. Collection formats expose `root.apb` and each level's `apb`. H5AD exposes `root: null` and its combined metadata once on the single level. Persistence and representation reuse the same projection helpers; there is no `shared` field. Optional layer `unit` and `scale` values come from `ParsedLevel.metadata["layer_descriptors"][<layer>]` when a producer records them.

Quantitative layers expose both logical `type` (`number` or `integer`) and physical matrix `dtype`, plus exact columnwise counts, moments, extrema and at most 100 per-observation box summaries. An integer layer may have `dtype: "Float64"` because missing matrix cells use NaN. Whole-layer quartiles use linear interpolation over either all finite values or a deterministic grid sample capped at 100,000 finite cells; nulls and nonfinite cells consume none of that budget. The method, emitted sample count and limit are explicit. NaN, positive infinity and negative infinity are counted separately and excluded. Categorical layers contain only their declared category count and fixed-size known versus missing-or-unknown counts; category codes never receive numerical statistics or box summaries.

The representation never contains matrix cells, complete observation or variable rows, distinct/category values, absolute filesystem paths in path-bearing provenance fields, or generated timestamps. At most 100 observation identities appear once in each level and quantitative summaries refer to them by index. Every bounded collection records total, emitted and truncated status. Absolute values stored under `path`, `paths`, `*_path`, or `*_paths` fields are reduced to their basename; slash-prefixed separators and patterns in other fields retain their exact meaning.

Provenance containers that AnnData persistence stores as JSON text (`rule_json`, `plan_json`, and aggregation metadata) are projected back into JSON objects or arrays. Invalid and scalar JSON text remains text, and the scientific artifact's stored `.uns` and extension metadata are not changed. Typed vendor parameters are application inputs and are never embedded into parsed provenance.

Sidecar publication invalidates any previous sidecar after the new scientific artifact succeeds, then uses a temporary file plus atomic replacement. If projection or publication fails, the producing call fails, the newly valid scientific artifact remains, and no stale sidecar can describe it.

## Format contracts

### Parquet

An APB2 Parquet result is a directory ending in `.parquet`, not one Parquet file. Its manifest
records levels, ordered logical names, key columns, Polars schemas, aligned and pairwise values,
provenance, and each layer's structural role. The reader deliberately rejects an ordinary vendor Parquet file.

### DuckDB

One `.duckdb` file contains generated physical tables and a versioned APB2 manifest. Logical names
and structural layer roles are metadata and are never interpolated into SQL identifiers.

### h5ad and h5mu

The h5 readers accept APB2-authored objects with tool namespaces directly under `uns["apb"]`: `parse`, `roles`, and extension-owned names such as `fasta` or `proteobench`. The containing object establishes ownership. MuData holds common provenance once; each embedded AnnData holds its rules, roles and results. There are no `shared` or `level` wrappers. Readers are not general importers for arbitrary third-party AnnData or MuData; older layouts are rejected rather than adapted.

`ParsedLevels.uns` and `ParsedLevel.uns` remain parse provenance. Independent post-parse sections live in each value's `metadata` mapping. Semantic roles are projected to `roles.columns` and `roles.layers` on AnnData only. Standalone H5AD recursively combines root and level mappings; conflicting leaves, even equal ones, fail before publication. The `storage` descriptor records ownership paths, including empty mappings, without copying values, so reads reconstruct the two in-memory owners exactly. It also records physical names, ordering, schemas, keys and matrix locations. Primary data remains exclusively in `X`.

Columnar manifests use root `apb` plus per-level `apb` records. Generic annotation stores its source descriptor once at `<tool>.provenance.annotation` on the root and its matching report at `<tool>.annotation` on each affected level; APB2 has no tool-name branches. See the specification's [version table](metadata_specification.md#versions) for current versions.

An h5ad writer requires exactly one level. An h5mu writer accepts one or more levels.

### Semantic rule roles

Schema `0.4` rules may assign semantic `roles` directly to var-column and measurement-layer entries. The packaged [role policy](../src/apb2/parserV2/vendor_parse_rules/schema/role_policy.json) owns both the vocabulary and its permitted declaration owners: `fasta_accessions` and `protein_assignment` belong to `var`, while `abundance` belongs to `layer`.

Conversion projects these declarations into per-level parse provenance. `column_roles` maps each semantic role to one logical var-column name. `layer_roles` maps each semantic role to the ordered names of all retained layers carrying it; several raw, normalized, MS1, MS2, LFQ, iBAQ, or peak-based layers may therefore share `abundance`. Source resolution removes absent optional layers before publishing this map.

The compact sidecar exposes the maps under each level's `apb.roles`, and all result backends preserve them without duplicating them in parse provenance. Reformatting does not recalculate or reinterpret the semantic assignments.

Semantic roles do not select AnnData `X`, alter encoding, or participate in occupancy validation. `measurements.primary_layer` chooses `X`; the structural roles below control occupancy.

### Structural layer roles and compatibility

Every backend persists `"measurement"` or `"auxiliary"` with each layer and restores the matching `MeasurementLayerRole` or `AuxiliaryLayerRole`. Structural roles are part of the [current versioned result contract](metadata_specification.md#versions); unsupported previous versions are rejected.

Every layer retained in `ParsedLevel.layers` remains part of the persisted result and is validated,
encoded, written, and read. Required-layer name checks still inspect the complete encoded mapping.
For h5 output, only measurement layers enter the occupancy comparison. Auxiliary layers therefore
cannot make a sparse measurement look populated, and their own sparsity does not trigger an
occupancy failure. The primary layer must have the measurement role.

## Fidelity

Parquet and DuckDB preserve the represented `ParsedLevels` value exactly, including:

- level, frame, column, and layer order;
- each layer's measurement or auxiliary role;
- Polars dtypes;
- null versus NaN;
- string and numeric-looking-string values;
- `obsm`, `varm`, `obsp`, and `varp`;
- root and per-level parse provenance;
- independent root and per-level extension metadata.

h5ad and h5mu intentionally apply the matrix encoding stored with the parsed result. Numeric text
becomes numeric values, configured factor strings become codes, and configured missing sentinels
become missing matrix entries. Reading, representing, rewriting, or reformatting that projected
representation preserves its quantitative-versus-factor semantics and passes stored factor codes
through without encoding them again, but it cannot recover the original layer tokens.

The logical primary layer is physically stored only in `X`; it is not duplicated in `adata.layers`. APB2 restores its logical name when reading the storage descriptor. Direct AnnData consumers therefore use `adata.X` for the primary values and `adata.layers` only for additional layers.

See the [Python API](api.md) for signatures and result types.
