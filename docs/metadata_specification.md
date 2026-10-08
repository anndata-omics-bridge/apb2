# APB metadata specification

**Status: normative APB2 result contract, 8 October 2026.** This document owns metadata placement, the record shape every producer shares, and persisted versions. Tool packages own their payloads and calculations; Studio is a consumer, not the authority. “Must” and “must not” describe the contract.

## Placement: a root part and level parts, never merged

Every result has one root part and one part per level, in every format:

| Format | Root part | Level part |
|---|---|---|
| h5mu | `mdata.uns["apb"]` | `mdata.mod["ion"].uns["apb"]` |
| h5ad | `adata.uns["apb"]` | `adata.uns["ion"]["apb"]` |
| Parquet, DuckDB | collection `apb` | per-level `apb` |
| JSON sidecar | `root.apb` | `levels[].apb` |

Parts must never be merged, so one key may appear in both. In memory the root part is `parse` (from `ParsedLevels.uns`) plus `ParsedLevels.metadata` and the typed `hierarchy`; a level part is `parse` (from `ParsedLevel.uns`), `roles` (from the typed column and layer roles) and `ParsedLevel.metadata`. Storage copies records unchanged; it never regroups or computes them.

APB2 owns `parse`, `roles`, `hierarchy` and `storage`, and an h5mu root's `annotation_tables` and `feature_relations`; extensions must not use these names. Every `uns["apb"]` part carries its own `storage` descriptor. An h5ad root descriptor names its single level and the `uns` key holding that level's part; an h5ad holds exactly one level and no annotation tables or feature relations.

## Records

A producer's record sits at `apb.<producer>`, or one key deeper when the producer keeps several named records, as Catalog does at `apb.catalog.<catalogue>`. Its fields, each optional, are:

- `schema_version`: text versioning the producer's payload
- `provenance`: versions, inputs and configuration
- `result`: the complete output
- `summary`: display metrics computed from that output
- `details`: references to the tables holding per-feature evidence

A record holds only the fields it uses, in whichever part needs them. Producers compute summaries from completed calculations; Studio displays them without recalculating them or reading logs.

### Summary entries

Each entry is `{name, label, value, unit, status}` plus an optional `layer` naming the logical quantity. Identity is the record's path plus `name` and `layer`; a repeated identity is rejected.

- `status` is `ok`, `attention` or `not_checked`
- `value` is a JSON scalar or `null`; `not_checked` must carry `null`
- `null` with `ok` or `attention` means computed but undefined
- Producers store NaN and ±infinity as `null`; APB2 rejects them in summary values
- A checked zero is an explicit `0`; a count of problems is `attention` above zero

### Detail references

`details` lists `{slot, name}` pairs naming a level's `varm` table or `layers` entry, or a collection's `annotation_tables` or `feature_relations` entry, by logical name. References are never pruned, so a copied record may name a table that its container does not hold.

## Producers

| Key | Writer | Root record | Level record |
|---|---|---|---|
| `parse` | apb2 conversion | `result`: observation keys and relationships | `provenance` (`rule_json`, `plan_json`, input preparation, sample matching), `result` (unknown tokens, layer diagnostics), `summary` |
| `prolfquapp`, `sdrf` | apb2 annotation | `schema_version`, `provenance.annotation` | `result.annotation`, `summary` |
| `fasta` | apb-fasta | `schema_version`, `provenance` per operation | `result`, `summary`, `details` |
| `aggregate` | apb-aggregate | none | `schema_version`, `provenance.lineage`, `summary`, `details` |
| `proteobench` | apb-proteobench | `schema_version`, `provenance` | `result`, `summary`, `details` |
| `catalog.<catalogue>` | apb-catalog | `result`: the resolution snapshot | `summary` |
| `export` | apb-export | `schema_version`, `provenance` | `provenance`, `summary` |

Generic annotation uses one mechanism for prolfquapp, SDRF and ProteoBench sample tables; the convention names the record. Sample annotations remain in `obs` and feature-aligned tables in `varm`; records reference them through `details` instead of copying them. Exports copy every upstream record unchanged and add their own `export` record, encoded with `UnsJsonCodec` in the same placement; a standalone export keys its level part by the name a MuData export would give that modality.

## Storage and representation

The primary quantity's only physical matrix must be `X`; `adata.layers` contains additional matrices. Its original logical name remains in the descriptor and is displayed as, for example, `Intensity · X`. Roles are explicit semantic metadata, not instructions to choose another primary matrix.

AnnData stores each `uns` list as one NumPy array and each key as an HDF5 link name, so h5ad and h5mu must store a value as JSON text when AnnData cannot hold it exactly. Such values are lists that are not empty and not one scalar type, integers outside 64 bits, text containing NUL, and mappings with a key that is empty, `.`, or contains `/` or NUL. The storage descriptor's `json_values` lists their key-segment paths, and readers decode exactly those paths. Every other mapping stays a native `uns` group. A top-level section name HDF5 cannot link is rejected. All `uns["apb"]` writes go through `UnsJsonCodec`.

Unknown extension metadata must survive storage; APB2 must not import producer packages to interpret it. Representation must reuse persistence projections: every document exposes `root.apb` and `levels[].apb`. The sidecar is a derived, bounded view, never part of `uns`; see [its lifecycle and redaction rules](result_io.md#compact-json-representation). Consumers must not merge the parts.

## Versions

| Persisted component | Current version |
|---|---|
| HDF5 storage descriptor | `6` |
| Parquet manifest | `7` |
| DuckDB manifest | `6` |
| Representation document | `5` |

Readers and writers must change together and reject unsupported older layouts explicitly. No aliases or automatic migrations are provided. Producer payload versions remain independently owned; a parsing-rule schema version is not a result-storage version. Historical artifacts must not be rewritten as a side effect of reading or viewing them.

## Enforcement

APB2 checks every record's `summary` and `details` when it writes or reads a result, without importing producers ([records.py](../src/apb2/parserV2/parse_quant/io/records.py)). [Record tests](../tests/parserV2/io/test_records.py), [format tests](../tests/parserV2/io/test_formats.py) and [representation tests](../tests/parserV2/io/test_json_representation.py) run through `make check`. Scientific payload and overwrite tests belong to the producing tools.
