# APB metadata specification

**Status: normative APB2 result contract, 2026-09-18.** This document owns metadata placement, composition and reconstruction across result formats. Tool packages own their payload schemas and calculations; Studio is a consumer, not the authority. “Must” and “must not” describe the contract; the enforcement section distinguishes implemented checks from remaining gaps.

## Ownership

The containing object establishes scope. Physical `uns["apb"]` must contain tool namespaces directly, without `shared` or `level` wrappers.

| Namespace | MuData root | Embedded AnnData |
|---|---|---|
| `parse` | Input and search-parameter provenance | Rule, resolved plan, parsing diagnostics |
| `roles` | Absent | `columns` and `layers` |
| `fasta` | `provenance[operation]` | Operation-specific validation summaries |
| `proteobench` | `provenance.annotation`, `provenance.scoring` | `annotation`, `scoring[quantity_name]` |

Common provenance must be stored once, not repeated in modalities. Source/configuration provenance and per-level outcomes are different information, not duplicate summaries. Collection annotation tables and feature relations retain their own metadata and ownership; they must not be copied into each level.

In memory, `ParsedLevels.uns` and `ParsedLevel.uns` remain root and local parse provenance. Their respective `metadata` dictionaries carry extensions; semantic roles are projected from local parse provenance to `apb.roles`. These public types are unchanged. Physical layout does not require changing computational data structures.

When retained rules already supply software, quantification-level and rule-schema identity, parse metadata must not repeat those fields. Dimensions come from the actual structures; semantic roles have one persisted home under `roles`.

## Standalone AnnData

H5AD must contain exactly one level and combine both owners into that AnnData's namespace:

```text
adata.uns["apb"]
├── parse
├── roles
├── fasta
│   ├── provenance
│   └── peptide_verification
├── proteobench
│   ├── provenance
│   │   ├── annotation
│   │   └── scoring
│   ├── annotation
│   └── scoring
│       └── Intensity
└── storage
```

Composition must recursively combine disjoint mappings without recognizing tool names. Overlapping mapping containers are permitted; overlapping leaves or mapping-versus-leaf conflicts must fail before publication, even when values compare equal. There is no precedence or silent overwrite.

The technical `storage` descriptor must record ownership paths, never copies of scientific values. `metadata_ownership` has `root` and `level` records, each containing `values` and `empty_objects` arrays of key-segment paths. Segment arrays preserve literal slashes in keys. Empty mappings, empty roots, nulls and unrelated extensions must round-trip; reads must reject incomplete, duplicated or invented ownership paths. These descriptor owner names are not scientific namespace wrappers.

Selecting one modality through APB2's storage-neutral result and writing H5AD must retain common provenance and reconstruct both original owners on read. Writing an extracted modality directly with third-party AnnData does not provide that guarantee.

## Tool operations

Generic annotation must store one source descriptor and operation configuration at root `<tool>.provenance.annotation`; each affected level stores matching counts, corrections and added-column names at `<tool>.annotation`. It must not add a parallel `annotation.<tool>` tree, repeated convention/source aliases or an extra metadata envelope. ProteoBench and prolfquapp use the same generic mechanism.

FASTA provenance contains database identity, sources and operation settings; validation summaries remain local. Sample annotations remain in `obs`; feature-aligned FASTA and ProteoBench tables remain in `varm`, referenced from metadata rather than copied there.

ProteoBench scoring must preserve existing annotation. Common versions, methods and selection mode belong in root scoring provenance; selected quantities derive from scoring keys. Each local scoring entry retains scores, roles, protein-mapping diagnostics and its aligned-table reference. Use `scoring.Intensity`, not `scoring.layers.Intensity` or an `X` alias. Logical names retain reversible escaping where required: `LFQ/Intensity` uses `LFQ%2FIntensity`, with the original `layer_name` retained. The tool must refuse existing scores rather than treating annotation alone as an overwrite conflict.

## Storage and representation

The primary quantity's only physical matrix must be `X`; `adata.layers` contains additional matrices. Its original logical name remains in the descriptor and is displayed as, for example, `Intensity · X`. Roles are explicit semantic metadata, not instructions to choose another primary matrix.

Parquet and DuckDB manifests must use collection `apb` and per-level `apb` records. Unknown extension metadata must survive storage; APB2 must not import FASTA or ProteoBench implementations to interpret it. `parse`, `roles` and `storage` are APB2-owned; collection `annotation_tables` and `feature_relations` are reserved. Retired `shared`/`level` envelopes are not extension slots.

Representation must reuse persistence projections. Collection documents expose `root.apb` and local `levels[].apb`; H5AD exposes `root: null` and the combined namespace once. No `shared` field or displayed storage value is permitted. The sidecar is a derived, bounded view, never part of `uns`; see [its lifecycle and redaction rules](result_io.md#compact-json-representation). Consumers must not independently merge metadata trees.

## Versions

| Persisted component | Current version |
|---|---|
| HDF5 storage descriptor | `3` |
| Parquet manifest | `4` |
| DuckDB manifest | `3` |
| Representation document | `4` |

Readers and writers must change together and reject unsupported older layouts explicitly. No aliases or automatic migrations are provided. Tool payload versions remain independently owned; a parsing-rule schema version is not a result-storage version. Historical artifacts must not be rewritten as a side effect of reading or viewing them.

## Enforcement

Already enforced: reserved writer sections, conflict rejection, ownership-path reconstruction, format versions, primary-only physical storage, aligned-frame invariants and cross-format regression tests. [Ownership tests](../tests/parserV2/io/test_metadata_ownership.py), [format tests](../tests/parserV2/io/test_formats.py) and [representation tests](../tests/parserV2/io/test_json_representation.py) run through `make check`. Scientific payload and overwrite tests belong to the producing tools.

Remaining hardening: generic extension writers still accept retired `shared`/`level` keys, and root readers can accept a misplaced `roles` section in a current-version envelope. Add symmetric envelope validation at APB2's read/write boundary, with malformed-current-version tests across all backends. Reject only structural violations; preserve unknown extensions. Do not add tool-specific payload validation or another independently maintained schema in Studio.
