# Read and write parsed results

APB2 result I/O operates on one storage-neutral `ParsedLevels` value. Every format crossing reads
that value and passes it to another writer; there is no DuckDB-to-Parquet or other backend shortcut.

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

from apb2.parserV2.parse_quant.result_io import ResultFormat, reader_for, writer_for

source = Path("results.parquet")
target = Path("results.duckdb")

parsed = reader_for(ResultFormat.PARQUET).read(source)
writer_for(ResultFormat.DUCKDB).write(parsed, target)
```

`read()` returns `ParsedLevels`. `write()` persists the supplied value and returns `None`.

## Path-inferred conveniences

Use the convenience functions when the paths already carry the format:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.result_io import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("results.duckdb"))
write_parsed_levels(parsed, Path("results.h5mu"))
```

The programmatic equivalent of the CLI command is:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.result_io import reformat

reformat(Path("results.parquet"), Path("results.duckdb"))
```

## Format contracts

### Parquet

An APB2 Parquet result is a directory ending in `.parquet`, not one Parquet file. Its manifest
records levels, ordered logical names, key columns, Polars schemas, aligned and pairwise values,
and provenance. The reader deliberately rejects an ordinary vendor Parquet file.

### DuckDB

One `.duckdb` file contains generated physical tables and a versioned APB2 manifest. Logical names
are metadata and are never interpolated into SQL identifiers.

### h5ad and h5mu

The h5 readers accept APB2-authored objects carrying the versioned result envelope under
`uns["apb"]["result"]`. They are not general importers for arbitrary third-party AnnData or MuData.

An h5ad writer requires exactly one level. An h5mu writer accepts one or more levels.

## Fidelity

Parquet and DuckDB preserve the represented `ParsedLevels` value exactly, including:

- level, frame, column, and layer order;
- Polars dtypes;
- null versus NaN;
- string and numeric-looking-string values;
- `obsm`, `varm`, `obsp`, and `varp`; and
- shared and per-level provenance.

h5ad and h5mu intentionally apply the matrix encoding stored with the parsed result. Numeric text
becomes numeric values, configured factor strings become codes, and configured missing sentinels
become missing matrix entries. Reading and rewriting that projected representation is idempotent,
but it cannot recover the original layer tokens.

See the [Python API](api.md) for signatures and result types.
