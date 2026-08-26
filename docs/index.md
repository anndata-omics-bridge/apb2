# APB2

APB2 converts quantitative proteomics vendor tables into AnnData or MuData and can move an
APB2-authored result between h5ad, h5mu, Parquet, and DuckDB without parsing the vendor source
again.

The two user workflows are deliberately separate:

```text
vendor table + parameter evidence
    -> apb2 convert
    -> .h5ad or .h5mu

APB2 result
    -> apb2 reformat
    -> another result format
```

## Convert a vendor table

Convert one quantification level to AnnData:

```bash
apb2 convert report.tsv ion --params search-parameters.txt --output results/ion
```

Omit the level to write every compatible level into MuData:

```bash
apb2 convert report.tsv --params search-parameters.txt --output results/all-levels
```

Continue with [vendor conversion](conversion.md).

## Read, write, or reformat a result

Change only the persisted result format:

```bash
apb2 reformat results/all-levels.h5mu results/all-levels.parquet
```

The corresponding Python API is:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.result_io import ResultFormat, reader_for, writer_for

parsed = reader_for(ResultFormat.PARQUET).read(Path("results.parquet"))
writer_for(ResultFormat.DUCKDB).write(parsed, Path("results.duckdb"))
```

Continue with [result reading and writing](result_io.md) or the [Python API](api.md).

## Supported result formats

| Format | Path | Levels | Result behavior |
| --- | --- | --- | --- |
| AnnData | `.h5ad` | exactly one | configured numeric/factor matrix projection |
| MuData | `.h5mu` | one or more | configured matrix projection per modality |
| APB2 Parquet dataset | `.parquet` directory | one or more | exact Polars values and schemas |
| DuckDB | `.duckdb` file | one or more | exact Polars values and schemas |

APB2 conversion does not perform FASTA annotation, protein inference, or ProteoBench scoring.

The repository [README](https://github.com/anndata-omics-bridge/apb2#readme) provides the compact
project overview. The complete design and dependency rules remain in the
[converter architecture](architecture_converter.md); the user pages summarize them without
replacing or shortening that decision record.
