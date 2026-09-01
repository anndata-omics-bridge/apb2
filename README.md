# apb2

Convert proteomics software output to AnnData (rules-driven parser, second generation)

Read the rendered [APB2 documentation](https://anndata-omics-bridge.github.io/apb2/) or its
[source index](docs/index.md).

## Convert

Use a packaged rule selected from the vendor parameter file and source header:

```bash
apb2 convert DATA LEVEL --params PARAMETER_FILE [--software VENDOR] [--output BASENAME]
```

Omit `LEVEL` to convert every compatible level into one shared-observation MuData container:

```bash
apb2 convert DATA --params PARAMETER_FILE [--software VENDOR] [--output BASENAME]
```

Use an explicit schema-0.3 rule document, with optional search-parameter evidence:

```bash
apb2 convert DATA LEVEL --rule-config RULES_JSON [--params PARAMETER_FILE] \
  [--params-software VENDOR] [--output BASENAME]
```

`LEVEL` is one of `ion`, `peptidoform`, `peptide`, `protein`, or `fragment`. The output basename
must not already carry the suffix APB2 appends: `.h5ad` with an explicit level, `.h5mu` without
one. A no-level conversion writes MuData even when only one level is compatible. `--strict`
promotes layer-contract warnings to errors. The command performs conversion only—FASTA annotation
and protein inference are outside Parser V2.

## Annotate samples

Attach a generic prolfquapp-style CSV/TSV table to any APB2 result format:

```bash
apb2 annotate INPUT ANNOTATION OUTPUT
```

The default prolfquapp behavior retains unmatched quantitative observations and writes null
annotation fields. `--unmatched error` requires complete coverage; `--unmatched drop` explicitly
subsets every observation-aligned value. ProteoBench-specific module annotation and scoring live
in the separate `apb-proteobench` package. See the [sample-annotation guide](docs/sample_annotation.md).

## Reformat a parsed result

Change only the persisted format; no vendor parsing or annotation runs:

```bash
apb2 reformat SOURCE TARGET
```

The suffix selects h5ad, h5mu, an APB2 Parquet directory dataset, or DuckDB. Programmatic callers
use the same explicit adapter boundary:

```python
from pathlib import Path

from apb2.result_facade import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("result.parquet"))
write_parsed_levels(parsed, Path("result.duckdb"))
```

`read_parsed_levels(source)` and `write_parsed_levels(parsed, target)` are path-inferred
conveniences. Parquet and DuckDB preserve Polars result values exactly; h5ad and h5mu apply the
stored numeric/factor matrix projection.

The CLI delegates conversion to Parser V2 and annotation to the independent annotation facade.

The controlling designs and dependency boundaries are documented in
[`docs/architecture_converter.md`](docs/architecture_converter.md) and
[`docs/architecture_annotation.md`](docs/architecture_annotation.md).

## Development

```bash
uv sync --group dev
make check
make docs
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

All Python commands run from the synchronized project `.venv`.
`make docs-serve` serves the user documentation locally. GitHub Actions publishes the strict
MkDocs build to GitHub Pages from `main`.

The rule JSON Schema is a packaged artifact. Developers regenerate it from the Parser V2 rule
package rather than through a user-facing CLI command:

```bash
uv run python -c 'from apb2.parserV2.vendor_parse_rules.schema_artifact import write_artifact; write_artifact()'
```
