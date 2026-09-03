# apb2

APB2 is a [rules-driven framework](docs/rule-based.md) for converting outputs from proteomics
software into AnnData or MuData. It supports ion, peptidoform, peptide, protein, and fragment
quantification levels and can also store the parsed data in Parquet or DuckDB.

“Rules-driven” means that declarative rule documents describe each vendor table: which columns
contain identifiers, measurements, and metadata, how those columns should be reshaped, and which
constraints the result must satisfy. One shared parser applies those rules, so a new or revised
input format can usually be supported by adding or updating a rule instead of writing a dedicated
reader.

Read the rendered [APB2 documentation](https://anndata-omics-bridge.github.io/apb2/) or its
[source index](docs/index.md). The [supported-software matrix](docs/supported_software.md) lists
every packaged software version, quantification level, vendor input type, table shape, and
parameter parser.

Choose the documentation for your interface:

- [CLI reference](docs/cli.md) and [command-line guides](docs/conversion.md)
- [Python API reference](docs/api.md)

## Motivation and origin

The work that became APB2 was discussed and started during the Copenhagen ProteoBench Hackathon,
13–17 April 2026, as one of the efforts to improve the backend of the
[ProteoBench platform](https://proteobench.cubimed.rub.de/). The hackathon included the public
[EuBIC-MS Seminar 2026 on 15 April](https://eubic-ms.org/events/latest-developments-and-tools-for-data-analysis/).

APB2 was also motivated by the vendor-specific readers maintained behind
[`prolfquapp::preprocess_software()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_software.R#L137)
and in
[`prolfquappPTMreaders`](https://github.com/prolfqua/prolfquappPTMreaders). We plan to move their
remaining input variants and PTM/site-level formats into APB2 so one rules-driven parser can serve
both prolfquapp and ProteoBench.

## Command-line interface

### Convert

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

### Reformat a parsed result

Change only the persisted format; no vendor parsing or annotation runs:

```bash
apb2 reformat SOURCE TARGET
```

The suffix selects h5ad, h5mu, an APB2 Parquet directory dataset, or DuckDB.

### Annotate samples

Attach a generic prolfquapp-style CSV/TSV table to any APB2 result format:

```bash
apb2 annotate INPUT ANNOTATION OUTPUT
```

The default prolfquapp behavior retains unmatched quantitative observations and writes null
annotation fields. `--unmatched error` requires complete coverage; `--unmatched drop` explicitly
subsets every observation-aligned value. ProteoBench-specific module annotation and scoring live
in the separate `apb-proteobench` package. See the
[sample-annotation guide](docs/sample_annotation.md).

## Python API

The file-to-file facades mirror the CLI operations. The compiler/parser APIs expose
storage-neutral values for custom pipelines. Result formats also have explicit adapters:

```python
from pathlib import Path

from apb2.result_facade import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("result.parquet"))
write_parsed_levels(parsed, Path("result.duckdb"))
```

Parquet and DuckDB preserve Polars result values exactly; h5ad and h5mu apply the stored
numeric/factor matrix projection. See the [Python API reference](docs/api.md) for vendor
conversion, annotation, result values, and errors.

## Architecture

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
Zensical build to GitHub Pages from `main`.

The rule JSON Schema is a packaged artifact. Developers regenerate it from the Parser V2 rule
package rather than through a user-facing CLI command:

```bash
uv run python -c 'from apb2.parserV2.vendor_parse_rules.schema_artifact import write_artifact; write_artifact()'
```
