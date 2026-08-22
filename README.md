# apb2

Convert proteomics software output to AnnData (rules-driven parser, second generation)

## Convert

Use a packaged rule selected from the vendor parameter file and source header:

```bash
apb2 convert DATA LEVEL --params PARAMETER_FILE [--software VENDOR] [--output BASENAME]
```

Use an explicit schema-0.3 rule document, with optional search-parameter evidence:

```bash
apb2 convert DATA LEVEL --rule-config RULES_JSON [--params PARAMETER_FILE] \
  [--params-software VENDOR] [--output BASENAME]
```

`LEVEL` is one of `ion`, `peptidoform`, `peptide`, `protein`, or `fragment`. The output basename
must not have an extension; `apb2` appends `.h5ad`. `--strict` promotes layer-contract warnings to
errors. The command performs conversion only—FASTA annotation and protein inference are outside
Parser V2.

The CLI imports only `apb2.parserV2`. Programmatic Parser V2 callers may also select the Parquet
writer without passing through the AnnData adapter.

## Development

```bash
uv sync --group dev
make check
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

All Python commands run from the synchronized project `.venv`.

The rule JSON Schema is a packaged artifact. Developers regenerate it from the Parser V2 rule
package rather than through a user-facing CLI command:

```bash
uv run python -c 'from apb2.parserV2.vendor_parse_rules.schema_artifact import write_artifact; write_artifact()'
```
