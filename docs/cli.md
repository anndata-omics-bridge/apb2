# CLI reference

APB2 exposes three commands:

```text
apb2 convert
apb2 reformat
apb2 annotate
```

Use `apb2 --help` or a command's `--help` for the installed version's generated Cyclopts reference.

## `apb2 convert`

```text
apb2 convert DATA [LEVEL] [OPTIONS]
```

| Argument or option | Meaning |
| --- | --- |
| `DATA` | Vendor result table |
| `LEVEL` | Optional quantification level; omit it to write every compatible level |
| `--params PATH` | Vendor search-parameter file |
| `--rule-config PATH` | Explicit schema-0.3 rule document |
| `--software NAME` | Disambiguate packaged rule selection |
| `--params-software NAME` | Select the parameter-file parser independently |
| `--output BASENAME` | Output basename without `.h5ad` or `.h5mu` |
| `--strict` | Promote layer-contract warnings to errors |

One of `--params` or `--rule-config` is required. An explicit `LEVEL` writes `.h5ad`; an omitted
level writes `.h5mu`.

See [Convert vendor results](conversion.md) for worked examples.

## `apb2 reformat`

```text
apb2 reformat SOURCE TARGET
```

`SOURCE` and `TARGET` must end in `.h5ad`, `.h5mu`, `.parquet`, or `.duckdb`. No additional option
changes the result semantics.

See [Read and write parsed results](result_io.md) for format contracts and fidelity.

## `apb2 annotate`

```text
apb2 annotate SOURCE ANNOTATION TARGET [OPTIONS]
```

| Argument or option | Meaning |
| --- | --- |
| `SOURCE` | Existing APB2 h5ad, h5mu, Parquet, or DuckDB result |
| `ANNOTATION` | Generic prolfquapp-style CSV/TSV table |
| `TARGET` | New annotated APB2 result; format selected by suffix |
| `--unmatched MODE` | `keep`, `error`, or `drop` behavior |
| `--include COLUMN` | In drop mode, also require a true Boolean annotation field |

ProteoBench module annotation is provided by the separate `apb-proteobench annotate` command. See
[Annotate samples](sample_annotation.md).

## Exit behavior

- `0`: operation completed successfully;
- `1`: expected input, selection, parsing, result-format, or writing failure; and
- `2`: command-line usage or rejected output naming.

Unexpected programming errors are not broadly swallowed; they remain visible with their traceback.
