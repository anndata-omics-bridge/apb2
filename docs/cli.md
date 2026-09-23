# CLI reference

APB2 exposes three commands:

| Command | Purpose | Guide |
| --- | --- | --- |
| `apb2 convert` | Parse a vendor table or result directory into HDF5, Parquet, or DuckDB | [Convert vendor results](conversion.md) |
| `apb2 reformat` | Change the storage format of an APB2 result | [Read and write results](result_io.md) |
| `apb2 annotate` | Attach sample metadata to an APB2 result | [Annotate samples](sample_annotation.md) |

Use `apb2 --help` or a command's `--help` for the installed version's generated Cyclopts reference.
The [support matrix](supported_software.md) lists compatible vendor versions, inputs, parameter
parsers, and quantification levels.

## `apb2 convert`

```text
apb2 convert DATA [LEVEL] [OPTIONS]
```

| Argument or option | Meaning |
| --- | --- |
| `DATA` | Vendor result table or directory containing named result tables |
| `LEVEL` | Optional quantification level; omit it to write every compatible level |
| `--params PATH` | Vendor search-parameter file; omit only for an explicit rule or a parameter-free packaged rule |
| `--rule-config PATH` | Explicit schema-0.8 rule document |
| `--software NAME` | Parameter-file grammar and result-rule hint; required without `--params` for parameter-free packaged rules such as `pb_custom` |
| `--format FORMAT` | `hdf5`, `parquet`, or `duckdb`; default `hdf5` |
| `--output BASENAME` | Output basename without the selected suffix |
| `--strict` | Promote layer-contract warnings to errors |

Use `--params` for ordinary packaged rules, `--software pb_custom` for a parameter-free ProteoBench Custom upload, or `--rule-config` for an explicit rule. A directory supplies multi-file table groups. An explicit `LEVEL` selects one decomposition; omission converts all supported levels. HDF5 uses `.h5ad` for an explicit level and `.h5mu` otherwise. Parquet and DuckDB retain their suffixes. Compatible one-to-one observation aliases are aligned; incompatible resolutions get separate key-qualified outputs, such as `result.raw_file.h5mu` and `result.experiment.h5mu`. The CLI reports every actual path; see [output naming](conversion.md#output-naming).

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

- `0`: operation completed successfully
- `1`: expected input, selection, parsing, result-format, or writing failure
- `2`: command-line usage or rejected output naming

Unexpected programming errors are not broadly swallowed; they remain visible with their traceback.
