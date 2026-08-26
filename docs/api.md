# Python API

The result-I/O API is the supported programmatic boundary for reading, transforming, and writing
APB2-authored results. It works with storage-neutral Polars values rather than AnnData or MuData
objects.

## Format selection

```python
from apb2.parserV2.parse_quant.result_io import ResultFormat
```

`ResultFormat` has four values:

```python
ResultFormat.H5AD
ResultFormat.H5MU
ResultFormat.PARQUET
ResultFormat.DUCKDB
```

## Readers and writers

```python
from pathlib import Path

from apb2.parserV2.parse_quant.result_io import (
    ParsedLevelsReader,
    ParsedLevelsWriter,
    ResultFormat,
    reader_for,
    writer_for,
)

reader: ParsedLevelsReader = reader_for(ResultFormat.PARQUET)
writer: ParsedLevelsWriter = writer_for(ResultFormat.DUCKDB)

parsed = reader.read(Path("results.parquet"))
writer.write(parsed, Path("results.duckdb"))
```

The capabilities are:

```python
class ParsedLevelsReader(Protocol):
    def read(self, source: Path, /) -> ParsedLevels: ...


class ParsedLevelsWriter(Protocol):
    def write(self, parsed: ParsedLevels, target: Path, /) -> None: ...
```

Concrete adapters are selected once by `reader_for()` or `writer_for()`. Callers do not need to
construct or discriminate among backend classes.

## Path helpers

```python
from apb2.parserV2.parse_quant.result_io import (
    read_parsed_levels,
    reformat,
    result_format_for,
    write_parsed_levels,
)
```

```python
result_format_for(path: Path, /) -> ResultFormat
read_parsed_levels(source: Path, /) -> ParsedLevels
write_parsed_levels(parsed: ParsedLevels, target: Path, /) -> None
reformat(source: Path, target: Path, /) -> None
```

These functions infer formats only from the supported suffixes. `reformat()` is a complete
storage-only use case, not a vendor conversion function.

## Result values

```python
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
```

`ParsedLevels` contains an ordered level mapping and shared JSON-compatible provenance. Each
`ParsedLevel` contains:

- `obs: ObsFinal`
- `var: VarFinal`
- `primary_layer_name: str`
- `layers: dict[str, FinalLayerTable]`
- `obsm: dict[str, polars.DataFrame]`
- `varm: dict[str, polars.DataFrame]`
- `obsp: dict[str, polars.DataFrame]`
- `varp: dict[str, polars.DataFrame]`
- `uns: dict[str, JsonValue]`

Layer tables remain wide Polars frames. Their leading columns are authored variable keys and their
remaining columns are observation values. They are not converted to NumPy arrays until an h5ad or
h5mu writer performs the matrix projection.

Pairwise frames have exactly `row`, `column`, and `value` columns. Positions are zero-based local
coordinates into the corresponding final axis.

## Errors

```python
from apb2.parserV2.parse_quant.errors import (
    InvalidResultError,
    ResultIOError,
    UnsupportedResultFormatError,
)
```

Catch `ResultIOError` for expected result-format failures. `UnsupportedResultFormatError` reports
an unsupported suffix; `InvalidResultError` reports an invalid in-memory or persisted result.

## Parser-owned API

A compiled parser still owns the one-level strategy contract:

```python
parsed_level = parser.parse()
parser.convert(parsed_level, Path("ion.h5ad"))
```

Parsing and result I/O therefore meet at `ParsedLevel`/`ParsedLevels`; neither computation nor the
result model imitates an AnnData container.

For compiler construction, rule-schema details, algorithms, and dependency boundaries, consult the
complete [converter architecture](architecture_converter.md). Nothing from that specification has
been moved into this user reference.
