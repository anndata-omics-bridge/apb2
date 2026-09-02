# Python API

APB2 exposes programmatic boundaries for vendor conversion, result I/O, and sample annotation.
The file-to-file facades mirror complete CLI operations; the compiler/parser APIs expose
storage-neutral Polars values for composition in larger applications.

## Convert vendor results

### File-to-file facade

Use the facade when the complete operation starts and ends with a file:

```python
from pathlib import Path

from apb2.parserV2.conversion_facade import (
    convert_all_from_packaged_rules,
    convert_from_packaged_rules,
)

convert_from_packaged_rules(
    data=Path("report.tsv"),
    level="ion",
    output=Path("results/ion.h5ad"),
    parameters_path=Path("search-parameters.txt"),
    software=None,
    parameters_software=None,
    checks="standard",
)

convert_all_from_packaged_rules(
    data=Path("report.tsv"),
    output=Path("results/all-levels.h5mu"),
    parameters_path=Path("search-parameters.txt"),
    software=None,
    parameters_software=None,
    checks="standard",
)
```

### Compiler and parser

Use the compiler/parser boundary to keep the parsed result in memory before persistence:

```python
from pathlib import Path

from apb2.parserV2.compile import AnnDataOutput, ParseRuleCompiler
from apb2.parserV2.detect_document import detect_rule_document, search_parameter_evidence
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_params.registry import parse_params

source = SingleFile(path=Path("report.tsv"))
parameters = parse_params(Path("search-parameters.txt"), software="spectronaut")
document = detect_rule_document(parameters, source).document
parser = ParseRuleCompiler(
    facade=ParseRuleFacade(document, "ion", search_parameter_evidence(parameters)),
    output=AnnDataOutput(checks="standard"),
).compile(source)

parsed = parser.parse()
parser.convert(parsed, Path("results/ion.h5ad"))
```

See [Convert vendor results](conversion.md) for rule selection, supported levels, validation, and
output naming.

## Annotate samples

The annotation API constructs an `Annotation` only after the source has been matched and validated
against one `ParsedLevels`:

```python
from pathlib import Path

from apb2.annotation.compiler import AnnotationCompiler
from apb2.result_facade import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("input.h5mu"))
parser = AnnotationCompiler().compile(Path("samples.tsv"))
annotation = parser.parse(parsed)

for level, match in annotation.matches.levels.items():
    print(level, match.coverage, match.corrections)

result = annotation.annotate()
write_parsed_levels(result.parsed, Path("annotated.h5mu"))
```

`AnnotationCompiler` loads and validates the generic delimited source once. The returned parser is
source-bound and can be parsed against several datasets, producing a separate dataset-bound
annotation each time.
`parse(parsed)` raises before constructing an annotation when the selected policy is invalid—for
example, when complete coverage was requested but cannot be met. `annotate()` uses the stored
matches and does not recompute them.

prolfquapp behavior is composed with `KeepUnmatchedAnnotation`,
`RequireCompleteAnnotation`, or `SelectAnnotatedObservations`. All tables and matching evidence
are Polars-backed values. External scientific interpreters use the public capabilities in
`apb2.annotation_extension`; APB2 does not select them by a convention enum.

## Read and write results

### Format selection

```python
from apb2.parserV2.parse_quant.io.formats import ResultFormat
```

`ResultFormat` has four values:

```python
ResultFormat.H5AD
ResultFormat.H5MU
ResultFormat.PARQUET
ResultFormat.DUCKDB
```

### Readers and writers

```python
from pathlib import Path

from apb2.parserV2.parse_quant.io.formats import (
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

### Path-inferred helpers

```python
from apb2.parserV2.parse_quant.io.formats import (
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

## Result model

```python
from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    FinalLayerRole,
    FinalLayerTable,
    MeasurementLayerRole,
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
h5mu writer performs the matrix projection. `FinalLayerTable.role` defaults to
`MeasurementLayerRole()`. Measurement layers may be primary and participate in h5 matrix-occupancy
comparisons. `AuxiliaryLayerRole()` is for numeric diagnostics such as counts or component IDs: the
writer still validates, encodes, stores, and restores these layers, but excludes them from occupancy
comparisons and does not allow one to be the primary layer.

The role-bearing layer field is:

```python
@dataclass(slots=True)
class FinalLayerTable:
    layer_name: str
    var_key_columns: tuple[str, ...]
    values: polars.DataFrame
    role: FinalLayerRole = field(default_factory=MeasurementLayerRole)
```

Pairwise frames have exactly `row`, `column`, and `value` columns. Positions are zero-based local
coordinates into the corresponding final axis.

### Quantitative helpers

Import each helper from the module that defines it:

```python
from collections.abc import Iterable

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import observation_labels
from apb2.parserV2.parse_quant.io.anndata_writer import (
    numeric_result_level,
    quantitative_layer_values,
)
```

Their public signatures are:

```python
observation_labels(count: int, reserved: Iterable[str]) -> tuple[str, ...]
quantitative_layer_values(parsed: ParsedLevel, layer_name: str, /) -> pl.DataFrame
numeric_result_level(parsed: ParsedLevel, /) -> ParsedLevel
```

`observation_labels()` creates collision-free positional value-column names for a wide layer.
`quantitative_layer_values()` applies the layer's stored APB2 numeric encoding and returns only the
variable-by-observation value block. `numeric_result_level()` validates that every layer value
column is numeric or null, then returns a shallow copy marked for plain-numeric writing. It copies
the provenance mapping and does not mutate the input level.

## Errors

```python
from apb2.parserV2.parse_quant.io.errors import (
    AnnDataLayerContractError,
    InvalidResultError,
    ResultIOError,
    UnsupportedResultFormatError,
)
```

Catch `ResultIOError` for expected result-format failures. `UnsupportedResultFormatError` reports
an unsupported suffix; `InvalidResultError` reports an invalid in-memory or persisted result.
`AnnDataLayerContractError` is a `ResultIOError` raised when the encoded layer set violates an h5
required-name check or the measurement-layer occupancy contract.

```python
class AnnDataLayerContractError(ResultIOError): ...
```

## Parser/result boundary

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
