# Python API

APB2 exposes programmatic boundaries for in-memory vendor parsing, result I/O, and sample annotation. The compiler/parser APIs expose storage-neutral Polars values for composition in larger applications; file-to-file vendor conversion belongs to the CLI command workflow.

## Discover packaged result rules

`get_rules()` lists packaged quant-result rules by APB2 category, without opening a vendor result or parameter file:

```python
from apb2.api import get_rules

for software in get_rules("DIA", level="ion"):
    print(software.software_name)
    for variant in software.variants:
        print(variant.software_version_pattern, variant.levels)
```

The initial categories are `DDA` and `DIA`; a rule can belong to both. Each variant identifies one packaged rule and retains its own supported quantification levels. `software_version_pattern` is the rule's regular-expression version range, not a list of tested releases. The catalogue reports rule availability, not whether an arbitrary file can be parsed without search parameters. Parameter-parser-only software and ProteoBench-specific UI aliases are not included. Unknown categories raise `ValueError`.

## Convert vendor results

Use the public compiler when another package needs canonical parsed values:

```python
from pathlib import Path

from apb2.api import ParseRuleCompiler, write_parsed_levels

compiler = ParseRuleCompiler(
    Path("report.tsv"),
    Path("search-parameters.txt"),
    requested_levels=("ion",),
    software="spectronaut",
    checks="standard",
)
parser = compiler.compile()
parsed_levels = parser.parse()
write_parsed_levels(parsed_levels, Path("results/ion.h5ad"))
```

Construction binds the physical source, chooses the parameter parser, parses typed parameters, detects compatible rules, and resolves the requested levels. `compile()` returns one `ParserCollection`; `parse()` returns canonical `ParsedLevels` and performs no write. `compiler.parameters` and `compiler.detection` retain the typed parameter and detection results. Storage is selected only by the target passed to `write_parsed_levels()`.

For a ProteoBench Custom upload, use `ParseRuleCompiler(Path("custom.txt"), None, software="pb_custom", requested_levels=("ion",))`. The packaged rule needs no parameter file; `compiler.parameters` is unavailable in this case, while `compiler.detection` still describes the selected rule.

For vendor exports uploaded without search parameters, the Python API provides a separate constructor:

```python
compiler = ParseRuleCompiler.from_software(
    Path("combined_ion.tsv"),
    software="fragpipe",
    requested_levels=("ion",),
)
parsed_levels = compiler.compile().parse()
```

`software` is required and names the result producer; for DIA-NN output from a FragPipe workflow, use `software="diann"`. APB2 matches the requested levels against that producer's packaged rules using their source columns and declared version signatures. For DIA-NN 2.x, the declared MS1 columns distinguish DDA from the DIA default without a separate acquisition argument. Ambiguous matches and search settings that columns cannot establish still raise errors. `compiler.detection.version` is `None`, `compiler.parameters` is unavailable, and the normal rule provenance remains in the parsed result. The CLI uses this same constructor for `apb2 convert DATA ion --software DIA-NN` when `--params` is omitted.

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
storage-only use case, not a vendor conversion function. Both writes publish an adjacent compact `<artifact>.apb.json` representation.

The public result facade also exposes `project_result(parsed, artifact=None)`, `sidecar_path(artifact)`, and `write_result_representation(parsed, artifact)` for consumers that need to inspect or republish the versioned representation explicitly. Omitting the artifact produces the same in-memory scientific document with `artifact: null`.

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

### Semantic conversion roles

For rule-driven conversion, `ParsedLevel.uns` contains `column_roles` and `layer_roles` inside parse provenance. `column_roles` maps a semantic role such as `protein_assignment` to one logical var-column name. `layer_roles` maps a role such as `abundance` to an ordered list of retained layer names. The allowed role/owner combinations come from the packaged [role policy](../src/apb2/parserV2/vendor_parse_rules/schema/role_policy.json).

These maps are JSON-compatible provenance rather than Python strategy objects. Consumers may use them to discover scientific meaning without knowing vendor-specific names, while readers and writers preserve them without interpretation.

### Structural layer roles

Layer tables remain wide Polars frames. Their leading columns are authored variable keys and their
remaining columns are observation values. They are not converted to NumPy arrays until an h5ad or
h5mu writer performs the matrix projection. `FinalLayerTable.role` defaults to
`MeasurementLayerRole()`. Measurement layers may be primary and participate in h5 matrix-occupancy
comparisons. `AuxiliaryLayerRole()` is for numeric diagnostics such as counts or component IDs: the
writer still validates, encodes, stores, and restores these layers, but excludes them from occupancy
comparisons and does not allow one to be the primary layer.

The structural-role-bearing layer field is:

```python
@dataclass(slots=True)
class FinalLayerTable:
    layer_name: str
    var_key_columns: tuple[str, ...]
    values: polars.DataFrame
    role: FinalLayerRole = field(default_factory=MeasurementLayerRole)
    semantics: FinalLayerSemantics = field(default_factory=QuantitativeLayerSemantics)
```

Pairwise frames have exactly `row`, `column`, and `value` columns. Positions are zero-based local
coordinates into the corresponding final axis.

### Quantitative helpers

Import the public helpers from the result facade:

```python
from collections.abc import Iterable

import polars as pl

from apb2.result_facade import observation_labels, quantitative_layer_values
```

Their public signatures are:

```python
observation_labels(count: int, reserved: Iterable[str]) -> tuple[str, ...]
quantitative_layer_values(parsed: ParsedLevel, layer_name: str, /) -> pl.DataFrame
```

`observation_labels()` creates collision-free positional value-column names for a wide layer. `quantitative_layer_values()` returns the already-canonical variable-by-observation value block for a quantitative layer; it performs no stored-plan interpretation or conversion.

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
