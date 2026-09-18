# Sample-annotation architecture

## Public object model

```python
parser = AnnotationCompiler(...).compile(annotation_source)
annotation = parser.parse(parsed)
result = annotation.annotate()
```

The compiler owns generic delimited-source loading. The source-bound parser owns source
interpretation, matching, and policy validation. A concrete `Annotation` is created only when it is
valid for the supplied dataset, and therefore stores that `ParsedLevels` and its completed
`AnnotationMatches`. Application cannot accidentally combine evidence from one dataset with
another.

Parsing, matching, policy validation, and application remain separate internal functions and
types. They are folded together only at the public construction boundary; matching is not hidden in
a dataclass initializer.

## Values and behavior

`AnnotationTable`, `AnnotationCoverage`, `LevelAnnotationMatch`, `AnnotationMatches`, and
`AnnotationResult` are frozen data values. They select no behavior. Runtime policies implement
keep, complete-coverage, or selection behavior, while concrete source packages own their
convention-specific diagnostic semantics.

prolfquapp validity depends on the configured application: keep accepts partial coverage, complete
rejects it, and selection validates its Boolean fields before construction. Every policy rejects a
level with zero matches because such an annotation is not an annotation for that dataset.

## Dependency direction

```text
apb2/
├── annotation_facade.py       generic result I/O + orchestration
├── annotation_extension.py    public external-interpreter capabilities
└── annotation/
    ├── compiler.py            generic delimited-source composition
    ├── prolfquapp.py           source-bound parser + bound annotation
    ├── source/                 CSV/TSV decoding
    ├── application/            retention and selection behavior
    ├── matching/               exact/fuzzy matching
    └── data/                   innermost values and errors
```

Modules directly in `annotation/` compose children. The child packages import only the innermost
data package. Annotation computation depends on the `ParsedLevels` value model but never on result
I/O, Pydantic rule documents, pandas, AnnData, or MuData. `annotation_facade.py` is the outer adapter
that reads and writes physical results.

External packages explicitly compose `make_annotation_table`, matching, an application policy,
and `record_annotation_provenance` through `apb2.annotation_extension`. APB2 does not discover or
branch on convention names. `apb-proteobench` is the first external interpreter.

The optional rule declaration `sample_annotation.matching` is a Pydantic storage schema. Parser V2
projects its exact/fuzzy mode and optional key normalization into JSON-compatible level provenance.
Annotation matching constructs its runtime matcher from that persisted value and never imports the
vendor-rule package. Normalization changes comparison keys only; collision checks and diagnostic
evidence remain owned by matching.

## Storage behavior

Annotation follows the [APB metadata specification](metadata_specification.md#tool-operations). It adds columns to each level's `obs`, records one source descriptor and configuration in `ParsedLevels.metadata[tool]["provenance"]["annotation"]`, and records coverage, corrections and bounded mismatch evidence in `ParsedLevel.metadata[tool]["annotation"]`. The tool key identifies the convention; there is no parallel `metadata["annotation"]` namespace or repeated convention alias.

Parse provenance remains in each owner's `uns`. Result writers project the tool namespaces directly into the owning object's `uns["apb"]`, composing root and local contributions only for standalone H5AD. Recording annotation preserves unrelated operations in the same tool namespace; no backend-specific or tool-name-specific annotation implementation exists.
