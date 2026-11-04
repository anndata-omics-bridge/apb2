# apb2 — agent rules

The closest `AGENTS.md` wins. Explicit user instructions override this file.

## Verified commands

| Task | Command |
| --- | --- |
| Synchronize | `uv sync --group dev` |
| Format, including the packaged rules.json layout | `make format` |
| Lint | `make lint` |
| Typecheck | `.venv/bin/pyright` |
| Dependencies | `.venv/bin/deptry .` |
| Tests | `.venv/bin/pytest -q` |
| Build | `uv build && .venv/bin/twine check dist/*` |
| Full gate | `make check` |
| Integration test | `uv run --project ../apb_studio corpus run routine convert` — the 13 routine datasets through the real CLI; see the workspace `AGENTS.md` |

Keep integration scope equal to the tool being changed. For `apb2 convert`, run only `uv run --project ../apb_studio corpus run routine convert`; do not run annotation, FASTA, ProteoBench, the other converter, or a full corpus unless the user explicitly requests broader coverage. Apply the same rule to FASTA work: run only a FASTA-focused workflow. If no such workflow exists, report that fact instead of substituting a broader pipeline.

## Code conventions

- Fully annotate every function and method in `src/` and `tests/`, including
  private functions, callbacks, generators, fixtures, and special methods.
- Standard Pyright strict and Ruff are mandatory. Do not create baselines,
  blanket exclusions, file-wide ignores, or unqualified `# type: ignore`.
- Ruff is the sole formatter and linter. Do not add Black, isort, Flake8, mypy,
  or another overlapping formatter/type checker.
- Keep `__init__.py` empty and import from defining modules inside this package. Other anndata_bridge packages import this one only from `apb2.api`, and it imports them only from theirs. The CLI imports this package only from `apb2.api` too.
- Use Google-style docstrings for public APIs and the configured 100-character
  line length.

## Architecture and import direction

These rules are mandatory for Parser V2 and for every newly introduced or
structurally refactored package. Existing legacy violations do not authorize new
ones.

For a package `A/` with child packages `A/B/`, `A/C/`, and `A/D/`:

- modules directly in `A/` may import from `A/B/` and `A/C/`;
- child packages must not import modules directly in `A/`;
- sibling-package imports must form a one-way directed acyclic graph;
- each child package may directly depend on at most one sibling package; and
- a module in `A/` owned only by `A/B/` moves into `A/B/`; genuine cross-child
  composition remains in `A/`.

For example, `A/B/ -> A/C/ -> A/D/` is permitted; reverse edges and a second direct
`A/B/ -> A/D/` edge are forbidden. If a child needs two siblings, move the composition
to `A/` or place the owned collaborators beneath that package.

For Parser V2, rule storage and parsing are sibling children. Their facade and
runtime composition belong in the parent; neither child imports the other or
the parent. Inside `parse_quant`, the one declared sibling edge is `io -> data`;
`parameters` remains independent, and neither `data` nor `parameters` imports
`io`. Physical readers and writers are parsing-owned, while computational modules
do not import those I/O modules.
Encode each concrete boundary in `.importlinter`; `make lint` and `make check`
must execute `lint-imports`, so the prose rule is also a merge-blocking check.

Parser V2 layers stay long until they are widened, last. A layer row names its cell by integer row numbers (`var_row`, `obs_row`), assigned once by one join per axis on the raw key columns, and by final positions (`var_pos`, `obs_pos`) once the axes are prepared. Duplicate resolution and value parsing run on these long rows; nothing widens or pivots a layer before `ObservationGrid.widen`. A parsed `FinalLayerTable` stays long too: `wide()` builds its matrix on request; writers and the JSON sidecar widen one block of one layer at a time (`LayerBlocks`), never a whole layer. Each parsed level's cells are parked in memory-mapped files (`LayerParking`), and so is each level a reader reads, so the levels of one conversion or one readback do not add up in memory. Identity stays in the raw and final key columns of the small axis frames; the integers only point at axis rows.

## Physical input preparation

- Join and preparation hooks are allowed in two cases: one conversion consumes several physical tables, files or sheets of one workbook, or a single file's rows are not any hierarchy level, so no rule can declare them as one. MSAngel's workbook is an example of the first: its protein-set sheet joins its peptide-ion sheet. MetaMorpheus FlashLFQ `AllQuantifiedPeaks.tsv` is one of the second: its rows are peaks, not ions.
- A hook only combines tables; `prepare_source.py` reads them, delimited files or the workbook sheets the hook names, so column selection, logging and input records stay in one place.
- A hook contains only the work that brings rows to a hierarchy level. Everything a rule document covers stays in the rule; [docs/rule-coverage.md](docs/rule-coverage.md) lists what that is.
- Any other single-file input must use the rule document and standard input reader; never add vendor-specific preprocessing for it.
- Do not use preparation hooks to repair duplicate rows, normalize columns, or otherwise transform a table whose rows already are a hierarchy level.
- Multi-file vendor inputs use their canonical directory as the source. Do not
  add per-file companion options to the CLI or conversion facade.

## Writing parsing rules

- A value per run and feature is a layer (`measurements.layers`). A value per run alone is an `obs` column (`columns.obs`). A value per feature, experiment-wide or taken from the spectral library, is a `var` column (`columns.var`), even where a long report repeats it in every run row: DIA-NN's `Global.Q.Value` and `Lib.Q.Value` are examples.
- Decide the placement from the tool's documentation, the data and what the column means. One dataset alone does not decide it: a column with the same value in every cell of one result says nothing about whether it varies per run, while one that differs across runs in any result is a layer.
- Write the reason for a placement in the entry's `note`, especially where a reader would expect the other one.
- A `var` column keeps a feature's first nonblank value (`"per_feature": "first"`, as most tables' `defaults` say) and reports features whose rows disagree as `var_disagreements`, so check in a real report that the rows agree. A tool that writes placeholders into some rows, as DIA-NN writes 1.0 into `Global.PG.Q.Value`, needs `"per_feature": "min"` (or `max`). Columns that describe one of a feature's repeated rows, as Proline lists a shared peptide once per protein set, take `"per_feature": "best_row"`, and the level names its ranking once, `"rank_rows": {"by": "<score column>", "best": "highest"}`: the level keeps only the row the score ranks best, for the feature table and every layer.
- A prepared table's sources carry the namespace its join adds, such as MaxQuant's `peptide.`; write it once per level as `source_prefix`, and on an entry only where it reads another table.
- A wide layer writes its header as text with `{sample}` where the sample's name goes, `"Intensity {sample}"`; the table declares the regex once as its `sample_pattern`, `(?P<sample>.+)` where the header holds the name alone. Only a header no fixed text describes stays a regex.
- Every rule declares `apb_Decoy` and `apb_Contaminant`; a report with no column marking such rows writes `{"name": "apb_Contaminant", "how": "unmarked"}`, as apb2 writes only what a rule declares.
- A column or layer that holds one value, or none, in a result is not created; the parse record keeps what it held. That says nothing about placement, so declare it where its meaning belongs. Keys, role and computed columns, the flags' inputs and the primary and sample layers are kept.
- A moved layer becomes a `var` entry that writes `"type": "number"` and `"required": false` where its table's `defaults` say otherwise: a required column decides which files the rule accepts.
- A rule states every real choice: a value with more than one option is written, once in the table's `base.defaults` for column and layer fields, or where it applies (`hierarchy`, `parameter_file`, `repeated_cells`, sequence and syntax settings). An entry writes only what differs from its table's `defaults`. A field with one possible value, such as a computed entry's `type` or `required`, is written nowhere; one whose absence means none, such as `note`, stays absent.
- A level whose required columns another level's tables also carry names the column that tells them apart as its `forbidden_columns`, as WOMBAT-P's peptidoform level names `charge`, so an ion table converts at the ion level only. Never drop the column in a hook, and never aggregate the other level's rows into this one.
- `"required": false` means a version sometimes leaves a column out. A column that some versions never write belongs in a rule per version, as Spectronaut 21's library q-values have their own rule beside 19–20's: optional across versions hides which versions carry the column, and apb-catalog names columns per version.
- The steps for a new format are in [docs/adding_a_format.md](docs/adding_a_format.md).

## Dependency rules

### MUST

- Declare every imported runtime dependency directly in `[project.dependencies]`.
- Put tests, linting, typing, building, and documentation tools in dependency
  groups; optional user-facing capabilities belong in extras.
- Update `pyproject.toml` and run `make check`; the repository commits no `uv.lock`.

### SHOULD

- Prefer the standard library, then an existing direct dependency, then a small,
  maintained, typed dependency.
- Keep source independent of test, build, documentation, and CLI-only packages.

### MUST NOT

- Depend on unpinned branches or undeclared transitive dependencies.
- Add parallel manifests, lockfiles, formatters, type checkers, or test runners.
- Silence a dependency or typing defect instead of fixing its source.

## Workflow

1. Preserve unrelated worktree changes.
2. Add or update focused tests with each behavioral change.
3. Run the smallest relevant check while iterating.
4. Run `make check` before handoff and report its actual result.


# APB2 implementation model

- APB2 intentionally uses a restricted subset of Python programming styles.

- Use dataclasses when generated construction would only assign declared fields. This includes values and configured behavior objects; methods belong on a dataclass when they operate cohesively on its configuration. Configuration and runtime-plan dataclasses should normally be frozen.

- Use an explicit constructor when construction validates, normalizes, derives state, or acquires resources. Do not handwrite an `__init__` whose body only copies arguments to attributes.

- Capabilities are Protocols. Protocols are defined by their consumer and describe the smallest capability required at an architectural boundary. Concrete implementations use structural typing and do not inherit from the Protocol.

- Behavior lives in classes. Parsers, writers, policies, normalizers, matchers, compilers and other runtime behavior are objects. Stateless behavior uses an ordinary class without a generated constructor; configured behavior may use a dataclass under the construction rule above. Prefer composition over inheritance.

- Free functions are reserved for composition and facades. Module-level make_*, *_for, compile_*, read_*, and write_* functions may construct/select objects or expose simple package operations. Domain behavior should not be implemented as an unrelated collection of free functions.

- Pydantic is a boundary technology. Pydantic models validate persisted configuration. Runtime algorithms consume validated dataclasses and behavior objects, not Pydantic storage models.

- No utility classes, mixins, service locators, runtime monkey-patching, implicit global state, or deep inheritance hierarchies.
