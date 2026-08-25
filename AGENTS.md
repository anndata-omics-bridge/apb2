# apb2 — agent rules

The closest `AGENTS.md` wins. Explicit user instructions override this file.

## Verified commands

| Task | Command |
| --- | --- |
| Synchronize | `uv sync --frozen --group dev` |
| Format | `.venv/bin/ruff format src tests && .venv/bin/ruff check --fix src tests` |
| Lint | `make lint` |
| Typecheck | `.venv/bin/pyright` |
| Dependencies | `.venv/bin/deptry .` |
| Tests | `.venv/bin/pytest -q` |
| Build | `uv build && .venv/bin/twine check dist/*` |
| Full gate | `make check` |
| Integration test | `make -C ../apb_studio corpus-routine` — 10 named fixtures through the real CLIs; see the workspace `AGENTS.md` |

Keep integration scope equal to the tool being changed. For `apb2 convert`, run only
`make -C ../apb_studio corpus-routine CORPUS_PIPELINE=apb2-convert`; do not run annotation,
FASTA, ProteoBench, the other converter, or a full corpus pipeline unless the user explicitly
requests broader coverage. Apply the same rule to FASTA work: run only a FASTA-focused workflow or
test target. If no such target exists, report that fact instead of substituting a broader pipeline.

## Code conventions

- Fully annotate every function and method in `src/` and `tests/`, including
  private functions, callbacks, generators, fixtures, and special methods.
- Standard Pyright strict and Ruff are mandatory. Do not create baselines,
  blanket exclusions, file-wide ignores, or unqualified `# type: ignore`.
- Ruff is the sole formatter and linter. Do not add Black, isort, Flake8, mypy,
  or another overlapping formatter/type checker.
- Keep `__init__.py` empty and import public objects from their defining modules.
- Use Google-style docstrings for public APIs and the configured 100-character
  line length.

## Architecture and import direction

These rules are mandatory for Parser V2 and for every newly introduced or
structurally refactored package. Existing legacy violations do not authorize new
ones.

For a package `A/` with child packages `A/B/` and `A/C/`:

- modules directly in `A/` may import from `A/B/` and `A/C/`;
- code under `A/B/` or `A/C/` must not import modules directly in `A/`;
- `A/B/` and `A/C/` must not import one another; and
- a module in `A/` owned only by `A/B/` moves into `A/B/`; genuine cross-child
  composition remains in `A/`.

For Parser V2, rule storage and parsing are sibling children. Their facade and
runtime composition belong in the parent; neither child imports the other or
the parent. Physical readers and writers are parsing-owned and stay inside the
parsing package, while computational modules do not import those I/O modules.
Encode each concrete boundary in `.importlinter`; `make lint` and `make check`
must execute `lint-imports`, so the prose rule is also a merge-blocking check.

## Dependency rules

### MUST

- Declare every imported runtime dependency directly in `[project.dependencies]`.
- Put tests, linting, typing, building, and documentation tools in dependency
  groups; optional user-facing capabilities belong in extras.
- Update `pyproject.toml` and `uv.lock` together and run `make check`.

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
