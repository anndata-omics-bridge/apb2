# AGENTS.md guidance

Use this concise block after verifying the actual package names and check commands. Put it in the
closest AGENTS.md governing the package. Do not paste speculative module names.

```markdown
## Architecture and import direction

Directory nesting is an import rule, not decoration. For a package `A/` with child packages
`A/B/`, `A/C/`, and `A/D/`:

- modules directly in `A/` may import from `A/B/` and `A/C/`;
- child packages must not import modules directly in `A/`;
- sibling-package imports must form a one-way directed acyclic graph;
- each child package may directly depend on at most one sibling package; and
- a module in `A/` owned only by `A/B/` moves into `A/B/`; genuine cross-child composition
  remains in `A/`.

For example, `A/B/ -> A/C/ -> A/D/` is permitted; the reverse edges and a second direct
`A/B/ -> A/D/` edge are forbidden. If a child needs two siblings, move the composing code to `A/`
or make the owned collaborators children of that package.

Apply the same rule recursively to `[NAME THE NESTED PACKAGES]`. `[NAME THE PARENT MODULES]` are
the authorized cross-child composition boundary. Do not add exceptions, compatibility re-exports,
or `utils`/`shared` buckets to bypass ownership.

The executable contracts live in `[PATH TO .importlinter]`. Run `[VERIFIED LINT COMMAND]`; CI and
`[VERIFIED CHECK COMMAND]` run the same `lint-imports` gate.
```

Also document scoped exceptions, if any, as temporary migration islands with an owner and removal
condition. “Legacy code” without a named boundary is not an exception.

Keep the rule short in AGENTS.md and point to the architecture document for the concrete tree.
AGENTS.md tells an agent what to preserve; the linter proves the edges, and the architecture
document explains why the components exist.
