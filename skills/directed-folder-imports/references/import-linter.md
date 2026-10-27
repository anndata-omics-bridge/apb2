# Import Linter enforcement

Use this reference after the target folder tree is agreed and at least one migrated slice exists.
Import Linter checks imports; it does not decide module ownership for you.

## Install and wire it

Add `import-linter` using the project's existing dependency manager. Keep the configuration at the
repository root and run `lint-imports` from the normal lint/check command and CI.

```ini
[importlinter]
root_packages =
    acme
include_external_packages = True
```

Do not add a second wrapper script when the existing task runner can call `lint-imports` directly.

## Parent with directed children

Suppose the parent contains four composition/application modules and three children. `parsing` may
depend on `rules`; `vendor_parameters` is independent:

```ini
[importlinter:contract:acme-directory-tree]
name = acme children follow the directed folder graph
type = layers
exhaustive = True
containers =
    acme
layers =
    conversion
    detect
    compile
    facade
    parsing | vendor_parameters
    rules
```

Layers are written from outer/highest to inner/lowest. This permits `parsing -> rules`, forbids
`rules -> parsing`, and keeps packages on one `|` line independent. Order parent modules only when
that dependency is intentional; use `|` for parent modules that should also remain independent.

`exhaustive = True` is important. A new top-level module must be assigned deliberately rather than
silently escaping the contract.

This particular contract encodes the intended edge exactly: packages on the same `|` line remain
independent, and `rules` is the one lower sibling target available to `parsing`. In a larger DAG,
however, a layers contract permits a package to import every sibling on every lower line. It does
not state the repository-wide maximum-one invariant by itself, and a later layer insertion can
silently widen the available targets. Pair it with the focused direct-edge check below.

## Nested child tree

Add another contract for a child that has its own inward leaves:

```ini
[importlinter:contract:parsing-directory-tree]
name = parsing descends into data and parameters
type = layers
exhaustive = True
containers =
    acme.parsing
layers =
    parser
    contracts
    decomposition | input_adapter | output_adapter
    errors
    data | parameters
```

One contract cannot infer every recursive folder boundary. Encode each architectural container
whose children are meant to be independent.

## Focused forbidden dependencies

Use a forbidden contract for framework isolation or a particularly important boundary:

```ini
[importlinter:contract:computation-has-no-storage-framework]
name = parsing computation has no storage framework
type = forbidden
source_modules =
    acme.parsing.data
    acme.parsing.parameters
    acme.parsing.contracts
forbidden_modules =
    pydantic
    sqlalchemy
    pandas
    numpy
```

Set `include_external_packages = True` when checking third-party imports.

Use `allow_indirect_imports = True` only when the architectural statement is genuinely about
reachability, not to make a direct-edge contract look stronger than intended.

## Enforce at most one sibling dependency

Import Linter proves the declared direction, but its layers contract does not constrain a package's
out-degree. Add a focused architecture test using Import Linter's underlying Grimp graph:

```python
from pathlib import PurePosixPath

import grimp


def test_each_child_depends_on_at_most_one_sibling() -> None:
    graph = grimp.build_graph("acme")
    children = {"acme.parsing", "acme.rules", "acme.vendor_parameters"}

    for child in children:
        child_modules = {
            module
            for module in graph.modules
            if module == child or module.startswith(f"{child}.")
        }
        imported_modules = set().union(
            *(graph.find_modules_directly_imported_by(module) for module in child_modules)
        )
        imported_siblings = {
            sibling
            for sibling in children - {child}
            if any(
                imported == sibling or imported.startswith(f"{sibling}.")
                for imported in imported_modules
            )
        }
        assert len(imported_siblings) <= 1, (child, imported_siblings)
```

Adapt the graph query to the installed Grimp version and test it against a deliberately invalid
fixture before trusting it. Count direct sibling-package targets, not transitive reachability: the
valid chain `B -> C -> D` gives `B` one direct sibling target, not two.

## What Import Linter does not replace

Use a focused architecture test for facts such as:

- which parent modules may compose several children;
- the maximum-one direct sibling target invariant above;
- the CLI imports only the application facade;
- package `__init__.py` files remain empty;
- a runtime module does not inspect schema discriminator strings;
- the installed wheel includes moved JSON/TOML resources.

Do not rewrite all imports in an AST test. That duplicates Import Linter and creates two graphs that
can drift.

## Ruff is not this tool

Ruff can ban specific imports or enforce style, but it does not model an arbitrary recursive
package tree with parent, child, sibling, and exhaustive ownership relationships. Keep Ruff as the
formatter/linter and Import Linter as the architectural graph check.

## Adoption in a wild project

If the entire legacy package cannot pass immediately:

1. contract the new/migrated package first;
2. prohibit new edges into known legacy areas;
3. list remaining legacy components explicitly in the migration plan;
4. expand the contract after each real move;
5. never use blanket ignores to manufacture a green graph.

The contract should be merge-blocking once its scoped package passes. Diagnostic graph counts may
remain non-blocking gauges; they are not substitutes for the explicit law.
