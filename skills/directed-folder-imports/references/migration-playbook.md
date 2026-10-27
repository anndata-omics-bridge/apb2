# Migration playbook

Use this reference when turning an existing package into the directed tree described by the skill.

## 1. Evidence report

Start with a short factual report:

```markdown
## Current package

Import root: `src/acme`
Public entry points: `acme.cli:main`, `acme.convert`
Package data: `acme/rules/*.json`
Baseline: 412 tests pass; import-linter is not installed

| From | To | Class | Why it exists now |
| --- | --- | --- | --- |
| `acme/parsing/parser.py` | `acme/rules/model.py` | sibling edge | parser receives storage schema |
| `acme/rules/loader.py` | `acme/configure.py` | upward | loader asks parent to construct runtime rules |
| `acme/output/writer.py` | `acme/parsing/parser.py` | second sibling target | writer already imports data |
```

Distinguish direct imports from indirect reachability. The migration changes direct ownership
edges; indirect cycles reveal the order in which slices must move.

## 2. Target-tree worksheet

Write the tree before editing:

```text
src/acme/
├── cli.py                    # imports only conversion.py
├── conversion.py             # application boundary
├── compile.py                # composes rules + parsing + parameters
├── rules/                    # storage documents; imports neither sibling
│   └── schema/               # inward-only Pydantic declarations
├── parsing/                  # storage-neutral runtime parsing
│   ├── data/                 # inward pipeline values
│   └── parameters/           # resolved parsing parameters
└── vendor_parameters/        # independent parameter-file grammar
```

Then state the graph:

```text
acme/*.py -> rules | parsing | vendor_parameters
rules -X-> parsing | vendor_parameters | acme/*.py
parsing -> rules
parsing -X-> vendor_parameters | acme/*.py
vendor_parameters -X-> rules | parsing | acme/*.py
```

The parent files are not a miscellaneous upper layer. Each must name a parent responsibility:
application boundary, facade/translation, detection, or construction/injection.

## 3. Resolving violations

### Child imports parent

Ask which parent value the child actually needs.

- If it is child-owned, move the value/module into the child.
- If it is a broad configuration object, project the exact child-owned parameter at the parent and
  pass it in.
- If it is an error defined by the child boundary, define it in the child.
- If it is orchestration, remove it from the child and call the child from the parent.

Do not create `parent_types.py` and keep the upward import.

### Sibling imports one sibling

One declared `B -> C` edge is valid when `C -X-> B` and `B` has no other direct sibling target.
Keep the import if `B` genuinely owns the workflow that consumes `C` and the edge matches the
declared folder DAG.

Use parent composition and injection when the direct edge would create a cycle or a second sibling
dependency:

```python
# B/ports.py — B owns the capability it consumes.
class Notifier(Protocol):
    def notify(self, recipient: str, subject: str, /) -> None: ...


# C/email.py — no import from B; structural conformance.
class EmailNotifier:
    def notify(self, recipient: str, subject: str, /) -> None:
        ...


# A/compile.py — parent knows and connects both children.
service = OrderService(notifier=EmailNotifier(...))
```

If the event type itself would create a reverse import, pass a consumer-owned value or have the
parent translate between the children. If `B` owns both collaborators, another valid correction is
to nest those collaborators under `B`; modules directly in `B/` can then compose its children.

### Parent module belongs to one child

Move it. A module is not parent-owned because callers historically imported it from the parent.
Update internal callers to the defining module. Preserve an old import only when it is documented
public API, not merely because tests once used it.

### Both siblings need the same code

Do not automatically create a third sibling named `shared`.

Choose among:

1. merge the children because they change and reuse together;
2. make the consumers parent-level modules and put the reusable value in an inward child;
3. let the parent translate/inject a small capability;
4. extract a genuinely independent package with its own public contract.

The correct choice changes the tree; a `shared` bucket avoids making that choice.

## 4. Migration ordering

A reliable order for a dense legacy package is:

1. checkpoint/record the baseline;
2. add characterization and architecture tests around one vertical slice;
3. create the new package boundary without compatibility re-exports;
4. move leaf schemas/values and their tests;
5. move computation behind plain typed inputs/outputs;
6. add parent facade and composition root;
7. cut the external CLI/application entry point to the new path;
8. delete the legacy slice and stale tests;
9. tighten Import Linter from “new package only” to the final graph;
10. repeat for the next slice.

This is how an established wild package can converge without requiring the entire repository to be
clean in one commit. Existing legacy violations are recorded islands, not permission for new edges.

## 5. Review questions

Ask these after every slice:

- Does every moved module have one owner?
- Can any child reach a parent?
- Do sibling edges form a DAG, and does every child directly target at most one sibling package?
- Is the parent doing composition/translation, or merely forwarding?
- Did a broad config/facade cross a boundary where a small value would suffice?
- Did a compatibility wrapper preserve an internal path nobody promised publicly?
- Are tests importing the defining module rather than a package re-export?
- Did package data paths, entry points, and installed-wheel resources survive the move?
- Does the linter contract describe the code that now exists, not a future aspiration?

## 6. Historical lessons captured from APB2

The APB2 migration demonstrated several general points:

- A new, directed namespace can be safer than mutating a heavily coupled legacy tree in place.
- The cutover must be real: once the CLI used Parser V2, legacy production modules and fallback
  routes were deleted rather than kept as a second implementation.
- Storage schemas became an inward child only after their ownership was clear; folder creation was
  the consequence of dependency direction, not the goal.
- A facade that consumed rule storage and produced parsing parameters belonged in their parent,
  not in either sibling.
- Physical readers and writers stayed in the parsing component because they depended only on
  parsing-owned values; an external framework dependency did not move them upward.
- Import Linter encoded the general graph. Small AST tests covered only additional facts such as
  which parent modules were allowed to know two children and whether package markers stayed empty.
- Incremental ownership commits made mistakes reversible and reviewable; the final legacy deletion
  produced the large line-count reduction, not a premature “less code” constraint.
