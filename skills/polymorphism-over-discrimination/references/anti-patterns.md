# Eight ways to do this wrong

> Read this before writing any remedy. Every item below was produced during real reviews and
> corrected only when challenged — these are the failure modes to expect, roughly in the order they
> tend to appear. **§3 and §4 are the two gates in SKILL.md's remedy section**, and they are where
> remedies go wrong most often.

## 1. Replacing `| None` with `| NoThing`

```python
fragments: Fragments | NoFragments = NO_FRAGMENTS       # ✗ Fragments is already a 2-member union
```

Three members spelled as two, and every consumer is back to asking *"is it the `NoFragments` one?"* — `is
None` under a new name. **The absent case is a member of the union, with a discriminator value and the
identity behaviour:**

```python
class NoFragments(RuleModel):
    label_strategy: Literal["none"] = "none"
    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df                                        # ✓ identity arm carries behaviour

type Fragments = Annotated[
    NoFragments | PositionalFragments | ColumnLabeledFragments, Field(discriminator="label_strategy")
]
```

A type split whose empty member has no method leaves the caller discriminating — that is shape 4.

## 2. Returning a token instead of the result

```python
class SumDuplicates:
    def aggfunc(self) -> str: return "sum"              # ✗ caller must dispatch on the string
```

The caller then writes `if aggfunc == "sum": ...`, and you have moved the branch one call deeper while
adding a stringly-typed hop. **Return the result, not the name of the operation:**

```python
class SumDuplicates:
    def combine(self, cells: CellContributions) -> Matrix: ...   # ✓ does the thing
```

## 3. Putting the behaviour on the persistence schema

A pydantic model describes **what may appear in a file**. It is not the computational model. Hanging
conversion logic on it fuses the file format with the computation — and it is why an agent that splits a
pydantic union finds nowhere legitimate to put the method and **stops at the split**.

```
Document (pydantic, knows the format)  →  factory (the one place a tag is read)  →  Runtime (behaviour)
```

The runtime type carries **no discriminator field** — it *is* the type. The document keeps its tag, because
a decoder needs it. Corollary, and it changes verdicts: **a combination-rejecting validator on a document
schema is correct and stays.** Someone can write that combination in the file and must be told. It is a
missing polymorphism only when the validated type is *also* what computes.

## 4. Inventing a method to justify a split

If, after splitting, you find yourself writing a method no caller wants, undo it.

> **If you cannot name the method that would go on the split types, it is a DTO and the validator is doing
> its job.** A missing polymorphism is missing *behaviour*. Two record shapes with no behaviour between them
> are a schema; splitting buys a second name and nothing else.

Cheapest test available — try to finish the sentence *"and then each type would implement …"*.

## 5. Treating the current module layout as a constraint

"A method here would import-cycle, so I will use `singledispatch` instead" — no. **A cycle means the code is
in the wrong file.** If an operation is a type's behaviour, the type and the operation belong in one module,
and moving them there is part of the remedy. Relatedly: **do not invent a layering rule to justify the
detour.** Check what the project's rules actually forbid before reasoning from an observed import habit.

`functools.singledispatch` is a legitimate encoding of the same polymorphism — reach for it when the
operation genuinely is not the type's behaviour (a second backend wanting a different conversion of the same
data), not to route around a file boundary.

## 6. Calling it a Builder

Pattern names are claims about structure. A function that takes a tag and returns one of N types is a
**factory function**; a `dict` lookup returning stateless singletons is not even that, it *selects*. GoF
Builder means incremental construction through a stepwise interface driven by a director. Naming:

- `make_<thing>(...)` — constructs a new object of one of N types
- `<thing>_for(...)` — selects among existing instances
- named constructor (`Format.build`) only when the return type **is** the class; a union has no class to
  hang it on

And do not reach for a Builder to tidy a factory: a builder's internal state is a record whose fields may or
may not be set yet — 2^N shapes validated at `build()`, which is the defect this skill exists to remove,
reintroduced with a respectable name. Builder earns its place when construction is caller-configurable
across many combinations, never for one fixed sequence.

## 7. Showing the union and stopping

A union with no factory is unfinished — nothing yet turns the document's tag into the type. **The factory is
also where an unimplemented variant fails**: a mode declared in the schema with no runtime class is a
missing registry entry, which raises once at construction time instead of partway through the work.

```python
_BY_MODE: dict[Mode, Policy] = {"error": ErrorOnDuplicates(), "aggregate": SumDuplicates()}
                                                     # "keep_all_as_raw_table": absent on purpose
```

Hazard: shared singletons are safe only while the types are frozen and stateless. If one gains state, the
dict must hold classes, not instances, or two runs share an accumulator.

## 8. Narrowing a signature instead of giving the thing a type

Shape 7 (`f(rule: ParseRule)` reading only `rule.axis.duplicates.mode`) tempts a cosmetic fix — narrow the
parameter. **A function reaches for the whole record because the thing it actually wants has no type yet.**
Give that thing a type and the parameter narrows for free; narrow the parameter and every branch survives.
Treat shape-7 hits as a queue of shape 1–6 candidates.

---

