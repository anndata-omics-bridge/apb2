---
name: polymorphism-over-discrimination
description: >-
  Find and fix code that decides what to do by asking what something is. Use this skill whenever
  reviewing or writing Python that contains an isinstance chain, an `if mode == "x" / elif` chain,
  `if record.field is None` where the arms do different work, a validator that rejects field
  combinations ("X is only valid when Y is Z"), a function returning a string that names an
  operation, or a record carrying a kind flag plus fields belonging to only one kind. Use it when
  adding an arm to any existing chain of those, when adding an optional field to a type you own,
  when introducing a `Missing`/`No`-prefixed class, and when naming something Builder, Factory,
  Strategy, or Visitor. Also use it for any request to review Python design, reduce branching,
  clean up conditionals, or audit a package for structural problems — even when the user does not
  say "polymorphism". Python only.
---

# Polymorphism over discrimination

**Branching on what something *is*, in order to decide what to *do*, is a missing polymorphism.
Put the behaviour on the type.**

The bound is half the rule. This is **not** "few `if` statements are good". Branching on what
*happened* — empty, absent, invalid, out of range — is ordinary control flow. `if not values:
return None` is correct Python and must not trigger an abstraction reflex. A review that flags
guard clauses is worse than no review, because it teaches the reader to distrust the whole report.

## When the task is a design review, start earlier

This protocol needs design judgement the sweep below does not have. If you were delegated only
the mechanical sweep, report its output and state plainly that the design question was not asked
— do not guess a target model.

The shapes and bounds below judge branches one at a time. A design review asks a prior question:
**what decides what, and where?** Before any sweep, build the decision inventory — one row per
point where the code decides what something *is* (every factory, suffix map, content probe, mode
field, flag):

| Decision | Evidence it uses | Where decided | Where consumed |
|---|---|---|---|

Then two rules:

1. **Decisions drawing on evidence from the same object are one decision.** A package that chose
   vendor by filename suffix at construction and acquisition mode from file content
   mid-extraction had two single dispatch points, each individually passing the bounds below —
   and together they were one undeclared factory: `build_parser(rawfile)`.
2. **A decision belongs where its evidence is.** A selecting value computed at level A and
   consumed at level C means the type should be chosen at A and the object passed down, not the
   flag.

The review's first deliverable is the **target object model**: the two or three lines the caller
should write. State it before enumerating findings; each finding is then a delta from that
model. A review that grades branches without stating the model has answered this skill's
question and not the review's — that is the difference between an audit and a design review.

## Get the worklist mechanically

Do not start by grepping for `isinstance`. Most of what that finds is correct code, and reading it
burns the review's attention before it reaches anything real.

```bash
python3 <this-skill-dir>/scripts/find_candidates.py <package-dir>          # readable worklist
python3 <this-skill-dir>/scripts/find_candidates.py <package-dir> --json    # machine-readable
```

One pass produces every candidate list below, already sorted, with the rejections that need no
judgement applied. On a 97-module package it rejected 57 % of `isinstance` sites outright and
ranked the remaining work in about two seconds.

**Its output is a floor, not the worklist.** This is measured, not a caveat for form's sake: in a
head-to-head run against the same skill without the script, the *unassisted* pass found the single
best defect in one subpackage — a `bool` field (shape 8) re-answering one question at four sites —
and the script-assisted pass missed it while working through the sections above. Every shape below is
invisible to a different query. So after the script, **read the two or three modules that look most
tangled**, with the shapes in mind and no grep. Budget it; it is where the findings the tool cannot
name come from.

**Read its `IDENTITY TYPES WITHOUT METHODS` section first.** That is consistently the highest-yield
output, for a reason worth understanding: a codebase that has adopted "replace `| None` with a named
absent case" grows those types by the dozen, and **nothing in that discipline makes anyone add the
method.** The modelling lands; the branch survives. On the package above: 18 such types, **0 with a
method**, discriminated in 8 modules.

> **⚠ That section's headline is the one number in this skill you must not optimise.** It counts
> *types without methods*, so **adding a method always improves it — including a method nobody
> calls**, which is `references/anti-patterns.md` §4 with a green dashboard. The honest fix is as
> often deletion: see **gate 3** in *The remedy*. On a later pass of the same package the count fell
> 14 → 5 by deleting nine types and adding none. Treat every one of these outputs as a gauge, never a
> gate; a CI check on this number would reward exactly the wrong edit.

## The one question

> Does this branch ask **what something is** (type, vendor, format, mode, strategy,
> which-variant), or **what happened** (missing, empty, out of range, this file lacks that column)?

*What it is* → continue. *What happened* → done, the `if` is correct.

**Ask it per condition, not per branch.** One `if` can join a finding and a non-finding with `and`:

```python
if (isinstance(params, StoredSearchParameters)          # which variant is this → FINDING
        and params.parameters.min_length is not None):  # did the vendor state one → correct
```

Judging that `if` as a unit gives the wrong answer whichever way you round.

## Bounds — cheapest disqualifier first

The first three need no reading. Most candidates die here, and that is the point: a shortlist you
can defend beats a long list you cannot.

**B1 — Is the `isinstance` target a class this package defines?** If not, you cannot give it a
method, so narrowing it is correct code at a parse boundary. `find_candidates.py` applies this for
you. The rejects are `str`, `dict`, `list`, `int`, `bool`, `h5py.Group`, `np.ndarray`, DataFrames.

Note carefully: **this is about where the foreign value is, not which package handles foreign
data.** `Tolerance.parse(value: object)` is the boundary; everything past it is yours and gets no
protection from this bound.

**B2 — Does the union have ≥ 2 consumer sites?** One site is B5 territory. Two in one module is
weak; **two modules is the threshold worth reporting.** Skipping this over-reports every small
result type. The script prints site and module counts per union.

> **B2 counts per union, so it can miss a module.** One package had six one-site absent types in a
> single file — each scored `weak`, all six below threshold — while that file ranked **5th in the
> package** for owned-target `isinstance`. Six non-findings summed to a real one and the bound could
> not say so. **Before clearing a module, check whether several sub-threshold unions share one
> consumer**; the script prints a `CLUSTERED WEAK TYPES` rollup for exactly this. The finding there
> is rarely "add methods to six types" — it is usually one thing all six work around, as it was: the
> six existed only
> to decide whether a key reached the output JSON, which `model_dump(exclude_none=True)` already did
> for a neighbouring field in the same function.

**B3 — For a mode-string candidate, does anything at the far end branch on the value?** grep the
literal. If nothing compares against it, the string is a **label**, not a discriminator — that is
duplication at worst.

**B4 — Guard clauses, validation, emptiness, boundary checks.** Also: *ordering* (`min > max`),
*uniqueness*, *set equality*, *non-emptiness* validators police **values**, not kinds. The validator
shape only counts when **one field is a mode, kind or strategy and the others are that mode's
payload**. Quick proxy: the condition compares a field to a **literal** (`self.mode == "absolute"`)
→ candidate; a field to another **field** (`self.min > self.max`) → validation.

**B5 — Two arms that will demonstrably never grow.** Say so and move on.

**And one destination that is not a defect:** a single dispatch point at a factory or composition
root is right. A `dict[str, Handler]` with one lookup is the cure. When a strategy needs runtime
configuration, the values may instead be factories that bind that configuration and return the
same `Handler` contract. The smell is the same case set appearing *again* past that lookup, or the
returned handler carrying its selection token downstream so another layer can dispatch on it.

A meta-selection such as `"all"` is also legitimate orchestration at this boundary. It expands one
request into several already-defined strategies; it does not define a new strategy and does not
belong in the numerical or domain behaviour. Keep the vocabularies distinct — for example,
`StrategyName` for executable strategies and `StrategySelection = StrategyName | Literal["all"]`
for the composition root — so `"all"` cannot accidentally flow into code that expects one
behaviour.

Two qualifications, both from a review this bound derailed. It holds only when that dispatch is
the **only** one answering its question: two single dispatch points keyed on properties of the
same input — a suffix map at `__init__`, a content probe mid-extraction — are fragments of one
factory, and each passes this bound alone. So **count the factories**: list every function
returning a union or protocol instance; two keyed on the same input are one. And the dispatch
must sit **where the evidence is** — a decision that fires once, but two frames below the level
that computed its input, is at the wrong level, not exonerated.

## The shapes

Ordered by how mechanically checkable they are. Findings from the top are evidence; from the bottom,
opinion. **Say which you are reporting** — a review that mixes them loses the reader's trust in both.

| # | Shape | Where the script reports it |
| --- | --- | --- |
| 1 | **A validator rejecting field combinations**, one field being a kind | `VALIDATOR MESSAGES` — then apply B4 |
| 2 | **The same case set branched on in more than one function** | derive from the `union:` line + site counts |
| 3 | **`if x.field is None` on a type you own**, arms doing different work | `OPTIONAL-FIELD DISCRIMINATION` (bare guards pre-removed) |
| 4 | **A union you already split, whose consumers still discriminate** | `IDENTITY TYPES WITHOUT METHODS` — **read this first** |
| 5 | **A function turning a mode into a string another function branches on** | `MODE-STRING RETURNS` — then apply B3 |
| 6 | **`isinstance` / `== "literal"` chains** | `OWNED-TARGET ISINSTANCE` — foreign already rejected |
| 7 | **A record passed whole where the body reads one field** | `WIDE-PARAMETER READS` — **reports nothing on its own** |
| 8 | **A `bool` field or flag selecting behaviour** | nothing — **only reading finds this** |

**Shape 8 deserves its own note, because it is the one the script cannot reach at all.**
`if params.uses_diann:` is `if vendor == "diann":` with the case set narrowed to two and its name
hidden inside the field name. It clears the bounds exactly like any other discriminator — same
question re-answered at several sites — and it is invisible to every query aimed at `isinstance` or a
string literal. When a flag's name contains a vendor, format, tool or mode, treat the flag as that
discriminator wearing a `bool`.

Two tells the script surfaces but cannot judge:

- **The message text, for shape 1.** *"X is only valid when Y is Z"* → Y is a type discriminator.
  *"X must not exceed Y"* → validation. This fires while the validator is being written, which is
  before the case set has spread to three modules.
- **The docstring, for shape 4.** An empty class whose docstring *describes behaviour* is a missing
  method. `"""Use the stored enzyme when present, otherwise Trypsin."""` on a class with no fields
  and no methods is the defect announcing itself in prose.

**Shape 4 must clear gate 2 before you report it, and the script cannot check that.** A pydantic
discriminated union is *supposed* to have no methods. If its only consumers are its own validator and
its factory, it is finished code and the branch is exactly where remedy step 6 puts it. One package
had `NoValuePattern | RegexValuePattern` scoring FINDING at 2 sites / 2 modules — one a schema
validator, one `make_layer_coercion`, with the runtime polymorphism already built alongside as
`LayerCoercion`. Acting on it would have converted a correct design into a layering violation. Check
where the two sites *are* before writing a remedy.

## The remedy, in order

**The first two steps are gates.** Both can disqualify the whole finding, and both are cheaper than
any edit. Asking them late means writing a full remedy before discovering it was not needed.

1. **Gate — name the question every arm answers.** *"Which token index is this?"*, *"How do several
   values in one cell become one?"* That name is the method name. **If you cannot name it, stop: it
   is a DTO and the validator is doing its job.** Highest-value question in the skill; costs one
   sentence. See `references/anti-patterns.md` §4.
2. **Gate — is the type a storage schema?** A pydantic model, a TOML/JSON shape, a table row? Then
   the behaviour does *not* go on it: the remedy is a runtime type plus a factory, and the schema's
   validators are **correct and stay**. Getting this wrong converts a clean finding into a layering
   violation. See `references/anti-patterns.md` §3. And note what a runtime type *is*: a genuinely
   different type carrying only what the computation needs. **A wrapper that holds the schema and
   forwards to it is not one** — it is a carpet, and it grows the type count you came to reduce.
3. **Gate — for an absent case: does it *do* anything?** This gate has two exits and only one of
   them adds code.
   - It does something → give it the method. `MissingSearchParameters.cleavage(default=…)` returns
     the default **and logs the fallback**; `MissingRuleVersion.accepts()` returns `True` because
     absent evidence excludes nothing; `MissingGeneName.cell()` yields `pd.NA`. Continue to step 5.
   - **It does nothing → delete the type.** It is `| None` wearing a class. Every consumer only ever
     asked it whether the value was there, which is *what happened*, and the one question at the top
     of this skill already ruled that ordinary control flow.

   The test is whether you can finish *"and then each type would implement …"*. When you cannot, the
   answer is not a weaker method — it is that the type should not exist.

   Where the `| None` may then live: a **storage boundary** — an adapter, a CLI, a parser — where
   absence is a fact about the world. Not in a computation signature; that rule does not move. In one
   package this deleted nine classes and removed nineteen `isinstance` sites, and it is the exit two
   review drafts missed while proposing eleven new wrapper classes instead. **A package whose stated
   defect is "18 types too many" is not repaired by a nineteenth.**

   Beware the near-miss: wrapping the present arm so you can narrow on *that* instead
   (`isinstance(payload, StoredNamespaceText)`) relocates the branch and satisfies step 7's letter
   while failing its intent entirely.
4. **Split the type**, or give the absent case an identity member. One class per case, each
   declaring only its own fields. Constraints the validator policed from outside become field
   declarations: `Field(min_length=1)`, a required `str` instead of `str | None`, a `Literal` for a
   forced name.
5. **Put the method on each class, and return the result — not a token naming it.**
6. **Write the factory.** A union with nothing to construct it is not finished. It is also where an
   unimplemented variant fails: a mode declared in the schema with no runtime class is a missing
   registry entry, raising once at construction time instead of partway through the work. Registry
   values can be ready instances for fixed strategies or construction callables for configured
   strategies. Bind variant-specific options here and return one uniform runtime contract; do not
   make consumers inspect the strategy name to recover those options.
7. **Delete the branches.** If any survive, the polymorphism moved rather than happened.

**Do not stop at step 4.** Splitting a type without moving behaviour onto it produces shape 4 —
measurably the most common defect this skill finds.

**Write the remedy only for findings you are actually reporting, and size it to the finding.** A
full before/after for something you have already graded below your own threshold is not thoroughness,
it is padding that buries the findings that matter — and it reads as a recommendation whatever the
surrounding hedge says. A weak or rejected candidate gets one line saying which bound it failed. If
the honest answer to the whole question is "no, this is fine", say that and stop; a short decisive
report is the right output, not a failure to find enough.

## Naming

Pattern names are claims about structure, and a wrong one sends every future reader hunting for
machinery that is not there.

- `make_<thing>(...)` — constructs a new object of one of N types. A factory function.
- `<thing>_for(...)` — selects among existing instances. A lookup; constructs nothing.
- a named constructor (`Format.build`) only when the return type **is** the class. A union has no
  class to hang it on.
- **Not `Builder`** unless construction is incremental, stepwise and director-driven. And do not
  reach for a Builder to tidy a factory: a builder's internal state is a record whose fields may or
  may not be set yet — 2^N shapes validated at `build()`, which is this skill's own target defect
  wearing a respectable name.

## Reporting

State for each finding: the **shape**, the **bound it clears**, whether it is **evidence or
judgement**, and the **remedy as code**. A finding with no polymorphic remedy is a complaint. A
finding that cannot name the bound it clears is not a finding.

Report what the bounds rejected too, with counts. It is the only way a reader can tell a
discriminating instrument from one that flags everything.

## References

- **`references/anti-patterns.md`** — eight ways to do this wrong, each observed in practice.
  **Read before writing any remedy**; §3 and §4 are the two gates above and they are where remedies
  go wrong most often.
- **`references/worked-examples.md`** — three complete before/after remedies at increasing
  difficulty: a plain runtime union, a document/runtime split with a factory, and a policy object
  replacing a laundered mode string.
- **`references/external/`** — read-only copies of two public design skills, kept for comparison and
  provenance only. They are *not* guidance for this skill and need not be read to use it; each carries
  its upstream URL and `npx skills add` command in its header.
- `design-principles` — the sibling skill stating the underlying
  principle abstractly, with its bounds and the parameter-typing principle this one leaves out. Reach
  for it when the question is *"is an abstraction warranted here at all"* rather than *"sweep this
  package"*.

## Improve this skill from use

After completing a task with this skill, reflect on whether its instructions or resources revealed a
gap, ambiguity, stale instruction, avoidable friction, or error. If concrete evidence surfaced, include
a brief `Skill feedback` note in the handoff or final response that names the affected file or section
and proposes the smallest useful correction. Do not invent feedback when no issue surfaced, and do not
edit the skill during an unrelated task without the user's authorization.
