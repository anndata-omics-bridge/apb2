# Three worked examples, easiest first

Each shows the branch set, the bounds it clears, and the complete remedy. They are ordered by how
much architecture is involved, which is the axis that actually makes these hard — not the number of
arms.

---

## 1. Plain runtime union — no architecture, just a method

**The easiest case, and the one to reach for when explaining the idea.** The types are already plain
frozen dataclasses in the same module as the functions branching on them, so nothing about layers or
factories arises.

Four functions in one module each ask *where does this modification sit?* and answer it per type:

```python
type ModificationLocation = ResidueLocation | TerminalLocation | TerminalOnlyLocation | UnlocalizedLocation

def _record_unknown_token(location: ModificationLocation, sequence_length: int, ...) -> None:
    if isinstance(location, ResidueLocation):
        unknown_tokens[location.sequence_index] = raw_token
    elif isinstance(location, TerminalLocation | TerminalOnlyLocation):
        index = -1 if location.position == "N-term" else sequence_length
        unknown_tokens[index] = raw_token
```

**Shapes:** 2 (four functions, one case set) and 6.
**Bounds cleared:** B1 — own types, defined in the same file. B2 — four consumer sites. The `if` asks
*what it is*.

The question all four arms answer is *which token index is this?*, so that is the method:

```python
@dataclass(frozen=True, slots=True)
class ResidueLocation:
    residue: str
    sequence_index: int
    def token_index(self, sequence_length: int) -> int:
        return self.sequence_index

@dataclass(frozen=True, slots=True)
class TerminalLocation:
    position: Literal["N-term", "C-term"]
    def token_index(self, sequence_length: int) -> int:
        return -1 if self.position == "N-term" else sequence_length

# every call site, no branch
unknown_tokens[location.token_index(sequence_length)] = raw_token
```

Adding a fifth location kind becomes one class the type checker forces you to complete, instead of
four functions nothing forces you to find.

---

## 2. A document type — where the method must *not* go

**The case that trips every automated attempt.** Same shape as above, but the type is a pydantic
model loaded from a config file, so putting `coerce()` on it fuses the file format with the
computation.

```python
class LayerDocument(RuleModel):                 # pydantic — describes what may appear in the TOML
    name: str
    source: str
    encoding_mode: EncodingMode = "numeric"
    categories: dict[str, int] = Field(default_factory=dict)
    missing_values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_encoding(self) -> LayerDocument:
        if self.encoding_mode == "factor" and self.missing_values:
            raise ValueError("'missing_values' is only valid for numeric layers.")   # ← shape 1 tell
```

and two consumers, in different modules, with the same three-arm chain:

```python
if layer.encoding_mode == "factor":            # module A          # module B has this verbatim
    return encode_factor(series, layer.categories)
if isinstance(layer.value_pattern, RegexValuePattern):
    return coerce_regex_numeric(...)
return coerce_numeric(series, layer.missing_values)
```

**Gate 1** — the question is *"coerce this column to numbers"*. Nameable, so continue.
**Gate 2** — **it is a storage schema.** So the remedy is a runtime type plus a factory, and
`_validate_encoding` **stays exactly where it is**: someone can write that combination in a TOML
file and must be told. It is validating input, which is its job.

```python
# rules/ — document unchanged, validator unchanged.

# converters/ — runtime. No mode field: the class is the mode.
@dataclass(frozen=True, slots=True)
class FactorLayer:
    name: str
    source: str
    categories: Mapping[str, int]              # non-empty by construction, not by validator
    def coerce(self, series: pd.Series) -> pd.Series:
        return encode_factor(series, self.categories)

@dataclass(frozen=True, slots=True)
class NumericLayer:
    name: str
    source: str
    missing_values: tuple[float, ...]
    values: PlainNumericValues | RegexNumericValues     # the pattern branch, also split
    def coerce(self, series: pd.Series) -> pd.Series:
        return self.values.coerce(series, self.missing_values)

type LayerReader = FactorLayer | NumericLayer

def make_layer(document: LayerDocument) -> LayerReader:      # the one place the flag is read
    ...
```

Both consumers become `layer.coerce(df[layer.source])`.

**What this buys, precisely.** The validator still catches the bad TOML, and downstream the
combination is **unrepresentable** — `FactorLayer` has no `missing_values` field. Both copies of the
chain vanish, and the regex compiles once at construction time instead of per column per run.

**The trap this example exists to prevent:** adding `coerce()` directly to `LayerDocument` removes
the same two branches and looks like a clean win. It also puts pandas in the schema module and
points a dependency the wrong way. Same branch count, worse architecture.

---

## 3. A laundered mode string — the branch you cannot grep for

**Hardest to see, because the discriminator never appears as a type.** One function converts a mode
into a string; a second converts the string back into behaviour.

```python
def _aggfunc_for(rule: ParseRule) -> str:               # mode  → "sum" | "first"
    mode = rule.axis.duplicates.mode
    if mode == "aggregate":
        return "sum"
    if mode == "keep_all_as_raw_table":
        raise NotImplementedError("not yet supported")
    return "first"

def _build_matrix(..., aggfunc: str) -> Matrix:         # string → behaviour
    if aggfunc == "sum":
        ...
    else:  # "first"
        ...
```

Plus four more sites elsewhere branching on `duplicates.mode` directly — six in total, two modules.

**Shapes:** 5 (the laundering) and 2 (the case set).
**B3 checked:** something *does* branch on the string at the far end (`_build_matrix`), so it is a
discriminator, not a label. Had nothing branched on it, this would be duplication and not a finding.

The question all six sites ask is *how do several values that land in one cell become one value?*

```python
@dataclass(frozen=True, slots=True)
class ErrorOnDuplicates:
    """Repeated keys are a rule error: combining is never permitted."""
    def combine(self, cells: CellContributions) -> Matrix:
        cells.raise_if_any_repeated()          # the error belongs here, not in a separate pre-pass
        return cells.single()

@dataclass(frozen=True, slots=True)
class SumDuplicates:
    def combine(self, cells: CellContributions) -> Matrix: ...     # the `aggfunc == "sum"` arm

@dataclass(frozen=True, slots=True)
class KeepFirstDuplicate:
    def combine(self, cells: CellContributions) -> Matrix: ...     # the `else` arm

type DuplicatePolicy = ErrorOnDuplicates | SumDuplicates | KeepFirstDuplicate

_BY_MODE: dict[DuplicateMode, DuplicatePolicy] = {
    "error": ErrorOnDuplicates(), "aggregate": SumDuplicates(), "keep_first": KeepFirstDuplicate(),
}                                              # "keep_all_as_raw_table": absent on purpose

def policy_for(duplicates: Duplicates) -> DuplicatePolicy:
    policy = _BY_MODE.get(duplicates.mode)
    if policy is None:
        raise NotImplementedError(f"duplicates.mode={duplicates.mode!r} is not yet supported")
    return policy
```

**Three things to notice, because each is a rule in itself.**

**No method returns `"sum"` or `"first"`.** An `aggfunc() -> str` accessor would recreate the exact
defect — a type converted to a token, passed across a boundary, branched on again. Return the
result, not the name of the operation.

**`ErrorOnDuplicates` raises, and that is its entire implementation.** It has no aggregation because
"error" *means* combining is not permitted, so the error belongs inside `combine` at the moment a
second contributor appears. That collapses two near-identical pre-pass functions into one arm.

**The missing `_BY_MODE` entry is the `NotImplementedError`.** `keep_all_as_raw_table` fails once, at
construction, naming the mode — instead of from inside a dispatch chain partway through a
conversion.

**One behaviour change to declare rather than hide:** the original pre-pass raised whenever duplicate
*keys* existed, even when the rows were all-null and would never have combined. Raising from inside
`combine` fires on duplicate *contributing values*. Arguably the better rule — it is what the mode
means — but it is a change, and a refactor claiming to preserve behaviour must say so and cover it
with a test.
