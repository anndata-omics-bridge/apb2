---
name: interview-to-spec
description: >-
  Explicit, opt-in PROSE interview (one question at a time, never the
  multiple-choice picker) that turns a fuzzy idea or rough TODO_ note into a
  TODO_<title>.md planning doc — requirements, design, implementation plan; a
  PLAN ONLY, never code. Trigger ONLY on explicit invocation — two phrase
  clusters: (1) "interview me", "explore this with me", "elicit / elucidate
  requirements", "help me think this through", "co-design this"; (2) "let's
  create a TODO_ for this", "convert this TODO_ into a design / implementation
  plan", "improve the existing TODO_". Do NOT self-activate on vague or
  underspecified requests.
---

# interview-to-spec

Turn a half-formed idea — or a rough `TODO_` note — into a spec the user actually agrees with, by **interviewing them
one question at a time** instead of guessing and building the wrong thing.

This skill produces a **planning document only — it never writes code.** The deliverable is one
`TODO/TODO_<title>.md` with three sections: **requirements, design, implementation plan**. "Implementation plan"
means the ordered steps to take *later*, not work done now.

## When to use

**Explicit and opt-in.** Activate only when the user asks to *interview / explore / elucidate requirements /
co-design*, or to *create, convert, or improve a `TODO_`* document — see the `description` for the exact trigger
phrases.

**Do NOT self-activate** because a request looks vague — if the user just asks for the work, do the work. And note
the key trap: "let's create a `TODO_` for this" means **start interviewing**, not write a file now. The file is what
you land at the **end**, after the conversation.

## The interview loop

This is the core protocol. Repeat it until the picture is clear:

1. **Ask ONE question** — in plain prose, in the message. One decision per turn.
2. **User answers** in their own words.
3. **Confirm understanding** — reflect back what you took from the answer, in a sentence or two, so they can catch a
   misread.
4. **User comments / corrects.**
5. **Draft the next question** off what you just learned, and ask it.
6. Repeat.

### Rules for the questions

- **One at a time — exactly one, never several.** The answer to one question usually reshapes the next, so asking
  ahead wastes turns; worse, a batch forces the user to label which question each reply answers, wasting their time.
- **Never the multiple-choice / single-choice question picker (`AskUserQuestion`) — for ANY question, including
  yes/no gates like "shall I create the file?".** A picker hands the user *your* pre-written answers and forecloses
  everything else — but what looks like a yes/no may draw a comment, a counter-question, or a new direction.
  **Keeping the exchange open is the point of this skill.** Ask in the message; let the user answer however they want,
  in their own words and vocabulary (see below).
- **Give them something to react to.** When you have a recommendation, state it and ask them to confirm or adjust —
  that is faster than an open void. But the recommendation is a draft, not a decision.
- **Confirm before moving on.** Every answer gets a short "here's what I understood" before the next question.

## Preserve the user's vocabulary

The reason prose beats a picker runs deeper than "nuance." A multiple-choice option enforces *your* phrasing; the
user's own answer carries **the terms and synonyms they actually use** for the problem. That vocabulary is signal —
capture it verbatim and carry it through the whole spec:

- **Requirements** — describe the problem in the user's words, not a paraphrase.
- **Design** — name components and concepts with the user's terms so the design reads back to them in their language.
- **Implementation plan** — carry the vocabulary into the *planned* names: the variable, function, and column names
  the steps will create, so the eventual code speaks the user's language instead of forcing a constant translation.

This is especially true in collaborative work like doc-writing, where the right answer is rarely a clean pick from a
list. When the user introduces a term, adopt it; if you must rename it, say so and why.

## The three sections are a center of gravity, not gates

The document grows three sections, and the conversation *drifts* through them in roughly this order:

1. **Requirements** — what and why: the user story, scope, what "done" looks like.
2. **Design** — how: architecture, the alternatives considered, the trade-offs.
3. **Implementation plan** — the ordered steps, files touched, and how it gets tested.

But the flow is **non-linear**: designing surfaces new requirements, planning surfaces design gaps. When that
happens, hop back and revise the earlier section. The sections are a center of gravity, **not gates — there are no
phase checkpoints**, and all three stay live as understanding develops.

## Entry points

- **Greenfield** ("let's create a `TODO_` for X") — no file yet. Start the interview at requirements.
- **Convert** ("turn this `TODO_` into a design / implementation plan") — **read the existing file first**, then
  interview to fill the gaps the target artifact needs.
- **Improve** ("refine the existing `TODO_`") — **read it first**, see which sections are thin or stale, and continue
  the interview from there.

Always read an existing `TODO_` before asking anything, so your questions build on what is already written.

## Producing the artifact

- **Write the file once, at the end.** Hold the whole conversation in context; do **not** write or update the file
  mid-interview — there is no need, and a half-written file just adds noise. The file is the thing you land at
  convergence, not during.
- **Never write silently.** When the picture converges, ask first — e.g. *"Now I have a clear picture — should I
  create `TODO/TODO_<title>.md`?"* — and write only on a yes. Pick a short, descriptive `<title>` — **lowercase,
  words separated by underscores** — and show it in that prompt so the user can correct it.
- **Check before writing.** If `TODO/TODO_<title>.md` already exists, surface it and ask whether to replace, merge,
  or pick a new title — never silently overwrite. (For Convert / Improve, the existing file is the one you update.)
- **If `TODO/` does not exist**, ask whether to create the folder rather than assuming. (If the user would rather not,
  fall back to an in-conversation summary.)

### TODO_ document template

```markdown
# <Title>

> One-line goal.

## Requirements
- User story / what we're actually trying to do (the *why*)
- Scope (in / out)
- Acceptance — what "done" looks like

## Design
- Approach / architecture
- Alternatives considered and why they were set aside
- Trade-offs and open risks

## Implementation plan
- [ ] Ordered steps
- Files / packages touched
- Test strategy

## Open questions
- Anything still unresolved
```

Keep sections that genuinely do not apply short or omit them; do not pad. Mark unresolved items under **Open
questions** rather than inventing answers.

## Convergence

You converge when open questions stop appearing and the three sections hold together — there is no checkpoint that
ends the interview. Signal it ("I think I have a clear picture") and offer to write the file (see *Producing the
artifact* for the mechanics). The user decides when it is done.

## Anti-patterns

- Firing this skill on your own because a request felt vague. It is opt-in.
- Asking several questions in one turn, or using the choice picker.
- Treating "let's make a `TODO_`" as "write a file now" instead of "start interviewing."
- Detailing the implementation-plan section on top of a requirement you have not confirmed — note the dependency and
  loop back; the sections are not gates.
- Writing the `TODO_` file mid-interview, or writing / overwriting it without asking first.
- Skipping the "here's what I understood" confirmation between questions.

## Related

Complements a separate **talking-to-humans** skill (planned — fgcz/skills PR #23) that calibrates *how* a single
reply is phrased. This skill governs the *shape of the conversation*; talking-to-humans the *shape of each message*
inside it.
