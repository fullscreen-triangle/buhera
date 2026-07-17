# Purpose Routines

**What you'll learn:** the tandem carry — Buhera's answer to the question
"as this session grows, what context is actually necessary for the next
step?" The answer is not "everything"; it's a small load-bearing subset
computed from the running history.

**Time:** ~10 minutes.

**Prerequisites:** [Basic routines](./basic-routines) and
[Kwasa-kwasa routines](./kwasa-kwasa-routines).

The theory is in `docs/sources/purpose-propagation.tex`. This tutorial is
the runtime side of the same paper: floor, residue, seek, necessity,
knapsack. You'll see each of them as an actual number after a few
dispatches.

---

## 0. The mental model in one paragraph

Your session accumulates dispatched acts. Every act has *content* (which
is regeneratable — you can always look at the payload again) and *residue*
(what specific distinction the act draws that a later act might depend on).
Content is disposable. Residue is not. When a new goal arrives, purpose
computes which past acts have residue the goal actually reaches (that's
the necessity operator), fits them under a token budget (the knapsack),
and drops the rest for free. **You carry the uncertainty, not the
knowledge.**

---

## 1. Session stats — the baseline

Every dispatch you make automatically feeds a Step into the purpose
session via a post-dispatch hook wired into the registry. Start clean and
watch the count.

**Cell 1.1** — Reset both the vaHera kernel and the purpose session.
```
:clear
dispatch("purpose-carry", { kind: "reset" })
```

**Expected** — two confirmations.

**Cell 1.2** — Check baseline stats.
```
dispatch("purpose-carry", { kind: "stats" })
```

**Expected**
```
purpose-carry: 0 steps in session
ambient floor β = 0.000
```

The floor is 0 because there are no shared-term edges yet (nothing to draw
edges between).

---

## 2. Populating the audit log

**Cell 2.1** — Store three thematically different notes.
```
memory store "note1" = "Technical University of Munich founded in 1868"
memory store "note2" = "AIMe Registry for Artificial Intelligence is at TU Munich"
memory store "note3" = "buy milk and bread from the corner shop"
```

**Expected** — three stored confirmations.

Each `memory store` dispatch was fed into the purpose session
automatically. Verify:

**Cell 2.2**
```
dispatch("purpose-carry", { kind: "stats" })
```

**Expected**
```
purpose-carry: 3 steps in session
ambient floor β = 5.000
```

The floor is positive now — the tandem-carry paper's Floor Theorem says
this MUST hold when the graph has any shared-term edges. Two of your notes
(note1 and note2) share the term "munich," which gives one edge of weight
1 in the raw graph, scaled by the default weight function to a positive
value. The floor is the minimum positive edge weight.

---

## 3. The sufficient test — the demo the paper is written around

**Cell 3.1** — Ask purpose-carry which of the three notes are necessary
for a goal about TUM and AIMe.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: ["tum", "aime"],
  budget: 500
})
```

**Expected** — a rich card showing:

```
session: 3 steps · floor: 5.000 · budget: 500
goal: tum, aime

keep: 2 · regenerable: 0 · dropped: 1
cost 22/500 (remaining 478), relaxation gap 0.023

[show breakdown ▾]
```

Click "show breakdown" and you'll see:

- **keep**: `act-1` (residue X), `act-2` (residue Y) — the TUM and AIMe notes
- **dropped**: `act-3` — the milk note

That's the paper's Free Drop Theorem: purpose-carry has decided the milk
note is not load-bearing for the goal and can be free-dropped from the
carried state.

---

## 4. Free-text goals

You don't need to hand-list terms. Pass a string and purpose-carry uses
the τ extractor in `src/lib/purpose-terms.js` (lowercase, alphanumeric,
drop stopwords, dedupe).

**Cell 4.1** — Same query, free-text.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: "when was Technical University Munich founded"
})
print("keep: {}", c.output_delta.keep)
```

**Expected** — a similar `keep` set to Cell 3.1. The extractor produced
terms like `technical`, `university`, `munich`, `founded` from your goal
string.

**Cell 4.2** — A goal that doesn't reach any of your notes.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: "photosynthesis in higher plants"
})
print("keep: {}", c.output_delta.keep.length)
print("dropped: {}", c.output_delta.dropped.length)
```

**Expected**
```
keep: 0
dropped: 3
```

Nothing in your session touches photosynthesis — everything is free to
drop. That's not a failure; it's the paper's honest-decline stance.

---

## 5. Explicit step registration

Sometimes you want to feed the session a step that didn't come from a
dispatch — a note, a paper title, a claim you want to be reachable.

**Cell 5.1**
```
dispatch("purpose-carry", {
  kind: "add",
  id: "manual1",
  content: "the Apollo 11 mission launched on 16 July 1969"
})
```

**Expected**
```
purpose-carry: added step "manual1" with 4 terms.
session now holds 4 steps.
```

**Cell 5.2** — Query for it.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: "apollo lunar mission"
})
print("keep: {}", c.output_delta.keep)
```

**Expected** — `manual1` in the keep set.

---

## 6. The knapsack under a tight budget

**Cell 6.1** — First check current session size.
```
dispatch("purpose-carry", { kind: "stats" })
```

**Cell 6.2** — Ask with an artificially tight budget.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: "tum munich university aime registry",
  budget: 20
})
print("kept cost: {}", c.output_delta.diagnostics.totalKeptCost)
print("budget: {}", c.output_delta.diagnostics.budgetRemaining)
print("relaxation gap: {}", c.output_delta.diagnostics.knapsackRelaxationGap)
```

**Expected** — a small `keep` set because most necessary steps didn't fit.
The relaxation gap is bounded by `cost_max / budget` (the paper's
Theorem 6.2); with budget = 20 and typical step costs of ~15, you'll see
a gap around 0.7.

---

## 7. Reset

**Cell 7.1**
```
dispatch("purpose-carry", { kind: "reset" })
```

**Expected**
```
purpose-carry: session reset.
```

The vaHera kernel still holds your notes — reset the kernel too if you
want a fully clean slate:
```
:clear
```

---

## 8. What you now know

- Every dispatch you make automatically becomes a Step in the purpose
  session — this is the post-dispatch feeder, wired once in the terminal.
- `purpose-carry` produces a partition (keep / regenerable / dropped) for
  any goal, respecting a token budget.
- The floor β is a positive number when the graph has any shared-term
  edges — that's the paper's Floor Theorem in action.
- The relaxation gap is bounded by `cost_max / budget`, per the paper's
  knapsack theorem.
- `add`, `stats`, `reset`, `carry` are the four instruction kinds. All
  others (`load`, `save`) are future extensions.

**Next up:** [Zangalewa routines](./zangalewa-routines) — the natural-
language interceptor. It sits above the federation and picks the right
module for a user's utterance.

---

## Troubleshooting

- **`ambient floor β = 0`** — nothing in the session shares any terms.
  Store notes with overlapping vocabulary and try again.
- **`stats` shows fewer steps than dispatches you've made** — some of your
  dispatches had empty term sets after τ extraction (e.g. an echo of a
  single stopword). Those don't produce a Step.
- **`carry` returns `ok:false, error: empty-goal`** — the goal string was
  too short after τ. Provide more content, or pass an explicit term array.
