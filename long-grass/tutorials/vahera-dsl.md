# The vaHera DSL

**What you'll learn:** vaHera as its own language — all 15 statement forms,
not just the memory/kernel half you've already used in passing. That means
the half you *haven't* used yet: `describe` / `resolve` / `spawn` /
`navigate to penultimate` / `complete trajectory` — the categorical
trajectory statements that give vaHera its name and that no other tutorial
here walks through on their own terms.

**Time:** ~15 minutes.

**Prerequisites:** [Basic routines](./basic-routines) for the terminal
itself (`:modules`, `:audit`, `:clear`) and the memory half of vaHera
(`memory store` / `memory find nearest` / `memory list` / `memory dump`) —
this tutorial does not re-teach those, it picks up where that one stopped.

**Runtime requirement:** none beyond the browser. vaHera lines are typed
directly into the terminal — no `dispatch("vahera", ...)` wrapper, no
module id. The terminal recognizes vaHera syntax and runs it against the
shared kernel directly.

**One rule that shapes every example below:** each terminal cell is its own
independent run of the vaHera interpreter. The kernel's *stored objects*
(everything `memory store` and `memory create at` put there) persist across
cells — that's the same shared kernel the whole session. But `describe` and
`resolve` bindings, and any `spawn`ed process, live only for the one cell
that created them; a later cell has no memory of a target or process an
earlier cell set up. **A full trajectory — `describe` through `complete
trajectory` — has to be one cell, newline-separated, never split across
separate Enters.** This tutorial writes every trajectory that way; treat any
example you see elsewhere that splits `spawn`/`navigate`/`complete` into
separate cells as something to merge into one before you run it.

---

## 0. What vaHera actually is

Everything in [Basic routines](./basic-routines) §3–4 — `memory store`,
`memory find nearest`, `kernel stats` — is real vaHera, but it's the part of
the language that behaves like a vector store: text in, S-coordinate out,
proximity search. That's one of vaHera's two halves.

The other half is what the name is about: a **trajectory** through a
categorical coordinate space, from a fixed initial point to a target,
passing through a *penultimate* state before completing. `describe` names a
target and gives it meaning; `resolve` computes its coordinate; `spawn`
starts a process aimed at that coordinate; `navigate to penultimate` walks
the process backward from the target toward the initial point, stopping one
step short; `complete trajectory` takes that last step and reports what's
nearest to where you landed. Five statements, always in this order, one
process at a time, one cell.

Start clean:

**Cell 0.1**
```
:clear
```

---

## 1. The coordinate: S(k, t, e)

Every target in vaHera lives at a point `S(k, t, e)` — three numbers, each
constrained to `[0, 1]`, standing for **k**nowledge, **t**ime, and
**e**ntropy. You never compute this by hand for text: `describe`/`resolve`
derive it from what you wrote, the same way `memory store` does. You *can*
place a bare anchor at an exact coordinate with `memory create at`, which is
useful for seeing the coordinate space directly before layering meaning on
top of it:

**Cell 1.1**
```
memory create at S(0.2, 1.0, -0.5)
```

**Expected** — this one is deliberately invalid: `e = -0.5` is outside
`[0, 1]`. This is checked when the coordinate is parsed, before it ever
reaches the kernel, so the failure is immediate and precise about which
axis and which value was out of range. The terminal renders a thrown error
in brackets:
```
[SCoord.e=-0.5 outside [0,1]]
```

**Cell 1.2** — the corrected version:
```
memory create at S(0.2, 1.0, 0.5)
```

**Expected** — a confirmation the anchor was allocated, with its ternary
address and tier (`RAM`, `L1`, `L2`, or `L3` — a function of the
coordinate's distance from the origin, nothing to do with actual memory
hierarchy; it's a naming borrowed from the substrate model, not a real
storage tier).

Distance between two coordinates is not Euclidean — it's a Fisher-metric
distance per axis, summed in quadrature. You don't need the formula to use
vaHera, only to know that "nearest" in `memory find nearest` and in
trajectory navigation means this distance, not raw coordinate subtraction.

---

## 2. describe / resolve — naming a target, computing its coordinate

**Cell 2.1**
```
describe SOD1 with "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
resolve SOD1
```

**Expected**
```
describe SOD1 -> S(...)
resolve SOD1 -> S(...)
```
Both lines report the same coordinate — `describe` computes it from the
text (the same embedding `memory store` uses underneath) and binds it to
the bare-identifier target name `SOD1`; `resolve` on an already-described
target just re-reports that binding. Neither statement stores anything
retrievable by `memory find` — the binding exists only for `spawn` to read,
and only within this cell.

**Cell 2.2** — what happens when you `resolve` a target you never
`describe`d, in a fresh cell:
```
resolve CYP2D6
```

**Expected** — this does *not* fail. `resolve` on an unbound target falls
back to embedding the *target name itself* — the literal string `"CYP2D6"`
— rather than raising an error:
```
resolve CYP2D6 -> S(...)
```
That coordinate is real but weak: it was derived from six characters, not
from a real description. This is a genuine sharp edge in the language —
forgetting to `describe` a target doesn't fail loudly, it silently degrades
what `resolve` has to work with. Always `describe` before you `resolve` if
you want the coordinate to mean anything.

---

## 3. spawn / navigate / complete — the trajectory itself, one cell

A **process** is spawned from a resolved target and carries it through the
two remaining trajectory statements. Because the target binding from §2
doesn't survive to a new cell, the whole thing — `describe`, `resolve`,
`spawn`, `navigate to penultimate`, `complete trajectory` — has to be
written as one cell:

**Cell 3.1**
```
describe SOD1 with "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
resolve SOD1
spawn analysis from SOD1
navigate to penultimate
complete trajectory
```

**Expected**
```
describe SOD1 -> S(...)
resolve SOD1 -> S(...)
spawn analysis from SOD1
navigate analysis steps=12
```
...and then, for `complete trajectory`, whatever the kernel finds nearest to
where the trajectory landed. If you ran Cell 1.2 earlier in this session,
that bare anchor is still the only thing in the kernel, so `complete` will
likely report *it* — a `note` result named `SOD1` whose `text` reads the
literal string `"null"`. That's not a bug: `memory create at` allocates an
object with no payload, and the note renderer has nothing better to show
you. It's a real, slightly odd result, and seeing it once is worth more
than being told to expect it. If nothing in the kernel is close enough,
you'll instead get:
```
no categorical match.
```
Either way, `complete trajectory` performs a genuine nearest-neighbor
lookup against whatever the kernel already holds — it never invents a
result, and it can just as easily hand you back an unrelated object that
happens to be nearest as it can report nothing.

A note on `navigate analysis steps=12`: that `12` is the kernel's
configured search depth (the constructor argument in `createRuntimeContext`
— 12, everywhere in this webtool), not a measurement of how far `SOD1`'s
coordinate was from the start. The backward walk always takes the full
configured depth regardless of target position; `steps` reports the
configuration, not the distance covered.

**Cell 3.2** — give `complete trajectory` two real candidates to choose
between, so the "nearest" in "nearest-neighbor" is actually doing work
instead of returning the only object in the kernel by default. Store two
notes on unrelated topics, then run the trajectory in one cell:
```
memory store "sod1_note" = "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
memory store "bread_note" = "sourdough bread recipe, flour water salt yeast"
describe SOD1 with "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
resolve SOD1
spawn analysis from SOD1
navigate to penultimate
complete trajectory
```

**Expected** — `complete trajectory`'s result is the `sod1_note` object:
name, address, coordinate, tier. It's found because the note's text and
`SOD1`'s description text are identical, so their S-coordinates coincide
exactly (S-distance 0), closer than `bread_note` could possibly be.

**Cell 3.3** — same two stored notes, but describe `SOD1` with the *other*
text instead:
```
describe SOD1 with "sourdough bread recipe, flour water salt yeast"
resolve SOD1
spawn analysis from SOD1
navigate to penultimate
complete trajectory
```

**Expected** — `complete trajectory` now returns `bread_note` instead. The
notes never moved; the trajectory's target did, and nearest-neighbor
followed it. This is the whole "nearest" claim made concrete: two objects,
one query text, and the result genuinely depends on what that text says.

---

## 4. process list — inspecting what you've spawned, and its own limit

**Cell 4.1** — run right after Cell 3.3, in a fresh cell:
```
process list
```

**Expected**
```
no processes
```
This is the same cross-cell boundary from the intro, seen from the other
side: the `analysis` process Cell 3.3 spawned lived only inside that cell's
`executeVahera` call. `process list` only ever shows processes spawned in
*its own* cell — which means, in practice, it always reports empty unless
you put it in the same cell as the `spawn` you want it to see:

**Cell 4.2**
```
describe SOD1 with "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
resolve SOD1
spawn analysis from SOD1
process list
```

**Expected** — one entry: `analysis`, state `ready` (spawned but not yet
navigated — `process list` ran before `navigate to penultimate` this time),
target coordinate. `process list` is diagnostic sugar for whatever a single
script did, not a durable session-wide roster.

---

## 5. Aspects — biasing a trajectory without changing the target

A comment line `# aspect: NAME` at the top of a script registers a
retrieval aspect. It's a directive to the *protein-demo* comparison path
specifically (`# aspect: compare:KEY` triggers a two-way protein comparison
in `complete trajectory`'s output when the target resolves against the
built-in protein database) — for plain-text targets like `SOD1` above, an
aspect other than `compare:...` is parsed and recorded but has no visible
effect on the result shape. It's real syntax, worth knowing exists, but not
something a plain-text script needs to reach for:

**Cell 5.1**
```
# aspect: full
describe SOD1 with "superoxide dismutase 1, copper-zinc, antioxidant enzyme"
resolve SOD1
```

**Expected** — identical output to Cell 2.1; the aspect line changes
nothing here.

---

## 6. demon sort / controller verify — the two you've seen, in context

These appeared in [Basic routines](./basic-routines) §4 as kernel
diagnostics; here's what they're actually diagnosing relative to what
you've just built.

**Cell 6.1**
```
demon sort
```

**Expected** — every object currently in the kernel (the `S(0.2,1.0,0.5)`
anchor from Cell 1.2, `sod1_note` and `bread_note` from Cell 3.2), sorted by
distance from the coordinate origin `S(0,0,0)`. This is the "Maxwell-demon
categorical sort" — a total order over everything you've allocated, independent of any
query.

**Cell 6.2**
```
controller verify
```

**Expected** — a triple-equivalence sample count and a rejection count.
Every `memory_create`, `navigate`, and `complete` statement you ran above
was checked against a kernel-level invariant as it happened. Cell 1.1's
rejected coordinate does **not** count toward this rejection total — it
never reached the kernel at all, because `SCoord` validation happens at
parse time, one layer earlier than the kernel's own PVE check.

---

## 7. What you now know

- vaHera has two halves: the memory/proximity half ([Basic
  routines](./basic-routines) §3–4: `memory store`, `memory find nearest`,
  `memory list`, `memory dump`) and the **trajectory** half taught here:
  `describe`, `resolve`, `spawn`, `navigate to penultimate`, `complete
  trajectory`.
- **Trajectory state does not survive across cells.** `describe`/`resolve`
  bindings and any `spawn`ed process exist only for the one cell that
  creates them — write the whole `describe` → `complete trajectory`
  sequence as one multi-line cell. The kernel's stored objects (`memory
  store`, `memory create at`) are the only thing that persists across
  cells in the same session.
- A target's coordinate `S(k, t, e)` is always derived from text via the
  same embedding `memory store` uses — `describe`/`resolve` never take a
  literal coordinate. `memory create at S(...)` is the only statement that
  places one directly, and it validates the `[0, 1]` bound strictly, at
  parse time.
- `resolve` on a target that was never `describe`d does not fail — it falls
  back to embedding the target's bare name, a much weaker signal than real
  description text.
- `complete trajectory` performs a real nearest-neighbor lookup against
  whatever the kernel holds at the time — it returns `no categorical
  match.` rather than fabricating a result when nothing is close.
- All 15 statement forms, the exact grammar, and more worked scripts live
  in the [vaHera reference](../knowledge-packs/vahera/reference.md) — this
  tutorial is the walkthrough; that file is the spec to check when you're
  writing a script and need to know if a form is valid.

**Next up:** [The Complete CKG Experiment](./complete-ckg-experiment) — the
same `describe`/`resolve`/`spawn` shape reappears there as `represent` /
`attach` / `dispatch`, one categorical node holding many modules' facts
instead of one process holding one trajectory — and unlike a vaHera
process, a CKG node *does* persist across cells by design, referenced by
its `tau` name.

---

## Troubleshooting

- **`unknown vaHera: ...`** — the line doesn't match any of the 15 forms
  exactly. Check quoting (double quotes only), check fixed phrases are
  verbatim (`navigate to penultimate`, not `navigate penultimate` or
  `navigate to the penultimate state`), and check the [reference
  grammar](../knowledge-packs/vahera/reference.md).
- **`[SCoord.<axis>=<value> outside [0,1]]`** — a `memory create at
  S(k,t,e)` coordinate had a component outside the valid range. This is a
  hard parse error, not a warning; fix the value and resend.
- **`[spawn: unresolved target NAME]`** — you ran `spawn NAME from TARGET`
  in a cell that never `describe`d or `resolve`d `TARGET` in that *same*
  cell. This is almost always the cross-cell state issue from the top of
  this tutorial: merge the `describe`/`resolve` lines back into the same
  cell as the `spawn`.
- **`[navigate: no active process]` / `[complete: no active process]`** —
  you ran `navigate to penultimate` or `complete trajectory` in a cell that
  never `spawn`ed a process in that same cell.
- **`complete trajectory` always returns `no categorical match.`** — the
  kernel has nothing near the target's coordinate. Store something with
  `memory store` (in the same cell or an earlier one — stored objects do
  persist) whose text is close in meaning to what you're navigating toward.
- **`process list` reports `no processes` right after you just spawned
  one** — you spawned it in a different cell. `process list` only sees
  processes spawned in its own cell.
