# Basic Routines

**What you'll learn:** how to talk to Buhera OS at all — the meta commands
that don't run any module, the vaHera memory DSL, and how to check what
you have available.

**Time:** ~5 minutes.

**Prerequisites:** none. Open the terminal at the home page and follow along.

Each numbered cell is one thing you type into the terminal. Expected output
is shown below each cell. If your output looks materially different, that's
a signal something is wrong (see the troubleshooting notes at the end of
each section).

---

## 0. Where you are

The terminal is the whole page. There's no filesystem, no processes in the
Unix sense, no windows. What there is: a **federation of modules** and an
**append-only audit log** that records every act you take. Every command
you type either dispatches to a module or asks the terminal about its own
state.

Start clean if you've been playing around:

**Cell 0.1**
```
:clear
```

**Expected**
```
(the memory kernel is reset; the audit log stays)
```

---

## 1. Meta commands — the ones that don't run modules

Meta commands start with `:`. They query the terminal's own state rather
than dispatching to any module.

**Cell 1.1** — See the federation.
```
:modules
```

**Expected** — a list of every registered module with its description
and example instructions. You should see at least: `echo`, `vahera`,
`lavoisier`, `purpose`, `zangalewa`, `graffiti`, `purpose-carry`,
`shapeshifter`, `sbs`, `scope`, `catalysts`, `compute`.

**Cell 1.2** — See what's happened so far in this session.
```
:audit
```

**Expected** — a list of act entries with monotone `act_id`, module id,
wall-clock time. If you just cleared, this is nearly empty.

**Cell 1.3** — Load a small demo corpus into vaHera.
```
:tour
```

**Expected** — five sample notes stored, followed by three example
`memory find nearest` queries against them.

**Cell 1.4** — Reset again.
```
:clear
```

---

## 2. The echo module — the smoke test

Echo does nothing but return its input. It's the simplest possible module
and a good sanity check.

**Cell 2.1** — String instruction.
```
dispatch("echo", "hello")
```

**Expected** — the terminal renders the string `hello` back to you and
appends an act to the audit log.

**Cell 2.2** — Structured instruction.
```
dispatch("echo", { kind: "note", body: "buy milk" })
```

**Expected** — the terminal renders the object back, echoed verbatim.

**Cell 2.3** — Confirm the log grew.
```
:audit
```

**Expected** — two `echo` entries, most recent first.

---

## 3. vaHera — the memory DSL

vaHera stores text at a semantic address (a three-number S-coordinate)
computed from the meaning of what you write. Retrieval is by proximity
in that coordinate space, not by keyword.

**Cell 3.1** — Store a note.
```
memory store "meeting" = "team retro on Thursday at 2pm about the Q3 launch"
```

**Expected** — a confirmation card showing the note was stored at some
computed S-coord.

**Cell 3.2** — Store two more.
```
memory store "recipe" = "bread flour, salt, water, yeast — 18-hour cold retard"
memory store "todo" = "buy milk and bread from the corner shop"
```

**Cell 3.3** — Retrieve by meaning, not keyword.
```
memory find nearest "when is the meeting"
```

**Expected** — the `meeting` note comes back first. The query didn't have
the word "meeting" in it, but the S-coordinates line up.

**Cell 3.4** — A different query.
```
memory find nearest "sourdough baking"
```

**Expected** — the `recipe` note comes back first.

**Cell 3.5** — See everything you've stored.
```
list
```

**Expected** — a table of your three notes with their S-coordinates.

**Cell 3.6** — Inspect one entry in detail.
```
dump meeting
```

**Expected** — the full stored object: coordinate, ternary address, tier,
payload.

---

## 4. Kernel diagnostics

**Cell 4.1** — Statistics.
```
kernel stats
```

**Expected** — a count of stored objects per tier of the substrate, PVE
verification counts, and other bookkeeping numbers.

**Cell 4.2** — Trace.
```
kernel trace
```

**Expected** — a sequence of internal events (`CMM.allocate`,
`DIC.retrieve`, `PSS.trajectory`, etc.) the kernel has emitted since boot.

---

## 5. Shorthands

The terminal recognises several plain-English shorthands that route to
vaHera.

**Cell 5.1** — Store using shorthand.
```
store note = "remember to write up the proposal"
```

**Cell 5.2** — Find using shorthand.
```
find "writing" k=2
```

**Cell 5.3** — Dump using shorthand.
```
dump note
```

---

## 6. What you now know

- Every command either targets a module (dispatch) or the terminal itself
  (meta commands starting with `:`).
- `:modules` lists the federation; `:audit` lists your session; `:clear`
  resets the kernel.
- Echo is a no-op; use it to test that dispatch works.
- vaHera stores and retrieves by semantic proximity, computed from
  content. The kernel keeps a running trace you can inspect.
- Shorthands (`store`, `find`, `dump`, `list`, `stats`) are convenience
  aliases for `memory ...` statements.

**Next up:** [Kwasa-kwasa routines](./kwasa-kwasa-routines) — the
orchestrator that composes many dispatches into a script.

---

## Troubleshooting

- **`:modules` shows fewer than 12 entries** — the deployment is on an
  older build. Refresh the page.
- **`memory find nearest` returns nothing** — the kernel was cleared.
  Store a few notes first, then retry.
- **A dispatch returns `ok:false`** — the returned `output_delta.lines`
  usually tell you exactly what's wrong.
