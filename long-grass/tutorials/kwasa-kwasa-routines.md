# Kwasa-kwasa Routines

**What you'll learn:** the orchestrator language (turbulance / kwasa-kwasa).
How to write functions, values, and propositions, and — most importantly —
how to dispatch to modules from inside a script so many modules cooperate
in one act.

**Time:** ~10 minutes.

**Prerequisites:** [Basic routines](./basic-routines).

Kwasa-kwasa is the orchestrator's language. The terminal detects
kwasa-kwasa syntax by the leading keyword (`funxn`, `item`, `proposition`,
`hypothesis`, `point`, `given`, `considering`, `within`, `research`,
`for each`) and routes the line to the turbulance interpreter instead of
to vaHera.

---

## 1. Values and functions

**Cell 1.1** — Define a function.
```
funxn double(x): return x * 2
```

**Expected** — an "ok" acknowledgement. The function is now defined in
this cell's script context.

**Cell 1.2** — Use it.
```
item r = double(21)
print("result: {}", r)
```

**Expected**
```
result: 42
```

**Cell 1.3** — Multi-line, all in one cell (kwasa-kwasa scripts can span
lines when you paste them together).
```
funxn greet(name):
  print("hello {}", name)
  return name
item n = greet("world")
```

**Expected**
```
hello world
```

---

## 2. Propositions and points

Kwasa-kwasa distinguishes *claims* (points) from *stances on claims*
(propositions). Both carry evidence and can compose.

**Cell 2.1** — A single claim.
```
point year: 1868
```

**Expected** — the value `year` is now a point with confidence 1.

**Cell 2.2** — A proposition with a motion (a stance).
```
proposition Greeting: motion Hello("world")
```

**Expected** — the proposition is registered in the script's context.

**Cell 2.3** — Inspect what the script produced.
```
item _ = 0
```

The trailing `_` receives nothing — but the whole script's return value
(the last `item`, `point`, or `proposition`) is shown in the artifact.

---

## 3. Dispatching to modules from a script

This is the load-bearing feature of kwasa-kwasa. Any registered module in
`:modules` can be called from inside a script.

**Cell 3.1** — Call echo from a script.
```
item out = dispatch("echo", "hello federation")
print(out.output_delta.value)
```

**Expected**
```
hello federation
```

Now check `:audit` — you'll see the echo dispatch recorded.

**Cell 3.2** — Route a vaHera statement through the orchestrator.
```
item stored = dispatch("vahera", "memory store \"n\" = \"hi\"")
item found  = dispatch("vahera", "memory find nearest \"hi\" k=1")
```

Same effect as typing the vaHera statements directly, but now they
compose in a script.

**Cell 3.3** — Chain two modules.
```
item ms = dispatch("lavoisier", "demo")
print("lavoisier returned {} records", ms.output_delta.summary.count)

item p = dispatch("purpose-carry", { kind: "stats" })
print("purpose session has {} steps", p.output_delta.stepCount)
```

**Expected**
```
lavoisier returned 135 records
purpose session has 3 steps
```

The purpose-carry step count includes every dispatch you've made — it's
running as a post-dispatch feeder. If this is your first session, the
count should be at least the number of dispatches you've made so far.

---

## 4. Conditionals and iteration

**Cell 4.1** — `given` (conditional evaluation).
```
item x = 7
given x > 5:
  print("big")
```

**Expected**
```
big
```

**Cell 4.2** — `for each`.
```
for each name in ["alice", "bob", "carol"]:
  print("hi {}", name)
```

**Expected**
```
hi alice
hi bob
hi carol
```

**Cell 4.3** — Combine with dispatch.
```
for each q in ["writing", "shopping", "meeting"]:
  item r = dispatch("vahera", "memory find nearest \"" + q + "\" k=1")
  print("q: {}", q)
```

**Expected** — three iterations, each dispatching a vaHera search.

---

## 5. Composition with purpose-carry

Now the interesting flow: dispatch some acts, then ask purpose-carry which
of them are load-bearing for a goal.

**Cell 5.1** — Reset for a clean run.
```
:clear
```

Also reset the purpose session:
```
dispatch("purpose-carry", { kind: "reset" })
```

**Cell 5.2** — Populate the audit log with three thematically different
acts.
```
dispatch("vahera", "memory store \"tum\" = \"Technical University of Munich founded in 1868\"")
dispatch("vahera", "memory store \"aime\" = \"AIMe Registry for Artificial Intelligence at TU Munich\"")
dispatch("vahera", "memory store \"milk\" = \"buy milk from the corner shop\"")
```

**Cell 5.3** — Ask purpose-carry what's necessary for a goal about TUM.
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: ["tum", "aime"],
  budget: 500
})
print("keep: {}", c.output_delta.keep)
print("dropped: {}", c.output_delta.dropped)
print("ambient floor β = {}", c.output_delta.ambientFloor)
```

**Expected**
```
keep: [act-1, act-2]
dropped: [act-3]
ambient floor β = 5
```

The TUM and AIMe acts are on-topic; the milk act is free-dropped.

---

## 6. What you now know

- Kwasa-kwasa (turbulance) is the orchestrator language. It detects itself
  by leading keyword.
- `funxn`, `item`, `point`, `proposition`, `given`, `for each` are the
  core forms.
- `dispatch("module-id", instruction)` is a first-class builtin. Every
  module in the federation is callable from a script.
- Scripts compose: you can chain `dispatch` calls, use `for each` to run
  many, feed results between them.
- Purpose-carry is fed automatically by every dispatch, so scripts have
  a "context awareness" as a built-in property, not something you must
  wire manually.

**Next up:** [Purpose routines](./purpose-routines) — the tandem carry in
depth, the theory behind the automatic session-feeder.

---

## Troubleshooting

- **The script errors on the first line** — check the leading keyword.
  If it doesn't start with a kwasa-kwasa keyword, the router treats it as
  a vaHera or plain-text input. Prefix with `item _ = ...` or start with
  `funxn`.
- **`dispatch(...)` says "unknown module"** — check `:modules` for the
  exact spelling of the module id.
- **`print("... {}", ...)` renders literally** — the `{}` substitution
  requires positional arguments after the format string. Use `+` for
  simple string concatenation instead.
