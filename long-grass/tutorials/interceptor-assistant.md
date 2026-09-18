# Interceptor: The AI Code Assistant

**What you'll learn:** the `interceptor` module — an AI assistant that turns
a plain-English description into a runnable Rust or TypeScript program,
executes it in a sandboxed subprocess, and hands the result to two other
pieces of the federation: vaHera, which stores every run's captured console
output as a searchable memory, and wind-tunnel, a stability-testing pass
that runs the program several times and reports whether it behaves the
same way each time. Three roles, one dispatch: generate, run, monitor,
test.

**Time:** ~35 minutes.

**Prerequisites:** [Basic routines](./basic-routines) for `dispatch(...)`
itself. [vaHera DSL](./vahera-dsl) helps for §5 (recalling stored run
output), though this tutorial explains everything it needs from vaHera
inline — you do not need to have written a `memory store` command by hand
before. Independent of spraypaint, graffiti, and the DSL-writer module.

**Runtime requirement:** a working deployment with Rust's `rustc` on
`PATH` (for Rust generation/execution — checked by
`/api/interceptor-run`; a clear 503-shaped error is returned if the
`interceptor-run` CLI binary, built from `crates/interceptor-run` at the
buhera repo root via `cargo build -p interceptor-run`, isn't found) and
Node 24+ (for TypeScript execution, via Node's built-in
`--experimental-strip-types` — no `tsx`/`ts-node` dependency needed). Code
*generation* additionally needs a working LLM provider (`OLLAMA_URL`,
`GEMINI_API_KEY`, `HUGGINGFACE_API_KEY`, or `OPENAI_API_KEY` in
`.env.local`); code *execution* and *testing* need none of that — you can
run and wind-tunnel-test hand-written code with no LLM configured at all.
Every cell and its "Expected" block below was run for real against this
exact codebase before being written down — nothing here is illustrative,
including the failures.

---

## 0. Three roles, one module

`interceptor` is not one thing doing three jobs — it's three existing
capabilities wired together, and understanding the seam between them
matters more than memorizing the dispatch syntax:

- **Generation** (`/api/interceptor-generate`) asks whichever LLM provider
  is configured (via the same `llm-cascade.js` cascade every other
  generative module in this federation uses — Ollama, then Gemini, then
  HuggingFace, then OpenAI, first one configured and reachable wins) to
  write one complete, runnable program for a task description. No
  compiler-in-the-loop repair step, unlike the [DSL writer
  module](./vahera-dsl) — general Rust and TypeScript have no single
  authoritative grammar to validate against and repair toward, so a bad
  generation surfaces as a failed *run* instead of a generation error.
- **Execution** (`/api/interceptor-run`) takes any Rust or TypeScript
  source — generated or hand-written, the module does not care which —
  and runs it in a locked-down subprocess: a wall-clock timeout, a 2 MiB
  cap per output stream, no network isolation beyond what the language
  runtime itself provides. Rust goes through a small dedicated CLI
  (`crates/interceptor-run`, built once with `cargo build -p
  interceptor-run`) that compiles with `rustc` fresh per call and runs the
  result; TypeScript goes straight through `node
  --experimental-strip-types`, so nothing needs installing beyond Node
  itself.
- **Monitoring and testing** are the two consumers of execution's output.
  vaHera stores every run's combined stdout+stderr as a named memory —
  the same `memory store`/`memory find nearest` machinery [vaHera
  DSL](./vahera-dsl) covers in full — so a run from ten minutes ago is
  recallable by describing what it printed, not by remembering which cell
  you ran it in. wind-tunnel runs the program several times and checks
  whether every run produced the same output; a program that always
  prints the same thing is "Phase-locked," one that never agrees with
  itself twice is "Turbulent."

Four instruction shapes reach these, and you'll use every one in this
tutorial (illustrative only — not a runnable cell; each shape gets a real
worked example starting in §1):

```text
dispatch("interceptor", { mode: "generate", language, task })
dispatch("interceptor", { mode: "run", language, code })
dispatch("interceptor", { mode: "test", language, code, runs? })
dispatch("interceptor", { mode: "assist", language, task, runs? })
```

`assist` is the one-shot path — generate, then run, then store in vaHera,
then wind-tunnel-test, as a single act. A plain string instruction (no
object) is sugar for `{ mode: "assist", language: "typescript", task:
<your string> }` — the shortest possible way to ask for something.

---

## 1. Running code you already have

Start with execution alone, no LLM involved, so you can see exactly what
the sandbox does before layering generation on top of it.

**Cell 1.1**
```
dispatch("interceptor", { mode: "run", language: "typescript", code: "console.log(2 + 2);" })
```

**Expected**
```
interceptor run   typescript

console output   ● ok   exit 0   214 ms
4

stored in vaHera as "interceptor-run-1-run" — recall it with memory find nearest
```

The renderer shows three things stacked: the code you sent (collapsible —
click "generated typescript ▾" to fold it), the console panel (stdout in
gray, stderr in red if there is any, a green "● ok" or red "● failed"
badge, exit code, wall-clock time), and a line telling you which vaHera
memory now holds this run's output. That storage happens automatically —
`run` mode always stores, whether or not you asked for it — because
"vaHera monitors the console output" is the module's job description, not
an opt-in.

Now the same call in Rust, to see the other execution path:

**Cell 1.2**
```
dispatch("interceptor", { mode: "run", language: "rust", code: "fn main() { println!(\"hello from rust\"); }" })
```

**Expected**
```
interceptor run   rust

console output   ● ok   exit 0   710 ms
hello from rust
```

Notice the timing difference — 710ms against TypeScript's 214ms. That gap
is `rustc` compiling the snippet fresh (there's no shared build cache
across sandboxed calls, deliberately — each run is isolated from every
other). TypeScript's path has no compile step at all; Node's
`--experimental-strip-types` flag strips type annotations at parse time
and runs the result immediately.

---

## 2. When the code fails

A sandbox that only shows you success cases is lying about the other half
of what "execution" means. Three distinct failure shapes exist, and the
module reports each one honestly rather than collapsing them into one
generic error.

**Cell 2.1** — a runtime exception:
```
dispatch("interceptor", { mode: "run", language: "typescript", code: "throw new Error('boom');" })
```

**Expected**
```
interceptor run   typescript

console output   ● failed   exit 1   139 ms
(no stdout)

Error: boom
    at Object.<anonymous> (C:\...\interceptor-ts-ZGnlNl\snippet.ts:1:7)
    ...
```

The stderr panel (red border, red text) carries Node's full stack trace,
including the temp file path the sandbox wrote your code to — that path
is real and expected to appear; the sandbox writes each run to a fresh
temp file rather than `eval`-ing a string in-process, so a stack trace
naturally points there instead of to a "real" source file. This is not a
leak worth worrying about — every run gets a throwaway directory, deleted
immediately after the run completes, win or lose.

**Cell 2.2** — a Rust compile error (a different failure stage entirely —
this program never ran at all):
```
dispatch("interceptor", { mode: "run", language: "rust", code: "fn main() { let x: i32 = \"nope\"; }" })
```

**Expected**
```
interceptor run   rust

console output   ● failed   ...
error[E0308]: mismatched types
 --> ...snippet.rs:1:26
  |
1 | fn main() { let x: i32 = "nope"; }
  |                    ---   ^^^^^^ expected `i32`, found `&str`
  ...
```

The raw `rustc` diagnostic lands in stderr verbatim — the sandbox doesn't
parse or reformat compiler errors, so what you see is exactly what you'd
get running `rustc` yourself. The underlying `output_delta.stage` field
(not shown in the panel, but present in the raw result) reads `"compile"`
here versus `"run"` for cell 2.1 — useful if you're ever inspecting
results programmatically rather than reading the rendered panel.

**Cell 2.3** — a timeout:
```
dispatch("interceptor", { mode: "run", language: "typescript", code: "while(true){}", timeoutMs: 1500 })
```

**Expected**
```
interceptor run   typescript

console output   ● failed   timed out
(no stdout)
```

No exit code (the process was killed, not exited), no stderr — the
program simply ran out of time. The default timeout is 10 seconds if you
don't pass `timeoutMs`; this cell shortens it to 1.5s so the tutorial
doesn't make you wait. Every mode accepts `timeoutMs`, capped server-side
at 30 seconds regardless of what you ask for — a sandbox with no upper
bound isn't a sandbox.

---

## 3. Generating code from a description

This is the part that needs an LLM provider configured. Ask for
TypeScript first:

**Cell 3.1**
```
dispatch("interceptor", { mode: "generate", language: "typescript", task: "print the first 10 fibonacci numbers, one per line" })
```

**Expected, when a provider is configured and reachable:** a generated
program in the code panel, with the provider/model that wrote it named
above it (e.g. `interceptor generate  ollama/llama3.2:3b`).

**Expected right now, on this deployment, honestly:**
```
{"ok":false,"language":"typescript","error":"openai HTTP 401","stage":"upstream"}
```

`OLLAMA_URL` is configured but nothing is listening there in this
deployment; the cascade falls through to Gemini and OpenAI, both of which
have keys configured that don't currently authenticate (`gemini` and
`openai` both return `HTTP 401` — confirmed by testing them directly,
independent of anything this module does). This mirrors [Spraypaint's own
honest §5](./spraypaint-search#5-the-internet-half) about its web-search
call failing the same way, for the same underlying reason: a stale or
misconfigured key, not a bug in the calling module. A module that hides
this behind a generic "something went wrong" would be worse than one that
names the exact upstream and status code, so that's what you get.

If you're reading this after a provider is fixed, cell 3.1 should instead
return real generated code — try it with `language: "rust"` too; the
system prompt for Rust generation constrains the model to standard-library
code only (no external crates), because the execution side compiles with
bare `rustc`, not `cargo`, so a generated `use rand::...` would fail to
compile through no fault of the sandbox.

---

## 4. Wind-tunnel: does it behave the same way twice?

Named after [fullscreen-triangle/wind-tunnel](https://github.com/fullscreen-triangle/wind-tunnel),
a Rust testing framework built on a specific claim: testing individual
functions in isolation (unit tests) or checking properties of individual
inputs (property-based tests) both miss *emergent* behavior — how a whole
call-cycle behaves when it's actually run, not how each piece behaves
alone. The full framework measures this with a borrowed-from-physics
vocabulary: semantic entropy, holonomy (how far a cycle's actual behavior
deviates from its declared spec), and a Kuramoto-style order parameter
measuring how "in sync" an ensemble of runs are with each other.

interceptor's wind-tunnel pass ports the *idea*, not the binary — there is
no existing indexed repository to run the real `wt` CLI against here, only
a program you just generated. The "cycle" becomes one program, run
repeatedly; the "declared spec" becomes its first successful run, treated
as the reference every later run is checked against; holonomy becomes the
fraction of output lines that differ from that reference; the order
parameter becomes the fraction of runs that matched it exactly.

**Cell 4.1** — a deterministic program, tested 3 times:
```
dispatch("interceptor", { mode: "test", language: "typescript", code: "console.log('stable output');", runs: 3 })
```

**Expected**
```
wind-tunnel test   typescript · 601 ms

wind-tunnel stability — 3 runs
Phase-locked        R = 100%
all runs produced identical output
```

**Cell 4.2** — the same test against a program that is deliberately
*not* deterministic:
```
dispatch("interceptor", { mode: "test", language: "typescript", code: "console.log(Math.random());", runs: 3 })
```

**Expected**
```
wind-tunnel test   typescript · 643 ms

wind-tunnel stability — 3 runs
Desynchronized       R = 33%
most runs disagree
deviations: run 1 (holonomy 0.50), run 2 (holonomy 0.50)
```

The order parameter R landed at 33% (1 of 3 runs — the reference run
itself — "matched," and by definition it always matches itself). The
regime name comes from a fixed threshold table: R ≥ 95% is
"Phase-locked," ≥ 75% "Synchronized," ≥ 45% "Partially-locked," ≥ 15%
"Desynchronized," anything lower "Turbulent." This is the honest signal
wind-tunnel exists to surface: a program that calls `Math.random()` (or
reads the clock, or depends on iteration order of an unordered
collection) is not a bug by itself, but if you *expected* deterministic
output and got "Desynchronized" back, that's the test doing its job.

**Cell 4.3** — Rust, to confirm the same pass works across both
languages (this one recompiles three times, so expect it to take
noticeably longer than the TypeScript cells):
```
dispatch("interceptor", { mode: "test", language: "rust", code: "fn main() { println!(\"deterministic\"); }", runs: 3 })
```

**Expected**
```
wind-tunnel test   rust · 1307 ms

wind-tunnel stability — 3 runs
Phase-locked        R = 100%
all runs produced identical output
```

A note on how the runs are scheduled: wind-tunnel runs are sequential, not
parallel, on purpose — a Rust run compiles fresh every time with no shared
build cache, and running several `rustc` invocations concurrently against
temp files in the same naming pattern would compete for resources in a
way that makes timing (and therefore truncation/timeout behavior)
unpredictable. `runs` is capped at 7 server-side; ask for more and you get
7.

---

## 5. Recalling a run's output through vaHera

Every `run` and `assist` call stores its combined stdout+stderr in vaHera
memory under a name like `interceptor-run-<n>-<mode>`. This is what "vaHera
monitors the output" means concretely: the output doesn't just render once
and disappear — it becomes a memory you can search back to by describing
what it said, the same way [vaHera DSL](./vahera-dsl) teaches you to
recall anything else you've stored.

Re-run cell 1.1 if you haven't already this session (its output — `"4"` —
is what you're about to search for), then:

**Cell 5.1**
```
memory find nearest "the number four" k=3
```

**Expected** — vaHera's nearest-neighbor search over everything stored
this session, which by now includes every interceptor run's output
alongside anything else you've stored by hand. The stored memory named
`interceptor-run-1-run` (holding the text `"4"`) should surface among the
top matches, exactly like any hand-stored memory would — vaHera doesn't
distinguish "memories a module wrote" from "memories you wrote with
`memory store` yourself." That's the whole point of routing the output
through vaHera rather than a bespoke run-history table: one recall
mechanism, already documented, already understood, now doing double duty.

If cell 1.1's output feels too short a memory to search meaningfully
(a single digit doesn't give BM25-style relevance much to work with),
try it again with something more textured:

**Cell 5.2**
```
dispatch("interceptor", { mode: "run", language: "typescript", code: "console.log('inventory check: 42 widgets in stock, 3 backordered');" })
```
then
```
memory find nearest "how many widgets are backordered" k=3
```

**Expected**: the run's own console output — the widgets line — as the
top or near-top hit. This is the pattern worth internalizing: you don't
need to remember *which cell* printed something, only *roughly what it
said*.

---

## 6. Everything at once: `assist`

Sections 3 through 5 walked the pipeline one stage at a time so each
piece's job was visible on its own. In practice, the shape you'll actually
reach for is `assist` — generate, run, vaHera-store, wind-tunnel-test, as
one dispatch:

**Cell 6.1**
```
dispatch("interceptor", { mode: "assist", language: "typescript", task: "print the squares of 1 through 5" })
```

or, using the plain-string sugar (identical to the above, with
`language: "typescript"` implied):

**Cell 6.1 (equivalent)**
```
dispatch("interceptor", "print the squares of 1 through 5")
```

**Expected, when a provider is configured:** one combined panel — the
generated code, the console output from running it, the vaHera memory
name it was stored under, and (only if the run succeeded — a program that
never produced output has nothing meaningful to test for stability) the
wind-tunnel stability readout underneath.

**Expected right now, on this deployment, honestly:** the same generation
failure as §3 — `assist` calls the same `/api/interceptor-generate` route
internally, so a misconfigured LLM provider fails `assist` at the same
first step. The panel shows the generation error and stops there; it does
not attempt to run or test code that was never produced. This is
deliberate — `assist` fails fast at whichever stage breaks first, rather
than silently skipping stages and rendering a misleadingly-partial result.

If you want to see `assist`'s full four-stage panel without depending on
generation working, you can exercise the same downstream stages directly
by chaining `run` → (read the vaHera line) → `test` yourself, as sections
1, 4, and 5 already did — `assist` is convenience, not a different code
path from what you've already run by hand in this tutorial.

---

## 7. What you now know

- `interceptor` wires together three previously-separate ideas:
  AI code **generation** (via the existing LLM cascade), sandboxed
  **execution** (Node's native TS type-stripping for TypeScript, a small
  dedicated Rust CLI for Rust), and two consumers of a run's output —
  **vaHera**, which stores it as a recallable memory, and **wind-tunnel**,
  a TS port of a Rust testing framework's core idea (does a program
  behave the same way every time it runs?).
- Four modes: `generate` (task → code, no execution), `run` (code →
  captured console output, always stored in vaHera), `test` (code → N
  runs → a stability report), `assist` (all of the above, chained).
- `run` and `assist` both store stdout+stderr as a vaHera memory
  automatically — you never have to ask for that separately, and once
  stored, it's recallable with the same `memory find nearest` you'd use
  for anything else in this session.
- Execution failures come in three distinct shapes the module reports
  honestly rather than collapsing: a runtime exception (`stage: "run"`,
  nonzero exit), a compile error (`stage: "compile"`, Rust only, the
  program never ran), and a timeout (`timed_out: true`, no exit code at
  all).
- wind-tunnel's order parameter R and its five named regimes
  (Phase-locked → Synchronized → Partially-locked → Desynchronized →
  Turbulent) measure run-to-run agreement, not correctness — a program
  can be perfectly "Phase-locked" and still be wrong, and perfectly
  "Turbulent" on purpose if it's supposed to involve randomness.
- Generation depends on a working LLM provider and fails loudly with the
  real upstream error when one isn't configured or doesn't authenticate,
  exactly like [Spraypaint's web-search half](./spraypaint-search#5-the-internet-half)
  — it does not fabricate plausible-looking code in that case.

**Next up:** [vaHera DSL](./vahera-dsl) if you skipped straight here — it
covers `memory store`/`memory find nearest` in full, which §5 above only
used, not explained from scratch.

---

## Troubleshooting

- **`interceptor-run binary not found`** — the `crates/interceptor-run`
  CLI hasn't been built on this machine. Run `cargo build -p
  interceptor-run` from the buhera repo root (not from `long-grass/`),
  or set `INTERCEPTOR_RUN_CLI` to an already-built binary's path. Only
  affects `language: "rust"` — TypeScript execution has no external
  binary dependency.
- **`openai HTTP 401` / `gemini HTTP 401` / any HTTP 4xx-5xx from
  generate mode** — the configured LLM provider's key isn't valid or
  isn't authorized. Not a bug in interceptor; check the key, or configure
  `OLLAMA_URL` to point at a locally running Ollama instead (it's tried
  first in the cascade, and needs no API key).
- **A `run` call's stderr shows a temp file path like
  `C:\...\interceptor-ts-XXXXXX\snippet.ts`** — expected. Every run is
  written to a fresh temp file and deleted immediately after; a stack
  trace naturally points there instead of to "real" source, since there
  is no real source file for AI-generated code.
- **wind-tunnel reports "Turbulent" or "Desynchronized" for code you
  expected to be deterministic** — check for anything reading the clock,
  generating random numbers, iterating an unordered collection, or
  depending on filesystem/timing state. This is the test doing its job,
  not a false positive; the fix is in the generated/tested code, not in
  interceptor.
- **A Rust `run`/`test` call is noticeably slower than the equivalent
  TypeScript call** — expected. Rust goes through a real `rustc` compile
  every single call, with no shared build cache across sandboxed
  invocations (each one is isolated from the others by design).
  TypeScript has no compile step at all.
- **Output looks truncated, or `truncated: true` appears in a raw
  result** — each stdout/stderr stream is capped at 2 MiB per run. A
  program that prints more than that gets killed mid-run rather than
  buffering unbounded output server-side.
