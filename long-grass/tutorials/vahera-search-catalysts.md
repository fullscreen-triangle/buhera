# vaHera Search Catalysts

**What you'll learn:** how search reaches vaHera — not as a 16th vaHera
statement (the [vaHera DSL tutorial](./vahera-dsl) already established
there are exactly 15, and inventing a new one would contradict the
reference grammar), but through graffiti's catalyst mechanism, which
vaHera's own kernel already plugs into. You'll compare three ways of
resolving the same claim — vaHera's own memory, a full-repository search,
and the internet — side by side, in one script.

**Time:** ~15 minutes.

**Prerequisites:** [The vaHera DSL](./vahera-dsl) (for `describe`/`resolve`
and the kernel's memory), [Spraypaint: Local and Internet
Search](./spraypaint-search) (for what the two new catalysts actually do —
this tutorial doesn't re-explain BM25 or grounding, it wires them into
graffiti), and passing familiarity with `dispatch("graffiti", ...)` — the
[Kwasa-kwasa routines](./kwasa-kwasa-routines) tutorial or `:modules`'
description of `graffiti` covers the `.grf` script shape if you haven't
seen it.

**Runtime requirement:** everything [Spraypaint: Local and Internet
Search](./spraypaint-search) requires, plus the `graffiti` module (bundled,
no extra setup). **One honest gap, stated up front:** the two catalysts
this tutorial wires in (`spraypaint_local`, `web_search`) were verified by
confirming their code matches exactly what graffiti's interpreter calls
(`provider({ currentClaim, args })`, checked directly against
`src/graffiti/lang/interpreter.ts`) and by testing the API routes they call
in isolation — but a live `.grf` script actually invoking them through a
browser session was not run end-to-end while writing this tutorial. If a
cell below doesn't behave as described, that's the seam to check first.

---

## 0. Why this isn't a vaHera statement

vaHera's grammar is closed: `describe`, `resolve`, `spawn`, `navigate to
penultimate`, `complete trajectory`, the six memory/kernel forms, and
`process list` — fifteen, no more, matching the Rust `buhera-vahera` crate
this JS interpreter mirrors. Search doesn't belong on that list, because
search isn't a categorical-trajectory primitive — it's a *catalyst*, a
resource a running computation can call out to, and graffiti already has a
first-class abstraction for exactly that (`namespace: "local" | "remote" |
"inference" | "composite"`, proven "namespace-neutral" by the theory
graffiti implements — the calculus doesn't care whether a catalyst is a
kernel lookup, a repo search, or a network call, they compose identically).

vaHera's kernel already sits behind one catalyst, `kernel_search` — this
tutorial adds two more behind the same interface, `spraypaint_local` and
`web_search`, so a `.grf` script can reach either without vaHera's own
grammar changing at all.

Start clean:

**Cell 0.1**
```
:clear
```

---

## 1. The three catalysts, side by side

`:modules` lists what's registered. Confirm all three search-shaped
catalysts are present before writing a script against them:

**Cell 1.1**
```
dispatch("graffiti", "demo")
```

**Expected** — the built-in demo runs unchanged (it doesn't touch the new
catalysts); this cell is only here to confirm `graffiti` itself dispatches
cleanly before adding anything to it. You should see a `greeting` project
yield `year: "1868"` via the `local_search` fixture catalyst.

**Cell 1.2** — put something in vaHera's own memory, so `kernel_search` has
material to compete with the other two on equal footing:
```
memory store "floor_note" = "a conditioned admissibility floor is a global minimum separation cost over a contact graph's structure"
```

---

## 2. One claim, three catalysts, one script

The comparison this tutorial exists to make: resolve "what is a conditioned
admissibility floor" through `kernel_search` (vaHera's own memory — exactly
the one note from Cell 1.2, nothing else), `spraypaint_local` (the whole
repository), and `web_search` (the internet), in one `.grf` script, and
read off what each catalyst actually returns as its claim and its power:

**Cell 2.1**
```
dispatch("graffiti", `
floor 0.02

catalyst kernel_search {
  namespace: local
  input: Region output: Claim
}
catalyst spraypaint_local {
  namespace: local
  input: Region output: Claim
}
catalyst web_search {
  namespace: remote
  input: Region output: Claim
}

project compare_floor {
  seek claim
    not{ "off topic" }
    toward{ conditioned_admissibility_floor }
    via{ kernel_search(query: "admissibility floor") }
    until converge
    yield claim
}
`)
```

**Expected** — a `compare_floor` project yielding `claim`, resolved through
`kernel_search` alone: the note stored in Cell 1.2 comes back essentially
verbatim (vaHera's memory holds exactly one thing, so there's nothing else
it could return), at whatever power `kernel_search`'s distance-based scaling
computed for this query.

**Cell 2.2** — the same script, `via{ spraypaint_local(...) }` instead:
```
dispatch("graffiti", `
floor 0.02

catalyst spraypaint_local {
  namespace: local
  input: Region output: Claim
}

project compare_floor {
  seek claim
    not{ "off topic" }
    toward{ conditioned_admissibility_floor }
    via{ spraypaint_local(query: "capability calculus six verdict starved refused") }
    until converge
    yield claim
}
`)
```

**Expected** — `claim` this time is a real passage snippet from the
repository — the `spraypaint_local` catalyst calls the same API as
`dispatch("spraypaint", ...)` with a fixed budget of 3 and no scene
restriction, and for this exact query that call's top hit is [Federated
Querying](./federated-querying)'s own opening lines (per [Spraypaint: Local
and Internet Search](./spraypaint-search) §1, which runs a scoped version
of the same query and shows the `hfq` module's source ranked right beside
it). The claim graffiti resolves to should be recognizably that passage
(path:line prefix included, per `createSpraypaintCatalyst`'s claim format),
not vaHera's stored note. Two completely different corpora, same script
shape, same seek statement — only the catalyst named in `via{}` changed
which one got searched.

**Cell 2.3** — `web_search` instead:
```
dispatch("graffiti", `
floor 0.02

catalyst web_search {
  namespace: remote
  input: Region output: Claim
}

project compare_floor {
  seek claim
    not{ "off topic" }
    toward{ conditioned_admissibility_floor }
    via{ web_search(query: "what is a conditioned admissibility floor contact graph") }
    until converge
    yield claim
}
`)
```

**Expected**, with a working `GEMINI_API_KEY`: `claim` is a grounded answer
from the internet, at whatever power `web_search` returns on success
(0.6, fixed — unlike the other two catalysts, this one doesn't scale power
by a distance or score, since a grounded LLM answer doesn't carry a natural
equivalent of "how close was the match"). **Under the current, honestly
broken key** (see [Spraypaint: Local and Internet
Search](./spraypaint-search) §5), expect this cell's `via{}` chain to
resolve at power 0 with a claim like `web_search:unreachable:...` — the
catalyst's own failure marker, not a crash. A `seek` whose only catalyst
returns power 0 will not converge in the usual sense; read whatever the
project's diagnostics report for this case as the real, current behavior of
a catalyst with no working backend, not a bug in the script.

---

## 3. All three in one project

Graffiti's calculus is built to run multiple catalysts toward the same
claim and combine them — that composability is the actual point of a
namespace-neutral catalyst abstraction, not just a convenience for writing
fewer scripts:

**Cell 3.1**
```
dispatch("graffiti", `
floor 0.02

catalyst kernel_search {
  namespace: local
  input: Region output: Claim
}
catalyst spraypaint_local {
  namespace: local
  input: Region output: Claim
}
catalyst web_search {
  namespace: remote
  input: Region output: Claim
}

project compare_floor {
  seek claim
    not{ "off topic" }
    toward{ conditioned_admissibility_floor }
    via{ kernel_search(query: "admissibility floor") }
    via{ spraypaint_local(query: "capability calculus six verdict starved refused") }
    via{ web_search(query: "conditioned admissibility floor") }
    until converge
    yield claim
}
`)
```

**Expected** — one `compare_floor` project, one yielded `claim`, but now
resolved from whichever catalyst the calculus's own convergence rule
favors given all three powers together — read the `diagnostics` field
`dispatch("graffiti", ...)` returns alongside `projects` to see which
catalyst actually won and why, rather than guessing from the claim text
alone.

---

## 3½. What a catalyst's `power` is not, read against `ladder`'s `climb`

Every catalyst above resolves to a `claim` and a `power` in `[0,1]` — and
it is tempting, having just done [Federated
Querying](./federated-querying), to read that `power` as the same kind of
quantity as `ladder`'s composed rung power in that tutorial's §4 (`climb`,
`composite_power`, the `subfloor` refusal). It isn't, and the difference is
worth making concrete rather than asserted, because conflating them is an
easy mistake once both numbers live in `[0,1]` and both get called "power."

**Cell 3½.1** — run the identical rung numbers `ladder` composed in
[Federated Querying](./federated-querying) §4, here, for reference:
```
dispatch("ladder", { op: "climb", powers: [0.45, 0.30, 0.55], target: 0.70 })
```

**Expected** — `verdict: "reached"`, a `composite_power` around `0.827`
(`1 - (0.55)(0.70)(0.45)`, per that tutorial's own derivation). This number
is **intensive-then-composed**: each rung's power was itself derived from a
real graph's local neighborhood (§3 of that tutorial), and the composition
law combining them is a specific, proven multiplicative formula — raise the
`target` past what they can jointly reach and the whole thing refuses
(`subfloor`, `M: 0`) *before* any further computation, by construction.

**Cell 3½.2** — now look at what a graffiti catalyst's `power` actually is,
by re-running Cell 2.1 above and reading the number `kernel_search` reports
for a middling match:
```
dispatch("graffiti", `
floor 0.02

catalyst kernel_search {
  namespace: local
  input: Region output: Claim
}

project probe_power {
  seek claim
    not{ "off topic" }
    toward{ some_unrelated_target }
    via{ kernel_search(query: "floor_note") }
    until converge
    yield claim
}
`)
```

**Expected** — a `power` computed by `createKernelSearchCatalyst`'s own
formula: `0.5 + 0.4 * (1 - normalisedDistance)`, clamped to `[0.5, 0.9]` —
a heuristic scaling of one catalyst's one lookup, invented per-catalyst (
`createSpraypaintCatalyst` uses a different formula again, scaling BM25
score instead of S-distance; `createWebSearchCatalyst` doesn't scale
anything at all, it's a fixed `0.6`). **There is no floor, no composition
law, and no refusal rule tying these numbers together** — graffiti's
namespace-neutrality theorem (§0 above) says the *calculus* treats every
catalyst's power identically regardless of namespace, but it says nothing
about what any individual catalyst's power *means* physically, and nothing
requires two catalysts' power scales to be comparable in the way two
`ladder` rungs derived from the same graph metric are. Reading "web_search
returned power 0.6" and "spraypaint_local returned power 0.73" as "the
internet search was less confident" is not a claim either number actually
supports — they're not measuring the same thing, only sharing a numeric
range by convention. If you need the composed, floor-backed guarantee
`ladder` gives you, `ladder` is the tool for that question, run alongside
graffiti rather than through it — nothing in this session wraps `ladder`
as a graffiti catalyst, and nothing about the catalyst contract in
`src/graffiti/orchestration/catalyst.ts` would make that wrapping
meaningful without also carrying the composition law across the boundary,
which is a real design question, not a one-line registration.

---

## 4. What you now know

- Search reaches vaHera through graffiti's catalyst mechanism, not through
  a new vaHera statement — `kernel_search` (already existed, backed by
  vaHera's own memory), `spraypaint_local` (this repo on disk), and
  `web_search` (the internet) are three catalysts behind one identical
  interface, freely mixable in a single `seek ... via{}` chain.
- The three have zero shared state: `kernel_search` only ever sees what
  `memory store` put in the current browser session; `spraypaint_local`
  only ever sees files actually on disk; `web_search` only ever sees
  whatever the grounding call's browsing tool retrieves live. The same
  query text against all three is a genuine comparison of three different
  corpora, not three views of one.
- A catalyst that can't do its job (no data, no network, no valid key)
  returns power 0 with a diagnostic claim string — it doesn't crash the
  script and it doesn't silently pretend to succeed.
- `:modules`' description of `graffiti` always lists the currently
  registered catalysts — check it if a `via{}` reference to a catalyst name
  ever fails to resolve.
- A catalyst's `power` is a per-catalyst heuristic scaling, not a composed,
  floor-backed guarantee — don't read it as the same kind of quantity as
  `ladder`'s `climb`/`subfloor` from [Federated
  Querying](./federated-querying) §4 just because both live in `[0,1]`.
  Nothing in graffiti's calculus ties catalyst powers to a global
  admissibility bound the way `ladder`'s composition law does; if a
  question needs that guarantee, reach for `ladder` directly.

**Previous:** [Spraypaint: Local and Internet Search](./spraypaint-search)
for the two new catalysts' own behavior in isolation, without graffiti in
the picture. [The vaHera DSL](./vahera-dsl) for `kernel_search`'s side —
vaHera's memory and trajectory grammar on its own terms. [Federated
Querying](./federated-querying) for `ladder`'s composed, floor-backed power
and `hfq`'s six-verdict model — the two engines §3½ above contrasts against
a graffiti catalyst's much simpler, uncomposed `power`.

---

## Troubleshooting

- **`unknown catalyst "spraypaint_local"` (or `web_search`)** — run
  `:modules` and check the `graffiti` entry's catalyst list; if either name
  is missing, the deployment hasn't picked up this tutorial's changes yet.
- **A `via{ web_search(...) }` chain never converges** — almost certainly
  the same broken-key state documented in [Spraypaint: Local and Internet
  Search](./spraypaint-search) §5, not something wrong with the script.
  Check `dispatch("spraypaint", { kind: "web", query: "test" })` directly
  first — if that alone fails, the graffiti path will too, for the same
  reason.
- **`spraypaint_local`'s claim looks like `spraypaint_local:no-hits:<query>`
  or `spraypaint_local:unreachable:<query>`** — the catalyst's own
  no-match / network-failure markers (mirrors `kernel_search`'s
  `:no-corpus:`/`:miss:` convention). Try the query directly via
  `dispatch("spraypaint", {kind:"ask", query:"..."})` to see the raw result
  and diagnose from there.
