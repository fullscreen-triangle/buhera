# Federated Querying: Three Methods, and What LinkML Alone Would Have Told You

**What you'll learn:** three real ways to answer the same question over an
ontology, run inside Buhera OS, then read against a plain LinkML/SPARQL-shape
view of the same question. The three methods are not toy illustrations —
they are the real querying engines from three prior projects, vendored
unmodified into this webtool and wired up as Buhera federation modules:

1. **The conventional pipeline** — ontology → RDF → reasoner closure →
   SPARQL. This is the foil the other two are measured against: a query
   result set that is either non-empty or empty, one bit of information.
2. **HFQ** (`dispatch("hfq", ...)`) — a plan language with a capability
   calculus and a **six-verdict** execution model
   (`answer`/`empty`/`surface`/`timeout`/`refused`/`starved`), each
   non-answer verdict carrying a named blocker
   (`model`/`engine`/`budget`/`corpus`). Vendored from
   `hegel/consequences/src/lib/hfq` — the same engine that answers Mark
   Doerr's five real biocatalysis questions and eight generic Chem-DCAT-AP
   queries.
3. **Ladder** (`dispatch("ladder", ...)`) — a contact-graph admissibility
   construction: an intensive derived power over a weighted graph, composed
   multiplicatively across rungs, that can refuse (`subfloor`) *before any
   commitment* when the declared rungs cannot reach a declared target.
   Vendored from `levinthal/enzymes/web`.

Both are real code, not summaries of it — `vendor/hfq/` and
`vendor/ladder/` in this repo are byte-copies of the source engines, run
through the same `dispatch(...)` mechanism as every other module here.

**Time:** ~20 minutes.

**Prerequisites:** [Basic routines](./basic-routines) for `dispatch(...)`
and the audit log.

**Runtime requirement:** none beyond the browser. Every cell below
dispatches against the real vendored engines — nothing here is illustrative
or mocked.

---

## 0. The question, and what a shape-only view can say about it

Take one of Mark Doerr's real questions (`mark_q1` in the HFQ plan
catalogue): *find a bacterial (not eukaryotic) enzyme that catalyses
benzylethylamine transamination and carries no cysteine in its sequence.*

A LinkML schema — or any shape language (JSON Schema, SHACL, OWL) — can
certify what a record of `Enzyme` looks like: it has an accession, an
organism, a catalysed-reaction link, a sequence field. It cannot certify
whether *this* question, against *this* data, has an answer that exists,
an answer that's empty because none exists, or no answer because some
source in the chain can't be asked what's needed. A SPARQL query against
data validated by that schema returns rows or no rows — one bit.

Run the same question through HFQ and watch it come back as more than one
bit.

**Cell 0.1** — list what's available:
```
dispatch("hfq", { kind: "list_presets" })
```

**Expected** — 24 preset plans across six sections: `verdicts`,
`translation`, `paper`, `semman4cat` (Mark Doerr's five questions),
`dcat` (eight generic Chem-DCAT-AP queries), `other`.

---

## 1. HFQ: run Mark Doerr's own question

**Cell 1.1**
```
dispatch("hfq", { kind: "preset", id: "mark_q1" })
```

**Expected** — a JSON document with `steps[]`, one entry per plan step, each
carrying a `verdict`. Read the step for the sequence scan: its payload
carries `_covered` and `_uncovered` counts, because one of the five
transaminases in this world has no sequence record at all. That enzyme is
**neither included nor excluded** by the no-cysteine scan — it was never
examined. A SPARQL query against the same data returns whatever rows
matched and says nothing about the gap; here the gap is a first-class,
named fact on the record.

**Cell 1.2** — the sharpest case, Q4-shaped (adapted from the paper): ask
for expected products at pH 9 in a named buffer.
```
dispatch("hfq", { kind: "preset", id: "mark_q4" })
```

**Expected** — read the two verdicts side by side. One step reports
`empty` — **no blocker** — because the condition was genuinely tested and
nothing was recorded at it; real chemistry, not a failure. Another step
reports `starved`, with `blocker: "corpus"` — that step was never reached
because an earlier one failed to supply what it needed. A shape-only view
sees the same thing for both: an empty result set. Here they are
distinguishable by construction, and the distinction is exactly the
content that matters to a chemist deciding whether to run the experiment
again or trust the "no."

---

## 2. HFQ: a plan refused before any request is issued

**Cell 2.1** — a plan asking a lookup-only source for something it never
declared:
```
dispatch("hfq", { kind: "run", source: "plan bad {\n  budget 10 requests\n  let x = from enzdb ask reactions_consuming(\"EC:1.1.1.1\")\n  emit x\n}" })
```

**Expected** — `halted_early: true`, `requests_issued: 0`, and the failing
step's verdict is `surface` with a `missing` features list. `enzdb`
declares `{lookup, link}`, not `pattern` — the capability check catches
this **before any request touches the fixture**. A schema check cannot see
this at all: capability (can this source answer this *shape* of question)
is a property of the query engine's contract with its source, not of the
record shape LinkML certifies. Two enzymes with identical schema-valid
records behind two different endpoints — one that supports pattern
queries, one that doesn't — are indistinguishable to a shape check and
produce two different verdicts here.

---

## 3. Ladder: a real contact-graph floor, not an assertion

Ladder derives a power number the way `mark_q1`'s sequence coverage was
computed — from real structure, not a declared constant.

**Cell 3.1** — build a small chain graph and derive its intensive power at
each vertex:
```
dispatch("ladder", "demo")
```

**Expected** — six vertices, each with a derived `power` in `[0,1]`, plus
the graph's `floor` (the minimum edge weight). This `power` is **intensive**:
it depends only on the vertex's local neighbourhood, not on anything
elsewhere in the graph. That property doesn't hold for the two nearby
formulas the notebook rejects for comparison — reachable via `derive`:

**Cell 3.2**
```
dispatch("ladder", { op: "derive", graph: { vertices: ["m","v0","v1","v2"], weights: { "v0|m": 1.0, "v1|m": 1.0, "v2|m": 1.0, "v0|v1": 2.0, "v1|v2": 0.3 }, medium: "m" }, vertex: "v0", radius: 1 })
```

**Expected** — `intensive`, `globalfloor`, `extensive` all reported
together. Add a distant, unrelated low-weight edge somewhere far from `v0`
and re-run: `intensive` is unchanged, `globalfloor` moves (because β is a
minimum over *every* edge in the graph, not just v0's neighbourhood), and
`extensive` moves too (because it normalises by the whole graph's total
weight). Only the intensive quantity is safe to compose across a chain of
independent measurements — which is exactly what a ladder does next.

---

## 4. Ladder: refusing before commitment

**Cell 4.1** — compose three rung powers and ask whether they clear a
target:
```
dispatch("ladder", { op: "climb", powers: [0.45, 0.30, 0.55], target: 0.70 })
```

**Expected** — `verdict: "reached"`, `M: 3` (one commitment per rung
climbed). The composite is `1 - (0.55)(0.70)(0.45) = 0.82675`, clearing the
0.70 target.

**Cell 4.2** — raise the target past what the same rungs can reach:
```
dispatch("ladder", { op: "climb", powers: [0.45, 0.30, 0.55], target: 0.95 })
```

**Expected** — `verdict: "subfloor"`, `M: 0`. The refusal happens **before
any rung is climbed** — nothing is committed, because the multiplicative
composition law makes the ceiling knowable in advance. A schema has no
construct for this at all: "this batch of measurements cannot possibly
reach the required confidence, don't bother running them" is a statement
about a global composition of local quantities, and no per-record
validation rule can express it.

---

## 5. Combining the three inside one Buhera session

HFQ's own `ladder` step kind (a real part of the engine you just ran in
§1–2, not new) composes rung powers exactly like §3–4 — but takes them as
bare declared numbers. Feed it a **derived** power from a real graph
instead of a hand-typed one, and a federated plan's admissibility check
becomes a computed quantity end to end:

**Cell 5.1** — derive a rung power from a real graph (as in §3):
```
dispatch("ladder", { op: "derive", graph: { vertices: ["m","v0","v1"], weights: { "v0|m": 1.0, "v1|m": 1.0, "v0|v1": 0.6 }, medium: "m" }, vertex: "v0", radius: 1 })
```

**Expected** — an `intensive` value; note it down (it will differ run to
run only if you change the graph — this one is deterministic).

**Cell 5.2** — hand that number to an HFQ plan's own ladder step, alongside
a real `from` step querying the ontology fixture:
```
dispatch("hfq", { kind: "run", source: "plan combined {\n  budget 50 requests\n  let acids = from chebi ask descendants_of(\"CHEBI:1\") within 10\n  let L = ladder over acids power 0.62, power 0.55 expect power 0.80\n  emit L\n}" })
```

(Replace one of the `power` literals with the number Cell 5.1 gave you, to
see the plan's `starved`/`answer` verdict respond to a graph-derived rather
than hand-typed rung.)

**Expected** — the `L` step's `composite_power` and verdict. This one
session has now run all three methods on related questions: HFQ's plan
engine (capability-checked federation), the ladder's real contact-graph
derivation (§3), and the ladder-as-plan-step composition (this cell) — the
same combination the paper's separation argument is about, but built from
data the run actually produced rather than three numbers asserted for the
demonstration.

---

## 6. What this means in practice

- A shape language (LinkML, JSON Schema, SHACL, OWL) answers "what does a
  record of this kind look like," and does it well. Nothing here disputes
  that or proposes replacing it.
- What a shape language cannot express, at any degree of schema
  enrichment: (a) the difference between "tested, nothing found" and
  "never reached" (§1's `empty`-vs-`starved` distinction), (b) a
  capability mismatch that must refuse a request before it's issued
  (§2), (c) an intensive, composable quantity over a graph's structure
  that a shape check has no construct to name (§3), and (d) a refusal
  decided by composition, before any commitment (§4).
- Running these methods costs nothing extra where a shape check is
  sufficient — `dispatch("hfq", ...)` and `dispatch("ladder", ...)` sit
  alongside every other Buhera module, callable from the same terminal,
  composable with each other (§5) and with the rest of the federation
  (`ckg`, `graffiti`, `vahera`) the same way.

Reset when you're done exploring — HFQ and ladder hold no session state
between dispatches, so there is nothing to clear.

---

## Troubleshooting

- **`dispatch("hfq", ...)` returns `ok:false` with `stage: "parse"`** — a
  plan source string must match the grammar exactly: `plan NAME { budget N
  requests  let x = from SOURCE ask PRED(...) ...  emit x }`. Check for a
  missing `budget` line or an unbalanced quote inside `ask(...)`.
- **A preset id returns an error** — `dispatch("hfq", { kind:
  "list_presets" })` lists every valid id; the biocatalysis ones are
  `mark_q1`–`mark_q5`, the generic ones `dcat_g1`–`dcat_g8`.
- **`dispatch("ladder", { op: "derive", ... })` errors** — `graph` needs
  `vertices` (array), `weights` (object keyed `"a|b"`, values > 0), and
  `medium` (the vertex name every item connects to).

---

## See also

[Spraypaint: Local and Internet Search](./spraypaint-search) uses this
tutorial's own header and the `hfq`/`ladder` module source as its worked
search examples, and its §4½ compares spraypaint's BM25 "clearing price"
against this tutorial's ladder `floor` — the same family of idea (a number
computed from the whole competitive field, not any one candidate) with a
real limit: a clearing price doesn't refuse anything the way a `subfloor`
verdict does. [vaHera search catalysts](./vahera-search-catalysts) §3½
goes further and contrasts a graffiti catalyst's simple, uncomposed `power`
against this tutorial's §4 composed, floor-backed `climb`/`subfloor` —
worth reading if you've used both and want the distinction made explicit
rather than assumed from both numbers living in `[0,1]`.
