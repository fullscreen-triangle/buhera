# Shape Is Not Admissibility

**What you'll learn:** why a data schema — LinkML, JSON Schema, SHACL, OWL, or
Buhera's own `represent` — can certify what a record *looks like* and still
say nothing about which *questions* the data underneath can actually answer.
We build both halves for real: a schema (a CKG node's `τ`) and a floor (a real
module's conditioned admissibility bound), then attach a second data source
with the same shape and a different floor, and watch a question flip from
admissible to inadmissible without a single field changing.

**Time:** ~15 minutes.

**Prerequisites:** [Basic routines](./basic-routines), and
[The Complete CKG Experiment](./complete-ckg-experiment) if you want the full
background on `represent` / `attach` / `dispatch` / `graph` — we reuse all
four here without re-deriving them.

**Runtime requirement:** none beyond the browser. Every cell below dispatches
against a real federation module (`ckg`, `cytochrome`, `sbs`) shipped with
this webtool — nothing here is illustrative or mocked.

---

## 0. The claim, stated once

LinkML answers "what does a record of this kind look like": classes, slots,
types, enums, compiled to JSON Schema, SHACL, OWL, SQL DDL, whatever a surface
needs. Nothing below disputes that, and nothing below proposes replacing it.

The question this tutorial works through is different: **given a schema that
is complete and honest, does a consumer holding it know which questions are
admissible against the data it describes?** The answer is no, and it isn't a
gap in any particular schema language's coverage — it's a theorem about what
*kind* of thing a schema can name. A schema is built from features you can
check on one record, or one record's local neighborhood, independently of
every other record (that's what makes a schema language tractable at all). An
admissibility floor is a property of the *whole* population — a global
minimum — and no local, per-record feature can encode a global minimum. This
is exactly the separation the `cytochrome` module's `floor` op and the `ckg`
module's `graph`/`report` ops already give us, live, for real.

Start clean:

**Cell 0.1**
```
dispatch("ckg", { op: "reset" })
```

**Expected**
```
runtime reset
```

---

## 1. Shape: a node's τ is a schema, nothing more

In [the CKG tutorial](./complete-ckg-experiment) a node's type `τ` stood in
for an ontology class. Here we use it for exactly what a LinkML class is: a
name plus a declared shape, asserting nothing about behavior yet. Represent
one node for each of two *data sources* — same reaction family, same declared
shape, about to diverge only in what floor their data actually has:

**Cell 1.1**
```
dispatch("ckg", { op: "represent", tau: "source_a", seed: 1 })
```

**Cell 1.2**
```
dispatch("ckg", { op: "represent", tau: "source_b", seed: 1 })
```

**Expected** (both cells)
```
node "source_a" represented at ckg/source_a
chunks: publish
```

Two nodes, identical τ, identical declared shape. If you generated a LinkML
schema for "P450 turnover record" and validated an instance from each source
against it, both would validate clean. Nothing distinguishes them yet — which
is the point. Shape alone is symmetric between them.

---

## 2. The floor: a property no single field carries

Now attach a *real* admissibility computation to each — the `cytochrome`
module's conditioned floor, `op: "floor"`. This is a physically-derived
β (discreteness + quadratic + convergence terms), not a toy number, and it is
exactly the kind of quantity a shape schema has no slot for: it isn't a
property of one record, it's a bound the whole conditioned population sits
above.

**Cell 2.1** — source A at reference conditions:
```
dispatch("ckg", { op: "attach", tau: "source_a", name: "floor", module: "cytochrome", instruction: { op: "floor" } })
```

**Cell 2.2**
```
dispatch("ckg", { op: "dispatch", tau: "source_a" })
```

**Cell 2.3** — source B, same op, perturbed conditions (this is the whole
experiment: same shape, different population):
```
dispatch("ckg", { op: "attach", tau: "source_b", name: "floor", module: "cytochrome", instruction: { op: "floor", conditions: { T: 250, depth: 3 } } })
```

**Cell 2.4**
```
dispatch("ckg", { op: "dispatch", tau: "source_b" })
```

Look at the two `β` values and their `dominant_term`. At the reference
conditions the convergence term usually dominates and β is effectively
condition-independent; move `depth` down and `T` away from reference and the
Q-term can take over — the note field says so explicitly, per record. Nothing
about `source_a`'s or `source_b`'s *shape* changed: same τ, same chunk name,
same op. Only the floor moved, and it moved for a reason no class/slot
declaration states, because no class/slot declaration is about a bound over
conditions in the first place.

---

## 3. Attach a second, independent shape source: same schema, different data

To make the separation concrete rather than numeric, give `source_b` a second
contributor of a *structurally different kind* — an SBS circuit, which is
Buhera's nearest thing to the weighted contact graph the floor is actually
defined over (nodes, weighted edges, a computed coherence). The circuit's
weakest edge is its own floor-like bottleneck, visible directly in the
conductances:

**Cell 3.1** — the whole `.sbs` source is the instruction, same pattern as
the CKG tutorial's redox ladder:
```
dispatch("ckg", { op: "attach", tau: "source_b", name: "contact_graph", module: "sbs", instruction: `circuit toy_source {
  node m    { mu: 0.0,    concentration: 1.0 }
  node v0   { mu: -50.0,  concentration: 1.0 }
  node p    { mu: -120.0, concentration: 1.0 }
  node xstar{ mu: -200.0, concentration: 1.0 }

  edge m  -> v0    { rate: 12.0, conductance: 4.0 }
  edge v0 -> p     { rate: 8.0,  conductance: 3.0 }
  edge p  -> xstar { rate: 1.5,  conductance: 0.4 }
}

observe toy_source
navigate from v0` })
```

**Cell 3.2**
```
dispatch("ckg", { op: "dispatch", tau: "source_b" })
```

Read the rendered circuit's coherence `R` and the `p→xstar` edge's
conductance against the other two. That edge is the graph's bottleneck —
exactly the role the paper's separation theorem gives the floor: a global
minimum, sitting on one edge, invisible to any per-node or per-record
description of `p` or `xstar` alone. Give `source_a` the same circuit with
that one edge strengthened, and the two sources still validate against
*identical shape* while their admissibility answer for "is `xstar` reachable
from `v0`" is no longer the same question with the same answer:

**Cell 3.3**
```
dispatch("ckg", { op: "attach", tau: "source_a", name: "contact_graph", module: "sbs", instruction: `circuit toy_source {
  node m    { mu: 0.0,    concentration: 1.0 }
  node v0   { mu: -50.0,  concentration: 1.0 }
  node p    { mu: -120.0, concentration: 1.0 }
  node xstar{ mu: -200.0, concentration: 1.0 }

  edge m  -> v0    { rate: 12.0, conductance: 4.0 }
  edge v0 -> p     { rate: 8.0,  conductance: 3.0 }
  edge p  -> xstar { rate: 9.0,  conductance: 3.5 }
}

observe toy_source
navigate from v0` })
```

**Cell 3.4**
```
dispatch("ckg", { op: "dispatch", tau: "source_a" })
```

Same node names, same edge names, same declared "shape" of the circuit
description. Only the weight on one edge differs between `source_a` and
`source_b`, and that alone is the entire content of the separation theorem:
**shape does not determine admissibility**, witnessed here by two sources
that validate identically and disagree on reachability.

---

## 4. Walk the trajectory instead of querying a schema

In the old pipeline you would now write a SPARQL query against a hand-built
ontology and hope "is `xstar` reachable" was expressible in it. Here you
don't query a schema you authored — you walk the graph the run actually
produced:

**Cell 4.1**
```
dispatch("ckg", { op: "carry" })
```

**Cell 4.2**
```
dispatch("ckg", { op: "graph" })
```

**Cell 4.3**
```
dispatch("ckg", { op: "report" })
```

The report's `contributions` are laid out per module — `cytochrome`'s floor
fact and `sbs`'s circuit fact sitting side by side on the *same* node,
because both converged on the same subtask. Neither module's output is
derivable from the other's: the floor doesn't retain edge-level structure,
and the circuit's conductances don't retain the physically-derived β. Both
are necessary; that's Buhera's actual data-modeling primitive — not a rival
schema language to LinkML, but this second, floor-bearing layer sitting
underneath whichever shape language a surface already uses.

---

## 5. What this means in practice

- A dataset-catalog listing, a metadata harvester, a FAIR-compliance
  check — these are shape questions. Adopt LinkML (Chem-DCAT-AP, say) for
  them unmodified; nothing above argues against that, and extending it for
  these purposes buys nothing.
- "Which enzymes have I not yet tried against this substrate, and would
  trying them tell me anything the floor doesn't already foreclose" is an
  admissibility question. No schema answers it at any degree of enrichment,
  for the same reason `source_a` and `source_b` above validate identically
  and disagree anyway: the floor is global, and every schema's declarable
  features are local by construction.
- `represent` + `attach` + `dispatch` + `graph`/`report` is already this
  composition, running: §1 was shape, §2–3 were the floor, §4 was walking
  the trajectory instead of querying an authored schema.

Reset when you're done exploring:

**Cell 5.1**
```
dispatch("ckg", { op: "reset" })
```
