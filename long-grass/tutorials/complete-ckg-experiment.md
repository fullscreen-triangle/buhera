# The Complete CKG Experiment

**What you'll learn:** how to hand a whole *problem* to the OS as a task and
let the federation solve it — instead of solving it yourself and handing the OS
the answer. We use a real one: building a causal knowledge graph over an assay
pipeline. The original version of this problem meant authoring an ontology,
running a reasoner across a knowledge graph, and querying it with SPARQL. Here
none of those three artifacts is built by hand. Each one turns out to be a
different *view of a single runtime trajectory* — and the graph the run
produces is one no ontology you could have written in advance describes.

**Time:** ~20 minutes.

**Prerequisites:** [Basic routines](./basic-routines), and it helps to have
skimmed [Shapeshifter routines](./shapeshifter-routines) so you've seen a real
federation module do real work.

**Runtime requirement:** none beyond the browser. The `ckg` module is pure JS
and ships with the webtool. It calls other federation modules (`echo`,
`shapeshifter`, `sbs`, `scope`, …) when you attach them; those that need a
linked package will say so, and the run records that as a fact rather than
halting.

---

## 0. The problem, the old way

The original task: take an assay pipeline — samples go in, get prepped,
measured on an instrument, the spectra get processed, hits get called — and
produce a causal knowledge graph so you can ask *what led to what*.

The old pipeline had three moving parts:

1. **An ontology model.** You sat down and wrote the schema by hand: what a
   `Sample` is, that `preparation` precedes `measurement`, what properties a
   `Hit` carries. Structure, decided up front, before any data moves.
2. **A reasoner.** You loaded the graph and ran a reasoner over it to
   materialize the *derived* facts — the edges the schema entails but you
   didn't state literally.
3. **SPARQL.** You queried the finished graph to read answers out.

It worked. But two things about it always grated. The graph could only ever
contain relations your ontology already anticipated — the reasoner deduces
*within* the schema, it never surprises you with a shape you didn't author. And
the whole thing gave you **results and no report**: a table of hits, with no
account of how the run got there.

We're going to rebuild all three parts as **one runtime trajectory** and get
both things back: a graph shaped by what actually happened, and a report of it.

---

## 1. The mapping

Here is the whole idea on one screen. Keep it in view; every cell below is an
instance of one of these three rows.

| The old artifact | Its runtime twin |
|---|---|
| **ontology model** | a node's `τ` (its type) plus the **chunk-bag** you attach to it — structure handed in, before any run |
| **reasoner** | `dispatch` running the federation. Derived facts are **emitted value-deltas**, not schema entailments. Which module reads which node is decided *during* the run |
| **SPARQL access** | `graph` / `report` — you *walk* the graph the run produced and read the account assembled from its emitted values |

The knowledge graph is not built and then queried. It **is** the runtime
trajectory: nodes are the subtasks the run touched, edges are the reads that
actually fired, facts are the values the modules emitted. Because the trajectory
emerges from the run rather than from a schema, the graph is one no hand-authored
ontology produces. That is the surprise we're after.

Let's start with a clean session.

**Cell 1.1**
```
dispatch("ckg", { op: "reset" })
```

**Expected**
```
runtime reset
```

---

## 2. The ontology, as τ and a chunk-bag

In the old world you'd write an OWL class for each pipeline stage. Here you
`represent` each stage as a **node** with a type `τ`. That's the entire
"schema": a type, and an address the node lives at. No properties, no
subclass axioms — those all become *runtime behavior* attached as chunks.

**Cell 2.1**
```
dispatch("ckg", { op: "represent", tau: "sample", seed: 1 })
```

**Expected**
```
node "sample" represented at ckg/sample
chunks: publish
```

Notice the `publish` chunk it added for you. That's the one piece of behavior
every node gets: it emits a *signal* the run uses to decide what reads what. We
come back to it in §4. Now the rest of the pipeline:

**Cell 2.2**
```
dispatch("ckg", { op: "represent", tau: "prep" })
```

**Cell 2.3**
```
dispatch("ckg", { op: "represent", tau: "measure" })
```

**Cell 2.4**
```
dispatch("ckg", { op: "represent", tau: "process" })
```

**Cell 2.5**
```
dispatch("ckg", { op: "represent", tau: "call" })
```

Five nodes, five types, in order. This is the ontology — but it asserts almost
nothing yet. Everything interesting is going to be *emitted*, not declared.

---

## 3. The reasoner, as attached federation modules

Now we make the nodes *do* something. In the old pipeline, the reasoner was one
engine you pointed at the whole graph. Here, each node's behavior is a **chunk**
that calls a real federation module. When that chunk runs, the module's output
is folded onto the node as a fact. The "reasoner" is just the federation
running.

Start simple — attach `echo` to the `sample` node. `echo` always answers, so
it's the clean case:

**Cell 3.1**
```
dispatch("ckg", { op: "attach", tau: "sample", name: "seen", module: "echo", instruction: "sample logged" })
```

**Expected**
```
attached echo as "seen" on "sample"
chunks: publish, seen
```

The node now has two chunks: `publish` (the signal) and `seen` (a live call to
`echo`). Now attach contributors that each assert a *different kind* of fact
across the pipeline. If a module's package isn't linked in this browser, don't
worry — you'll see that recorded as a fact, not an error that stops anything:

**Cell 3.2**
```
dispatch("ckg", { op: "attach", tau: "measure", name: "spectra", module: "shapeshifter", instruction: "demo" })
```

**Cell 3.3**
```
dispatch("ckg", { op: "attach", tau: "process", name: "pathway", module: "sbs", instruction: "demo" })
```

**Cell 3.4**
```
dispatch("ckg", { op: "attach", tau: "call", name: "cooccurrence", module: "scope", instruction: "demo" })
```

Look at what just happened to a single subtask like `measure`. Its facts are
about to come from `shapeshifter` — forward-simulated spectra, retention time,
fragmentation coherence — a completely different *kind* of assertion than the
pathway `sbs` will fold onto `process`, or the microscopy co-occurrence `scope`
will fold onto `call`. This is the move worth pausing on: one node is a meeting
point where *any* module that has something to say about that subtask leaves a
fact. Think of representing an enzyme this way — one node — and attaching
shapeshifter to check its physico-chemical properties, sbs to derive its
reaction pathway, scope to find its co-occurrence under a microscope. No single
ontology models all three of those relations at once; the node accretes them all
because each module converged on the same subtask. That accretion is the graph
you're building, and no schema authored it.

We now have a schema (§2) *and* behavior (§3). Nothing has run. In the old
world you'd now invoke the reasoner. Here, running the reasoner is running the
nodes — which is the next section, and it's also where the graph's shape gets
decided.

---

## 4. Dispatch judges nothing

Run one node. `dispatch` executes **every** chunk on it, records what each one
emitted, and **judges nothing** — there is no exit code, no ok/fail verdict, no
"the reasoner rejected this." A contributor that throws doesn't halt the run;
its error becomes a recorded value like any other.

**Cell 4.1**
```
dispatch("ckg", { op: "dispatch", tau: "sample" })
```

**Expected**
```
dispatched "sample" — 2 chunk(s), judged nothing
emitted: signal, fact:echo
```

Two emissions: the `signal` from `publish`, and a `fact:echo` — the echo
module's output, folded onto the node. That fact is a *derived* fact in exactly
the reasoner's sense, except no schema entailed it: the federation produced it
by running.

This "judges nothing" property is the whole reason the next step can surprise
you. Nothing has been pruned for being wrong, so the trajectory is free to take
a shape your schema never sanctioned.

---

## 5. The trajectory emerges

Here's the move that has no analogue in the old pipeline. We run a **carry**
across all five nodes. The carry walks the nodes in order; at each one it reads
the `signal` that node emitted and *reaches forward* by that signal's
magnitude, landing an edge on the node it reaches. A node that inherited a
bigger seed reaches further.

The crucial part: the reach is a value **produced during this run**. So the set
of edges — the graph's shape — is decided by the run, not by you.

**Cell 5.1**
```
dispatch("ckg", { op: "carry" })
```

**Expected**
```
carry ran to completion — N edge(s) induced this run
```

(The exact edge count depends on the seeds; that's the point.) Now project the
trajectory as a knowledge graph and look at it:

**Cell 5.2**
```
dispatch("ckg", { op: "graph" })
```

You get nodes (the subtasks the run touched), edges (the carrier reads that
actually landed — the run-induced causal relation), and facts (every value the
modules emitted). This is your SPARQL replacement: you **walk** the graph the
run produced. You did not query a graph you authored — you couldn't have
authored this one.

---

## 6. Seeding a different node reshapes the graph

To feel that the graph's shape is genuinely the run's and not the schema's,
change *where the run starts* and re-carry. Same five nodes, same chunk-bags —
only the seed moves. Push a large seed onto `measure` and carry again:

**Cell 6.1**
```
dispatch("ckg", { op: "seed", tau: "measure", value: 3 })
```

**Cell 6.2**
```
dispatch("ckg", { op: "carry" })
```

**Cell 6.3**
```
dispatch("ckg", { op: "graph" })
```

The edge set is different. The node set didn't change and the ontology didn't
change — only the run did, and the graph followed the run. In the old pipeline
this was impossible: the reasoner's output is a function of the schema, so
re-running it over the same schema gives the same entailments. Here the
trajectory is a function of the *run*.

---

## 7. The report — the thing the original never gave you

The original pipeline handed back results with no account of how it got them.
Now assemble the report:

**Cell 7.1**
```
dispatch("ckg", { op: "report" })
```

You get a per-contributor account: which modules asserted facts, on which
nodes, how many acts ran, how many were errors — and the full audit, the
run-to-completion witness showing every act appeared and none was gated on a
verdict. Toggle the audit open in the rendered report to see the raw acts.

An error fact, if any of your contributors' packages weren't linked, is listed
as a fact — reported, not hidden, not treated as a failure. That's the
non-judging runtime being honest with you.

---

## 8. Reproducing the protocol, not the results

Last piece. In science you want to reproduce a *method*, and you expect the
*results* to vary run to run. The fingerprint hashes the protocol — each node's
`τ`, its sorted chunk names, and any edit addresses — and pointedly **excludes**
the emitted values.

**Cell 8.1**
```
dispatch("ckg", { op: "fingerprint" })
```

Run the whole experiment again from §1 with different seeds and the graph will
differ every time — but this fingerprint stays the same, because the *protocol*
didn't change. That's the right reproducibility guarantee for a system whose
results are supposed to emerge: same method, same fingerprint; different run,
different graph.

---

## What you did

You represented a problem as a task in the OS and let the federation solve it:

- the **ontology** became nodes' `τ` and their chunk-bags (§2);
- the **reasoner** became the federation running, its derived facts *emitted*
  rather than entailed (§3–§4);
- **SPARQL** became walking the graph the run produced (§5, §7);
- and because the runtime **judges nothing**, the trajectory was free to take a
  shape no schema anticipated — different again the moment you moved the seed
  (§6), yet reproducible as a *protocol* (§8).

The graph you got is not one you could have authored in advance. That is the OS
surprising you — which was the whole point of handing it the problem instead of
the answer.
