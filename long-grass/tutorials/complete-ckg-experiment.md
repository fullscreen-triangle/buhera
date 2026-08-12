# The Complete CKG Experiment

**What you'll learn:** how to hand a whole *problem* to the OS as a task and
let the federation solve it — instead of solving it yourself and handing the OS
the answer. We use a real one: building a causal knowledge graph of
**cytochrome P450** turnover — its electron-transfer chain, oxidation states,
spectroscopic signatures, and reaction pathways. The original version of this
problem meant authoring an ontology of the catalytic cycle, running a reasoner
across a knowledge graph, and querying it with SPARQL. Here none of those three
artifacts is built by hand. Each one turns out to be a different *view of a
single runtime trajectory* — and the graph the run produces is one no ontology
you could have written in advance describes.

**Time:** ~20 minutes.

**Prerequisites:** [Basic routines](./basic-routines), and it helps to have
skimmed [Shapeshifter routines](./shapeshifter-routines) so you've seen a real
federation module do real work.

**Runtime requirement:** none beyond the browser. The `ckg` module is pure JS
and ships with the webtool. It calls other federation modules (`cytochrome`,
`shapeshifter`, `sbs`, …) when you attach them; those that need a
linked package will say so, and the run records that as a fact rather than
halting.

---

## 0. The problem, the old way

The original task: take one enzyme — **cytochrome P450**, the heme monooxygenase
that turns C–H bonds over — and produce a causal knowledge graph of its turnover
so you can ask *what led to what*: which electron hop feeds which oxidation state,
which state carries the spectroscopic signature, which reaction family the
metabolite came out of.

The old pipeline had three moving parts:

1. **An ontology model.** You sat down and wrote the schema by hand: what an
   `Enzyme` is, that `reduction` precedes `oxo-formation`, what properties a
   `Metabolite` carries. Structure, decided up front, before any chemistry moves.
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

In the old world you'd write an OWL class for each stage of the catalytic cycle.
Here you `represent` each subtask of P450 turnover as a **node** with a type `τ`.
That's the entire "schema": a type, and an address the node lives at. No
properties, no subclass axioms — those all become *runtime behavior* attached as
chunks. Five nodes stand for the five things you'd ask a P450 model about:

| Node `τ` | The P450 subtask it stands for |
|---|---|
| `enzyme` | the resting holoenzyme — heme, participants, catalytic states |
| `spectra` | the spectroscopic readout — Soret, Raman, a virtual MS acquisition |
| `redox` | the NADPH→FAD→FMN→heme electron-transfer chain |
| `metabolite` | the reaction family the product came out of |
| `recovery` | the empty-dictionary node — recover full state from a partial one |

**Cell 2.1**
```
dispatch("ckg", { op: "represent", tau: "enzyme", seed: 1 })
```

**Expected**
```
node "enzyme" represented at ckg/enzyme
chunks: publish
```

Notice the `publish` chunk it added for you. That's the one piece of behavior
every node gets: it emits a *signal* the run uses to decide what reads what. We
come back to it in §4. Now the rest of the cycle:

**Cell 2.2**
```
dispatch("ckg", { op: "represent", tau: "spectra" })
```

**Cell 2.3**
```
dispatch("ckg", { op: "represent", tau: "redox" })
```

**Cell 2.4**
```
dispatch("ckg", { op: "represent", tau: "metabolite" })
```

**Cell 2.5**
```
dispatch("ckg", { op: "represent", tau: "recovery" })
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

Start with the `enzyme` node. Attach the `cytochrome` module twice — once for
the seven-state catalytic orbit, once for the participant/carrier cut. Each
`attach` names a *different chunk*, so one node carries both facts without
collision:

**Cell 3.1**
```
dispatch("ckg", { op: "attach", tau: "enzyme", name: "states", module: "cytochrome", instruction: { op: "states" } })
```

**Expected**
```
attached cytochrome as "states" on "enzyme"
chunks: publish, states
```

**Cell 3.2**
```
dispatch("ckg", { op: "attach", tau: "enzyme", name: "participants", module: "cytochrome", instruction: { op: "participants", reaction: "aliphatic-hydroxylation" } })
```

Now the `spectra` node. Two contributors assert *different kinds* of readout on
it: `cytochrome` folds the resonance-Raman Fe=O signature, and `shapeshifter`
runs a real virtual-MS acquisition of the substrate and its oxidised metabolite.
If a module's package isn't linked in this browser, don't worry — you'll see that
recorded as a fact, not an error that stops anything:

**Cell 3.3**
```
dispatch("ckg", { op: "attach", tau: "spectra", name: "raman", module: "cytochrome", instruction: { op: "raman" } })
```

**Cell 3.4** — the shapeshifter `.ss` script (a virtual orbitrap acquisition of a
CYP substrate + oxidised metabolite). The whole script is the `instruction`:
```
dispatch("ckg", { op: "attach", tau: "spectra", name: "ms", module: "shapeshifter", instruction: `objective p450_metabolite_scan:
  target: "CYP substrate + oxidised metabolite, positive mode"

instrument orbi:
  kappa: 1e12
  ref_frequency: 10e6

phase acquire:
  records = lavoisier.instrument.run_experiment(classes: ["PC"], polarity: "+", analyser: "orbitrap", mz_window: [150, 500])
  field = lavoisier.observe.partition_field(records: records)` })
```

The `redox` node carries the electron-transfer chain two ways: as the native
`cytochrome` Marcus fact, and as an `sbs` circuit — the CPR→heme redox ladder,
where FMN→heme is the smallest conductance (the rate-limiting hop):

**Cell 3.5**
```
dispatch("ckg", { op: "attach", tau: "redox", name: "et", module: "cytochrome", instruction: { op: "electron-transfer" } })
```

**Cell 3.6** — the SBS circuit; the whole `.sbs` source is the `instruction`:
```
dispatch("ckg", { op: "attach", tau: "redox", name: "ladder", module: "sbs", instruction: `circuit p450_electron_transfer {
  node NADPH { mu: -320.0, concentration: 1.0, compartment: "cytoplasm" }
  node FAD   { mu: -220.0, concentration: 1.0 }
  node FMN   { mu: -190.0, concentration: 1.0 }
  node heme  { mu: -170.0, concentration: 1.0 }

  edge NADPH -> FAD  { rate: 30.0, conductance: 3.0 }
  edge FAD   -> FMN  { rate: 50.0, conductance: 5.0 }
  edge FMN   -> heme { rate: 5.0,  conductance: 0.5 }
}

observe p450_electron_transfer
perturb p450_electron_transfer { factor: 0.1 }
navigate from heme` })
```

Finally the `metabolite` node — the reaction family the product came out of,
sorted by aperture ΔM:

**Cell 3.7**
```
dispatch("ckg", { op: "attach", tau: "metabolite", name: "pathway", module: "cytochrome", instruction: { op: "pathway", reaction: "aliphatic-hydroxylation" } })
```

Look at what just happened to a single subtask like `redox`. It now carries two
facts of a *completely different kind*: the `cytochrome` module's Marcus
electron-transfer claim (categorical distance, λ, the rate-limiting hop) and the
`sbs` circuit's coherence `R` and flux visibility `V` for the same chain. This is
the move worth pausing on: one node is a meeting point where *any* module that
has something to say about that subtask leaves a fact. The `enzyme` node holds
its catalytic states *and* its participant cut; the `spectra` node holds a Raman
signature *and* a virtual-MS workspace. No single ontology models all of those
relations at once; the node accretes them all because each module converged on
the same subtask. That accretion is the graph you're building, and no schema
authored it.

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
dispatch("ckg", { op: "dispatch", tau: "enzyme" })
```

**Expected**
```
dispatched "enzyme" — 3 chunk(s), judged nothing
emitted: signal, fact:cytochrome#states, fact:cytochrome#participants
```

Three emissions: the `signal` from `publish`, and two `fact:cytochrome` values —
the seven-state catalytic orbit and the participant/carrier cut, each folded onto
the node under its own chunk name. Those are *derived* facts in exactly the
reasoner's sense, except no schema entailed them: the federation produced them by
running.

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

Each fact isn't a stringified summary — it's the module's **whole output**. A
`fact:sbs` on `redox` carries the P450 electron-transfer circuit `sbs` actually
built, its coherence `R` and flux visibility `V`, the S-entropy observation it
rendered on the GPU. A `fact:shapeshifter` on `spectra` carries the produced
virtual-MS workspace. Click a fact open (`▸`) and the module's own chart draws inline —
the identical panel you'd see if you dispatched that module directly. The graph
node is a *meeting point* holding each contributor's real artifact, not a note
that one ran.

---

## 6. Seeding a different node reshapes the graph

To feel that the graph's shape is genuinely the run's and not the schema's,
change *where the run starts* and re-carry. Same five nodes, same chunk-bags —
only the seed moves. Push a large seed onto `spectra` and carry again:

**Cell 6.1**
```
dispatch("ckg", { op: "seed", tau: "spectra", value: 3 })
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

The original pipeline handed back results with no account of how it got them —
and, worse, when the modules themselves produce charts and rich readouts, a
"report" that only counts them has thrown the actual findings away. So this
report is a **dossier**: it collects, per contributing module, the real output
each one asserted — the same charts and metrics the module renders on a direct
dispatch — and lays them out as the account of the run.

**Cell 7.1**
```
dispatch("ckg", { op: "report" })
```

Read it top to bottom. Under each contributor (`cytochrome`, `sbs`,
`shapeshifter`) you get one entry per subtask it spoke on. Each entry leads with a
**findings headline** — for `cytochrome`, the seven-state orbit / Raman signature
/ Marcus chain / pathway apertures; for `sbs`, the node/edge counts, coherence
`R`, flux visibility `V`, the backend it solved on; for `shapeshifter`, the
virtual-MS workspace it produced — and then expands (`▸ show chart`) into the
module's **own artifact**: the seven-state orbit chart, the P450 electron-transfer
MetricsDashboard, the mass-spec panels. This is not a summary *of* the modules'
work. It *is* the modules' work, assembled. The report renders exactly what the
modules render — which is why it is finally a report and not a tally.

Below the dossier, toggle the **audit** open: the run-to-completion witness
showing every act appeared and none was gated on a verdict.

An error fact, if any of your contributors' packages weren't linked, is listed
as a contribution with its error headline — reported, not hidden, not treated as
a failure. That's the non-judging runtime being honest with you.

---

## 8. The `recovery` node — a graph that stores nothing

The fifth node has been sitting empty. It is the point of the whole exercise, so
attach the `cytochrome` module's admissibility **floor** to it and run it:

**Cell 8.1**
```
dispatch("ckg", { op: "attach", tau: "recovery", name: "floor", module: "cytochrome", instruction: { op: "floor" } })
```

**Cell 8.2**
```
dispatch("ckg", { op: "dispatch", tau: "recovery" })
```

The `cyp_floor` fact this folds is the conditioned admissibility floor
`β = disc + Q + conv`: at categorical depth d = 9 the conversion term dominates,
so β becomes effectively condition-independent — the node can answer for states
it never stored, because admissibility is *recomputed*, not looked up.

That is the **empty-dictionary** discipline, and it is exactly what the whole CKG
run has been doing. No node held a table of answers; every fact was recovered by
running the contributor when the node was dispatched. Two P450 papers make this
precise and measure it:

- **Database recovery** (monograph Paper 13, the paper lesson `10_db-recovery`
  performs): given a partial observation, the receiver synthesises the full state
  by constraint propagation to a self-consistent fixed point — Kirchhoff balance,
  loop consistency, backward trajectories — **not** by forward simulation from a
  stored record. `stored = 0`; it recovers what it never stored.
- **The empty dictionary for cytochrome P450** (monograph Paper 16): the
  *resident* state is **O(1) — 561 bytes** (20 coordinate rows plus an encoding
  rule), constant while a derived cache would grow across nine orders of magnitude
  (≈357 MB at protein scale). Queries are answered **6/6** against a
  no-stored-entry scheme where an index control answers **0/6**, and the resident
  state's hash is unchanged before and after querying. Any large cache is an
  *optional* O(N·k) accelerator, not the thing that holds the knowledge.

So the graph you built is an empty dictionary too: its `τ`-and-chunk skeleton is
the 561-byte resident state, and every fact on it is recovered by the run — which
is why moving the seed (§6) gives a different graph over the *same* skeleton, and
why the protocol, not the results, is the thing you reproduce next.

---

## 9. Reproducing the protocol, not the results

Last piece. In science you want to reproduce a *method*, and you expect the
*results* to vary run to run. The fingerprint hashes the protocol — each node's
`τ`, its sorted chunk names, and any edit addresses — and pointedly **excludes**
the emitted values.

**Cell 9.1**
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
- the graph turned out to be an **empty dictionary** — a 561-byte skeleton whose
  every fact is recovered by the run, not stored (§8);
- and because the runtime **judges nothing**, the trajectory was free to take a
  shape no schema anticipated — different again the moment you moved the seed
  (§6), yet reproducible as a *protocol* (§9).

The graph you got is not one you could have authored in advance — a P450 model
that no hand-written ontology of the catalytic cycle produces. That is the OS
surprising you — which was the whole point of handing it the problem instead of
the answer.
