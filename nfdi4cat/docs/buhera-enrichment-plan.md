# Deriving What the Graph Cannot State: A Plan to Enrich the nfdi4cat Transaminase Knowledge Graph with the Buhera Derivation Layer

**Author:** Kundai Farai Sachikonye
**Audience:** Prof. Dr. Mark Doerr, nfdi4cat
**Status:** Design proposal — for review prior to implementation
**Date:** 2026-07-26

---

## Abstract

The nfdi4cat transaminase knowledge graph (henceforth *the KG*) is a FAIR, SPARQL-queryable RDF artefact describing three PLP-dependent aminotransferase reactions (EC 2.6.1.1/.2/.3). It is correct, well-sourced, and complete *as a record of asserted facts*. This proposal does not seek to replace, re-derive, or reimplement any part of it. Instead it identifies three classes of knowledge that the KG structurally *cannot contain in its current form* — not because of an authoring oversight, but because each is a fact that must be **computed** rather than **asserted** — and proposes a purely additive layer, the **Buhera Derivation Layer**, that computes each class and writes it back as further triples on the *same* subjects, under explicit provenance.

The organising principle is a single sentence: **the conventional pipeline asserts facts; Buhera derives facts, and writes them back as more triples on the same graph.** Every enrichment produces triples of the form `?existing_subject ?new_predicate ?computed_value`, each stamped `prov:wasGeneratedBy buhera:<module>`. Nothing in the base graph is altered or removed. The enrichment is detachable: dropping the Buhera-generated named graph returns the artefact bit-for-bit to its current state. The relationship is therefore *strictly upside* — the base pipeline remains the reviewable floor, and Buhera is an optional ceiling.

We characterise three gaps, map each to concrete Buhera OS modules that already exist, specify the exact triples each will emit against the KG's real IRI schema, and give an integration architecture that honours both the nfdi4cat FAIR commitments (ChEBI/KEGG/EC anchoring, `skos:exactMatch`, `dcterms` provenance) and Buhera's own frozen module contract (`dispatch(id, instruction)` over the `ActResult` trait).

---

## 1. Introduction

### 1.1 What the KG is today

The KG is produced by a small, disciplined Python pipeline (`transaminase_kg`). Its structure is worth stating precisely because the proposal is defined *relative to it*:

- **`reference.py`** is the single source of truth: nine `Compound` records (each with a verified ChEBI accession, KEGG id, molecular formula as an element-count map, and net charge) and three `Reaction` records (each with an EC number, KEGG reaction id, systematic name, donor/acceptor/product roles, and the shared PLP cofactor). Every value carries an external identifier; nothing is unsourced.
- **`graph.py`** is a pure transform over that reviewed data. It mints individuals under `https://w3id.org/nfdi4cat/transaminase-kg/resource/` and types them against the ontology namespace `…/ontology#` (prefix `ta:`). Each species gets `skos:exactMatch` to its ChEBI IRI and `skos:closeMatch` to KEGG; each enzyme gets `ta:ecNumber` plus `skos:exactMatch` to the EC resolver; each reaction gets its donor/acceptor/product roles, `ta:catalyzedBy`, reversibility, and mechanism.
- The artefact is served over SPARQL 1.1 (`api.py`), shipped as OWL (T-Box) + Turtle (A-Box), containerised, and tested (31 tests, mass/charge balance among them).

This is a model FAIR deliverable. The proposal takes it as a fixed, trusted substrate.

### 1.2 The distinction the proposal turns on

There are two kinds of statement one can put in a knowledge graph:

1. **Asserted facts** — statements a human curator wrote down from an authoritative source. "ALT has EC number 2.6.1.2." "L-alanine has ChEBI accession 16977." The KG is entirely, and appropriately, of this kind.

2. **Derived facts** — statements that are *entailed by* or *computable from* the asserted facts plus a body of domain method, but which no curator has written down because writing them down by hand would be error-prone, unscalable, or simply not the curator's job. "The monoisotopic mass of L-alanine is 89.0477 Da." "These three reactions form a star topology around a shared 2-oxoglutarate/L-glutamate hub." "The mass balance of R00258 closes to zero." "The recommended name *alanine transaminase* is evidenced by KEGG ENZYME entry 2.6.1.2, passage *N*."

The central observation is that **a correct, complete graph of asserted facts is still missing every derived fact** — and that the derived facts are often exactly what a downstream consumer (a modeller, a reviewer, a search system, a reasoner) actually needs. The Buhera Derivation Layer's job is to compute derived facts and return them to the graph as first-class, provenance-stamped triples, so that what was previously latent in method becomes queryable in SPARQL.

### 1.3 Three gaps

We group derivable facts into three gaps, in increasing order of ambition and decreasing order of conventionality:

- **Gap 1 — The graph does not contain its own correctness.** Mass balance and charge balance are *proven* (in `pytest`), but the proof lives outside the graph. A consumer querying the graph cannot ask "is this reaction balanced, and by what margin?" The correctness is real but not queryable.

- **Gap 2 — The graph does not contain anything that was not hand-typed.** There are no computed chemical properties (monoisotopic mass, predicted spectra, structural fingerprints), and no computed links to the literature that grounds the asserted names and roles. Everything present was typed by a curator; nothing was *derived*.

- **Gap 3 — The graph does not contain its own emergent structure.** The three reactions are asserted in isolation. The fact that they *share* a hub — every one consumes 2-oxoglutarate and produces L-glutamate, every one depends on PLP — is true of the data but is nowhere stated as a triple. The network-level structure is emergent and unrecorded.

Each gap is addressed by a distinct part of the Buhera OS federation. §§4–6 treat them in turn. §3 gives the shared architecture; §7 the provenance and detachability guarantees; §8 the evaluation plan; §9 scope and honest limitations.

---

## 2. Design principles

The proposal inherits and respects two independent contracts, and adds one of its own.

**P1 — nfdi4cat FAIR contract (inherited).** All enrichment triples reuse community vocabularies where they exist (ChEBI, KEGG, EC, PROV-O, QUDT/OM for quantities, SKOS for mappings). New predicates are minted under the existing `ta:` ontology namespace only when no community term fits, and are documented in the T-Box. Every enrichment subject is an IRI that already exists in the base graph — we never mint a parallel identity for an entity the KG already names.

**P2 — Buhera module contract (inherited).** Every derivation is performed by a Buhera OS module conforming to the frozen trait in `registry.js`: a stable `id`, a synchronous `describe()`, an `async execute(instruction, actBudget)` returning an `ActResult` (`{ok, output_delta, residue, completed}`), and an `outputCell()` sufficiency descriptor. Derivations are invoked through the single funnel `dispatch("<id>", instruction)`. No module invents a new dispatch path; each occupies one of the three sanctioned roles (Solver, Router/Interceptor, Meta/Context).

**P3 — Additivity and detachability (new, load-bearing).** The Buhera layer writes only into a dedicated named graph, `…/enrichment`. It never issues an `UPDATE` against a base triple. The base graph is loaded read-only. The composite served to consumers is `base ⊎ enrichment`; removing `enrichment` is a no-op on `base`. This is the single guarantee that makes the layer *pure upside*: at worst it is inert; it can never regress the deliverable.

---

## 3. Architecture of the derivation layer

### 3.1 The write-back shape

Every enrichment, regardless of gap, produces triples of exactly one shape:

```
<existing-subject>  <predicate>  <computed-object> .
<statement-node>    prov:wasGeneratedBy  buhera:<module-id> .
<statement-node>    prov:generatedAtTime "…"^^xsd:dateTime .
<statement-node>    prov:wasDerivedFrom  <inputs…> .
```

where the computed triple is reified (or attached via a PROV-O `prov:Activity` / RDF-star annotation, chosen at implementation time — see §7) so that provenance attaches to the *derivation*, not merely to the subject. The subject is always an IRI the base graph already minted: a `res:species/*`, `res:enzyme/*`, or `res:reaction/*` node.

### 3.2 The dispatch funnel

Derivations run through Buhera OS's single dispatch cycle. Concretely, the enrichment build is a sequence of calls:

```
dispatch("honjo",     { kind: "check-balance", reaction: "R00258", … })
dispatch("lavoisier", { kind: "predict-ms",    species:  "CHEBI:16977", … })
dispatch("graffiti",  { kind: "evidence",       ec: "2.6.1.2", … })
dispatch("purpose-carry", { kind: "topology", scope: "all-reactions" })
```

Each returns an `ActResult` whose `output_delta` carries the computed values; a thin adapter (the *triple-emitter*, §3.3) turns each `ActResult` into RDF written to the enrichment graph. Because dispatch is the same funnel the interactive terminal uses, every derivation is also runnable *live* against the KG from the Buhera terminal — the enrichment build and the interactive demo are the same code path, which is exactly the demonstration story to show nfdi4cat.

### 3.3 The triple-emitter adapter

A single new component — call it `kg-emit` — sits between Buhera and rdflib. It:

1. loads the base graph read-only,
2. drives the dispatch sequence,
3. maps each module's `output_delta` to triples via a per-module schema (declared once, reviewed like any other mapping),
4. writes them to the `enrichment` named graph with PROV-O stamps,
5. serialises `enrichment` as a standalone Turtle side-car.

`kg-emit` is the *only* new code that touches both worlds. Everything upstream is an unmodified Buhera module; everything downstream is standard rdflib/SPARQL. This keeps the seam small, auditable, and removable.

### 3.4 Integration seam

Three seam options exist; we recommend the first for the demo and note the others for production:

- **Side-car merged at load (recommended).** `api.py`'s `load_graph()` gains an optional second parse of `enrichment.ttl` into a named graph. One conditional, fully detachable, zero risk to the base build. This is the seam that best demonstrates the "pure ceiling" property.
- **Enrichment endpoint.** A `/enrich` route runs `kg-emit` on demand and returns the side-car. Good for live demos; more moving parts.
- **Build-time step.** `build-kg` gains an opt-in `--enrich` flag. Cleanest for reproducible artefacts; couples the two builds more tightly.

The recommendation is to ship seam 1 and keep 2–3 as documented options.

---

## 4. Gap 1 — Making the graph contain its own correctness

### 4.1 The gap, precisely

`tests/` proves that every reaction is mass- and charge-balanced by summing the element-count maps and charges of substrates and products. This is a genuine correctness property. But it is *asserted nowhere in the graph*. A consumer who loads the Turtle without the test suite has no way to ask the graph whether R00258 balances, nor to discover the residual (which should be the zero vector). Correctness is a property of the data that the data does not record about itself.

### 4.2 The module: `honjo`

`honjo` is Buhera's cheminformatics solver. Its role here is narrow and defensible: given a reaction's substrate and product species (each already carrying a ChEBI-verified formula and charge in `reference.py`), it recomputes the elemental and charge balance independently of the Python test, and returns the per-element residual vector and the charge residual. Because it recomputes rather than trusts, its output is an *independent check* that also happens to be serialisable.

### 4.3 Triples emitted

For each `res:reaction/<key>`:

```turtle
res:reaction/ALT  ta:massBalanced      true ;
                  ta:chargeBalanced     true ;
                  ta:massResidual       "0"^^xsd:integer ;   # summed over elements
                  ta:chargeResidual     "0"^^xsd:integer .
```

with a PROV-O activity node recording `prov:wasGeneratedBy buhera:honjo` and `prov:used` the participating species IRIs. Optionally, per-element residuals are emitted as a small blank-node structure for full transparency.

### 4.4 Why this is the safe first gap

Gap 1 is the least contestable: it computes a property the project already trusts (the tests pass), and merely *publishes* it into the graph. If Buhera's `honjo` and the Python test ever disagree, that disagreement is itself valuable (it means one of them has a bug) — but the expected outcome is exact agreement, which is a clean, verifiable demonstration that the derivation layer is sound before we ask anyone to trust its more ambitious outputs. **Gap 1 is where trust in the layer is established.**

---

## 5. Gap 2 — Making the graph contain derived chemistry and derived evidence

This is the flashiest gap and, per the project's own assessment, where the framework shines. It has two sub-parts: derived *chemical properties* and derived *literature evidence*.

### 5.1 Derived chemical properties — the module: `lavoisier`

`lavoisier` is Buhera's mass-spectrometry / molecular-property solver (a Solver role: "forward-simulate mass spectra"). From the ChEBI structure of a species it can compute properties that are entailed by the molecule but absent from the KG:

- **Monoisotopic and average molecular mass**, from the formula already in `reference.py` — a pure, deterministic computation, ideal as the first derived property.
- **Predicted MS fragmentation** — a forward simulation producing a small predicted peak list, which is genuinely *new* information (nowhere in the base graph) and directly useful to an analytical-chemistry consumer.
- **Structural fingerprints / descriptors** — where a ChEBI structure is resolvable.

#### Triples emitted

For each `res:species/<key>`:

```turtle
res:species/L-alanine
    ta:monoisotopicMass  "89.04768"^^xsd:decimal ;   # QUDT/OM unit: dalton
    ta:averageMass       "89.0932"^^xsd:decimal .

res:species/L-alanine  ta:hasPredictedSpectrum  _:spec1 .
_:spec1  a ta:PredictedMassSpectrum ;
         ta:peak [ ta:mz "90.055"^^xsd:decimal ; ta:relIntensity "100"^^xsd:decimal ] ,
                 [ ta:mz "44.049"^^xsd:decimal ; ta:relIntensity "62"^^xsd:decimal ] ;
         prov:wasGeneratedBy buhera:lavoisier .
```

Masses attach as typed quantities (QUDT/OM units for FAIR unit semantics); predicted spectra attach as a structured node the T-Box gains a class for. All PROV-stamped.

### 5.2 Derived evidence — the modules: `graffiti` + `purpose`

The KG asserts recommended names, systematic names, and cofactor identity. These come from KEGG ENZYME and BRENDA (`provenance()` says so) — but *at the granularity of the whole dataset*, not per-fact, and with no link to the specific passage that grounds each assertion. Gap 2's second half closes this.

- **`graffiti`** is Buhera's search-program solver (executes a `.grf` search program; Solver role). Its `spraypaint` organ retrieves passages. Given an EC number or a compound name, it locates the specific literature/database passage that evidences the asserted fact.
- **`purpose`** is the symbol/passage indexing organ underneath. Together they turn "sourced from KEGG/BRENDA" (a dataset-level `dcterms:source`) into per-fact evidence links.

Each evidence link carries a **confidence** derived from the χ character-invariant (the min-cut-residual the Buhera search organs compute), so consumers can filter by strength of evidence.

#### Triples emitted

```turtle
res:enzyme/2.6.1.2
    ta:evidencedBy [ a ta:Evidence ;
                     ta:source        <https://identifiers.org/kegg.enzyme:2.6.1.2> ;
                     ta:passage       "…recommended name: alanine transaminase…" ;
                     ta:confidence    "0.94"^^xsd:decimal ;   # χ min-cut-residual
                     prov:wasGeneratedBy buhera:graffiti ] .
```

This is the enrichment that turns a static curated claim into a *traceable* one: a reviewer can follow `ta:evidencedBy` to the exact grounding passage, with a machine-readable confidence.

### 5.3 Why Gap 2 is the derivation engine

Gaps 1 and 3 publish facts *about* the existing data. Gap 2 introduces facts that were *never in the data in any form* — a predicted spectrum, a monoisotopic mass, a specific evidencing passage. This is the "derivation engine" in the strict sense: new scientific content, computed on demand, written back queryably. It is also the most modular — each property or evidence type is one more `dispatch` and one more emit-schema entry, so the engine grows monotonically without touching anything already shipped.

---

## 6. Gap 3 — Making the graph contain its own emergent structure

### 6.1 The gap, precisely

Read `reference.py` and the shared structure is unmistakable: all three reactions take `OXOGLUTARATE` as amino-acceptor, all three yield `L_GLUTAMATE` as amino-product, all three depend on `PLP`. The three enzymes are not three isolated facts — they are a **star network around a shared metabolic hub**. Yet the graph, which asserts each reaction independently, contains no triple stating that this hub exists or that these reactions share it. The topology is emergent in the data and unrecorded in the graph.

### 6.2 The module: `purpose-carry` (Meta/Context role) with the kernel gate

`purpose-carry` is Buhera's Meta module: it reads the accumulated audit log / session state and returns *derived* information about the whole, not about any single step. This is exactly the right instrument for emergent structure — it reasons over the set of derivations, not over one reaction. Backed by the kernel's gate (contribution / holonomy / sufficiency / order-parameter `R_ens`), it identifies the shared-hub topology and emits it as explicit relational triples.

### 6.3 Triples emitted

```turtle
res:species/2-oxoglutarate  a ta:MetabolicHub ;
    ta:sharedAcceptorOf  res:reaction/ALT , res:reaction/AST , res:reaction/CysAT .

res:species/L-glutamate  a ta:MetabolicHub ;
    ta:sharedProductOf   res:reaction/ALT , res:reaction/AST , res:reaction/CysAT .

res:reaction/ALT  ta:sharesHubWith  res:reaction/AST , res:reaction/CysAT .
res:reaction/AST  ta:sharesHubWith  res:reaction/ALT , res:reaction/CysAT .
res:reaction/CysAT ta:sharesHubWith res:reaction/ALT , res:reaction/AST .

res:enzyme/2.6.1.2  ta:sharesCofactorWith  res:enzyme/2.6.1.1 , res:enzyme/2.6.1.3 .
```

plus an order-parameter annotation `ta:ensembleCoherence "…"^^xsd:decimal` recording the gate's `R_ens` for the reaction set — a single number stating *how tightly* this cluster coheres, which generalises directly to larger reaction sets.

### 6.4 Why Gap 3 is the most "categorical OS"

Gaps 1 and 2 are per-entity. Gap 3 is *relational*: it states facts whose subject is the network, not any node. This is where Buhera's categorical machinery earns its place — the kernel's order parameter turns a qualitative observation ("these share a hub") into a quantitative, queryable claim ("this set coheres to degree `R_ens`"), and the same machinery scales without change to the KEGG-sized reaction networks nfdi4cat ultimately cares about. On three reactions it is a clean demonstration; on three thousand it is the only tractable way to surface the structure.

---

## 7. Provenance, detachability, and correctness guarantees

### 7.1 Provenance

Every enrichment triple is attached to a PROV-O `prov:Activity`:

```turtle
_:act_honjo_R00258  a prov:Activity ;
    prov:wasAssociatedWith  buhera:honjo ;
    prov:used               res:species/L-alanine , res:species/2-oxoglutarate , … ;
    prov:generatedAtTime    "2026-07-26T…"^^xsd:dateTime .
```

The computed fact is linked to the activity via RDF-star (`<< s p o >> prov:wasGeneratedBy … `) or standard reification, decided at implementation time by which the target triplestore indexes better. Either way: **no enrichment triple is anonymous.** A consumer can always ask "which Buhera module produced this, from what inputs, when?"

### 7.2 Detachability (the load-bearing guarantee)

Because the enrichment lives entirely in a separate named graph and is never merged into `base` at rest, the following invariant holds and will be tested (§8):

> `DROP GRAPH <…/enrichment>` returns the served artefact to byte-for-byte the current deliverable.

This is what lets nfdi4cat adopt the layer with zero risk: it is not a fork of the pipeline, it is a strictly-optional overlay.

### 7.3 Correctness

- **Gap 1** is self-checking: `honjo`'s residuals must equal the Python test's residuals (both zero). Divergence is a caught bug, not a silent error.
- **Gap 2** properties are deterministic where possible (masses) and clearly labelled as *predicted* where not (spectra) — never conflated with measured data.
- **Gap 3** claims are entailments over the asserted roles; they are checkable by a SPARQL query against the base graph alone, so the enrichment can be validated *against its own inputs*.

---

## 8. Evaluation plan

The proposal is falsifiable. We will demonstrate:

1. **Detachability test.** Automated test asserting `base` is unchanged by loading and dropping `enrichment` (bit-identical serialisation).
2. **Gap 1 agreement test.** `honjo` residuals equal `pytest` residuals for all three reactions.
3. **Gap 2 determinism test.** Monoisotopic masses reproduce reference values to specified tolerance; predicted spectra are stable across runs.
4. **Gap 3 entailment test.** The emitted `sharesHubWith` / `MetabolicHub` triples are exactly recoverable by a SPARQL query over the base graph — i.e. the derivation asserts nothing not entailed by the inputs.
5. **Provenance completeness test.** Every triple in `enrichment` has a `prov:wasGeneratedBy` path to a Buhera module.
6. **End-to-end demo.** From the Buhera terminal, live `dispatch` of each derivation against the loaded KG, showing the same code path serves both the batch build and the interactive session.

Success is: all six pass, and a SPARQL query that could not be answered before the layer (e.g. "return every reaction, its mass residual, its predicted product spectrum, and its hub partners, with per-fact provenance") returns rows after it.

---

## 9. Scope, and honest limitations

### 9.1 What is in scope

The three gaps above, delivered against the existing three-reaction KG, using the real Buhera modules `honjo`, `lavoisier`, `graffiti`/`purpose`, and `purpose-carry`, through the frozen dispatch trait, written back under PROV-O into a detachable named graph.

### 9.2 What is deliberately out of scope

Buhera also contains modules whose value appears only when the data has a **process or time axis** — live-process monitoring, closed-loop bioprocess control, temporal/manufacturing scheduling. The transaminase KG is a *static* structural artefact; these modules would have nothing to act on here, and including them would be dishonest padding. We name them and exclude them explicitly: **they are where Buhera goes when nfdi4cat's data grows a process/time dimension** (operando catalysis, reactor telemetry, time-series experiments), which is a natural and large follow-on but not this proposal.

### 9.3 Known risks

- **Structure resolvability (Gap 2 spectra/fingerprints).** Where ChEBI does not expose a resolvable structure, `lavoisier` falls back to formula-only properties (masses) and omits the spectrum rather than inventing one. The graph never carries a fabricated structure-derived value.
- **Evidence precision (Gap 2 evidence).** `graffiti` returns the best-matching passage with a χ-confidence; low-confidence links are emitted with their low score, not suppressed and not inflated. Consumers filter on `ta:confidence`.
- **Predicate minting.** Every new `ta:` predicate is documented in the T-Box and, where a community term exists (QUDT/OM units, PROV-O provenance, SKOS mappings), the community term is used in preference to a minted one.

---

## 10. Summary

The nfdi4cat KG asserts facts faithfully. The three things it structurally cannot do — record its own correctness, contain anything not hand-typed, or state its own emergent structure — are each a matter of *derivation*, not assertion, and each maps cleanly onto a Buhera OS module that already exists. The Buhera Derivation Layer computes all three and writes them back as provenance-stamped triples in a detachable named graph, through the single frozen `dispatch` funnel, touching the base pipeline through exactly one small, removable adapter.

The result is strictly additive: at worst inert, at best a graph that answers questions the base graph cannot — with every new answer traceable to the module that derived it. It is offered not as a replacement for the pipeline but as its optional, verifiable ceiling.
