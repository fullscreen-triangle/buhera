# CKG Modelling Facts — the ground truth, each tied to a method and a script

This is the fact ledger for the **causal-knowledge-graph (CKG) experiment**. Every
row is a fact the federation can *emit onto a CKG node*, tied to two things:

1. **Grounding** — the paper / method / monograph the number or claim comes from.
2. **DSL script** — the exact `dispatch(...)` cell that produces it, and the
   `attach` cell that folds it onto a CKG node as a value-delta.

It exists because the module *integration notes* (`docs/modules/*-integration-notes.md`)
are **wiring documents** — they tell you how to install and call a module, not what
scientific claim each output stands on. The groundings live in the **module source**
and in the **CKG runtime paper**, and that is what is cited here. Where a fact has no
external-paper grounding — only an in-repo definition — this document says so plainly
rather than inventing a citation.

## How a fact lands on a CKG node

A CKG node is a triple `(τ, Chunks(node), Vals(node))` (paper `def:node`). A module
becomes a **chunk** on a node; when the node is dispatched (directly or by the carry),
the chunk calls the real module and folds its `output_delta` onto the node as a value
keyed `fact:<module>#<chunk>` — see [`ckg-module.js`](../../src/lib/modules/ckg-module.js)
`moduleChunk()`. The chunk name disambiguates the channel, so **one node can carry many
facts from the same module** without collision. The runtime *judges nothing*: a module
that errors folds an `error` value, not a halt (paper `thm:no-exit`).

The two-cell shape for every fact below is therefore:

```
dispatch("ckg", { op: "represent", tau: "<node>", seed: 1 })
dispatch("ckg", { op: "attach", tau: "<node>", name: "<chunk>", module: "<module>", instruction: <instr> })
```

then `{ op: "carry" }`, `{ op: "graph" }`, `{ op: "report" }`, `{ op: "fingerprint" }`
to run, project, account for, and hash the trajectory.

The authoritative list of fact-kinds a node can carry is the set of `case` labels in
`deriveFindings()` in [`ckg-module.js`](../../src/lib/modules/ckg-module.js): the
runtime facts (Part A), the four modelling-DSL facts — cytochrome `cyp_*` (Part B1),
shapeshifter `shapeshifter_run` (B4), sbs `sbs_result` (B3), honjo (B2) — and the other
federation-module facts (Part C).

---

## Part A — Runtime facts (grounding: the CKG runtime paper)

These are facts about the *run itself*, not about any domain. Grounding is the paper
`docs/causal-knowledge-graph-runtime/causal-knowledge-graph-runtime.tex`; each claim is
also checked by a script in that paper's `validation/` directory.

| Fact | Grounding (paper claim) | Validation script | DSL that produces it |
| --- | --- | --- | --- |
| A node is a triple `(τ, Chunks, Vals)`; structure is handed in before any run | `def:node` | — | `dispatch("ckg", { op: "represent", tau: "sample", seed: 1 })` |
| Edges are **run-induced** ("a value emitted at u was read at v"), never declared | `prop:no-edges` | `trajectory_emergence.py` | `dispatch("ckg", { op: "carry" })` → `{ op: "graph" }` (edge list) |
| Trajectory **emerges** from the run; it is not authored | `thm:emergence` | `trajectory_emergence.py` | seed then carry: `{ op: "seed", tau, value }` then `{ op: "carry" }` |
| The run has **no exit code / no verdict**; errors become recorded facts | `thm:no-exit` | `run_to_completion.py` | any `{ op: "dispatch", tau }` — an erroring chunk folds an `error` value |
| Every act runs; nothing is gated on a verdict (**run to completion**) | `cor:completion` | `run_to_completion.py` | `{ op: "report" }` → `audit` array is the completion witness |
| Multi-chunk execution: `dispatch(node) = ⟨dispatch(c) : c ∈ Chunks(node)⟩` | `def:exec` | — | attach ≥2 chunks to one τ, then `{ op: "dispatch", tau }` |
| **Intended non-determinism**: a different seed reshapes which reads happen | `prop:nondeterminism` | `nondeterminism_provenance.py` | re-`{ op: "seed" }` a node then re-`{ op: "carry" }` — graph changes |
| **Resolution / prefix containment**: a k-prefix edit has blast radius bounded by `b^(D−k)` | `prop:prefix`, `def:resolution` | `resolution_control.py` | `{ op: "fingerprint", edits: [["ckg","measure"]] }` |
| Reproducibility relocates from results to **protocol + provenance**: the fingerprint hashes τ + sorted chunk names + edit addresses and **excludes values** | `prop:prefix` (fingerprint invariance) | `resolution_control.py` | `{ op: "fingerprint" }` — same protocol ⇒ same hash across different seeds |

The three projections that replace the old ontology→reasoner→SPARQL pipeline
(grounding: `ckg-module.js` module header):

| Old pipeline stage | CKG runtime replacement | DSL |
| --- | --- | --- |
| ontology model | node `τ` + attached chunk-bag | `{ op: "represent" }`, `{ op: "attach" }` |
| reasoner (schema entailment) | federation dispatch (emitted value-deltas) | `{ op: "dispatch" }`, `{ op: "carry" }` |
| SPARQL access | walk the graph the run produced | `{ op: "graph" }`, `{ op: "report" }` |

---

## Part B1 — Cytochrome-P450 domain facts (grounding: the P450 monograph + real `.shk` lessons)

Source: [`src/lib/modules/cytochrome-module.js`](../../src/lib/modules/cytochrome-module.js).
Its header states: *"Every number below is read from the monograph, not invented."*
Each op returns an `output_delta` whose `kind` has a headline case in
`deriveFindings()`. Dispatch directly as `dispatch("cytochrome", { op: ... })`, or fold
onto a CKG node with `attach`:

```
dispatch("ckg", { op: "attach", tau: "<node>", name: "<chunk>",
                  module: "cytochrome", instruction: { op: "<op>", ... } })
```

**The native example script for each fact** is the real Shakespeare `.shk` lesson in
`levinthal/cytochrome/src/data/lessons.js` (11 plays of increasing categorical depth, each
*performing* one monograph paper and checked against a validation `oracle`). The lesson is
the `.shk` grounding the directive asks for; the `{ op }` cell is the CKG adapter that folds
the same fact onto a node. The lesson's `oracle` field carries the validated number.

| Fact-kind (`delta.kind`) | The claim / numbers (from the monograph) | Grounding method | Native `.shk` lesson (monograph paper · oracle) | DSL: `dispatch("cytochrome", …)` |
| --- | --- | --- | --- | --- |
| `cyp_electron_transfer` | NADPH→FAD→FMN→heme chain; categorical distance `d_C = 4`; **Marcus λ = 0.85 eV**; Fe centre (17.26, 11.53, 24.76) Å; FMN→heme rate-limiting at 5×10⁶ s⁻¹, ΔM 7.60 | **Marcus electron-transfer theory** | `07_electron-chain` (Paper 4 · multi-hop-et-chain; oracle: hops FAD→FMN→heme-Fe, femtosecond, Δs_orbital=0 conserved) | `{ op: "electron-transfer" }` (also the `"demo"` / bare-string default) |
| `cyp_compound_i` | Compound I as a `d_C = 1` aperture of depth **ΔM = ln2 ≈ 0.693**; mechanism **PCET**; KIE ≈ 1.7; S-coordinate (0.86, 0.515, 0.595); Mössbauer isomer shift 0.11 mm/s | **Rittle–Green Compound I observables**; proton-coupled electron transfer | `05_compound-i` (Paper 5 · compound-i-formation; oracle: Fe(IV)=O, heterolysis ΔM 0.693, lifetime ≈1 ms, licensed by cycle closure) | `{ op: "compound-i" }` |
| `cyp_pathway` | Reaction families sorted by aperture ΔM: S-ox (0.28) < N-ox (0.32) < N-dealk (0.50) < O-dealk (0.58) < aliphatic C–H (0.65); large KIE ⇒ C–H broken in the RDS (HAT), KIE≈1 ⇒ direct O-transfer | **Kinetic isotope effect (KIE)** as HAT diagnostic; hydrogen-atom-transfer / rebound mechanism | `06_ch-rebound` (Paper 6 · ch-activation-rebound; oracle: HAT + oxygen rebound, HAT_ΔM 0.65, product R-OH) | `{ op: "pathway", reaction: "aliphatic-hydroxylation" }` |
| `cyp_states` | Seven-state closed catalytic orbit (resting → substrate-bound → ferrous → oxy-ferrous → peroxo → Compound I → product); orbit **ΣΔM = 4.963**; each state carries an (n,ℓ,m,s) address | P450 catalytic cycle (closed-orbit / ping-pong analogue) | `04_catalytic-cycle` (Paper 14 · seven-state-closed-orbit; oracle: 7 states, 8 transitions, ΣΔM 4.963, orbit closed) | `{ op: "states" }` |
| `cyp_spectroscopy` | Combined Soret + EPR + Raman signature (headline blob) | UV/vis + EPR + resonance-Raman spectroscopy | `03_resting-state` (Paper 3 · cyp3a4-resting-substrate-bound; oracle: Fe³⁺ low-spin, S=½, Cys442 thiolate) | `{ op: "spectroscopy" }` |
| `cyp_soret` | **Soret band 417 → 392 nm** (Δ −25) on resting → Compound I; reads porphyrin π→π* oxidation | UV/vis (Soret) spectroscopy | `03_resting-state` → `05_compound-i` (resting Fe³⁺ LS to Fe(IV)=O; the Soret shift reads that transition) | `{ op: "soret" }` |
| `cyp_epr` | Low-spin Fe³⁺ rhombic **g-tensor (2.42, 2.25, 1.92)** — the CYP resting-state fingerprint | EPR spectroscopy | `03_resting-state` (Paper 3; oracle: resting Fe³⁺ low-spin S=½ — the state the g-tensor fingerprints) | `{ op: "epr" }` |
| `cyp_raman` | **Fe=O stretch 795 cm⁻¹**, shifting to 758 cm⁻¹ on ¹⁶O→¹⁸O (Δ −37) — confirms an oxo-ferryl oscillator | Resonance-Raman spectroscopy + isotope substitution | `05_compound-i` (Paper 5; the Fe(IV)=O species whose stretch the isotope shift confirms) | `{ op: "raman" }` |
| `cyp_isoform` | 57 human CYPs on a base-3 address manifold; **CYP2D6 phenotype ΔM {UM .27, EM .55, IM .75, PM 2.50}**; **CYP2C9\*3 ΔM 3.60**; CYP3A4 depth 5.69 | Pharmacogenomics (isoform / allele ΔM shifts) | `08_isoform-diversity` (Paper 12 · 57-isoform-taxonomy; oracle: 57 isoforms, families depth 3, isoforms depth 6) + variant re-cut `09_variant-effect` (Paper 15; ARG144CYS coherence 0.83→0.79) | `{ op: "isoform", cyp: "CYP2D6", phenotype: "PM" }` |
| `cyp_participants` | The **participant/carrier cut**: heme Fe (protoporphyrin IX), FAD, FMN are *carriers* (bound once, in no equation); substrate + O₂ + 2e⁻ + 2H⁺ → product + H₂O are *participants* | The levinthal transaminase kernel (`pingpong-bibi-conditioned-floor`) ported to P450 | `03_resting-state` closure `complex CYP3A4(cofactor "heme", residue "CYS442", solvent "H2O_axial")` names the carriers; `04_catalytic-cycle` tracks the participants | `{ op: "participants", reaction: "aliphatic-hydroxylation" }` |
| `cyp_floor` | Conditioned admissibility floor **β = disc + Q + conv**; at categorical depth d=9 the conversion term dominates ⇒ β effectively condition-independent; Q-regime ⇒ conditions govern comparability | The conditioned floor from `conditioned_floor.py` (levinthal) | `10_db-recovery` (Paper 13 · database-recovery; oracle: recovery = constraint-propagation fixed point, **stored = 0**, converged) — the empty-dictionary lesson | `{ op: "floor" }` |

The cytochrome module also ships two **cross-DSL scripts** (source: `cytochrome-module.js`,
exports `P450_ET_SBS` and `P450_MS_SS`), so the same P450 reaction node can carry an
`sbs` circuit fact and a `shapeshifter` spectra fact beside the native cytochrome facts:

- `P450_ET_SBS` — the NADPH→FAD→FMN→heme redox chain as an **SBS circuit** (μ in kJ/mol
  tracking the reduction-potential ladder; FMN→heme is the smallest conductance, the
  rate-limiting hop). Fold via `attach … module: "sbs"`.
- `P450_MS_SS` — a virtual orbitrap acquisition of a CYP substrate + oxidised metabolite,
  as a **`.ss` shapeshifter script**. Fold via `attach … module: "shapeshifter"`.

### External-paper grounding for the recovery / empty-dictionary facts

The `cyp_floor` fact and the `recovery` CKG node (§8 of the flagship tutorial) are the
"stores-nothing" claim, and it is grounded in **two real cytochrome papers on disk**, each
with a `validation/` directory of Python scripts and `results/*.json` whose numbers match
the `.tex` to the digit:

- **Database recovery** — `cytochrome/publications/informatics/database-recovery/database-recovery.tex`
  (Monograph **Paper 13**, the paper the `.shk` lesson `10_db-recovery` performs). 8 validation
  scripts (`01_information_capacity.py … 08_full_recovery_table.py`) + `results/*.json` + 8
  rendered PNG panels. The claim: given a partial observation, the receiver synthesises the
  full state by **constraint propagation to a self-consistent fixed point** (Kirchhoff
  balance, loop consistency, backward trajectories) — **not** by forward simulation from a
  stored record. `stored = 0`.
- **The empty dictionary for cytochrome P450** —
  `cytochrome/publications/informatics/empty-dictionary-p450/empty-dictionary-p450.tex`
  (Monograph **Paper 16**). Verified labels + numbers:
  - `prop:resident` / `tab:scaling` (`01_storage_scaling.py`): the *resident* state is
    **O(1) = 561 bytes** (20 coordinate rows + an encoding rule), constant while a derived
    cache grows 0.001 → 356.6 → 891 541 MB across n = 1…10⁹ (nine orders of magnitude);
    `resident_is_constant`: true.
  - `tab:query` / `prop:graded` (`02_query_without_entries.py`): the empty scheme answers
    **6/6** queries against an index control that answers **0/6**; the resident state's hash
    is **unchanged** before and after querying; graded recovery ρ = **0.382** over **190**
    pairs vs a 200-shuffle null (mean −0.004, max 0.252). The identity task is 1.00 vs 1.00
    (non-discriminating — the paper flags this honestly rather than claiming a win).
  - `thm:attribution` / `prop:cache-optional` (`03_paper2_reconciliation.py`): the "~350 MB"
    figure recomputes to **356.6 MB** and is the O(N·k) **cache** — *optional*, not the O(1)
    resident state; answers are identical with the cache dropped.

  **Two honesty flags carried, not propagated:** (1) the paper labels itself "Paper 16" in
  its header but "the seventeenth paper" in its conclusion — a monograph-numbering
  inconsistency, not a claim about the CKG; (2) the graded-recovery significance is reported
  `p < 0.005` in prose while the backing JSON records `p = 0.0` (a 200-shuffle floor, not a
  vanishing p-value). Neither number is repeated as fact here; both are noted so the ledger
  is honest about its own sources. The user's own source in that paper's bibliography
  (`sachikonye2025c`) is **not** cited.

---

## Part B2 — Cheminformatics (honjo `.hj`)

The honjo module folds a `.hj` cheminformatics run onto a CKG node. The native example
scripts are the four real programs in
`borgia/honjo-masamune/honjo/examples/{carbon,water,salt,track}.hj` (web-mirrored in
`honjo/src/pages/playground.js`), and the grounding is honjo's own **stdlib chemistry**
(valence / VSEPR / Δ-thickness in `honjo/src/stdlib.ts`) plus the DSL paper
`borgia/honjo-masamune/docs/honjo-dsl/honjo-masamune-dsl.tex` (*"The Honjo Masamune
Language — A Cut-Primitive Domain-Specific Language for Cheminformatics"*).

| Fact | The claim / real output | Grounding | Native `.hj` example |
| --- | --- | --- | --- |
| individuation | an atom is a length-one **cut** against the medium; C (Z=6) → `[He] 2s² 2p²`, term **³P₀**, **vacancy = 4** | honjo stdlib (electron-configuration + term-symbol rules) | `carbon.hj` (`C := cut 6` then `observe C`) |
| bond + closure geometry | `close O(H,H)` drives every vacancy to zero ⇒ **2:1 stoichiometry, bent, ≈104.5°**; a bond is admitted only if `delta > 0` (lowers thickness) | honjo stdlib VSEPR + the "no separation is free" floor | `water.hj` (`OH := O ~ H when delta > 0`; `W := close O(H,H)`) |
| ionic contact + negative control | `close Na(Cl)` → NaCl **1:1**; a closed-shell partner forms **no** bond — `Na ~ Ne` yields `exists = false` (Ne has vacancy 0) | honjo stdlib (open- vs closed-shell admissibility) | `salt.hj` (the `dead := Na ~ Ne` line is the built-in negative control) |
| causal tracking / amalgamation | tracking O through the process is admissible iff it **S-entropy-converges**; representation switching (mass, charge, time) allowed; the **amalgamation** is the result | `import honjo.causal`; the causal-propagation-table method | `track.hj` (`track O in W with reps mass, charge, time until converge yield amalgamation`) |

**Grounding strength:** paper + stdlib prose. These four chemistry examples have **no
per-example oracle JSON** — the honjo repo's validation JSONs check its physics papers, not
`carbon/water/salt/track.hj`. The outputs above are read from the example comments and the
stdlib rules they exercise, which is stated plainly rather than dressed as an oracle pass.

---

## Part B3 — Systems biology (SBS `circuit`)

The `sbs` module folds an S-entropy circuit observation (`sbs_result`: nodes, edges,
coherence **R**, flux visibility **V**, backend) onto a node. The native example scripts are
the six real circuits in `hegel/consequences/src/lib/sbs/dsl/examples.js`, and the grounding
is the paper `hegel/docs/publication/bayesian-network/st-stellas-circuits.tex` (*"S-Entropy
Coordinate Analysis of Electrical Circuits: … Tri-Dimensional Phase Space Navigation"*).

| Fact | The claim / real number | Native `circuit` example (`examples.js`) |
| --- | --- | --- |
| baseline circuit + perturbation | 10-step glycolysis circuit; `perturb { factor: 0.1 }` reads the disease response; μ Glucose = **−917.0** kJ/mol | `glycolysis` (the `DEFAULT_EXAMPLE`) |
| catalyst convergence | two catalysts combine as `1 − (1−0.3)(1−0.7)` = **0.79** (κ₁₂) — convergence, not addition | `catalyst_convergence` (inline comment computes 0.79) |
| triple equivalence | O ≅ C ≅ P as mutually reachable nodes; symmetric edges at **conductance 3.14** | `triple_equivalence` (`edge X -> Y { conductance: 3.14 }` both ways) |
| miracle principle | a locally-infeasible sub-task is admissible if the whole circuit closes | `unconstrained_subtask` |
| restoration | perturb then `restore` recovers flux visibility **V > 0.9** | `drug_design` |
| recursive depth | nested triples give **3ᵈ** substates (3 / 9 / 27) | `recursive_triple` |

**Grounding strength:** paper + the inline example numbers in `examples.js`. There is **no
on-disk oracle JSON** for these six circuits — the numbers above are the example comments and
the paper's own figures, and that is stated plainly.

---

## Part B4 — Mass spectrometry (shapeshifter `.ss`)

The `shapeshifter` module folds a virtual-MS workspace (`shapeshifter_run`: records,
S-entropy, tensor…) onto a node. The native example scripts are the eleven real `SS_*`
programs (virtual files `hello_lipid.ss`, `targeted_lipidomics.ss`, `proteomics_experiment.ss`,
`temporal_acquisition.ss`, `partition_addresses.ss`, and the S-entropy / SEBD / tensor / db
scripts) in `lavoisier/web/src/sandbox/ShapeshifterSandbox.js`. Grounding is the paper
`lavoisier/docs/publication/st-stellas-spectrometry.tex` (*"S-Entropy Spectrometry … Empty
Dictionary Systems …"*) **plus on-disk oracle JSON** —
`lavoisier/validation/ion_journeys/aggregate_summary.json` and the per-experiment
`validation/experiment_results/*/…_results.json`.

| Fact | The claim / real number | Native `.ss` example | Oracle |
| --- | --- | --- | --- |
| lipid acquisition | PC(32:0) precursor **m/z 734.5694**, positive mode, orbitrap | `hello_lipid.ss` / `targeted_lipidomics.ss` (`{ name: "PC(32:0)", mz: 734.5694 }`) | experiment_results lipid JSON |
| proteomics fragment ladder | lysine **[M+H]⁺ 147.1128** (immonium/fragment series) | `proteomics_experiment.ss` (`precursor_mz = 147.1128`) | `ion_journeys/aggregate_summary.json` (34 ions, 1292 theorems, pass_rate 1.0) |
| partition addressing | records are assigned S-entropy **partition addresses** (a ternary field over the acquisition) | `partition_addresses.ss` (`lavoisier.observe.partition_field(records)`) | partition validation JSON |
| tolerance window | matches are read within a **5 ppm** window | the `targeted_lipidomics` / db-search scripts | experiment_results JSON |

**Grounding strength: FULL** — paper + on-disk oracle, the cytochrome-grade pairing. The
`aggregate_summary.json` is a real validation artifact (an MS/MS theorem-check over 34 spike-
protein ions, all passing); the 734.5694 / 147.1128 constants live in the `SS_*` scripts and
their per-experiment result JSONs. The P450-specific acquisition the flagship tutorial folds
(`P450_MS_SS`, a CYP substrate + oxidised-metabolite scan) is grammatical against this same
`.ss` grammar.

---

## Part C — Federation-module facts (structural / allocation / identity quantities)

These modules carry **no domain content of their own** — the "empty-dictionary"
discipline (stated explicitly in the `purpose` and `musande` integration notes). They
emit *structural, allocation, and identity* quantities that become facts on a CKG node.
Only the fact-kinds with a headline case in `deriveFindings()` are listed as landing
cleanly on a node; each row gives the grounding paper/method from the module's own
integration note and the DSL that produces it.

| Module (id) | Fact-kind on node | The quantity | Grounding (paper / method) | DSL |
| --- | --- | --- | --- | --- |
| `shapeshifter` | `shapeshifter_run` | workspace values (records, S-entropy, tensor…) from a `.ss` run | lavoisier mass-spec interpreter (`@lavoisier/shapeshifter`); see **Part B4** | `dispatch("shapeshifter", <SS_* script>)` — e.g. the `P450_MS_SS` scan or `hello_lipid.ss` |
| `sbs` | `sbs_result` | cellular circuit + S-entropy observation: nodes, edges, coherence **R**, flux visibility **V**, backend | Systems-Biology-Shaders circuit model; see **Part B3** | `dispatch("sbs", <circuit source>)` — e.g. `P450_ET_SBS` or `glycolysis` from `examples.js` |
| `scope` | `scope_run` | microscopy observation count + summary | SCOPE microscopy REPL | `dispatch("scope", <scope block>)` |
| `lavoisier` | `lavoisier_run` | instrument run: record count, average S-entropy | lavoisier instrument model | `dispatch("lavoisier", <instrument block>)` — a `lavoisier.instrument.run_experiment(...)` phase |
| `graffiti` | `graffiti_result` | repo scan: project count, **ambient floor β** | graffiti/spraypaint: BM25 + water-filling; Stoer–Wagner min-cut identity (Invariants 1–4) | `dispatch("graffiti", <query>)` |

### Federation modules whose grounding is identity/allocation math (from the integration notes)

These are registered and dispatchable, but their native output-kinds fall through
`deriveFindings`' `default` case (they still land as facts via the `summary` headline).
Groundings are from the module integration notes, verbatim:

| Module (id) | Quantity it asserts | Grounding paper / method | DSL |
| --- | --- | --- | --- |
| `graffiti` / spraypaint | `fingerprint` (blake3) + **χ** (Stoer–Wagner min-cut ≥ floor); never-resetting `committed_count`; ranked context slice (search-not-fetch); water-filling `allocation` + clearing price `p*` | Four invariants: conserved identity, never-resetting count, search-not-fetch, exclusive phases | `dispatch("graffiti", "…")` |
| `smith` (agent-smith / musande) | agent **χ** (exact min-cut bipartition); realised floor **β**; water-filling attention; per-agent `count` never decreases (I2) | split-attention paper (`agent/society` DSL; parse→typecheck→compile→tick/town) | `dispatch("smith", <agent/society source>)` |
| `purpose` / `dsl-writer` | ranked context slice (`file:line [kind] name`); compiled vaHera fragment (dry-run); generated DSL + federation `confidence` | empty-dictionary principle; FKAC (N-draft federation + aggregate-floor confidence) | `dispatch("dsl-writer", { dslId, instructions, execute })` |
| `srn` (pylon) | agent **χ**; monotone count **m**; separation price **sep(e)**; attention price **p★**; Kuramoto order parameter **r** | `network-yield-computing-allocation.tex`; `split-attention-agents.tex` | `dispatch("srn", { kind: "nl", text: "…" })` |
| `zangalewa` | research card: title + S-coordinate **(S_k, S_t, S_e)** + sections + references | Minimum Sufficient Interceptor (coordinate-only hand-off) | `dispatch("zangalewa", "what is p53?")` |
| `desk` | standing intent; acts split into **necessary** (δS > 0) vs **purposeless** (δS = 0) | tagged-intent scoring against committed act history | `dispatch("desk", { kind: "surface" })` |

---

## Honesty note on groundings

- **Part A** groundings are load-bearing claims of the CKG paper, each with a named
  theorem/proposition label and a validation script. These are the strongest.
- **Part B1** groundings are the P450 monograph as encoded in `cytochrome-module.js`, now
  each tied to the real `.shk` lesson (with its own oracle) that performs the same paper. The
  physical theories named (Marcus, PCET/Rittle–Green, KIE/HAT, Soret/EPR/Raman,
  pharmacogenomic ΔM) are standard and correctly attached to the numbers the source
  carries. The *categorical* framing (ΔM apertures, address manifold, conditioned floor)
  is this project's own construction, ported from levinthal — not an external citation.
- **Part C** groundings are taken from each module's integration note. Those notes are
  wiring documents; the paper/method column reflects what the note itself claims the
  module implements (min-cut identity, split-attention, network-yield allocation,
  Kuramoto coherence, Minimum Sufficient Interceptor). No external citation is asserted
  beyond what the source and notes state.

### The grounding gradient across the four modelling modules

The four modelling DSLs are **not** grounded to the same strength, and this ledger says so
rather than implying uniform paper+oracle backing:

| Module | Native example scripts | Paper | On-disk oracle | Strength |
| --- | --- | --- | --- | --- |
| **cytochrome** (`.shk`) | 11 lessons in `lessons.js` | P450 monograph (Papers 3–16) | ✅ each lesson's `oracle` field (verdict PASS) | **Full** — paper + per-example oracle |
| **shapeshifter** (`.ss`) | 11 `SS_*` scripts | `st-stellas-spectrometry.tex` | ✅ `ion_journeys/aggregate_summary.json` + experiment_results JSON | **Full** — paper + on-disk oracle |
| **honjo** (`.hj`) | `carbon/water/salt/track.hj` | `honjo-masamune-dsl.tex` | ❌ no per-example JSON (the repo's JSONs validate physics papers, not these 4 chemistry examples) | **Medium** — paper + stdlib prose |
| **sbs** (`circuit`) | 6 circuits in `examples.js` | `st-stellas-circuits.tex` | ❌ no oracle file (numbers are inline example comments) | **Medium** — paper + inline example numbers |

For the recovery / empty-dictionary facts specifically, the grounding is stronger still: two
dedicated on-disk cytochrome papers (**database-recovery**, **empty-dictionary-p450**) each
with a full `validation/` directory whose `results/*.json` match the `.tex` to the digit —
see *External-paper grounding* under Part B1.

Nothing in this ledger is fabricated from memory. Every number in Part B1 is traceable to a
constant in `cytochrome-module.js` and a `.shk` lesson oracle; every honjo/SBS/shapeshifter
fact to a named on-disk example script (and, where the gradient says so, an oracle JSON);
every claim in Part A to a labelled result in the CKG `.tex`; every row in Part C to a module
integration note. Where a fact has only in-repo or example-comment grounding, the gradient
table above states that plainly.
