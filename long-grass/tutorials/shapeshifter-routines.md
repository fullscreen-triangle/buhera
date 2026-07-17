# Shapeshifter Routines

**What you'll learn:** how to use the Shape Shifter DSL — lavoisier's
notebook language for mass-spectrometry workflows. You'll compile and
run `.ss` scripts inside the Buhera terminal, one cell at a time, and
see forward-simulated spectra, S-entropy panels, partition addresses,
and fragment coherence.

**Time:** ~15 minutes.

**Prerequisites:** [Basic routines](./basic-routines).

**Runtime requirement:** the `@lavoisier/shapeshifter` package must be
linked (see `docs/modules/lavoisier-integration-notes.md`). If not, every
dispatch below returns "runtime not installed."

---

## 0. Two modules, one library

There are two distinct entry points to lavoisier in the federation:

- **`lavoisier`** — a thin wrapper that calls `runExperiment()` with a
  config object and returns predicted records. Good for one-shot
  simulations.
- **`shapeshifter`** — the full DSL. Compile `.ss` scripts with phase
  blocks that invoke lavoisier capabilities and return workspace values
  that render as inline charts.

This tutorial is about shapeshifter. If you just want a canned mass-spec
demo, `dispatch("lavoisier", "demo")` is quicker.

---

## 1. The demo

**Cell 1.1**
```
dispatch("shapeshifter", "demo")
```

**Expected** — a compile log, then an inline chart grid of 135 PC-lipid
records: precursor m/z on x-axis, intensity on y-axis, coloured by adduct
(`[M+H]+`, `[M+Na]+`, `[M+K]+`).

If you see the log but no chart, the shape-shifter to chart renderer
mapping is off; see the troubleshooting section.

---

## 2. Single-phase scripts — the common case

You do NOT need a full `objective`/`instrument`/`phase` program. A single
`phase` block is enough.

**Cell 2.1** — Compute an S-entropy coordinate from frequencies.
```
dispatch("shapeshifter", "phase p:\n  se = lavoisier.observe.sentropy(frequencies: [1200, 1550, 2900])")
```

**Expected** — an S-entropy coordinate panel showing the three-axis
`(S_k, S_t, S_e)` value.

**Cell 2.2** — Run a small lipidomics experiment.
```
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PC\"], polarity: \"+\", analyser: \"orbitrap\")")
```

**Expected** — the same 135-record chart grid you saw in the demo.

**Cell 2.3** — Compute partition addresses.
```
dispatch("shapeshifter", "phase p:\n  a = lavoisier.observe.ternary_address(coord: {S_k: 0.4, S_t: 0.2, S_e: 0.3}, depth: 6)")
```

**Expected** — the ternary address of that point at depth 6, rendered as
a short digit string like `0.1.2.0.1.1`.

---

## 3. Multi-value phases

A phase can produce multiple values, and each renders as its own panel.

**Cell 3.1**
```
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PC\", \"PE\"], polarity: \"+\", analyser: \"orbitrap\")\n  se = lavoisier.observe.sentropy(frequencies: [1200, 1550, 2900])\n  a = lavoisier.observe.ternary_address(coord: {S_k: 0.3, S_t: 0.3, S_e: 0.3}, depth: 4)")
```

**Expected** — three panels stacked: the records grid, the S-entropy
panel, and the ternary address.

---

## 4. Fragment / MSMS capabilities

Shape Shifter exposes several MSMS-side probes.

**Cell 4.1** — Phase coherence over a fragmentation ladder.
```
dispatch("shapeshifter", "phase p:\n  pc = lavoisier.msms.phase_coherence(analyte: \"PC(34:1)\", collisionEnergy_eV: 25)")
```

**Expected** — a subharmonic-bar panel showing coherence at each
fragmentation step.

**Cell 4.2** — Virtual tensor summary.
```
dispatch("shapeshifter", "phase p:\n  vt = lavoisier.msms.virtual_tensor(analyte: \"PC(34:1)\", polarity: \"+\")")
```

**Expected** — a tensor report card summarising the virtual-tensor
components.

**Cell 4.3** — Impossible-ion probe (crossing symmetries).
```
dispatch("shapeshifter", "phase p:\n  ii = lavoisier.msms.impossible_ions(analyte: \"PC(34:1)\", polarity: \"+\")")
```

**Expected** — a crossing-symmetry probe table.

---

## 5. Purpose-domain capabilities

Shape Shifter's `purpose` namespace runs domain-reduction and matching
under a stated experimental purpose.

**Cell 5.1** — Domain reduction under a stated purpose.
```
dispatch("shapeshifter", "phase p:\n  d = lavoisier.purpose.domain(purpose: \"identify phosphatidylcholines in serum\", classes: [\"PC\", \"PE\", \"PS\"])")
```

**Expected** — a purpose-domain reduction card showing which classes
survived the reduction.

**Cell 5.2** — Purpose match.
```
dispatch("shapeshifter", "phase p:\n  m = lavoisier.purpose.match(purpose: \"identify phosphatidylcholines\", records: [])")
```

**Expected** — a match card (empty records is a no-op input; provide real
records from a previous cell to see a real match).

---

## 6. Explicit programs

If you want to write a full script with `objective` and `instrument`
blocks:

**Cell 6.1**
```
dispatch("shapeshifter", {
  kind: "run",
  source: `objective o:
  target: "PC(34:1)"
  polarity: "+"
  analyser: "orbitrap"

phase p:
  r = lavoisier.instrument.run_experiment(classes: ["PC"], polarity: o.polarity, analyser: o.analyser)
  se = lavoisier.observe.sentropy(frequencies: [1200, 1550, 2900])
`
})
```

**Expected** — the objective is registered, the phase runs, and both the
records grid and the S-entropy panel render.

---

## 7. Reading a records value

Every workspace value has a `kind` that selects which panel renders it.
For records:

| kind           | panel                                    |
|----------------|------------------------------------------|
| `records`      | full record chart grid                   |
| `cells`        | ΔP cell registry table                   |
| `addresses`    | partition address table                  |
| `sentropy`     | S-entropy coordinate                     |
| `subharmonics` | fragment subharmonic bars                |
| `tensorReport` | virtual-tensor summary                   |
| `impossible`   | crossing-symmetry probe                  |
| `transient`    | single-transient contents                |
| `complement`   | SWIFT antistate                          |
| `domain`       | purpose-domain reduction                 |
| `validation`   | dual-path validation                     |
| `scalar` / `string` / `number` | value card               |
| anything else  | JSON dump                                |

If a value renders as a JSON dump you didn't want, its `kind` is not one
the terminal has a renderer for.

---

## 8. Composing with the rest of the OS

Shape Shifter dispatches feed the purpose session like every other
module. So a workflow of "run three experiments, then decide which to
keep for a synthesis" is natural.

**Cell 8.1** — Populate.
```
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PC\"], polarity: \"+\", analyser: \"orbitrap\")")
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PE\"], polarity: \"-\", analyser: \"tof\")")
dispatch("shapeshifter", "phase p:\n  r = lavoisier.instrument.run_experiment(classes: [\"PS\"], polarity: \"+\", analyser: \"quadrupole\")")
```

**Cell 8.2** — Purpose-carry decides.
```
dispatch("purpose-carry", {
  kind: "carry",
  goal: "phosphatidylcholine orbitrap positive mode identification"
})
```

**Expected** — the PC/orbitrap/positive act in `keep`, the others in
`dropped` or `regenerable`.

---

## 9. What you now know

- Shape Shifter's dispatch entry accepts a bare string as `.ss` source.
  You almost always want one `phase` block per cell.
- Capabilities are `lavoisier.<namespace>.<function>` — the namespace
  tells you what kind of panel you'll get.
- The workspace-value `kind` field selects the renderer; unrecognised
  kinds fall back to JSON.
- Shape Shifter is a real DSL, not just a wrapper. You can chain
  capabilities, produce multiple values per phase, and pipe results
  between cells.

**Next up:** [Scope routines](./scope-routines) — the helicopter
microscopy DSL, structured differently: cell-by-cell accumulation into a
script, with `visualise()` cells that produce charts.

---

## Troubleshooting

- **"runtime is not installed"** — the `@lavoisier/shapeshifter` link is
  broken. Follow the recovery instructions in
  `docs/modules/lavoisier-integration-notes.md`.
- **Compile log but no chart** — the returned workspace value's `kind`
  has no renderer. Check the console for the last value's kind and add
  a case to `WorkspaceValue.js` if it's missing.
- **"Unknown function `lavoisier.foo.bar`"** — the `fn` name isn't
  dispatched by the compiler. Check spelling against the capability list
  in the lavoisier integration notes.
- **Records chart is empty** — the experiment produced zero records
  matching your polarity/analyser combo. Try `analyser: "orbitrap"` and
  `polarity: "+"` (the most permissive combination).
