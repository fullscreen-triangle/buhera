# Scope Routines

**What you'll learn:** how to use SCOPE — the helicopter microscopy DSL —
inside Buhera OS. Unlike shapeshifter (which compiles a script per cell),
SCOPE is a **REPL**: each cell adds a declaration to a growing program,
and cells that reach a `visualise(...)` produce a chart against a linked
image.

**Time:** ~15 minutes.

**Prerequisites:** [Basic routines](./basic-routines).

**Runtime requirement:** the `scope-lang` package must be linked to the
helicopter source, not the stub (see
`docs/modules/helicopter-integration-notes.md`). If not, every cell below
returns "runtime not installed."

---

## 0. What makes SCOPE different

Shape Shifter compiles a fresh script per dispatch. SCOPE **accumulates**.
Each cell adds a construct — a coordinate space, a channel definition, a
morphism, a goal, a dispatch rule — to the same growing script. When a
cell reaches a `visualise(...)` call, the accumulated script runs against
a linked image and produces a chart.

That means the natural workflow is:

1. Link an image once.
2. Declare a coordinate space.
3. Declare channels.
4. Write morphisms (`observe(...) |> ...`).
5. Ask for a visualisation.

Each of those is one cell. You never re-declare things you already
declared.

---

## 1. State and reset

**Cell 1.1** — See the current state.
```
dispatch("scope", { kind: "state" })
```

**Expected** — the accumulated script so far, plus any linked images and
the running session's status. On a fresh page: mostly empty.

**Cell 1.2** — Reset (clear all accumulated declarations).
```
dispatch("scope", { kind: "reset" })
```

**Expected**
```
scope: session reset.
```

---

## 2. Natural syntax — no wrapper needed

The router in `BuheraTerminal.js` recognises a SCOPE cell by its leading
keyword or by a morphism assignment. You can just type SCOPE directly.

**Cell 2.1** — Declare a coordinate space.
```
coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.10  lambda_t 0.05 }
```

**Expected** — the declaration is added to the accumulated script. No
chart yet; that comes with `visualise`.

**Cell 2.2** — Declare channels.
```
channels { GFP 488nm  Hoechst 405nm  BF }
```

**Expected** — three channels registered.

**Cell 2.3** — Confirm the script grew.
```
dispatch("scope", { kind: "state" })
```

**Expected** — the state now includes both declarations.

---

## 3. Morphisms — the observation and processing chain

Morphisms have the form `name = observe(...) |> step |> step |> ...`. Each
step transforms the observed image.

**Cell 3.1** — Simple morphism.
```
seg = observe(Hoechst) |> segment(threshold: 0.4) |> label
```

**Expected** — the morphism `seg` is added.

**Cell 3.2** — Chain more steps.
```
counts = observe(GFP) |> segment(threshold: 0.3) |> count
```

**Expected** — `counts` is added.

---

## 4. Goals and rules

Goals declare what the script is trying to compute. Rules constrain how.

**Cell 4.1** — Goal.
```
goal { count_cells(seg): "count Hoechst-positive nuclei" }
```

**Cell 4.2** — Rule.
```
rule { min_intensity(Hoechst, 0.2): "reject dim regions" }
```

**Expected** — both added to the script.

---

## 5. Visualise — the render step

A `visualise(...)` cell asks SCOPE to run the accumulated script against
a linked image and render a chart.

**Cell 5.1** — Assumes an image has been linked to the session (via the
image-picker UI or `linkScopeImage` in the terminal harness). If no image
is linked, this call will report so.
```
dispatch("scope", "visualise(seg)")
```

**Expected** — a chart showing the segmentation result, or a message
saying no image is linked.

---

## 6. Explicit dispatch shape

When you want to be explicit (e.g. inside a kwasa-kwasa script):

**Cell 6.1**
```
item r = dispatch("scope", { kind: "cell", source: "seg = observe(Hoechst) |> segment(threshold: 0.4) |> label" })
```

**Expected** — same effect as typing the morphism directly.

**Cell 6.2** — Inspect the state through a script.
```
item s = dispatch("scope", { kind: "state" })
print("script length: {}", s.output_delta.script.length)
```

---

## 7. A complete micro-workflow

Reset, link an image (via UI), and run through a full analysis.

**Cell 7.1** — Reset.
```
dispatch("scope", { kind: "reset" })
```

**Cell 7.2** — Space and channels.
```
coordinate_space { field 200 x 200 µm  depth 8  lambda_s 0.05  lambda_t 0.02 }
channels { DAPI 405nm  GFP 488nm  BF }
```

**Cell 7.3** — Morphisms.
```
nuclei = observe(DAPI) |> segment(threshold: 0.5) |> label
signal = observe(GFP) |> mask(nuclei) |> intensity
ratio = signal / count(nuclei)
```

**Cell 7.4** — Goal.
```
goal { report(ratio): "GFP intensity per nucleus" }
```

**Cell 7.5** — Visualise.
```
dispatch("scope", "visualise(ratio)")
```

**Expected** — the ratio chart, if an image is linked.

---

## 8. Composing with the rest of the OS

Every SCOPE cell dispatch is fed into the purpose session, so you can
still do the tandem carry over an accumulated microscopy analysis.

**Cell 8.1** — After running the workflow above, ask what's necessary for
a follow-up about GFP intensity.
```
dispatch("purpose-carry", {
  kind: "carry",
  goal: "GFP intensity segmentation ratio"
})
```

**Expected** — the SCOPE cells that touched GFP, mask, and intensity in
`keep`; unrelated acts in `dropped`.

---

## 9. What you now know

- SCOPE is a REPL: cells accumulate into one growing script. Reset with
  `{ kind: "reset" }` when starting a new analysis.
- Natural syntax works — the terminal detects SCOPE keywords and morphism
  assignments and routes to the scope module without needing
  `dispatch(...)`.
- The five construct kinds are `coordinate_space`, `channels`, morphism
  assignments, `goal`, and `rule`.
- A `visualise(...)` call runs the accumulated script against a linked
  image and renders a chart.
- Explicit dispatch (`dispatch("scope", { kind: "cell", source: ... })`)
  is available for kwasa-kwasa composition.

---

## Troubleshooting

- **"runtime is not installed"** — `node_modules/scope-lang` is pointing
  at the stub. Follow the recovery in
  `docs/modules/helicopter-integration-notes.md`.
- **"no image linked"** — SCOPE needs a linked image to visualise. Use
  the image-picker UI in the terminal, or call `linkScopeImage(url)` in
  code.
- **Morphism unknown** — the `observe(...) |> step` step name isn't
  recognised. Check the helicopter runtime's supported morphism steps.
- **The state gets huge across sessions** — reset between analyses. SCOPE
  accumulates within the browser page's lifetime.
