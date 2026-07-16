# Using SCOPE inside Buhera OS

How to run SCOPE — the microscopy-analysis DSL — from the Buhera OS terminal
(`long-grass`). SCOPE runs there as a **REPL**: you type declarations cell by
cell, they accumulate into one growing program (the "script so far"), and a
cell that reaches a `visualise(...)` renders a chart against a linked image.

---

## 0. Architecture at a glance

Three intentionally-separated layers — don't merge them:

- **SCOPE (a module / "player")** — correctness only. Lives as the npm package
  `scope-lang`, whose source is in `helicopter/scope-lang/src/`
  (`compiler/`, `runtime/` incl. `mic/`, `mic-engine/`, `session.ts`, `index.ts`).
- **Buhera OS frontend (`architecture/buhera/long-grass/`)** — a Next.js Pages
  Router app. The terminal lives in `src/components/BuheraTerminal.js`; modules
  live in `src/lib/modules/` and are registered against the Module trait in
  `registry.js`.
- **Orchestrator (kwasa-kwasa / turbulance)** — a *separate* project. SCOPE is
  **not** driven through it; SCOPE cells route straight to the scope module,
  bypassing the orchestrator by design.

The package is consumed by long-grass through the bare import `"scope-lang"`,
resolved by a `node_modules/scope-lang` junction.

---

## 1. ⚠️ Current state: the junction points at a STUB

Right now `long-grass/node_modules/scope-lang` is a junction to
`long-grass/vendor/scope-lang-stub` — a placeholder whose `createSession()`
returns `{ ok:false, error:"scope-lang runtime is not installed…" }`. This
exists so the Buhera build passes without the real package. **Every SCOPE cell
will fail with "runtime is not installed" until you swap the junction to the
real package.**

To activate the real SCOPE runtime, repoint the junction (PowerShell, run from
`long-grass/`):

```powershell
# remove the stub junction (does NOT delete vendor/scope-lang-stub itself)
Remove-Item node_modules\scope-lang -Force -Recurse

# link the real package source in helicopter
New-Item -ItemType Junction `
  -Path node_modules\scope-lang `
  -Target C:\Users\kunda\Documents\vision\helicopter\scope-lang
```

Notes:
- Use a **junction**, not `npm link`. `npm link` fails on this machine — long-grass
  has a git-URL dep (`lavoisier`) and re-running install chokes on a stale
  git-clone cache dir. The junction gives live propagation without npm.
- The real `scope-lang` ships **TS source** (`exports` → `src/index.ts`); Next
  transpiles it, so edits in `helicopter/scope-lang/src/` propagate live.
- `scope-lang/package.json` must expose `"./package.json": "./package.json"` in
  its exports map, or Next's resolver complains.
- To go back to the safe stub for a deploy that shouldn't carry the runtime,
  repoint the junction back at `vendor/scope-lang-stub`.

Verify: `npx next build` in long-grass should compile the whole app including
the linked TS source with type-checking on.

---

## 2. Two ways to reach SCOPE from the terminal

### a) Natural SCOPE syntax (recommended)

The router in `BuheraTerminal.js` recognises a line as a SCOPE cell when it
begins with a SCOPE keyword **or** is a morphism assignment, and sends it
straight to the scope module. Triggers:

- Leading keyword — one of:
  `coordinate_space`, `channels ` / `channels{`, `goal ` / `goal{`,
  `rule `, `dispatch ` / `dispatch{`
- Morphism assignment matching `^<ident> = observe(` — e.g. `seg = observe(...`

So you just type SCOPE, no wrapper needed.

### b) Explicit dispatch

You can also call the module directly:

```
dispatch("scope", "coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.10  lambda_t 0.05 }")
dispatch("scope", { kind: "state" })
dispatch("scope", { kind: "reset" })
```

---

## 3. The REPL model (Jupyter-style)

You write SCOPE's **existing** constructs **without** the `scope name { … }`
wrapper — one or more of: `coordinate_space`, `channels`, a `morphism`
(`name = observe(...) |> …`), `goal`, `dispatch`. There is **no new syntax**
(no import / funxn / link-file).

Cells accumulate into a session namespace:

- **Defining cells** (a `coordinate_space`, `channels`, or a morphism that does
  *not* reach `visualise`) → **acknowledged** ("define"). No image needed.
- **Executing cells** (a morphism whose chain reaches `|> visualise(mode)`, or a
  `dispatch`) → run the phases against the linked image and **return a chart**.
  An image is required only here — exactly as the sandbox needs an image to run.

Each cell is type-checked in the context of **all prior definitions** (the whole
merged program), so cell 3 can refer to what cell 1 defined. Redefining a name
replaces it.

---

## 4. Linking an image

An executing cell needs a linked image. Use the meta-command:

```
:scope load https://example.com/cells.png
```

- **PNG / JPEG only.** The loader decodes via the browser's `createImageBitmap`
  (fetch → blob → `createImageBitmap` → canvas → `getImageData` → RGBA →
  grayscale `Float32Array` normalised to `[0,1]`). TIFF is **not** supported —
  browsers can't decode it via `createImageBitmap`; you'll get a clear error.
- Link a PNG/JPEG URL (not a local path).

---

## 5. Session meta-commands

| Command          | Effect                                                        |
|------------------|---------------------------------------------------------------|
| `:scope`         | Show session state (coordinate_space / channels / morphisms / goal / dispatch / image linked?). |
| `:scope load <url>` | Decode a PNG/JPEG URL and link it as the session image.    |
| `:scope reset`   | Clear all definitions **and** the linked image.               |

---

## 6. A worked session

```text
# 1. define the world (a define-cell → ack, no image needed)
coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.10  lambda_t 0.05 }

# 2. link a PNG/JPEG image
:scope load https://example.com/hela_dapi.png

# 3. an executing cell: observe → visualise → CHART
seg = observe(load(db="cells", dataset="cells", image="hela_dapi.png"), n=4) |> visualise(scale_field)

# 4. inspect what the session knows
:scope

# 5. start over
:scope reset
```

Grammar notes (verified against the real parser):
- `coordinate_space` fields are `field W x H µm`, `depth N`, `lambda_s`,
  `lambda_t` — **not** `field_of_view`.
- A morphism is `name = observe(load(db=…, dataset=…, image=…), n=N) |> step |> …`.
- `visualise(mode)` is the chart-producing step. Modes include: `raw_image`,
  `scale_field`, `segmentation`, `distance_map`, `geodesic`, `spectral_power`,
  `entropy_trajectory`, `uncertainty_bar`, `scale_histogram`.

---

## 7. What you get back

An executing cell returns a `scope_run` output that the terminal renders with
the `ArtifactScope` component: the S-entropy sum + components (`Sk + St + Se = 1`),
distance ± uncertainty, goal pass/fail chips, field dims + SNR, and a
collapsible log.

The underlying `ScopeResult` carries:
- `chartData` — `spectralPower`, `entropyTrajectory`, `scaleHistogram`,
  `uncertaintyBar`.
- `visualData` — `rawImage`, `scaleField`, `segmentationMask`, `distanceMap`,
  `geodesicPath`, `pointCloud` (all `Float32Array`s).

**Known follow-up:** `ArtifactScope` currently renders the summary + goal chips
but does **not** yet draw the d3 field/chart from `visualData` / `chartData`.
That's the next visual iteration (d3 is already available in long-grass).

---

## 8. Quick checklist to get running

1. Repoint the `node_modules/scope-lang` junction from the stub to
   `helicopter/scope-lang` (§1).
2. `npx next build` (or `npm run dev`) in `long-grass` — it should compile the
   linked TS source cleanly.
3. Open the Buhera terminal.
4. Type a `coordinate_space { … }` cell → expect an ack.
5. `:scope load <png-or-jpeg-url>` → expect "image linked (W×H)".
6. Type a morphism ending in `|> visualise(scale_field)` → expect a `scope_run`
   chart card.
7. `:scope` to confirm state; `:scope reset` to clear.
</content>
</invoke>
