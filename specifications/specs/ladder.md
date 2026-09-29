# ladder — the catalytic-ladder contact-graph engine

| | |
|---|---|
| **Registry id** | `ladder` |
| **Layer** | science |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/levinthal` · `enzymes/web/src/lib/engine.js` @ `eca6596` (origin/main) |
| **Vendored at** | `long-grass/vendor/ladder/src/engine.js` (byte-exact up to line endings) |
| **TS binding** | native — `registry-ts/src/modules/ladder.ts`, bound in `long-grass/src/lib/modules/ladder-module.js` |
| **Rust binding** | none — the engine is JS (a port of `shk_core.py`) |

## 1. Purpose

A **contact graph** has items plus a medium vertex adjacent to every item, with strictly positive weights; its floor is the smallest weight. At a vertex, the **intensive power** is β = clamp(1 − localFloor/σ, 0, 1), where σ is the local separation cost — the minimum cut over the vertex's radius-r ball. β is invariant under extending the graph far from the vertex; the global-floor and extensive variants are the near-miss and control. Rung powers compose multiplicatively: 1 − ∏(1 − pᵢ). A costed **machine** climbs a ladder: probing, deriving and observing are free; each climbed rung costs one commitment; and a ladder whose declared rungs cannot reach a declared target is **refused before any commitment** (`subfloor`).

This is what makes an HFQ plan's `ladder over … power P` step a computed quantity rather than an assertion: derive the powers here, hand them to the plan.

## 2. What is bound

`ContactGraph`, `chainGraph` (seeded `mulberry32`), `powerIntensive` / `powerGlobalFloor` / `powerExtensive`, the four compositions and their diagnostics, and `Machine.runVerdict`. Every verdict — `reached`, `short`, `subfloor`, `empty` — is the engine's, in one shape. (The former long-grass adapter re-implemented the subfloor check, returned a different shape for it, and built an unrelated chain graph to satisfy the `Machine` constructor.)

The machine reads its graph only for the per-commit residue (the graph's floor). `climb` accepts an optional real `graph`; without one, the residues are `null` rather than the floor of an invented graph.

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"` | the notebook's chain (`n = 6`, seed 7) with the intensive power at every item |
| `{op: "chain", n: 2..500, seed?, mediumWeight?, lo?, hi?}` | a seeded chain contact graph |
| `{op: "derive", graph: {vertices, weights: {"a\|b": w}, medium}, vertex, radius?}` | β at a vertex, with the two controls |
| `{op: "compose", powers}` | multiplicative, additive, max, mean; sensitivity; gap trajectory |
| `{op: "climb", powers, target?, gap0?, graph?}` | the machine's verdict, commitments M, trace |

`derive` enumerates every subset of the ball (2^|ball|; radius 8 on a 40-item chain took 4.5 s), so graphs are capped at **20 items**; larger requests are refused rather than left to run.

## 4. Output delta

`ladder_chain`, `ladder_derived`, `ladder_compose`, `ladder_climb` (`{verdict, payload, composite, target, M, trace, residues}`). Invalid weights (≤ 0) and empty edge sets are refused with the engine's message.

## 5. Residue

`climb`: the distance still to the target — 0 when `reached`; target − achieved when `short`; the shortfall when `subfloor`; 1 for `empty`. Commitments M are reported in the delta. All other operations: 0.

## 6. Side effects and hazards

None: pure and deterministic (its only randomness is the seeded PRNG).

## 7. Conformance

- `registry-ts/test/modules.test.ts` — the demo's six powers; compose [0.45, 0.30, 0.55] = 0.82675; climb to 0.70 is `reached` with M = 3, residue 0 and null residues; to 0.95 is `subfloor` with M = 0 and residue = shortfall 0.12325; a 26-item derive is refused.
- `long-grass/test/hfq-ladder-modules.test.mjs` — the host's behaviour tests.
