# @buhera/spraypaint

TypeScript bindings for the **spraypaint** split-attention search binary, plus the
**crossfilter → query propagation** loop that makes its charts editable.

This module is deliberately *binary-backed*: it never simulates results. Every
number a UI draws — water-filling clearing price `p*`, per-scene allocation, the
χ identity invariant, the never-resetting committed count — comes from real
`spraypaint … --json` output. It is the antithesis of a mock engine.

## The canonical artifact: `AskQuery`

Everything runnable is one object:

```ts
interface AskQuery { query: string; budget: number; scenes: string[] | null; flat: boolean }
```

It maps 1:1 to `spraypaint ask <query> -k <budget> [--scenes a,b] [--flat]`.
Three input surfaces — a typed prompt, a hand-edited field, and a **chart
gesture** — are peers that all write to this one object.

## The loop (why charts are bidirectional)

```
   AskQuery ──ask──▶ AskResult ──draw──▶ charts
      ▲                                     │
      └────── applyDiff ◀── invert*() ◀─────┘  (a gesture)
```

A chart gesture is *inverted* into a `QueryDiff` — a typed, structured change to
the `AskQuery` — then applied. Re-running redraws the charts. Two reasons this
must be bidirectional:

1. **The query is the only runnable artifact.** If charts were read-only, only
   users fluent enough to hand-edit `-k`/`--scenes` could steer a live search.
   Gestures-as-query-edits keep the chart surface a first-class input, equal to
   the prompt.
2. **The abstractions are new.** `p*`, χ, committed count have no pre-existing
   intuition. Watching a drag rewrite `-k 12 → -k 20` *teaches, by operation*,
   that "the price line I'm dragging IS the budget." The diff is the Rosetta
   Stone between an unfamiliar chart and the query grammar.

Unlike the prototype web app (which string-replaced a `.grf` DSL and could emit
malformed source), these inversions edit a typed object — an invalid query is
unrepresentable.

## Gestures → real flags

| Gesture (chart) | Inversion | Real binary effect |
|---|---|---|
| Click a scene bar | `invertSceneToggle` | add/remove `--scenes a,b` |
| Drag the p* / clearing-price band | `invertPriceDrag` | move `-k` to admit more/fewer passages |
| Budget +/- stepper | `invertBudgetStep` | change `-k` |
| Flat / grouped toggle | `invertFlatToggle` | `--flat` |

## Usage

```ts
import { SpraypaintClient, SpraypaintSession, invertSceneToggle } from "@buhera/spraypaint";
import { NodeRunner } from "@buhera/spraypaint/runner-node"; // server-side only

const client = new SpraypaintClient(new NodeRunner(), { root: "/path/to/repo" });
const session = new SpraypaintSession(client, {
  query: "water-filling attention", budget: 12, scenes: null, flat: false,
});

const result = await session.run();               // real ask, increments count
const diff = invertSceneToggle(session.state().query, result.allocation, "docs");
if (diff) session.applyGesture(diff, "crossfilter"); // query changes, marked stale
const next = await session.run();                 // re-run → charts reattach

session.undo(); // unified history: prompt/editor/crossfilter edits are peers
```

## React + D3 components (`@buhera/spraypaint/react`)

The charts are an integral part of the implementation — they *are* the
crossfilter surface. Import from the `/react` subpath (peer deps: react ≥18,
d3 ≥7):

```tsx
import { SpraypaintPanel } from "@buhera/spraypaint/react";

<SpraypaintPanel client={client} initialQuery={{ query: "water-filling", budget: 12, scenes: null, flat: false }} />
```

Components:

- **`SpraypaintPanel`** — the whole loop in one component: query bar + allocation
  chart + results + identity badge, wired to a session. The reference layout.
- **`AllocationChart`** — the primary crossfilter surface. Water-filling
  allocation bars (blue above p*, grey below) + BM25 box plots + the p* line.
  **Click a scene bar → toggle `--scenes`. Drag the p* line → change `-k`.**
- **`ResultsList`** — ranked passages grouped by scene; scene headers also toggle.
- **`IdentityBadge`** — χ ≥ floor, committed count, p*, fingerprint (the invariants
  made visible, since they have no pre-existing intuition).
- **`useSpraypaintSession`** — the hook that binds a `SpraypaintSession` to React
  (state + `run`/`applyGesture`/`applyAndRun`/`undo`/`redo`). Build a bespoke
  layout on this if `SpraypaintPanel` isn't the shape you want.

Every gesture returns a `QueryDiff`; the panel calls `applyAndRun`, so a click or
drag advances the `AskQuery`, re-invokes the real binary, and redraws. That is the
chart→query propagation, live.

## Architecture

- **`types.ts`** — interfaces typed against real `--json` output; `AskQuery` +
  serialisation. No runtime deps.
- **`client.ts`** — `SpraypaintClient` over a `SpraypaintRunner` abstraction
  (local binary or HTTP service). Browser-safe.
- **`runner-node.ts`** — `NodeRunner` via `child_process`. The *only*
  Node-coupled module; imported from the `/runner-node` subpath.
- **`crossfilter.ts`** — the `invert*` functions + `applyDiff`. The mechanism.
- **`undo.ts`** — `QueryHistory`: full-snapshot stack, three sources as peers.
- **`session.ts`** — `SpraypaintSession`: the closed loop as one headless object
  (query, result, stale flag, history). A UI wires charts to this.
- **`react/`** — the components above; the only part that pulls react + d3.

## Invariant fidelity

The session honours the binary's four invariants: `run()` uses `ask` (increments
the committed count, Inv 2); `preview()` uses `--dry-run` (a zero-act read-out,
no increment, Inv 3); `index` vs `ask` are the exclusive construction/commitment
phases (Inv 4); `client.identity()` / `verify()` expose the χ conservation
(Inv 1) and the full certificate.

## Build

```
npm install          # @types/node + typescript (dev only)
npm run typecheck
npm run build        # → dist/
```
