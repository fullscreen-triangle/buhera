# smith — Agent Smith (split-attention synchronised agents)

| | |
|---|---|
| **Registry id** | `smith` |
| **Layer** | coordination |
| **Language** | `smith` (`.smith`) |
| **Upstream** | `fullscreen-triangle/musande` · `web/src/lib/agent-smith` @ `0fc4aae` |
| **Vendored at** | `long-grass/vendor/agent-smith/src` (byte-exact) |
| **TS binding** | native — `registry-ts/src/modules/smith.ts`, bound in `long-grass/src/lib/modules/smith-module.js` |
| **Rust binding** | none — see §7 |

## 1. Purpose

Agent Smith declares **agents**: an agent has a finite weighted **self-graph** of parts, a **purpose** (the attractor of a strongly convex drive), and a bounded **attention budget** divided over the **scenes** where it may act. The paper (*Split-Attention Synchronised Agents*, musande `epistemology/split-attention-synchronised-agents`) proves that every agent carries a conserved, non-local, strictly positive identity invariant — its **character χ**, the minimum inter-part cut, lower-bounded by the **realised floor** β_Self (T1) — and that the optimal split of attention over concurrent scenes under diminishing returns is **water-filling** at a single price p★ (T2).

`purpose minimise φ` makes a **character** (standing purpose, never halts); `purpose reach o` makes a **task-agent** (halts at quiescence). The runtime (a *town*) steps every agent through **observe → diagnose → commit** against a shared solution state Ω, under four invariants: identity χ conserved (I1), committed count monotone (I2), search-not-fetch — agents re-read Ω and store no answers (I3), phase exclusivity between construction and commitment (I4).

## 2. Which implementation, and why

musande holds three Smith front ends:

| Implementation | Status |
|---|---|
| `smith-ide/src/compiler` | self-declared **stub** ("Stub compiler … will be replaced"); regex-based, nondeterministic (`Math.random`), drops society members' `self`/budget |
| `web/src/lib/agent-smith` | **canonical** — recursive-descent parser, typechecker implementing the paper's typing rules, town runtime; newest (2026-08-12) |
| `crates/agent-smith` (Rust) | a port of the JS that **lags it**: realised floor is a singleton cut (on a path graph a–9–b–1–c–9–d it reports floor 9; the true floor is 1), and its potential list rejects 8 of the 10 tutorials |

The module wraps the canonical engine. long-grass previously ran `src/lib/smith`, a port of the stub (a permissive dialect with no typechecker, whose own test inputs the real parser rejects); it was removed in this revision (findings U-smi-5).

## 3. Language

Whitespace, commas and `//` comments are free; newlines are not significant. A source holds **exactly one** program — an agent or a society.

```
program    := agent | society
agent      := "agent" ID "{" agentItem* "}"
agentItem  := "purpose" ("minimise" | "minimize") ID      -- ID must be a registered strongly convex potential
            | "purpose" "reach" ID
            | "scenes" "{" ("scene" ID "serves" ID "with" ID)* "}"
            | "self" "{" ("parts" idList | "separations" sepList)* "}"
            | "budget" NUM | "floor" NUM | "coherence" "keeps"? idList
idList     := "{" (ID ("," ID)*)? "}"
sepList    := "{" (sep ("," sep)*)? "}"           sep := "(" ID "," ID ":" NUM ")"
society    := "society" ID "{" (agent | "tie" "(" ID "," ID ":" NUM ")" | "couple" NUM | ID)* "}"
```

Typing rules (the typechecker, not the parser):

- **Self** — parts non-empty and distinct; separations join known, distinct parts; every cost ≥ floor > 0; the graph is connected.
- **Purpose** — `reach` always types; `minimise φ` types only if φ is a registered strongly convex potential (`backlog`, `forge_residual`, `heat_residual`, `verdict_confirmed`, `residual`, `distance_to_goal`, …).
- **Scenes** — at least one; no duplicates; every scene `serves` the declared target; every scene has a hook.
- **Agent** — `budget > 0`; `coherence keeps` names known parts.
- **Society** — tie costs > 0.

Hooks (`with h`) are opaque by design: "the language can say a scene serves a purpose, never how it serves it."

A character with a triangle self-graph (χ = 4, floor 2):

```smith
agent clerk {
  purpose minimise backlog
  scenes {
    scene counter serves backlog with serve_hook
    scene filing  serves backlog with file_hook
  }
  self {
    parts { memory, manner, patience }
    separations {
      (memory, manner: 2), (manner, patience: 3), (patience, memory: 2)
    }
  }
  budget 1.0
  floor  2.0
}
```

A task-agent that halts at quiescence (the module's `"demo"`):

```smith
agent rerun_exp {
  purpose reach verdict_confirmed
  scenes {
    scene integrate serves verdict_confirmed with kuramoto_hook
    scene tabulate  serves verdict_confirmed with aggregate_hook
    scene report    serves verdict_confirmed with emit_hook
  }
  self { parts { data, method, result, verdict }
    separations { (data, method: 2), (method, result: 2), (result, verdict: 2), (verdict, data: 2) } }
  budget 1.0
  floor  2.0
  coherence keeps { method, result }
}
```

Common rejections: a separation cost below the floor; a disconnected self-graph; `minimise` of an unregistered potential (use `reach` for bespoke outcomes); a scene serving a target other than the purpose.

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` | check: parse, typecheck, characterise every agent |
| `"demo"` | run the task-agent example above |
| `{source, run?: bool, maxTicks?: int ≤ 500}` | check, then optionally run the town — **always with models off** |

## 5. Output delta

`agent_generated`, in the shape the `ArtifactSmith` renderer reads:

```jsonc
{ "kind": "agent_generated", "ok": true, "summary": "…",
  "agents": [{ "name", "regime": "character|task", "chi", "floor", "nonLocal", "chiPartition": [[…], […]], "count", "state" }],
  "society": { "chi", "side", "couple" } | null,
  "diagnostics": [{ "message", "line", "severity": "error" }],
  "steps"?: [{ "tick", "agent", "regime", "outcome", "limit", "tier", "price", "residual", "delta", "count", "scene", … }],
  "finalCounts"?: { "<agent>": n }, "residuals"?: { "<target>": gap }, "quiescent"?, "done"? }
```

## 6. Residue

- **Check**: the number of diagnostics (0 when typed).
- **Run**: Σ over purpose targets of max(0, residual − lowest normalised floor among agents pursuing it) — the engine's own remaining distance to purpose. (The former residue, a sum of realised floors, measured size, not remaining work.)

`completed` is true when the town is done or quiescent.

## 7. Why no Rust binding

The Rust crate gives different answers from the JS on the same programs (singleton-cut floor, **U-smi-1**; short potential list, **U-smi-2**; unordered `HashMap` residuals, **U-smi-3**). Binding it as `smith` would break contract equivalence. It becomes a candidate once it tracks the JS.

## 8. Side effects and hazards

- The adapter's runs make no network calls. The engine's own `defaultCtx()` enables models, and its transport POSTs the user's keys to an app route; the adapter always passes `useModel: false` (**U-smi-4** records the default).
- Module-level state (a `uid` counter; a PRNG used only for graphs of more than 20 parts) makes internal ids differ between builds. The adapter exposes no ids.
- `minimise φ` rejects any φ outside the registry; use `reach` for bespoke outcomes.

## 9. Conformance

- `registry-ts/test/modules.test.ts` — `smith: the canonical front end types the demo and runs it deterministically, models off`, `smith: typing rules come from the real checker`.
- `long-grass/test/smith.test.mjs` — six cases over the real engine (Scout χ = 5, regimes, refusals, society, deterministic flattened trace, a refused program does not run).
- `long-grass/test/knowledge-packs.test.mjs` — both examples above validate through the real parser + typechecker.
