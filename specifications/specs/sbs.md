# sbs — Systems Biology Shaders

| | |
|---|---|
| **Registry id** | `sbs` |
| **Layer** | science |
| **Language** | `sbs` (`.sbs`) |
| **Upstream** | `fullscreen-triangle/hegel` · `consequences/src/lib/sbs` @ `63d09c4` |
| **Vendored at** | `long-grass/vendor/sbs` (subset; `index.js` and `package.json` carry recorded local changes) |
| **TS binding** | native: `registry-ts/src/modules/sbs.ts`, bound in `long-grass/src/lib/modules/sbs-module.js` |
| **Rust binding** | none. The Rust crate has no `.sbs` language; its calculus is the separate module [`sbs-core`](sbs-core.md) |

## 1. Purpose

SBS models a metabolic network as a circuit:

- **Nodes** are metabolites with a chemical potential μ.
- **Edges** are reactions with a conductance; the flux on an edge is `G·|Δμ|`.

Each node is observed as an S-entropy triple (S_e: potential, S_k: flux, S_t: degree). The run extracts two scalars:

- **coherence R**: the mean pairwise Spearman ρ of the three coordinates, mapped to [0, 1];
- **flux visibility V**: a flux-weighted geometric mean of perturbed/healthy flux ratios; V = 1 means healthy.

The observation is a fragment shader on WebGL2, with a CPU mirror. The `.sbs` language declares the circuit and the acts on it.

## 2. Language

```
program   := stmt*
stmt      := "circuit" IDENT "{" stmt* "}" | "node" IDENT props? | "edge" IDENT "->" IDENT props?
           | "observe" expr | "perturb" expr props? | "restore" expr | "navigate" "from"? expr
           | "catalyst" IDENT props? | "cascade" "(" expr, … ")" | "convert" expr "from" KW "to" KW
           | "let" IDENT "=" expr | "fn" IDENT "(" params ")" "{" stmt* "}" | "for" IDENT "in" expr "{" … "}"
           | "if" expr "{" … "}" ("else" …)? | "import" IDENT ("as" IDENT)? "from" STRING | "export" stmt | expr
props     := "{" (key ":" expr ","?)* "}"
node props: mu, mu0, potential, concentration (default 1), compartment (default "cytoplasm"), boundary
edge props: rate (default 1), conductance (default rate)
perturb   : { factor } (default 0.1)
```

Comments are `//` and `/* */`. Semicolons are optional. Operators are `|> || && == != < > <= >= + - * / % **`, unary `- !`, and `#(a,b,c)` / `triple(a,b,c)` literals.

Engine behaviour a script author must know. These are upstream facts; the adapter warns about the silent ones:

- A node's non-zero `mu` is used verbatim. When `mu` is absent or 0, μ = μ0 + RT·ln c.
- `perturb` ignores every prop except `factor`, **including `edge:`**. The product of all factors is applied to the single highest-healthy-flux edge.
- `restore` deletes perturbations. It does not treat the circuit. For treatment, use the `therapy` instruction.
- `navigate` always starts from the highest-μ node, and it is triggered by the substring `navigate` anywhere in the source.
- A second `circuit` block's edges are not re-indexed.
- `import` data is inert, and zero-valued props cannot be expressed.

```sbs
circuit glycolysis {
  node Glucose  { mu: -917.0, concentration: 5.0, compartment: "cytoplasm" }
  node G6P      { mu: -1760.0, concentration: 0.083 }
  node F6P      { mu: -1755.0, concentration: 0.014 }
  node Pyruvate { mu: -472.0, concentration: 0.051 }
  edge Glucose -> G6P      { rate: 230.0, conductance: 464.1 }
  edge G6P     -> F6P      { rate: 100.0, conductance: 3.35 }
  edge F6P     -> Pyruvate { rate: 150.0, conductance: 0.85 }
}
observe glycolysis
perturb glycolysis { factor: 0.1 }
```

```sbs
circuit pair {
  node A { mu: -900.0, concentration: 5.0 }
  node B { mu: -1200.0, concentration: 1.0 }
  edge A -> B { rate: 10.0 }
}
observe pair
```

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"`, `""`, `null` | The 10-node glycolysis demo |
| `string` / `{kind:"run", source, preferCPU?}` | Compile and observe; `preferCPU` forces the deterministic CPU path |
| `{kind:"check", source}` | The engine's own checker, `checkSBS` |
| `{kind:"compile", source}` | Raw compiler output (AST, emitted JS and GLSL, circuit) |
| `{kind:"therapy", source, maxEdges?}` | Run, then `suggestTherapy`: an l1-sparse restorative perturbation |
| `{kind:"script", id}` / `{kind:"list_scripts"}` | The upstream example scripts |

## 4. Output deltas

`sbs_result` keeps the historical shape that the `MetricsDashboard` renderer reads:

```jsonc
{ "kind": "sbs_result", "summary": "SBS: 10 nodes, 9 edges  R=0.592  V=0.118  [cpu]",
  "circuit": { "numNodes", "numEdges", "nodes", "edges", "compartments" }, "metrics": { "R", "V", "Se", "Sk", "St", "fluxHealthy", "fluxCurrent", "backend", "renderTimeMs" },
  "navigation": [{ "nodeId", "name", "mu" }] | null, "observations": [], "perturbations": [{ "idx", "factor" }],
  "warnings": [/* engine */], "adapter_warnings": [/* see §2 */], "backend": "webgl2|cpu", "therapy"?: [{ "idx", "factor" }] }
```

`backend` and `renderTimeMs` are host-local (spec 02 §4).

## 5. Residue

`1 − V`: how far the observed flux pattern is from the healthy baseline. It is 0 when unperturbed or when nothing was observed. R is not a distance and is not used.

This replaces the previous `nodes + edges`, which was a size, not a residue.

## 6. Side effects and hazards

- A fresh WebGL2 canvas is created on every observation, and contexts are never explicitly released. Many dispatches in one tab can exhaust the browser's context limit, so use `preferCPU` for batch work.
- Shader loops are bounded at 256 nodes and 256 edges. Beyond that, GPU metrics are wrong and the adapter warns.
- GPU arithmetic is float32 and CPU arithmetic is float64.
- A 1-node circuit gives R = NaN, and the adapter warns.

## 7. Conformance

- `registry-ts/test/modules.test.ts`: three `sbs:` cases (CPU demo with residue = 1 − V; the edge-prop warning; the checker).
- long-grass: SBS validator accept and reject.
- The upstream-drift check verifies the vendored subset byte-for-byte (`scripts/vendor-sync.mjs`).

## 8. Upstream notes

- **U-sbs-1.** `perturb { edge: … }` is ignored. The TypeScript compiler tree (`dsl/compiler/*.ts`) keeps the edge target, but it also rejects `edge` as a prop key, so 4 of the 7 shipped scripts fail there.
- **U-sbs-2.** Parser errors carry `line: 0`, with the real line only inside the message text. The adapter's validator recovers it from the message.
