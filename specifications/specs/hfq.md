# hfq — the hegel federated query interpreter

| | |
|---|---|
| **Registry id** | `hfq` |
| **Layer** | coordination |
| **Language** | `hfq` (`.hfq` plans) |
| **Upstream** | `fullscreen-triangle/hegel` · `consequences/src/lib/hfq` @ `63d09c4` |
| **Vendored at** | `long-grass/vendor/hfq/src` (byte-exact up to line endings) |
| **TS binding** | native: `registry-ts/src/modules/hfq.ts`, bound in `long-grass/src/lib/modules/hfq-module.js` |
| **Rust binding** | none. No Rust HFQ exists in hegel; a port would violate rule M1 |

## 1. Purpose

A **plan** names abstract sources and declares a request budget. The engine runs the paper's pipeline: parse → resolve → **check** → **allocate** → execute → emit.

- **check** decides capability containment before any contact (Req(ρ) ⊆ Capset(Src)). An ill-capable plan is refused having issued zero requests.
- **allocate** water-fills the budget on a shadow price.

Every step ends in one of six verdicts:

| verdict | blocker |
|---|---|
| `answer` | — |
| `empty` | — (an answer, just an empty one) |
| `surface` | model |
| `timeout` | engine |
| `refused` | budget |
| `starved` | corpus |

The central claim ("cor:onebit") is that a boolean success interface collapses five of these into one bit. This module never does that.

## 2. Language

The language is line-oriented. A clause starts on a line whose head is `plan`, `budget`, `let`, `emit`, `assert` or `}`, and continues onto the following lines. `#` starts a comment.

```
plan       := "plan" NAME "{" clause* "}"
budget     := "budget" NUM ("request" | "requests")?                         -- required
let-from   := "let" x "=" "from" SOURCE "ask" PRED "(" ARGS ")" ("with" ?v "in" y)* ("within" NUM)?
              ("else fail unresolved")? ("when starved emit partial")?        -- the last is parsed but inert
let-map    := "let" x "=" "map" y "via" MAP ("then via" MAP)* ("expect partial" NUM)?
let-ladder := "let" x "=" "ladder over" y ("power" P)+ ("expect power" E)?   -- P ∈ [0,1]
let-set    := "let" x "=" ("union" | "intersect") y z+ | "join" y z "on" ATTR | "filter" y "where" ATTR OP VALUE
emit       := "emit" x ("with provenance")? ("as extension of" "TEXT" ("because" REASON)?)?
            | "emit divergence(" y "," z ")" ("as" NAME)?
```

Sources resolve against one of four fixture worlds:

- `main`: chebi, rhea, enzdb
- `paper`: CHEBI, RHEA, KEGG
- `tiny`
- `biocat`: TAX, RXN, SEQ, PROV, INST

A plan that mixes worlds fails at `resolve`.

```hfq
plan healthy_chain {
  budget 200 requests
  let acids = from chebi ask descendants_of("CHEBI:1") within 10
  let kegg  = map acids via chebi2kegg expect partial 0.6
  let rxns  = from rhea ask reactions_consuming(?c) with ?c in kegg within 60
  emit rxns
}
```

```hfq
plan mark_q1 {
  budget 200 requests
  let enzymes = from RXN
      ask enzyme_of("RXN:TA-benzylethylamine")
      within 10
  let bacterial = from RXN
      ask typed_as("KIND:bacterial-enzyme", "KIND:eukaryotic-enzyme")
      with ?e in enzymes
      within 10
  let candidates = intersect enzymes bacterial
  let keyed = map candidates via enz_to_seq
      expect partial 0.75
  let no_cys = from SEQ
      ask excluding("C", ?k)
      with ?k in keyed
      within 20
  emit no_cys with provenance
}
```

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"`, `""` | `healthy_chain`, the paper's worked example |
| `string` / `{kind:"run", source}` | Run a plan |
| `{kind:"preset", id}` | One of the 24 preset plans, including Mark Doerr's `mark_q1`–`mark_q5` and `dcat_g1`–`dcat_g8` |
| `{kind:"check", source}` | Parse plus world resolution, issuing no requests |
| `{kind:"list_presets"}` | List preset ids, sections and blurbs |

## 4. Output delta

`hfq_result`: `{summary, result, verdicts:{answer:n,…}, blockers:{budget:n,…}, halted_early, refused_statically}`. `result` is the engine's six-verdict document, verbatim.

## 5. Residue and `ok`

**Residue** is the number of steps whose verdict is neither `answer` nor `empty`. `empty` is an answer, and by the library's own definition it has no blocker.

**`ok`**:
- A static capability refusal is a legitimate typed outcome (contract A1): `ok: true`, `refused_statically: true`, and the refused steps count as residue.
- `ok: false` only when the plan could not be run at all: stage `parse`, `resolve` or `execute`.

## 6. Determinism and side effects

Fully deterministic. `mark_q1` produced a byte-identical 6279-byte result on two runs. There is no I/O, by construction. Importing `biocat.js` mutates the shared `PREDICATE_FEATURES` map at load time, despite `sideEffects:false`.

## 7. Conformance

- `registry-ts/test/modules.test.ts`: three `hfq:` cases:
  - `mark_q1` runs in the biocat world;
  - `ill_capability` is refused statically with `ok:true`;
  - `empty_answer` / `budget_trap` verdict and blocker tallies.
- long-grass: `test/hfq-ladder-modules.test.mjs` (5 hfq cases, including all 24 presets and zero requests on refusal) and the validator accept and reject.

## 8. Upstream notes

- **U-hfq-1.** `runPlan` cannot take a budget override. `Executor.run(plan, budget)` can, but the world builders are private to `index.js`, so the act budget cannot be mapped to the plan budget yet.
- **U-hfq-2.** The static capability check needs the world registry, which `runPlan` does not expose. The validator therefore stops at world resolution.
- **U-hfq-3.** `when starved emit partial` is parsed but ignored by the executor.
