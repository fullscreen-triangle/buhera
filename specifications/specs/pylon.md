# pylon — SRN glyphs, the yield market, process agents

| | |
|---|---|
| **Registry id** | `pylon` |
| **Layer** | coordination |
| **Language** | `srn` (SRN glyph source, `.srn`) |
| **Upstream** | `fullscreen-triangle/pylon` · `ts/src` @ `5fdb8cd` |
| **Vendored at** | `long-grass/vendor/pylon/dist` (build product; verified byte-identical to a fresh build at this commit) |
| **TS binding** | native: `registry-ts/src/modules/pylon.ts`, bound in `long-grass/src/lib/modules/pylon-module.js` |
| **Rust binding** | none; see §8 |

## 1. Purpose

pylon is the distributed resource-allocation runtime:

- **Unit of work.** The Sango Rine Shumba (SRN) *glyph*, addressed by a partition coordinate (n, ℓ, m, s) with shell capacity 2n².
- **Individuation.** A glyph is individuated by a mandatory negation clause `not{…}`.
- **Placement.** A **yield market** prices each slot at its separation cost, and clearing yields an assignment where no single reassignment improves yield by more than τ₀.
- **Execution.** Every allocated task becomes a persistent, goal-directed **process agent**. It moves toward its goal by exactly TICK = 10⁻³ per step, holds a monotone committed counter M, and on attainment asks an occupation callback for its next goal.

Before this revision, long-grass vendored pylon but called none of it. This module is its first integration.

## 2. Language

```
glyph  := "|" IDENT ":" "(" INT "," INT "," INT "," SPIN ")" "|" clause+
clause := "not" "{" … "}"      -- required (no-negation-boundary otherwise)
        | "do"  "{" … "}"      -- required
        | "to"  "{" … "}"      -- required
        | "as"  "{" … "}"      -- optional
SPIN   := "+" | "-" | "+1" | "-1" | "1" | ""     (any order of clauses; braces balanced)
```

Constraints: n ≥ 1, 0 ≤ ℓ < n, |m| ≤ ℓ.

Clause text is **opaque**. The not-guard is an environment-key presence test, and in a `Cluster` the environment is empty, so it never fires. `do`, `to` and `as` are carried but never interpreted. Routing uses exact coordinate equality, falling back to all nodes.

```srn
|task : (2,1,0,+)| not { unreachable-key } do { emit self.n } to { * }
```

```srn
|identity : (2,1,0,+)|
      not { n != 2 }
      do  { emit self.n }
      to  { n = 2, * }
      as  { id }
```

## 3. Instructions

The module owns one `Cluster`: nodes n1 (2,1,0,+), n2 (2,1,1,+) and n3 (1,0,0,+).

| Instruction | Effect |
|---|---|
| `"demo"`, `string` / `{kind:"submit", source, goal?}` | Parse, clear the market, start an agent. Returns the yield |
| `{kind:"step", agent}` | Drive the agent: **one act-budget unit = one tick** (contract M6); `completed` when retired |
| `{kind:"agents"}` / `{kind:"snapshot"}` | Inspect |
| `{kind:"clear", agents, slots, payoffs?}` | The pure market (`clearMarket`); `payoffs[agent][slot]` defaults to 1 |
| `{kind:"reset"}` | Fresh cluster |

## 4. Output deltas

- `pylon_yield`: `{yield:{ok, agent, committedStep, residual, allocation:{node, slot, price}, value}, agent:{id, residual, committed, state}}`
- `pylon_agent`
- `pylon_agents`
- `pylon_snapshot`
- `pylon_market`: `{assignment, prices}`

## 5. Residue

The agent's **goal distance**: Euclidean distance in its goal space, decreasing by exactly 10⁻³ per tick. It is geometric, and it is not progress on the glyph body, which pylon never executes. The delta says which agent it belongs to.

## 6. Hazards

The adapter avoids these rather than papering over them:

- `Cluster.fromSnapshot` can loop forever (a restored agent retires and stops advancing M). Restore is **not exposed**.
- `clearMarket(agents, [])` throws. An empty slot list is refused first.
- Capacity is not enforced. Agents stack when they outnumber slots, and retired agents are never evicted, so a node's price climbs with every exact-target submit.
- `orderParameter()` / `isPhaseLocked()` mutate the Kuramoto bank. They are never polled.
- `digestOf` truncates UTF-16 code units to 8 bits.

## 7. Conformance

`registry-ts/test/modules.test.ts`:
- `pylon: submit allocates an agent; stepping drives residual down to retirement`
- `…glyph without a negation boundary is rejected by the real parser`
- `…clear refuses an empty slot list`

long-grass: `library-federation.test.mjs` (dispatch and audit through the facade; SRN validator).

## 8. Why no Rust binding

pylon's Rust crates implement **different models**:
- `srn-node` uses a structured JSON `/eval`, SHA-1 labels and the key `trajectory_count`.
- `srn-fleet` has a cost-based greedy clearing with `unplaced` acts.

Neither is wire- or semantics-compatible with the TypeScript runtime. Binding one as `pylon` would give the two hosts different answers under one id. Per rule M1, the Rust binding stays `none` until upstream chooses one source of truth (**U-pyl-1**).

**Related finding:** long-grass's existing `/api/srn` proxy sends `{expression}` to srn-node's `/eval` and calls `/probe` and `/gossip`. srn-node expects the structured `EvalRequest` and serves `/network/probe` and `/network/gossip` (**U-pyl-2**, a long-grass defect, recorded here and not fixed in this revision).
