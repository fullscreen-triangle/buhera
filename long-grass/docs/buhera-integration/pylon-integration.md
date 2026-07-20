# Pylon Integration

**Status:** long-grass-side contract for the `pylon` distributed
resource-allocation runtime.
**Depends on:** `docs/sources/sango-rine-shumba-expression-protocol.tex`
(the SRN expression language pylon speaks), and
`docs/sources/network-yield.tex` (the theory of allocation-as-agents).
**Paired with:** `musande-integration.md`. Musande and pylon share
the same runtime notion of "agent"; they differ only in what an agent
is *for*. Read that document first.

This document is the specification of what long-grass expects `pylon`
to be, how long-grass will consume it, and — crucially — the division
of responsibility between long-grass (the OS surface), musande
(vocational agents), and pylon (compute-resource agents).

Every design choice cites the theorem in the two source papers that
motivates it.

---

## 0. What pylon is, in one line

Pylon is the distributed resource-allocation layer of Buhera OS. It is
the runtime that (a) speaks Sango Rine Shumba (SRN) as its
transmission unit, (b) implements the compute-tick–bounded scheduler
and yield market of the network-yield paper, and (c) instantiates each
allocated compute packet as a **process agent** — a standing,
goal-directed entity with the same runtime shape musande gives
vocational NPCs.

Buhera OS does not allocate compute. Long-grass does not allocate
compute. Pylon does. Long-grass surfaces the results.

---

## 1. Why this exists — the missing layer

Before pylon, the picture had a hole. Musande gave us townspeople
(smiths, scribes, market-runners) using modules (shapeshifter, vahera,
scope). But when a smith actually starts working — when the shapeshifter
DSL script the smith authored has to run — *something has to allocate a
CPU, a GPU shard, a memory budget, a wall-clock slot* on some physical
machine in the cluster. That "something" is not Buhera OS's job. Buhera
OS is a categorical OS — it names, dispatches, and audits. It does not
own the fleet.

Pylon owns the fleet. It:

- Receives SRN expressions describing what needs computing, where, with
  what target
  (`sango-rine-shumba-expression-protocol.tex`, §5)
- Runs the cell-partition scheduler with compute-tick `τ₀` as the
  physical/algorithmic/market quantum
  (`network-yield.tex`, §4, §7)
- Clears the yield market — matches tasks to execution slots at prices
  equal to separation costs
  (`network-yield.tex`, Thm 7.1, "Three-way Equivalence")
- Instantiates each allocated task as a **process agent** with a
  standing goal, monotone committed-step counter, and the four musande
  invariants
  (`network-yield.tex`, §10)
- Provides liveness: no queue starves, because stalled queues raise
  their own price and summon capacity
  (`network-yield.tex`, Thm 8.1)

In short: pylon is the plumbing that makes musande's townspeople able
to actually do things across a cluster.

---

## 2. The critical unification: compute agents ARE musande agents

This is the load-bearing insight. Section 10 of `network-yield.tex`
proves that every allocated compute packet is already, in full, a
process agent in the sense the paper defines — with a standing goal
(`τ(x)`), a felt drive (`r(x, ·)`), an ever-advancing internal state
(`loc(x)`), an ability to initiate (raise `sep(e, A)` when stalled),
and, once you add the one goal-succession primitive (Def. 10.2 of the
paper), *persistence past completion*.

The four musande invariants (identity `χ`, monotone count `m`,
search-not-fetch, phase exclusion) map onto pylon's agents cleanly:

| Musande invariant | Pylon realisation |
|---|---|
| I1 conserved identity `χ` | The agent's `AgentId` and its bounded self-graph (the SRN partition coordinates `(n, ℓ, m, s)` it occupies) |
| I2 monotone committed count `m` | The committed-step record `M(t)` from Thm 9.1 of network-yield |
| I3 search-not-fetch | Every re-invocation walks the current self-graph; the SRN expression is evaluated *at the current receiver frame*, per Thm 4.2 (Receiver-Relativity) of the SRN paper |
| I4 exclusive construction/commitment phases | Cell-partition scheduler alternates observation phase (construction, filling the self-graph) and action phase (commitment, executing `A(cell(h))`) |

Meaning: **pylon does not need a separate agent runtime**. It uses
musande's `Agent` / `Town` classes, feeding them SRN-derived scenes,
SRN-derived drives, and the compute-tick as the water-fill quantum.

This is what makes the OS coherent: the smith-in-the-workshop and the
GPU-slot-executing-that-workshop-DSL are the same kind of entity. The
long-grass terminal shows both in `/town`. When you address a smith,
you're addressing a musande agent whose scenes are pylon-allocated
compute slots. When a compute slot stalls, it raises `sep`, and pylon's
yield market re-routes work — which the smith experiences as its own
felt drive rising and being met.

---

## 3. Package identity

- **Name:** `@buhera/pylon`
- **Language:** TypeScript for the caller-facing API and browser client;
  the actual scheduler + SRN evaluator + trajectory encoder are likely
  Rust (for cluster deployment) with a WASM build for the browser
  demo. Long-grass consumes the WASM/TS surface only; the Rust core is
  a deployment concern outside this contract.
- **Distribution:** vendored under `long-grass/vendor/pylon/` (same
  pattern as `@buhera/purpose`, `@buhera/musande`,
  `@lavoisier/shapeshifter`, `scope-lang`).
- **Runtime targets:** browser (Next.js client bundle) for the
  single-machine demo; Node (for the `/api/pylon/*` routes) for cluster
  proxy.
- **Dependencies on other Buhera packages:** may import `@buhera/musande`
  for the `Agent` class (since pylon's allocated tasks *are* agents).
  Must not import anything from long-grass.
- **License:** matches the rest of the stack.

---

## 4. Versioning contract

Same as purpose and musande: semver, pre-1.0 minor bumps may break
shapes, at 1.0.0 the API surface in §5 is locked.

---

## 5. Non-negotiable design principles

These follow from the two source papers. Violating any of them breaks
the theorems long-grass depends on.

1. **SRN is the transmission unit.** Every request into pylon, every
   response out, every inter-node message inside the cluster: an SRN
   expression. Not JSON, not gRPC. The SRN paper's Structural
   Incorruptibility theorem (Thm 6.1) depends on there being no parser
   — timing trajectories decode to a fixed grammar. Pylon must preserve
   this.

2. **Every SRN expression carries a mandatory `not` boundary.** The
   SRN paper's Corollary 2.1 (Mandatory Negation Boundary) is
   *runtime-enforced*: expressions arriving without a `not` clause are
   rejected. Pylon rejects them; long-grass never fabricates
   expressions without one.

3. **Receiver-relativity is honoured.** The same SRN expression
   evaluated on two nodes legitimately produces two different results
   (SRN Thm 4.2). Pylon does not "reconcile" them. Long-grass does not
   expect a canonical answer.

4. **The compute-tick τ₀ is the single quantum.** Physical resolution
   floor, algorithmic closure threshold, and market lot size are all
   τ₀ (`network-yield.tex`, Thm 7.1). Pylon exposes τ₀ as a read-only
   constant; long-grass reads it but does not choose it.

5. **Monitor and control are architecturally separate.**
   `network-yield.tex` Thm 4.4 (Monitor–Control Separation) requires
   monitoring subsystems to have no direct path to the action stream
   except through the cell-partition map. Pylon enforces this;
   long-grass MUST NOT ask pylon to bypass the cell map (e.g., "just
   run this expression right now, don't wait for the scheduler").

6. **Structural incorruptibility.** No monitoring payload can influence
   the scheduling action stream (`network-yield.tex`, Thm 9.1). The
   action channel carries only cell indices. Long-grass respects this:
   when it observes pylon state, it reads cell indices, not raw
   payloads.

7. **Allocated compute IS an agent.** No separate "task" and "agent"
   abstractions. Pylon exposes each allocated slot as a musande
   `Agent` instance (or a lightweight compatible interface). The
   goal-succession primitive from `network-yield.tex` Def. 10.2 is
   pylon's contribution — musande gets it for free by delegating.

8. **Errors are typed, never thrown from the core.** Same discipline
   as purpose and musande.

9. **Forest openness (SRN Thm 8.1).** Any machine that can evaluate
   SRN expressions is a peer. Pylon has no enrolment,
   no certificate authority, no membership list. Long-grass respects
   this: it does not maintain a whitelist of pylon nodes.

---

## 6. The public API (locked at 1.0)

### 6.1 Types

```typescript
/** The compute-tick τ₀, the single quantum. Read-only. */
export const TICK: number; // seconds; typically 1e-3 to 1e-6

/** SRN partition coordinates (n, ℓ, m, s). */
export type Coord = readonly [number, number, number, number];

/** An SRN expression, as text (source form) or as a parsed AST. */
export type SrnExpr = string | SrnAst;
export interface SrnAst { /* opaque; produced by parseSrn() */ }

/** A cluster node identified by its receiver frame. */
export interface Node {
  id: string;               // opaque, unique per node
  frame: Coord;             // the node's partition coordinates
  capacity: number;         // c(e), thread capacity
  taskDuration: number;     // ℓ(e), expected task duration in ticks
  maxRate: number;          // v̄(e), maximum throughput rate
}

/** A completion target τ(x) in the receiver's disposition space. */
export type Target = ReadonlyArray<number>;

/** A yield-market price p(e) = sep(e, A). */
export type Price = number;

/** The result of a submitted expression. */
export type Yield =
  | {
      ok: true;
      agent: AgentId;               // the process agent instantiated
      committedStep: number;        // M(t), monotone
      residual: number;             // r(x, e), current
      allocation: {
        node: string;
        slot: number;
        price: Price;
      };
      /** Receiver-relative result from the current node's evaluation. */
      value: unknown;
    }
  | {
      ok: false;
      error:
        | { kind: 'no-negation-boundary' }
        | { kind: 'malformed-srn'; at: number }
        | { kind: 'no-capacity'; price: Price }
        | { kind: 'boundary-rejects'; frame: Coord }
        | { kind: 'stalled'; ticks: number; priceRising: boolean };
    };
```

### 6.2 Parsing and encoding

```typescript
/** Parse SRN source. Fails on missing `not` clause. */
export function parseSrn(source: string): SrnAst;

/** Encode an SRN expression as a timing trajectory
 *  (SRN paper §7). For transport across nodes. */
export function encodeTrajectory(expr: SrnExpr):
  { deltas: number[]; channels: number };

/** Decode a timing trajectory back to an SRN expression. */
export function decodeTrajectory(deltas: number[], channels: number):
  SrnAst;
```

### 6.3 The Cluster class

```typescript
export interface ClusterConfig {
  /** Nodes participating in this cluster. Membership is open per SRN
   *  Thm 8.1 (Forest Theorem): this is a hint, not an authoritative
   *  list. Additional nodes that arrive with valid SRN capability
   *  join automatically. */
  nodes: ReadonlyArray<Node>;

  /** Utilisation cost g_u; must satisfy g_u(0)=0, g_u'>0, g_u''>0
   *  (network-yield §7, Ax. Cost). */
  utilisationCost?: (v: number) => number;

  /** Kuramoto coupling K for scheduler phase-lock (network-yield §9).
   *  Should exceed K_c* = 2σ_ω/π to achieve R≥0.95 lock. */
  coupling?: number;
}

export class Cluster {
  constructor(config: ClusterConfig);

  /** Read-only inspectors. */
  readonly tick: number;
  nodes(): ReadonlyArray<Node>;
  price(nodeId: string): Price;                 // p(e) = sep(e, A)
  yield(): number;                              // network yield
  orderParameter(): { R: number; psi: number }; // Kuramoto R, ψ
  isPhaseLocked(): boolean;                     // R ≥ 0.95

  /** Submit an SRN expression for evaluation somewhere in the cluster.
   *  Pylon selects the receiver based on the expression's `to { region }`
   *  clause and the current market clearing. Returns a Yield with a
   *  running Agent handle; the agent persists past initial completion
   *  via goal succession if a succession map was supplied. */
  submit(
    expr: SrnExpr,
    options?: {
      /** Occupation map γ for goal succession (network-yield §10.2).
       *  If absent, the agent completes and retires. */
      succession?: (attained: Target, hist: unknown[]) => Target | null;
      /** Timeout in ticks; if the agent stalls beyond this without
       *  price attracting capacity, returns { ok: false, stalled }. */
      timeoutTicks?: number;
    },
  ): Promise<Yield>;

  /** Broadcast an SRN meta-expression (installs into every reachable
   *  node's live registry per SRN §6). Used for protocol updates. */
  broadcastMeta(expr: SrnExpr): Promise<{ reached: number }>;

  /** Return the process agent (musande Agent) currently running for
   *  the given AgentId. */
  agent(id: AgentId): Agent | null;

  /** List all live process agents in the cluster. */
  liveAgents(): ReadonlyArray<Agent>;

  /** Persistence. Snapshots the cluster's assignment state; agent
   *  state is snapshotted via musande. */
  snapshot(): ClusterSnapshot;
  static fromSnapshot(snap: ClusterSnapshot): Cluster;
}
```

### 6.4 Pure operators (for tests, for lightweight callers)

```typescript
/** Compute yield (network-yield Def. 3.4). */
export function yieldOf(
  assignment: Map<AgentId, string>,
  agents: ReadonlyArray<Agent>,
  nodes: ReadonlyArray<Node>,
  utilisationCost: (v: number) => number,
): number;

/** Compute separation cost sep(e, A) for one node. */
export function separationCost(
  node: string,
  assignment: Map<AgentId, string>,
  agents: ReadonlyArray<Agent>,
  nodes: ReadonlyArray<Node>,
): Price;

/** Clear the yield market. Returns assignment + prices at
 *  deterministic closure (network-yield Thm 7.1). */
export function clearMarket(
  agents: ReadonlyArray<Agent>,
  nodes: ReadonlyArray<Node>,
  tick: number,
): { assignment: Map<AgentId, string>; prices: Map<string, Price> };
```

### 6.5 Exports summary

- **Types:** `Coord`, `SrnExpr`, `SrnAst`, `Node`, `Target`, `Price`,
  `Yield`, `ClusterConfig`, `ClusterSnapshot`.
- **Constants:** `TICK`.
- **Functions:** `parseSrn`, `encodeTrajectory`, `decodeTrajectory`,
  `yieldOf`, `separationCost`, `clearMarket`.
- **Classes:** `Cluster`.
- **Re-exports from `@buhera/musande`:** `Agent`, `AgentId`. (Pylon
  agents ARE musande agents.)

---

## 7. Behavioural guarantees (what long-grass tests against)

Each cites the source theorem. Long-grass will hold integration tests
that assert each of these.

1. **Boundary rejection (SRN Cor. 2.1).** Any SRN expression submitted
   without a `not` clause returns `{ ok: false, error: 'no-negation-boundary' }`.
2. **Receiver-relativity (SRN Thm 4.2).** The same expression submitted
   to two clusters with different frames returns different `value`
   fields, and both are ok:true.
3. **Determinism per node (SRN Thm 4.1).** The same expression
   submitted twice to the same node in the same partition state
   returns the same `value`.
4. **Replay immunity (SRN Thm 6.2).** Replaying a past submission
   never returns the identical result — because `self.M` (the node's
   partition count) has advanced.
5. **Three-way Equivalence (network-yield Thm 7.1).** After
   `clearMarket()`, no single-agent reassignment improves yield by
   more than τ₀; the returned prices are exactly the separation costs.
6. **Forced optimal utilisation (network-yield Thm 7.3).** At closure,
   every node runs at v* satisfying `g_u(v*) - v* g_u'(v*) = 0`.
7. **Liveness (network-yield Thm 8.1).** Every live agent's residual
   reaches zero in finite time; stalled agents' prices rise and attract
   capacity.
8. **Monitor–control separation (network-yield Thm 4.4).** No path
   exists in the API for monitoring output to bypass the cell-partition
   map and directly influence assignment.
9. **Structural incorruptibility (network-yield Thm 9.1).** The
   committed-step counter M is monotone across submits, snapshots, and
   restarts; no API path decrements it.
10. **Forest openness (SRN Thm 8.1).** A new node arriving with SRN
    capability joins `cluster.nodes()` without enrolment.
11. **Phase-lock (network-yield Thm 9.2).** With coupling ≥ K_c*, the
    scheduler order parameter R settles ≥ 0.95; drops below 0.95
    surface as a first-class fault signal.
12. **Agent persistence (network-yield Thm 10.4).** An agent with a
    succession map runs indefinitely, one finite-time goal at a time;
    without one, it retires at first completion.

---

## 8. What's deferred (deliberately)

- **Stochastic arrivals.** All results are stated for deterministic
  arrivals. Poisson / heavy-tailed extensions are future work.
- **Incentive-compatibility of the yield market.** We prove clearing,
  not strategy-proofness.
- **Continuous external arrivals.** Liveness is proved over finite
  task sets (extended to indefinite by succession).
- **Fault-domain isolation.** Node failure detection is the phase-lock
  drop; recovery policy is caller-supplied.
- **A user-facing SRN editor.** Long-grass may add one later; pylon
  ships only `parseSrn` / `encodeTrajectory` / `decodeTrajectory`.

---

## 9. Non-goals

- Any protocol other than SRN. No HTTP task submission; the HTTP proxy
  in long-grass wraps `submit()` around a JSON-→-SRN adapter, but the
  wire format between pylon nodes is SRN trajectories.
- Cryptographic membership. Forest openness (SRN Thm 8.1) is
  incompatible with any credential system.
- A separate agent framework. Pylon delegates to `@buhera/musande`.
- Any centralised state (registries, catalogs, coordinators). Every
  registry is local per SRN §6.

---

## 10. Integration surface in long-grass

The files long-grass will add when pylon ships. All are long-grass's
own; pylon's public surface stays untouched.

### 10.1 `src/lib/cluster/pylon.js`

The single `Cluster` singleton per page:

```js
import { Cluster, TICK } from '@buhera/pylon';

export const cluster = new Cluster({
  nodes: /* seeded from env or discovered */,
  utilisationCost: v => v * v,
  coupling: 2.0,
});

export const tick = TICK;
```

### 10.2 `src/lib/cluster/pylon-module.js`

A module (in the Buhera federation sense) that lets other modules
dispatch compute to the cluster. Registered like `purpose-carry-module`:

```js
export const pylonModule = {
  id: 'pylon',
  describe: () => 'distributed resource allocation via SRN',
  execute: async (instruction, actBudget) => {
    const expr = instruction.srn ?? synthesizeSrn(instruction);
    const result = await cluster.submit(expr, { timeoutTicks: actBudget });
    return { ok: result.ok, cell: /* renderer */, ... };
  },
  outputCell: (result) => /* React component */,
};
```

Note: `synthesizeSrn` is long-grass's SRN codegen for callers that
don't hand-write SRN — it wraps a task in a minimal `|task:coord| not
{...} do {...} to {...}` skeleton.

### 10.3 `src/pages/api/pylon/submit.js`

A server route wrapping `cluster.submit()`. Long-grass validates the
expression parses (`parseSrn` throws on missing `not`), then forwards.
Keeps any cluster-side secrets server-side; browser code hits this
proxy.

### 10.4 `src/lib/cluster/audit-feeder.js`

Every pylon submit → committed step becomes an entry in the long-grass
audit log. Since pylon agents ARE musande agents, this feeds the same
audit stream that musande already feeds; the source field distinguishes
`{ source: 'pylon-alloc' }` from `{ source: 'musande-npc' }`.

### 10.5 `src/pages/cluster/index.js`

A `/cluster` page showing:
- Every node in `cluster.nodes()` with its frame `(n, ℓ, m, s)`,
  capacity, current price, and current live agents.
- The yield `cluster.yield()` and phase-lock indicator
  `cluster.isPhaseLocked()` in the header.
- The list of currently in-flight SRN expressions (compute-agents),
  each linking to its musande `/town/[agent]` page — because they're
  the same agent.

### 10.6 Terminal integration

- `:cluster` opens `/cluster`.
- `:tick` prints the current `TICK` and `cluster.orderParameter().R`.
- A new meta-command `:srn <expr>` submits an SRN expression via
  `cluster.submit()` and streams the yield result into the terminal
  as a normal act (with a new committed-step id).

### 10.7 No changes to the module registry

Pylon does not replace the module registry. It slots in as *one more
module* whose particular job is compute allocation. The federation
principle stays intact.

---

## 11. The three-way architectural picture

With pylon in place, Buhera OS has three co-equal runtime layers, each
governed by its own theory paper and each speaking its own DSL:

```
  Users / Agents (townspeople)                 Layer: OS surface
      ↓ dispatch via module DSLs               Owner: long-grass
      ↓ carry via @buhera/purpose
      ↓
  Vocational agents (musande)                  Layer: agency
      ↓ present() over scenes                  Owner: @buhera/musande
      ↓ water-fill attention, Kuramoto sync    Paper: split-attention
      ↓                                                synchronised-agents
      ↓
  Compute-allocation agents (pylon)            Layer: physical execution
      ↓ SRN expressions across cluster         Owner: @buhera/pylon
      ↓ yield-market clearing at τ₀            Papers: sango-rine-shumba
      ↓                                                network-yield

  Modules (echo, vahera, lavoisier, purpose,   Layer: infrastructure
    zangalewa, graffiti, shapeshifter, sbs,    Owner: their own repos
    scope, catalysts, compute, ...)            Contract: Module trait
```

Buhera OS = long-grass = the top layer, plus the federation registry
and the terminal that presents everything to the user. It does not do
agency (musande does). It does not do compute allocation (pylon does).
It composes them.

---

## 12. Long-grass smoke test — the "sufficient test"

Passing criterion for the pylon integration is a demo that exercises
all three layers together.

1. `:town` — three musande agents visible: smith, scribe, crier.
2. `:cluster` — three pylon nodes visible with frames, prices, phase-lock
   indicator green.
3. In the terminal:

   ```
   memory store "iron-order" = "three horseshoes by Friday"
   ```

   Vahera dispatches; audit-feeder converts to musande interaction;
   smith commits (present() returns ok:true); smith's response is an
   SRN expression describing the shapeshifter DSL script it wants to
   run.

4. Long-grass hands that SRN expression to pylon via
   `pylonModule.execute()`. Pylon's `submit()` parses the expression
   (has a `not` clause — passes), water-fills attention across nodes,
   the yield market clears at price `p(node2) = 0.4`, node 2 gets the
   allocation, a compute-agent is instantiated with the shapeshifter
   task as its `τ(x)`, and starts executing.

5. In `/cluster`, the new compute-agent appears at node 2. In `/town`,
   the smith is shown as "at the forge" (occupied — Thm 10.1 of
   network-yield: never idle while goal unmet).

6. Compute-agent completes; its goal-succession map is
   `attained => next_horseshoe(attained)`; it starts on the next
   horseshoe automatically. The smith's `present()` fires again on the
   completion notification; a new act is committed; `M(smith)` and
   `M(compute-agent)` both advance.

7. `:rerun <act-id-of-first-horseshoe>` — the smith re-presents against
   the same interaction; its self-graph has grown; the SRN expression
   it produces differs (perhaps a bulk-run script instead of a single);
   pylon allocates differently because the request is different.
   Different act, different `M`, same self-consistency.

If that end-to-end story works, the integration is done.

---

## 13. Open questions pylon owns

Long-grass hands each of these to pylon. Any reasonable answer is
fine; long-grass consumes what pylon picks.

- Whether the cluster core is Rust, Go, or TypeScript.
- The specific timing-trajectory encoding (SRN §7 fixes the shape but
  leaves the numerical scheme open).
- The specific Kuramoto integrator (as with musande).
- The KKT solver for market clearing.
- Whether snapshots are JSON or a custom binary format.
- The wire protocol between pylon nodes below SRN — QUIC, raw UDP,
  TCP-with-timing-extraction, anything.
- The physical resolution to which τ₀ is set at deployment time
  (nanoseconds on commodity hardware; microseconds on browser-only
  demos).
- The discovery mechanism by which new nodes join the forest
  (mDNS, gossip, WebRTC, manual URL — all conform to Forest openness).

---

## 14. Change management

Same rules as purpose and musande.
- Changes to §6 or §7: coordination point (proposal → review → version
  bump).
- Changes to §8 or §13: pylon's call alone.
- New long-grass-side capabilities pylon needs: issue in the long-grass
  repo; added under `src/lib/cluster/` so pylon stays clean.

---

## 15. Sequencing with musande

- **If musande ships first:** long-grass integrates it standalone; the
  town runs in a single browser process; every "compute" is local
  synchronous JS. This is a legitimate demo.
- **If pylon ships first:** long-grass integrates it standalone; SRN
  expressions run across a cluster, but the OS still speaks in modules
  and dispatches, not townspeople. Also a legitimate demo (arguably
  the "distributed compute" demo).
- **When both ship:** the sufficient test in §12 becomes possible, and
  the OS finally reads as an inhabited, computed-across-a-cluster town.
  This is the target.

Long-grass does not require them to arrive in any particular order.
The current OS state is coherent as a stopping point until either
lands.

---

## 16. What this integration UNBLOCKS

Once pylon lands:

- **The OS runs across a cluster** without any changes to the module
  federation, the terminal, or the audit log. All the existing
  machinery works because pylon presents as one more module.
- **SRN becomes visible as the OS's inter-node substrate.** Users can
  hand-write SRN expressions in the terminal (`:srn ...`) and see
  where they land, at what price, with what receiver-relative result.
- **The forest metaphor becomes real.** New nodes joining the cluster
  are visible in `/cluster` seconds after they come up, with no
  configuration.
- **Compute-agents and vocational-agents become interchangeable in
  the town view.** A user browsing `/town` sees smiths (musande) and
  forges (pylon compute-agents) as the same kind of entity — because
  by Section 10 of `network-yield.tex`, they are.
- **The Uni Greifswald pitch gets its scale story.** "This is my town
  on one machine. Now watch what happens when I add three more nodes"
  — Forest Theorem fires; the townspeople keep working; the yield
  market absorbs the new capacity; nothing else changes.
- **The Buhera OS claim "categorical operating system" earns itself
  at cluster scale.** Categories name what work is; agents do it;
  pylon puts it on hardware. Three layers, one town.

---

## 17. Timeline expectations (informational, not contractual)

- Long-grass does zero prep work until pylon v0.1 ships. Everything
  in §10 will happen in one focused session when the package lands,
  same pattern as purpose and musande.
- Long-grass will exercise §5 principles 1, 2, 5, 6 (SRN structural
  requirements) as soon as pylon can parse+evaluate a single
  expression in a browser, before cluster capability lands. That's a
  useful early milestone.

The current OS state remains a coherent stopping point until pylon
arrives.
