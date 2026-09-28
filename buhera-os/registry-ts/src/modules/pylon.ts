/* ============================================================================
 * pylon — SRN glyphs, the yield market and process agents (specification
 * specs/pylon.md). Wraps @buhera/pylon (pylon/ts), vendored unmodified.
 *
 * The module owns one Cluster (contract M2). `submit` parses an SRN glyph,
 * clears the market over the cluster's slots and instantiates a process
 * agent; `step` drives agents (one act-budget unit = one agent tick). The
 * residue it reports is the agent's goal distance — a geometric quantity in
 * the agent's goal space, NOT progress on the glyph's body (which pylon never
 * executes). Known engine hazards are avoided, not papered over: snapshot
 * restore is not exposed (it can loop forever), and an empty slot list is
 * refused before it reaches clearMarket (which throws).
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

/** Structural view of the pylon exports this adapter uses. */
export interface PylonEngine {
  parseSrn(source: string): unknown;
  isParseError(x: unknown): boolean;
  Cluster: new (cfg: { nodes: PylonNodeSpec[]; coupling?: number }) => PylonCluster;
  clearMarket(
    agents: string[],
    slots: Array<{ id: string; capacity: number; maxRate: number }>,
    payoff: (agent: string, slot: string) => number,
    tick: number,
  ): { assignment: Map<string, string>; prices: Map<string, number> };
  TICK: number;
}

export interface PylonNodeSpec {
  id: string;
  frame: readonly [number, number, number, number];
  capacity: number;
  taskDuration: number;
  maxRate: number;
}

export interface PylonCluster {
  submit(expr: string, opts?: { goal?: number[] }): { ok: boolean; agent?: string; error?: { kind: string }; [k: string]: unknown };
  agent(id: string): { residual(): number; committedStep(): number; currentState(): string; step?(): string } | null;
  liveAgents(): Array<{ id: string; residual(): number; committedStep(): number; currentState(): string }>;
  snapshot(): unknown;
  nodes(): unknown[];
  price(nodeId: string): number;
}

export const PYLON_ID = "pylon";

/** Default cluster: the three reference nodes of the SRN test suite's first shells. */
const DEFAULT_NODES: PylonNodeSpec[] = [
  { id: "n1", frame: [2, 1, 0, 1], capacity: 2, taskDuration: 1, maxRate: 1 },
  { id: "n2", frame: [2, 1, 1, 1], capacity: 2, taskDuration: 1, maxRate: 1 },
  { id: "n3", frame: [1, 0, 0, 1], capacity: 1, taskDuration: 1, maxRate: 1 },
];

const DEMO_GLYPH = "|task : (2,1,0,+)| not { unreachable-key } do { emit self.n } to { * }";

export function pylonValidate(engine: PylonEngine) {
  return (source: string): Validation => {
    const r = engine.parseSrn(source);
    if (!engine.isParseError(r)) return { ok: true, errors: [] };
    const e = r as { kind: string; message?: string };
    return {
      ok: false,
      errors: [{ message: e.kind === "no-negation-boundary" ? "no negation boundary: a glyph needs a not{…} clause" : (e.message ?? e.kind) }],
    };
  };
}

export function pylonDsl(engine: PylonEngine): DslEntry {
  return { id: "srn", label: "SRN glyph", extension: ".srn", moduleId: PYLON_ID, packId: "srn", validate: pylonValidate(engine) };
}

const agentView = (a: { id?: string; residual(): number; committedStep(): number; currentState(): string }, id?: string) => ({
  id: a.id ?? id,
  residual: a.residual(),
  committed: a.committedStep(),
  state: a.currentState(),
});

export function makePylonModule(engine: PylonEngine, nodes: PylonNodeSpec[] = DEFAULT_NODES): Module {
  let cluster = new engine.Cluster({ nodes });
  const expected = 'SRN glyph source, "demo", or { kind: submit|step|agents|snapshot|clear|reset, … }';
  return {
    id: PYLON_ID,
    describe: () => ({
      id: PYLON_ID,
      description:
        "pylon — Sango Rine Shumba glyphs, a separation-cost yield market and persistent process agents. Submit a " +
        "glyph to allocate it to a node and start an agent; step agents toward their goals; clear a market.",
      instructions: [
        'dispatch("pylon", "demo")',
        'dispatch("pylon", { kind: "submit", source: "|t : (2,1,0,+)| not { k } do { emit self.n } to { * }", goal: [0.02, 0.02] })',
        'dispatch("pylon", { kind: "step", agent: "a0" })',
        'dispatch("pylon", { kind: "clear", agents: ["x", "y"], slots: [{ id: "e0", capacity: 1, maxRate: 1 }] })',
      ],
      dsl: "srn",
      binding: "native",
    }),

    execute(instruction: Instruction, actBudget = 1): ActResult {
      const obj = typeof instruction === "object" && instruction && !Array.isArray(instruction) ? instruction : null;
      const kind =
        instruction == null || instruction === "" || instruction === "demo" || typeof instruction === "string"
          ? "submit"
          : obj?.["kind"];
      try {
        if (kind === "submit") {
          const source =
            typeof instruction === "string" && instruction !== "demo" && instruction !== ""
              ? instruction
              : typeof obj?.["source"] === "string"
                ? (obj["source"] as string)
                : DEMO_GLYPH;
          const goal = Array.isArray(obj?.["goal"]) ? (obj!["goal"] as Json[]).filter((x): x is number => typeof x === "number") : undefined;
          const y = cluster.submit(source, goal ? { goal } : undefined);
          if (!y.ok) {
            return fail([`pylon: ${y.error?.kind ?? "rejected"}`], y.error?.kind ?? "rejected");
          }
          const a = y.agent ? cluster.agent(y.agent) : null;
          return done(
            { kind: "pylon_yield", summary: `pylon: agent ${y.agent} allocated`, yield: y as unknown as Json, agent: a ? agentView(a, y.agent) : null },
            a ? a.residual() : 0,
          );
        }
        if (kind === "step") {
          const id = obj?.["agent"];
          if (typeof id !== "string") return invalid(PYLON_ID, '{ kind: "step", agent: string }');
          const a = cluster.agent(id);
          if (!a || typeof a.step !== "function") return fail([`pylon: no live agent "${id}"`], "unknown agent");
          for (let i = 0; i < Math.max(1, actBudget); i++) {
            if (a.currentState() === "retired") break;
            a.step();
          }
          const retired = a.currentState() === "retired";
          return {
            ok: true,
            output_delta: { kind: "pylon_agent", summary: `pylon: ${id} ${a.currentState()} (residual ${a.residual().toFixed(4)})`, agent: agentView(a, id) },
            // Residue = the agent's goal distance (geometric; not glyph progress).
            residue: retired ? 0 : a.residual(),
            completed: retired,
          };
        }
        if (kind === "agents") {
          const live = cluster.liveAgents().map((a) => agentView(a));
          return done({ kind: "pylon_agents", summary: `pylon: ${live.length} agent(s)`, agents: live }, 0);
        }
        if (kind === "snapshot") {
          return done({ kind: "pylon_snapshot", summary: "pylon: cluster snapshot", snapshot: cluster.snapshot() as Json }, 0);
        }
        if (kind === "reset") {
          cluster = new engine.Cluster({ nodes });
          return done({ kind: "text", lines: ["pylon: cluster reset"] }, 0);
        }
        if (kind === "clear") {
          const agents = Array.isArray(obj?.["agents"]) ? (obj!["agents"] as Json[]).filter((x): x is string => typeof x === "string") : [];
          const slots = Array.isArray(obj?.["slots"]) ? (obj!["slots"] as Array<{ id: string; capacity: number; maxRate: number }>) : [];
          if (slots.length === 0) return fail(["pylon clear: at least one slot is required"], "no slots");
          const payoffs = (obj?.["payoffs"] ?? {}) as Record<string, Record<string, number>>;
          const r = engine.clearMarket(agents, slots, (a, s) => payoffs[a]?.[s] ?? 1, engine.TICK);
          return done(
            {
              kind: "pylon_market",
              summary: `pylon: cleared ${agents.length} agent(s) over ${slots.length} slot(s)`,
              assignment: Object.fromEntries(r.assignment),
              prices: Object.fromEntries(r.prices),
            },
            0,
          );
        }
        return invalid(PYLON_ID, expected);
      } catch (err) {
        return fail([`pylon: unexpected error — ${errorText(err)}`], errorText(err));
      }
    },

    outputCell: () => ({ kind: "pylon_cell" }),
  };
}
