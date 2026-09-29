/* ============================================================================
 * ladder — the catalytic-ladder contact-graph engine (specification
 * specs/ladder.md). Wraps levinthal's enzymes/web engine.js (a port of
 * shk_core.py), vendored.
 *
 * Every verdict comes from the engine's own Machine.runVerdict: `reached`,
 * `short`, `subfloor` (refused before any commitment) or `empty`, in one
 * shape. The costed machine commits once per climbed rung; derive, observe
 * and probe are free.
 *
 * `derive` enumerates every subset of a vertex's ball (2^|ball|), so graphs
 * are capped at MAX_DERIVE_ITEMS items — beyond that a request is refused,
 * never left to run for minutes.
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";

interface Graph {
  items(): string[];
  medium: string;
  floor: number;
  total: number;
  weights: Map<string, number>;
}

interface Verdict {
  label: "reached" | "short" | "subfloor" | "refused" | "empty";
  payload: Record<string, unknown>;
}

/** The subset of the vendored ladder engine this adapter uses. */
export interface LadderEngine {
  ContactGraph: new (vertices: string[], weights: Map<string, number>, medium: string) => Graph;
  chainGraph(n: number, rng: () => number, mediumWeight?: number, lo?: number, hi?: number): Graph;
  mulberry32(seed: number): () => number;
  powerIntensive(g: Graph, v: string, radius: number): number;
  powerGlobalFloor(g: Graph, v: string, radius: number): number;
  powerExtensive(g: Graph, v: string, radius: number): number;
  composeMultiplicative(ps: number[]): number;
  composeAdditive(ps: number[]): number;
  composeMax(ps: number[]): number;
  composeMean(ps: number[]): number;
  sensitivity(ps: number[]): unknown;
  gapTrajectory(ps: number[]): unknown;
  residualFraction(ps: number[]): unknown;
  Machine: new (graph: { floor: number | null }, epsilon?: number) => {
    M: number;
    trace: string[];
    residues: Array<number | null>;
    runVerdict(ps: number[], target: number, gap0?: number): Verdict;
  };
}

export const LADDER_ID = "ladder";
export const MAX_DERIVE_ITEMS = 20;
const MAX_CHAIN = 500;

function describeGraph(g: Graph) {
  return {
    vertices: g.items(),
    medium: g.medium,
    floor: g.floor,
    total: g.total,
    edges: [...g.weights.entries()].map(([edge, weight]) => ({ edge, weight })),
  };
}

function numbers(x: Json | undefined): number[] | null {
  return Array.isArray(x) && x.length > 0 && x.every((p) => typeof p === "number" && Number.isFinite(p)) ? (x as number[]) : null;
}

function num(x: Json | undefined, fallback: number): number | null {
  if (x === undefined || x === null) return fallback;
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

export function makeLadderModule(engine: LadderEngine): Module {
  const expected = '"demo" or { op: "chain" | "derive" | "compose" | "climb", … }';

  function graphFrom(g: Json | undefined): Graph | string {
    if (!g || typeof g !== "object" || Array.isArray(g)) return "graph must be { vertices, weights: { \"a|b\": w }, medium }";
    const vertices = g["vertices"];
    const medium = g["medium"];
    const weights = g["weights"];
    if (!Array.isArray(vertices) || typeof medium !== "string" || !weights || typeof weights !== "object" || Array.isArray(weights)) {
      return "graph must be { vertices, weights: { \"a|b\": w }, medium }";
    }
    return new engine.ContactGraph(vertices.map(String), new Map(Object.entries(weights).map(([k, v]) => [k, Number(v)])), medium);
  }

  function climb(powers: number[], target: number | null, gap0: number, graph: Graph | null): ActResult {
    // The machine reads its graph only for the per-commit residue (the graph's
    // floor). Without a real graph there is no floor to report: residues are null.
    const machine = new engine.Machine(graph ?? { floor: null });
    const v = machine.runVerdict(powers, target ?? 0, gap0);
    const composite = engine.composeMultiplicative(powers);
    // Residue: the distance still to the target — 0 when reached; the gap left
    // when short; the shortfall when the ladder cannot reach it at all.
    const p = v.payload as { achieved?: number; shortfall?: number };
    const residue =
      v.label === "reached" ? 0
      : v.label === "short" ? Math.max(0, (target ?? 0) - (p.achieved ?? 0))
      : v.label === "subfloor" ? Math.max(0, p.shortfall ?? 0)
      : 1;
    return done(
      {
        kind: "ladder_climb",
        summary: `ladder: ${v.label} — composite ${composite.toFixed(6)}${target != null ? ` vs target ${target}` : ""}, M = ${machine.M}`,
        verdict: v.label,
        payload: v.payload,
        composite,
        target,
        M: machine.M,
        trace: machine.trace,
        residues: machine.residues,
      },
      residue,
    );
  }

  return {
    id: LADDER_ID,
    describe: () => ({
      id: LADDER_ID,
      description:
        "Ladder — the catalytic-ladder contact-graph engine: derive the intensive power β = 1 − localFloor/σ at a " +
        "vertex of a weighted contact graph, compose rung powers multiplicatively (1 − ∏(1 − pᵢ)), and climb with a " +
        "costed machine that commits once per rung and refuses (subfloor) before any commitment when the declared " +
        "rungs cannot reach the declared target.",
      instructions: [
        'dispatch("ladder", "demo")',
        'dispatch("ladder", { op: "chain", n: 6, seed: 7 })',
        'dispatch("ladder", { op: "derive", graph: { vertices, weights: { "a|b": w }, medium }, vertex, radius: 1 })',
        'dispatch("ladder", { op: "compose", powers: [0.45, 0.30, 0.55] })',
        'dispatch("ladder", { op: "climb", powers: [0.45, 0.30, 0.55], target: 0.70 })',
      ],
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      const instr: Record<string, Json> =
        typeof instruction === "string" ? { op: instruction || "demo" } : instruction && typeof instruction === "object" && !Array.isArray(instruction) ? instruction : {};
      const op = instr["op"] ?? "demo";
      try {
        if (op === "demo") {
          // The notebook's chain (enzymes/web main.js: seed 7), with the intensive
          // power derived at every item — real rung powers from a real graph.
          const g = engine.chainGraph(6, engine.mulberry32(7));
          const items = g.items();
          return done(
            {
              kind: "ladder_chain",
              summary: `ladder: chain of ${items.length} items, floor ${g.floor.toFixed(4)}`,
              graph: describeGraph(g),
              powers: items.map((vertex) => ({ vertex, power: engine.powerIntensive(g, vertex, 1) })),
            },
            0,
          );
        }
        if (op === "chain") {
          const n = num(instr["n"], 6);
          const seed = num(instr["seed"], 7);
          if (n == null || seed == null || !Number.isInteger(n) || n < 2 || n > MAX_CHAIN) return invalid(LADDER_ID, `{ op: "chain", n: 2..${MAX_CHAIN}, seed? }`);
          const opt = (x: Json | undefined) => (typeof x === "number" && Number.isFinite(x) ? x : undefined);
          const g = engine.chainGraph(n, engine.mulberry32(seed), opt(instr["mediumWeight"]), opt(instr["lo"]), opt(instr["hi"]));
          return done({ kind: "ladder_chain", summary: `ladder: chain of ${n}, floor ${g.floor.toFixed(4)}`, graph: describeGraph(g) }, 0);
        }
        if (op === "derive") {
          const g = graphFrom(instr["graph"]);
          if (typeof g === "string") return invalid(LADDER_ID, g);
          const vertex = instr["vertex"];
          const radius = num(instr["radius"], 1);
          if (typeof vertex !== "string" || radius == null || !Number.isInteger(radius) || radius < 0) return invalid(LADDER_ID, "{ op: \"derive\", graph, vertex, radius?: integer ≥ 0 }");
          if (g.items().length > MAX_DERIVE_ITEMS) {
            return fail([`ladder: derive enumerates every subset of the vertex's ball; graphs are capped at ${MAX_DERIVE_ITEMS} items`], "graph too large");
          }
          return done(
            {
              kind: "ladder_derived",
              vertex,
              radius,
              intensive: engine.powerIntensive(g, vertex, radius),
              globalfloor: engine.powerGlobalFloor(g, vertex, radius),
              extensive: engine.powerExtensive(g, vertex, radius),
            },
            0,
          );
        }
        if (op === "compose") {
          const powers = numbers(instr["powers"]);
          if (!powers) return invalid(LADDER_ID, "{ op: \"compose\", powers: [non-empty numbers] }");
          return done(
            {
              kind: "ladder_compose",
              multiplicative: engine.composeMultiplicative(powers),
              additive: engine.composeAdditive(powers),
              max: engine.composeMax(powers),
              mean: engine.composeMean(powers),
              sensitivity: engine.sensitivity(powers),
              gapTrajectory: engine.gapTrajectory(powers),
              residualFraction: engine.residualFraction(powers),
            },
            0,
          );
        }
        if (op === "climb") {
          const powers = numbers(instr["powers"]);
          const target = instr["target"] == null ? null : num(instr["target"], 0);
          const gap0 = num(instr["gap0"], 1);
          if (!powers || gap0 == null || (instr["target"] != null && target == null)) return invalid(LADDER_ID, "{ op: \"climb\", powers, target?, gap0?, graph? }");
          let graph: Graph | null = null;
          if (instr["graph"] != null) {
            const g = graphFrom(instr["graph"]);
            if (typeof g === "string") return invalid(LADDER_ID, g);
            graph = g;
          }
          return climb(powers, target, gap0, graph);
        }
        return invalid(LADDER_ID, expected);
      } catch (err) {
        // The engine throws on non-positive weights and empty edge sets.
        return fail([`ladder: ${errorText(err)}`], errorText(err));
      }
    },

    outputCell: () => ({ kind: "ladder_cell" }),
  };
}
