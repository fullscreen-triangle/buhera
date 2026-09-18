/* ============================================================================
 * Ladder Module Adapter
 *
 * Wraps @levinthal/ladder — the real catalytic-ladder contact-graph engine —
 * as a Buhera module. Vendored unmodified from levinthal/enzymes/web
 * (src/lib/engine.js, itself a direct port of shk_core.py): the same code
 * that computes the intensive derived power beta = 1 - (local floor / local
 * separation cost) over a weighted contact graph, and composes a sequence of
 * such powers multiplicatively into a ladder that can legitimately refuse
 * (subfloor) before any commitment when the declared rungs cannot reach a
 * declared target.
 *
 * This module's `derive` op computes a REAL rung power from a graph and a
 * vertex — the thing the HFQ engine's `ladder` step (see hfq-module.js /
 * vendor/hfq/src/execute.js's runLadder) takes as a bare declared number.
 * Combining them — derive real rung powers here, then hand them to an HFQ
 * plan's `ladder over ... power P, power P, ...` step — is what makes a
 * federated query's ladder step a computed quantity instead of an assertion.
 *
 * Instruction shapes:
 *   • { op: "chain", n, seed?, mediumWeight?, lo?, hi? }
 *       — build a random chain contact graph (chainGraph) and return its
 *         edges/floor/total, deterministic for a given seed.
 *   • { op: "derive", graph, vertex, radius? }
 *       — compute the intensive power at `vertex` (and the two controls,
 *         globalfloor/extensive, for comparison) over a supplied graph
 *         { vertices, weights, medium }.
 *   • { op: "compose", powers }
 *       — composeMultiplicative(powers), plus the additive/max/mean
 *         alternatives and the sensitivity of each rung.
 *   • { op: "climb", powers, target?, gap0? }
 *       — run the costed Machine: free derive/observe, one commitment per
 *         climbed rung, refusing (subfloor) before committing if the
 *         declared rungs cannot reach a declared target.
 *   • "demo" / "" — a small worked chain, matching the notebook's cell 1.
 * ========================================================================== */

import {
  ContactGraph,
  chainGraph,
  mulberry32,
  powerIntensive,
  powerGlobalFloor,
  powerExtensive,
  composeMultiplicative,
  composeAdditive,
  composeMax,
  composeMean,
  sensitivity,
  gapTrajectory,
  residualFraction,
  staticReachable,
  Machine,
} from "@levinthal/ladder";

/**
 * Build a ContactGraph from a plain JSON-shaped instruction: `vertices` an
 * array, `weights` an object keyed "a|b" (as sent over dispatch — a Map
 * cannot round-trip through JSON), `medium` the medium vertex name.
 */
function graphFromPlain(g) {
  const weights = g.weights instanceof Map ? g.weights : new Map(Object.entries(g.weights || {}));
  return new ContactGraph(g.vertices, weights, g.medium);
}

function describeGraph(g) {
  return {
    vertices: g.items(),
    medium: g.medium,
    floor: g.floor,
    total: g.total,
    edges: [...g.weights.entries()].map(([k, w]) => ({ edge: k, weight: w })),
  };
}

export const ladderModule = {
  id: "ladder",

  describe() {
    return {
      id: "ladder",
      description:
        "Ladder — the real catalytic-ladder contact-graph engine, vendored " +
        "unmodified from levinthal/enzymes/web. Derives an intensive power " +
        "beta over a weighted contact graph, composes rung powers " +
        "multiplicatively, and refuses (subfloor) before any commitment when " +
        "declared rungs cannot reach a declared target. Feed derived powers " +
        "into an hfq `ladder over ... power P` step to make a federated " +
        "query's ladder a computed quantity rather than an assertion.",
      instructions: [
        'dispatch("ladder", "demo")',
        'dispatch("ladder", { op: "chain", n: 6, seed: 42 })',
        'dispatch("ladder", { op: "derive", graph: <ContactGraph JSON>, vertex: "v2", radius: 1 })',
        'dispatch("ladder", { op: "compose", powers: [0.45, 0.30, 0.55] })',
        'dispatch("ladder", { op: "climb", powers: [0.45, 0.30, 0.55], target: 0.70 })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const instr = typeof instruction === "string" ? { op: instruction || "demo" } : instruction || {};
    const op = instr.op === "" ? "demo" : instr.op;

    try {
      if (op === "demo" || op === undefined) {
        const g = chainGraph(6, mulberry32(42));
        const items = g.items();
        const powers = items.map((v) => powerIntensive(g, v, 1));
        return ok({
          kind: "ladder_chain",
          graph: describeGraph(g),
          powers: items.map((v, i) => ({ vertex: v, power: powers[i] })),
          message: `chain graph of ${items.length} items, floor=${g.floor.toFixed(4)}`,
        });
      }

      if (op === "chain") {
        const n = instr.n ?? 6;
        const rng = mulberry32(instr.seed ?? 42);
        const g = chainGraph(n, rng, instr.mediumWeight, instr.lo, instr.hi);
        return ok({ kind: "ladder_chain", graph: describeGraph(g) });
      }

      if (op === "derive") {
        if (!instr.graph || !instr.vertex) return fail("derive: requires { graph, vertex }");
        const g = graphFromPlain(instr.graph);
        const radius = instr.radius ?? 1;
        const intensive = powerIntensive(g, instr.vertex, radius);
        const globalfloor = powerGlobalFloor(g, instr.vertex, radius);
        const extensive = powerExtensive(g, instr.vertex, radius);
        return ok({
          kind: "ladder_derived",
          vertex: instr.vertex,
          radius,
          intensive,
          globalfloor,
          extensive,
          note:
            "intensive is the only one invariant under graph extension far " +
            "from vertex; globalfloor and extensive are the near-miss and " +
            "control candidates the notebook rejects for that reason",
        });
      }

      if (op === "compose") {
        const powers = instr.powers;
        if (!Array.isArray(powers) || !powers.length) return fail("compose: requires non-empty powers[]");
        return ok({
          kind: "ladder_compose",
          multiplicative: composeMultiplicative(powers),
          additive: composeAdditive(powers),
          max: composeMax(powers),
          mean: composeMean(powers),
          sensitivity: sensitivity(powers),
          gapTrajectory: gapTrajectory(powers),
          residualFraction: residualFraction(powers),
        });
      }

      if (op === "climb") {
        const powers = instr.powers;
        if (!Array.isArray(powers) || !powers.length) return fail("climb: requires non-empty powers[]");
        const target = instr.target ?? null;
        const composite = composeMultiplicative(powers);
        if (target !== null && !staticReachable(powers, target)) {
          return ok({
            kind: "ladder_climb",
            verdict: "subfloor",
            composite,
            target,
            shortfall: target - composite,
            M: 0,
            message: "refused before any commitment — declared rungs cannot reach declared target",
          });
        }
        // A fresh graph/machine per climb call — the machine's M is a
        // per-run cost ledger, not shared state across dispatches.
        const g = chainGraph(Math.max(powers.length + 1, 2), mulberry32(1));
        const machine = new Machine(g);
        const verdict = machine.runVerdict(powers, target ?? 0, instr.gap0 ?? 1);
        return ok({
          kind: "ladder_climb",
          verdict: verdict.label,
          payload: verdict.payload,
          composite,
          target,
          M: machine.M,
          trace: machine.trace,
        });
      }

      return fail(`unknown op "${op}" — try chain | derive | compose | climb`);
    } catch (err) {
      return fail(err && err.message ? err.message : String(err));
    }
  },

  outputCell(_instruction) {
    return { kind: "ladder_cell" };
  },
};

function ok(output_delta) {
  return { ok: true, output_delta, residue: 0, completed: true };
}

function fail(message) {
  return {
    ok: false,
    output_delta: { kind: "text", lines: [`ladder: ${message}`] },
    residue: 0,
    completed: true,
    error: message,
  };
}
