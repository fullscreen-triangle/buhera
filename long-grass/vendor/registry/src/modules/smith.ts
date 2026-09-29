/* ============================================================================
 * smith — Agent Smith: split-attention synchronised agents (specification
 * specs/smith.md). Wraps musande's canonical engine (web/src/lib/agent-smith:
 * recursive-descent parser + typechecker + town runtime), vendored.
 *
 * Every run uses useModel:false. The engine's default context turns models on
 * and POSTs the user's provider keys to an app route; a Buhera act must be
 * deterministic and must not move keys, so the adapter never enables it.
 *
 * The delta keeps the historical long-grass shape (kind "agent_generated",
 * agents / diagnostics / steps / finalCounts) that the ArtifactSmith renderer
 * reads; the engine's per-tick records are flattened into `steps`.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

interface SmithAgent {
  name: string;
  regime: string;
  chi: number;
  chiSide?: Set<string> | string[];
  chiNonLocal?: boolean;
  floor: number;
  floorNorm?: number;
  self: { parts: string[] };
  purpose: { mode: string; target: string };
  count: number;
  state: string;
}

interface SmithBuild {
  ok: boolean;
  errors: Array<{ message: string; line?: number | null }>;
  program: { kind: "agent" | "society"; name?: string; agents: SmithAgent[] } | null;
}

interface SmithStep {
  tick: number;
  records: Array<Record<string, unknown> & { agent: string }>;
  residuals: Record<string, number>;
  quiescent: boolean;
  done: boolean;
}

/** The subset of the vendored agent-smith engine this adapter uses. */
export interface SmithEngine {
  build(source: string): SmithBuild;
  makeTown(program: NonNullable<SmithBuild["program"]>): {
    agents: SmithAgent[];
    society: { chi: number; side: string[]; couple: number | null } | null;
  };
  defaultCtx(opts: { useModel: boolean }): unknown;
  runTown(town: unknown, ctx: unknown, maxTicks: number): Promise<SmithStep[]>;
}

export const SMITH_ID = "smith";

/** musande's EXAMPLE_TASK — a task-agent that halts at quiescence. */
const DEMO_SOURCE = `agent rerun_exp {
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
}`;

const finiteOrNull = (x: number) => (Number.isFinite(x) ? x : null);

function agentView(a: SmithAgent) {
  const side = a.chiSide ? [...a.chiSide] : [];
  const rest = a.self.parts.filter((p) => !side.includes(p));
  return {
    name: a.name,
    regime: a.regime,
    chi: finiteOrNull(a.chi),
    floor: finiteOrNull(a.floor),
    nonLocal: !!a.chiNonLocal,
    chiPartition: side.length ? [side, rest] : [a.self.parts],
    count: a.count,
    state: a.state,
  };
}

export function smithValidate(engine: SmithEngine) {
  return (source: string): Validation => {
    const b = engine.build(source);
    if (b.ok) return { ok: true, errors: [] };
    const errors = b.errors.map((e) => (typeof e.line === "number" ? { message: e.message, line: e.line } : { message: e.message }));
    return { ok: false, errors: errors.length ? errors : [{ message: "rejected" }] };
  };
}

export function smithDsl(engine: SmithEngine): DslEntry {
  return { id: "smith", label: "Agent Smith", extension: ".smith", moduleId: SMITH_ID, packId: "smith", validate: smithValidate(engine) };
}

export function makeSmithModule(engine: SmithEngine): Module {
  const expected = 'Agent Smith source, "demo", or { source, run?, maxTicks? }';
  return {
    id: SMITH_ID,
    describe: () => ({
      id: SMITH_ID,
      description:
        "Agent Smith — compile split-attention synchronised agents: parse and typecheck (connected self-graph, " +
        "costs ≥ floor, strongly convex potentials), compute each agent's character χ and realised floor, and " +
        "optionally run the town (observe → diagnose → commit, water-filled attention) deterministically.",
      instructions: [
        'dispatch("smith", "demo")',
        'dispatch("smith", "agent a { purpose reach done scenes { scene s serves done with h } self { parts { p, q } separations { (p, q: 3) } } budget 1 floor 2 }")',
        'dispatch("smith", { source, run: true, maxTicks: 12 })',
      ],
      dsl: "smith",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      let source: string | null = null;
      let run = false;
      let maxTicks = 30;
      if (instruction == null || instruction === "" || instruction === "demo") {
        source = DEMO_SOURCE;
        run = instruction === "demo";
      } else if (typeof instruction === "string") source = instruction;
      else if (typeof instruction === "object" && !Array.isArray(instruction) && typeof instruction["source"] === "string") {
        source = instruction["source"];
        run = instruction["run"] === true;
        const mt = instruction["maxTicks"];
        if (typeof mt === "number" && Number.isInteger(mt) && mt > 0) maxTicks = Math.min(mt, 500);
      }
      if (source == null) return invalid(SMITH_ID, expected);

      let b: SmithBuild;
      try {
        b = engine.build(source);
      } catch (err) {
        return fail([`smith: unexpected error — ${errorText(err)}`], errorText(err));
      }
      const diagnostics = b.errors.map((e) => ({ message: e.message, line: e.line ?? null, severity: "error" }));
      if (!b.ok || !b.program) {
        return {
          ok: false,
          output_delta: { kind: "agent_generated", ok: false, agents: [], diagnostics, summary: `smith: ${diagnostics.length} error(s)` },
          // Check-only residue: the diagnostics still to fix.
          residue: diagnostics.length,
          completed: true,
          error: diagnostics[0]?.message ?? "check failed",
        };
      }

      const town = engine.makeTown(b.program);
      if (!run) {
        return done(
          {
            kind: "agent_generated",
            ok: true,
            summary: `smith: ${town.agents.length} agent(s) typed`,
            agents: town.agents.map(agentView),
            society: town.society,
            diagnostics,
          },
          0,
        );
      }

      let history: SmithStep[];
      try {
        history = await engine.runTown(town, engine.defaultCtx({ useModel: false }), maxTicks);
      } catch (err) {
        return fail([`smith: run error — ${errorText(err)}`], errorText(err));
      }
      const last = history.at(-1);
      const steps: Array<Record<string, unknown>> = history.flatMap((h) => h.records.map((r) => ({ tick: h.tick, ...r })));
      const finalCounts = Object.fromEntries(town.agents.map((a) => [a.name, a.count]));
      // Residue: per purpose target, how far the shared residual still is above
      // the lowest floor any agent pursuing it can reach — the engine's own
      // remaining distance to purpose.
      const floors = new Map<string, number>();
      for (const a of town.agents) {
        const f = a.floorNorm ?? 0;
        floors.set(a.purpose.target, Math.min(floors.get(a.purpose.target) ?? Infinity, f));
      }
      const residuals = last?.residuals ?? {};
      const residue = Object.entries(residuals).reduce((s, [t, r]) => s + Math.max(0, r - (floors.get(t) ?? 0)), 0);
      return {
        ok: true,
        output_delta: {
          kind: "agent_generated",
          ok: true,
          summary: `smith: ${history.length} tick(s), ${steps.filter((s) => s["outcome"] === "commit").length} commit(s)`,
          agents: town.agents.map(agentView),
          society: town.society,
          diagnostics,
          steps,
          finalCounts,
          residuals,
          quiescent: last?.quiescent ?? false,
          done: last?.done ?? false,
        },
        residue,
        completed: !!(last && (last.done || last.quiescent)),
      };
    },

    outputCell: () => ({ kind: "agent_generated_cell" }),
  };
}
