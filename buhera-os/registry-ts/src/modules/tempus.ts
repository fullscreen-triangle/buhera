/* ============================================================================
 * tempus — the stella-lorraine Tempus timing language (specification
 * specs/tempus.md). Wraps the web library (web/src/lib/tempus), vendored.
 *
 * A Tempus program declares cells (intervals in ΔP-space), sync channels, a
 * composition and `when <cell> do <action>` rules. `compile` runs the
 * engine's own static checks. `simulate` drives the engine's SEEDED synthetic
 * event generator: ΔP values are pseudo-random, and actions are labels that
 * never execute — the delta says so, and anomaly counts are not quality.
 *
 * The Rust `tempus` crate (an LHC trigger kernel with no text grammar) is a
 * different system and is not bound here; see the spec's Rust section.
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

export interface TempusDiag {
  severity: "error" | "warning" | "info";
  message: string;
  pos?: { line?: number; col?: number } | number;
}

/** The subset of the vendored tempus library this adapter uses. */
export interface TempusEngine {
  compile(src: string): {
    ok: boolean;
    diagnostics: TempusDiag[];
    runtime: unknown | null;
    registry: unknown | null;
  };
  createSimulator(
    program: unknown,
    cfg: { totalEvents: number; noiseSigma: number; seed: number; batchSize: number },
  ): { generateBatch(): TempusEvent[]; isDone(): boolean; getEvents(): TempusEvent[] };
  compileConstruct?(src: string): { ok: boolean; diagnostics: TempusDiag[]; scene: unknown };
  compileComposition?(src: string): { ok: boolean; diagnostics: TempusDiag[]; scene: unknown };
}

export interface TempusEvent {
  cell: string;
  phase: string;
  actionFired?: string | null;
  [k: string]: unknown;
}

export const TEMPUS_ID = "tempus";

/** The "reactor coolant" lesson program (web/src/lib/tempus/lessons.ts). */
const DEMO_SOURCE = `sync coolant at 10.0e6 freq
cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7) action 1
cell HOT      bounds ( 5.0e-7, 2.0e-6) action 2
cell CRITICAL bounds ( 2.0e-6, 1.0e-5) action 3
compose d=1 channels coolant into coolant_traj
when NOMINAL  do emit status_ok
when WARM     do emit status_warn
when HOT      do begin
                 emit status_hot;
                 fire reduce_power
               end
when CRITICAL do begin
                 emit scram;
                 fire emergency_shutdown
               end`;

const lineOf = (d: TempusDiag): number | undefined =>
  typeof d.pos === "object" && d.pos && typeof d.pos.line === "number" ? d.pos.line : undefined;

function diagErrors(diags: TempusDiag[]) {
  return diags
    .filter((d) => d.severity === "error")
    .map((d) => {
      const line = lineOf(d);
      return line === undefined ? { message: d.message } : { message: d.message, line };
    });
}

/** JSON-safe copy (the engine's runtime holds ES Maps). */
function plain(v: unknown): Json {
  return JSON.parse(
    JSON.stringify(v, (_k, x) => (x instanceof Map ? Object.fromEntries(x) : x instanceof Set ? [...x] : x)),
  ) as Json;
}

export function tempusValidate(engine: TempusEngine) {
  return (source: string): Validation => {
    const r = engine.compile(source);
    const errors = diagErrors(r.diagnostics);
    return r.ok && errors.length === 0 ? { ok: true, errors: [] } : { ok: false, errors: errors.length ? errors : [{ message: "rejected" }] };
  };
}

export function tempusDsl(engine: TempusEngine): DslEntry {
  return { id: "tempus", label: "Tempus", extension: ".tempus", moduleId: TEMPUS_ID, packId: "tempus", validate: tempusValidate(engine) };
}

function num(o: Record<string, Json>, k: string, dflt: number): number {
  const v = o[k];
  return typeof v === "number" && Number.isFinite(v) ? v : dflt;
}

export function makeTempusModule(engine: TempusEngine): Module {
  const expected = 'Tempus source, "demo", or { kind: compile|simulate|construct|compose, source, … }';
  return {
    id: TEMPUS_ID,
    describe: () => ({
      id: TEMPUS_ID,
      description:
        "Tempus — the stella-lorraine timing language: cells over the timing residual ΔP, sync channels, " +
        "trajectory composition and when→action rules, statically checked; plus a seeded synthetic-event " +
        "simulator (ΔP values are pseudo-random; actions are labels and never execute).",
      instructions: [
        'dispatch("tempus", "demo")',
        'dispatch("tempus", { kind: "simulate", source, totalEvents: 200, seed: 42 })',
        'dispatch("tempus", { kind: "compose", source: "space d=3 nmax=8\\nrefine Sk 1\\nrefine St 1\\nspin +" })',
      ],
      dsl: "tempus",
      binding: "native",
    }),

    execute(instruction: Instruction, actBudget = 1): ActResult {
      const obj = typeof instruction === "object" && instruction && !Array.isArray(instruction) ? instruction : null;
      const kind = obj ? obj["kind"] : "compile";
      let source: string | null = null;
      if (instruction == null || instruction === "" || instruction === "demo") source = DEMO_SOURCE;
      else if (typeof instruction === "string") source = instruction;
      else if (obj && typeof obj["source"] === "string") source = obj["source"];
      if (source == null || typeof kind !== "string") return invalid(TEMPUS_ID, expected);

      try {
        if (kind === "construct" || kind === "compose") {
          const f = kind === "construct" ? engine.compileConstruct : engine.compileComposition;
          if (!f) return fail([`tempus: this engine build does not expose ${kind}`], "unavailable");
          const r = f(source);
          const errors = diagErrors(r.diagnostics);
          return {
            ok: r.ok,
            output_delta: { kind: `tempus_${kind}`, summary: `tempus ${kind}: ${r.ok ? "ok" : "rejected"}`, scene: plain(r.scene), diagnostics: plain(r.diagnostics) },
            residue: errors.length,
            completed: true,
            ...(r.ok ? {} : { error: errors[0]?.message ?? "rejected" }),
          };
        }

        const c = engine.compile(source);
        const errors = diagErrors(c.diagnostics);
        if (!c.ok || !c.runtime) {
          return {
            ok: false,
            output_delta: {
              kind: "tempus_compile",
              summary: `tempus: ${errors.length} error(s)`,
              diagnostics: plain(c.diagnostics),
              lines: errors.map((e) => `  ${e.line !== undefined ? `line ${e.line}: ` : ""}${e.message}`),
            },
            residue: errors.length,
            completed: true,
            error: errors[0]?.message ?? "compile failed",
          };
        }
        if (kind === "compile") {
          return done(
            { kind: "tempus_compile", summary: "tempus: compiled", diagnostics: plain(c.diagnostics), registry: plain(c.registry) },
            0,
          );
        }
        if (kind !== "simulate" || !obj) return invalid(TEMPUS_ID, expected);

        const totalEvents = Math.max(1, Math.floor(num(obj, "totalEvents", 200)));
        const batchSize = Math.max(1, Math.floor(num(obj, "batchSize", 50)));
        const sim = engine.createSimulator(c.runtime, {
          totalEvents,
          noiseSigma: num(obj, "noiseSigma", 0.1),
          seed: Math.floor(num(obj, "seed", 42)),
          batchSize,
        });
        // One act-budget unit = one engine batch (contract M6).
        for (let i = 0; i < Math.max(1, actBudget) && !sim.isDone(); i++) sim.generateBatch();
        const events = sim.getEvents();
        const byCell: Record<string, number> = {};
        const byAction: Record<string, number> = {};
        const byPhase: Record<string, number> = {};
        for (const e of events) {
          byCell[e.cell] = (byCell[e.cell] ?? 0) + 1;
          byPhase[e.phase] = (byPhase[e.phase] ?? 0) + 1;
          if (e.actionFired) byAction[e.actionFired] = (byAction[e.actionFired] ?? 0) + 1;
        }
        const remaining = Math.max(0, totalEvents - events.length);
        return {
          ok: true,
          output_delta: {
            kind: "tempus_simulation",
            summary: `tempus: ${events.length}/${totalEvents} synthetic event(s), ${byCell["anomaly"] ?? 0} anomaly`,
            synthetic: true,
            generated: events.length,
            total_events: totalEvents,
            by_cell: byCell,
            by_action: byAction,
            by_phase: byPhase,
            events: plain(events.slice(-200)),
            truncated: events.length > 200,
          },
          // Residue = fraction of requested events not yet generated (progress
          // only; says nothing about the program).
          residue: remaining / totalEvents,
          completed: remaining === 0,
        };
      } catch (err) {
        return fail([`tempus: unexpected error — ${errorText(err)}`], errorText(err));
      }
    },

    outputCell: () => ({ kind: "tempus_cell" }),
  };
}
