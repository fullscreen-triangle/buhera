/* ============================================================================
 * sbs — Systems Biology Shaders (specification specs/sbs.md).
 *
 * Wraps @sachikonye/sbs (hegel consequences/src/lib/sbs). The engine is
 * injected; this adapter only turns instructions into engine calls and engine
 * results into ActResults. It preserves the historical long-grass delta
 * (kind "sbs_result" with circuit/metrics/navigation/observations/
 * perturbations/warnings) that the MetricsDashboard renderer reads, and adds
 * the engine's known silent failure modes as explicit warnings.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

/** The subset of @sachikonye/sbs this adapter depends on. */
export interface SbsEngine {
  runSBS(source: string, opts?: { preferCPU?: boolean }): SbsResult;
  checkSBS(source: string): { valid: boolean; errors: Array<{ message: string; line?: number }> };
  suggestTherapy?(circuit: unknown, perturbation: unknown, maxEdges?: number): Array<{ idx: number; factor: number }>;
  compileSBS?(source: string): Record<string, unknown>;
  SCRIPTS?: Record<string, string>;
}

export interface SbsResult {
  ok: boolean;
  errors?: Array<{ message: string; line?: number }>;
  warnings?: Array<{ message: string; line?: number }>;
  circuit?: { numNodes: number; numEdges: number; [k: string]: unknown } | null;
  metrics?: { R: number; V: number; backend?: string; [k: string]: unknown } | null;
  navigation?: unknown;
  observations?: unknown[];
  perturbations?: Array<{ idx: number; factor: number }>;
}

export const SBS_ID = "sbs";

/** WebGL shader loop bound (observation.frag.js); larger circuits are wrong on GPU. */
const SHADER_LIMIT = 256;

const DEMO_SOURCE = `// Glycolysis — the canonical SBS demo circuit
circuit glycolysis {
  node Glucose  { mu: -917.0, concentration: 5.0, compartment: "cytoplasm" }
  node G6P      { mu: -1760.0, concentration: 0.083 }
  node F6P      { mu: -1755.0, concentration: 0.014 }
  node FBP      { mu: -2600.0, concentration: 0.031 }
  node G3P      { mu: -1290.0, concentration: 0.14 }
  node BPG13    { mu: -2356.0, concentration: 0.001 }
  node PG3      { mu: -1515.0, concentration: 0.1 }
  node PG2      { mu: -1510.0, concentration: 0.03 }
  node PEP      { mu: -1263.0, concentration: 0.023 }
  node Pyruvate { mu: -472.0, concentration: 0.051 }

  edge Glucose  -> G6P      { rate: 230.0, conductance: 464.1 }
  edge G6P      -> F6P      { rate: 100.0, conductance: 3.35 }
  edge F6P      -> FBP      { rate: 150.0, conductance: 0.85 }
  edge FBP      -> G3P      { rate: 80.0,  conductance: 1.0 }
  edge G3P      -> BPG13    { rate: 200.0, conductance: 11.3 }
  edge BPG13    -> PG3      { rate: 300.0, conductance: 0.12 }
  edge PG3      -> PG2      { rate: 180.0, conductance: 7.27 }
  edge PG2      -> PEP      { rate: 100.0, conductance: 1.21 }
  edge PEP      -> Pyruvate { rate: 500.0, conductance: 4.64 }
}

observe glycolysis
perturb glycolysis { factor: 0.1 }
navigate from Pyruvate
`;

function summarize(r: SbsResult): string {
  if (!r.circuit) return "SBS: compiled (no circuit declared)";
  const parts = [`${r.circuit.numNodes} nodes, ${r.circuit.numEdges} edges`];
  if (r.metrics) parts.push(`R=${r.metrics.R.toFixed(3)}`, `V=${r.metrics.V.toFixed(3)}`, `[${r.metrics.backend}]`);
  return `SBS: ${parts.join("  ")}`;
}

/** Warnings for the engine's documented silent failure modes (spec §Hazards). */
function adapterWarnings(source: string, r: SbsResult): string[] {
  const w: string[] = [];
  const c = r.circuit;
  if (c && r.metrics?.backend === "webgl2" && (c.numNodes > SHADER_LIMIT || c.numEdges > SHADER_LIMIT)) {
    w.push(`circuit exceeds the WebGL shader bound (${SHADER_LIMIT}); GPU metrics are unreliable — rerun with preferCPU`);
  }
  if (r.metrics && Number.isNaN(r.metrics.R)) w.push("coherence R is NaN (circuit too small for a rank correlation)");
  if (/perturb[^\n{]*\{[^}]*\bedge\s*:/.test(source)) {
    w.push("perturb { edge: … } is ignored by the engine: the highest-flux edge is perturbed");
  }
  if ((source.match(/^\s*circuit\s/gm) || []).length > 1) {
    w.push("more than one circuit block: the engine does not offset the second block's edge indices");
  }
  return w;
}

function sourceOf(instruction: Instruction, engine: SbsEngine): string | null {
  if (instruction == null || instruction === "" || instruction === "demo") return DEMO_SOURCE;
  if (typeof instruction === "string") return instruction;
  if (typeof instruction === "object" && !Array.isArray(instruction)) {
    const src = instruction["source"];
    if (typeof src === "string") return src;
    const id = instruction["id"];
    if (instruction["kind"] === "script" && typeof id === "string") return engine.SCRIPTS?.[id] ?? null;
  }
  return null;
}

export function sbsValidate(engine: SbsEngine) {
  return (source: string): Validation => {
    const v = engine.checkSBS(source);
    if (v.valid) return { ok: true, errors: [] };
    const errors = (v.errors ?? []).map((e) => {
      // The engine reports line 0 and puts the position in the message text.
      const m = /line\s+(\d+)/i.exec(e.message);
      const line = e.line && e.line > 0 ? e.line : m ? Number(m[1]) : undefined;
      return line === undefined ? { message: e.message } : { message: e.message, line };
    });
    return { ok: false, errors: errors.length ? errors : [{ message: "rejected" }] };
  };
}

export function sbsDsl(engine: SbsEngine): DslEntry {
  return { id: "sbs", label: "SBS", extension: ".sbs", moduleId: SBS_ID, packId: "sbs", validate: sbsValidate(engine) };
}

export function makeSbsModule(engine: SbsEngine): Module {
  const expected = '.sbs source, "demo", or { kind: run|check|compile|therapy|script|list_scripts, … }';
  return {
    id: SBS_ID,
    describe: () => ({
      id: SBS_ID,
      description:
        "Systems Biology Shaders — compile and run .sbs scripts: build a cellular circuit, render the S-entropy " +
        "observation (WebGL2, CPU fallback), return coherence R, flux visibility V and backward navigation.",
      instructions: [
        'dispatch("sbs", "demo")',
        'dispatch("sbs", { kind: "run", source, preferCPU: true })',
        'dispatch("sbs", { kind: "therapy", source, maxEdges: 3 })',
        'dispatch("sbs", { kind: "script", id: "glycolysis.sbs" })',
      ],
      dsl: "sbs",
      binding: "native",
    }),

    execute(instruction: Instruction): ActResult {
      const kind = typeof instruction === "object" && instruction && !Array.isArray(instruction) ? instruction["kind"] : null;
      if (kind === "list_scripts") {
        const ids = Object.keys(engine.SCRIPTS ?? {});
        return done({ kind: "sbs_scripts", summary: `SBS: ${ids.length} script(s)`, scripts: ids }, 0);
      }
      const source = sourceOf(instruction, engine);
      if (source == null) return invalid(SBS_ID, expected);

      if (kind === "check") {
        const v = sbsValidate(engine)(source);
        return done({ kind: "dsl_validation", dsl: "sbs", ok: v.ok, errors: v.errors }, v.errors.length);
      }
      if (kind === "compile") {
        if (!engine.compileSBS) return fail(["sbs: this engine build does not expose compileSBS"], "unavailable");
        const c = engine.compileSBS(source);
        return done({ kind: "sbs_compiled", summary: `SBS: compiled (${c["success"] ? "ok" : "failed"})`, compiled: c }, 0);
      }

      const preferCPU =
        typeof instruction === "object" && instruction && !Array.isArray(instruction) && instruction["preferCPU"] === true;
      let r: SbsResult;
      try {
        r = engine.runSBS(source, preferCPU ? { preferCPU: true } : undefined);
      } catch (err) {
        return fail([`sbs: unexpected error — ${errorText(err)}`], errorText(err));
      }
      if (!r.ok) {
        const msgs = (r.errors ?? []).map((e) => `  line ${e.line ?? 0}: ${e.message}`);
        return fail(["sbs: compile failed", ...msgs], r.errors?.[0]?.message ?? "compile failed");
      }

      let therapy: unknown = undefined;
      if (kind === "therapy") {
        if (!engine.suggestTherapy) return fail(["sbs: this engine build does not expose suggestTherapy"], "unavailable");
        const maxEdges =
          typeof instruction === "object" && instruction && !Array.isArray(instruction) && typeof instruction["maxEdges"] === "number"
            ? (instruction["maxEdges"] as number)
            : 3;
        therapy = r.circuit ? engine.suggestTherapy(r.circuit, r.perturbations ?? [], maxEdges) : [];
      }

      const V = r.metrics?.V;
      // Residue = 1 − V: distance of the observed flux pattern from the healthy
      // baseline (0 unperturbed). Without metrics nothing was observed: 0.
      const residue = typeof V === "number" && Number.isFinite(V) ? Math.max(0, 1 - V) : 0;
      return done(
        {
          kind: "sbs_result",
          summary: summarize(r),
          circuit: r.circuit ?? null,
          metrics: r.metrics ?? null,
          navigation: r.navigation ?? null,
          observations: r.observations ?? [],
          perturbations: r.perturbations ?? [],
          warnings: r.warnings ?? [],
          adapter_warnings: adapterWarnings(source, r),
          backend: r.metrics?.backend ?? null,
          ...(therapy !== undefined ? { therapy } : {}),
        },
        residue,
      );
    },

    outputCell: () => ({ kind: "sbs_cell" }),
  };
}
