/* ============================================================================
 * hfq — the hegel federated query interpreter (specification specs/hfq.md).
 *
 * Wraps @hegel/hfq, vendored unmodified. A plan is run through the engine's
 * own pipeline (parse → resolve → check → allocate → execute → emit) against
 * a local fixture world; the six-verdict document is returned verbatim under
 * `result`. A static capability refusal is a legitimate typed outcome
 * (contract A1): ok stays true, and the delta says `refused_statically`.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

/** The subset of @hegel/hfq (+ its /plans subpath) this adapter uses. */
export interface HfqEngine {
  runPlan(source: string): HfqResult;
  parse(source: string): unknown;
  selectWorld(plan: unknown): { world: string | null; unknown: string[] };
  PLANS: Array<{ id: string; section: string; blurb: string; source: string }>;
  SECTIONS: Record<string, unknown>;
}

export interface HfqStep {
  step: string;
  verdict: "answer" | "empty" | "surface" | "timeout" | "refused" | "starved";
  blocker?: string;
  [k: string]: unknown;
}

export interface HfqResult {
  ok: boolean;
  stage?: string;
  error?: string;
  line?: number;
  plan?: string;
  world?: string;
  steps: HfqStep[];
  requests_issued?: number;
  halted_early?: boolean;
  emitted?: Record<string, unknown>;
  [k: string]: unknown;
}

export const HFQ_ID = "hfq";

const DEMO_SOURCE = `plan healthy_chain {
  budget 200 requests
  let acids = from chebi ask descendants_of("CHEBI:1") within 10
  let kegg  = map acids via chebi2kegg expect partial 0.6
  let rxns  = from rhea ask reactions_consuming(?c) with ?c in kegg within 60
  emit rxns
}`;

/** Verdicts with no blocker: the step did what it could (def:blocker). */
const UNBLOCKED = new Set(["answer", "empty"]);

function tally(steps: HfqStep[]): { verdicts: Record<string, number>; blockers: Record<string, number> } {
  const verdicts: Record<string, number> = {};
  const blockers: Record<string, number> = {};
  for (const s of steps) {
    verdicts[s.verdict] = (verdicts[s.verdict] ?? 0) + 1;
    if (s.blocker) blockers[s.blocker] = (blockers[s.blocker] ?? 0) + 1;
  }
  return { verdicts, blockers };
}

function summarize(r: HfqResult): string {
  if (!r.ok) return `HFQ: failed at stage "${r.stage}" — ${r.error}`;
  const parts = Object.entries(tally(r.steps).verdicts).map(([v, n]) => `${n} ${v}`);
  return `HFQ: ${r.plan} [${r.world}] — ${parts.join(", ")}, ${r.requests_issued} request(s)`;
}

export function hfqValidate(engine: HfqEngine) {
  return (source: string): Validation => {
    let plan: unknown;
    try {
      plan = engine.parse(source);
    } catch (err) {
      const e = err as { message?: string; line?: number };
      const message = e?.message ?? errorText(err);
      return { ok: false, errors: [typeof e?.line === "number" ? { message, line: e.line } : { message }] };
    }
    const w = engine.selectWorld(plan);
    if (w.unknown.length) {
      return { ok: false, errors: [{ message: `unknown source(s): ${w.unknown.join(", ")} (no fixture world declares them)` }] };
    }
    return { ok: true, errors: [] };
  };
}

export function hfqDsl(engine: HfqEngine): DslEntry {
  return { id: "hfq", label: "HFQ plan", extension: ".hfq", moduleId: HFQ_ID, packId: "hfq", validate: hfqValidate(engine) };
}

export function makeHfqModule(engine: HfqEngine): Module {
  const expected = 'plan source, "demo", { kind: "run", source }, { kind: "preset", id }, { kind: "check", source } or { kind: "list_presets" }';
  return {
    id: HFQ_ID,
    describe: () => ({
      id: HFQ_ID,
      description:
        "HFQ — the hegel federated query interpreter, vendored unmodified. Runs a plan (parse → resolve → check → " +
        "allocate → execute → emit) against a local fixture world and returns a six-verdict document " +
        "(answer/empty/surface/timeout/refused/starved) with a named blocker where one applies.",
      instructions: [
        'dispatch("hfq", "demo")',
        'dispatch("hfq", { kind: "preset", id: "mark_q1" })',
        'dispatch("hfq", { kind: "check", source })',
        'dispatch("hfq", { kind: "list_presets" })',
      ],
      dsl: "hfq",
      binding: "native",
    }),

    execute(instruction: Instruction): ActResult {
      const obj = typeof instruction === "object" && instruction && !Array.isArray(instruction) ? instruction : null;
      if (obj?.["kind"] === "list_presets") {
        return done(
          {
            kind: "hfq_presets",
            summary: `HFQ: ${engine.PLANS.length} preset plan(s)`,
            sections: Object.keys(engine.SECTIONS),
            plans: engine.PLANS.map((p) => ({ id: p.id, section: p.section, blurb: p.blurb })),
          },
          0,
        );
      }

      let source: string | null = null;
      if (instruction == null || instruction === "" || instruction === "demo") source = DEMO_SOURCE;
      else if (typeof instruction === "string") source = instruction;
      else if (obj && (obj["kind"] === "run" || obj["kind"] === "check") && typeof obj["source"] === "string") source = obj["source"];
      else if (obj?.["kind"] === "preset" && typeof obj["id"] === "string") {
        const p = engine.PLANS.find((x) => x.id === obj["id"]);
        if (!p) return fail([`hfq: unknown preset "${obj["id"]}"`], "unknown preset");
        source = p.source;
      }
      if (source == null) return invalid(HFQ_ID, expected);

      if (obj?.["kind"] === "check") {
        const v = hfqValidate(engine)(source);
        return done({ kind: "dsl_validation", dsl: "hfq", ok: v.ok, errors: v.errors }, v.errors.length);
      }

      let r: HfqResult;
      try {
        r = engine.runPlan(source);
      } catch (err) {
        return fail([`hfq: unexpected error — ${errorText(err)}`], errorText(err));
      }
      if (!r.ok) {
        return {
          ok: false,
          output_delta: { kind: "hfq_result", summary: summarize(r), result: r },
          residue: 0,
          completed: true,
          error: r.error ?? "plan failed",
        };
      }
      const { verdicts, blockers } = tally(r.steps);
      const blocked = r.steps.filter((s) => !UNBLOCKED.has(s.verdict)).length;
      return done(
        {
          kind: "hfq_result",
          summary: summarize(r),
          result: r,
          verdicts,
          blockers,
          halted_early: r.halted_early === true,
          refused_statically: r.halted_early === true && !!r.emitted && "refusal" in r.emitted,
        },
        // Residue = steps that did not reach an unblocked verdict. `empty` is
        // an answer (no blocker) and does not count.
        blocked,
      );
    },

    outputCell: () => ({ kind: "hfq_cell" }),
  };
}
