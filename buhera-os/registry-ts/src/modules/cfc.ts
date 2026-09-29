/* ============================================================================
 * cfc — cause-for-concern: thermodynamic cycle-consistency experiments on
 * metabolic circuits (specification specs/cfc.md). Wraps syndrome's JS port
 * of the reference implementation (lexer, parser, kernel, interpreter),
 * vendored.
 *
 * A run is atomic and returns the engine's record verbatim. Its terminal
 * status is a result, not a failure: OK, NEGATIVE (an assertion failed — a
 * real scientific finding) and INVALID (the reference failed its own check —
 * no conclusion is licensed) are all ok:true acts. Only ERROR (the program
 * could not be run) is ok:false.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

interface CfcVerdict {
  verdict: string;
  tolerance?: { dataAvailable?: boolean };
}

interface CfcRecord {
  status: "OK" | "NEGATIVE" | "INVALID" | "ERROR";
  committedMeasurements: number;
  verdicts: CfcVerdict[];
  error: string | null;
  errorLine?: number | null;
  [key: string]: unknown;
}

/** The subset of the vendored cfc engine this adapter uses. */
export interface CfcEngine {
  parse(src: string): unknown;
  runSource(src: string, name?: string): CfcRecord;
  EXAMPLES?: Record<string, string>;
  DEFAULT_FILE?: string;
}

export const CFC_ID = "cfc";

/** Sets and Maps in the record (witness sets, circuit tables) become arrays. */
function plain(value: unknown): unknown {
  if (value instanceof Map) return Object.fromEntries([...value.entries()].map(([k, v]) => [String(k), plain(v)]));
  if (value instanceof Set) return [...value].map(plain);
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]));
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

export function cfcValidate(engine: CfcEngine) {
  return (source: string): Validation => {
    try {
      engine.parse(source);
      return { ok: true, errors: [] };
    } catch (err) {
      const line = (err as { line?: unknown }).line;
      return { ok: false, errors: [typeof line === "number" ? { message: errorText(err), line } : { message: errorText(err) }] };
    }
  };
}

export function cfcDsl(engine: CfcEngine): DslEntry {
  return { id: "cfc", label: "cause-for-concern", extension: ".cfc", moduleId: CFC_ID, packId: "cfc", validate: cfcValidate(engine) };
}

export function makeCfcModule(engine: CfcEngine): Module {
  const expected = 'cfc source, "demo", or { kind: "run" | "check", source, name? } | { kind: "example", name }';
  return {
    id: CFC_ID,
    describe: () => ({
      id: CFC_ID,
      description:
        "cause-for-concern — build and solve metabolic circuits, compute cycle holonomies and their two error floors " +
        "(numerical, data), and admit three-valued verdicts (CONSISTENT / UNDECIDABLE / INCONSISTENT) that never " +
        "exist without their tolerance. Returns the full run record.",
      instructions: [
        'dispatch("cfc", "demo")',
        'dispatch("cfc", "<source>")',
        'dispatch("cfc", { kind: "check", source })',
        'dispatch("cfc", { kind: "example", name: "02_undecidable.cfc" })',
      ],
      dsl: "cfc",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      const examples = engine.EXAMPLES ?? {};
      let kind = "run";
      let source: string | null = null;
      let name = "<program>";
      if (instruction === "demo" && engine.DEFAULT_FILE) {
        source = examples[engine.DEFAULT_FILE] ?? null;
        name = engine.DEFAULT_FILE;
      } else if (typeof instruction === "string") source = instruction;
      else if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
        if (typeof instruction["kind"] === "string") kind = instruction["kind"];
        if (typeof instruction["name"] === "string") name = instruction["name"];
        if (kind === "example") {
          source = examples[name] ?? null;
          if (source == null) return fail([`cfc: no example "${name}" (have: ${Object.keys(examples).join(", ")})`], "unknown example");
          kind = "run";
        } else if (typeof instruction["source"] === "string") source = instruction["source"];
      }
      if (source == null || !["run", "check"].includes(kind)) return invalid(CFC_ID, expected);

      if (kind === "check") {
        const v = cfcValidate(engine)(source);
        return v.ok
          ? done({ kind: "cfc_check", ok: true, summary: "cfc: program parses" }, 0)
          : { ok: false, output_delta: { kind: "cfc_check", ok: false, errors: v.errors }, residue: v.errors.length, completed: true, error: v.errors[0]!.message };
      }

      let rec: CfcRecord;
      try {
        rec = engine.runSource(source, name);
      } catch (err) {
        // The interpreter rethrows unexpected JS errors; contain them (R2).
        return fail([`cfc: runtime failure — ${errorText(err)}`], errorText(err));
      }
      const record = plain(rec) as Record<string, unknown>;
      const undecided = rec.verdicts.filter((v) => v.verdict === "UNDECIDABLE").length;
      const delta = {
        kind: "cfc_record",
        summary: `cfc: ${rec.status} — ${rec.verdicts.length} verdict(s), ${rec.committedMeasurements} committed measurement(s)`,
        ...record,
      };
      if (rec.status === "ERROR") {
        return { ok: false, output_delta: delta, residue: 1, completed: true, error: rec.error ?? "cfc: the program could not be run" };
      }
      // Residue: the questions this run could not settle — UNDECIDABLE verdicts
      // (the data floor swallows the holonomy), plus one if the reference was
      // INVALID (no conclusion is licensed at all).
      return done(delta, undecided + (rec.status === "INVALID" ? 1 : 0));
    },

    outputCell: () => ({ kind: "cfc_record_cell" }),
  };
}
