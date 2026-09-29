/* ============================================================================
 * honjo — Honjo Masamune, the cut calculus for chemistry (specification
 * specs/honjo.md). Wraps borgia's honjo bundle (lexer, parser, checker,
 * Cut-IR lowering, shell derivation, interpreter), vendored.
 *
 * Programs are straight-line (there are no loops), so an act runs the whole
 * program; the cut count M is honjo's own monotone clock and is reported, not
 * re-derived. Values carry honjo's per-value floor and residue — a property
 * of the chemistry, which is NOT the act residue (that is always 0 here: a
 * program that ran has no remaining work).
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

interface HonjoRun {
  cutCount: number;
  floor: number;
  named: Record<string, unknown>;
  log: string[];
  ok: boolean;
}

/** The subset of the vendored honjo bundle this adapter uses. */
export interface HonjoEngine {
  compile(src: string): unknown;
  evaluate(src: string): HonjoRun;
  deriveAtom(z: number): unknown;
  renderValue(value: unknown): string;
}

export const HONJO_ID = "honjo";

/** borgia honjo-masamune/honjo/examples/track.hj. */
const DEMO = `-- track.hj — tracking an item through a process (the causal table)
floor 1.0
import honjo.causal

O := cut 8
H := cut 1
W := close O(H, H)

path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path
`;

function plain(value: unknown): unknown {
  if (value instanceof Map) return Object.fromEntries([...value.entries()].map(([k, v]) => [String(k), plain(v)]));
  if (value instanceof Set) return [...value].map(plain);
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]));
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

/** honjo errors carry `line` and `col` (their messages say "at L:C", not "line N"). */
function diagnostic(err: unknown): { message: string; line?: number; column?: number } {
  const e = err as { line?: unknown; col?: unknown };
  const out: { message: string; line?: number; column?: number } = { message: errorText(err) };
  if (typeof e?.line === "number") out.line = e.line;
  if (typeof e?.col === "number") out.column = e.col;
  return out;
}

export function honjoValidate(engine: HonjoEngine) {
  return (source: string): Validation => {
    try {
      engine.compile(source);
      return { ok: true, errors: [] };
    } catch (err) {
      return { ok: false, errors: [diagnostic(err)] };
    }
  };
}

export function honjoDsl(engine: HonjoEngine): DslEntry {
  return { id: "honjo", label: "Honjo Masamune", extension: ".hj", moduleId: HONJO_ID, packId: "honjo", validate: honjoValidate(engine) };
}

export function makeHonjoModule(engine: HonjoEngine): Module {
  const expected = 'honjo source, "demo", { kind: "run" | "compile", source }, or { kind: "derive", z }';
  return {
    id: HONJO_ID,
    describe: () => ({
      id: HONJO_ID,
      description:
        "Honjo Masamune — chemistry as cuts: generate atoms (shell structure derived, Z = 1..118), form bonds, close " +
        "compounds (stoichiometry and geometry follow), track items through processes. Every value carries a floor and a " +
        "residue; the cut count M is a monotone clock.",
      instructions: [
        'dispatch("honjo", "demo")',
        'dispatch("honjo", "floor 1.0\\nO := cut 8\\nH := cut 1\\nW := close O(H, H)\\nobserve W")',
        'dispatch("honjo", { kind: "derive", z: 26 })',
      ],
      dsl: "honjo",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      let kind = "run";
      let source: string | null = null;
      if (instruction === "demo") source = DEMO;
      else if (typeof instruction === "string") source = instruction;
      else if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
        if (typeof instruction["kind"] === "string") kind = instruction["kind"];
        if (typeof instruction["source"] === "string") source = instruction["source"];
        if (kind === "derive") {
          const z = instruction["z"];
          if (typeof z !== "number" || !Number.isInteger(z)) return invalid(HONJO_ID, '{ kind: "derive", z: <integer> }');
          try {
            return done({ kind: "honjo_atom", z, atom: plain(engine.deriveAtom(z)) }, 0);
          } catch (err) {
            return fail([`honjo: ${errorText(err)}`], errorText(err));
          }
        }
      }
      if (source == null || !["run", "compile"].includes(kind)) return invalid(HONJO_ID, expected);

      if (kind === "compile") {
        const v = honjoValidate(engine)(source);
        return v.ok
          ? done({ kind: "honjo_result", ok: true, summary: "honjo: program compiles", log: [] }, 0)
          : { ok: false, output_delta: { kind: "honjo_result", ok: false, errors: v.errors }, residue: 1, completed: true, error: v.errors[0]!.message };
      }

      let r: HonjoRun;
      try {
        r = engine.evaluate(source);
      } catch (err) {
        // Compile errors, and range errors that surface only at run (cut 200).
        const d = diagnostic(err);
        return { ok: false, output_delta: { kind: "honjo_result", ok: false, errors: [d], log: [] }, residue: 1, completed: true, error: d.message };
      }
      const render = (v: unknown) => {
        try {
          return engine.renderValue(v);
        } catch {
          return String(v);
        }
      };
      const values = Object.fromEntries(Object.entries(r.named).map(([k, v]) => [k, render(v)]));
      const delta = {
        kind: "honjo_result",
        ok: r.ok,
        summary: `honjo: M = ${r.cutCount} cut(s), floor ${r.floor}${r.ok ? "" : " — an assertion aborted the program"}`,
        log: r.log,
        cutCount: r.cutCount,
        floor: r.floor,
        values,
        named: plain(r.named),
      };
      if (!r.ok) return { ok: false, output_delta: delta, residue: 0, completed: true, error: "assert aborted" };
      return done(delta, 0);
    },

    outputCell: () => ({ kind: "honjo_result_cell" }),
  };
}
