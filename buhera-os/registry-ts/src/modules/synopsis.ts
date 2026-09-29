/* ============================================================================
 * synopsis — the synopsis genomic scripting language (specification
 * specs/synopsis.md). Wraps gospel's TypeScript front end (synopsis/ts/src:
 * tokeniser, parser, Stage-B checker), vendored.
 *
 * Upstream ships NO evaluator, deliberately (gospel IDE-PLAN.md): a caller can
 * parse and typecheck a program, it cannot run one. So this module checks,
 * parses and tokenises; a request to run is refused rather than approximated.
 *
 * Refusals are SynopsisError subclasses carrying `className` (the name the
 * conformance corpus compares against) and a 1-based `line`; they are read
 * directly, not scraped from the message.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, DslError, Validation } from "../dsl.ts";

/** The subset of the vendored synopsis front end this adapter uses. */
export interface SynopsisEngine {
  parse(src: string): unknown;
  checkProgram(program: never): unknown;
  tokenise(src: string): unknown[];
}

export const SYNOPSIS_ID = "synopsis";

interface Refusal {
  className: string;
  message: string;
  line: number | null;
}

/** A SynopsisError is recognised by shape: `className` + nullable `line`. */
function asRefusal(err: unknown): Refusal | null {
  if (err instanceof Error && typeof (err as { className?: unknown }).className === "string") {
    const line = (err as { line?: unknown }).line;
    return { className: (err as unknown as { className: string }).className, message: err.message, line: typeof line === "number" ? line : null };
  }
  return null;
}

/** Maps (the AST's ordered parameter blocks) become ordered [key, value] pairs. */
function plain(value: unknown): unknown {
  if (value instanceof Map) return [...value.entries()].map(([k, v]) => [k, plain(v)]);
  if (value instanceof Set) return [...value].map(plain);
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]));
  }
  return value;
}

type Checked = { ok: true; report: unknown } | { ok: false; refusal: Refusal } | { ok: false; crash: string };

function runCheck(engine: SynopsisEngine, source: string): Checked {
  try {
    return { ok: true, report: engine.checkProgram(engine.parse(source) as never) };
  } catch (err) {
    const refusal = asRefusal(err);
    // A non-SynopsisError is the front end failing on its input (e.g. a
    // truncated program), not a refusal it chose; it is still a rejection.
    return refusal ? { ok: false, refusal } : { ok: false, crash: errorText(err) };
  }
}

export function synopsisValidate(engine: SynopsisEngine) {
  return (source: string): Validation => {
    const c = runCheck(engine, source);
    if (c.ok) return { ok: true, errors: [] };
    const e: DslError =
      "refusal" in c
        ? c.refusal.line != null
          ? { message: `${c.refusal.className}: ${c.refusal.message}`, line: c.refusal.line }
          : { message: `${c.refusal.className}: ${c.refusal.message}` }
        : { message: `front end failed on this input: ${c.crash}` };
    return { ok: false, errors: [e] };
  };
}

export function synopsisDsl(engine: SynopsisEngine): DslEntry {
  return { id: "synopsis", label: "synopsis", extension: ".syp", moduleId: SYNOPSIS_ID, packId: "synopsis", validate: synopsisValidate(engine) };
}

export function makeSynopsisModule(engine: SynopsisEngine): Module {
  const expected = 'synopsis source, or { kind: "check" | "parse" | "tokens", source }';
  return {
    id: SYNOPSIS_ID,
    describe: () => ({
      id: SYNOPSIS_ID,
      description:
        "synopsis — the genomic scripting language: parse and typecheck a program (frames, residues, required " +
        "parameters, termination) and emit the checker's report (frames, parameters, residues, claims, bounds). " +
        "There is no evaluator upstream; a program is checked, never run.",
      instructions: [
        'dispatch("synopsis", "<source>")',
        'dispatch("synopsis", { kind: "parse", source })',
        'dispatch("synopsis", { kind: "tokens", source })',
      ],
      dsl: "synopsis",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      let kind = "check";
      let source: string | null = null;
      if (typeof instruction === "string") source = instruction;
      else if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
        if (typeof instruction["kind"] === "string") kind = instruction["kind"];
        if (typeof instruction["source"] === "string") source = instruction["source"];
      }
      if (kind === "run" || kind === "evaluate") {
        return fail(
          ["synopsis: there is no evaluator — upstream exports parse and check only (gospel IDE-PLAN.md)"],
          "no evaluator exists",
        );
      }
      if (source == null || !["check", "parse", "tokens"].includes(kind)) return invalid(SYNOPSIS_ID, expected);

      if (kind === "tokens") {
        try {
          const tokens = engine.tokenise(source);
          return done({ kind: "synopsis_tokens", count: tokens.length, tokens: plain(tokens) }, 0);
        } catch (err) {
          return refused(asRefusal(err), errorText(err));
        }
      }
      if (kind === "parse") {
        try {
          return done({ kind: "synopsis_ast", ast: plain(engine.parse(source)) }, 0);
        } catch (err) {
          return refused(asRefusal(err), errorText(err));
        }
      }
      const c = runCheck(engine, source);
      if (c.ok) return done({ kind: "synopsis_report", ok: true, summary: "synopsis: program checks", report: plain(c.report) }, 0);
      return "refusal" in c ? refused(c.refusal, c.refusal.message) : refused(null, c.crash);
    },

    outputCell: () => ({ kind: "synopsis_report_cell" }),
  };
}

/** A refusal is a result, not a crash: ok:false with the checker's class and line. */
function refused(r: Refusal | null, fallback: string): ActResult {
  const message = r ? `${r.className}: ${r.message}` : `front end failed on this input: ${fallback}`;
  return {
    ok: false,
    output_delta: { kind: "synopsis_report", ok: false, summary: `synopsis: ${message}`, refusal: r ?? { className: null, message: fallback, line: null } },
    residue: 1,
    completed: true,
    error: message,
  };
}
