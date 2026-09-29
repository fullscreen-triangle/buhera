/* ============================================================================
 * scope — SCOPE, the microscopy-analysis language (specification
 * specs/scope.md). Wraps helicopter's scope-lang (compiler: lexer, parser,
 * type checker with the attainability analyses; runtime: observe → catalyze
 * → access → measure → visualise over an image), vendored.
 *
 * SCOPE runs as a REPL: cells accumulate into one growing program in a
 * session the module owns (R6), and a cell that reaches `visualise` runs the
 * program against the linked image. The session is per module instance; the
 * host reaches it through `linkImage` / `resetSession` (a terminal command
 * decodes an image and links it) rather than a module-level global.
 *
 * Residue of an executing cell: the declared goals the result did not meet.
 * (The former residue, sEntropy.sum, is normalised to 1 by the engine on
 * every run and so carried no information.)
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, DslError, Validation } from "../dsl.ts";

export interface ScopeImage {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

interface CellResult {
  ok: boolean;
  kind: "chart" | "define" | "error" | "noop";
  log: string[];
  result?: { goalStatus?: Array<{ passed: boolean }>; [key: string]: unknown };
  error?: string;
}

interface Session {
  setImage(image: ScopeImage): void;
  state(): { hasCoordinateSpace: boolean; hasChannels: boolean; morphisms: string[]; hasGoal: boolean; hasDispatch: boolean; hasImage: boolean };
  run(cell: string): Promise<CellResult>;
}

type CompileError = { kind: string; message?: string; line?: number; col?: number; [key: string]: unknown };

/** The subset of the vendored scope-lang this adapter uses. */
export interface ScopeEngine {
  compile(source: string): { ok: boolean; errors: CompileError[]; warnings: unknown[]; log: string[] };
  createSession(): Session;
}

export const SCOPE_ID = "scope";

/** A SCOPE module plus the host's handles on its session. */
export interface ScopeModule extends Module {
  linkImage(image: ScopeImage): void;
  resetSession(): void;
}

function describeError(e: CompileError): DslError {
  const { kind, message, line, col, ...rest } = e;
  const detail = message ?? Object.entries(rest).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ");
  const out: DslError = { message: `${kind}: ${detail}` };
  if (typeof line === "number" && line > 0) out.line = line;
  if (typeof col === "number" && col > 0) out.column = col;
  return out;
}

/** Whole-program validation: parse + type check (depth, cell overlap, entropy budget, grounding). */
export function scopeValidate(engine: ScopeEngine) {
  return (source: string): Validation => {
    const c = engine.compile(source);
    if (c.ok) return { ok: true, errors: [] };
    return { ok: false, errors: c.errors.length ? c.errors.map(describeError) : [{ message: c.log.join("\n") || "rejected" }] };
  };
}

export function scopeDsl(engine: ScopeEngine): DslEntry {
  return { id: "scope", label: "SCOPE", extension: ".scope", moduleId: SCOPE_ID, packId: "scope", validate: scopeValidate(engine) };
}

function isImage(x: unknown): x is ScopeImage {
  const o = x as { data?: unknown; width?: unknown; height?: unknown } | null;
  return !!o && typeof o === "object" && !!o.data && typeof o.width === "number" && typeof o.height === "number";
}

export function makeScopeModule(engine: ScopeEngine): ScopeModule {
  let session = engine.createSession();
  const expected = 'SCOPE source (a REPL cell), or { kind: "cell", source } | { kind: "check", source } | { kind: "load", image } | "state" | "reset"';

  return {
    id: SCOPE_ID,
    linkImage: (image) => session.setImage(image),
    resetSession: () => {
      session = engine.createSession();
    },
    describe: () => ({
      id: SCOPE_ID,
      description:
        "SCOPE — microscopy analysis as a REPL: declare channels, a coordinate space, goals and morphisms " +
        "(observe → catalyze → access → measure → visualise) cell by cell; a cell that visualises runs the program " +
        "against the linked image and returns the measurement, its uncertainty, S-entropy and goal status.",
      instructions: [
        'dispatch("scope", "coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.10  lambda_t 0.05 }")',
        'dispatch("scope", { kind: "check", source: "<whole program>" })',
        'dispatch("scope", "state")',
        'dispatch("scope", "reset")',
      ],
      dsl: "scope",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      if (instruction === "reset" || (instruction && typeof instruction === "object" && !Array.isArray(instruction) && instruction["kind"] === "reset")) {
        session = engine.createSession();
        return done({ kind: "text", lines: ["scope: session reset"] }, 0);
      }
      if (instruction === "state" || (instruction && typeof instruction === "object" && !Array.isArray(instruction) && instruction["kind"] === "state")) {
        const s = session.state();
        return done(
          {
            kind: "text",
            lines: [
              `coordinate_space: ${s.hasCoordinateSpace ? "set" : "—"}`,
              `channels: ${s.hasChannels ? "set" : "—"}`,
              `morphisms: ${s.morphisms.length ? s.morphisms.join(", ") : "—"}`,
              `goal: ${s.hasGoal ? "set" : "—"}  dispatch: ${s.hasDispatch ? "set" : "—"}`,
              `image: ${s.hasImage ? "linked" : "not linked"}`,
            ],
          },
          0,
        );
      }
      let source: string | null = null;
      if (typeof instruction === "string") source = instruction;
      else if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
        const kind = instruction["kind"];
        if (kind === "load") {
          if (!isImage(instruction["image"])) return invalid(SCOPE_ID, '{ kind: "load", image: { data, width, height } }');
          const img = instruction["image"] as unknown as ScopeImage;
          session.setImage(img);
          return done({ kind: "text", lines: [`scope: image linked (${img.width}×${img.height})`] }, 0);
        }
        if (kind === "check" && typeof instruction["source"] === "string") {
          const v = scopeValidate(engine)(instruction["source"]);
          return v.ok
            ? done({ kind: "text", lines: ["scope: program compiles"] }, 0)
            : { ok: false, output_delta: { kind: "text", lines: v.errors.map((e) => e.message), errors: v.errors }, residue: v.errors.length, completed: true, error: v.errors[0]!.message };
        }
        if (kind === "cell" && typeof instruction["source"] === "string") source = instruction["source"];
      }
      if (source == null) return invalid(SCOPE_ID, expected);

      let cell: CellResult;
      try {
        cell = await session.run(source);
      } catch (err) {
        return fail([`scope: ${errorText(err)}`], errorText(err));
      }
      if (cell.kind === "noop") return done({ kind: "text", lines: ["(empty cell)"] }, 0);
      if (cell.kind === "error") {
        return { ok: false, output_delta: { kind: "text", lines: cell.log.length ? cell.log : [cell.error ?? "error"] }, residue: 1, completed: true, error: cell.error ?? "error" };
      }
      if (cell.kind === "define") return done({ kind: "text", lines: cell.log }, 0);
      const unmet = (cell.result?.goalStatus ?? []).filter((g) => !g.passed).length;
      return done({ kind: "scope_run", result: cell.result, log: cell.log }, unmet);
    },

    outputCell: () => ({ kind: "scope_cell" }),
  };
}
