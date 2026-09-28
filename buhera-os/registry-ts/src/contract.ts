/* ============================================================================
 * The Module contract — specification 02-module-contract.md.
 *
 * Field-for-field twin of buhera-registry::contract (Rust). Everything that
 * crosses the contract is JSON-shaped so an act can be forwarded between the
 * TypeScript and Rust hosts without translation.
 * ========================================================================== */

/** JSON value. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * What a caller asks a module to do: DSL source / a bare verb (string), or
 * an object whose `kind` names the operation. Each module's specification
 * enumerates the shapes it accepts.
 */
export type Instruction = Json;

/** Renderable payload; `kind` names the renderer. */
export type OutputDelta = { kind: string; [key: string]: unknown };

/** The outcome of one act. */
export interface ActResult {
  /** Did the act achieve what the instruction asked for? */
  ok: boolean;
  /** What the act produced. null only on the registry's contained-throw path. */
  output_delta: OutputDelta | null;
  /** Work remaining, in the module's declared unit. 0 = nothing left. */
  residue: number;
  /** Did the act finish within its budget? */
  completed: boolean;
  /** Machine-readable failure reason when ok is false. */
  error?: string;
}

/** The cell a host allocates to render an instruction's output. */
export interface OutputCell {
  kind: string;
}

/** How a module reaches its engine. */
export type BindingKind = "native" | "remote" | "bridge";

/** A module's self-description. */
export interface Descriptor {
  id: string;
  description: string;
  instructions: string[];
  dsl?: string;
  binding: BindingKind;
}

/** A federation member. */
export interface Module {
  readonly id: string;
  describe(): Descriptor;
  execute(instruction: Instruction, actBudget: number): Promise<ActResult> | ActResult;
  outputCell?(instruction: Instruction): OutputCell;
}

/* ── Result constructors (mirror ActResult::done/fail/invalid/text) ─────── */

export function done(outputDelta: OutputDelta, residue: number): ActResult {
  return { ok: true, output_delta: outputDelta, residue, completed: true };
}

export function fail(lines: string[], error: string): ActResult {
  return { ok: false, output_delta: { kind: "text", lines }, residue: 0, completed: true, error };
}

export function invalid(moduleId: string, expected: string): ActResult {
  return fail([`${moduleId}: instruction must be ${expected}`], "invalid instruction");
}

export function text(lines: string[], residue = 0): ActResult {
  return done({ kind: "text", lines }, residue);
}

/** The conventional kind of an instruction (string itself, or `.kind`). */
export function instructionKind(instruction: Instruction): string | null {
  if (typeof instruction === "string") return instruction;
  if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
    const k = instruction["kind"];
    return typeof k === "string" ? k : null;
  }
  return null;
}

/** Normalise an instruction to an object: a string becomes `{ kind: s }`. */
export function asObject(instruction: Instruction, stringField = "kind"): Record<string, Json> {
  if (typeof instruction === "string") return { [stringField]: instruction };
  if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) return instruction;
  return {};
}

/** A string field of an object instruction. */
export function fieldStr(instruction: Instruction, key: string): string | null {
  if (instruction && typeof instruction === "object" && !Array.isArray(instruction)) {
    const v = instruction[key];
    return typeof v === "string" ? v : null;
  }
  return null;
}

/** Error text of an unknown thrown value. */
export function errorText(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
