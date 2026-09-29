/**
 * st-Hurbert compiler facade: source → tokens → AST → (validate) → run.
 *
 * `compile` validates without executing (the ground-truth check an AI-generated
 * program must pass before it runs — empty-dictionary principle: a real compiler
 * judges the code, nothing is memorised). `run` compiles then interprets.
 */

import { lex, LexError } from "./lexer";
import { parse, ParseError } from "./parser";
import { interpret, RunResult, RuntimeError } from "./interpreter";
import type { Program } from "./ast";
import type { Federation } from "../model";

export interface CompileError {
  stage: "lex" | "parse" | "runtime";
  message: string;
  line?: number;
  col?: number;
}

export interface CompileResult {
  ok: boolean;
  program?: Program;
  error?: CompileError;
}

/** Validate st-Hurbert source into an AST without executing it. */
export function compile(source: string): CompileResult {
  try {
    const tokens = lex(source);
    const program = parse(tokens);
    return { ok: true, program };
  } catch (e) {
    if (e instanceof LexError) return { ok: false, error: { stage: "lex", message: e.message, line: e.line, col: e.col } };
    if (e instanceof ParseError) return { ok: false, error: { stage: "parse", message: e.message, line: e.line, col: e.col } };
    return { ok: false, error: { stage: "parse", message: (e as Error).message } };
  }
}

export interface RunOutcome {
  ok: boolean;
  result?: RunResult;
  error?: CompileError;
}

/** Compile and interpret st-Hurbert source against an analysed federation. */
export function run(source: string, fed: Federation): RunOutcome {
  const compiled = compile(source);
  if (!compiled.ok) return { ok: false, error: compiled.error };
  try {
    const result = interpret(compiled.program!, fed);
    return { ok: true, result };
  } catch (e) {
    if (e instanceof RuntimeError) return { ok: false, error: { stage: "runtime", message: e.message } };
    return { ok: false, error: { stage: "runtime", message: (e as Error).message } };
  }
}

export type { RunResult } from "./interpreter";
export type { Program } from "./ast";
export { LexError } from "./lexer";
export { ParseError } from "./parser";
export { RuntimeError } from "./interpreter";
