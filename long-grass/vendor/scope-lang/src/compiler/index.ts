// SCOPE Compiler — public API

import { Lexer } from './lexer';
import { Parser, ParseError } from './parser';
import { typeCheck, formatErrors, formatWarnings, CompileError, CompileWarning } from './type-checker';
import type { ScopeProgram } from './ast';

export type { ScopeProgram } from './ast';
export type { CompileError, CompileWarning } from './type-checker';
export { Lexer, TokenType } from './lexer';
export { Parser, ParseError } from './parser';
export { typeCheck, formatErrors, formatWarnings } from './type-checker';

export interface CompileResult {
  ok: boolean;
  program: ScopeProgram | null;
  errors: CompileError[];
  warnings: CompileWarning[];
  /** Human-readable console lines, formatted like the spec examples */
  log: string[];
}

/** @deprecated Use compile() */
export const compileScope = (source: string): CompileResult => compile(source);

export function compile(source: string): CompileResult {
  const log: string[] = [];

  // ── Stage 1: Lex + Parse ─────────────────────────────────────────────────
  let program: ScopeProgram;
  try {
    program = Parser.fromSource(source).parse();
  } catch (err) {
    if (err instanceof ParseError) {
      const e: CompileError = { kind: 'ParseError', message: err.message, line: err.line, col: err.col };
      const line = `[PARSE ERROR] line ${err.line} col ${err.col}\n  ${err.message}`;
      log.push(line);
      return { ok: false, program: null, errors: [e], warnings: [], log };
    }
    const msg = err instanceof Error ? err.message : String(err);
    const e: CompileError = { kind: 'ParseError', message: msg, line: 0, col: 0 };
    log.push(`[PARSE ERROR] ${msg}`);
    return { ok: false, program: null, errors: [e], warnings: [], log };
  }

  // ── Stage 2: Type Check ──────────────────────────────────────────────────
  const { errors, warnings, program: resolved } = typeCheck(program);

  for (const line of formatWarnings(warnings)) log.push(line);
  for (const line of formatErrors(errors)) log.push(line);

  if (errors.length > 0) {
    return { ok: false, program: resolved, errors, warnings, log };
  }

  // Emit COMPILE log line summarising the program
  const depth = resolved.coordinateSpace?.depth ?? resolved.morphisms[0]?.expr.observe.depth ?? '?';
  const cells = (resolved.channels?.items ?? []).filter(i => i.kind === 'CellItem');
  if (cells.length > 0) {
    for (const cell of cells) {
      const c = cell as any;
      log.push(`[COMPILE]  cell=${c.name}  bounds=(${c.boundsLow},${c.boundsHigh})  depth=${depth}`);
    }
  } else {
    log.push(`[COMPILE]  default  depth=${depth}  morphisms=${resolved.morphisms.length}`);
  }

  if (warnings.length > 0) {
    return { ok: true, program: resolved, errors: [], warnings, log };
  }

  return { ok: true, program: resolved, errors: [], warnings: [], log };
}
