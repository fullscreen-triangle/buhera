// scope-lang — SCOPE DSL compiler + runtime, packaged for embedding.
//
// This is the public surface consumed by host applications (e.g. the Buhera OS
// `scope` module). It re-exports the compiler front end, the whole-program
// runtime, and the shared numeric engine.

// ── Compiler ────────────────────────────────────────────────────────────────
export { compile, compileScope, typeCheck, formatErrors, formatWarnings } from './compiler';
export type { CompileResult, CompileError, CompileWarning, ScopeProgram } from './compiler';

// ── Runtime (whole-program execution) ────────────────────────────────────────
export { runScope } from './runtime/runtime';
export type { ImagePayload, ScopeResult } from './runtime/runtime';

// ── REPL session (incremental, cell-at-a-time execution) ─────────────────────
export { ScopeSession, createSession } from './session';
export type { CellResult, SessionState } from './session';

// ── Numeric engine (mic-engine) ──────────────────────────────────────────────
// Exposed so hosts can render mic-engine-style fields directly if needed.
export {
  estimateScaleField,
  fastMarchingDistance,
  computeEntropyMetrics,
  segmentImage,
} from './mic-engine';
