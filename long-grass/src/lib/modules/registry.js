/* ============================================================================
 * Module Registry — long-grass facade over @buhera/registry.
 *
 * The Buhera federation lives here. The semantics (dispatch, audit log,
 * post-dispatch hooks, containment of throwing modules) are specified in
 * specifications/architecture/03-registry.md and implemented once, in the
 * TypeScript registry library (buhera-os/registry-ts, vendored at
 * vendor/registry) — the twin of the Rust buhera-registry crate.
 *
 * This file keeps the historical free-function API so every module and page
 * keeps working unchanged; each function delegates to one process-lifetime
 * Registry instance. A module exposes:
 *   id          : string identifier (e.g. "vahera", "sbs")
 *   execute     : (instruction, actBudget) => ActResult | Promise<ActResult>
 *   outputCell  : (instruction) => OutputCell (for sufficiency checks)
 *   describe    : () => { id, description, instructions, dsl?, binding? }
 * ========================================================================== */

import { Registry } from "@buhera/registry";

const _registry = new Registry();

/** The underlying Registry (for hosts that need the typed API). */
export function getRegistry() {
  return _registry;
}

export function getAuditLog() {
  return _registry.auditLog();
}

export function clearAuditLog() {
  _registry.clearAuditLog();
}

/** Bind a module under its id, replacing any previous binding (R5). */
export function register(mod) {
  return _registry.register(mod);
}

export function unregister(moduleId) {
  _registry.unregister(moduleId);
}

export function listModules() {
  return _registry.list();
}

export function getModule(moduleId) {
  return _registry.get(moduleId);
}

/**
 * Register a hook that runs after every dispatched act (R4). The hook
 * receives the audit-log entry. Returns an unregister function.
 */
export function onDispatch(hook) {
  return _registry.onDispatch(hook);
}

export function clearDispatchHooks() {
  _registry.clearHooks();
}

/**
 * Dispatch one act (R1–R3). Throws for an unknown module id; a module that
 * throws is contained and audited as { ok:false, output_delta:null, … }.
 */
export async function dispatch(moduleId, instruction, actBudget = 1) {
  return _registry.dispatch(moduleId, instruction, actBudget);
}
