/* ============================================================================
 * Module Registry
 *
 * The Buhera federation lives here. Each module conforms to the Module
 * trait (operations-architecture.md §5.1) and is dispatched by ID.
 *
 * A module exposes:
 *   id          : string identifier (e.g. "vahera", "purpose")
 *   execute     : (instruction, actBudget) => Promise<ActResult>
 *   outputCell  : (instruction) => OutputCell (for sufficiency checks)
 *   describe    : () => { id, description, instructions }
 *
 * Every act dispatched through this registry appends to the audit log.
 * ========================================================================== */

let _nextActId = 1;

// --------------------------------------------------------------------------
// Audit log (in-memory for v1; persistence is a later concern).
// --------------------------------------------------------------------------

const _auditLog = [];

export function getAuditLog() {
  return _auditLog.slice();
}

export function clearAuditLog() {
  _auditLog.length = 0;
}

// --------------------------------------------------------------------------
// Module registry.
// --------------------------------------------------------------------------

const _modules = new Map();

export function register(mod) {
  if (!mod || !mod.id || typeof mod.execute !== "function") {
    throw new Error("register: module must have id and execute()");
  }
  _modules.set(mod.id, mod);
}

export function unregister(moduleId) {
  _modules.delete(moduleId);
}

export function listModules() {
  return Array.from(_modules.values()).map((m) =>
    typeof m.describe === "function" ? m.describe() : { id: m.id }
  );
}

export function getModule(moduleId) {
  return _modules.get(moduleId) || null;
}

// --------------------------------------------------------------------------
// Dispatch: the one entry point. Per architecture doc §5.4 (in JS form).
// --------------------------------------------------------------------------

export async function dispatch(moduleId, instruction, actBudget = 1) {
  const mod = _modules.get(moduleId);
  if (!mod) {
    throw new Error(`dispatch: unknown module "${moduleId}"`);
  }

  const t0 = Date.now();
  let result;
  try {
    result = await mod.execute(instruction, actBudget);
  } catch (err) {
    result = {
      ok: false,
      output_delta: null,
      residue: 0,
      completed: true,
      error: err.message || String(err),
    };
  }

  const entry = {
    act_id: _nextActId++,
    module_id: moduleId,
    instruction,
    act_budget: actBudget,
    result,
    wall_clock_ms: Date.now() - t0,
    timestamp: new Date().toISOString(),
  };
  _auditLog.push(entry);

  return result;
}
