/* ============================================================================
 * The module registry — specification 03-registry.md.
 *
 * Twin of buhera-registry::Registry. Semantics:
 *   R1 unknown module     → dispatch throws (caller error; not audited)
 *   R2 module throws      → contained: {ok:false, output_delta:null, residue:0,
 *                           completed:true, error}
 *   R3 act ids            → from 1, +1 per dispatch, never reused
 *   R4 hooks              → after the audit append, in order, each isolated
 *   R5 register           → replaces, returns the previous binding (or null)
 *   R6 module state       → lives in the module value, between acts
 * ========================================================================== */

import type { ActResult, Descriptor, Instruction, Module } from "./contract.ts";
import { errorText } from "./contract.ts";

export interface AuditEntry {
  act_id: number;
  module_id: string;
  instruction: Instruction;
  act_budget: number;
  result: ActResult;
  wall_clock_ms: number;
  timestamp: string;
}

export type DispatchHook = (entry: AuditEntry) => void;

export class UnknownModuleError extends Error {
  readonly moduleId: string;
  constructor(moduleId: string) {
    super(`dispatch: unknown module "${moduleId}"`);
    this.name = "UnknownModuleError";
    this.moduleId = moduleId;
  }
}

export class Registry {
  #modules = new Map<string, Module>();
  #audit: AuditEntry[] = [];
  #hooks: DispatchHook[] = [];
  #nextAct = 1;

  register(mod: Module): Module | null {
    if (!mod || typeof mod.id !== "string" || !mod.id || typeof mod.execute !== "function") {
      throw new Error("register: module must have id and execute()");
    }
    const prev = this.#modules.get(mod.id) ?? null;
    this.#modules.set(mod.id, mod);
    return prev;
  }

  unregister(moduleId: string): Module | null {
    const prev = this.#modules.get(moduleId) ?? null;
    this.#modules.delete(moduleId);
    return prev;
  }

  ids(): string[] {
    return [...this.#modules.keys()].sort();
  }

  list(): Descriptor[] {
    return this.ids().map((id) => {
      const m = this.#modules.get(id) as Module;
      return typeof m.describe === "function"
        ? m.describe()
        : { id, description: "", instructions: [], binding: "native" };
    });
  }

  get(moduleId: string): Module | null {
    return this.#modules.get(moduleId) ?? null;
  }

  has(moduleId: string): boolean {
    return this.#modules.has(moduleId);
  }

  async dispatch(moduleId: string, instruction: Instruction, actBudget = 1): Promise<ActResult> {
    const mod = this.#modules.get(moduleId);
    if (!mod) throw new UnknownModuleError(moduleId);

    const t0 = Date.now();
    let result: ActResult;
    try {
      result = await mod.execute(instruction, actBudget);
    } catch (err) {
      result = { ok: false, output_delta: null, residue: 0, completed: true, error: errorText(err) };
    }

    const entry: AuditEntry = {
      act_id: this.#nextAct++,
      module_id: moduleId,
      instruction,
      act_budget: actBudget,
      result,
      wall_clock_ms: Date.now() - t0,
      timestamp: new Date().toISOString(),
    };
    this.#audit.push(entry);

    for (const hook of [...this.#hooks]) {
      try {
        hook(entry);
      } catch (err) {
        // Best-effort: a failing hook never breaks the dispatch.
        console.warn("registry: post-dispatch hook failed", err);
      }
    }
    return result;
  }

  auditLog(): AuditEntry[] {
    return this.#audit.slice();
  }

  clearAuditLog(): void {
    this.#audit.length = 0;
  }

  /** Register a post-dispatch hook; returns its unregister function. */
  onDispatch(hook: DispatchHook): () => void {
    if (typeof hook !== "function") throw new Error("onDispatch: hook must be a function");
    this.#hooks.push(hook);
    return () => {
      const i = this.#hooks.indexOf(hook);
      if (i >= 0) this.#hooks.splice(i, 1);
    };
  }

  clearHooks(): void {
    this.#hooks.length = 0;
  }
}
