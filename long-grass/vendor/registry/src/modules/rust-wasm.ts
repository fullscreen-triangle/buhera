/* ============================================================================
 * Rust modules hosted in-process through buhera-wasm: ndombolo, windtunnel,
 * tracker, heihachi, olduvai, levinthal, mekaneck (specifications specs/<id>.md).
 *
 * These adapters contain no module logic at all: the descriptor, the act and
 * the language validator are the Rust module's own, executed in wasm. That
 * makes the TypeScript and Rust hosts contract-equivalent by construction
 * (specification 02 §4) — there is one implementation, compiled twice.
 * ========================================================================== */

import type { Module } from "../contract.ts";
import type { DslEntry } from "../dsl.ts";
import type { WasmEngine } from "../wasm.ts";
import { wasmDslSummaries } from "../wasm.ts";

export const WASM_MODULE_IDS = ["heihachi", "levinthal", "mekaneck", "ndombolo", "olduvai", "tracker", "windtunnel"] as const;

export function makeWasmModules(engine: WasmEngine): Module[] {
  return engine.describe().modules.map((d) => ({
    id: d.id,
    describe: () => d,
    execute: (instruction, actBudget) => engine.dispatch(d.id, instruction, actBudget),
    outputCell: () => ({ kind: `${d.id}_cell` }),
  }));
}

export function makeWasmDsls(engine: WasmEngine): DslEntry[] {
  return wasmDslSummaries(engine).map((s) => ({ ...s, validate: (src: string) => engine.validate(s.id, src) }));
}

/** The language each wasm module executes (null = none), for lazy registration. */
const WASM_DSL: Record<(typeof WASM_MODULE_IDS)[number], string | null> = {
  heihachi: "mishima",
  levinthal: null,
  mekaneck: "mekaneck",
  ndombolo: "turbulance",
  olduvai: null,
  tracker: null,
  windtunnel: "wt",
};

/**
 * Wasm modules that register synchronously and load the engine on first use
 * (a browser host should not fetch ~640 KB at page mount). Until the engine
 * has loaded, `describe()` says so plainly; afterwards it is the Rust
 * module's own descriptor. A failed load is an act result, never a throw.
 */
export function makeLazyWasmModules(load: () => Promise<WasmEngine>): Module[] {
  let engine: Promise<WasmEngine> | null = null;
  let loaded: WasmEngine | null = null;
  const get = () =>
    (engine ??= load().then(
      (e) => (loaded = e),
      (err) => {
        engine = null; // allow a retry on the next act
        throw err;
      },
    ));
  return WASM_MODULE_IDS.map((id) => {
    const dsl = WASM_DSL[id];
    return {
      id,
      describe: () =>
        loaded?.describe().modules.find((d) => d.id === id) ?? {
          id,
          description: `${id} — Rust module (wasm); engine loads on first use`,
          instructions: [`dispatch("${id}", "demo")`],
          ...(dsl ? { dsl } : {}),
          binding: "native",
        },
      async execute(instruction, actBudget) {
        let e: WasmEngine;
        try {
          e = await get();
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          return { ok: false, output_delta: { kind: "text", lines: [`${id}: wasm engine failed to load — ${msg}`] }, residue: 0, completed: true, error: "engine unavailable" };
        }
        return e.dispatch(id, instruction, actBudget);
      },
      outputCell: () => ({ kind: `${id}_cell` }),
    } satisfies Module;
  });
}
