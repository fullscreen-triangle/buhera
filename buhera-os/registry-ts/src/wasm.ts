/* ============================================================================
 * The buhera-wasm engine — the Rust modules that compile to wasm32
 * (ndombolo, windtunnel, tracker), loaded in-process.
 *
 * ABI (buhera-os/crates/buhera-wasm): every call takes one UTF-8 JSON
 * document written into memory obtained from bw_alloc, and returns one packed
 * u64 `(ptr << 32) | len` pointing at a UTF-8 JSON reply the caller frees with
 * bw_free. The Rust module executes the act; the TypeScript registry that
 * calls it is the one that audits (so each act is recorded exactly once).
 *
 * A panic in wasm32-unknown-unknown traps the instance. The engine then
 * re-instantiates from the same compiled module so the next act starts clean,
 * and the trapping call surfaces as a thrown error (contained by the registry,
 * rule R2).
 * ========================================================================== */

import type { ActResult, Descriptor, Json } from "./contract.ts";
import type { DslSummary, Validation } from "./dsl.ts";

interface Exports {
  memory: WebAssembly.Memory;
  bw_alloc(len: number): number;
  bw_free(ptr: number, len: number): void;
  bw_describe(): bigint;
  bw_dispatch(ptr: number, len: number): bigint;
  bw_validate(ptr: number, len: number): bigint;
}

export interface WasmEngine {
  describe(): { modules: Descriptor[]; dsls: Array<{ id: string; label: string; extension: string; module_id: string; pack_id: string }> };
  dispatch(module: string, instruction: Json, actBudget: number): ActResult;
  validate(dsl: string, source: string): Validation;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

/** Compile (async) and instantiate the engine — the browser path. */
export async function loadWasmEngine(source: BufferSource | WebAssembly.Module): Promise<WasmEngine> {
  const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
  return instantiateWasmEngine(module);
}

/** Synchronous load from bytes — for Node hosts (API routes, validators) that
 *  need the engine inside a synchronous call. */
export function loadWasmEngineSync(bytes: BufferSource): WasmEngine {
  return instantiateWasmEngine(new WebAssembly.Module(bytes));
}

export function instantiateWasmEngine(module: WebAssembly.Module): WasmEngine {
  let ex = new WebAssembly.Instance(module, {}).exports as unknown as Exports;

  const call = (fn: (p: number, l: number) => bigint, input: unknown): unknown => {
    const bytes = enc.encode(JSON.stringify(input));
    const ptr = ex.bw_alloc(bytes.length);
    new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
    let packed: bigint;
    try {
      packed = fn(ptr, bytes.length);
    } catch (err) {
      // Trapped: the instance is poisoned; replace it before rethrowing.
      ex = new WebAssembly.Instance(module, {}).exports as unknown as Exports;
      throw err;
    }
    ex.bw_free(ptr, bytes.length);
    return read(packed);
  };

  const read = (packed: bigint): unknown => {
    const outPtr = Number(packed >> 32n);
    const outLen = Number(packed & 0xffffffffn);
    const text = dec.decode(new Uint8Array(ex.memory.buffer, outPtr, outLen));
    ex.bw_free(outPtr, outLen);
    return JSON.parse(text);
  };

  return {
    describe: () => read(ex.bw_describe()) as ReturnType<WasmEngine["describe"]>,
    dispatch(moduleId, instruction, actBudget) {
      const r = call((p, l) => ex.bw_dispatch(p, l), { module: moduleId, instruction, act_budget: actBudget }) as
        | ActResult
        | { error: string };
      if ("error" in r && !("ok" in r)) throw new Error(r.error);
      return r as ActResult;
    },
    validate(dsl, src) {
      const r = call((p, l) => ex.bw_validate(p, l), { dsl, source: src }) as Validation | { error: string };
      if ("error" in r && !("ok" in r)) throw new Error(r.error);
      return r as Validation;
    },
  };
}

/** Normalise the Rust DSL summary (snake_case) to the TS DslSummary shape. */
export function wasmDslSummaries(engine: WasmEngine): DslSummary[] {
  return engine.describe().dsls.map((d) => ({
    id: d.id,
    label: d.label,
    extension: d.extension,
    moduleId: d.module_id,
    packId: d.pack_id,
  }));
}
