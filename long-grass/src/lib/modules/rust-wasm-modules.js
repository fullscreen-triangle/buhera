/* ============================================================================
 * Rust modules in the browser — ndombolo, windtunnel, tracker, executed by
 * the Rust adapters themselves compiled to wasm (buhera-os/crates/buhera-wasm;
 * specification 07 §4). Registered at mount, loaded on first use from
 * /wasm/buhera_modules.wasm (installed by scripts/build-wasm.mjs).
 * ========================================================================== */

import { loadWasmEngine } from "@buhera/registry";
import { makeLazyWasmModules } from "@buhera/registry/modules";

export const WASM_URL = "/wasm/buhera_modules.wasm";

async function fetchEngine() {
  const res = await fetch(WASM_URL);
  if (!res.ok) throw new Error(`GET ${WASM_URL}: HTTP ${res.status}`);
  return loadWasmEngine(await res.arrayBuffer());
}

export const rustWasmModules = makeLazyWasmModules(fetchEngine);
