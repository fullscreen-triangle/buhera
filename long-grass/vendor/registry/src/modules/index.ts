/* ============================================================================
 * @buhera/registry/modules — the TypeScript federation.
 *
 * createFederation(engines) registers one module (and its language, if it
 * has one) for every engine the host supplies. Engines are injected, never
 * imported here: the host owns the vendored engine copies (long-grass/vendor)
 * and this library stays engine-free, exactly like the Rust buhera-registry.
 *
 * With every engine supplied the result must pass catalogue conformance
 * (test/conformance.test.ts, specification 05).
 * ========================================================================== */

import { DslRegistry } from "../dsl.ts";
import { Registry } from "../registry.ts";
import type { WasmEngine } from "../wasm.ts";
import { hfqDsl, makeHfqModule, type HfqEngine } from "./hfq.ts";
import { makePylonModule, pylonDsl, type PylonEngine } from "./pylon.ts";
import { GATEWAY_MODULES, makeRemoteModule, type GatewayTransport } from "./remote.ts";
import { makeWasmDsls, makeWasmModules } from "./rust-wasm.ts";
import { makeSbsModule, sbsDsl, type SbsEngine } from "./sbs.ts";
import { makeTempusModule, tempusDsl, type TempusEngine } from "./tempus.ts";
import { makeZangalewaModule, type InterceptorClientCtor } from "./zangalewa-dsl.ts";

export * from "./hfq.ts";
export * from "./pylon.ts";
export * from "./remote.ts";
export * from "./rust-wasm.ts";
export * from "./sbs.ts";
export * from "./tempus.ts";
export * from "./zangalewa-dsl.ts";

export interface Engines {
  /** buhera-wasm: ndombolo, windtunnel, tracker. */
  wasm?: WasmEngine;
  sbs?: SbsEngine;
  hfq?: HfqEngine;
  pylon?: PylonEngine;
  tempus?: TempusEngine;
  /** The vendored interceptor client class, plus the broker URL. */
  zangalewa?: { Client: InterceptorClientCtor; baseUrl?: string };
  /** buhera-gateway, for the Rust-only modules reached remotely. */
  gateway?: GatewayTransport;
}

export interface Federation {
  registry: Registry;
  dsls: DslRegistry;
}

export function createFederation(engines: Engines, into?: Federation): Federation {
  const registry = into?.registry ?? new Registry();
  const dsls = into?.dsls ?? new DslRegistry();
  if (engines.wasm) {
    for (const m of makeWasmModules(engines.wasm)) registry.register(m);
    for (const d of makeWasmDsls(engines.wasm)) dsls.register(d);
  }
  if (engines.sbs) {
    registry.register(makeSbsModule(engines.sbs));
    dsls.register(sbsDsl(engines.sbs));
  }
  if (engines.hfq) {
    registry.register(makeHfqModule(engines.hfq));
    dsls.register(hfqDsl(engines.hfq));
  }
  if (engines.pylon) {
    registry.register(makePylonModule(engines.pylon));
    dsls.register(pylonDsl(engines.pylon));
  }
  if (engines.tempus) {
    registry.register(makeTempusModule(engines.tempus));
    dsls.register(tempusDsl(engines.tempus));
  }
  if (engines.zangalewa) {
    registry.register(makeZangalewaModule(engines.zangalewa.Client, engines.zangalewa.baseUrl));
  }
  if (engines.gateway) {
    for (const d of GATEWAY_MODULES) registry.register(makeRemoteModule(d, engines.gateway));
  }
  return { registry, dsls };
}
