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
import { cfcDsl, makeCfcModule, type CfcEngine } from "./cfc.ts";
import { hfqDsl, makeHfqModule, type HfqEngine } from "./hfq.ts";
import { honjoDsl, makeHonjoModule, type HonjoEngine } from "./honjo.ts";
import { makeLadderModule, type LadderEngine } from "./ladder.ts";
import { makePylonModule, pylonDsl, type PylonEngine } from "./pylon.ts";
import { GATEWAY_MODULES, makeRemoteModule, type GatewayTransport } from "./remote.ts";
import { makeWasmDsls, makeWasmModules } from "./rust-wasm.ts";
import { makeSbsModule, sbsDsl, type SbsEngine } from "./sbs.ts";
import { makeScopeModule, scopeDsl, type ScopeEngine } from "./scope.ts";
import { makeShapeshifterModule, type ShapeshifterEngine } from "./shapeshifter.ts";
import { makeSmithModule, smithDsl, type SmithEngine } from "./smith.ts";
import { makeSpectralModule, type SpectralEngine } from "./spectral.ts";
import { makeSthurbertModule, sthurbertDsl, type SthurbertEngine } from "./sthurbert.ts";
import { makeSynopsisModule, synopsisDsl, type SynopsisEngine } from "./synopsis.ts";
import { makeTempusModule, tempusDsl, type TempusEngine } from "./tempus.ts";
import { makeZangalewaModule, type InterceptorClientCtor } from "./zangalewa-dsl.ts";

export * from "./cfc.ts";
export * from "./hfq.ts";
export * from "./honjo.ts";
export * from "./ladder.ts";
export * from "./pylon.ts";
export * from "./remote.ts";
export * from "./rust-wasm.ts";
export * from "./sbs.ts";
export * from "./scope.ts";
export * from "./shapeshifter.ts";
export * from "./smith.ts";
export * from "./spectral.ts";
export * from "./sthurbert.ts";
export * from "./synopsis.ts";
export * from "./tempus.ts";
export * from "./zangalewa-dsl.ts";

export interface Engines {
  /** buhera-wasm: ndombolo, windtunnel, tracker. */
  wasm?: WasmEngine;
  sbs?: SbsEngine;
  hfq?: HfqEngine;
  pylon?: PylonEngine;
  tempus?: TempusEngine;
  smith?: SmithEngine;
  synopsis?: SynopsisEngine;
  cfc?: CfcEngine;
  sthurbert?: SthurbertEngine;
  honjo?: HonjoEngine;
  shapeshifter?: ShapeshifterEngine;
  ladder?: LadderEngine;
  spectral?: SpectralEngine;
  scope?: ScopeEngine;
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
  if (engines.smith) {
    registry.register(makeSmithModule(engines.smith));
    dsls.register(smithDsl(engines.smith));
  }
  if (engines.synopsis) {
    registry.register(makeSynopsisModule(engines.synopsis));
    dsls.register(synopsisDsl(engines.synopsis));
  }
  if (engines.cfc) {
    registry.register(makeCfcModule(engines.cfc));
    dsls.register(cfcDsl(engines.cfc));
  }
  if (engines.sthurbert) {
    registry.register(makeSthurbertModule(engines.sthurbert));
    dsls.register(sthurbertDsl(engines.sthurbert));
  }
  if (engines.honjo) {
    registry.register(makeHonjoModule(engines.honjo));
    dsls.register(honjoDsl(engines.honjo));
  }
  if (engines.shapeshifter) registry.register(makeShapeshifterModule(engines.shapeshifter));
  if (engines.ladder) registry.register(makeLadderModule(engines.ladder));
  if (engines.spectral) registry.register(makeSpectralModule(engines.spectral));
  if (engines.scope) {
    registry.register(makeScopeModule(engines.scope));
    dsls.register(scopeDsl(engines.scope));
  }
  if (engines.zangalewa) {
    registry.register(makeZangalewaModule(engines.zangalewa.Client, engines.zangalewa.baseUrl));
  }
  if (engines.gateway) {
    for (const d of GATEWAY_MODULES) registry.register(makeRemoteModule(d, engines.gateway));
  }
  return { registry, dsls };
}
