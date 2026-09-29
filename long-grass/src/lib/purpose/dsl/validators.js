/**
 * The DSL registry — long-grass facade over @buhera/registry's DslRegistry
 * (specification specifications/architecture/04-dsl-registry.md).
 *
 * Every language a Buhera module owns is registered here with its REAL
 * front end, normalised to one contract:
 *
 *     validate(source) -> { ok: boolean, errors: [{ message, line?, column? }] }
 *
 * The validator is the empty-dictionary ground truth: purpose ships no DSL
 * facts, it only proposes code and lets the DSL's own compiler judge it. A
 * generated script is "valid" iff the module's own compiler accepts it.
 *
 * Each entry also records which module executes the code (moduleId, for
 * dispatch) and which knowledge pack grounds generation (packId).
 *
 * Server-side only (dsl-generator, /api/dsl-generate): the Turbulance, .wt,
 * mishima and sangoma validators are the Rust front ends compiled to wasm, loaded
 * synchronously from public/wasm on first use.
 */

import fs from "fs";
import path from "path";
import { DslRegistry, fromThrowing, loadWasmEngineSync } from "@buhera/registry";
import { cfcDsl, hfqDsl, honjoDsl, pylonDsl, sbsDsl, smithDsl, sthurbertDsl, synopsisDsl, tempusDsl } from "@buhera/registry/modules";
import * as sbsEngine from "@sachikonye/sbs";
import * as pylonEngine from "@buhera/pylon";
import { parseVahera } from "@/lib/vahera";
import { hfqEngine } from "@/lib/modules/hfq-module";
import { tempusEngine } from "@/lib/modules/tempus-module";
import { smithEngine } from "@/lib/modules/smith-module";
import { synopsisEngine } from "@/lib/modules/synopsis-module";
import { cfcEngine } from "@/lib/modules/cfc-module";
import { sthurbertEngine } from "@/lib/modules/sthurbert-module";
import { honjoEngine } from "@/lib/modules/honjo-module";

/**
 * vaHera — `parseVahera(src)` THROWS on the first invalid line, embedding
 * "line N:" in the message. Parsing is pure (no kernel needed), so it is a
 * safe pre-flight check before we hand the source to dispatch("vahera").
 */
export const validateVahera = fromThrowing(parseVahera);

const registry = new DslRegistry();
registry.register({ id: "vahera", label: "vaHera", extension: ".vhr", moduleId: "vahera", packId: "vahera", validate: validateVahera });
registry.register(sbsDsl(sbsEngine));
registry.register(hfqDsl(hfqEngine));
registry.register(pylonDsl(pylonEngine));
registry.register(tempusDsl(tempusEngine));
registry.register(smithDsl(smithEngine));
registry.register(synopsisDsl(synopsisEngine));
registry.register(cfcDsl(cfcEngine));
registry.register(sthurbertDsl(sthurbertEngine));
registry.register(honjoDsl(honjoEngine));

// Rust front ends via wasm: registered now, engine loaded on first validate.
let _wasm = null;
function wasmEngine() {
  if (!_wasm) {
    const file = path.join(process.cwd(), "public", "wasm", "buhera_modules.wasm");
    _wasm = loadWasmEngineSync(fs.readFileSync(file));
  }
  return _wasm;
}
registry.register({
  id: "turbulance", label: "Turbulance (ndombolo)", extension: ".tb", moduleId: "ndombolo", packId: "turbulance",
  validate: (src) => wasmEngine().validate("turbulance", src),
});
registry.register({
  id: "wt", label: "Wind Tunnel (.wt)", extension: ".wt", moduleId: "windtunnel", packId: "wt",
  validate: (src) => wasmEngine().validate("wt", src),
});
registry.register({
  id: "mishima", label: "mishima", extension: ".mma", moduleId: "heihachi", packId: "mishima",
  validate: (src) => wasmEngine().validate("mishima", src),
});
registry.register({
  id: "sangoma", label: "sangoma", extension: ".sgn", moduleId: "heihachi", packId: "sangoma",
  validate: (src) => wasmEngine().validate("sangoma", src),
});

/** The registry instance (typed API). */
export function getDslRegistry() {
  return registry;
}

/** List the DSL ids this generator can currently target. */
export function listDsls() {
  return registry.ids();
}

/** Look up a DSL entry, or null if unknown. */
export function getDsl(dslId) {
  return registry.get(dslId);
}

/**
 * Validate `source` against the named DSL's real compiler. Throws if the
 * DSL id is unknown (a programming error, distinct from invalid source
 * which returns {ok:false}).
 */
export function validate(dslId, source) {
  return registry.validate(dslId, source);
}
