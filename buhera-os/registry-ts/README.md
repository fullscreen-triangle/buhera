# @buhera/registry

The TypeScript twin of the Rust crates `buhera-registry` and `buhera-modules`. The normative text is in `specifications/architecture/02`–`07` and `specifications/specs/*.md`.

| Export | What |
|---|---|
| `@buhera/registry` | `Module`, `ActResult`, `Descriptor` (the contract); `Registry` (dispatch, audit log, hooks; rules R1–R6); `DslRegistry`; `conformance(catalogue, "ts", …)`; the wasm loader |
| `@buhera/registry/modules` | Adapter factories (`makeSbsModule`, `makeHfqModule`, `makePylonModule`, `makeTempusModule`, `makeZangalewaModule`, `makeWasmModules`, `makeLazyWasmModules`, `makeRemoteModule`) and `createFederation(engines)` |

Engines are **injected**, never imported. The host owns the vendored engine copies (long-grass keeps them in `long-grass/vendor/`), so this package never pulls an engine's code or dependencies.

```ts
import { createFederation } from "@buhera/registry/modules";
const { registry, dsls } = createFederation({ sbs, hfq, pylon, tempus, wasm, zangalewa, gateway });
await registry.dispatch("hfq", { kind: "preset", id: "mark_q1" });
dsls.validate("tempus", source);
```

`wasm/buhera_modules.wasm` holds the Rust modules ndombolo, windtunnel and tracker, adapters included. It is built by `node scripts/build-wasm.mjs`.

The source is erasable TypeScript: Node 22.18+ runs it directly, and bundlers transpile it.

```sh
npm test                 # registry semantics, catalogue conformance with the real engines, module behaviour
npx tsc -p tsconfig.json # strict typecheck
```

long-grass consumes a mirror at `long-grass/vendor/registry`, written by `node scripts/sync-registry-ts.mjs`; `--check` fails on drift.
