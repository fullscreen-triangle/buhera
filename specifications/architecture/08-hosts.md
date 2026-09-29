# 08 — Hosts

**Status:** normative

A *host* is a process that owns one registry. There are four.

## 1. long-grass (TypeScript; browser + Next API routes)

- **Registry:** `src/lib/modules/registry.js`, a facade over one `@buhera/registry` `Registry`. It keeps the historical free functions (`register`, `dispatch`, `onDispatch`, `getAuditLog`, …), so all pre-existing modules and pages work unchanged.
- **Bootstrap:** `src/lib/runtime/bootstrap.js` registers, in order:
  1. the host-local modules: vahera, echo, lavoisier, purpose, zangalewa (coordinate extractor), graffiti, desk, dsl-writer, srn, ckg, cytochrome, gateway, triangle, spraypaint, interceptor, and others;
  2. the library federation.
- **The library federation in long-grass:**

| module | file | binding |
|---|---|---|
| `sbs` | `sbs-module.js` | native, `makeSbsModule(@sachikonye/sbs)` |
| `hfq` | `hfq-module.js` | native, `makeHfqModule(@hegel/hfq)` |
| `pylon` | `pylon-module.js` | native, `makePylonModule(@buhera/pylon)` |
| `tempus` | `tempus-module.js` | native, `makeTempusModule(@stella-lorraine/tempus)` |
| `smith` | `smith-module.js` | native, `makeSmithModule(@musande/agent-smith)` |
| `synopsis` | `synopsis-module.js` | native, `makeSynopsisModule(@gospel/synopsis)` |
| `cfc` | `cfc-module.js` | native, `makeCfcModule(@syndrome/cfc)` |
| `sthurbert` | `sthurbert-module.js` | native, `makeSthurbertModule(@bloodhound/sthurbert)` |
| `honjo` | `honjo-module.js` | native, `makeHonjoModule(@borgia/honjo)` |
| `shapeshifter` | `shapeshifter-module.js` | native, `makeShapeshifterModule(@lavoisier/shapeshifter)` |
| `ladder` | `ladder-module.js` | native, `makeLadderModule(@levinthal/ladder)` |
| `spectral` | `spectral-module.js` | native, `makeSpectralModule(@gospel/spectral)` |
| `scope` | `scope-module.js` | native, `makeScopeModule(scope-lang)`; the terminal links images through `linkScopeImage` |
| `zangalewa-dsl` | `zangalewa-dsl-module.js` | remote (broker; `NEXT_PUBLIC_ZANGALEWA_BROKER`) |
| `ndombolo`, `windtunnel`, `tracker`, `heihachi`, `olduvai`, `levinthal`, `mekaneck` | `rust-wasm-modules.js` | native (wasm, lazy from `/wasm/buhera_modules.wasm`) |
| `sbs-core` | `gateway-remote-modules.js` | remote (gateway session from `gateway-module.js`) |

- **DSL registry:** `src/lib/purpose/dsl/validators.js`, a facade over `DslRegistry` that carries all seven catalogue languages with their real front ends. It is server-only: it loads the wasm validators synchronously from `public/wasm`.
- **Knowledge packs:** `knowledge-packs/<id>/`. Six are generated from the specifications (`scripts/build-packs.mjs`); vaHera's is hand-written. Every tagged example is validated by `test/knowledge-packs.test.mjs`.
- **Next config:** `transpilePackages` includes the three TypeScript-source vendored packages. `next build` passes.

## 2. buhera-gateway (Rust; HTTPS)

- `POST /api/dispatch` and `GET /api/modules` (spec 07 §2).
- A per-account `Registry` from `buhera_modules::federation` with `filesystem: false`, covering vahera, ndombolo, windtunnel, tracker and sbs-core.
- The pre-existing `/api/run` is unchanged, except that it now renders through the shared `buhera_vahera::render_result`.

## 3. Host infrastructure that is not a module: the scheduler

stella-lorraine's residue-driven scheduler lives at `long-grass/src/lib/scheduler`. Its provenance is recorded in `vendor.json` and it is byte-identical to upstream. It is an orchestrator: `P = descent / max(residue − threshold, floor)`, which feeds converging tasks, finishes near ones and starves stalled ones. It reaches modules through an `OsPort`.

Nothing instantiates it yet. When a host wires it, the port MUST map the contract as follows:

| OsPort `ActResult` | from the module ActResult |
|---|---|
| `residue` | `residue`, whose meaning each module spec declares |
| `completed` | `completed`, or `true` when `ok` is false (a failed act terminates the task rather than stalling it) |
| `cost` | the act budget spent |

The scheduler re-sends the same `nextInstruction` every act. Only modules whose acts are resumable (pylon `step`, tempus `simulate`) make progress under it.

## 4. The buhera-os binaries and the wasm module

`repl` and `demo` are unchanged; they drive the kernel directly. `buhera-wasm` is a host in its own right: it owns a registry of the three wasm-safe modules, registered explicitly, and executes without auditing (spec 07, W1).
