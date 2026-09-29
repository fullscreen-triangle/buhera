# 09 — Conformance: what proves each claim

**Status:** normative · Results recorded 2026-09-29

## 1. How to run everything

```sh
# Rust: registry semantics, vendored engines' own suites, module behaviour, catalogue conformance, gateway
cd buhera-os
cargo test -p buhera-registry
cargo test -p buhera-modules --features full
cargo test --workspace

# wasm artifact
node scripts/build-wasm.mjs

# TypeScript library: registry semantics, catalogue conformance with real engines, module behaviour
cd buhera-os/registry-ts && npm test && npx tsc -p tsconfig.json

# long-grass host
node scripts/sync-registry-ts.mjs --check
node scripts/build-packs.mjs --check
cd long-grass && npm test && npx next build

# sourcing integrity
node scripts/vendor-sync.mjs --check
```

## 2. Results at this revision

| Suite | Result |
|---|---|
| `buhera-registry` (R1–R6, D1, RFC 3339) | 8 passed |
| vendored `ndombolo-core` (41 unit, 11 differential vs Python oracle) | 52 passed |
| vendored `wt-dsl` | 19 passed |
| vendored `hegel-sbs` | 6 passed |
| vendored `tracker-chi` | 2 passed |
| vendored `zangalewa-dsl` (with `ZANGALEWA_PACKS` → long-grass packs, set in `buhera-os/.cargo/config.toml`) | 20 passed |
| `buhera-modules --features full` (catalogue C1–C5 + module behaviour) | 11 passed |
| `buhera-wasm` (native) | 3 passed |
| `buhera-gateway` (36 existing + 4 `/api/dispatch`) | 40 passed |
| `cargo test --workspace` (default features; every crate above plus kernel, substrate, vahera, embed, os) | 222 passed, 0 failed |
| `registry-ts` (R1–R6, D1, catalogue C1–C5, 26 module cases incl. wasm, lazy wasm, remote, smith, the full synopsis corpus, the cfc and honjo example corpora, st-Hurbert) | 34 passed; `tsc` strict clean |
| long-grass `npm test` | 100 passed |
| long-grass `next build` | success |
| `vendor-sync --check` | 19 verified, 1 built, 0 failed |

## 3. Claim → proof

| Claim | Proof |
|---|---|
| Both registries implement identical semantics | the same seven cases in `buhera-registry/tests/registry_semantics.rs` and `registry-ts/test/registry-semantics.test.ts` |
| Each library federation is exactly the catalogue | `rust_federation_conforms_to_the_catalogue`, `the TS federation conforms to the catalogue` |
| Rust-in-TS is the same implementation | wasm cases assert the same numbers as the Rust cases (ndombolo 0.9; windtunnel incomplete 2/1/⅓; tracker 3 blocks / 2 fragments) |
| Adapters wrap real engines | tests use vendored engines only; `sbs-core` reproduces hegel's CLI numbers (R = 0.5919, V = 0.1175) |
| Vendored copies are the upstream engines | `vendor-sync --check` byte comparison at the recorded commit |
| Validators are the languages' own compilers | `library-federation.test.mjs`: every language accepts a valid and rejects a broken script through the long-grass facade |
| Packs ground generation in valid code | `knowledge-packs.test.mjs` validates every tagged pack example |
| Remote acts are audited on both hosts, and isolated per account | gateway `a_real_act_runs_and_is_audited_per_account`; TS `remote (gateway)` |
| Transport failures are results | `zangalewa-dsl: an unreachable broker…`, `remote … unreachable`, `lazy wasm … failed load` |
