# 10 — Findings

Reading ten upstream targets closely surfaced defects in them and in Buhera. This register records each one and what was done. **Fixed** means fixed in this repository in this revision. **Upstream** items were not changed: they belong to their repositories, and are recorded here as change requests. Nothing upstream was modified.

## 1. Fixed in Buhera in this revision

| # | Where | Defect | Fix |
|---|---|---|---|
| F1 | `long-grass/src/lib/server/exec-sandbox.js` | Model-generated programs inherited the server's full environment, **including API keys**. Verified: a generated program could print `OPENAI_API_KEY` | Children get an allowlisted environment (`childEnv`). Verified: the program now prints "no key" |
| F2 | `crates/interceptor-run` | Pipe deadlock: stdout/stderr were read only after exit, so any program writing more than about 64 KiB blocked until killed. It was reported `timed_out` with a silently clipped transcript | Pipes are drained on threads while the child runs. A 200 000-line program now completes in 1.65 s, keeping the first 2 MiB and flagging `truncated` |
| F3 | `crates/interceptor-run` | Work-dir suffix `Instant::now().elapsed()` was ~0, so directories were unique only per PID | Wall-clock nanoseconds |
| F4 | `long-grass/src/lib/wind-tunnel` | When every run crashed, identical crashes scored R = 1, "Phase-locked", residue 0 | No successful run means R = 0, "Turbulent", `all_failed: true`, and a failed run never counts as agreement. The misleading claim that its regimes follow wind-tunnel's was also corrected |
| F5 | `long-grass/knowledge-packs/vahera` + `test/dsl-validators` | The pack grounded generation with `S(0.2, 1.0, -0.5)`, which the parser rejects. The test asserted it was valid and was failing | In-range example, and the [0, 1] rule stated in the pack |
| F6 | `long-grass` `sbs` / `hfq` adapters | Residue was a size (`nodes + edges`, `steps.length`), not remaining work | `1 − V`; blocked steps |
| F7 | long-grass vendored SBS, HFQ, scheduler | No recorded provenance | `vendor.json` entries, byte-verified |
| F8 | `buhera-gateway` | The vaHera renderer was private, so a second path would have duplicated it | Moved to `buhera_vahera::render_result` and shared |

## 2. Upstream change requests

| # | Repo | Finding |
|---|---|---|
| U-tmp-1 | stella-lorraine | `crates/tempus` does not compile: `TempusError` lacks `Clone` (`typecheck.rs:61, :192`) |
| U-tmp-2 | stella-lorraine | Sequential composition never passes the carry forward, so multi-path trigger programs compute "last path fires" and not the OR (2 of 41 tests fail once U-tmp-1 is patched) |
| U-tmp-3 | stella-lorraine | η_C counts rejected events as accepted |
| U-pyl-1 | pylon | The TS runtime and the Rust `srn-node`/`srn-fleet` are not wire- or semantics-compatible; one source of truth is needed |
| U-pyl-2 | buhera/long-grass | `/api/srn` sends `{expression}` to srn-node `/eval` and calls `/probe` and `/gossip`. srn-node expects structured `EvalRequest` and serves `/network/probe` and `/network/gossip` |
| U-pyl-3 | pylon | `Cluster.fromSnapshot` can loop forever; `clearMarket` with no slots throws; capacity is not enforced |
| U-pyl-4 | pylon | ⚠ The local-only commit `ff22aaa` rotates a credential inside a **tracked** `.claude/settings.local.json`. Pushing it publishes the credential |
| U-sbs-1 | hegel | `perturb { edge: … }` is ignored; the TS compiler tree rejects `edge` as a prop key (4 of 7 shipped scripts) |
| U-sbs-2 | hegel | Parser errors carry `line: 0` |
| U-sbsc-1 | hegel | `sbs` (Rust) compiles ~185 crates for a library that uses two dependencies; make the rest optional |
| U-sbsc-2 | hegel | `partial_cmp().unwrap()` panics on NaN; the solver clock blocks wasm |
| U-hfq-1/2 | hegel | `runPlan` exposes neither a budget override nor the world registry (for a static-only check) |
| U-hfq-3 | hegel | `when starved emit partial` is parsed and ignored |
| U-wt-1 | wind-tunnel | `scope.include/exclude/language` and `cycles.through` are parsed and never read |
| U-wt-2 | wind-tunnel | `measure` lives in the binary; expose it from a library crate |
| U-wt-3 | wind-tunnel | `serve` opens a request-supplied traces path, contrary to its own comment |
| U-trk-1 | bloodhound | tracker is binary-only; split it into lib + bin |
| U-ndo-1 | kwasa-kwasa | `CellResult`/`CellError`/`StoreChange` are not `Serialize` |
| U-zng-1 | zangalewa | The Gemini key goes in the URL; reqwest errors include the URL, so they can leak it into `providerErrors` |
| U-zng-2 | zangalewa | Upstream's DSL registry has vaHera only; add entries for the other Buhera languages, or a `generate_with(validator)` |
| U-zng-3 | buhera/long-grass | `dsl-writer` reimplements the zangalewa loop in JS and collapses drafts to one; migrate to `zangalewa-dsl` |
| U-zng-4 | zangalewa | Five crates are manifest-only; `consciousness-core` fails to compile (20 errors) |

## 3. Naming collisions, recorded and not renamed

long-grass's host-local `zangalewa` (zoom-climb's coordinate extractor) and `interceptor` (an NL → program → run assistant) share names with upstream projects they do not wrap. They were not renamed, because tutorials depend on them. The upstream Zangalewa is the module `zangalewa-dsl`, and the upstream interceptor is its transport.
