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
| F9 | `long-grass/src/lib/smith` | long-grass ran a port of musande's `smith-ide` **stub** compiler (regex-based, no typechecker, `Math.random` in its output); its own test programs are rejected by the real Agent Smith parser | Removed; the `smith` module wraps the canonical `web/src/lib/agent-smith` (parser + typechecker + town), vendored byte-exact |
| F10 | `long-grass` `smith` adapter | Residue was the sum of realised floors (a size); runs enabled models by default | Residue is the shared residual still above the reachable floor; every run passes `useModel: false` |
| F11 | `long-grass/src/lib/sandboxes/honjo/honjo.js` | A stale copy of the honjo bundle (pre-`shell.ts`: Z ≤ 18, `3P0` term notation) backed the honjo sandbox | Removed; the sandbox imports the vendored `@borgia/honjo` |
| F12 | `long-grass` `shapeshifter` adapter | A global runtime failure (the interpreter swallows it into an empty result) was reported `ok: true`; residue was `workspace.length` (output, not remaining work) | `ok: false` on the interpreter's `error: runtime` line; residue = warnings + unresolved pending lookups; timing lines moved out of the terminal stream |
| F13 | `long-grass` `ladder` adapter | `climb` re-implemented the subfloor check with a different result shape, built an unrelated chain graph to satisfy the `Machine`, and reported residue 0 after commitments | every verdict from `Machine.runVerdict`; optional real graph, otherwise null residues; residue = distance still to the target; `derive` capped at 20 items |
| F14 | `long-grass/vendor/shapeshifter`, `vendor/ladder` | No recorded provenance; shapeshifter was a stale snapshot (`162ae1b`) | `vendor.json` entries; shapeshifter upgraded to `dacc197` (additive: `experiment.js`, nine operations); ladder recorded at origin/main |
| F15 | `long-grass` `scope` adapter; `test/resolver.mjs` | Residue was `sEntropy.sum`, which the engine normalises to 1 on every run; the session was a module-level global; the Node test resolver could not follow directory imports to `index.ts`, so SCOPE had never been tested outside the browser | residue = declared goals not met; per-instance session with host handles; the resolver resolves `index.ts`, and SCOPE runs in both test suites |

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
| U-zng-5 | buhera/long-grass | `src/lib/zangalewa/` holds byte copies of zoom-climb's `prompt.ts`, `coord-extract.ts` and `desk/*` that nothing imports |
| U-zng-6 | zangalewa + buhera | vaHera validity differs: long-grass rejects S-coordinates outside [0, 1] (without a line number), zoom-climb's TS parser and the Rust `zangalewa-dsl` accept them, and zoom-climb's pack teaches `S(0.2, 1.0, -0.5)`; the three must agree |
| U-smi-1 | musande | `crates/agent-smith` computes the realised floor as a singleton cut: on the path a–9–b–1–c–9–d it reports 9, the JS (and the definition) give 1 |
| U-smi-2 | musande | The Rust `CONVEX_POTENTIALS` list is short of the JS registry; 8 of the 10 tutorials fail to typecheck in Rust |
| U-smi-3 | musande | Rust residuals live in a `HashMap`, so trace order is nondeterministic natively |
| U-smi-4 | musande | `defaultCtx()` defaults `useModel: true`, and the model transport POSTs the user's provider keys to an app route; a library default should be models-off |
| U-smi-5 | musande | `smith-ide/src/compiler` is a self-declared stub that drops society members' `self` and `budget`; retire it or point the IDE at `web/src/lib/agent-smith` |
| U-syn-1 | gospel | `synopsis/rs` parses but does not check, so it cannot be bound as `synopsis` (it would accept programs the TS checker refuses) |
| U-syn-2 | gospel | Some truncated inputs raise a JS `TypeError` from inside the parser rather than a `ParseError` |
| U-cfc-1 | syndrome | `webtool` `npm test` runs zero tests: the script globs `test/*.test.mjs`, the files are `test/*.mjs` |
| U-cfc-2 | syndrome | `measure`, `localize`, `close` and gap closure are lexed but not implemented; `import` binds a placeholder that returns `<name.attr>` strings |
| U-sth-1 | bloodhound | `thrust/src/lib/repo-lens/chi.ts` is a TS port of `thrust/tracker/src/chi.rs`; two χ implementations can drift, and their field names already differ |
| U-sth-2 | bloodhound | The st-Hurbert number lexer accepts `1.2.3`; identifiers cannot start with a digit, so such repositories cannot be navigated to; `KEYWORDS` is exported and never enforced |
| U-hjo-1 | borgia | The committed bundle `src/lib/honjo.js` is ahead of `honjo/src/stdlib.ts` (a `shells` field in `individuate`); rebuild or reconcile |
| U-hjo-2 | borgia | `honjo-rs` supports Z = 1..18 only, has no Cut-IR lowering and prints `3P0` for `3P_0`; `lib.rs` exposes `serve` (TcpListener, SystemTime) from the library |
| U-hjo-3 | borgia | The web workbench labels files `.hnj`; the spec and CLI use `.hj` |
| U-hei-1 | heihachi | Running a program (driving the record graph per seek/construct) lives in the daemon's axum handler `server.rs::run`, not a library function; extract a `run_source(&mut Runtime, lang, src, backend)` |
| U-hei-2 | heihachi | `class_reached` matches channels named after rungs, which only the audio integrations emit, so in a fresh graph every seek "converges" with `classes: []` |
| U-hei-3 | heihachi | `required_power 0.8` is hard-coded in both the CLI and the server |
| U-hei-4 | heihachi | The mishima parser silently skips unknown top-level tokens, so `observe`/`assert`/`emit` lines in a `.mma` are neither parsed nor validated |
| U-wgb-1 | verum | `philharmonic/crates/dsl-wagenbau` fails 9 of its 20 tests at HEAD `bdef6b1`: the statement splitter rejects the continuation lines of a `part` (`expresses` on the next line is "not a wagenbau declaration"), so the default `STANDARD_CHASSIS` does not build. wagenbau is deferred until it does |
| U-old-1 | olduvai-exchange | The local clone is nine commits ahead of origin; `olduvai-core` is identical at both, and the vendor records origin/main |
| U-lev-1 | levinthal | `levinthal-core` declares `nalgebra`, `num-traits`, `num-complex` and `rand` but uses none; `rand` → `getrandom` blocks wasm32 |
| U-lev-2 | levinthal | `levinthal-msms` and `AminoAcid::molecular_weight` sum free amino-acid average masses labelled monoisotopic: +18.01 Da per residue (PEPTIDE 925.92 vs 799.36 Da) |
| U-lev-3 | levinthal | `levinthal-folding` is nondeterministic (`thread_rng`, no seed), fails wasm32, and at 13.2 THz with any usable `dt` the phases re-randomise every step (`fold` reported 10 164 000 000 ATP cycles in 11 steps) |
| U-lev-4 | levinthal | `TernaryString::to_sentropy` returns a constant (0.5, 0.5, 0.5); `to_cell_bounds` computes its offset from the string length, not the trit values |
| U-cyt-1 | buhera/long-grass | `cytochrome-module.js` is hand tables, not an engine wrapper: its cycle ΔM values sum to 4.053 while it reports the oracle's 4.963, and its test asserts only the constant |
| U-ss-1 | lavoisier | No Shapeshifter front end rejects arbitrary text (`compileStage` of garbage is `ok` with warnings); the web interpreter swallows runtime errors, assigns kinds by value shape rather than producer, and reorders integer-named phases |
| U-scp-1 | helicopter | `scope-lang/src/runtime` ships ~1.8 kLOC of legacy executors and API clients (network fetches, `NEXT_PUBLIC_HF_TOKEN`) that nothing reachable from `index.ts` imports |

## 3. Naming collisions, recorded and not renamed

long-grass's host-local `zangalewa` (zoom-climb's coordinate extractor) and `interceptor` (an NL → program → run assistant) share names with upstream projects they do not wrap. They were not renamed, because tutorials depend on them. The upstream Zangalewa is the module `zangalewa-dsl`, and the upstream interceptor is its transport.
