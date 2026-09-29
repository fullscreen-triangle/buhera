# 11 — The second survey: every candidate, and what became of it

**Status:** normative record · 2026-09-29

The first federation (specification 01) integrated the ten targets the user named. The second pass followed nine more links and then searched the user's repositories for languages and engines nobody had listed. This document records **every candidate examined**, the disposition, the evidence it rests on, and — for anything not integrated — exactly what would change the answer. Dispositions are decided by the rules already in force: wrap, never reimplement (M1); a validator must be the language's own front end (L1); a `none` binding with a stated reason beats a port; and a component whose own tests fail is not bound.

Evidence was gathered read-only: upstream repositories were inspected at their committed HEAD, suites were run in scratch target directories, and nothing upstream was modified.

## 1. Integrated in this pass

| Module | Language | Source | Rust | TS | Evidence |
|---|---|---|---|---|---|
| `smith` | Agent Smith `.smith` | musande `web/src/lib/agent-smith` | — | native | canonical parser + typechecker + town; the smith-ide compiler is a stub (U-smi-5) |
| `shapeshifter` | (`.ss`, not registered: no front end rejects anything, U-ss-1) | lavoisier `web/src/lib` @ `dacc197` | — | native | the only implementation of the paper's generative library; formalised with F12 |
| `ladder` | — | levinthal `enzymes/web/src/lib/engine.js` | — | native | byte-identical to upstream; formalised with F13 |
| `spectral` | — | gospel `vivid-symbolism/src/lib` | — | native | spectral embedding, shader-kernel ranking, matched filter; planted motifs recovered exactly |
| `scope` | SCOPE `.scope` | helicopter `scope-lang/src` | — | native | byte-identical to upstream; residue fixed and first tested outside the browser (F15) |
| `synopsis` | synopsis `.syp` | gospel `synopsis/ts/src` | — | native | the upstream conformance corpus (4 positive, 16 negative) runs as the test; no evaluator exists upstream, so none is exposed |
| `cfc` | cause-for-concern `.cfc` | syndrome `cause-for-concern/webtool/src/cfc` | — | native | Python ↔ JS parity checked; all five examples reproduce their statuses |
| `sthurbert` | st-Hurbert `.sth` | bloodhound `thrust/src/lib/repo-lens` | — | native | a real lexer → parser → interpreter; χ computed by the engine |
| `honjo` | Honjo Masamune `.hj` | borgia `honjo-masamune/src/lib/honjo.js` | — (twin lags) | native | 37 upstream tests; four examples run with their cut counts |
| `heihachi` | mishima `.mma`, sangoma `.sgn` | heihachi `micro-kernel/daemon/src/{lang,graph}` | native | native (wasm) | 32 upstream tests in-workspace; wasm-clean |
| `olduvai` | — | olduvai-exchange `crates/olduvai-core` | native | native (wasm) | 181 + 7 upstream tests in-workspace; pure |
| `levinthal` | — | levinthal `crates/levinthal-core` | native | native (wasm) | 40 + 2 upstream tests; wasm-clean once unused deps are dropped |
| `mekaneck` | Mekaneck `.mck` | mekaneck `chatelier/crates/{lang,algebra}` | native | native (wasm) | 72 upstream tests in-workspace (incl. the fixtures that pin its TS mirror); the README's runs reproduce on both hosts; `thiserror 2` declared locally |

## 2. Already running in long-grass, to be formalised

These engines are vendored and bound in long-grass today but are not catalogue members, so nothing checks their provenance or conformance. (Shapeshifter, ladder and SCOPE were formalised in this pass; see §1, F12–F15.) Each has a concrete defect in its current adapter, recorded here so formalisation fixes it rather than enshrining it.

| Module | Language | Upstream | Adapter defect to fix on formalisation |
|---|---|---|---|
| `graffiti` | Graffiti `.grf` | graffiti `web/src/graffiti` | residue is the count of yields, not the engine's `ClaimValue.residue`; `actBudget` ignored (it maps to `maxCatalystInvocations`); upstream is ahead (`5d90402f`, additive) |

## 3. Deferred — an upstream defect or a missing piece blocks binding

| Candidate | Repository | Blocking finding | What unblocks it |
|---|---|---|---|
| wagenbau `.wgb` | verum `philharmonic/crates` | 9 of 20 upstream tests fail at HEAD; the default chassis does not build (U-wgb-1) | upstream fixes continuation lines; then vendor vesicle-abi/kernel/lang + dsl-wagenbau (wasm-clean) |
| vitruvius `.vvs` | hehahe `musculo-skeletal/web/src/lang` | none in the engine (115/116 vitest, the one failure a timeout); runs take up to 15 s and import a JSON rig table | a Node JSON-import path in the test hooks, and running acts off the request thread |
| mutapa reactor | mutapa `web/src/lib/experiment-engine.js` | none (pure, seeded); not yet run here | a module over `Reactor` stepping, with `actBudget` = steps |
| MPL `.mpl` | bene-gesserit `phosphate/src/components/mpl` | the only export evaluates (no validator can avoid running); `T(k,d)` disagrees with the spec; the budget is never enforced; `ceil(log₃ N)` is wrong at exact powers | upstream exports `tokenize`/`parse` and fixes `T`, the budget and the log |
| mee `.mee` | pugachev-cobra `web/src/lib/mee` | the parser loops forever on both repository programs; `require` inside ES modules | a progress guard in the scene loop; static imports |
| brutscript | brut `web/src/brutscript` | its own default script fails at three stages; every `layer … from <source>` is refused | upstream fixes the `from` scope check and `index.ts:53` (`TT.Error` is 85, not 9) |
| dendra `.dra` | buhera-west `web/src/dendra` | only the lexer exists (milestone M1) | M2 (parser) and M3 (checker) |
| Shakespeare `.shk` | levinthal `cytochrome/src/helpers/shakespeare.js` | the interpreter ignores unrecognised lines and never evaluates `assert`; no validator can meet L1 | a `parse(src) → {ok, errors}` over the grammar in `shakespeare-language.tex`; the module could be bound without a DSL entry before then |
| scilang `.sci` | greifswald `scilang_proto` (Python) | no TS or Rust front end — binding it means porting ~1.4 kLOC | the user's decision to port (it is the closest to the LARA work), with the corpus (4 valid, 8 broken) as the conformance suite |
| closure | closure `closure-kernel`, `closure-runtime` | AGPL-3.0 | the user's licensing decision |
| catcount | stella-lorraine `trans_planckian/catcount` | no language; headline outputs rest on framework constants | a request to federate the physics calculators |
| zangalewa-dsl, native TS binding | zangalewa `zoom-climb/src/lib/dsl` | `generate` imports its registry at module scope, so a byte-exact copy cannot be given Buhera's validators | a one-line upstream refactor so `generate` accepts `{getDsl, buildPackContext, providers}` |
| levinthal-msms, levinthal-folding | levinthal | wrong masses (U-lev-2); nondeterministic and non-informative dynamics (U-lev-3) | fixes upstream |
| honjo-rs | borgia | Z ≤ 18, no lowering, different term notation (U-hjo-2) | parity with the TS |
| agent-smith (Rust) | musande | singleton-cut floor, short potential list (U-smi-1..3) | parity with the JS |
| synopsis (Rust) | gospel | parse only (U-syn-1) | a Rust checker passing the same corpus |
| Shapeshifter reference (`shapeshifter-ref`) | lavoisier `catalogue/web/src/lib/ss` | none in the engine: a bit-exact JS port of the paper's Python reference (its `check_port.mjs` gate passes), whose ladder operations are pure; but its AST and kind function are incompatible with the bound interpreter, and its ladder semantics overlap the `ladder` module | the user's choice of which module owns ladder semantics; it would then be a second module, never merged into `shapeshifter` |

## 4. Skipped — no engine to bind

| Candidate | Repository | Reason |
|---|---|---|
| catscript `.cat` | stella-lorraine | Python-only; `spectrum` returns canned tuples and `validate` a constant dict of `True` |
| cynegeticus | sighthound | a stub: no evaluator, and the parser cannot parse its own `S(…)` literal (the lexer never emits `TokenType.S`) |
| Turbulance variants (7) | kwasa-kwasa, hegel, nebuchadnezzar, four-sided-triangle, moriarty, borgia, gospel | none adds grammar ndombolo-core lacks; four do not build |
| lavoisier-buhera `.bh` | lavoisier | a different language that happens to share a name; a stub that does not compile |
| hieronymus scope-compiler | helicopter | an older ancestor of the vendored scope-lang |
| enzymes/webtool | levinthal | superseded by `enzymes/web` (its README says so); its `engine.js` is the same body wrapped for `window`, and does not load in Node despite its README |
| zoom-climb `bridge/`, `desk/`, `prompt.ts`, `coord-extract.ts` | zangalewa | `bridge/` is the predecessor of the interceptor broker Buhera already binds; `desk/` is a single-user GitHub/Hugging Face indexer with no language; the other two belong to the old research-card module (long-grass keeps dead copies, U-zng-5) |
| mogadishu, huygens, conclave, berlin, faraday, fourth-stomach, lagrangian | — | no pure engine reachable from either host (pyo3/tokio throughout, empty manifests, LaTeX only, Python only, GPU only) |

## 5. Drift and defects found in Buhera itself

| Where | Finding | Status |
|---|---|---|
| `long-grass/src/lib/smith` | a port of the smith-ide stub | removed (F9) |
| `long-grass/src/lib/sandboxes/honjo/honjo.js` | stale honjo (Z ≤ 18) | removed (F11) |
| `long-grass/src/lib/modules/cytochrome-module.js` | not an engine wrapper: hand tables whose cycle ΔM sums to 4.053 while it reports the reference 4.963; the test checks only the constant | open — re-base on the lesson oracle or relabel as curated tables (U-cyt-1) |
| `long-grass/src/lib/zangalewa/` | byte copies of zoom-climb's `prompt.ts`, `coord-extract.ts`, `desk/*` that nothing imports | open — dead code (U-zng-5) |
| vaHera validity | long-grass's parser rejects coordinates outside [0, 1]; zoom-climb's TS parser and the vendored Rust `zangalewa-dsl` accept them, and zoom-climb's pack teaches `S(0.2, 1.0, -0.5)`. A chunk the broker's agent accepts can be refused when long-grass dispatches it | open — the three parsers must agree (U-zng-6) |
| `long-grass/src/lib/lavoisier/shapeshifter/` | a third copy of the Shapeshifter compiler (the `162ae1b` logic with rewritten imports) that nothing imports; the rest of `src/lib/lavoisier` backs the host-local `lavoisier` module | open — dead code |
| `long-grass/src/lib/purpose/dsl-generator.js` | integrates drafts into one (the loop's contract returns every accepted draft) and frames packs as domain facts | open (U-zng-3) |
