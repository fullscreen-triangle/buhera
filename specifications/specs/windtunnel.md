# windtunnel — the `.wt` assertion language

| | |
|---|---|
| **Registry id** | `windtunnel` |
| **Layer** | observation |
| **Language** | `wt` (`.wt`) |
| **Upstream** | `fullscreen-triangle/wind-tunnel` · `crates/wt-dsl` @ `50c2b59` |
| **Vendored at** | `buhera-os/vendor/wt-dsl/src` (byte-exact) |
| **Rust binding** | native: `buhera-modules/src/windtunnel.rs` (`measure` spawns `$BUHERA_WT_BIN` when configured) |
| **TS binding** | native, via wasm (`buhera-wasm`); `measure` is unavailable in the browser |

## 1. Purpose

Wind Tunnel analyses software as a coupled system. It argues that local tests are bounded observers and cannot certify global behaviour. Its measurement is `WT(E, Λ) = (R_dyn, S_flat_est, H, D, δS)`, where:

- R is a Kuramoto-style coherence;
- H is holonomy around call cycles;
- D is the set of decoherence zones;
- δS is the contribution scores.

The `.wt` language is upstream's own "deterministic half of adjudication". It states what to analyse and which lines the numbers must not cross, and it produces a verdict: `pass`, `fail`, or `incomplete`.

This module wraps that language. Measuring a real source tree (tree-sitter indexing, cycle enumeration, trace replay) is the `wt` binary's job. The module reaches it only through `measure`, and only on a host where an operator has configured it.

## 2. Upstream and vendoring

- `wt-dsl` depends only on serde and thiserror, and deliberately depends on none of the analysis crates. Its 19 upstream tests pass in the Buhera workspace.
- The `wt` CLI and the analysis crates (`wt-index`, `wt-static`, …) are **not** vendored. They pull tree-sitter C grammars and an HTTP client.

## 3. Language

The language is line-oriented. A block header sits at column 0 and ends in `:`. Its directives are indented beneath it. `#` starts a comment outside quotes. The parser collects **every** error, each with a 1-based line; a missing `scope` reports line 0.

```
scope <Name>:        repo "s" | include "glob" | exclude "glob" | language <ident>
analyse: | analyze:  static
                     cycles [through [a, b, …]] [max_depth N]
                     purpose [ablate [a, …]]
                     dynamic traces "dir"
assert:              regime <op> <Regime>                  -- Turbulent < ApertureDominated < HierarchicalCascade < Coherent < PhaseLocked
                     r_est | r_dyn | k_c | s_flat_est <op> <number>     -- op ∈ >= > <= < ==   (== within 1e-9)
                     no holonomy_violations | no cycles
                     purposeless none [in [a, …]]
report:              format text|json | include <section>
```

`cycles`, `purpose` and `dynamic` each imply `static`. An assertion over a metric that was not measured is **skipped**. It is never passed. A script with no assertions passes vacuously, and the module flags that with `vacuous: true`.

```wt
scope PaymentFlowAnalysis:
    repo "github.com/owner/repo"
    include "src/payments/**"
    include "src/ledger/**"
    exclude "**/__tests__/**"
    language typescript

analyse:
    static
    cycles through ["checkout", "ledger", "reconciler"] max_depth 5
    purpose ablate ["audit_log", "retry_middleware"]

assert:
    regime >= Coherent
    r_est >= 0.75
    no holonomy_violations
    purposeless none in ["checkout", "ledger"]

report:
    format json
    include regime_map
```

```wt
scope Minimal:
    include "src/**"

assert:
    r_est >= 0.5
    no cycles
```

## 4. Instructions

| Instruction | Effect |
|---|---|
| `"demo"`, `""` | Parse the canonical script above |
| `string` | Parse `.wt` source → `windtunnel_script` |
| `{kind:"parse", script}` | Same |
| `{kind:"evaluate", script, metric}` | Adjudicate the assertions against a caller-supplied `MetricView` `{regime?, r_est?, r_dyn?, k_c?, s_flat_est?, holonomy_violations?, cycle_candidates?, purposeless?}` |
| `{kind:"measure", script, project}` | Rust host only: `$BUHERA_WT_BIN check <tmp.wt> <project> --json --backend treesitter`. Otherwise `bridge unavailable` |

## 5. Output deltas

- `windtunnel_script`: `{scope, summary, script}`. `script` is the parsed AST.
- `windtunnel_check`: `{scope, verdict, passed, failed, skipped, results:[{source, outcome, detail}], metric, measured_by: "caller"|"wt", vacuous?}`.

## 6. Residue

`(failed + skipped) / max(1, assertions)`, the fraction of assertions not yet satisfied. A skipped assertion was never checked, so it counts as work left. A parse error yields residue 1 with `ok: false`. `ok` means a verdict was produced; the verdict itself is the answer.

## 7. Bindings and side effects

- **Evaluate and parse:** pure, identical on both hosts (one adapter, compiled twice).
- **Measure:** spawns a process, writes a temp `.wt` file, and **`wt` writes `<project>/.wt/graph.json`**. It is pinned to `--backend treesitter`: upstream's default `auto` backend POSTs whole source files to a hosted model when tree-sitter fails.

## 8. Hazards (upstream, observed)

- **Scope is ignored.** `scope.include/exclude/language/repo`, `cycles.through` and `report.include` are parsed and never read by `wt`. A scoped script analyses the whole project.
- **Dynamic holonomy is structurally zero on indexed graphs.** Trace ids are file stems, while cycle ids are `path::name`, so they never match.
- **"Ablation" is not re-execution.** In a coherent run, most units are flagged purposeless.
- The regime spelling differs across `wt` subcommands (`HierarchicalCascade` vs `Hierarchical cascade`).
- long-grass's `src/lib/wind-tunnel` is a **different** measurement (run-to-run output agreement of generated code). It is not `wt`, and its regime bands are its own. Its all-crash scoring bug (identical crashes scored R = 1, "Phase-locked") was fixed in this revision.

## 9. Conformance

- Rust:
  - `windtunnel_evaluates_assertions_against_a_supplied_metric`
  - `windtunnel_parse_errors_are_collected_with_lines`
- TS (wasm): two cases, asserting the same numbers: verdict `incomplete`, 2 passed, 1 skipped, residue 1/3.
- long-grass: the `.wt` validator accepts and rejects through wasm.

## 10. Upstream notes

- **U-wt-1.** Honour `scope.include/exclude` in `wt check`.
- **U-wt-2.** Expose `measure` from a library crate, so a Rust host can link it instead of spawning the CLI.
- **U-wt-3.** `serve` opens a request-supplied `dynamic traces` path, despite its comment saying no request path is opened.
