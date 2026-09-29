# mekaneck — Mekaneck (substrate-neutral inquiry)

| | |
|---|---|
| **Registry id** | `mekaneck` |
| **Layer** | language |
| **Language** | `mekaneck` (`.mck`) |
| **Upstream** | `fullscreen-triangle/mekaneck` · `chatelier/crates/{lang,algebra}` @ `de81b8b` |
| **Vendored at** | `buhera-os/vendor/mekaneck` (crates, fixtures and examples byte-exact, upstream layout kept; `Cargo.toml`s local) |
| **Rust binding** | native — `buhera-modules/src/mekaneck.rs` (feature `mekaneck`) |
| **TS binding** | native, the same Rust module compiled to wasm |

## 1. Purpose

Mekaneck asks one kind of question of a record: *which state is this, told apart from what?* Its single primitive, `seek`, has three obligations the front end enforces:

- **Exclusion.** A seek without `excluding` is a parse error: a positive description alone does not determine a target (Thm 4.3).
- **Independent support.** A `via` clause must name at least three mutually independent catalysts; fewer cannot survive the loss of a member (Thm 6.2), and the type checker refuses them.
- **Closure, not threshold.** Evaluation terminates when the catalysts' cells close, not at a confidence level. A seek is `Resolved { cell }` or `Declined { cells }`, and **a declination is a result** (Thm 6.7) — the language says "these cells are incompatible", not "error".

A substrate declares its receivers, observable, events and **floor**; a floor that is not supplied when checking is a warning (the positive result's falsifiability is then unchecked), never silently assumed.

## 2. What is bound

`diagnose` (the check), and the upstream CLI's run glue — `parse` → `typecheck` → `FixedSubstrate` of caller-supplied cells → `eval_seek` for every `let` — reproduced exactly. There is no real substrate in-host: the cell each catalyst reads is the caller's to state (the `mekaneck-substrates` crate and its analysis pipeline are a later instruction, if wanted). The TypeScript mirror upstream (`web/src/languages/mekaneck`) is a front end only; it is not bound — the TS host runs the Rust module itself through wasm.

The vendored `lang` crate keeps its fixtures test, which pins upstream's TS mirror to the Rust front end with exact line/column diagnostics.

## 3. Language

`#` starts a comment; statements end with `;`.

```
program   := decl*
decl      := "substrate" ID "{" "receivers" ":" expr ";" "observable" ":" expr ";" "events" ":" expr ";" "floor" ":" expr ";" "}"
           | "catalyst" ID ":" expr "independent" ID ("," ID)* ";"
           | "let" ID "=" "seek" expr "excluding" expr ("via" "(" ID ("," ID)* ")")? "until" "closure" ";"
           | "report" ID ";"
expr      := ID "(" (expr ("," expr)*)? ")" | ID | STRING | NUMBER
```

A coherence regime, told apart from every other state:

```mekaneck
# Individuate a coherence regime against the rest of the record.
#
# The exclusion clause is mandatory: a target is not determined by a positive
# description alone, so `excluding` says what the regime is being told apart
# from. Termination is by closure, not by a confidence threshold.

substrate Osc {
  receivers  : recordings("cohort-A");
  observable : coherence_index();
  events     : label_change();
  floor      : asymptotic_separation();   # falsifiable; not sample_minimum()
}

catalyst spectral  : band_decomposition() independent surrogate, phase;
catalyst surrogate : phase_randomised()   independent spectral, phase;
catalyst phase     : locking_value()      independent spectral, surrogate;

let regime =
  seek        target_state("high-coherence")
  excluding   all_other_states()
  via         (spectral, surrogate, phase)
  until       closure;

report regime;
```

The instrument quantum made explicit — a floor declared as a property of the sensor:

```mekaneck
substrate CardiacQuantum {
  receivers  : series("intraday_base");
  observable : heart_rate_bpm();
  events     : rate_change();
  floor      : instrument_quantum();   # 1 bpm — a property of the sensor
}

catalyst spectral  : band_power()        independent surrogate, temporal;
catalyst surrogate : shuffled_baseline() independent spectral, temporal;
catalyst temporal  : run_length()        independent spectral, surrogate;

let rate_regime =
  seek        target_state("elevated")
  excluding   resting_states()
  via         (spectral, surrogate, temporal)
  until       closure;

report rate_regime;
```

Refused: dropping `excluding` (parse error, with its line); `via (spectral, surrogate)` (type error: fewer than three independent catalysts); running with a catalyst whose cell was not supplied (runtime refusal).

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "check", source, floors?: {Substrate: number}}` | diagnose |
| `"demo"` | run `coherence.mck` with every catalyst at `high` (the README's run) |
| `{kind: "run", source, floors?, cells: {catalyst: "cell"}}` | evaluate every seek |

## 5. Output delta

`mekaneck_check` — `{ok, summary, diagnostics: [{severity, message, line, column}]}`. `mekaneck_result` — `{ok, summary, evaluations: [{binding, evaluation: {outcome: {outcome: "resolved", cell} | {outcome: "declined", cells}, record, trace: [{catalyst, cell, record}], reached}}]}`; on refusal, `{ok: false, evaluations: [], diagnostics}`.

## 6. Residue

`check`: error diagnostics to fix. `run`: 0 — Resolved and Declined are both complete results; the engine's own `record` counter stays in the delta. 1 on a refusal.

## 7. Bindings

One implementation compiled twice (native, wasm); `lang` and `algebra` read no clock, filesystem or randomness and forbid `unsafe`. Upstream pins `thiserror 2`; the vendored `Cargo.toml`s declare it directly beside the workspace's `thiserror 1`.

## 8. Conformance

- Vendored suites: `cargo test -p mekaneck-algebra -p mekaneck-lang` — 33 + 8 algebra, 28 + 1 lang (the fixtures test, pinning the TS mirror), and doctests.
- `buhera-modules` unit tests: the README's two runs (all `high` → resolved `high` at record 1; `phase = mixed` → declined with 2 cells, `ok: true`); the exclusion, independence and missing-cell refusals.
- `registry-ts/test/modules.test.ts` — the same through wasm.
- `long-grass/test/library-federation.test.mjs`, `knowledge-packs.test.mjs`.
