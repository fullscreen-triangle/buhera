# heihachi — mishima and sangoma (the micro-kernel's languages)

| | |
|---|---|
| **Registry id** | `heihachi` |
| **Layer** | language |
| **Languages** | `mishima` (`.mma`, primary) and `sangoma` (`.sgn`) |
| **Upstream** | `fullscreen-triangle/heihachi` · `micro-kernel/daemon/src/{lang,graph}` @ `bd521db` |
| **Vendored at** | `buhera-os/vendor/heihachi` (`src/lang`, `src/graph`, `examples/` byte-exact; `lib.rs` and `Cargo.toml` a local shim) |
| **Rust binding** | native — `buhera-modules/src/heihachi.rs` (feature `heihachi`) |
| **TS binding** | native, the same Rust module compiled to wasm (`buhera_modules.wasm`) |

## 1. Purpose

heihachi is "a runtime and two languages for making records, organised as an accumulating record of authored decisions rather than as a pipeline of processing steps". **mishima** computes over the record the runtime accumulates: a *seek* names what it looks for by what it excludes and climbs a ladder of rungs until closure. **sangoma** constructs new material against declared targets: a *construct* declares stages and the thresholds its output must satisfy, and the compiler answers whether that is reachable with the stages at hand and, if not, by how much it falls short.

Both share one lexer and one rule: no literal of zero resolution is writable. Every numeric literal can carry its own floor (`6.0#0.5`), and a non-positive floor is a lex error.

## 2. What is bound

`parse` then `check` for each language; a refusal is data, never a thrown error: a `Diagnostic{severity, rule, message, remedy, line, column}`, and `remedy` is never empty. Checks:

| Rule | Meaning |
|---|---|
| `rule:floor-declared` | a program must declare its ambient floor |
| `rule:floor-positivity` | the ambient floor must be strictly positive |
| `rule:floor-negotiation` | a declared floor finer than the render path resolves is refused |
| `rule:mandatory-not` | a seek without a `not` clause does not specify a region |
| `rule:coherence` | closure needs at least three rungs |
| `rule:saturation` | a ladder whose composite power is below 0.5 cannot discriminate |
| `rule:undiscriminated` | (warning) |

Composite power of a ladder is `1 − ∏(1 − kᵢ)`. The module reports it per seek or construct.

The record graph (`graph/`: `Runtime`, `Value::reading`, the min-cut contact graph, `seek_to_closure`) is vendored with the languages because upstream treats them as one unit, but the module does not drive it: running a program is orchestrated inside the daemon's HTTP handler rather than a library function (**U-hei-1**), and in a fresh graph closure is vacuous (every seek "converges" with no classes; **U-hei-2**).

## 3. Language

`--` starts a comment.

```
mishima  := "floor" NUM seek*
seek     := "seek" ID
              "not"    "{" ID ("," ID)* "}"                 -- mandatory: what the search excludes
              "toward" "{" "region" "(" ID ")" "}"
              "via"    "{" rung (">>" rung)* "}"            -- ≥ 3 rungs to reach closure
              "until"  "closure"
              ("otherwise" "decline")?
              "yield"  ID
rung     := "rung" ID "at" NUM                               -- a power k in (0, 1)

sangoma  := "floor" NUM medium? species* construct*
medium   := "medium" ID "{" "ceiling" ":" NUM "}"
species  := "species" ID "{" (ID "{" (ID ":" NUM ","?)* "}")* "}"
construct:= "construct" ID "{" ("stage" ID)* "target" "{" (ID (">=" | "<=") NUM "#" NUM)* "}" "via" "{" rung (">>" rung)* "}" "}"
```

A mishima seek — the exclusions are the specification:

```mishima
-- recall.mma
-- Finding work you already made. The exclusions are the specification:
-- twenty versions were rejected, and what they were rejected FOR is the
-- only description of the survivor that was ever precise.

floor 0.02                 -- nothing here resolves finer

seek reese_growl
  not    { thin, undistorted, mono }
  toward { region(that_2019_growl) }
  via    { rung spectral   at 0.45
        >> rung annotation at 0.30
        >> rung model      at 0.55 }
  until  closure
  otherwise decline
  yield  found
```

A sangoma construct — declared by what it must satisfy; composite power 1 − (0.60)(0.65)(0.45) = 0.8245:

```sangoma
floor 0.02

medium air {
  ceiling: -1.0
}

species reese {
  source { operators: 2, ratio: 1.0, index: 3.5 }
  motion { rate: 0.3, depth: 0.8 }
}

construct bass {
  stage fm_source
  stage resample
  stage saturate

  target {
    crest    >= 6.0#0.5      -- half a decibel is what a meter delivers
    midrange >= 0.40#0.05
    width    <= 0.85#0.02
  }

  via { rung fm_source at 0.40
     >> rung resample  at 0.35
     >> rung saturate  at 0.55 }
}
```

Refused: a seek with no `not` clause (`rule:mandatory-not`); `floor 0.0` (`rule:floor-positivity`); a target floor such as `crest >= 6.0#0.001` checked against a render path that resolves only 0.05 (`rule:floor-negotiation`).

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` | check as mishima |
| `"demo"` | check `reese.sgn` |
| `{kind?: "check", language: "mishima" \| "sangoma", source, backend_resolution? = 0.001, required_power? = 0.8}` | parse and check |

The defaults are upstream's CLI and server values (`required_power` 0.8 is hard-coded upstream; **U-hei-3**).

## 5. Output delta

`heihachi_check` — `{language, accepted, summary, diagnostics: [{severity, rule, message, remedy, line, column}], declarations: [{kind, name, defaulted}], ladders: [{kind: seek|construct, name, line, rungs: [{name, power}], composite_power}], backend_resolution, required_power}`. The parsed program itself is not emitted (sangoma's species table is a `HashMap`, whose order is not deterministic).

## 6. Residue

Error diagnostics still to fix; 0 when the program is accepted. A refused program is `ok: false`.

## 7. Bindings

One implementation, compiled twice: the Rust module is linked natively and also built into `buhera_modules.wasm` (`lang/` and `graph/` read no clock, no filesystem, no randomness). The TypeScript host's `heihachi`, `mishima` and `sangoma` are that wasm.

## 8. Side effects and hazards

None. The mishima parser skips unknown top-level tokens, so `observe`, `assert` and `emit` lines in a `.mma` are neither parsed nor validated (**U-hei-4**).

## 9. Conformance

- Vendored suite: `cargo test -p heihachi-lang` — 32 upstream tests.
- `buhera-modules` unit tests: both examples check clean; the demo's composite is 0.8245; refusals carry rule, line and remedy; floor negotiation refuses a render path coarser than the declared floors.
- `registry-ts/test/modules.test.ts` — the same numbers through wasm.
- `long-grass/test/library-federation.test.mjs`, `knowledge-packs.test.mjs`.
