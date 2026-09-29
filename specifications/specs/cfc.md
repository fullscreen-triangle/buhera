# cfc — cause-for-concern (thermodynamic cycle consistency)

| | |
|---|---|
| **Registry id** | `cfc` |
| **Layer** | science |
| **Language** | `cfc` (`.cfc`) |
| **Upstream** | `fullscreen-triangle/syndrome` · `cause-for-concern/webtool/src/cfc` @ `6cf5588` |
| **Vendored at** | `long-grass/vendor/cfc/src` + `long-grass/vendor/cfc/examples.js` (byte-exact) |
| **TS binding** | native — `registry-ts/src/modules/cfc.ts`, bound in `long-grass/src/lib/modules/cfc-module.js` |
| **Rust binding** | none — see §7 |

## 1. Purpose

cfc writes experiments that ask whether a metabolic network's thermodynamic annotations are **cycle-consistent**. A network is a circuit: species carry a standard potential μ°, a concentration and (optionally) an uncertainty σ; reactions are edges with a conductance. Solving gives μ = μ° + RT ln c and fluxes J = GΔμ. Around every cycle the potential differences must sum to zero; the sum that remains — the **holonomy** — is the defect.

The language holds four commitments in its grammar and runtime:

1. **A verdict cannot exist without its tolerance.** `admit H tolerance T` is the only production that yields a verdict.
2. **A tolerance cannot be conjured.** Each cycle has a numerical floor (machine precision over the potential range) and a data floor (from the species' σ). Where σ is missing the data floor is *undefined*, never defaulted.
3. **Verdicts are three-valued.** CONSISTENT when |H| ≤ ε_num; UNDECIDABLE when ε_num < |H| ≤ ε_data; INCONSISTENT when |H| > ε*.
4. **INVALID is not NEGATIVE.** A reference that fails its own check licenses no conclusion at all; a failed assertion is a real (negative) finding.

## 2. Which implementation, and why

| Implementation | Status |
|---|---|
| `prototype/cfc/*.py` | the Python reference (2 090 LOC, stdlib only; 28 tests) |
| `webtool/src/cfc/*.js` | **bound** — the JS port; parity checked on all four Python examples (status, clock, every verdict, holonomies, witness sets) |

`measure`, `localize`, `close` and the gap-closure machinery are lexed upstream but not implemented; `import` binds a placeholder module (**U-cfc-2**).

## 3. Language

`--` starts a comment. A program declares a numerical `floor`, builds and solves circuits, derives potentials and cycle bases with builtins, and admits verdicts per cycle.

```
program    := stmt*
stmt       := "floor" NUM
            | "let" ID ":=" expr
            | "circuit" ID "{" ( species | reaction )* "solve" "yield" ID "}"
            | "holonomy" "of" ID "in" expr "yield" ID
            | "tolerance" "of" ID "with" ID "yield" ID
            | "admit" ID "tolerance" ID "yield" ID              -- the only verdict producer
            | "witness" "of" ID "yield" ID
            | "foreach" ID "in" expr "{" stmt* "}"
            | "where" cond "collect" ID "into" ID
            | "assert" cond "emit" STRING ( "otherwise" ("invalid" | "decline") "emit" STRING )?
            | "report" expr ("," expr)* | "emit" STRING
species    := "species" ID ":" "mu0" ":" NUM "," "concentration" ":" NUM ( "," "sigma" ":" NUM )?
reaction   := "reaction" ID ":" ID "->" ID "," "k" ":" NUM
builtins   := centre_potentials | minimum_cycle_basis | fundamental_basis | perturb_edge | perturb
            | size | abs | edge | count_where
```

The third verdict in action — the same 3 kJ/mol defect is resolved when σ = 0.02 and UNDECIDABLE when σ = 3.0:

```cfc
-- 02_undecidable.cfc
--
-- The third verdict is not a formality.
--
-- The same 3 kJ/mol defect is applied twice: once to a circuit whose
-- thermodynamics is well characterised, once to one annotated with
-- realistic component-contribution uncertainties.
--
-- A two-valued test must call the second case either CONSISTENT --
-- denying a signal that is real -- or INCONSISTENT -- asserting a
-- defect the data cannot resolve. Neither is warranted.

floor 1e-9

circuit WellCharacterised {
  species A : mu0 : -100.0, concentration : 1.0, sigma : 0.02
  species B : mu0 : -140.0, concentration : 1.0, sigma : 0.02
  species D : mu0 : -180.0, concentration : 1.0, sigma : 0.02

  reaction AB : A -> B, k : 0.1
  reaction BD : B -> D, k : 0.1
  reaction DA : D -> A, k : 0.1

  solve yield C_good
}

let G := centre_potentials(C_good)
let BG := minimum_cycle_basis(G)
let G_def := perturb_edge(G, "BD", 3.0)

foreach loop in BG {
  holonomy of loop in G_def yield hg
  tolerance of loop with WellCharacterised yield tg
  admit hg tolerance tg yield vg
  report "well_characterised", loop, hg, tg, vg

  assert vg == INCONSISTENT
    emit "3 kJ/mol resolved at sigma = 0.02"
}

circuit PoorlyCharacterised {
  species P : mu0 : -100.0, concentration : 1.0, sigma : 3.0
  species Q : mu0 : -140.0, concentration : 1.0, sigma : 3.0
  species R : mu0 : -180.0, concentration : 1.0, sigma : 3.0

  reaction PQ : P -> Q, k : 0.1
  reaction QR : Q -> R, k : 0.1
  reaction RP : R -> P, k : 0.1

  solve yield C_poor
}

let H := centre_potentials(C_poor)
let BH := minimum_cycle_basis(H)
let H_def := perturb_edge(H, "QR", 3.0)

foreach loop in BH {
  holonomy of loop in H_def yield hp
  tolerance of loop with PoorlyCharacterised yield tp
  admit hp tolerance tp yield vp
  report "poorly_characterised", loop, hp, tp, vp

  assert vp == UNDECIDABLE
    emit "same defect is UNDECIDABLE at sigma = 3.0"
}

emit "the verdict tracks data quality, as it must"
```

The validity gate — a reference that fails its own check is `otherwise invalid`, and a defect is localised to a witness set, never to a named loop:

```cfc
floor 1e-9

circuit Reference {
  species Glucose  : mu0 : -917.0,  concentration : 5.0,   sigma : 1.2
  species G6P      : mu0 : -1760.0, concentration : 0.5,   sigma : 2.4
  species FBP      : mu0 : -2600.0, concentration : 0.1,   sigma : 3.1
  species G3P      : mu0 : -1510.0, concentration : 0.05,  sigma : 2.0
  species Pyruvate : mu0 : -474.0,  concentration : 0.1,   sigma : 0.9

  reaction HK   : Glucose -> G6P,      k : 0.10
  reaction PFK  : G6P -> FBP,          k : 0.05
  reaction ALD  : FBP -> G3P,          k : 0.08
  reaction PK   : G3P -> Pyruvate,     k : 0.12
  reaction GNG  : Pyruvate -> Glucose, k : 0.02
  reaction SHNT : G6P -> G3P,          k : 0.03

  solve yield C_ref
}

let C_centred := centre_potentials(C_ref)
let B := minimum_cycle_basis(C_centred)

foreach loop in B {
  holonomy of loop in C_centred yield h_ref
  tolerance of loop with Reference yield t_ref
  admit h_ref tolerance t_ref yield v_ref
  assert v_ref == CONSISTENT
    emit "reference cycle consistent"
    otherwise invalid
      emit "reference network fails its own consistency check"
}

let C_pert := perturb_edge(C_centred, "SHNT", 40.0)

foreach loop in B {
  holonomy of loop in C_pert yield h
  tolerance of loop with Reference yield t
  admit h tolerance t yield v
  where v == INCONSISTENT collect loop into flagged
}

witness of flagged yield W
report "witness_set", W
```

Rejected at parse time: `admit h yield v` (no `tolerance` clause — a verdict may not be produced without one).

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "run", source, name?}` | run the program; return the record |
| `"demo"` | run `01_validity_gate.cfc` |
| `{kind: "example", name}` | run one of the five upstream examples |
| `{kind: "check", source}` | parse only |

## 5. Output delta

`cfc_record` — the engine's record verbatim (Sets as arrays): `{status: OK | NEGATIVE | INVALID | ERROR, floor, committedMeasurements, environment{machU, RT, zAlpha}, circuits, cycles, tolerances, verdicts[{verdict, holonomy{loop, value, absValue, length, potentialRange}, tolerance{numerical, data, star, dataAvailable}, line}], assertions, emissions, reports, witnessSet, error}`, plus `summary`. `cfc_check` for `check`.

`ok` is **false only for ERROR** (the program could not be run). OK, NEGATIVE and INVALID are results.

## 6. Residue

The questions the run could not settle: the number of UNDECIDABLE verdicts, plus one if the reference was INVALID. A run is atomic; `committedMeasurements` (which ticks on every holonomy) is the engine's own clock and is reported, not budgeted.

## 7. Why no Rust binding

There is no Rust implementation; the Python reference cannot run in either host.

## 8. Side effects and hazards

None: pure JS, deterministic, always terminating (`foreach` iterates finite lists; there is no `while`). The interpreter rethrows unexpected JS errors; the adapter contains them (R2). Upstream `npm test` runs zero tests (its glob is `*.test.mjs`, the files are `*.mjs`; **U-cfc-1**), so this module's test is the one that runs the examples.

## 9. Conformance

- `registry-ts/test/modules.test.ts` — all five upstream examples reproduce their statuses (OK, OK, INVALID, ERROR, OK), residues 0/1/1/–/0, every verdict carries a numeric tolerance, the demo's witness set is `["SHNT"]` at clock 5; `admit` without `tolerance` is rejected with a line.
- `long-grass/test/library-federation.test.mjs`, `knowledge-packs.test.mjs`.
