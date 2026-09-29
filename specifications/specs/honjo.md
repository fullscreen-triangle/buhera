# honjo — Honjo Masamune (chemistry as cuts)

| | |
|---|---|
| **Registry id** | `honjo` |
| **Layer** | science |
| **Language** | `honjo` (`.hj`) |
| **Upstream** | `fullscreen-triangle/borgia` · `honjo-masamune/src/lib/honjo.js` @ `28a633a` |
| **Vendored at** | `long-grass/vendor/honjo/honjo.js` + `long-grass/vendor/honjo/examples/` (byte-exact) |
| **TS binding** | native — `registry-ts/src/modules/honjo.ts`, bound in `long-grass/src/lib/modules/honjo-module.js` |
| **Rust binding** | none in this revision — see §7 |

## 1. Purpose

Honjo Masamune has one computational primitive, the **cut**: individuating a part from the rest of a bounded whole at a strictly positive, irreducible cost, the floor φ > 0. Generating an atom, forming a bond and tracking an item through a process are the same operation at different arities. Every value carries a floor and a residue; `floor 0` is rejected at compile time ("the sharp cut is not expressible"); the cut count M is a monotone clock.

Atoms are derived, not looked up: `cut Z` derives the ground-state configuration by aufbau (with its exceptions), the Hund term, period and group, for Z = 1..118. `close` drives every vacancy to zero, and stoichiometry and geometry follow.

## 2. Which implementation, and why

| Implementation | Status |
|---|---|
| `honjo/src/*.ts` → `src/lib/honjo.js` | **bound** — the zero-dependency ESM bundle the honjo web app runs; 37 upstream tests |
| `honjo-rs` (Rust) | behind the TS: Z = 1..18 only, no Cut-IR lowering, `3P0` for `3P_0` (**U-hjo-2**); ships a loopback HTTP server in the library (`pub mod serve`) |
| `honjo-py/hjm` | a **different dialect** (`deloc`, graph imports, cell counts); not `.hj` |

The bundle is four lines ahead of its TypeScript source (a `shells` field; **U-hjo-1**). long-grass's honjo sandbox previously ran a stale copy (pre-`shell.ts`: Z ≤ 18, `3P0`); it now imports the vendored bundle.

## 3. Language

`--` starts a comment. Programs are straight-line: there are no loops.

```
program := stmt*
stmt    := "floor" NUM | "import" QNAME | "module" …
         | ID ":=" expr
         | "observe" expr ("as" ID)?
         | "assert" expr RELOP expr ("emit" STRING)?
         | "track" ID "in" ID ("with" "reps" ID ("," ID)*)? "until" ("converge" | "diverge" | cond) ("yield" ID)?
expr    := "cut" NUM                               -- individuate element Z
         | expr "~" expr ("when" cond)?             -- a bond: admitted only if it lowers thickness
         | "close" ID "(" args ")" ("by" ID)?       -- drive every vacancy to zero
         | QNAME "(" args ")" | ID | NUM ("#" NUM)? -- a number may carry its own floor
```

Water — cuts to closure; geometry by maximal separation:

```honjo
-- water.hj — cuts to closure, geometry by maximal separation
floor 1.0

O := cut 8
H := cut 1

-- a bond is a cut between two items, admitted only if it lowers thickness
OH := O ~ H when delta > 0
observe OH

-- close drives every vacancy to zero: stoichiometry + geometry follow
W := close O(H, H)        -- 2:1, bent, ~104.5 deg
observe W
assert W.valence == closed emit "water did not close"
```

Ionic contact, and a closed shell that forms no bond:

```honjo
-- salt.hj — ionic contact: Na ~ Cl (both open-shell, bond lowers thickness)
floor 1.0

Na := cut 11
Cl := cut 17
observe Na
observe Cl

NaCl := close Na(Cl)      -- 1:1
observe NaCl

-- a closed-shell partner forms no bond (Ne has vacancy 0)
Ne := cut 10
dead := Na ~ Ne when delta > 0
observe dead              -- exists=false
```

Tracking an item through a process — the amalgamation is the result:

```honjo
floor 1.0
import honjo.causal

O := cut 8
H := cut 1
W := close O(H, H)

path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path
```

Rejected: `floor 0` (type error at 1:1); an unbound identifier (type error with its line and column); a stray character (lex error). `cut 200` compiles and is refused when it runs ("beyond the named elements (1..118)").

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "run", source}` | compile and run |
| `"demo"` | run `track.hj` |
| `{kind: "compile", source}` | lex, parse, check, lower — no run |
| `{kind: "derive", z}` | `deriveAtom(z)`: configuration, term, period, group |

## 5. Output delta

`honjo_result` — `{ok, summary, log, cutCount, floor, values: {name: rendered line}, named: {name: value}}`; on refusal `{ok: false, errors: [{message, line?, column?}]}`. `honjo_atom` — `{z, atom}`.

An aborted assertion is `ok: false` with `error: "assert aborted"`; the log and values up to the abort are kept.

## 6. Residue

Always 0, `completed: true`: programs are straight-line and run to completion. honjo's own per-value `residue` (floor plus vacancy) is chemistry, reported inside `named` — it is not the act residue.

## 7. Why no Rust binding in this revision

`honjo-rs` would give different answers from the bound TS for Z > 18 and for term notation (**U-hjo-2**), so binding it as `honjo` would break contract equivalence. Its library core is otherwise pure and passes a wasm32 check once `serve` is excluded; it becomes a candidate when it reaches parity.

## 8. Side effects and hazards

None: pure, deterministic. Range errors surface at run time, not compile time. The web app labels files `.hnj` while the spec and CLI use `.hj` (**U-hjo-3**); the registry uses `.hj`.

## 9. Conformance

- `registry-ts/test/modules.test.ts` — all four upstream examples validate and run with their cut counts (carbon 1, salt 4, track 6, water 5); water closes bent at 104.5°; `floor 0` refused at 1:1; an unbound identifier at line 2; `cut 200` refused at run; `derive 26` is Fe.
- `long-grass/test/library-federation.test.mjs`, `knowledge-packs.test.mjs`.
