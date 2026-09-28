# tempus — the Tempus timing language

| | |
|---|---|
| **Registry id** | `tempus` |
| **Layer** | science |
| **Language** | `tempus` (`.tempus`) |
| **Upstream** | `fullscreen-triangle/stella-lorraine` · `web/src/lib/tempus` @ `4491e88` |
| **Vendored at** | `long-grass/vendor/tempus/src` (byte-exact) |
| **TS binding** | native: `registry-ts/src/modules/tempus.ts`, bound in `long-grass/src/lib/modules/tempus-module.js` |
| **Rust binding** | none; see §7 |

## 1. Purpose

In Tempus, the single runtime datum is the **timing residual ΔP**, the gap between a reference clock and a received tick.

- A program declares **cells**, which are intervals in ΔP-space, each mapped to an action.
- Execution measures ΔP, finds its cell, and dispatches that cell's action.
- Channels are declared with `sync`, and grouped into a trajectory with `compose`.

The compiler runs static checks: duplicate cells, empty or inverted bounds, undeclared cells in `when` with a Levenshtein "did you mean", coverage gaps, and overlaps.

The engine also ships a **seeded synthetic-event simulator**:

- ΔP values are drawn by a Mulberry32 PRNG inside a cell chosen in proportion to its width, plus Gaussian noise.
- Actions are labels. **Nothing executes.**

The delta says so (`synthetic: true`), and anomaly counts must never be read as a quality measure.

Two further surfaces live in the same library:
- **construct**: wave and slit interference scenes;
- **compose**: refinement cycles to a partition state and an element.

## 2. Language

Keywords are case-insensitive, and comments start with `--`.

```
program := decl*
decl    := "cell" IDENT "bounds" "(" NUM "," NUM ")" "action" NUM
         | "sync" IDENT "at" NUM "freq"
         | "compose" "d" "=" NUM "channels" IDENT ("," IDENT)* "into" IDENT
         | "when" IDENT "do" stmt
stmt    := "emit" IDENT | "fire" IDENT ("(" NUM (","? NUM)* ")")? | "wait" NUM
         | "begin" (stmt ";"?)* "end"
```

Diagnostics come in three severities:
- **Errors:** duplicate cell; no cells; `lo ≥ hi`; undeclared `when` cell; a `compose` channel without `sync`; no channels.
- **Warnings:** `compose d` ≠ channel count; coverage gap; overlapping cells (the first match wins).
- **Info:** a cell with no `when`.

Lexer quirks:
- A leading `-` is part of a number only after `(`, `,`, `=` or at the start of input, so `action -1` lexes as `action 1`.
- Unknown characters are dropped, so `.5` lexes as `5`.

```tempus
sync coolant at 10.0e6 freq
cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7) action 1
cell HOT      bounds ( 5.0e-7, 2.0e-6) action 2
cell CRITICAL bounds ( 2.0e-6, 1.0e-5) action 3
compose d=1 channels coolant into coolant_traj
when NOMINAL  do emit status_ok
when WARM     do emit status_warn
when HOT      do begin
                 emit status_hot;
                 fire reduce_power
               end
when CRITICAL do begin
                 emit scram;
                 fire emergency_shutdown
               end
```

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"`, `string` / `{kind:"compile", source}` | Compile and statically check |
| `{kind:"simulate", source, totalEvents?=200, batchSize?=50, noiseSigma?=0.1, seed?=42}` | Seeded synthetic run; **one act-budget unit = one engine batch** (M6); `completed` when all requested events exist |
| `{kind:"construct", source}` / `{kind:"compose", source}` | The other two surfaces |

## 4. Output deltas

- `tempus_compile`: `{diagnostics, registry}`, or `{diagnostics, lines}` on error.
- `tempus_simulation`: `{synthetic: true, generated, total_events, by_cell, by_action, by_phase, events (last 200), truncated}`.
- `tempus_construct` / `tempus_compose`: `{scene, diagnostics}`.

ES `Map`s in engine output are converted to plain objects.

## 5. Residue

- compile, construct and compose: the number of error diagnostics.
- simulate: the fraction of requested events not yet generated. This is progress only.

## 6. Hazards

- A trajectory dispatches only the **first** action of the last event's cell, so the second statement of a `begin…end` never fires.
- Time uses only the first channel's frequency.

## 7. Why no Rust binding

stella-lorraine's Rust crate `tempus` is a **different system**. It is an LHC Level-1 trigger kernel: symmetric timing cells bound to channels, AND-of-cells paths, an OR of paths, and a vaHera JSON AST dispatched through a five-subsystem kernel. It has no text grammar. At `4491e88`:

1. **It does not compile**: `TempusError` lacks `Clone` (`typecheck.rs:61, :192`). **U-tmp-1**
2. With that patched, 2 of 41 tests fail. Sequential composition never passes the carried value to the next stage (`vahera.rs:282` only substitutes an explicit `Hole("input")`). As a result, multi-path trigger programs compute "last path fires" instead of the OR. **U-tmp-2**
3. The efficiency figure η_C counts rejected events as accepted. **U-tmp-3**

Binding it would violate M1 (wrap working engines, never patch or reimplement). The binding becomes available when U-tmp-1 and U-tmp-2 are fixed upstream. The smallest honest surface then is `TempusProgram::decide(&[TimingResidual])`.

## 8. Related: the scheduler

stella-lorraine's residue-driven scheduler (`web/src/lib/scheduler`) is byte-identical to `long-grass/src/lib/scheduler`. Its provenance is now recorded in `vendor.json`. It is host infrastructure, not a module (spec 08 §3).

## 9. Conformance

`registry-ts/test/modules.test.ts`:
- `tempus: the coolant lesson compiles; simulate is seeded and paged by the act budget`: deterministic under a seed; 2 budget units × 25-event batches = 50 events; residue 0.5.
- `tempus: diagnostics lesson surfaces the engine's did-you-mean`.
