# shapeshifter — Shapeshifter (virtual mass-spectrometry experiments)

| | |
|---|---|
| **Registry id** | `shapeshifter` |
| **Layer** | science |
| **Language** | `.ss` — **not registered** as a DSL; see §2 |
| **Upstream** | `fullscreen-triangle/lavoisier` · `web/src/lib/{shapeshifter,experiment,partition}`, `spectral/dbSearch.js` @ `dacc197` |
| **Paper** | lavoisier `oxford/publications/shapeshifter-ms-syntax/shapeshifter-ms-syntax.tex` |
| **Vendored at** | `long-grass/vendor/shapeshifter` (byte-exact; `package.json` local) |
| **TS binding** | native — `registry-ts/src/modules/shapeshifter.ts`, bound in `long-grass/src/lib/modules/shapeshifter-module.js` |
| **Rust binding** | none — no Rust `.ss` front end exists (`lavoisier-buhera` is a different language, `.bh`) |

## 1. Purpose

Shapeshifter describes mass-spectrometry experiments as programs: `objective`, `instrument`, `dataset`, `target_list`, `phase` and `validate` blocks whose phase statements call lavoisier operations — run a virtual lipidomics or proteomics experiment, compute partition addresses, compile timing cells from a target list, search MS/MS by SEBD, map records to an S-entropy field. The paper gives it a formal semantics: phases run in source order and the workspace grows monotonically (Thm 6.6, 6.9); values are typed by the operation that produced them (Thm 5.3); failures are either **local** (skip the entity, warn) or **global** (refuse the run) and nothing fails silently (Prop 6.12); compiling is effect-free and certifies structure only (Thm 7.6, Rem 7.8).

## 2. Which implementation, and why no DSL entry

Four things carry the name:

| Implementation | Status |
|---|---|
| `shapeshifter-py` | the paper's reference; reproduces its execution table exactly; no generative library (§9) |
| `catalogue/web/src/lib/ss` | a bit-exact JS port of the Python (its `check_port.mjs` gate passes); replays for spectral operations; no generative library |
| **`web/src/lib/shapeshifter`** | **bound** — the only implementation of the paper's generative standard library (instrument runs, cells, partition, MS/MS, observe, purpose) |
| `lavoisier-buhera` | a different language (`.bh`) that shares the name; a stub that does not compile |

**No front end rejects arbitrary text** (**U-ss-1**). `compileStage("this is not shapeshifter at all ((( ]]")` returns `ok: true` with two warnings, in the bound interpreter and in the catalogue port alike. A DSL validator must not execute (L2), so the only admissible validator is `compileStage` — and a validator that cannot reject would tell a generation loop that every draft is valid. Shapeshifter is therefore a module without a registered language until upstream has a rejecting front end. `{kind: "compile"}` still returns the structural warnings.

Where the bound interpreter departs from the paper (recorded, not patched): kinds are assigned by value shape rather than producer (an empty `run_experiment` is `list`, not `records`); integer-named phases run in numeric, not source, order; `dataset` blocks are skipped; the acquisition namespace of §8 (`read_mzml`, `link_dda`, …) is not implemented; `db.*` operations return unresolved `pending` sentinels.

## 3. The shape of a program

For reference (from the paper, §9.2; runs on the bound interpreter):

```
objective TargetedPanel:
    target: "phospholipids with per-class ranges"

phase Design:
    panel = [
        { class: "PC",  carbons: [30, 40], db: [0, 4] },
        { class: "PE",  carbons: [32, 38], db: [0, 3] }
    ]

phase VirtualRun:
    records = lavoisier.instrument.run_experiment(
        classes: panel,
        adducts: ["[M+H]+", "[M+Na]+"],
        filters: { rt: [8.0, 18.0], mz: [600, 900] }
    )
```

Arguments are named (a positional argument is silently dropped — the paper's own §9.4(b) returns zero addresses for that reason); a bracket inside a string is still counted for line continuation (paper L2).

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "run", source}` | compile, then execute every phase |
| `"demo"` | a PC lipid run followed by a partition field over its records |
| `{kind: "compile", source}` | compile only (effect-free) |

## 5. Output delta

`shapeshifter_run` — `{ok, summary, result: {type, data, summary?}, workspace: [{name, kind, value}], term: [{stream, text}], timing: [..], log: [{level, message}], pending: [names], diagnostics}`. Timing lines (`compiled in X ms`) are moved from `term` to `timing` so the rest is deterministic.

## 6. Residue and failure

- **Global failure** (e.g. every requested lipid class unknown: "No valid lipid classes"): `ok: false`, residue 1. The interpreter itself swallows the exception and returns an empty result; the adapter reads its `error: runtime` line and refuses.
- **Otherwise** `ok: true`, residue = warnings (unknown operations, skipped entities) + unresolved `pending` lookups — what the run did not accomplish. (The former residue, `workspace.length`, counted output.)

## 7. Side effects and hazards

None at run time: the bound operations are pure JS. `spectral/dbSearch.js` (the network client) is imported by the compiler but never called. Output can be large (a §9.1 run holds ~600 records).

## 8. Conformance

- `registry-ts/test/modules.test.ts` — the demo's workspace in declaration order with residue 0 and timing moved out; an all-unknown class set refused `ok: false`; one unknown operation plus one `db.search` gives residue 2 with `pending: ["y"]`.
- `long-grass/test/cytochrome-ckg.test.mjs`, `ckg-federation.test.mjs` — the P450 and tutorial programs through the module.
