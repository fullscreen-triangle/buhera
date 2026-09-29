# synopsis — the genomic scripting language

| | |
|---|---|
| **Registry id** | `synopsis` |
| **Layer** | science |
| **Language** | `synopsis` (`.syp`) |
| **Upstream** | `fullscreen-triangle/gospel` · `synopsis/ts/src` @ `f4b695b` |
| **Vendored at** | `long-grass/vendor/synopsis/src` (byte-exact), corpus at `long-grass/vendor/synopsis/corpus/corpus.json` |
| **TS binding** | native — `registry-ts/src/modules/synopsis.ts`, bound in `long-grass/src/lib/modules/synopsis-module.js` |
| **Rust binding** | none — see §7 |

## 1. Purpose

synopsis writes genomic analyses — motif scanning, spectral homology, variant adjudication, annotation transfer — as programs whose every threshold, frame and residue is on the page. A comparison returns a value **and a residue**, and the residue must be recorded or explicitly dropped before its frame closes; parameters have **no defaults**; loops are bounded by construction. The point is that the method is recoverable from the script: a checked program's *report* lists the frames it used, every parameter it set, the residues it carried, and the claims it makes.

The language is the executable face of gospel's paper; the Python reference (`validation/lang.py`) and the TypeScript front end are held to one conformance corpus of positive and negative programs.

## 2. What exists, and what does not

| Implementation | Status |
|---|---|
| `synopsis/ts/src` | tokeniser, parser, **Stage-B checker**; exports `parse`, `checkProgram`, `check`, `accepts`, the error hierarchy. **No evaluator** — deliberately (gospel `IDE-PLAN.md`) |
| `synopsis/rs` | Rust tokeniser + parser only; no checker |
| `validation/lang.py` | the Python reference the corpus is extracted from |

So this module **checks** programs, it never **runs** them. A request to run is refused with `error: "no evaluator exists"` — not approximated.

## 3. Language

A program opens its inputs, optionally declares named methods, then works inside one or more **frames** (`under name { … }`), and ends with `report to "file"`. `#` starts a comment. Every form is keyword-led; there is no `while`, no `if`, no recursion and no indexing, so every program terminates and no program can reach into a sequence position.

```
program   := open* method* frame+ "report" "to" STRING
open      := "open" ID "=" STRING
method    := "method" ID "=" ID "(" params ")"
frame     := "under" ID "{" stmt* "}"
stmt      := "let" ID "=" expr
           | "bind" ID "," ID "=" "compare" expr "against" expr "by" ID "(" params? ")"   -- value, residue
           | "relax" ID "until" "quiescent" "{" params "}"
           | "for" ID "in" "items" "(" ID ")" "where" cond "{" stmt* "}"
           | "claim" STRING "=" ID
           | "record" ID ("," ID)*  |  "drop" ID
expr      := "project" ID "by" ID "(" params? ")"
           | "detect" ("peaks" | "top") "in" ID "{" params "}"
           | "unit" ID "anchors" NUM  |  "nearest_unit" ID "to" ID
           | "response" ID "by" ID
           | "corr" "from" ID "to" ID "by" ID
           | "align" "central" "(" ID "," ID ")" "response" "(" ID "," ID ")" "under" ID "," ID "{" params "}"
params    := ID "=" (NUM | ID) ((";" | ",") ID "=" (NUM | ID))*
```

**Required parameters** (Thm 9.7 — there are no defaults; omission is a `ParameterError`):

| Form | Required |
|---|---|
| `detect peaks` | `z`, `min_distance`, `min_score` |
| `detect top` | `k`, `depth` |
| `align` | `theta` |
| `relax` | `eta`, `theta` (both > 0, else `TerminationError`) |
| `smith_waterman` | `match`, `mismatch`, `gap` |

**Result types** `profile`, `ranked`, `verdict` are deliberately not unified: `detect top` of a profile is a `TypeError`, as is comparing embeddings of different dimension.

**Refusals** form a hierarchy; the corpus records the most specific class and a checker may report a strict subclass, never a superclass:

`SynopsisError` ⊃ `ParseError`, `TypeError` ⊃ {`ArityError`, `ScopeError`, `ResidueError`, `ParameterError`, `TerminationError`}.

A motif scan — the residue of the comparison is recorded, so the frame closes cleanly:

```synopsis
# Locate a motif in a target by coherent multichannel matched filtering.
open motif  = "motif.fa"
open target = "chr7_region.fa"

under nucleotide {
    let q = project motif  by channels(dna)
    let t = project target by channels(dna)
    bind r, res_r = compare q against t by xcorr(normalised)
    let hits = detect peaks in r {
        z            = 4.0 ;
        min_distance = 30  ;
        min_score    = 0.35
    }
    claim "motif occurs at least once above z=4" = hits
    record hits, res_r
}

report to "motif_scan.report"
```

Spectral homology with an exact re-rank — every threshold is written down:

```synopsis
open query = "query.fa"
open db    = "swissprot_subset.fa"

under residue_space {
    let e  = project query by spectral(coeffs = 8)
    let DB = project db    by spectral(coeffs = 8)
    bind s, res_s = compare e against DB by shader(cosine)
    let cands = detect top in s {
        k      = 200 ;
        depth  = 12
    }
    bind final, res_f = compare e against cands by smith_waterman(
        match = 2 ; mismatch = -1 ; gap = -2
    )
    claim "top 20 by exact rerank" = final
    record final, res_s, res_f
}

report to "homology.report"
```

Variant adjudication — `align` must carry both the central and the response correspondence (dropping `response(…)` is an `ArityError`), and `relax` must declare positive bounds:

```synopsis
open wt  = "wildtype.solved"
open var = "variant.solved"

method pi_edge_raise = perturb_incident(magnitude = 0.10)

under cell_receiver {
    let Nwt  = project wt  by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt  anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA by pi_edge_raise
    let rB = response cB by pi_edge_raise
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from rA to rB by species_name
    let v = align central(cA, cB) response(rA, rB)
            under phi_c, phi_r { theta = 0.01 }
    relax v until quiescent { eta = 0.005 ; theta = 0.01 }
    claim "variant is functionally equivalent to wild type" = v
    record v
}

report to "adjudicate.report"
```

Common refusals: a residue bound by `bind` and never recorded or dropped (`ResidueError`); a name used in a frame other than the one that binds it (`ScopeError`); a missing required parameter (`ParameterError`); `relax` with `eta = 0` (`TerminationError`); a missing `report to` (`ParseError`).

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "check", source}` | parse + typecheck; emit the checker's report |
| `{kind: "parse", source}` | the AST (ordered parameter maps become `[key, value]` pairs) |
| `{kind: "tokens", source}` | the token stream |
| `{kind: "run" \| "evaluate", …}` | refused: `error: "no evaluator exists"` |

## 5. Output delta

- `synopsis_report` — `{ok, summary, report: {frames, parameters, residues, abandoned, claims, bounds, iterations, responses}}` on success; `{ok: false, summary, refusal: {className, message, line}}` on refusal.
- `synopsis_ast` — `{ast}`; `synopsis_tokens` — `{count, tokens}`.

A refusal is `ok: false` with the checker's class and 1-based line. A front-end crash on malformed input (a non-`SynopsisError`, e.g. on truncated source) is also a rejection, reported as `front end failed on this input: …` (**U-syn-2**).

## 6. Residue

0 when the program checks; 1 when it is refused (the checker reports one refusal at a time). synopsis's own *residues* — the second value of every `compare` — are language data and appear in the report, not in the act residue.

## 7. Why no Rust binding

`synopsis/rs` parses but does not check (**U-syn-1**). A Rust `synopsis` binding would accept programs the TS binding refuses, breaking contract equivalence. It becomes a candidate when the Rust checker lands and passes the same corpus.

## 8. Side effects and hazards

None: the front end is pure TypeScript with no I/O. `open` names files; nothing is read.

## 9. Conformance

- `registry-ts/test/modules.test.ts` — the whole upstream corpus (4 positive, 16 negative): positives check with residue 0; each negative is refused with its declared class or a subclass (`isSubclassOf`), through both `dispatch` and the DSL registry. Parse/tokens deltas are plain JSON; `run` is refused.
- `long-grass/test/library-federation.test.mjs` — valid and residue-leaking programs through the long-grass facade.
- `long-grass/test/knowledge-packs.test.mjs` — the three examples above validate.
