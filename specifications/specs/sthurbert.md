# sthurbert — st-Hurbert (repository query language)

| | |
|---|---|
| **Registry id** | `sthurbert` |
| **Layer** | observation |
| **Language** | `sthurbert` (`.sth` — a Buhera assignment; upstream has no file extension) |
| **Upstream** | `fullscreen-triangle/bloodhound` · `thrust/src/lib/repo-lens` @ `d614427` |
| **Vendored at** | `long-grass/vendor/sthurbert/src` (byte-exact, minus `ai.ts`, `analyse.ts`, `engine/`) |
| **TS binding** | native — `registry-ts/src/modules/sthurbert.ts`, bound in `long-grass/src/lib/modules/sthurbert-module.js` |
| **Rust binding** | none — see §7 |

## 1. Purpose

st-Hurbert is "the informed user's path into the repo lens": a small language that navigates and slices an **analysed federation** of repositories. A repository is analysed into its symbol index and its **character** — χ, the minimum cut of the co-occurrence graph of its files, with the salient blocks, the fragments and the cut side. The language asks questions of that analysis: which repositories, which symbols, which files carry the sense, how fragmented is it.

Upstream uses `compile` as the gate for model-generated programs ("the ground-truth check an AI-generated program must pass before it runs"), which is exactly how the DSL registry uses it.

## 2. What is bound

The lexer → parser → interpreter in `sthurbert/`, over `model.ts` (the federation), `chi.ts` (`computeCharacter`) and `lineage.ts` (image lineage, reported as absent when there is none). `github.ts` is vendored for its types only; its `fetch` code is never reached from the language. `ai.ts` (the LLM front door) is not vendored.

The module keeps the federation as state (R6). Repositories enter by `load` from symbol indexes — the `{name, kind, file, line, snippet}` rows a `.purpose/index.json` holds — and each one's character is computed by the engine's own `computeCharacter`.

## 3. Language

Statements are separated by `;` or newlines; `--` and `#` start comments. `slice` and `find` narrow the working set cumulatively.

```
program := stmt ((";" | NEWLINE) stmt)*
stmt    := "navigate" (ID | "*")
         | "slice" KIND? ("where" cond)?
         | "show" ("sense" | "chi" | "salient" | "fragments" | "files" | "symbols" | "lineage" | "regime" | "health")
         | "find" STRING
         | "compose" stmt                                  -- a no-op prefix
cond    := and ("or" and)*        and := cmp ("and" cmp)*
cmp     := ("name" | "kind" | "file" | "line") op value
op      := "==" | "!=" | "~" | "contains" | ">" | "<" | ">=" | "<="    -- numeric ops need a number
```

Output caps: 100 symbols, 30 finds, 200 files, a 50-row data sample.

Upstream's own examples (thrust `ai.ts`):

```sthurbert
navigate * ; show chi
navigate myrepo ; slice fn where name contains parse ; show symbols
navigate * ; find "entropy" ; show files
slice where kind == class and file contains model ; show symbols
```

Rejected: `show nope` (`parse error (1:6): unknown show target "nope"`), `slice where line > abc` (a numeric operator needs a number). `navigate zzz` parses, and is refused at run time when `zzz` is not in scope.

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "query", source}` | run against the loaded federation |
| `{kind: "load", repos: [{name, path?, symbols}]}` | add repositories; χ is computed by the engine |
| `{kind: "compile", source}` | lex + parse only |
| `"demo"` | load a small, openly synthetic repository and query it |
| `"reset"` | clear the federation |

## 5. Output delta

`repo_query` — `{ok, summary, blocks: [{label, kind, lines, data?}]}` where `kind` ∈ sense, chi, salient, fragments, files, symbols, lineage, regime, health, slice, find, scope; on refusal `{ok: false, blocks: [], error: {message, line?, column?}}`.

## 6. Residue

0 for an answered query, 1 for a refused one. The engine defines no working-set residue, and `run` returns only capped samples, so none is invented.

## 7. Why no Rust binding, and the χ overlap

There is no Rust st-Hurbert. Note that `chi.ts` is thrust's TypeScript port of the `chi.rs` the `tracker` module vendors byte-exact from the same commit; the two can drift (**U-sth-1**). Their field names differ (`core_blocks`/`coreBlocks`, `cut_side`/`cutSide`, salient `{block, degree}`/`{file, weight}`).

## 8. Side effects and hazards

None: pure over an in-memory federation. The number lexer accepts `1.2.3` (read as 1.2), and a repository whose name starts with a digit cannot be navigated to (**U-sth-2**).

## 9. Conformance

- `registry-ts/test/modules.test.ts` — refusal on an empty federation; the demo's chi and symbols blocks; persistence across acts; parse refusals carry line and column; `reset` clears.
- `long-grass/test/library-federation.test.mjs`, `knowledge-packs.test.mjs`.
