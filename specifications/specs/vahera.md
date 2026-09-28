# vahera — the Buhera OS language

| | |
|---|---|
| **Registry id** | `vahera` |
| **Layer** | language |
| **Language** | `vahera` (`.vhr`) |
| **Source** | this repository: `buhera-os/crates/buhera-vahera` + `buhera-kernel` |
| **Rust binding** | native: `buhera-modules/src/vahera.rs` |
| **TS binding** | none in the library federation. long-grass keeps its own host-local `vahera` module (`src/lib/modules/vahera-module.js`, in-browser kernel), registered after the library federation |

## 1. Purpose

vaHera is the OS's own declarative language. It lets a program:

- bind text to categorical coordinates in S-entropy space;
- navigate a trajectory backward to its penultimate state, then apply the completion morphism;
- store and retrieve content at its content address;
- run the zero-cost categorical sort;
- verify the triple equivalence (oscillatory ≡ categorical ≡ partition).

It is registered in the Rust federation so that:

1. the Rust DSL registry carries the OS language;
2. the gateway's `/api/dispatch` reaches the same kernel semantics as `/api/run`.

Results render through `buhera_vahera::render_result`, which was moved out of the gateway for this purpose, so both paths render identically.

## 2. Language

The language has 15 statement forms, one per line. `#` lines are comments.

| Statement | Effect |
|---|---|
| `describe <name> with "<text>"` | bind text content to a categorical coordinate |
| `resolve <name>` | compute or look up the coordinate |
| `spawn <program> from <name>` | create a categorical process |
| `navigate to penultimate` | backward-navigate to the penultimate state |
| `complete trajectory` | apply the completion morphism |
| `memory create at S(k, t, e)` | allocate at an explicit coordinate; each axis in [0, 1] |
| `memory store "<name>" = "<text>"` | store at the content coordinate |
| `memory find nearest "<text>" k=<n>` | categorical retrieval |
| `memory list` · `memory dump <name>` | inspect |
| `demon sort` · `controller verify` | zero-cost sort · triple-equivalence diagnostics |
| `kernel stats` · `kernel trace` · `process list` | kernel inspection |

```vahera
memory store "groceries" = "milk, eggs, bread and coffee"
memory store "travel" = "flight to Munich on Friday morning"
memory find nearest "shopping list" k=1
demon sort
controller verify
```

## 3. Instructions and state

| Instruction | Effect |
|---|---|
| `"demo"` / source string / `{kind:"run", source}` | Execute on the module's kernel (depth 12) |
| `{kind:"reset"}` | Fresh kernel |

The kernel persists between acts (M2). On the gateway it is per account.

## 4. Output and residue

`vahera_result`: `{results:[render_result(…)], trace}`. Residue is the number of rendered results, a count; vaHera has no distance-to-solution.

## 5. Conformance

- Rust: `vahera_kernel_state_persists_between_acts`.
- Gateway: `a_real_act_runs_and_is_audited_per_account` (per-account isolation).
- long-grass: the vaHera validator tests (`dsl-validators.test.mjs`). Its grammar-conformance case was corrected in this revision: it had asserted that `S(0.2, 1.0, -0.5)` is valid after the parser began enforcing [0, 1]. The knowledge pack gave the same out-of-range example and was corrected too.
