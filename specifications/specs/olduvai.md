# olduvai — intrinsic addressing (the Olduvai exchange core)

| | |
|---|---|
| **Registry id** | `olduvai` |
| **Layer** | coordination |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/olduvai-exchange` · `crates/olduvai-core/src` @ `a7b520d` (origin/main) |
| **Vendored at** | `buhera-os/vendor/olduvai-core/src` (byte-exact; local `Cargo.toml`) |
| **Rust binding** | native — `buhera-modules/src/olduvai.rs` (feature `olduvai`) |
| **TS binding** | native, the same Rust module compiled to wasm |

## 1. Purpose

The Olduvai exchange addresses everything intrinsically: a thing's S-coordinates (s_k, s_t, s_e) ∈ [0, 1]³ are refined one axis at a time into a **ternary address** (up to 12 trits), so similarity is simply the length of the shared prefix and a prefix trie is a nearest-neighbour index. When a query's full address is unoccupied, the trie backs off to the deepest occupied prefix and **says how many trits of resolution it gave up** — "a result found by backing off six trits is a different kind of answer from an exact one, and saying so is the system naming its own gaps."

The core also water-fills an attention budget across scenes (the price p★ below which a scene is not asked at all) and checks the coherence of a closed foreman cycle. It is pure and deterministic: no clock, no randomness, no I/O. This is the same addressing idea as vaHera's S-coordinates, in a form another system already uses in production.

## 2. What is bound

`Address::{encode, decode, common_prefix_len}`, `Trie::{insert, nearest, ranked}`, `agent::water_fill`, `foreman::check_cycle`. The module keeps one trie of labelled points as state (R6).

Not bound: the agent/self checks (`agent::check`, χ, realised floor) duplicate the `smith` module's concepts with a different implementation; proposals and provenance belong to the exchange's workflow.

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"` | insert three labelled points, then ask for the nearest to a fourth |
| `{kind: "encode", coords: {s_k, s_t, s_e}, depth? = 12}` | the address |
| `{kind: "decode", address}` | the centre of the address's cell |
| `{kind: "compare", a, b}` | shared-prefix length |
| `{kind: "insert", entries: [{label, address \| coords, depth?}]}` | add to the trie |
| `{kind: "nearest", address \| coords, depth?}` | deepest occupied prefix, its labels, and the resolution lost |
| `{kind: "ranked", address \| coords}` | every label ranked by shared prefix |
| `{kind: "water_fill", scenes: [{name, gain_k > 0}], budget ≥ 0}` | allocations, price p★, total gain |
| `{kind: "check_cycle", cycle}` | foreman coherence of a closed cycle |
| `"reset"` | clear the trie |

## 4. Output delta

`olduvai_address`, `olduvai_coords`, `olduvai_similarity`, `olduvai_trie`, `olduvai_nearest` (`{query, prefix, matched_depth, requested_depth, exact, resolution_lost, values}`), `olduvai_ranked`, `olduvai_water_fill` (`{allocations: [{scene, allocation, marginal_gain}], price, total_gain, budget_used}`), `olduvai_cycle`. Refusals (out-of-range coordinates, malformed addresses) are `text` with `ok: false`.

## 5. Residue

`nearest`: the number of trits of resolution the fallback gave up (0 on an exact hit). Every other operation: 0.

## 6. Bindings

One implementation compiled twice (native, wasm), as for `heihachi`.

## 7. Side effects and hazards

None. The recorded commit is origin/main: the local clone is nine commits ahead, but `crates/olduvai-core` is identical at both, so the provenance points at published history.

## 8. Conformance

- Vendored suite: `cargo test -p olduvai-core` — 181 unit/property tests + 7 doctests (proptest as a local dev-dependency).
- `buhera-modules` unit tests: encode/decode round-trips to within half a cell (depth 6 = two trits per axis); `nearest` reports the resolution it gave up as its residue and never returns a distant point; out-of-range coordinates are refused.
- `registry-ts/test/modules.test.ts` — through wasm; the trie persists across acts.
