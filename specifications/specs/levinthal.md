# levinthal — folding as backward trajectory completion

| | |
|---|---|
| **Registry id** | `levinthal` |
| **Layer** | science |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/levinthal` · `crates/levinthal-core/src` @ `eca6596` (origin/main) |
| **Vendored at** | `buhera-os/vendor/levinthal-core/src` (byte-exact; local `Cargo.toml`) |
| **Rust binding** | native — `buhera-modules/src/levinthal.rs` (feature `levinthal`) |
| **TS binding** | native, the same Rust module compiled to wasm |

## 1. Purpose

levinthal's thesis: "protein folding is not forward search through conformational space but backward derivation through categorical space." A state is a partition coordinate (n, l, m, s) with n ≥ 1, 0 ≤ l < n, |m| ≤ l, s = ±½, and capacity C(n) = 2n². Given a goal (the native structure), the trajectory is derived **backward** to the origin (1, 0, 0, s) by repeatedly taking the predecessor under the selection rules; there is no search. Each residue of a sequence maps to an S-entropy coordinate (S_k from hydrophobicity, S_t from volume, S_e from charge class).

## 2. What is bound, and what is not

Bound: `PartitionState::{new, capacity, cumulative_capacity}`, `Trajectory::{complete, is_continuous, coherence_profile, max_depth, max_complexity}`, `amino_acid::parse_sequence` with each residue's `sentropy`, `category`, `hydrophobicity`, `charge`.

Not bound, deliberately:

| Upstream | Why not |
|---|---|
| `TernaryString::to_sentropy`, `to_cell_bounds` | placeholders: the first returns a constant (0.5, 0.5, 0.5), the second ignores the trit values (**U-lev-4**) |
| `coherence_proxy` as a physical quantity | it is a proxy; reported only inside the trajectory's profile |
| molecular weight | sums free amino-acid **average** masses, labelled monoisotopic: every residue is ~18.01 Da heavy (PEPTIDE: 925.92 vs 799.36 Da; **U-lev-2**) |
| `levinthal-msms` | same mass defect; no identification code — `shapeshifter` owns fragmentation |
| `levinthal-folding` | nondeterministic (`thread_rng`, no seed), fails wasm32, and at 13.2 THz with any usable `dt` the phases re-randomise every step (**U-lev-3**) |

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"` | complete the goal (3, 2, 1, +½) |
| `{kind: "complete", goal: {n, l, m, s}}` | the backward trajectory |
| `{kind: "analyze", sequence}` | per-residue S-entropy, category counts, net charge |
| `{kind: "capacity", depth}` | C(n) and the cumulative capacity |

## 4. Output delta

`levinthal_trajectory` — `{goal, states: [{n, l, m, s}], continuous, coherence_profile, max_depth, max_complexity}` (spin as ±0.5, not the variant name); `levinthal_analysis` — `{length, net_charge, category_counts, mean_sentropy, residues: [{code, code3, category, hydrophobicity, charge, sentropy}]}`; `levinthal_capacity`. Invalid coordinates and unknown residues are refused with the engine's message.

## 5. Residue

0: every operation is a closed-form derivation that completes in one act.

## 6. Bindings

One implementation compiled twice (native, wasm). The vendored `Cargo.toml` declares only `serde` and `thiserror`; upstream's unused `nalgebra`, `num-traits`, `num-complex` and `rand` (which pulls `getrandom` and blocks wasm32) are dropped (**U-lev-1**).

## 7. Side effects and hazards

None. An empty sequence is refused (the upstream CLI produced `NaN` for it).

## 8. Conformance

- Vendored suite: `cargo test -p levinthal-core` — 40 unit tests + 2 doctests.
- `buhera-modules` unit tests: the demo reproduces the upstream CLI's trajectory (1,0,0)→(2,0,0)→(3,0,0)→(3,1,1)→(3,2,1); `l < n` is enforced; `X` is not an amino acid; C(3) = 18.
- `registry-ts/test/modules.test.ts` — the same trajectory through wasm.
