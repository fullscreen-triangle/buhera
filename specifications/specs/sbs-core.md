# sbs-core — the SBS observation calculus (Rust)

| | |
|---|---|
| **Registry id** | `sbs-core` |
| **Layer** | science |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/hegel` · `sbs/src` @ `63d09c4` |
| **Vendored at** | `buhera-os/vendor/hegel-sbs/src` (byte-exact; the manifest declares only serde + quick-xml) |
| **Rust binding** | native: `buhera-modules/src/sbs_core.rs` |
| **TS binding** | remote: `POST /api/dispatch` on buhera-gateway (`registry-ts/src/modules/remote.ts`) |

## 1. Purpose

hegel's Rust crate implements the SBS observation calculus without the language: S-entropy triples, coherence R, flux visibility V, greedy backward navigation, SBML import, and an l1 restoration. It is a separate module from [`sbs`](sbs.md) because it is not the same computation:

| | `sbs` (JS) | `sbs-core` (Rust) |
|---|---|---|
| Input | `.sbs` source | explicit circuit JSON, SBML, or the demo |
| Chemical potential | `mu` verbatim when non-zero | always μ + RT·ln c |
| Spearman ρ | ordinal ranks, `1−6Σd²/…`, NaN for n = 1 | average ranks with ties, Pearson on ranks, 1 for n < 2 |
| V with zero flux | 0 | 1 |
| Perturbation | top-flux edge only | explicit `(edge, factor)` |
| Restoration | compensating factor `1/current` | factor 1.0 replacements |

For the glycolysis demo the two agree on R (0.5919) and V (0.1175 vs 0.1176). Per-node triples differ.

## 2. Vendoring

Upstream's manifest declares 17 dependencies; the library uses two. The vendored manifest declares exactly those two (serde, quick-xml) and does not build the CLI (`autobins = false`). All 6 upstream tests pass in-workspace.

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"`, `null` | Glycolysis demo, edge 0 × 0.1 |
| `{kind:"observe", circuit? \| sbml?, perturbations?:[{edge, factor}]}` | Observe; with no `circuit`/`sbml`, uses the demo circuit |
| `{kind:"restore", …, max_edges?}` | As `observe`, plus `therapy` from `find_optimal_perturbation` |

`circuit` = `{nodes:[{name, mu?, concentration?, compartment?}], edges:[{src, dst, conductance?, rate?}]}`, with indices into `nodes`.

## 4. Output delta

`sbs_core_result`: `{num_nodes, num_edges, coherence, visibility, metrics:{r:{value, rho_ek, rho_et, rho_kt}, v:{value}, s_entropy, flux_healthy, flux_current, backward_path}, perturbations, backend:"cpu", compute_time_us, therapy?}`.

`compute_time_us` is host-local.

## 5. Residue

`1 − V`.

## 6. Hazards the adapter guards

The engine panics:

- on an out-of-range edge endpoint (`Circuit::add_edge` indexes `nodes[src]`);
- on NaN (`partial_cmp().unwrap()` in navigation and ranking).

The adapter validates indices and finiteness first and returns `invalid circuit` / `invalid perturbation` (contract I2). The solver reads `std::time::Instant`, which panics on `wasm32-unknown-unknown`. That is why the TypeScript binding is `remote` rather than wasm.

## 7. Conformance

- `sbs_core_reproduces_the_upstream_demo_numbers`: R = 0.5919, V = 0.1175 (hegel's own `sbs demo --perturb`), residue = 1 − V.
- `sbs_core_refuses_out_of_range_indices_instead_of_panicking`.
- TS: `remote (gateway): …` (transport contract).
- Gateway: `/api/dispatch` tests.

## 8. Upstream notes

- **U-sbsc-1.** Make `wgpu`, `rayon`, `clap` and friends optional features. They are unused by the library but compiled for every dependent (~185 crates, ~10 min cold).
- **U-sbsc-2.** Replace `partial_cmp().unwrap()` with `total_cmp`, and make the solver clock injectable, so the crate can target wasm.
- **U-sbsc-3.** `restore` in the CLI re-solves with factor-1.0 replacements, which is tautologically the healthy circuit.
