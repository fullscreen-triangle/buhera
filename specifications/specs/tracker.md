# tracker — the repo character invariant χ

| | |
|---|---|
| **Registry id** | `tracker` |
| **Layer** | observation |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/bloodhound` · `thrust/tracker/src/chi.rs` @ `d614427` |
| **Vendored at** | `buhera-os/vendor/tracker-chi/src/chi.rs` (byte-exact), with a local shim (`lib.rs`, `purpose.rs`) |
| **Rust binding** | native: `buhera-modules/src/tracker.rs` |
| **TS binding** | native, via wasm; `character` only |

## 1. Purpose

bloodhound's repo-federation tracker tracks a group of repositories and each one's *conserved sense*. That sense is **χ**, the character invariant. A repository's `purpose` index is read as a finite weighted graph:

- **Vertices** are files.
- **Edges** are weighted by shared path components plus cross-file references, where a distinctive name in one file appears in another file's snippets.

χ is the **Stoer–Wagner global minimum cut of the largest connected component**. It has three properties:

- **Positive**, by the floor theorem, on a connected graph.
- **Conserved** under relabelling. The upstream test proves this invariant, I1.
- **Non-local**, in that it names a *region*: `cut_side`.

The fragment count reports the islands honestly.

## 2. Upstream and vendoring

The tracker is a binary crate with private modules and no library API, so only `chi.rs` is vendored, verbatim. Two local shim files make it a library:

- `lib.rs` re-roots the module.
- `purpose.rs` carries only the `Symbol`/`Index` structs `chi.rs` reads.

The upstream bridge that spawns the `purpose` CLI is deliberately left out. The upstream unit tests inside `chi.rs` pass in-workspace.

**Exposed:** the read-only half of the tracker.

**Deliberately not dispatchable** (spec 07, B5):
- `add` / `drift` advance the monotone act counter and write `federation.json`.
- `sync` pushes to external remotes.
- `profile git-setup --apply` rewrites global git config.

## 3. Instructions

| Instruction | Effect |
|---|---|
| `{kind:"character", index:{root, symbols:[{name, kind, file, line, snippet}]}, repo?}` | χ of the given `purpose` index. Pure; both hosts |
| `{kind:"character_at", path}` | Read `<path>/.purpose/index.json` and compute χ. Rust host with filesystem only |
| `{kind:"list", root}` | Read `<root>/.tracker/federation.json`. Rust host with filesystem only |

On the gateway and in wasm, `filesystem` is false. The two filesystem instructions then return `unavailable on this host`.

## 4. Output deltas

`repo_character`: `{repo, chi, blocks, core_blocks, fragments, cut_side:[file], salient:[{block, degree}], beta}`.

`repo_federation`: `{repos:[{name, path, remote, chi, committed}]}`.

## 5. Residue

Always 0. χ is a conserved invariant, not a distance. A change in χ means the structure changed; it does not mean work remains.

## 6. Hazards

- χ depends on `purpose`'s indexer and ignore rules. Upgrading `purpose` alone can move χ, as happened with the 2026-07-28 indexer fix.
- Containment weights connect every pair of files that share a top-level directory, so χ is shaped by directory layout.

## 7. Conformance

- Rust: `tracker_character_of_a_small_index` (3 blocks, 2 fragments, χ > 0, residue 0).
- TS (wasm): the same assertions, plus the filesystem-refusal case.
- Gateway: `filesystem_operations_are_off_on_the_gateway`.

## 8. Upstream notes

- **U-trk-1.** Split the tracker into a `lib` plus a `bin`, so χ and the registry can be depended on rather than vendored file by file.
- **U-trk-2.** `run` is a stub. It always returns `ExecutionMissing` until the network-yield organ is wired.
