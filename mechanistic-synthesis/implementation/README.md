# Purpose — mechanistic-synthesis implementation

Rust workspace implementing the frozen Purpose framework described in
[`long-grass/integration.md`](../../long-grass/integration.md).

**Scope of this workspace (Stage 0 MVP)**: the four crates enumerated in
§2.1 of the integration contract — `purpose-core`, `purpose-operations`,
`purpose-domains-protein`, `purpose-cli`. Nothing more. Later stages
(`purpose-kernel`, `purpose-interceptor`, `purpose-cascade`,
`purpose-factory`, `purpose-aperture`) live in separate crates and
depend on this surface without modifying it.

## Layout

```
mechanistic-synthesis/implementation/
├── Cargo.toml                         # workspace manifest
├── rust-toolchain.toml                # stable, rustfmt, clippy
└── crates/
    ├── purpose-core/                  # VaHera, Value, Type, Operation,
    │                                  # Domain, Resolver, Error, typecheck
    ├── purpose-operations/            # Provider, OperationRegistry,
    │                                  # Executor, UniprotProvider,
    │                                  # ProteinSummaryProvider
    ├── purpose-domains-protein/       # Protein resolver + domain +
    │                                  # register_providers
    └── purpose-cli/                   # `purpose` binary
```

## Frozen surface (contract §2.1)

Eight symbols are permanent under semver:

| Symbol | Crate |
| --- | --- |
| `VaHera` | `purpose-core` |
| `Value` | `purpose-core` |
| `Type` | `purpose-core` |
| `Operation` | `purpose-core` |
| `Domain` | `purpose-core` |
| `Resolver` trait | `purpose-core` |
| `Provider` trait | `purpose-operations` |
| `OperationRegistry::register` | `purpose-operations` |

Additions are allowed. Renames and removals are major-version events.

## Prerequisites

Install rustup (channel `stable`):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh        # unix
# or https://rustup.rs for the Windows installer
```

Verify:

```bash
cargo --version
rustc --version
```

## Build

```bash
cd mechanistic-synthesis/implementation
cargo build --release
```

The binary lands at `target/release/purpose` (or `purpose.exe` on
Windows).

## Test

```bash
cargo test --workspace
```

## CLI usage

```bash
./target/release/purpose query "Tell me about SOD1"
./target/release/purpose query "Tell me about SOD1" --raw       # JSON only
./target/release/purpose query "Tell me about SOD1" --fragment  # compiled vaHera
./target/release/purpose operations                              # registered ops
./target/release/purpose introspect                              # domain + ops JSON
```

`--raw` emits a single JSON `Value` on stdout and nothing else; it is
the wire contract consumed by the web tool (see §5 and §9.4 of the
integration contract).

## How the web tool integrates

The Next.js site at [`long-grass/`](../../long-grass) has a `/purpose`
page and an `/api/purpose` route. The API route spawns the CLI as a
subprocess with `--raw`, parses stdout as JSON, and returns it to the
page. No Rust crate is imported from the web side.

Binary resolution order used by the web API route:

1. `$PURPOSE_CLI` env var, if set and the path exists.
2. `../mechanistic-synthesis/implementation/target/release/purpose[.exe]`
3. `../mechanistic-synthesis/implementation/target/debug/purpose[.exe]`

Run the site with the release binary in path:

```bash
cargo build --release -p purpose-cli
cd ../long-grass
npm run dev
# visit http://localhost:3000/purpose
```

## What this workspace does NOT do

Per the integration contract §1.1:

* No kernel scheduler (that is `purpose-kernel`, future).
* No presentation / focus / session UI (that is `purpose-interceptor`, future).
* No training loop (that is `purpose-factory`, future).
* No model weights (loaded by providers on demand).
* No domain content (lives in external substrates — here, UniProt).
* No atomic-precision scheduling, cross-domain coordination, or
  evolutionary optimisation (those belong to Zangalewa).

Adding any of the above is a new crate against this frozen surface,
never a modification to these four crates.

## Next steps (in order of increasing coupling)

1. A second domain crate (e.g. `purpose-domains-chemistry` against
   ChEMBL/PubChem) — Path A in integration.md §3.
2. Additional provider kinds in `purpose-operations/src/providers/`
   (HuggingFace Inference, local Candle, Python subprocess, SQL) —
   Path B in §4.
3. `purpose-kernel` crate wrapping `Executor` with CMM/PSS/DIC/PVE/TEM.
4. `purpose-cascade` crate providing the k-ary `Resolver` tree.
5. `purpose-interceptor` crate owning presentation + focus + session +
   MSI.
