# Buhera OS

A standalone categorical operating system. Boots in milliseconds, allocates
content at categorical addresses in S-entropy space, navigates trajectories
backward to penultimate states, and synthesizes answers without storing
property tables.

This workspace has **no dependencies** on `mechanistic-synthesis/`,
`long-grass/`, the legacy `src/`, or any external project. It is a
self-contained OS.

## Layout

```
buhera-os/
├── crates/
│   ├── buhera-substrate/   # S-entropy coords, Fisher metric, ternary
│   │                       # addresses, backward navigation, embeddings.
│   ├── buhera-kernel/      # Five subsystems (CMM/PSS/DIC/PVE/TEM) +
│   │                       # Kernel orchestrator.
│   ├── buhera-vahera/      # vaHera parser + interpreter (15 statement types).
│   └── buhera-os/          # Binary crate: `demo` and `repl`.
├── data/
│   └── nist_compounds.json # Sample compound database for the demo.
└── examples/
    └── demo.bvh            # vaHera source demonstrating end-to-end use.
```

## Quick start

```sh
# Build everything.
cargo build --release

# Run the NIST demo end-to-end.
cargo run --release -p buhera-os --bin demo

# Open an interactive vaHera REPL against a live kernel.
cargo run --release -p buhera-os --bin repl

# Execute a vaHera source file directly.
cargo run --release -p buhera-os --bin demo -- examples/demo.bvh
```

## vaHera statements (v0.1.0)

| Statement                                  | Effect                                   |
|--------------------------------------------|------------------------------------------|
| `describe <name> with "<text>"`            | Bind text content to a categorical coord |
| `resolve <name>`                           | Compute / look up the coord for `<name>` |
| `spawn <program> from <name>`              | Create a categorical process to `<name>` |
| `navigate to penultimate`                  | Backward-navigate to the penultimate     |
| `complete trajectory`                      | Apply the completion morphism            |
| `memory create at S(<k>,<t>,<e>)`          | Allocate at an explicit coord            |
| `memory store "<name>" = "<text>"`         | Store text at its content coord          |
| `memory find nearest "<text>" k=<n>`       | Categorical retrieval                    |
| `memory list`                              | List all allocated objects               |
| `memory dump <name>`                       | Print payload + coord + address of one   |
| `demon sort`                               | Zero-cost categorical sort               |
| `controller verify`                        | Triple-equivalence diagnostics           |
| `kernel stats`                             | Per-subsystem statistics                 |
| `kernel trace`                             | Activity log                             |
| `process list`                             | All spawned processes and their states   |

## Theoretical grounding

The S-entropy substrate, the five subsystems, and the vaHera language are
each grounded in the Buhera framework's published theorems:

- **Bounded phase space → oscillatory necessity → partition coordinates
  `(n, ℓ, m, s)` with capacity `C(n) = 2n²`** drives the categorical
  addressing scheme.
- **Backward trajectory completion** terminates at the *penultimate state*,
  one isomorphism step short of the endpoint, because that gap is the
  structural requirement for recognition.
- **The triple equivalence** (oscillatory ≡ categorical ≡ partition) is
  monitored at every kernel tick by TEM.

The code here is a direct, faithful Rust port of the Python OS reference
under `../driven/system/`; the Python remains as a regression oracle.

## License

MIT.
