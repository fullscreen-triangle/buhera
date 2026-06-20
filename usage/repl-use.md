# Buhera OS — Install and Use

A practical guide. Five minutes to a working REPL on a fresh machine.

---

## What it is

A small categorical operating system. You give it text (sentences,
notes, anything). It files each piece at a *categorical address* — a
triple of numbers between 0 and 1, derived from the meaning of the
text. Later, when you ask a question, the OS computes the address of
your question and returns the stored notes whose addresses are
closest.

There is no chemistry, no domain model, no required database. The OS
runs as a single executable on Windows, Linux, or macOS, and you talk
to it through a vaHera REPL.

---

## 1. Prerequisites

You need exactly two things:

1. **Rust toolchain (stable)**. If you don't have it, install via
   [rustup](https://rustup.rs):

   - **Windows (PowerShell)**:
     ```pwsh
     winget install Rustlang.Rustup
     # or, download and run rustup-init.exe from rustup.rs
     ```
   - **Linux/macOS**:
     ```sh
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     ```

   After install, open a fresh terminal and check:
   ```sh
   cargo --version    # should print "cargo 1.7x.0 ..." or later
   rustc --version
   ```

2. **About 1 GB of disk space and ~5 minutes for the first build**.
   The dependency tree (ONNX Runtime + tokenizers) is large; subsequent
   builds are seconds.

That's it. No Python, no Docker, no separate database server, no
external API keys.

---

## 2. Get the code

If you already have a clone of the Buhera repository, skip this step.
Otherwise:

```sh
git clone https://github.com/fullscreen-triangle/buhera
cd buhera/buhera-os
```

If you're working from a local copy and don't know the path, the
workspace is `buhera-os/` at the repo root (it is a sibling of
`driven/`, `long-grass/`, and `mechanistic-synthesis/`).

---

## 3. Build the OS

From the `buhera-os/` directory:

```sh
cargo build --release
```

First run takes roughly 3–5 minutes (it compiles ONNX Runtime, the
tokenizer, and ~40 supporting crates). After that, rebuilds are 1–2
seconds.

If you only want the lexical version (no model, no ONNX Runtime, builds
in ~30 seconds), you can skip the heavy build for now by always passing
`--lexical` to the binaries (see below). The first invocation without
`--lexical` will pay the build cost lazily.

---

## 4. Run the REPL

This is the primary way to interact with the OS.

```sh
cargo run --release -p buhera-os --bin repl
```

On the very first run, the OS will download the embedding model
(~133 MB, full-precision BGE-Small-EN-v1.5; trained specifically for
retrieval and outperforms MiniLM on synonym queries) into your
fastembed cache:

- **Windows**: `%LOCALAPPDATA%\fastembed\cache\`
- **Linux**: `~/.cache/fastembed/`
- **macOS**: `~/Library/Caches/fastembed/`

Subsequent runs are instant.

You'll see a prompt:

```
buhera-os repl  (depth=12, embedder=all-MiniLM-L6-v2, objects loaded=0)
type something to search, or :help for the menu, :quit to exit
buhera>
```

### A first session

Type these one at a time:

```
store laundry  = "I need to do laundry and clean the kitchen this weekend"
store shopping = "buy milk eggs bread and coffee from the supermarket"
store run      = "go for a run on Saturday morning before it gets hot"
store flight   = "book a flight to Munich for the conference next month"
store code     = "refactor the database connection pool to use async"
```

Now ask:

```
find "shopping list"
find "morning workout"
find "database refactor"
find "trip to Germany"
list
stats
```

You'll see the OS pull `shopping` for `"shopping list"` (synonym
match via the model), `run` for `"morning workout"` (synonym match),
`code` for `"database refactor"` (literal token match boosts), and
`flight` for `"trip to Germany"` (semantic match via the model).

### REPL shortcuts

These are convenience aliases; the full vaHera syntax also works.

| Shortcut                         | Full vaHera                                       |
| -------------------------------- | ------------------------------------------------- |
| `store <name> = "<text>"`        | `memory store "<name>" = "<text>"`                |
| `find "<text>"`                  | `memory find nearest "<text>" k=3`                |
| `find "<text>" k=5`              | `memory find nearest "<text>" k=5`                |
| `list`                           | `memory list`                                     |
| `dump <name>`                    | `memory dump <name>`                              |
| `sort`                           | `demon sort`                                      |
| `stats`                          | `kernel stats`                                    |
| `trace`                          | `kernel trace`                                    |
| `procs` or `ps`                  | `process list`                                    |
| `verify`                         | `controller verify`                               |
| A bare line with no keyword      | `memory find nearest "<line>" k=3`                |

### REPL commands

| Command   | Effect                                          |
| --------- | ----------------------------------------------- |
| `:help`   | Show the full statement menu                    |
| `:tour`   | Run a short guided demo against the live kernel |
| `:clear`  | Reset the kernel to an empty state              |
| `:quit`   | Exit (also `Ctrl-D`)                            |

### REPL command-line flags

```sh
cargo run --release -p buhera-os --bin repl -- \
    --depth 12 \
    --lexical \
    --no-overlap
```

| Flag                    | Effect                                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------- |
| `--depth <N>`           | Ternary-address depth used by CMM. Default 12 (4096 cells per axis). Higher = finer.   |
| `--lexical`             | Skip the semantic model. Use the deterministic hash-bag embedder. Faster, no download. |
| `--no-overlap`          | Disable the token-overlap re-ranking pass.                                             |
| `--data <PATH>`         | Pre-load a JSON compound/object database (see `data/nist_compounds.json` for shape).   |

---

## 5. Run the demo (non-interactive)

A scripted version that runs a vaHera program from a file and exits:

```sh
cargo run --release -p buhera-os --bin demo
```

That uses a built-in plain-text script. Or:

```sh
cargo run --release -p buhera-os --bin demo -- --script examples/notes-wide.bvh
```

The demo accepts the same `--lexical` and `--no-overlap` flags.

### Writing your own vaHera script

Create a file `myscript.bvh`:

```
# Anything after '#' on a line is a comment.

memory store "morning"  = "wake at 6, shower, coffee, check email"
memory store "evening"  = "dinner with the team, then early to bed"
memory store "project"  = "finish the proposal draft and send to Sarah"

memory find nearest "writing tasks" k=3
memory find nearest "what to do tonight" k=3

memory list
kernel stats
```

Run it:

```sh
cargo run --release -p buhera-os --bin demo -- --script myscript.bvh
```

---

## 6. The full vaHera language

Every statement that the REPL accepts.

| Statement                                | Effect                                                                |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `describe <name> with "<text>"`          | Bind a name to the categorical coord of the text.                     |
| `resolve <name>`                         | Look up (or compute) the coord for `<name>`.                          |
| `spawn <program> from <name>`            | Create a categorical process targeting `<name>`'s coord.              |
| `navigate to penultimate`                | Backward-navigate the active process to the penultimate state.        |
| `complete trajectory`                    | Apply the completion morphism from penultimate to final.              |
| `memory create at S(<k>,<t>,<e>)`        | Allocate an empty object at the explicit coord.                       |
| `memory store "<name>" = "<text>"`       | Store `<text>` at its content coord, tagged `<name>`.                 |
| `memory find nearest "<text>" k=<n>`     | Return the `n` stored objects closest to `<text>`'s coord.            |
| `memory list`                            | List every stored object.                                             |
| `memory dump <name>`                     | Show the full payload + coord + address of one object.                |
| `demon sort`                             | Zero-cost categorical sort by S-distance to the origin.               |
| `controller verify`                      | Triple-equivalence diagnostics (TEM).                                 |
| `kernel stats`                           | Per-subsystem statistics (CMM, PSS, DIC, PVE, TEM).                   |
| `kernel trace`                           | Activity log across all subsystems.                                   |
| `process list`                           | List spawned processes with their states.                             |

---

## 7. Running the tests

```sh
cargo test --release
```

You should see ~58 tests pass across `buhera-substrate`, `buhera-kernel`,
`buhera-vahera`, `buhera-embed`, and `buhera-os`.

---

## 8. Troubleshooting

### "cargo: command not found"

The Rust toolchain isn't on your PATH. Open a fresh terminal after
installing rustup. On Windows PowerShell, you may need to log out and
back in, or do `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.

### Build fails with `-C lto` errors on first build

If your clone of the Buhera repository has an old
`.cargo/config.toml` at the root that forces `lto=fat`, modern rustc
will reject it. The `buhera-os/` workspace is shipped with a working
config; if you've copied stale settings, replace
`buhera/.cargo/config.toml` with a minimal version containing only
`[build] target-dir = "target"`.

### Semantic embedder fails at runtime

If the model download fails (corporate firewall, no internet), the OS
falls back automatically to the lexical embedder. You'll see a line
like `(semantic embedder failed: ...; falling back to lexical)`. You
can also force lexical mode with `--lexical`.

### "every query returns the same notes"

Two causes. Either you're using `--lexical` on very short queries (the
hash-bag embedding lacks signal), or you have only a handful of notes
all dominated by the same few stopwords. Try:

- Run without `--lexical`.
- Store longer, more distinctive notes.
- Use queries that share at least one word with the target notes
  (this triggers the overlap re-ranker).

### REPL won't rebuild because `repl.exe` is locked

If you have an existing REPL session running, exit it (`:quit`) before
running `cargo build` or `cargo run` again. Windows holds the
executable lock as long as it's executing.

---

## 9. What the OS does under the hood

When you `store note = "..."`:

1. The text goes to the embedder (semantic model by default).
2. The model produces a 384-D vector; a deterministic projection maps
   it to `(S_k, S_t, S_e) ∈ [0,1]³`.
3. CMM (Categorical Memory Manager) computes a 12-character ternary
   address from the coord, classifies the tier (L1–RAM by `||S||₂`),
   and stores the payload.
4. TEM (Triple Equivalence Monitor) records a sample at this coord.
5. PVE (Proof Validation Engine) verifies the statement type.

When you `find "..."`:

1. The query embeds to a query coord.
2. CMM does a proximity scan (Fisher-metric S-distance).
3. DIC (Demon I/O Controller) performs surgical retrieval of the top-k.
4. The OS applies the token-overlap re-ranker (unless `--no-overlap`).
5. The result is printed with each hit's name, address, and distance.

The whole loop is deterministic given the inputs. Two identical
sessions on two machines produce identical outputs.

---

## 10. Beyond the REPL

The OS is a library too. The four crates are:

- `buhera-substrate` — S-entropy coords, Fisher metric, ternary
  addresses, backward navigation. No external deps.
- `buhera-kernel` — CMM, PSS, DIC, PVE, TEM, and the orchestrator.
- `buhera-vahera` — parser and interpreter for the 15 vaHera statements.
- `buhera-embed` — semantic and lexical text embedders.
- `buhera-os` — the `demo` and `repl` binaries plus shared helpers.

If you want to embed Buhera into your own Rust application, depend on
`buhera-kernel` and `buhera-vahera`, plug in a `TextEmbedder` from
`buhera-embed`, and call `execute_vahera_with(source, &mut kernel,
&molecules, &adapter)`. Everything else is internal.

For a no-install browser version that you can show to someone in 30
seconds, see [`webtool.md`](./webtool.md). It's a JavaScript
implementation of the same kernel and vaHera — same semantics, no
Rust toolchain required — meant for first-look demos before someone
commits to the desktop install.
