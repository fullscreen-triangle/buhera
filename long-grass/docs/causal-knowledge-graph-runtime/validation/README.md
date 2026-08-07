# Validation suite

Executable models that empirically check the five load-bearing claims of
*A Runtime Without a Program: Execution as Propagation over a Causal Knowledge
Graph*. Each validator asserts its claim (non-zero exit on failure) and writes
one figure panel into `../figures/`.

Run everything:

```bash
python validation/run_all.py
```

Requires only `matplotlib` (tested with 3.10 on CPython 3.14).

## Files

| File | Claim (paper) | What it shows |
|---|---|---|
| `ckg_runtime.py` | — | Shared substrate: `Node` (task+chunks+values), `Runtime.dispatch` (runs *all* chunks, judges nothing, no exit code), `Module` (read/transform/emit), `provenance_fingerprint` (hashes the protocol, never the values). Mirrors the shape of `long-grass/src/lib/modules/registry.js`. |
| `trajectory_emergence.py` | Thm. 5, Prop. 1 | Two runs over one fixed node set induce different causal edge relations; their symmetric difference is non-empty, so no static schedule reproduces both. |
| `run_to_completion.py` | Thm. 2, Cor. 4, Def. 3 | A chunk that raises becomes a recorded `error` value; the run continues past it; the runtime exposes no `exit_code`/`ok`. Also: a node with two chunks (shapeshifter + graffiti) runs both. |
| `nondeterminism_provenance.py` | Prop. 8 | 400 runs of one imported setup spread over result values while sharing a single provenance fingerprint — reproducibility is of the protocol, not the numbers. |
| `resolution_control.py` | Def. 6, Prop. 7 | Editing three full-depth addresses changes exactly those objects; the other nine are byte-identical. Coarse import and surgical redesign are one address space at two depths. |

## Design note

`ckg_runtime.py` contains no comparison-to-expectation anywhere — no `==` against
a "correct" result, no verdict, no exit code. That inertia is deliberate: the
paper's theorems are properties of a runtime that judges nothing, so the
substrate the validators run on must judge nothing either. The `assert`s live in
the *validators* (which play the role of downstream consumers holding
expectations of their own), never in the runtime.
