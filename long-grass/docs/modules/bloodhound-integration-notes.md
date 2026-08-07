# Bloodhound Integration Notes

**Status:** Notes (companion to `cross-repo-confluence-design.md`)
**Date:** 2026-07-27
**Scope:** Two concrete integration questions, answered against code that already
exists in the federation — not hypotheticals.

1. How the thrust web tool (Repo Lens / the confluence cockpit) can be embedded
   **inside Buhera OS web**.
2. How **any other repo** can consume the Bloodhound Rust CLI(s).

Both answers turn out to be the *same* pattern, and — importantly — **that pattern is
already implemented and proven in this codebase** by `@buhera/spraypaint`. These notes
therefore read less as "here is a design to build" and more as "here is the existing
seam to reuse, applied twice."

---

## 0. The one pattern behind both answers: binary-backed bindings with a swappable Runner

`buhera-os/spraypaint-ts` (`@buhera/spraypaint`) is a TypeScript package that binds a
Rust CLI (`spraypaint`) to a web/React UI. Its defining decision is stated in its own
source:

> "The client never simulates results. Every method shells out to the installed
> `spraypaint` executable and returns the parsed JSON… The binary may be local (Node
> `child_process`) or behind a service (HTTP). `SpraypaintClient` depends only on a
> `Runner`, so the same client code works in a Next.js route handler, a CLI, or a test
> with a fake runner."

That is *exactly* the deployment shape the confluence design settled on in §7.6
(downloaded local native engine ↔ static tab, no hosted backend). The seam is a
single interface:

```ts
interface Runner {              // spraypaint calls it SpraypaintRunner
  run(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }>;
}
```

- **`NodeRunner`** (in a separate `runner-node.ts` so the rest of the package imports
  no Node built-ins) executes the real binary via `child_process.execFile`.
- A **browser build** imports *no* Node built-ins; it talks to a `Runner` that is an
  HTTP client hitting a route (or the localhost engine) which *itself* holds a
  `NodeRunner`.

So "embed the web tool in Buhera OS web" and "let another repo use the Rust CLI" are
both: **write a thin client that emits CLI args and parses `--json` output, behind a
`Runner` you can point at `child_process` or HTTP.** Everything below is that, made
specific.

**Design rule carried over:** the client decides *what* to run and *how to interpret
it*; the Runner decides *where* it runs. Never fold the two together — that is what
keeps the same code usable in a Next.js route, a local engine, and a test.

---

## 1. Embedding the thrust web tool inside Buhera OS web

### 1.1 What "the web tool" actually is

The thrust confluence cockpit is the `repo-lens` page (`thrust/src/pages/repo-lens.js`)
plus its component set (`SalientTreemap`, `FragmentGraph`, the notebook cells) and one
serverless route, `thrust/src/pages/api/repo-lens/ai.js` (the AI proxy that keeps
provider keys server-side). It is a Next.js **pages-router** surface: JS pages,
Tailwind, `@/*` alias, d3 + framer-motion.

Two honest facts shape how it embeds:

- The **cockpit UI is portable** — it is React + d3; it can live anywhere React runs.
- The **data sources are not part of the UI** — today the page fetches
  `api.github.com` from the browser and the AI proxy from `/api/repo-lens/ai`.
  Tomorrow (confluence §7.6) it fetches a **localhost engine**. Embedding must decide
  *which* data sources come along.

### 1.2 Three embedding options, cheapest to deepest

**Option A — iframe / linked route (fastest, weakest coupling).**
Deploy thrust as-is; Buhera OS web links to it or frames it. Zero shared code. Use this
only as a stopgap: an iframe cannot share Buhera's `AskQuery`/session state, and the
two toolbars will feel like two apps. Reject for anything but a demo.

**Option B — extract the cockpit into a shared React package (recommended).**
Mirror what `@buhera/spraypaint` already does for `spraypaint`: pull the confluence
cockpit out of the thrust pages tree into a **framework-agnostic React package** (call
it `@bloodhound/confluence-ui`) exporting the notebook + cells as components that take
their data via props/a client, not via hardcoded `fetch`. Then:

- **thrust** imports it and wires the client to browser-`fetch`/its AI route (its
  current behaviour, unchanged for the standalone site).
- **Buhera OS web** imports the *same package* and wires the client to *its own*
  runner — which is the localhost engine, or a Buhera route handler holding a
  `NodeRunner`.

This is the spraypaint architecture (`./react` export + a client that depends only on a
Runner) applied to the confluence cockpit. The UI is written once; each host supplies
the data seam.

```
@bloodhound/confluence-ui   (React components: notebook, cells, D3 charts)
        │ props: { client, onDispatch, theme }
        ├── used by thrust site      → client = browser fetch + /api/repo-lens/ai
        └── used by buhera-os web    → client = localhost engine (or buhera route)
```

**Option C — a Buhera-native cell (deepest, most "one system").**
Buhera OS web already has a canonical runnable artifact — `AskQuery` mapping 1:1 to
`spraypaint ask …`, with charts that invert gestures back into query diffs. The
confluence cockpit could become **another cell type in that same surface**: a
confluence query is a runnable artifact peer to `AskQuery`, its correspondence-map /
pipeline / regime-map charts sit beside spraypaint's water-filling charts, and both
write to one session/undo stack. This is the truest expression of "N repos as one
library" — the two tools stop being two tools. It is also the most work, because it
requires reconciling confluence's client with Buhera's session/crossfilter loop.

**Recommendation:** build Option B now (it is the reusable substrate and it is exactly
the proven spraypaint shape), and treat Option C as the destination once the confluence
client is stable — the extra work in C is *integration into Buhera's session model*,
which only pays off after the cockpit's own data seam exists (B).

### 1.3 The concrete extraction checklist (Option B)

1. **Make the cockpit take a `client` prop.** Everywhere `repo-lens.js` calls
   `fetch('https://api.github.com/…')` or `analyseFederation(...)`, route it through a
   `ConfluenceClient` object (same role as `SpraypaintClient`). The component imports no
   transport; it calls `client.analyse(...)`, `client.correspond(...)`, etc.
2. **Split transport into a Runner**, exactly as spraypaint splits `runner-node.ts`:
   - `runner-fetch` (browser): hits GitHub / the AI route / the localhost engine.
   - `runner-node` (Node): `child_process` to `purpose`/`tracker`/`wt` for the local
     engine or a route handler.
   The cockpit bundle imports **neither** directly — the host injects one.
3. **Theme via tokens, not hardcoded hex.** thrust's palette (primary `#2A9D8F`, accent
   `#F4A261`, danger `#E63946`, dark `#0a0a0f`, surface `#16161f`, muted `#8888aa`)
   becomes a `theme` prop / CSS variables so Buhera OS web can restyle without forking.
4. **The AI proxy is a host responsibility.** `/api/repo-lens/ai` keeps provider keys
   server-side. When embedded in Buhera OS web, Buhera supplies the equivalent route (or
   the localhost engine's `/ai` endpoint). The extracted package must **not** carry a
   key — it names a client method (`client.generate(...)`) and the host wires the
   endpoint. (Same honesty split as confluence §7.5: dispatch/data can be local, but the
   model endpoint stays where the key legitimately lives.)
5. **Keep the pages-router site working.** thrust keeps `repo-lens.js` as a thin page
   that mounts the package with the browser client — so extracting the package does not
   break the standalone site.

### 1.4 Honesty flags (embedding)

- **Two routers.** thrust is pages-router; if Buhera OS web is app-router or a non-Next
  React app, the *package* must be router-agnostic (plain components). Do not export a
  Next `page` — export components. (Option B already requires this.)
- **d3 + framer-motion are peer deps**, mirror spraypaint's `peerDependenciesMeta`
  (d3/react optional-peer pattern) so the host controls versions and the cockpit does
  not double-bundle d3.
- **The GitHub-token story must survive the move.** thrust promises the token "stays in
  your browser; sent only to api.github.com." Inside Buhera OS web with a localhost
  engine, the token instead lives in the *engine* (§7.6). Either is fine; what is *not*
  fine is silently routing the token through a Buhera-hosted server. The embedding must
  preserve one of the two honest homes (browser-only, or local engine).

---

## 2. How any other repo can use the Bloodhound Rust CLI

"The Bloodhound Rust CLI" here means the confluence/federation binaries — `purpose`
(installed, v0.1.0), the federation `tracker`, `wt` (wind-tunnel), and the forthcoming
local engine — any Rust executable that emits `--json`. The question "how can another
repo use it" has **four distinct answers by consumer type**, and the right one depends
on who the consumer is.

### 2.1 Consumer is a shell / CI / a human — just call the binary

The lowest tier and the one to design for first. Every binary follows the `purpose`
contract: installable per-project (`cargo install --path <crate>`), single-repo with a
`--root` to target others, `--json` for machine output.

```sh
cargo install --path purpose            # or: cargo install purpose
purpose index --root ../lavoisier       # build that repo's self-graph
purpose ask "peak alignment" --root ../lavoisier --json
```

Another repo "uses" the CLI by adding it as a **dev/tooling dependency documented in its
README** and calling it in scripts or CI. No coupling to Bloodhound's internals; the
only contract is argv + `--json` shape. This is already how the tracker consumes
`purpose` (it *shells out*, reimplements nothing — tracker design §5.1).

### 2.2 Consumer is a TypeScript / web project — ship binary-backed bindings

If the consuming repo has a web UI, it should **not** re-parse argv by hand. It should
depend on a bindings package modelled on `@buhera/spraypaint`:

```
@bloodhound/purpose-ts        (or a unified @bloodhound/cli-ts)
  ├── types.ts        AskQuery-equivalent + queryToArgs() + result types
  ├── client.ts       Client(runner, opts) — decides what to run, parses --json
  ├── runner-node.ts  NodeRunner — child_process.execFile (Node only)
  ├── runner-fetch    HTTP runner for browser → localhost engine / route
  └── react/          optional React hooks/components (peer-dep react, d3)
```

The consuming repo picks a runner:

```ts
import { PurposeClient } from "@bloodhound/purpose-ts";
import { NodeRunner } from "@bloodhound/purpose-ts/runner-node";

// In a Next.js route handler (Node): shell to the real binary.
const client = new PurposeClient(new NodeRunner({ bin: "purpose" }), { root });
const result = await client.ask("peak alignment");
```

```ts
// In the browser: same client, HTTP runner pointing at the local engine.
const client = new PurposeClient(new FetchRunner("http://localhost:7423"), { root });
```

The **client code is identical**; only the Runner changes. That is the entire benefit,
and it is copy-the-spraypaint-package work, not new invention.

### 2.3 Consumer is another Rust crate — depend on the library, not the binary

If the consumer is Rust, a binary is the wrong seam. Each tool should be a **workspace
with a `lib` crate + a thin `bin` crate** (the pattern buhera-os already uses:
`buhera-substrate`/`-kernel`/`-vahera` libs, `buhera-os` the binary). Then another Rust
repo adds a path/git dependency on the *lib* and calls the API directly — no subprocess,
no JSON round-trip:

```toml
# consumer Cargo.toml
[dependencies]
purpose-core = { git = "https://github.com/…/bloodhound", package = "purpose-core" }
```

Prefer this whenever the consumer is Rust and latency/typing matter (e.g. the local
engine embedding `purpose`'s indexer in-process rather than shelling to it). Keep the
binary as the *interop* seam (§2.1/§2.2); keep the lib as the *Rust-native* seam.

### 2.4 Consumer is the confluence local engine — the aggregator case

The confluence engine (§7.6) is itself "another repo using the CLI," and it is the
motivating one. It consumes the CLIs in **both** modes above:

- **Rust-native (§2.3)** for the hot paths it wants in-process — indexing, χ min-cut.
- **Subprocess (§2.1)** for tools it treats as black boxes — `wt`, `gh`.

It then **re-exposes** all of it over a localhost HTTP API that the web tools (§1, §2.2)
consume via a Fetch runner. So the engine is simultaneously a *consumer* of the CLIs and
a *Runner provider* for the browser. This is the keystone that makes §1 and §2 the same
story: the engine is where "another repo uses the CLI" and "the web tool needs a
backend-that-isn't-hosted" meet.

### 2.5 Contract every consumer can rely on (and the honest limits)

What a consuming repo may depend on:

- **argv + `--json` output shape**, versioned. This is the stable interface. (spraypaint
  makes each subcommand's JSON a typed result; do the same for `purpose`/`tracker`/`wt`.)
- **Exit codes as verdicts**, not just success/failure — spraypaint's runner *resolves*
  (never rejects) on nonzero exit because some subcommands (`verify`) exit nonzero as a
  legitimate answer. A consumer must inspect `exitCode`, not assume nonzero = crash.
- **`--root` to target a repo other than cwd**; **`index` before `ask`** (the
  construction/commit phase split — tracker I3/I4).

Honest limits a consumer must design around (verified against `purpose` today):

- **`purpose` ranking is raw substring** (+3 name, +1 snippet, top-20; no idf, no config,
  no synonym map). A consumer needing recall must issue **distinctive multi-term
  queries** and merge several `ask` calls — do not expect one query to be complete.
- **No config surface** on `purpose` (no `.purpose/config.toml`, no env). A consumer
  cannot tune it; it adapts around it.
- **The index is a snapshot.** A consumer owns staleness detection: if git HEAD moved,
  re-`index` before trusting `ask` (search-not-fetch freshness, tracker I3).
- **network-yield / a hosted executor is not assumed.** Every consumer degrades to
  read-only search + tracking when execution organs are absent — the same
  graceful-degradation discipline the tracker and confluence both hold.

---

## 3. Summary — one seam, two questions

| Question | Answer | Proven by |
|---|---|---|
| Embed the web tool in Buhera OS web | Extract the cockpit into a router-agnostic React package taking a `client` prop (Option B); optionally make it a Buhera-native cell (Option C). | `@buhera/spraypaint`'s `./react` export + client-depends-only-on-Runner |
| Let another repo use the Rust CLI | Shell to the binary + `--json` (shell/CI); ship binary-backed TS bindings with a swappable Runner (web); depend on the lib crate (Rust); the local engine does all three and re-exposes over localhost. | `@buhera/spraypaint` (TS bindings), buhera-os lib/bin split (Rust), tracker→purpose (shell-out) |

The single unifying fact: **Bloodhound's tools are binaries that emit `--json`, and the
consumer supplies a Runner that decides where they run.** Buhera OS already demonstrates
this end-to-end with `spraypaint`. Both integrations are that same pattern, pointed at
the confluence CLIs.
