# Buhera OS — Terminal Tutorials

This is a hands-on guide to the current state of Buhera. Everything here
you can type into the terminal at [long-grass.vercel.app](https://long-grass.vercel.app)
(or your local `npm run dev`) and see it work. No installation required
beyond having the page open.

Buhera is a small operating system for finite reasoning agents. It has
seven registered modules ("the federation"), a natural-language
orchestrator (turbulance), a memory kernel with a ternary substrate, and
an append-only audit log. Every act flows through the same dispatch
mechanism.

The tutorial goes from the very basics to real workflows in seven
tracks. You do not need to follow them in order after Track 0, but each
assumes the terminal is in a fresh state (`:clear` resets vahera, and
`dispatch("purpose-carry", { kind: "reset" })` resets the purpose
session).

---

## Track 0 — Orientation

Before doing anything else, learn where you are.

### 0.1 See the federation

```
:modules
```

You should see seven registered modules. In cascade priority order they
are:

| Module | Purpose |
|--------|---------|
| `vahera` | memory / recall DSL with a ternary kernel |
| `echo` | smoke-test module — returns whatever you send |
| `lavoisier` | virtual mass-spec forward simulator |
| `purpose` | federated LLM synthesis (needs API key) |
| `zangalewa` | Minimum Sufficient Interceptor (needs API key) |
| `graffiti` | individuation-theoretic search DSL |
| `purpose-carry` | tandem carry — minimum-sufficient context for a goal |

### 0.2 See the audit log

```
:audit
```

An empty log is shown as `(audit log is empty)`. Every act you dispatch
will be appended here with a monotone act ID.

### 0.3 Check available LLM providers

Two modules (`purpose` and `zangalewa`) and one of graffiti's catalysts
call out to LLM APIs. To see which providers the server has keys for,
paste this URL into your browser (change the host if running locally):

```
https://long-grass.vercel.app/api/providers
```

You'll get a JSON response like:

```json
{
  "ok": true,
  "available": ["gemini", "huggingface", "openai"],
  "cascade_order": ["ollama", "gemini", "huggingface", "openai"],
  "active": "gemini"
}
```

`active` is the provider the LLM cascade will try first. `available` is
the list, in cascade priority order, of every provider currently
configured on this deployment.

### 0.4 Read the welcome banner

The terminal boots with a WELCOME message that summarises the syntax.
If you didn't read it, scroll up. Every example below builds on it.

---

## Track 1 — vaHera: memory and recall

vaHera is the memory subsystem. It stores notes at semantically-derived
addresses in a ternary S-entropy substrate and retrieves them by
proximity to a query. Everything vaHera does is deterministic — no LLM
in the loop.

### 1.1 Store your first notes

Type these one line at a time:

```
memory store "meeting" = "team retro on Thursday at 2pm about the Q3 launch"
memory store "recipe" = "bread flour, salt, water, yeast — 18-hour cold retard"
memory store "todo" = "buy milk and bread from the corner shop"
```

Each store operation returns a small confirmation card.

### 1.2 List what you've stored

```
list
```

You'll see the three notes with their computed S-coordinates.

### 1.3 Find by semantic proximity

```
memory find nearest "when is the meeting"
```

You should get back the `meeting` note ranked first, because "meeting"
and "team retro" share semantic content that lands close in the
S-entropy space.

Try:

```
memory find nearest "sourdough baking"
```

You should get `recipe` first. Notice: vaHera doesn't do exact keyword
matching — it embeds the query, computes S-distance to every stored
object, and ranks by proximity.

### 1.4 Inspect one object

```
dump meeting
```

Shows the full stored object: coordinates, ternary address, tier,
payload.

### 1.5 Kernel diagnostics

```
kernel stats
```

Shows how many objects live in each tier of the substrate, PVE
verification counts, and other bookkeeping.

```
kernel trace
```

Shows the sequence of internal events the kernel has emitted since
boot. Useful when you want to understand what's happening under the
hood.

### 1.6 A guided tour

If you want vaHera to load a demo corpus for you:

```
:tour
```

This stores five sample notes and demonstrates find operations against
them.

### 1.7 Reset

```
:clear
```

Wipes the kernel back to empty. Useful before starting a new track.

---

## Track 2 — Turbulance: orchestrating dispatches

Turbulance (also called kwasa-kwasa) is the orchestrator language.
Anything you can type as a single vaHera statement, you can also embed
inside a turbulance script that does more.

### 2.1 A one-line turbulance script

Type this into the terminal — the terminal detects turbulance syntax
by the leading keyword and routes accordingly:

```
funxn double(x): return x * 2
```

Now:

```
item r = double(21)
print("result: {}", r)
```

### 2.2 Propositions

```
proposition Greeting: motion Hello("world")
```

Propositions carry claims through a computation. See the graffiti track
for how these tie into the search calculus.

### 2.3 Dispatching to a module from turbulance

The load-bearing feature: turbulance can call any module in the
federation.

```
item out = dispatch("echo", "hello federation")
print(out.output_delta.value)
```

You should see `hello federation` printed back. Behind the scenes:

1. Turbulance evaluated `dispatch(...)` as a builtin
2. The module registry looked up `echo` and called its `execute()`
3. Echo returned an ActResult; the registry logged it
4. Turbulance handed the result back to your script

Look at `:audit` — there's a new entry.

### 2.4 Route vaHera through turbulance

```
item stored = dispatch("vahera", "memory store \"note\" = \"hello\"")
item found = dispatch("vahera", "memory find nearest \"hello\" k=1")
```

Same operation as typing the vaHera statements directly, but now they
compose in a script.

---

## Track 3 — Lavoisier: virtual mass spectrometry

Lavoisier is a forward-simulation mass spectrometer. You describe an
analyte class (lipidomics or proteomics), ionisation conditions, and an
analyser, and it synthesises the predicted spectra.

### 3.1 The demo run

```
item ms = dispatch("lavoisier", "demo")
print("records: {}", ms.output_delta.summary.count)
```

The demo runs a PC (phosphatidylcholines) lipidomics experiment on an
orbitrap in positive-ion mode, chain range 30:0 – 38:4, m/z window
400–1000. You'll see a summary card in the terminal — top-hit records,
m/z range, intensity range, per-class breakdown, principal-shell
histogram. Click "show N records" to expand.

### 3.2 A custom run

```
item ms = dispatch("lavoisier", {
  kind: "virtual_run",
  experimentType: "lipidomics",
  classSpecs: [
    { classKey: "PE", Xmin: 32, Xmax: 40, Ymin: 0, Ymax: 6 }
  ],
  polarity: "-",
  analyser: "tof",
  collisionEnergy_eV: 40,
  mzWindow: [200, 1400]
})
```

Same shape as the demo, but you supply the config. PE
(phosphatidylethanolamines) in negative mode on a TOF at high collision
energy.

### 3.3 Proteomics

```
item ms = dispatch("lavoisier", {
  kind: "virtual_run",
  experimentType: "proteomics",
  proteinSpecs: [{ classKey: "tryptic_short" }],
  polarity: "+",
  analyser: "orbitrap"
})
```

Switches the analyte enumeration from lipids to tryptic peptides.
Everything downstream (adduct chemistry, m/z window, MS² fragmentation)
follows automatically.

### 3.4 What lavoisier does not do

- It does not fetch real mzML files. That's a separate instruction kind
  (`parse_mzml`) that will land later.
- It does not search a real spectral library. The output is the
  forward-simulated prediction, not a match against a database.
- It has no LLM in the loop and never calls the network.

---

## Track 4 — Graffiti: individuation-theoretic search

Graffiti implements the `.grf` search calculus. A `seek` statement is
the primitive: it individuates a claim from an unbounded medium by
combining catalysts under a coherence condition.

### 4.1 The demo seek

```
item g = dispatch("graffiti", "demo")
print("floor: {}", g.output_delta.ambient_floor)
```

The demo runs a single seek that resolves `founding_year_of(TUM)`
against a fixture catalyst. The result renders as a card showing the
ambient floor β and the yielded claims.

### 4.2 Write your own .grf script

Graffiti scripts are ordinary strings — you can pass one as the
instruction directly.

```
item g = dispatch("graffiti", `
  floor 0.02

  catalyst local_search {
    namespace: local
    input: Region output: Claim
  }

  project apollo {
    seek year
      not { "disputed dates" }
      toward { launch_date_of("Apollo 11") }
      via { local_search(year) }
      until converge
      yield year
  }
`)
```

The registered catalysts in the graffiti module are:

| Catalyst | Namespace | Power | Backing |
|----------|-----------|-------|---------|
| `local_search` | local | 0.7 | in-memory fixture table |
| `kernel_search` | local | up to 0.9 | vaHera's live kernel via S-distance |
| `hf_inference` | inference | 0.6 | HuggingFace chat (via `/api/hf-inference`) |
| `restate` | inference | 0.5 | deterministic mock transform |

### 4.3 Kernel-backed search

This is the interesting composition. Anything you store in vaHera
becomes searchable through graffiti's `kernel_search` catalyst:

```
memory store "tum" = "Technical University of Munich, founded 1868"
memory store "aime" = "AIMe Registry for Artificial Intelligence"

item g = dispatch("graffiti", `
  floor 0.02
  catalyst kernel_search { namespace: local input: Region output: Claim }
  project P {
    seek fact
      not { "hearsay" }
      toward { "founding year TUM" }
      via { kernel_search(fact) }
      until converge
      yield fact
  }
`)
print(g.output_delta.projects.P.fact)
```

One kernel serves both modules. Storage in vaHera is retrieval-corpus
for graffiti.

### 4.4 Multi-catalyst coherence

The paper proves you need at least three mutually reinforcing catalysts
to ground a claim against a dissenting source. That looks like:

```
item g = dispatch("graffiti", `
  floor 0.02
  catalyst local_search { namespace: local input: Region output: Claim }
  catalyst kernel_search { namespace: local input: Region output: Claim }
  catalyst hf_inference { namespace: inference input: Region output: Claim }

  project triangle {
    seek verified
      not { "single-source claims" }
      toward { "when was TUM founded" }
      via {
        local_search(verified)
        >> kernel_search(verified)
        >> hf_inference(verified)
      }
      until converge otherwise decline
      yield verified
  }
`)
```

Three catalysts spanning two namespaces (`local`, `inference`). If any
one dissents, `until converge otherwise decline` will return an honest
"decline" rather than fabricating a consensus.

---

## Track 5 — Zangalewa & Purpose: LLM-backed modules

These two modules call an LLM. They need an API key set on the
deployment (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `HUGGINGFACE_API_KEY`
in `.env.local` locally or in Vercel environment variables in
production). If no key is set, both modules return a clear "no
provider configured" ActResult.

To check what's configured before you dispatch:

```
GET /api/providers
```

### 5.1 Zangalewa — the interceptor

Zangalewa is the Minimum Sufficient Interceptor: it takes a
natural-language question and returns an S-coordinate plus a
fully-rendered research card. One call, structured output, no chatter.

```
item z = dispatch("zangalewa", "what is p53?")
print(z.output_delta.title)
```

The rendered card shows:
- Title (with type/class annotation)
- S-coordinate `(S_k, S_t, S_e)` on `[0,1]^3`
- 3–5 sections (function, structure, mechanism, clinical, …)
- Up to 3 references with clickable URLs
- Which provider actually answered the call

Try a more specific query:

```
dispatch("zangalewa", "how does CRISPR-Cas9 double-strand break repair work")
```

Or something that stresses the S-coord estimation:

```
dispatch("zangalewa", "explain quantum entanglement to a 10-year-old")
```

The last one should return a card with high `S_e` (multi-step
explanation required) and moderate `S_k` (broad concept).

### 5.2 Purpose — federated synthesis

Purpose runs a knapsack-allocated federated cascade to produce a
concise synthesis document. It picks up relevant knowledge packs
(server-side files bundled with the codebase) as context, then calls
the LLM cascade.

```
item p = dispatch("purpose", "compare the composition-inflation theorem to a Turing machine's state space")
print(p.output_delta.synthesis)
```

The result renders as a synthesis card with the model, provider,
aggregate floor β from the federation, and the synthesis text.

Followups:

```
dispatch("purpose", {
  kind: "synthesise",
  description: "explain kwasa-kwasa's dispatch model",
  followups: ["how does it interact with the module registry?"]
})
```

### 5.3 What zangalewa and purpose do NOT do

- Neither has memory between calls. Each dispatch is independent.
- Neither writes to vaHera or to the purpose-carry session on its own —
  though every dispatch is logged in the audit trail.
- Neither performs multi-turn conversation. If you need context, you
  build it into the instruction.

That last point is what the next track is for.

---

## Track 6 — Purpose-Carry: the tandem carry

This is the newest module and probably the most conceptually different.
Purpose-carry implements the tandem carry from the paper "Carry the
Uncertainty, Not the Knowledge". Its job is to look at the accumulated
audit log, and given a goal, tell you the minimum set of past acts you
need to carry forward as context.

Every act you dispatch through the registry automatically becomes a
`Step` in the purpose session (via a post-dispatch hook). Purpose-carry
then computes the tandem: seek → necessary → knapsack → partition.

### 6.1 See how many steps have accumulated

```
dispatch("purpose-carry", { kind: "stats" })
```

Returns the current step count and the ambient floor β of the graph
derived from all recorded steps.

### 6.2 The sufficient test

Store three notes with vaHera. Two of them are on-topic for a goal,
one is off-topic:

```
memory store "note1" = "TUM was founded in 1868"
memory store "note2" = "AIMe Registry is at TU Munich"
memory store "note3" = "buy milk from the corner shop"
```

Now ask purpose-carry which of the three you should carry as context
if your next act is about "TUM" and "AIMe":

```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: ["tum", "aime"],
  budget: 500
})
```

You should get back a card showing:
- **keep**: `note1` and `note2` (the on-topic ones)
- **dropped**: `note3` (the milk one — free-dropped)
- **regenerable**: possibly empty (no diamond redundancy in this small
  graph)
- **ambientFloor**: a positive number (the paper's Floor Theorem)
- **diagnostics**: total kept cost, budget remaining, knapsack
  relaxation gap

Click "show breakdown" to see the per-step residues.

### 6.3 Free-text goal

You can pass a string instead of a term array; τ (the term extractor
in `src/lib/purpose-terms.js`) will lowercase, drop stopwords, and
extract the terms for you:

```
dispatch("purpose-carry", {
  kind: "carry",
  goal: "when was Technical University Munich founded"
})
```

### 6.4 Explicit step registration

If you want to feed a step that didn't come from a dispatch:

```
dispatch("purpose-carry", {
  kind: "add",
  id: "manual1",
  content: "the Apollo 11 mission launched on 16 July 1969"
})
```

### 6.5 Reset the session

```
dispatch("purpose-carry", { kind: "reset" })
```

Wipes all steps. Useful between tracks.

### 6.6 What purpose-carry does not do

- It doesn't fetch or store content. Steps are (id, terms, cost)
  handles — content lives in the caller (vaHera, another module, or a
  provided payload).
- It doesn't call an LLM. Everything is pure graph computation.
- It doesn't decide *what* the next act should be. It only prunes
  history to what's necessary for a goal you supply.

---

## Track 7 — Composition: putting it all together

The most interesting demos are the ones that use several modules in one
flow. A few worked examples.

### 7.1 Prime the kernel, then let zangalewa answer using your context

```
memory store "recent" = "I met with the AIMe team on Thursday about a paper submission"
memory store "goal" = "want to know when the submission deadline is"

item p = dispatch("purpose-carry", {
  kind: "carry",
  goal: ["aime","deadline","submission"],
  budget: 300
})

item z = dispatch("zangalewa", "given the AIMe conversation context, what's the submission deadline for the Buhera paper?")
```

Purpose-carry tells you which past acts are on-topic. In a real
orchestrator you would forward those payloads to zangalewa as context.
(That composition step is the caller's job — purpose-carry returns
step ids, not content; you look up the content and pass it in.)

### 7.2 Graffiti seek with a kernel corpus

Store a small corpus, then seek against it:

```
memory store "einstein" = "Albert Einstein published special relativity in 1905"
memory store "planck" = "Max Planck introduced quantum hypothesis in 1900"
memory store "bohr" = "Niels Bohr proposed the atomic model in 1913"

item g = dispatch("graffiti", `
  floor 0.02
  catalyst kernel_search { namespace: local input: Region output: Claim }
  project physics {
    seek year
      not { "unsourced" }
      toward { "when did special relativity get published" }
      via { kernel_search(year) }
      until converge
      yield year
  }
`)
print(g.output_delta.projects.physics.year)
```

### 7.3 Zangalewa followup, then purpose synthesis

```
item z = dispatch("zangalewa", "what is CRISPR")
item p = dispatch("purpose", {
  kind: "synthesise",
  description: "given a beginner's overview of CRISPR, explain its use in gene therapy",
  followups: ["what are the safety concerns?"]
})
```

Zangalewa gives you a research card (one call, structured output);
purpose gives you a longer synthesis (federated cascade). Two
complementary LLM entry points.

### 7.4 Lavoisier then purpose-carry

Run a mass-spec simulation, then ask purpose-carry to isolate the
relevant history for a followup:

```
item ms = dispatch("lavoisier", "demo")

// now the audit log has a lavoisier act with lipidomics-adjacent terms
dispatch("purpose-carry", {
  kind: "carry",
  goal: ["phosphatidylcholine","spectra","orbitrap"]
})
```

The lavoisier act will surface in the `keep` set because its
instruction carried terms that overlap the goal.

### 7.5 Audit trail

At any point:

```
:audit
```

You see every act with its ID, module, wall-clock time, and instruction
summary. This is the "committed history" the paper talks about — the
substrate that purpose-carry filters and that any future orchestration
layer would build on.

---

## Common commands cheat-sheet

Meta commands (Buhera terminal built-ins):

```
:modules       list all registered modules
:audit         show recent dispatched acts
:tour          load a demo corpus into vaHera
:proteins      load a hardcoded biology database
:clear         reset the vaHera kernel
:help          show every vaHera statement
:quit          (browser tab — this one is decorative)
```

vaHera statements you'll use most:

```
memory store "name" = "text"
memory find nearest "query" k=5
memory list
memory dump <name>
kernel stats
kernel trace
demon sort
controller verify
```

Turbulance keywords the router recognises as opening a script:

```
funxn  item  proposition  hypothesis  point  given  considering
within  research  for each
```

Dispatch shapes across the federation:

```
dispatch("echo", <anything>)
dispatch("vahera", "<vaHera source>")
dispatch("lavoisier", "demo" | { kind: "virtual_run", ... })
dispatch("graffiti", "demo" | "<.grf source>")
dispatch("zangalewa", "<natural language>")
dispatch("purpose", "<description>" | { kind: "synthesise", ... })
dispatch("purpose-carry", { kind: "carry" | "add" | "stats" | "reset", ... })
```

Server routes worth knowing:

```
GET  /api/providers          which LLM providers are configured
POST /api/extract            zangalewa → LLM cascade with structured output
POST /api/hf-inference       graffiti's hf_inference catalyst
POST /api/purpose            legacy Rust-binary purpose bridge
POST /api/purpose-federation new JS-federation purpose
```

---

## What to try when things don't work

### "no provider configured"

Zangalewa or purpose was called but no LLM API key is set on the
deployment. Fix by adding one of `OLLAMA_URL`, `GEMINI_API_KEY`,
`HUGGINGFACE_API_KEY`, or `OPENAI_API_KEY` to the environment. On
Vercel: project settings → environment variables → redeploy.

### "unknown module"

The dispatch called a module id that isn't registered. Check `:modules`
for the exact spelling. Common trap: `purpose` vs `purpose-carry` are
two different modules.

### "purpose-carry: goal is empty after τ-extraction"

τ dropped every token — either the goal string was too short (all
words <3 chars), or every word was a stopword. Try a longer, more
specific goal, or pass an explicit term array:

```
dispatch("purpose-carry", { kind: "carry", goal: ["tum", "aime"] })
```

### zangalewa returns HTTP 429

An LLM provider is rate-limited or out of quota. Try `/api/providers`
to see which one is active, then either wait, switch providers, or
force a specific one:

```
POST /api/extract  { "utterance": "...", "provider": "gemini" }
```

### A dispatched module errors

Every ActResult carries an `error` field on failure and a `lines`
array in `output_delta.kind = "text"` showing what went wrong. The
same information appears in `:audit`.

---

## Where to go from here

The federation is deliberately open. Adding a new module means writing
one file that conforms to the Module trait in
`src/lib/modules/registry.js`, registering it in the terminal's mount
effect, and (optionally) writing a renderer for its output shape.

Every proof in the papers under `docs/sources/` corresponds to a
component you can inspect at runtime:

| Paper | Runtime surface |
|-------|-----------------|
| composition-inflation | `kernel stats` — the ternary substrate is the paper's category |
| scheduling-mechanism | `:audit` — the monotone act ID is the paper's trajectory count |
| semantic-causal-propagation | graffiti's `.grf` interpreter |
| minimum-sufficient-interceptor | zangalewa's S-coord extraction |
| federated-knapsack | purpose's federation.js aggregate floor |
| purpose-propagation | `@buhera/purpose` and the purpose-carry module |

Every one of these is currently running on the deployment. This
tutorial is not describing what Buhera *will* do — it's describing what
Buhera does right now, tonight, when you type into the terminal.
