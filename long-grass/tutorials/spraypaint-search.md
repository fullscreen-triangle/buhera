# Spraypaint: Local and Internet Search

**What you'll learn:** the `spraypaint` module — full-text passage retrieval
over this repository on disk (real BM25 ranking, real water-filling budget
allocation across directories), and internet search through the same
module's `web` action. Both return results shaped for the terminal to
render as more than plain text: a ranked, expandable result list plus an
interactive chart of how the search budget actually split across the
codebase.

**Time:** ~15 minutes.

**Prerequisites:** [Basic routines](./basic-routines) for `dispatch(...)`
itself. Independent of vaHera and of graffiti — this tutorial covers
`dispatch("spraypaint", ...)` directly. [vaHera search
catalysts](./vahera-search-catalysts) picks up from here and shows the same
two backends reached through graffiti's `seek`/`via` chains instead.

**Runtime requirement:** a working deployment with the `spraypaint` CLI
installed server-side (checked by `/api/spraypaint`; the module reports a
clear error if it's missing rather than failing silently) for local search,
and a valid `GEMINI_API_KEY` for internet search. Every cell and its
"Expected" block below was run for real against this exact codebase before
being written down — nothing here is illustrative.

---

## 0. What spraypaint actually is, and isn't

`spraypaint` is a separate tool from vaHera's own `memory find nearest` —
worth stating plainly, because both return "the nearest thing to your
query" and it is easy to conflate them. vaHera's memory only knows what you
explicitly `memory store`d in this browser session; it is empty until you
put something in it. Spraypaint searches **files already on disk** — the
whole checked-out repository, indexed once, queried many times. They share
no state. Running `dispatch("spraypaint", ...)` does not read or write
anything `memory store` touched, and vice versa.

Spraypaint's ranking has two parts, both real and worth understanding
before the results look like magic or noise:

- **BM25** scores each passage against your query terms — a standard
  full-text relevance score, not semantic/embedding similarity like
  vaHera's S-coordinates. Exact and near-exact word matches win; a query
  about "floors" will not find a passage that only says "minima" unless
  both words happen to appear somewhere nearby.
- **Water-filling** then allocates a fixed total result budget *across
  directories* ("scenes" — spraypaint's name for a top-level folder like
  `long-grass/` or `buhera-os/`), rather than just taking the global top-K
  by score. A scene with many strong hits doesn't necessarily get every
  slot — the allocator spreads budget toward scenes with real matches
  first, so a query doesn't return ten hits from one file when three other
  directories also have something relevant.

---

## 1. A first query, on the paper this tutorial keeps coming back to

The [data-modeling tutorial](./data-modeling-capability) walks
`docs/data-modeling-capability/data-modeling-capability.tex` — the paper on
why a data schema (LinkML, JSON Schema...) cannot certify *admissibility*,
only *shape*. Ask spraypaint about that paper's own central claim, in the
same words a reader chasing the argument would type:

**Cell 1.1**
```
dispatch("spraypaint", { kind: "ask", query: "admissibility floor global minimum", budget: 5 })
```

**Expected** — a ranked list of real passages from this repository. Running
this for real returns the paper's own abstract and the `data-modeling-capability.md`
tutorial as the top two hits — not fabricated, this is what the actual
index returns for this actual query:
```
long-grass/tutorials/data-modeling-capability.md:1-40   score 17.83
  "# Shape Is Not Admissibility"
long-grass/tutorials/data-modeling-capability.md:30-69  score 16.18
  "The question this tutorial works through is different: **given a schema that"
long-grass/docs/data-modeling-capability/data-modeling-capability.tex:69-100  score 14.03
  "the first ..."
```
Below the list, an interactive bar chart renders the `allocation` array —
every scene (top-level directory) that had *any* matching passages, its
`available` count as a track and its `allocated` count as a filled bar.
Hover a bar for the exact numbers. For this query, on this repo, the
allocator gave all 5 budget slots to the `long-grass` scene — every
directory searched, only one had passages worth spending budget on.

A note on latency: this call took **~15 seconds** end to end when tested,
because the API route spawns the `spraypaint` CLI fresh per request and the
index it reads back is large (built over the whole `buhera` repository, not
just `long-grass`). This is a real, currently-unoptimized cost — not a
network delay, not a bug — worth expecting rather than assuming something
hung.

---

## 2. Reading the numbers spraypaint reports about itself

**Cell 2.1** — the same query, a different subject:
```
dispatch("spraypaint", { kind: "ask", query: "LinkML capability set containment", budget: 5 })
```

**Expected** — this time every one of the top hits lands in the `.tex`
paper itself (the phrase is closer to the paper's own vocabulary than to
the tutorial's looser prose):
```
...data-modeling-capability.tex:240-279  score 21.60
  "A LinkML schema declares a set of classes, each with a set of slots, e"
...data-modeling-capability.tex:61-100   score 19.80
...data-modeling-capability.tex:119-158  score 19.38
```
Above the result list, the header line reports `price` (spraypaint's
internal name for the water-filling clearing price — the marginal score a
passage needed to clear to receive budget; higher price means a more
competitive query, not an error) and `committed_count` (a monotone counter
spraypaint keeps across every real `ask` it has ever answered against this
index — it only goes up, by design, so you can tell whether a number you're
looking at reflects a fresh search or a cached one).

---

## 3. Restricting the search to one scene

Budget only went to one scene in both queries above because that's genuinely
where the matches are — but you can restrict the search up front instead of
discovering that after the fact, which matters once an index spans many
unrelated projects (this one does: `buhera-os`, `docs`, `driven`,
`arxiv_submissions`, and more, all in the same index):

**Cell 3.1**
```
dispatch("spraypaint", { kind: "ask", query: "separation theorem shape schema", budget: 5, scenes: ["long-grass"] })
```

**Expected** — the same shape of result, but the allocation chart now shows
only the `long-grass` row — every other scene is excluded from
consideration entirely, not just scored zero. Useful once you know which
part of a large index your question actually lives in.

---

## 4. Building the index yourself

Every query above ran against an index someone already built. Building one
is itself a dispatchable action — worth doing once so you've seen the whole
lifecycle, not just the query half:

**Cell 4.1**
```
dispatch("spraypaint", { kind: "index" })
```

**Expected** — a confirmation with real counts: how many documents, how
many passages (spraypaint's fixed-size text windows — not one passage per
file, several per file for anything longer than the window), how many
scenes, and an `identity_fingerprint` (a hash spraypaint uses internally to
detect whether the underlying files changed between builds — Buhera OS's
own "conserved identity" check, unrelated to anything vaHera does with the
same word). Reindexing the whole repository takes real time — expect
minutes, not seconds, the larger the codebase gets.

**A rule worth carrying over from the CLI itself, even though this module
doesn't expose it as a dispatch option:** run `spraypaint ask "..." --dry-run`
directly in a terminal on your own machine before committing to an `ask`
you're not sure about. `--dry-run` prints the query terms, the clearing
price, and the per-scene allocation without incrementing
`committed_count` — but note it always prints plain diagnostic text, not
JSON, even with `--json` also passed; that's a real CLI quirk, not
something to route through this module's JSON-only path.

---

## 5. The internet half

The same module reaches the internet, through a completely different
mechanism: an LLM's own web-browsing tool (Gemini's Google Search
grounding), not spraypaint itself — spraypaint has no network client of any
kind, by design, and never will reach outside the filesystem it indexes.

**Cell 5.1**
```
dispatch("spraypaint", { kind: "web", query: "what is a conditioned admissibility floor" })
```

**Expected**, when a valid `GEMINI_API_KEY` is configured: a grounded
answer — real prose generated with real web search behind it — followed by
a numbered source list with clickable links, and (if the model's response
included them) the literal search queries it issued.

**Expected right now, on this deployment, honestly**: this call currently
fails with `gemini HTTP 401` — the `GEMINI_API_KEY` presently configured
does not authenticate against Google's API (confirmed by testing it
directly, independent of anything this module does). The module surfaces
this plainly rather than masking it:
```
spraypaint web: gemini HTTP 401
```
This is included deliberately, not glossed over: a search tool that fails
open (silently returning nothing, or returning a plausible-looking but
fabricated answer) is worse than one that fails loud. If you're reading
this after the key has been fixed, Cell 5.1 should instead return the
grounded-answer shape described above — if it still doesn't, that's a
regression worth reporting, not an expected state.

---

## 6. Local and internet, side by side, one cell

**Cell 6.1**
```
dispatch("spraypaint", { kind: "both", query: "admissibility floor global minimum" })
```

**Expected** — one result panel, two stacked sections: "local (spraypaint)"
rendering exactly Cell 1.1's output, and "internet (web search)" rendering
exactly Cell 5.1's output (or its 401 failure, under current conditions) —
the same underlying calls, run in parallel, presented as one notebook cell
with two outputs instead of two separate dispatches. This is the shape to
reach for whenever a question is worth checking against both what this
codebase actually says and what the wider internet says about the same
claim.

---

## 7. What you now know

- `spraypaint` searches files on disk; it shares no state with vaHera's
  `memory store`/`memory find nearest`, which searches only what you've
  explicitly stored this session.
- Ranking is BM25 (lexical, term-based) within scenes, budget-allocated
  across scenes by a water-filling algorithm — not semantic similarity.
  A query's exact wording matters more here than it does for vaHera's
  S-coordinate proximity search.
- `dispatch("spraypaint", { kind: "ask", query, budget?, scenes? })` for
  local search, `{ kind: "web", query }` for internet search (via an LLM's
  browsing tool, not spraypaint itself — spraypaint never leaves the
  filesystem), and `{ kind: "both", query }` for both side by side.
- `{ kind: "index" }` rebuilds the searchable index; expect it to take real
  time on a large repository.
- The web half depends on a working LLM API key and fails loudly, with a
  clear error, when one isn't configured or doesn't authenticate — it does
  not fabricate a plausible-looking answer in that case.

**Next up:** [vaHera search catalysts](./vahera-search-catalysts) — the
same two backends, reached from a graffiti `.grf` script's `seek ... via{}`
chain instead of a direct `dispatch`, compared side by side against
vaHera's own `kernel_search`.

---

## Troubleshooting

- **`spraypaint ask: spraypaint CLI not found`** — the server this
  deployment runs on doesn't have the `spraypaint` binary installed, or
  `SPRAYPAINT_CLI` points somewhere wrong. This is a deployment issue, not
  something fixable from the terminal.
- **A local search takes 10+ seconds** — expected on a large index; the API
  route spawns the CLI fresh per call rather than keeping a server
  resident, so every call pays index-load cost, not just query cost.
- **`spraypaint web: gemini HTTP 401` (or any HTTP 4xx/5xx)** — the
  configured `GEMINI_API_KEY` isn't valid or isn't authorized for this
  call. Not a bug in the module; check the key.
- **A local search returns nothing for a query you're sure should hit** —
  remember BM25 needs actual term overlap. Try the words more literally
  (closer to how the target text is actually phrased) before concluding
  nothing matches.
