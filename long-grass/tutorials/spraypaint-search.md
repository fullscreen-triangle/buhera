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
two backends reached through graffiti's `seek`/`via` chains instead. If
you've done [Federated Querying](./federated-querying), the water-filling
"clearing price" §2 below introduces will look familiar — it's the same
family of idea as that tutorial's ladder floor: a single number computed
from the whole competitive field of candidates, not assignable to any one
result on its own.

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

[Federated Querying](./federated-querying) walks the same underlying claim
this whole family of tutorials keeps returning to: a shape language
(LinkML, JSON Schema, SHACL, OWL) can certify what a record *looks like*
and still say nothing about whether a given question against that data has
an answer, no answer, or was never reached at all. That tutorial makes the
case with `hfq`'s six-verdict execution model and `ladder`'s contact-graph
floor. Ask spraypaint about that argument's own central claim, in the same
words a reader chasing it would type — searching the actual prose that
makes the argument, not a paraphrase of it:

**Cell 1.1**
```
dispatch("spraypaint", { kind: "ask", query: "capability calculus six verdict starved refused", budget: 5, scenes: ["long-grass"] })
```

**Expected** — a ranked list of real passages from this repository. Running
this for real lands on the tutorial's own header and the `hfq` module's
real source as the top two hits — not fabricated, this is what the actual
index returns for this actual query:
```
long-grass/tutorials/federated-querying.md:1-40      score 27.62
  "# Federated Querying: Three Methods, and What LinkML Alone Would Have Told You"
long-grass/src/lib/modules/hfq-module.js:61-100      score 26.99
  "/** One-line summary of a run: verdict tally, for the audit log and text fallbac"
long-grass/tutorials/federated-querying.md:90-129    score 19.79
  "**Expected** — read the two verdicts side by side. One step reports"
```
Notice the hit into the *module's own source file*, not just the tutorial
that walks it — spraypaint indexes the whole repository, code and prose
alike, so a query about a real engineering concept can surface the
implementation itself, which is a genuinely different (and often more
precise) thing to read than a tutorial's paraphrase of it.

Below the list, an interactive bar chart renders the `allocation` array —
every scene (top-level directory) that had *any* matching passages, its
`available` count as a track and its `allocated` count as a filled bar.
Hover a bar for the exact numbers. Restricting to `scenes: ["long-grass"]`
here is deliberate, not incidental — §3 below explains why searching the
unrestricted index for this exact query would have buried the useful hits
under noise from other projects sharing the same vocabulary.

A note on latency: this call took **~15 seconds** end to end when tested,
because the API route spawns the `spraypaint` CLI fresh per request and the
index it reads back is large (built over the whole `buhera` repository, not
just `long-grass`). This is a real, currently-unoptimized cost — not a
network delay, not a bug — worth expecting rather than assuming something
hung.

---

## 2. Reading the numbers spraypaint reports about itself

**Cell 2.1** — the same idea, a different subject — ladder's intensive vs.
extensive power distinction:
```
dispatch("spraypaint", { kind: "ask", query: "intensive extensive power contact graph medium vertex", budget: 5, scenes: ["long-grass"] })
```

**Expected** — this time every one of the top hits lands in the `ladder`
module's own real implementation, ranked above the tutorial that explains
it (the phrase is closer to the code's own vocabulary — variable and
function names — than to the tutorial's looser prose):
```
long-grass/tutorials/federated-querying.md:121-160  score 35.06
long-grass/src/lib/modules/ladder-module.js:1-40    score 33.67
long-grass/src/lib/modules/ladder-module.js:61-100  score 27.17
  "function graphFromPlain(g) {"
```
This is worth sitting with for a moment: the highest-scoring passage is a
divider line (`---`) inside the tutorial markdown, immediately before the
prose that explains this exact concept — a reminder that BM25 scores
*terms*, and a passage sitting right next to a term-dense heading can score
well even when the passage itself is nearly empty. Read the neighboring
lines, not just the exact `start_line`–`end_line` window, when a top hit
looks thin.

Above the result list, the header line reports `price` (spraypaint's
internal name for the water-filling clearing price — the marginal score a
passage needed to clear to receive budget; higher price means a more
competitive query, not an error) and `committed_count` (a monotone counter
spraypaint keeps across every real `ask` it has ever answered against this
index — it only goes up, by design, so you can tell whether a number you're
looking at reflects a fresh search or a cached one).

---

## 3. Restricting the search to one scene — and why getting it wrong is a
real failure mode, not a cosmetic one

Both queries above sent every one of their budget slots to `long-grass`
because that's genuinely where the matching content lives — but the index
this deployment searches spans a much larger checkout: `buhera-os`, `docs`,
`driven`, `arxiv_submissions`, `category`, `kernel`, `substrate`, and more,
over twenty scenes in total, most of them unrelated projects that happen to
share this parent repository. Restricting to a scene up front matters, and
it's worth seeing what happens when you restrict to the *wrong* one, not
just the right one — because both are one line of instruction away from
each other and only one of them is a mistake:

**Cell 3.1** — the same query as §2.1, restricted to `docs` instead of
`long-grass`:
```
dispatch("spraypaint", { kind: "ask", query: "admissibility floor conditioned separation", budget: 8, scenes: ["docs"] })
```

**Expected** — real results, confidently ranked, about something else
entirely. `docs` is a real scene with real content — it just isn't the
content this query is actually chasing:
```
docs/theory/stellas-constant.md:181-220  score 5.39
docs/theory/buhera.md:511-550            score 5.02
docs/theory/time.md:61-100               score 4.94
```
Compare the scores against §2.1's unrestricted run (18–27) — an order of
magnitude lower, because these are weak, incidental term overlaps
("conditioned," "separation" used in an unrelated theoretical sense), not
the concept the query meant. **This is the real failure mode of scene
restriction**: it never reports "wrong scene, zero results" — a wrong
restriction returns a confident-looking, real, ranked list, from content
that has nothing to do with what you meant. Nothing about the JSON shape or
the UI distinguishes a right-scene result from a wrong-scene one; only the
score magnitude and your own judgment of the snippets do. Treat a
suspiciously low top score as a signal to check `scenes` before trusting
the list.

**Cell 3.2** — the corrected restriction:
```
dispatch("spraypaint", { kind: "ask", query: "admissibility floor conditioned separation", budget: 8, scenes: ["long-grass"] })
```

**Expected** — scores back in the 11–23 range, results back in the
`hfq`/`ladder`/`cytochrome` neighborhood — the same shape of outcome as
§1–2, restricted rather than incidental this time. The allocation chart
now shows only the `long-grass` row, every other scene excluded from
consideration entirely (not just scored zero and hidden) — the mechanism is
identical between Cell 3.1 and Cell 3.2, only the scene name differs, and
that one difference is the entire distance between a useful answer and a
plausible-looking wrong one.

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

## 4½. What spraypaint's "clearing price" has in common with a ladder
floor, and where the analogy breaks

If you've done [Federated Querying](./federated-querying), §3 there derived
a `ladder` graph's `floor` — the minimum edge weight over the whole
structure, a genuinely global quantity no single vertex or edge can report
on its own. `price` in every result above (Cell 1.1's header line, Cell
2.1's, Cell 3.1/3.2's) is doing something structurally similar: it's the
marginal BM25 score a passage needed to reach in order to receive a slot of
budget, computed once from the *entire competitive field* of candidate
passages, not attached to any one of them individually. Neither number
exists until you've looked at everything that could have competed for the
same resource — a floor is the minimum over every edge, a clearing price is
the score threshold that exactly exhausts the budget across every
candidate.

The analogy is real but limited, and the limit is worth naming so it
doesn't quietly turn into an overclaim: a ladder's `floor` is a genuine
admissibility bound — the paper's whole argument in [Federated
Querying](./federated-querying) §3–4 is that this number determines what
questions can be *answered at all*, refusing before commitment when a
target is unreachable (§4's `subfloor` verdict). Spraypaint's `price` isn't
an admissibility bound in that sense — it doesn't refuse anything, and a
low-price query still returns whatever cleared it, however weak the match
(exactly what Cell 3.1 above demonstrated: a real, ranked, *wrong* answer,
not a refusal). Run the same query at two different budgets to see this
distinction concretely — the price moves, but nothing about it tells you
whether the results it let through are actually relevant:

**Cell 4½.1**
```
dispatch("spraypaint", { kind: "ask", query: "admissibility floor conditioned separation", budget: 3, scenes: ["long-grass"] })
```

**Cell 4½.2**
```
dispatch("spraypaint", { kind: "ask", query: "admissibility floor conditioned separation", budget: 15, scenes: ["long-grass"] })
```

**Expected** — a lower `price` at budget 15 than at budget 3 (more slots to
fill means a weaker passage can still clear), and a longer results list,
but no verdict, no refusal, nothing structurally like `ladder`'s
`subfloor`. Compare this directly against [Federated Querying](./federated-querying)
§4's Cell 4.2, where raising the *target* past what the rungs can reach
produces an explicit `verdict: "subfloor"`, `M: 0` — a refusal decided
before any commitment. Spraypaint has no equivalent concept: it always
returns its best `budget` candidates, confident or not, and the only thing
distinguishing a strong answer from a weak one is a number you have to read
and judge yourself. That difference is not a shortcoming particular to this
implementation — it's the actual shape of the distinction the whole
"shape vs. admissibility" family of tutorials on this site keeps drawing:
a ranking is a *shape*-adjacent operation (score everything, return the
top-K), and admissibility is the separate, harder claim that a question
can or cannot be answered at all, which nothing about ranking speaks to
either way.

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
