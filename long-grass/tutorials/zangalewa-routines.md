# Zangalewa Routines

**What you'll learn:** how to use the Minimum Sufficient Interceptor —
Buhera's natural-language front door. Zangalewa takes an utterance and
returns a structured S-coordinate plus a fully-rendered research card in
a single call. This tutorial also covers the shared LLM cascade (Ollama →
Gemini → HuggingFace → OpenAI) that zangalewa and purpose both use.

**Time:** ~10 minutes.

**Prerequisites:** [Basic routines](./basic-routines).

**Runtime requirement:** the server must have at least one LLM provider
configured. Check with `GET /api/providers` before starting. If none is
active, the tutorial's calls will fail cleanly with a "no provider
configured" message; the shape of the calls is still worth reading.

---

## 0. The mental model

Zangalewa is a small model dedicated to *coordinate extraction* — given a
natural-language utterance, produce a structured address in a bounded
semantic space. The MSI paper (`docs/sources/minimum-sufficient-inter-
ceptor.tex`) proves this task requires exponentially less capacity than a
general-purpose LLM. In the deployed system that means a call to
zangalewa is cheap: one structured JSON response, ~2 seconds, ~1000
tokens.

Purpose is different — it produces a *synthesis document* over
knowledge-pack context. Longer output, more expensive.

Both use the same cascade under the hood: [src/lib/server/llm-
cascade.js](../../src/lib/server/llm-cascade.js).

---

## 1. Check what's configured

Open a new browser tab to your deployment (or `http://localhost:3000` if
running locally) and go to:
```
/api/providers
```

**Expected** — JSON like:
```json
{
  "ok": true,
  "available": ["gemini", "huggingface", "openai"],
  "cascade_order": ["ollama", "gemini", "huggingface", "openai"],
  "active": "gemini"
}
```

`active` is the provider the cascade will pick first. If `available` is
empty, none of the LLM-backed calls in this tutorial will work; skip to
the troubleshooting section.

---

## 2. Your first zangalewa call

**Cell 2.1** — Ask a simple factual question.
```
item z = dispatch("zangalewa", "what is p53?")
print(z.output_delta.title)
```

**Expected** — the terminal renders a research card with:

- **Title**: `p53 · tumour suppressor · TP53 · 393 aa` (or similar)
- **Caption**: a one-sentence description
- **Provider bar**: `via: gemini · gemini-2.0-flash`
- **S-coord**: `S_k=0.30, S_t=0.20, S_e=0.30` or similar
- **3–5 sections**: function, structure, mechanism, clinical, ...
- **References**: up to 3 clickable citations

The `print(z.output_delta.title)` also prints the title as a plain string.

**Cell 2.2** — The audit log records it.
```
:audit
```

**Expected** — a `zangalewa` entry with wall-clock time (usually 1–3
seconds).

---

## 3. Different query shapes

Zangalewa adapts to what it's asked. Try a mechanistic query:

**Cell 3.1**
```
dispatch("zangalewa", "how does CRISPR-Cas9 double-strand break repair work")
```

**Expected** — a card whose sections talk about the mechanism (Cas9
cleavage, NHEJ vs. HDR pathways, PAM recognition). S-coord will typically
have moderate S_e (multi-step inference).

**Cell 3.2** — Pedagogical framing.
```
dispatch("zangalewa", "explain quantum entanglement to a 10-year-old")
```

**Expected** — the same schema (title + sections + references) but the
body language shifts to accessible framing. S_e will be higher (multi-
step) and S_k moderate (broad concept).

---

## 4. Reading the S-coord

The three-axis S-coord tells you *how the query situates in the semantic
space*. It's what a real MSI-paper implementation would use to route to a
downstream module. Right now the coord is displayed but not yet used for
routing — the routing layer is a future addition to the OS.

**Cell 4.1**
```
item z = dispatch("zangalewa", "what is the molecular weight of caffeine")
print("S_k: {}", z.output_delta.coord.S_k)
print("S_t: {}", z.output_delta.coord.S_t)
print("S_e: {}", z.output_delta.coord.S_e)
```

**Expected**
```
S_k: 0.25
S_t: 0.20
S_e: 0.20
```

Low across the board — narrow entity, well-settled knowledge, single-step
lookup.

**Cell 4.2** — Compare with a research-frontier query.
```
item z = dispatch("zangalewa", "current understanding of dark matter distribution in the Milky Way halo")
print("S_t (should be high): {}", z.output_delta.coord.S_t)
print("S_e (should be high): {}", z.output_delta.coord.S_e)
```

**Expected** — S_t and S_e both >= 0.6.

---

## 5. Forcing a specific provider

If the deployment has multiple providers configured, you can force one by
including it in the instruction. Useful for A/B comparison or when one
provider is rate-limited.

**Cell 5.1** — Force Gemini.
```
dispatch("zangalewa", { utterance: "what is p53", provider: "gemini" })
```

**Cell 5.2** — Force OpenAI (if configured).
```
dispatch("zangalewa", { utterance: "what is p53", provider: "openai" })
```

Compare the two cards. Different providers give different phrasing but
the schema is identical because the API constrains it via JSON schema.

---

## 6. Composing zangalewa with purpose-carry

Zangalewa dispatches automatically feed the purpose session (like every
dispatch does). So a workflow of "ask, ask, ask, then compute the carry
toward a synthesis goal" is natural.

**Cell 6.1** — Reset and populate.
```
dispatch("purpose-carry", { kind: "reset" })
dispatch("zangalewa", "what is p53")
dispatch("zangalewa", "how does MDM2 regulate p53")
dispatch("zangalewa", "p53 mutations in cancer")
dispatch("zangalewa", "the plot of Casablanca")
```

**Cell 6.2** — Which of those are load-bearing for a p53 synthesis?
```
item c = dispatch("purpose-carry", {
  kind: "carry",
  goal: "p53 tumour suppressor regulation and cancer"
})
print("keep: {}", c.output_delta.keep.length)
print("dropped: {}", c.output_delta.dropped.length)
```

**Expected** — the first three zangalewa acts in `keep`, the Casablanca
act in `dropped`.

That's the OS working end-to-end: LLM calls feeding a decision plane
that decides which calls' residues carry forward.

---

## 7. Purpose (synthesis) — the sibling module

Purpose is not zangalewa. It runs a knapsack-allocated federated cascade
over LLMs and returns a longer synthesis document with knowledge-pack
context. Same LLM cascade underneath.

**Cell 7.1**
```
item p = dispatch("purpose", "explain the tandem carry from the purpose paper in one paragraph")
print(p.output_delta.synthesis)
```

**Expected** — a paragraph-length synthesis with a provider bar showing
which model answered.

**Cell 7.2** — Structured form with follow-ups.
```
dispatch("purpose", {
  kind: "synthesise",
  description: "compare zangalewa and purpose",
  followups: ["what's the same LLM cascade doing differently in each?"]
})
```

---

## 8. What you now know

- Zangalewa is the natural-language front door. One call, one card, one
  S-coord.
- Purpose is longer synthesis. Same cascade, different prompt shape.
- Both surface which provider answered so you can debug the cascade.
- Every LLM-backed dispatch is fed into the purpose-carry session; carry
  after a batch of zangalewa calls decides which are load-bearing.
- `/api/providers` tells you the cascade state before you spend tokens.

**Next up:** [Shapeshifter routines](./shapeshifter-routines) — the
lavoisier DSL for mass-spec workflows.

---

## Troubleshooting

- **`no provider configured`** — no LLM key is set. Add one to
  `.env.local` (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or
  `HUGGINGFACE_API_KEY`) and restart the dev server, or configure it in
  the Vercel dashboard for the deployed instance.
- **`HTTP 429` or `insufficient_quota`** — the active provider is rate-
  limited or out of quota. Force a different provider via
  `dispatch("zangalewa", { utterance: "...", provider: "openai" })`, or
  wait.
- **`model not supported by provider`** — HuggingFace's free tier
  restricts which models are served. Set `HF_MODEL` to a model your token
  supports, or let the cascade fall through to Gemini.
- **The rendered card is missing sections** — the LLM returned a partial
  response. Try again; the schema is strict but generation can occasionally
  drop optional fields.
