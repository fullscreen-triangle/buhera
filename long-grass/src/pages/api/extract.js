// API route: zangalewa coord extractor.
//
// POST /api/extract   body: { utterance: string, provider?: string }
//   -> RenderResult  (see src/components/render-leaves/types)
//   -> { ok: false, error, stage?, provider? }
//
// Runs through the shared LLM cascade (Ollama → Gemini → HF* → OpenAI),
// asking for structured JSON matching the RESPONSE_SCHEMA. HF is skipped
// automatically when a schema is required (it can't natively enforce one).

import { chatCascade } from "@/lib/server/llm-cascade";

const MAX_UTTERANCE_LEN = 4096;

const SYSTEM_PROMPT = `You are a research synthesis engine. Given a scientific query, produce a concise, surgical information card containing only the information most directly relevant to the query.

RULES:
- Emit exactly one "research" leaf.
- Title: one line identifying the target, with type/class, e.g. "p53 · tumour suppressor · TP53 · 393 aa" or "caffeine · C8H10N4O2 · 194.19 g/mol".
- Kind: one short word for the category, e.g. "protein", "compound", "concept", "gene", "disease", "reaction", "technique".
- Sections: 3-5 short sections. Each heading is a single lowercase word or hyphenated phrase. Each body is ONE short paragraph of 1-3 sentences. No bullet lists inside bodies. No restating of the question.
- Tag: one-line clinical or practical framing, if applicable. Empty string if not applicable.
- References: up to 3 key citations. Prefer canonical primary references. Use empty array if none.

DO NOT:
- Write "In summary..." or "Overall..." or any meta-commentary.
- Apologise or disclaim.
- Include information unrelated to the query's explicit target.

COORD:
Emit an S-entropy coord (S_k, S_t, S_e) in [0,1]^3:
- S_k (knowledge entropy): 0.2 = narrow/specific entity, 0.8 = broad concept.
- S_t (temporal entropy): 0.2 = well-settled classical, 0.8 = active frontier.
- S_e (evolution entropy): 0.2 = simple lookup, 0.8 = multi-step inference.

CAPTION:
One-sentence description of what was synthesised. 15 words max.

The output is the information. The form is the schema. Nothing else.`;

const RESPONSE_SCHEMA = {
  name: "research_render_result",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      caption: { type: "string" },
      leaves: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            leaf: { type: "string", enum: ["research"] },
            coord: {
              type: "object",
              additionalProperties: false,
              properties: {
                S_k: { type: "number" },
                S_t: { type: "number" },
                S_e: { type: "number" },
              },
              required: ["S_k", "S_t", "S_e"],
            },
            params: {
              type: "object",
              additionalProperties: false,
              properties: {
                kind: { type: "string" },
                title: { type: "string" },
                tag: { type: "string" },
                sections: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    properties: {
                      heading: { type: "string" },
                      body: { type: "string" },
                    },
                    required: ["heading", "body"],
                  },
                },
                references: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    properties: {
                      citation: { type: "string" },
                      url: { type: "string" },
                    },
                    required: ["citation", "url"],
                  },
                },
              },
              required: ["kind", "title", "tag", "sections", "references"],
            },
          },
          required: ["leaf", "coord", "params"],
        },
      },
    },
    required: ["caption", "leaves"],
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { utterance, provider } = req.body || {};
  if (typeof utterance !== "string" || utterance.trim().length === 0) {
    return res
      .status(400)
      .json({ ok: false, error: "utterance (non-empty string) required" });
  }
  if (utterance.length > MAX_UTTERANCE_LEN) {
    return res
      .status(413)
      .json({ ok: false, error: `utterance exceeds ${MAX_UTTERANCE_LEN} chars` });
  }

  const result = await chatCascade({
    system: SYSTEM_PROMPT,
    user: utterance,
    jsonSchema: RESPONSE_SCHEMA,
    maxTokens: 2048,
    temperature: 0.2,
    only: provider ? [provider] : undefined,
  });

  if (!result.ok) {
    return res.status(result.stage === "provider" ? 503 : 502).json(result);
  }

  // Parse the model's JSON string into the RenderResult.
  let renderResult;
  try {
    renderResult = JSON.parse(result.content);
  } catch {
    return res.status(502).json({
      ok: false,
      provider: result.provider,
      model: result.model,
      error: "model returned content that is not valid JSON",
      raw: result.content.slice(0, 512),
      stage: "content-parse",
    });
  }

  // Surface the provider used so the client renderer can show it.
  renderResult._meta = { provider: result.provider, model: result.model };
  return res.status(200).json(renderResult);
}
