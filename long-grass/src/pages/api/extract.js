// API route: zangalewa coord extractor.
//
// POST /api/extract   body: { utterance: string }
//   -> RenderResult (see src/components/render-leaves/types)
//   -> { ok: false, error, stage? } on failure
//
// Calls OpenAI's structured-output API with the zangalewa system prompt
// and JSON schema. The utterance becomes one research leaf + an S-coord.
//
// Requires OPENAI_API_KEY in .env.local. No key -> 503 with clear message.

const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MODEL = "gpt-4o-mini";
const MAX_UTTERANCE_LEN = 4096;

// Pulled from src/lib/zangalewa/prompt.ts. Kept inline so this route has no
// TypeScript compile step in the way — Next.js API routes can import .ts,
// but keeping the prompt colocated avoids an import cycle if prompt.ts ever
// grows non-portable dependencies.
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
- Pad sections. Short is good. Empty is bad.

COORD:
Emit an S-entropy coord (S_k, S_t, S_e) in [0,1]^3. This is a structural address for the target:
- S_k (knowledge entropy): 0.2 = narrow/specific entity, 0.8 = broad concept.
- S_t (temporal entropy): 0.2 = well-settled classical knowledge, 0.8 = active research frontier.
- S_e (evolution entropy): 0.2 = simple lookup, 0.8 = requires multi-step inference.

CAPTION:
One-sentence caption describing what was synthesised. 15 words max.

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

  const { utterance } = req.body || {};
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

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      ok: false,
      error: "no OPENAI_API_KEY configured. set it in .env.local.",
      stage: "provider",
    });
  }

  const model = process.env.ZANGALEWA_MODEL || DEFAULT_MODEL;

  let upstream;
  try {
    upstream = await fetch(OPENAI_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: utterance },
        ],
        response_format: { type: "json_schema", json_schema: RESPONSE_SCHEMA },
        max_tokens: 2048,
      }),
    });
  } catch (err) {
    return res.status(502).json({
      ok: false,
      error: `upstream fetch failed: ${err.message || String(err)}`,
      stage: "network",
    });
  }

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => "");
    return res.status(502).json({
      ok: false,
      error: `openai returned HTTP ${upstream.status}`,
      details: errText.slice(0, 512),
      stage: "upstream",
    });
  }

  let body;
  try {
    body = await upstream.json();
  } catch (err) {
    return res.status(502).json({
      ok: false,
      error: "openai returned non-JSON",
      stage: "parse",
    });
  }

  const content = body.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    return res.status(502).json({
      ok: false,
      error: "openai response missing choices[0].message.content",
      stage: "shape",
    });
  }

  let renderResult;
  try {
    renderResult = JSON.parse(content);
  } catch (err) {
    return res.status(502).json({
      ok: false,
      error: "content is not valid JSON",
      raw: content.slice(0, 512),
      stage: "content-parse",
    });
  }

  return res.status(200).json(renderResult);
}
