// API route: HuggingFace inference for the graffiti hf_inference catalyst.
//
// POST /api/hf-inference   body: { claim: string, prompt?: string }
//   -> { ok: true, refined: string, model: string }
//   -> { ok: false, error, stage? }
//
// Uses the HuggingFace Inference Router chat-completions endpoint with a
// small, ungated instruct model. Keeps the API key server-side.

const HF_URL =
  "https://router.huggingface.co/hf-inference/v1/chat/completions";
const DEFAULT_MODEL =
  process.env.GRAFFITI_HF_MODEL || "meta-llama/Meta-Llama-3-8B-Instruct";
const DEFAULT_PROMPT =
  "You are a claim-refinement catalyst. Given a raw claim, return a " +
  "single crisp, factually careful restatement. Reply with the refined " +
  "claim only, no preamble or commentary.";

const MAX_CLAIM_LEN = 4096;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { claim, prompt } = req.body || {};
  if (typeof claim !== "string" || claim.trim().length === 0) {
    return res
      .status(400)
      .json({ ok: false, error: "claim (non-empty string) required" });
  }
  if (claim.length > MAX_CLAIM_LEN) {
    return res
      .status(413)
      .json({ ok: false, error: `claim exceeds ${MAX_CLAIM_LEN} chars` });
  }

  const apiKey = process.env.HUGGINGFACE_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      ok: false,
      error: "no HUGGINGFACE_API_KEY configured. set it in .env.local.",
      stage: "provider",
    });
  }

  const system = typeof prompt === "string" && prompt.length > 0
    ? prompt
    : DEFAULT_PROMPT;

  let upstream;
  try {
    upstream = await fetch(HF_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages: [
          { role: "system", content: system },
          { role: "user", content: claim },
        ],
        max_tokens: 512,
        temperature: 0.2,
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
      error: `hf returned HTTP ${upstream.status}`,
      details: errText.slice(0, 512),
      stage: "upstream",
    });
  }

  let body;
  try {
    body = await upstream.json();
  } catch {
    return res.status(502).json({
      ok: false,
      error: "hf returned non-JSON",
      stage: "parse",
    });
  }

  const refined = body.choices?.[0]?.message?.content;
  if (typeof refined !== "string") {
    return res.status(502).json({
      ok: false,
      error: "hf response missing choices[0].message.content",
      stage: "shape",
    });
  }

  return res
    .status(200)
    .json({ ok: true, refined: refined.trim(), model: DEFAULT_MODEL });
}
