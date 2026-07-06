// API route: LLM inference for the graffiti hf_inference catalyst.
//
// POST /api/hf-inference   body: { claim: string, prompt?: string }
//   -> { ok: true, refined, provider, model }
//   -> { ok: false, error, stage? }
//
// Path kept as /api/hf-inference for backwards compatibility with the
// graffiti catalyst (createHfInferenceCatalyst). Internally it now runs
// through the shared llm-cascade so any provider (Ollama, Gemini, HF,
// OpenAI) works — whichever is configured.

import { chatCascade } from "@/lib/server/llm-cascade";

const DEFAULT_PROMPT =
  "You are a claim-refinement catalyst. Given a raw claim, return a " +
  "single crisp, factually careful restatement. Reply with the refined " +
  "claim only — no preamble, no commentary, no meta-language.";

const MAX_CLAIM_LEN = 4096;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { claim, prompt, provider } = req.body || {};
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

  const system =
    typeof prompt === "string" && prompt.length > 0 ? prompt : DEFAULT_PROMPT;

  const result = await chatCascade({
    system,
    user: claim,
    maxTokens: 512,
    temperature: 0.2,
    only: provider ? [provider] : undefined,
  });

  if (!result.ok) {
    return res.status(result.stage === "provider" ? 503 : 502).json(result);
  }

  return res.status(200).json({
    ok: true,
    refined: result.content,
    provider: result.provider,
    model: result.model,
  });
}
