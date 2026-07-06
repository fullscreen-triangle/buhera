// GET /api/providers
//
// Returns which LLM providers are currently configured on the server.
// Never returns actual API keys or model output — just the list of
// available provider names in cascade priority order.
//
// Handy for debugging "why is dispatch not doing anything" without
// spending an API call.

import { availableProviders } from "@/lib/server/llm-cascade";

export default function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }
  const available = availableProviders();
  return res.status(200).json({
    ok: true,
    available,
    cascade_order: ["ollama", "gemini", "huggingface", "openai"],
    active: available[0] ?? null,
  });
}
