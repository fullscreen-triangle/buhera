// API route: purpose federation.
//
// Runs the JS federation's knowledge-pack retrieval + prompt assembly, then
// hits the shared LLM cascade (Ollama → Gemini → HF → OpenAI) for the
// synthesis call. Keeps the same JSON contract as before.
//
// Contract:
//   POST /api/purpose-federation
//     body: { description: string, followups?: string[], field?: string|null, provider?: string }
//   -> { ok: true, synthesis, model, provider, floor, federation, packs_used }
//   -> { ok: false, error, stage?, provider? }

import { selectPacks, buildPackContext } from "@/lib/purpose/knowledge-packs";
import {
  getFederationModels,
  aggregateFloor,
  federationMetadata,
} from "@/lib/purpose/federation";
import { chatCascade } from "@/lib/server/llm-cascade";

const MAX_DESCRIPTION_BYTES = 32 * 1024;
const MAX_FOLLOWUP_COUNT = 32;
const PACK_BUDGET_TOKENS = 60_000;
const SYSTEM_PROMPT =
  "You are Purpose, a federated research synthesis engine. " +
  "Produce a concise, well-cited synthesis document.";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { description, followups, field, provider } = req.body || {};
  if (typeof description !== "string" || description.trim().length === 0) {
    return res
      .status(400)
      .json({ ok: false, error: "description (non-empty string) is required" });
  }
  if (description.length > MAX_DESCRIPTION_BYTES) {
    return res
      .status(413)
      .json({ ok: false, error: "description exceeds 32 KiB" });
  }
  const cleanFollowups = Array.isArray(followups)
    ? followups.slice(0, MAX_FOLLOWUP_COUNT).map(String)
    : [];

  // Pack selection is pure — no network call. Safe to do before checking
  // provider availability.
  let packContext = "";
  let packsUsed = [];
  try {
    const haystack = [description, ...cleanFollowups].join("\n");
    const packs = selectPacks(haystack, { budget: PACK_BUDGET_TOKENS });
    packContext = buildPackContext(packs.map((p) => p.id));
    packsUsed = packs.map((p) => ({ id: p.id }));
  } catch (err) {
    packContext = "";
    packsUsed = [];
    console.warn("purpose-federation: pack selection failed", err);
  }

  const userText = [
    packContext ? `Context:\n${packContext}\n` : "",
    field ? `Field: ${field}\n` : "",
    `Description:\n${description}`,
    cleanFollowups.length
      ? `\nFollowups:\n${cleanFollowups.map((f) => `- ${f}`).join("\n")}`
      : "",
  ]
    .join("")
    .trim();

  const result = await chatCascade({
    system: SYSTEM_PROMPT,
    user: userText,
    maxTokens: 2048,
    temperature: 0.3,
    only: provider ? [provider] : undefined,
  });

  if (!result.ok) {
    return res.status(result.stage === "provider" ? 503 : 502).json(result);
  }

  // Federation metadata still comes from the JS lib — it computes the
  // aggregate floor across the paper's declared federation models, which
  // is a bookkeeping artefact, not the LLM we actually called.
  const federationIds = getFederationModels();
  const floor = aggregateFloor([...federationIds, result.model]);
  const meta = federationMetadata(federationIds);

  return res.status(200).json({
    ok: true,
    synthesis: result.content,
    model: result.model,
    provider: result.provider,
    floor,
    federation: meta,
    packs_used: packsUsed,
  });
}
