// API route: purpose federation
//
// Server-side wrapper around the JS federation in src/lib/purpose/. Runs on
// Node — reads knowledge packs from disk (fs), reads env-var API keys, and
// hits HuggingFace / Anthropic. The client-side purpose-module.js posts here
// and never sees an API key.
//
// Contract:
//   POST /api/purpose-federation
//     body: { description: string, followups?: string[], field?: string|null }
//   ->  { ok: true, synthesis, model, floor, federation, packs_used }
//   ->  { ok: false, error, stage? }

import { getProvider, synthesisModel } from "@/lib/purpose/llm";
import { selectPacks, buildPackContext } from "@/lib/purpose/knowledge-packs";
import {
  getFederationModels,
  aggregateFloor,
  federationMetadata,
} from "@/lib/purpose/federation";

const MAX_DESCRIPTION_BYTES = 32 * 1024; // 32 KiB
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

  const { description, followups, field } = req.body || {};
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

  let provider;
  try {
    provider = getProvider();
  } catch (err) {
    return res.status(503).json({
      ok: false,
      error: err.message || String(err),
      stage: "provider",
    });
  }

  const model = synthesisModel();
  const federationIds = getFederationModels();

  // Assemble the prompt: knowledge-pack context (if any) + description +
  // followups. The pack step is a pure retrieval-side scoring — no LLM
  // call — so it's safe to do before hitting the network.
  let packContext = "";
  let packsUsed = [];
  try {
    const haystack = [description, ...cleanFollowups].join("\n");
    const packs = selectPacks(haystack, { budget: PACK_BUDGET_TOKENS });
    packContext = buildPackContext(packs.map((p) => p.id));
    packsUsed = packs.map((p) => ({ id: p.id }));
  } catch (err) {
    // Non-fatal — synthesise without packs if the retrieval crashed.
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

  try {
    const synthesis = await provider.chat({
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userText }],
      model,
      maxTokens: 2048,
    });

    const floor = aggregateFloor([...federationIds, model]);
    const meta = federationMetadata(federationIds);

    return res.status(200).json({
      ok: true,
      synthesis,
      model,
      floor,
      federation: meta,
      packs_used: packsUsed,
    });
  } catch (err) {
    return res.status(502).json({
      ok: false,
      error: err.message || String(err),
      stage: "chat",
    });
  }
}
