/**
 * FKAC federation orchestrator.
 *
 * Realizes the Federated Knapsack-Allocated Cascade paper as operational
 * code: instead of a single LLM call, run N drafts in PARALLEL, then fuse
 * them with an integration pass. Per-draft floor priors compose into an
 * aggregate floor (Theorem 5.3: joint receiver fails only when all fail),
 * which becomes the confidence badge on the result.
 *
 * Draft axis. `chatCascade` hard-codes each provider's model from env and
 * takes no model override, so the honest parallel axis available today is
 * the set of AVAILABLE PROVIDERS — each drafts once with the model it
 * actually used (res.model), and the FKAC floor attaches to that real model
 * id. `draftFn`/`integrateFn` are injectable so the orchestrator is unit-
 * testable without live LLMs, and so a future per-model router swaps in
 * without touching the fusion logic.
 *
 * This module is generation-only: it returns a candidate string. Validation
 * (against the DSL's real compiler) and the repair loop live in
 * dsl-generator.js.
 */

import { chatCascade, availableProviders } from "@/lib/server/llm-cascade";
import {
  aggregateFloor,
  confidenceFromFloor,
  floorOf,
  getIntegrationModel,
} from "@/lib/purpose/federation";

/**
 * Default draft function: one schema-constrained call restricted to a single
 * provider. Returns { ok, provider, model, content } — `content` is the
 * raw model output (a JSON string when jsonSchema is set).
 */
async function defaultDraftFn({ provider, system, user, jsonSchema, maxTokens, temperature }) {
  const res = await chatCascade({
    system,
    user,
    jsonSchema,
    maxTokens,
    temperature,
    only: [provider],
  });
  return res;
}

/**
 * Default integration function: fuse the surviving drafts into one candidate.
 * Uses the full cascade (no `only`) so the strongest available provider does
 * the integration. `getIntegrationModel()` is recorded in metadata but the
 * cascade picks the actual model per provider env.
 */
async function defaultIntegrateFn({ system, user, jsonSchema, maxTokens, temperature }) {
  return chatCascade({ system, user, jsonSchema, maxTokens, temperature });
}

/**
 * Run the federation.
 *
 * @param {object}   args
 * @param {string}   args.system         system prompt (grounding + task)
 * @param {string}   args.user           user prompt (the instructions)
 * @param {object}   [args.jsonSchema]   schema forcing structured output, e.g. { code }
 * @param {string[]} [args.providers]    draft axis; defaults to availableProviders()
 * @param {number}   [args.maxTokens]
 * @param {number}   [args.temperature]
 * @param {Function} [args.draftFn]      injectable draft fn (testing / future router)
 * @param {Function} [args.integrateFn]  injectable integration fn
 * @param {Function} [args.extract]      map a draft result -> candidate string
 * @returns {Promise<{ok, content, drafts, federation, error?}>}
 */
export async function runFederation({
  system,
  user,
  jsonSchema,
  providers,
  maxTokens = 1024,
  temperature = 0.3,
  draftFn = defaultDraftFn,
  integrateFn = defaultIntegrateFn,
  extract = defaultExtract,
}) {
  const axis = providers && providers.length ? providers : availableProviders();
  if (axis.length === 0) {
    return {
      ok: false,
      error:
        "no LLM provider configured. Set one of OLLAMA_URL, GEMINI_API_KEY, OPENAI_API_KEY.",
      stage: "provider",
      drafts: [],
      federation: null,
    };
  }

  // ── Parallel drafting ────────────────────────────────────────────────────
  const settled = await Promise.all(
    axis.map((provider) =>
      draftFn({ provider, system, user, jsonSchema, maxTokens, temperature })
        .then((res) => ({ provider, res }))
        .catch((err) => ({ provider, res: { ok: false, error: String(err) } }))
    )
  );

  const drafts = settled.map(({ provider, res }) => ({
    provider,
    ok: !!res.ok,
    model: res.model || null,
    content: res.ok ? extract(res) : null,
    error: res.ok ? null : res.error || "draft failed",
  }));

  const survivors = drafts.filter((d) => d.ok && d.content != null && d.content !== "");
  if (survivors.length === 0) {
    // Surface the per-draft reasons so failures are diagnosable without a
    // manual probe (e.g. an upstream 429 quota error on every provider).
    const reasons = drafts.map((d) => `${d.provider}: ${d.error || "empty draft"}`).join("; ");
    return {
      ok: false,
      error: `all federation drafts failed (${reasons})`,
      stage: "draft",
      drafts,
      federation: null,
    };
  }

  // ── FKAC floor over the surviving models ─────────────────────────────────
  const survivingModels = survivors.map((d) => d.model).filter(Boolean);
  const floor = aggregateFloor(survivingModels);
  const federation = {
    draft_models: survivingModels,
    integration_model: getIntegrationModel(),
    per_model_floors: survivingModels.map(floorOf),
    aggregate_floor: floor,
    confidence: confidenceFromFloor(floor),
    draft_count: survivors.length,
  };

  // ── Integration ──────────────────────────────────────────────────────────
  // A single surviving draft needs no fusion — return it directly with the
  // floor metadata. Multiple survivors are fused by the integration pass.
  if (survivors.length === 1) {
    return { ok: true, content: survivors[0].content, drafts, federation };
  }

  const integrationUser = buildIntegrationPrompt(user, survivors);
  const integ = await integrateFn({
    system,
    user: integrationUser,
    jsonSchema,
    maxTokens,
    temperature,
  });

  if (!integ.ok) {
    // Integration failed — fall back to the lowest-floor surviving draft so
    // the caller still gets a usable candidate.
    const best = survivors
      .map((d) => ({ d, f: d.model ? floorOf(d.model) : 100 }))
      .sort((a, b) => a.f - b.f)[0].d;
    return {
      ok: true,
      content: best.content,
      drafts,
      federation: { ...federation, integration_fell_back: true },
    };
  }

  return { ok: true, content: extract(integ), drafts, federation };
}

/**
 * Default extractor: when a jsonSchema with a `code` field was used, the
 * content is a JSON string `{ "code": "..." }`; pull the code out. Otherwise
 * return the raw content trimmed.
 */
function defaultExtract(res) {
  const raw = res.content;
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  try {
    const obj = JSON.parse(trimmed);
    if (obj && typeof obj.code === "string") return obj.code;
  } catch {
    // Not JSON — fall through to raw. Some providers wrap code in fences.
  }
  return stripCodeFence(trimmed);
}

/** Remove a leading/trailing ```lang fence if the model added one. */
function stripCodeFence(text) {
  const fence = /^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$/;
  const m = fence.exec(text.trim());
  return m ? m[1] : text;
}

/** Build the integration prompt that fuses the surviving drafts. */
function buildIntegrationPrompt(originalUser, survivors) {
  const blocks = survivors
    .map(
      (d, i) =>
        `--- Draft ${i + 1} (from ${d.provider}${d.model ? `, ${d.model}` : ""}) ---\n${d.content}`
    )
    .join("\n\n");
  return [
    "You are integrating several candidate solutions into one final answer.",
    "The original request was:",
    "",
    originalUser,
    "",
    "Here are the candidate drafts:",
    "",
    blocks,
    "",
    "Produce the single best result, correcting any errors and combining the",
    "strongest elements. Return it in the same required output format.",
  ].join("\n");
}
