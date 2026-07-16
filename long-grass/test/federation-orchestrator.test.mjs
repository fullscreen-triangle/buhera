// Unit tests for the FKAC federation orchestrator (A3), using injected
// draftFn/integrateFn so no live LLM is needed. Verifies parallel drafting,
// single-vs-multi survivor paths, floor/confidence, and graceful degradation.
import test from "node:test";
import assert from "node:assert/strict";

import { runFederation } from "../src/lib/purpose/federation-orchestrator.js";

// A draftFn that returns a fixed {code} JSON per provider, as chatCascade would.
function fakeDraft(codeByProvider, models = {}) {
  return async ({ provider }) => {
    if (!(provider in codeByProvider)) return { ok: false, provider, error: "no model" };
    return {
      ok: true,
      provider,
      model: models[provider] || `${provider}-model`,
      content: JSON.stringify({ code: codeByProvider[provider] }),
    };
  };
}

test("single surviving draft returns directly, no integration", async () => {
  let integrated = false;
  const out = await runFederation({
    system: "s",
    user: "u",
    jsonSchema: { type: "object" },
    providers: ["ollama"],
    draftFn: fakeDraft({ ollama: "memory list" }),
    integrateFn: async () => {
      integrated = true;
      return { ok: true, content: JSON.stringify({ code: "SHOULD NOT RUN" }) };
    },
  });
  assert.equal(out.ok, true);
  assert.equal(out.content, "memory list");
  assert.equal(integrated, false, "integration must be skipped for one survivor");
  assert.equal(out.federation.draft_count, 1);
  assert.ok(out.federation.confidence >= 0 && out.federation.confidence <= 1);
});

test("multiple survivors are fused by integration", async () => {
  const out = await runFederation({
    system: "s",
    user: "u",
    jsonSchema: { type: "object" },
    providers: ["ollama", "openai"],
    draftFn: fakeDraft(
      { ollama: "memory list", openai: "kernel stats" },
      { ollama: "llama3.2:3b", openai: "gpt-4o-mini" }
    ),
    integrateFn: async () => ({ ok: true, content: JSON.stringify({ code: "memory list\nkernel stats" }) }),
  });
  assert.equal(out.ok, true);
  assert.equal(out.content, "memory list\nkernel stats");
  assert.equal(out.federation.draft_count, 2);
  // Aggregate floor over two models must be <= the single-model floors
  // (federation only lowers the floor).
  assert.ok(out.federation.aggregate_floor <= Math.min(...out.federation.per_model_floors));
});

test("a failed draft is dropped; survivors still proceed", async () => {
  const out = await runFederation({
    system: "s",
    user: "u",
    providers: ["ollama", "gemini"],
    draftFn: async ({ provider }) =>
      provider === "ollama"
        ? { ok: true, provider, model: "llama3.2:3b", content: JSON.stringify({ code: "demon sort" }) }
        : { ok: false, provider, error: "upstream 500" },
    integrateFn: async () => ({ ok: true, content: "unused" }),
  });
  assert.equal(out.ok, true);
  assert.equal(out.content, "demon sort");
  assert.equal(out.federation.draft_count, 1);
});

test("all drafts failing yields ok:false at the draft stage", async () => {
  const out = await runFederation({
    system: "s",
    user: "u",
    providers: ["ollama"],
    draftFn: async ({ provider }) => ({ ok: false, provider, error: "dead" }),
  });
  assert.equal(out.ok, false);
  assert.equal(out.stage, "draft");
});

test("no providers yields ok:false at the provider stage", async () => {
  const out = await runFederation({ system: "s", user: "u", providers: [] });
  assert.equal(out.ok, false);
  assert.equal(out.stage, "provider");
});

test("integration failure falls back to lowest-floor draft", async () => {
  const out = await runFederation({
    system: "s",
    user: "u",
    providers: ["ollama", "openai"],
    draftFn: fakeDraft(
      { ollama: "memory list", openai: "kernel stats" },
      { ollama: "Qwen/Qwen2.5-7B-Instruct", openai: "Qwen/Qwen2.5-72B-Instruct" }
    ),
    integrateFn: async () => ({ ok: false, error: "integration upstream error" }),
  });
  assert.equal(out.ok, true);
  // 72B has the lower floor prior, so the openai draft should win the fallback.
  assert.equal(out.content, "kernel stats");
  assert.equal(out.federation.integration_fell_back, true);
});
