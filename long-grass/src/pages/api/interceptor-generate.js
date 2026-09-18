// API route: interceptor code generation.
//
// The "generate" half of the interceptor module — turns a natural-language
// task description into a single runnable Rust or TypeScript program via the
// existing LLM cascade (llm-cascade.js). No compiler-based repair loop here
// (unlike dsl-generate.js's DSL path): general Rust/TS has no single
// authoritative validator to repair against, so a bad generation surfaces as
// a failed run instead — interceptor-run.js reports it, the caller decides
// whether to regenerate.
//
// Contract:
//   POST /api/interceptor-generate
//     body: { language: "typescript"|"rust", task: string }
//   -> { ok: true, language, code, provider, model }
//   -> { ok: false, language, error, stage? }

import { chatCascade } from "@/lib/server/llm-cascade";

const MAX_TASK_BYTES = 8 * 1024;

const SYSTEM_PROMPTS = {
  typescript:
    "You are a code generator. Given a task description, output ONE complete, " +
    "runnable TypeScript program that accomplishes it and prints its result(s) " +
    "to stdout via console.log. Use only Node.js built-ins (no npm packages, no " +
    "imports that require installation). Output ONLY the code — no markdown " +
    "fences, no explanation, no comments about what you're doing.",
  rust:
    "You are a code generator. Given a task description, output ONE complete, " +
    "runnable Rust program (must include `fn main()`) that accomplishes it and " +
    "prints its result(s) to stdout via println!. Use only the Rust standard " +
    "library (no external crates — the code is compiled with `rustc` alone, no " +
    "Cargo.toml). Output ONLY the code — no markdown fences, no explanation.",
};

function stripFences(text) {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:\w+)?\n([\s\S]*?)\n```$/);
  return fenced ? fenced[1].trim() : trimmed;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { language, task } = req.body || {};
  const lang = String(language || "typescript").toLowerCase();

  if (!SYSTEM_PROMPTS[lang]) {
    return res.status(400).json({
      ok: false,
      error: `language must be "typescript" or "rust"; got "${language}"`,
    });
  }
  if (typeof task !== "string" || !task.trim()) {
    return res.status(400).json({ ok: false, error: "task (non-empty string) is required" });
  }
  if (task.length > MAX_TASK_BYTES) {
    return res.status(413).json({ ok: false, error: "task exceeds 8 KiB" });
  }

  let result;
  try {
    result = await chatCascade({
      system: SYSTEM_PROMPTS[lang],
      user: task,
      maxTokens: 2048,
      temperature: 0.2,
    });
  } catch (err) {
    return res.status(500).json({
      ok: false,
      language: lang,
      error: `generation crashed: ${err.message || String(err)}`,
    });
  }

  if (!result.ok) {
    const status = result.stage === "provider" ? 503 : 502;
    return res.status(status).json({
      ok: false,
      language: lang,
      error: result.error,
      stage: result.stage || "upstream",
    });
  }

  return res.status(200).json({
    ok: true,
    language: lang,
    code: stripFences(result.content),
    provider: result.provider,
    model: result.model,
  });
}
