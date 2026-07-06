/* ============================================================================
 * LLM provider cascade (server-side only).
 *
 * One provider abstraction shared by every API route that needs a chat
 * completion. Preference order:
 *
 *   1. Ollama          if OLLAMA_URL is set (local dev, free)
 *   2. Gemini          if GEMINI_API_KEY is set (generous free tier)
 *   3. HuggingFace     if HUGGINGFACE_API_KEY is set
 *   4. OpenAI          if OPENAI_API_KEY is set
 *
 * Each provider exposes the same interface:
 *
 *   chat({ system, user, jsonSchema?, maxTokens?, temperature? })
 *     -> { ok: true, content, provider, model }
 *     -> { ok: false, provider, error, status? }
 *
 * `content` is a string. When `jsonSchema` is supplied the provider is
 * asked to return JSON conforming to it; parsing is left to the caller so
 * routes can add their own validation.
 *
 * Callers pick the provider they want by calling pickProvider() (no arg)
 * for the highest-priority available, or pickProvider("gemini") for a
 * specific one.
 *
 * IMPORTANT: this file must ONLY be imported from API routes / server
 * components. It uses env vars and network fetches that assume Node.
 * ========================================================================== */

const PROVIDERS = ["ollama", "gemini", "huggingface", "openai"];

// ─── Provider availability ────────────────────────────────────────────────

function isAvailable(name) {
  switch (name) {
    case "ollama":       return !!process.env.OLLAMA_URL;
    case "gemini":       return !!process.env.GEMINI_API_KEY;
    case "huggingface":  return !!process.env.HUGGINGFACE_API_KEY;
    case "openai":       return !!process.env.OPENAI_API_KEY;
    default:             return false;
  }
}

export function availableProviders() {
  return PROVIDERS.filter(isAvailable);
}

export function pickProvider(preferred) {
  if (preferred && isAvailable(preferred)) return preferred;
  for (const name of PROVIDERS) {
    if (isAvailable(name)) return name;
  }
  return null;
}

// ─── Ollama ──────────────────────────────────────────────────────────────

async function chatOllama({ system, user, jsonSchema, maxTokens, temperature }) {
  const url = process.env.OLLAMA_URL.replace(/\/$/, "") + "/api/chat";
  const model = process.env.OLLAMA_MODEL || "llama3.2:3b";
  try {
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        // Ollama's `format` option supports JSON-schema-constrained output.
        format: jsonSchema ? jsonSchema : undefined,
        options: {
          num_predict: maxTokens ?? 2048,
          temperature: temperature ?? 0.2,
        },
        stream: false,
      }),
    });
    if (!upstream.ok) {
      const details = await upstream.text().catch(() => "");
      return {
        ok: false,
        provider: "ollama",
        model,
        error: `ollama HTTP ${upstream.status}`,
        details: details.slice(0, 512),
      };
    }
    const body = await upstream.json();
    const content = body?.message?.content;
    if (typeof content !== "string") {
      return {
        ok: false,
        provider: "ollama",
        model,
        error: "ollama response missing message.content",
      };
    }
    return { ok: true, provider: "ollama", model, content: content.trim() };
  } catch (err) {
    return {
      ok: false,
      provider: "ollama",
      model,
      error: err.message || String(err),
    };
  }
}

// ─── Gemini ──────────────────────────────────────────────────────────────

async function chatGemini({ system, user, jsonSchema, maxTokens, temperature }) {
  const key = process.env.GEMINI_API_KEY;
  const model = process.env.GEMINI_MODEL || "gemini-2.0-flash";
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
  try {
    const generationConfig = {
      maxOutputTokens: maxTokens ?? 2048,
      temperature: temperature ?? 0.2,
    };
    if (jsonSchema) {
      generationConfig.responseMimeType = "application/json";
      generationConfig.responseSchema = geminifyJsonSchema(jsonSchema);
    }
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig,
      }),
    });
    if (!upstream.ok) {
      const details = await upstream.text().catch(() => "");
      return {
        ok: false,
        provider: "gemini",
        model,
        error: `gemini HTTP ${upstream.status}`,
        details: details.slice(0, 512),
      };
    }
    const body = await upstream.json();
    const content = body?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (typeof content !== "string") {
      return {
        ok: false,
        provider: "gemini",
        model,
        error: "gemini response missing candidates[0].content.parts[0].text",
      };
    }
    return { ok: true, provider: "gemini", model, content: content.trim() };
  } catch (err) {
    return {
      ok: false,
      provider: "gemini",
      model,
      error: err.message || String(err),
    };
  }
}

/**
 * Gemini's `responseSchema` is JSON-Schema-shaped but strips a few OpenAI
 * quirks (no `additionalProperties`, no `strict`, no top-level `name`).
 * If the caller passes an OpenAI-shaped schema (with { name, schema }),
 * we unwrap it and drop the incompatible keys.
 */
function geminifyJsonSchema(schema) {
  const inner = schema.schema ? schema.schema : schema;
  return stripSchemaQuirks(inner);
}

function stripSchemaQuirks(node) {
  if (Array.isArray(node)) return node.map(stripSchemaQuirks);
  if (node && typeof node === "object") {
    const out = {};
    for (const [k, v] of Object.entries(node)) {
      if (k === "additionalProperties" || k === "strict") continue;
      out[k] = stripSchemaQuirks(v);
    }
    return out;
  }
  return node;
}

// ─── HuggingFace ─────────────────────────────────────────────────────────

async function chatHuggingFace({ system, user, maxTokens, temperature }) {
  const key = process.env.HUGGINGFACE_API_KEY;
  // HF's `hf-inference` router restricts free-tier model access; the exact
  // supported list drifts. Users should set HF_MODEL explicitly to a model
  // they know is enabled on their token, or (recommended) use Gemini or
  // Ollama instead — both are configured as higher-priority in the cascade.
  const model = process.env.HF_MODEL || "Qwen/Qwen2.5-7B-Instruct";
  const url = "https://router.huggingface.co/hf-inference/v1/chat/completions";
  try {
    const upstream = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        max_tokens: maxTokens ?? 2048,
        temperature: temperature ?? 0.2,
      }),
    });
    if (!upstream.ok) {
      const details = await upstream.text().catch(() => "");
      return {
        ok: false,
        provider: "huggingface",
        model,
        error: `huggingface HTTP ${upstream.status}`,
        details: details.slice(0, 512),
      };
    }
    const body = await upstream.json();
    const content = body?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      return {
        ok: false,
        provider: "huggingface",
        model,
        error: "hf response missing choices[0].message.content",
      };
    }
    return {
      ok: true,
      provider: "huggingface",
      model,
      content: content.trim(),
    };
  } catch (err) {
    return {
      ok: false,
      provider: "huggingface",
      model,
      error: err.message || String(err),
    };
  }
}

// ─── OpenAI ──────────────────────────────────────────────────────────────

async function chatOpenAI({ system, user, jsonSchema, maxTokens, temperature }) {
  const key = process.env.OPENAI_API_KEY;
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";
  const url = "https://api.openai.com/v1/chat/completions";
  try {
    const body = {
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      max_tokens: maxTokens ?? 2048,
      temperature: temperature ?? 0.2,
    };
    if (jsonSchema) {
      body.response_format = {
        type: "json_schema",
        json_schema: jsonSchema.schema
          ? jsonSchema
          : { name: "structured", strict: true, schema: jsonSchema },
      };
    }
    const upstream = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(body),
    });
    if (!upstream.ok) {
      const details = await upstream.text().catch(() => "");
      return {
        ok: false,
        provider: "openai",
        model,
        error: `openai HTTP ${upstream.status}`,
        details: details.slice(0, 512),
      };
    }
    const b = await upstream.json();
    const content = b?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      return {
        ok: false,
        provider: "openai",
        model,
        error: "openai response missing choices[0].message.content",
      };
    }
    return { ok: true, provider: "openai", model, content: content.trim() };
  } catch (err) {
    return {
      ok: false,
      provider: "openai",
      model,
      error: err.message || String(err),
    };
  }
}

// ─── Dispatch ─────────────────────────────────────────────────────────────

/**
 * Call a specific provider by name.
 */
export async function chat(providerName, opts) {
  switch (providerName) {
    case "ollama":       return chatOllama(opts);
    case "gemini":       return chatGemini(opts);
    case "huggingface":  return chatHuggingFace(opts);
    case "openai":       return chatOpenAI(opts);
    default:
      return {
        ok: false,
        provider: providerName,
        error: `unknown provider: ${providerName}`,
      };
  }
}

/**
 * Try providers in cascade order (Ollama → Gemini → HF → OpenAI). Returns
 * the first successful result; if all fail, returns the last error.
 *
 * `only` restricts the cascade to a subset (e.g. skip HF entirely).
 * `preferJsonSchema` filters out providers that can't do structured output
 * when a schema is required.
 */
export async function chatCascade(opts) {
  const wantSchema = !!opts.jsonSchema;
  const only = opts.only ? new Set(opts.only) : null;
  const noSchemaProviders = new Set(["huggingface"]);
  const order = PROVIDERS.filter((p) => {
    if (only && !only.has(p)) return false;
    if (!isAvailable(p)) return false;
    if (wantSchema && noSchemaProviders.has(p)) return false;
    return true;
  });

  if (order.length === 0) {
    return {
      ok: false,
      provider: null,
      error:
        "no LLM provider configured. Set one of OLLAMA_URL, GEMINI_API_KEY, " +
        "HUGGINGFACE_API_KEY, OPENAI_API_KEY in .env.local.",
      stage: "provider",
    };
  }

  let lastError = null;
  for (const provider of order) {
    const res = await chat(provider, opts);
    if (res.ok) return res;
    lastError = res;
  }
  return {
    ok: lastError ? false : false,
    provider: lastError?.provider ?? null,
    error: lastError?.error ?? "cascade exhausted",
    details: lastError?.details,
    stage: "upstream",
  };
}
