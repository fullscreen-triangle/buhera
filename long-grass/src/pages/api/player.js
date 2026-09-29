// API route: the RAG player — free text in, a validated vaHera script out.
//
// The default "player" on the blank surface. Whatever the user writes, in
// whatever words, the retrieval (four-sided-triangle's individuator over the
// user's folders) and the user's personal model (the providers and notes in
// their model settings) together write the vaHera script that starts the run.
// The script is validated by vaHera's own parser, repaired until it parses
// (lib/purpose/dsl-generator.js), and returned — the browser runs it, since
// the kernel and the runtime graph live there.
//
// Contract:
//   POST /api/player
//     body: {
//       utterance: string,
//       project?: string,                               the active project
//       rag?:   { enabled?, folders?: string[], extensions?: string[] },
//       model?: { providers?: string[], temperature?: number, instructions?: string },
//       programs?: [{ id, description }]                modules `spawn` may name
//     }
//   -> { ok: true,  script, retrieval, generation }
//   -> { ok: false, stage, error, retrieval?, generation?, script? }

import { generateDsl } from "@/lib/purpose/dsl-generator";
import { availableProviders } from "@/lib/server/llm-cascade";
import { isLocalRequest, resolveFolders, retrieve } from "@/lib/server/rag";

const MAX_UTTERANCE = 8 * 1024;
const MAX_PROGRAMS = 60;

function programLines(programs) {
  return (Array.isArray(programs) ? programs : [])
    .slice(0, MAX_PROGRAMS)
    .filter((p) => p && typeof p.id === "string" && /^\S+$/.test(p.id))
    .map((p) => `- ${p.id}: ${String(p.description || "").split(/(?<=[.:—])\s/)[0].slice(0, 140)}`);
}

// What the model is asked. The retrieved claim is context, labelled with its
// grounding status so the model can weigh it; it is never presented as fact.
function buildInstructions({ utterance, retrieval, instructions, programs }) {
  const parts = [`The user wrote: ${utterance}`];
  if (instructions && instructions.trim()) {
    parts.push("", `The user's standing notes to their model: ${instructions.trim()}`);
  }
  if (retrieval && retrieval.claim) {
    parts.push("", `Retrieved from the user's own sources (status: ${retrieval.status}):`, retrieval.claim);
  } else if (retrieval && retrieval.status === "contested" && retrieval.classes) {
    parts.push("", "The user's sources disagree; the competing readings are:",
      ...retrieval.classes.map((c) => `- ${c.representative}`));
  }
  const progs = programLines(programs);
  if (progs.length) {
    parts.push(
      "",
      "Programs `spawn PROGRAM from TARGET` may name. Each runs the registered module of",
      "that name on the target's description (the TARGET must be described first):",
      ...progs
    );
  }
  parts.push(
    "",
    "Write the vaHera script that carries this out. Name each thing the request concerns",
    "with `describe TARGET with \"...\"`, spawn the programs that do the work on it, and use",
    "memory statements to store or recall. If the request is only to store or find something,",
    "memory statements alone are right. Use only the 15 statement forms."
  );
  return parts.join("\n");
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }
  const { utterance, project = "default", rag = {}, model = {}, programs = [] } = req.body || {};
  if (typeof utterance !== "string" || !utterance.trim()) {
    return res.status(400).json({ ok: false, stage: "input", error: "utterance (non-empty string) is required" });
  }
  if (utterance.length > MAX_UTTERANCE) {
    return res.status(413).json({ ok: false, stage: "input", error: `utterance exceeds ${MAX_UTTERANCE} bytes` });
  }

  // ── 1. retrieval ─────────────────────────────────────────────────────────
  let retrieval = { status: "declined", reason: "retrieval is off (RAG settings)" };
  if (rag.enabled !== false) {
    const { folders, missing, refused } = resolveFolders(rag.folders, isLocalRequest(req));
    try {
      retrieval = await retrieve({ query: utterance, project, folders, extensions: rag.extensions });
    } catch (err) {
      retrieval = { status: "declined", reason: `retrieval failed: ${err.message || err}` };
    }
    if (missing.length) retrieval.missing = missing;
    if (refused) retrieval.refused = `${refused} folder(s) from the browser were not read: this server is not local`;
  }

  // ── 2. generation by the personal model ─────────────────────────────────
  const configured = availableProviders();
  const wanted = Array.isArray(model.providers) && model.providers.length
    ? model.providers.filter((p) => configured.includes(p))
    : configured;
  if (wanted.length === 0) {
    return res.status(200).json({
      ok: false,
      stage: "model",
      error: configured.length
        ? "none of the providers chosen in model settings is configured on this server"
        : "no model configured on this server (OLLAMA_URL, GEMINI_API_KEY or OPENAI_API_KEY)",
      retrieval,
    });
  }

  const gen = await generateDsl({
    dslId: "vahera",
    instructions: buildInstructions({ utterance, retrieval, instructions: model.instructions, programs }),
    maxRepairs: 3,
    federationOpts: {
      providers: wanted,
      temperature: typeof model.temperature === "number" ? Math.max(0, Math.min(1.5, model.temperature)) : 0.3,
    },
  });

  const generation = {
    providers: wanted,
    repairs: gen.repairs,
    attempts: gen.attempts,
    confidence: gen.federation?.confidence ?? null,
  };
  if (!gen.ok) {
    return res.status(200).json({
      ok: false,
      stage: "generation",
      error: (gen.errors || []).map((e) => e.message || String(e)).join("; ") || "generation failed",
      script: gen.code || null,
      retrieval,
      generation,
    });
  }
  return res.status(200).json({ ok: true, script: gen.code.trim(), retrieval, generation });
}
