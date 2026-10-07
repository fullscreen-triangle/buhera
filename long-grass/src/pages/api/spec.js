// API route for understanding a specification.
//
//   POST /api/spec { action: "model", url }                  → the classes and properties the
//                                                              specification states (lib/server/spec-model.js)
//   POST /api/spec { action: "draft", request, context, previous?, error? }
//                                                            → a Mermaid flowchart drafted by a model from
//                                                              your notes; the browser checks it parses
//
// A model is read from the page as kept in the library (read it first, or
// this reads it now). Models are cached in memory per page and read date.
// Same access as /api/web: this machine, or a signed-in member.

import { allowed } from "@/lib/server/session";
import { libraryDir, readKept, readUrl } from "@/lib/server/web";
import { modelFrom } from "@/lib/server/spec-model";
import { chatCascade } from "@/lib/server/llm-cascade";

const cache = new Map(); // url@read → model

const SYSTEM = [
  "You draw workflows as Mermaid flowcharts.",
  "Answer with Mermaid source only: no prose, no code fences.",
  "Start with `flowchart LR`. Use short node ids (letters and digits) and put labels in quotes, e.g. A[\"Plan the run\"].",
  "Activities are rounded: A([\"…\"]). Instruments and people are hexagons: D{{\"…\"}}. Data are rectangles.",
  "Label arrows with the verb that links them: A -->|generates| B.",
  "Use only what the notes say. If the notes do not say something, leave it out rather than invent it.",
].join("\n");

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "method not allowed" });
  const who = await allowed(req);
  if (!who.ok) return res.status(who.status).json({ ok: false, error: who.error });
  const { action, url, request, context, previous, error } = req.body ?? {};

  try {
    if (action === "model") {
      const dir = libraryDir();
      let kept = readKept(url, dir);
      if (!kept) {
        await readUrl(url, { local: who.local, dir });
        kept = readKept(url, dir);
      }
      if (!kept?.raw) return res.status(404).json({ ok: false, error: `${url} could not be read` });
      const key = `${kept.url}@${kept.read}`;
      if (!cache.has(key)) cache.set(key, modelFrom(kept.raw, kept.url));
      const model = cache.get(key);
      if (!model.classes.length) {
        return res.status(422).json({ ok: false, error: "this page states no classes: no property tables (Property, Range, Card) and no LinkML schema" });
      }
      return res.status(200).json({ ok: true, model });
    }

    if (action === "draft") {
      if (!String(request || "").trim()) return res.status(400).json({ ok: false, error: "what should the diagram show?" });
      const user = [
        `Draw: ${request}`,
        context ? `\nThe notes and sources to draw from:\n${String(context).slice(0, 12_000)}` : "",
        previous && error ? `\nYour previous answer did not parse as Mermaid (${error}). Correct it:\n${previous}` : "",
      ].join("\n");
      const r = await chatCascade({ system: SYSTEM, user, temperature: 0.2, maxTokens: 1200 });
      if (!r.ok) return res.status(502).json({ ok: false, error: r.error || "no model answered" });
      const mermaid = String(r.content).replace(/^```(?:mermaid)?\s*/i, "").replace(/```\s*$/, "").trim();
      return res.status(200).json({ ok: true, mermaid, provider: r.provider, model: r.model });
    }

    return res.status(400).json({ ok: false, error: `unknown action "${action}"` });
  } catch (e) {
    return res.status(502).json({ ok: false, error: e.message || String(e) });
  }
}
