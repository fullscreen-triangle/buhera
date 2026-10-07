/* ============================================================================
 * spec — understand a specification from what it states.
 *
 * A specification read with `web` (an HTML profile with property tables, like
 * DCAT-AP, or a LinkML schema, like DCAT-AP+) becomes a model: its classes,
 * their properties, ranges and obligations (pages/api/spec.js). From the
 * model, without guessing:
 *
 *   model      what it defines: classes, properties, what is mandatory
 *   diagram    a class diagram around one class, or the workflow its PROV-O
 *              terms describe (lib/spec/diagram.js)
 *   compare    what one specification changes about another — DCAT-AP+
 *              about DCAT-AP (lib/spec/compare.js)
 *
 * And, from your own words:
 *   mermaid    a diagram you wrote (paste Mermaid on the blank screen)
 *   draft      a workflow your model drafts from a plan's notes, checked to
 *              parse and labelled as the model's
 *
 * Instruction shapes:
 *   { kind: "model", url }
 *   { kind: "diagram", url, view?: "classes"|"flow", focus?, depth?, attributes?, against? }
 *   { kind: "compare", a, b }
 *   { kind: "mermaid", text, title? }
 *   { kind: "draft", request, item? }
 * ========================================================================== */

import { postJSON } from "@/lib/auth/headers";
import { classDiagram, flowDiagram } from "@/lib/spec/diagram";
import { compare } from "@/lib/spec/compare";
import { mermaidError } from "@/lib/spec/mermaid";
import { getItems, itemsInProject } from "@/lib/surface/planning";

const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });
const failed = (what, error) => done({ kind: "text", lines: [`spec ${what}: ${error}`] }, false);

const models = new Map(); // url → model, for this session
async function model(url) {
  if (models.has(url)) return { ok: true, model: models.get(url) };
  const r = await postJSON("/api/spec", { action: "model", url });
  if (r.ok) models.set(url, r.model);
  return r;
}

const summary = (m) => ({
  classes: m.classes.map((c) => ({
    name: c.name,
    label: c.label,
    anchor: c.anchor,
    isA: c.isA,
    properties: c.properties.length,
    mandatory: c.properties.filter((p) => p.min > 0).map((p) => p.label || p.name),
  })),
  properties: m.classes.reduce((s, c) => s + c.properties.length, 0),
});

// What a plan item holds, as text a model can draw from.
function itemContext(item) {
  if (!item) return "";
  const lines = [`# ${item.title}`, item.notes || ""];
  for (const s of item.steps) lines.push(`- step: ${s.text}`);
  // Notes and what was found, by title; a kept diagram by its title only —
  // its source would crowd out the notes for a small model.
  for (const r of item.refs) lines.push(`- ${r.source} ${r.cite}: ${r.title || ""}${r.note ? ` — note: ${r.note}` : ""}`);
  return lines.join("\n");
}

export const specModule = {
  id: "spec",

  describe() {
    return {
      id: "spec",
      description:
        "Understand a specification from what it states: its classes and obligations, class diagrams, the workflow " +
        "its PROV-O terms describe, and what a profile changes about the specification it extends. Read the page with `web` first.",
      instructions: [
        "diagram https://semiceu.github.io/DCAT-AP/releases/3.0.1/ around Dataset",
        "workflow https://nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml",
        "compare https://semiceu.github.io/DCAT-AP/releases/3.0.1/ with https://nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml",
        "draw the lab workflow from my notes",
      ],
    };
  },

  async execute(inst) {
    inst = inst || {};
    switch (inst.kind) {
      case "model": {
        const r = await model(inst.url);
        if (!r.ok) return failed("model", r.error);
        return done({ kind: "spec_model", url: inst.url, title: r.model.title, version: r.model.version || null, source: r.model.kind, ...summary(r.model) });
      }
      case "diagram": {
        const r = await model(inst.url);
        if (!r.ok) return failed("diagram", r.error);
        const m = r.model;
        const view = inst.view === "flow" ? "flow" : "classes";
        let highlight = [];
        if (inst.against) {
          const base = await model(inst.against);
          if (base.ok) highlight = compare(base.model, m).addedClasses.map((c) => c.name);
        }
        const focus = view === "classes" ? inst.focus || (m.classes.some((c) => c.name === "Dataset") ? "Dataset" : null) : null;
        const depth = Number(inst.depth) || 1;
        const attributes = inst.attributes || "mandatory";
        const mermaid = view === "flow" ? flowDiagram(m, { highlight }) : classDiagram(m, { focus, depth, attributes, highlight });
        return done({
          kind: "spec_diagram", url: inst.url, title: m.title, view, focus, depth, attributes, against: inst.against || null, highlight,
          classes: m.classes.map((c) => c.name), mermaid, by: "spec",
        });
      }
      case "compare": {
        const [a, b] = await Promise.all([model(inst.a), model(inst.b)]);
        if (!a.ok) return failed("compare", `${inst.a}: ${a.error}`);
        if (!b.ok) return failed("compare", `${inst.b}: ${b.error}`);
        const diff = compare(a.model, b.model);
        const highlight = diff.addedClasses.map((c) => c.name);
        return done({ kind: "spec_compare", aUrl: inst.a, bUrl: inst.b, ...diff, flow: flowDiagram(b.model, { highlight }), flowBase: flowDiagram(a.model) });
      }
      case "mermaid": {
        const error = await mermaidError(inst.text);
        return done({ kind: "spec_diagram", title: inst.title || "your diagram", mermaid: inst.text, by: "you", error });
      }
      case "draft": {
        const open = itemsInProject(getItems()).filter((i) => i.status !== "done" && i.status !== "dropped");
        const item = (inst.item && getItems().find((i) => i.id === inst.item)) || open[0] || null;
        const context = itemContext(item);
        let r = await postJSON("/api/spec", { action: "draft", request: inst.request, context });
        if (!r.ok) return failed("draft", r.error);
        let error = await mermaidError(r.mermaid);
        let attempts = 1;
        while (error && attempts < 3) {
          const again = await postJSON("/api/spec", { action: "draft", request: inst.request, context, previous: r.mermaid, error });
          if (!again.ok) break;
          r = again;
          error = await mermaidError(r.mermaid);
          attempts++;
        }
        return done({ kind: "spec_diagram", title: inst.request, mermaid: r.mermaid, by: "model", provider: r.provider, model: r.model, attempts, error, from: item ? { id: item.id, title: item.title } : null });
      }
      default:
        return failed("", `unknown kind "${inst.kind}"`);
    }
  },

  outputCell() {
    return { kind: "spec_cell" };
  },
};
