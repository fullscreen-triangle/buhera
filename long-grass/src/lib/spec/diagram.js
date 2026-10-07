/* ============================================================================
 * Drawing a specification's model (lib/server/spec-model.js) as Mermaid.
 *
 *   classDiagram(model, { focus, depth, attributes, highlight })
 *       the classes around `focus` (every class if none), their mandatory
 *       (or all) data properties as members, object properties as labelled
 *       associations with cardinality, and is-a as inheritance
 *
 *   flowDiagram(model, { highlight })
 *       the workflow a model describes, read from its PROV-O terms: what an
 *       activity used (prov:used), who or what carried it out
 *       (prov:wasAssociatedWith), where (prov:atLocation), which activity
 *       informed it (prov:wasInformedBy), and what it generated
 *       (prov:wasGeneratedBy / prov:generated). Inputs on the left, outputs
 *       on the right.
 *
 * Both are pure and return Mermaid text, so a diagram can be shown, copied,
 * edited and kept like any other text. Nothing is drawn that the model does
 * not state.
 * ========================================================================== */

const PROV = "http://www.w3.org/ns/prov#";
const byName = (model) => new Map(model.classes.map((c) => [c.name, c]));
const id = (name) => String(name).replace(/[^A-Za-z0-9_]/g, "_");
const quote = (s) => String(s).replace(/"/g, "'");
const cardText = (p) => p.card.replace(/\*/g, "n");

/** The class's ancestors, nearest first. */
export function ancestors(model, name, map = byName(model)) {
  const out = [];
  const queue = [...(map.get(name)?.isA || [])];
  while (queue.length) {
    const n = queue.shift();
    if (out.includes(n)) continue;
    out.push(n);
    queue.push(...(map.get(n)?.isA || []));
  }
  return out;
}

/** Classes within `depth` association/inheritance steps of `focus`. */
export function neighbourhood(model, focus, depth = 1) {
  const map = byName(model);
  if (!focus || !map.has(focus)) return new Set(map.keys());
  const keep = new Set([focus]);
  let frontier = [focus];
  for (let d = 0; d < depth; d++) {
    const next = [];
    for (const n of frontier) {
      const c = map.get(n);
      const out = [...(c?.isA || []), ...(c?.properties || []).map((p) => p.rangeClass).filter(Boolean)];
      const into = model.classes.filter((k) => k.isA.includes(n) || k.properties.some((p) => p.rangeClass === n)).map((k) => k.name);
      for (const m of [...out, ...(d === 0 ? into : [])]) {
        if (map.has(m) && !keep.has(m)) { keep.add(m); next.push(m); }
      }
    }
    frontier = next;
  }
  return keep;
}

export function classDiagram(model, { focus = null, depth = 1, attributes = "mandatory", highlight = [], direction = "LR" } = {}) {
  const map = byName(model);
  const keep = neighbourhood(model, focus, depth);
  const marked = new Set(highlight);
  const lines = ["classDiagram", `  direction ${direction}`];
  for (const name of keep) {
    const c = map.get(name);
    const members = c.properties
      .filter((p) => !p.rangeClass && (attributes === "all" || (attributes === "mandatory" && p.min > 0)))
      .map((p) => `    +${id(p.name)} ${quote(p.range)} [${cardText(p)}]`);
    const label = c.label && c.label !== name ? `["${quote(c.label)}"]` : "";
    lines.push(members.length ? `  class ${id(name)}${label} {\n${members.join("\n")}\n  }` : `  class ${id(name)}${label}`);
  }
  for (const name of keep) {
    const c = map.get(name);
    for (const sup of c.isA) if (keep.has(sup)) lines.push(`  ${id(sup)} <|-- ${id(name)}`);
    // Only the focus's own associations at depth 1 when focused, every association otherwise:
    // a focused diagram is about the focus, not about its neighbours' neighbours.
    if (focus && depth === 1 && name !== focus) continue;
    for (const p of c.properties) {
      if (p.rangeClass && keep.has(p.rangeClass)) lines.push(`  ${id(name)} --> "${p.card}" ${id(p.rangeClass)} : ${quote(p.label || p.name)}`);
    }
  }
  const hl = [...keep].filter((n) => marked.has(n));
  if (hl.length) {
    lines.push("  classDef added fill:#0f3d36,stroke:#2dd4bf,color:#e8ecef");
    lines.push(`  cssClass "${hl.map(id).join(",")}" added`);
  }
  if (focus && keep.has(focus)) {
    lines.push("  classDef focus stroke:#e8ecef,stroke-width:2px");
    if (!marked.has(focus)) lines.push(`  cssClass "${id(focus)}" focus`);
  }
  return lines.join("\n");
}

// ── the workflow, from PROV-O ────────────────────────────────────────────

const ROLE = {
  [`${PROV}used`]: "used",
  [`${PROV}wasGeneratedBy`]: "generatedBy",
  [`${PROV}generated`]: "generated",
  [`${PROV}wasAssociatedWith`]: "carriedOutBy",
  [`${PROV}atLocation`]: "at",
  [`${PROV}wasInformedBy`]: "informedBy",
};

function kindOf(model, name, map) {
  const chain = [name, ...ancestors(model, name, map)];
  const uris = chain.map((n) => map.get(n)?.uri || "");
  const has = (local, word) => uris.some((u) => u === PROV + local) || chain.some((n) => n === word);
  if (has("Plan", "Plan")) return "plan";
  if (has("Activity", "Activity")) return "activity";
  if (has("Agent", "Agent") || chain.includes("AgenticEntity")) return "agent";
  if (has("Location", "Location")) return "place";
  return "entity";
}

const SHAPE = {
  activity: (n, l) => `${n}(["${l}"])`,
  agent: (n, l) => `${n}{{"${l}"}}`,
  plan: (n, l) => `${n}[/"${l}"/]`,
  place: (n, l) => `${n}[("${l}")]`,
  entity: (n, l) => `${n}["${l}"]`,
};

/** → { mermaid, edges: [{ from, to, role, via }] } — edges is what the diagram states, for citing. */
export function flowEdges(model) {
  const map = byName(model);
  const edges = [];
  for (const c of model.classes) {
    for (const p of c.properties) {
      const role = ROLE[p.uri];
      if (!role || !p.rangeClass) continue;
      const via = `${c.name}.${p.name}`;
      if (role === "generatedBy") edges.push({ from: p.rangeClass, to: c.name, role, via, label: p.label || p.name });
      else if (role === "generated") edges.push({ from: c.name, to: p.rangeClass, role, via, label: p.label || p.name });
      else edges.push({ from: p.rangeClass, to: c.name, role, via, label: p.label || p.name });
    }
  }
  return edges;
}

export function flowDiagram(model, { highlight = [], direction = "LR" } = {}) {
  const map = byName(model);
  const edges = flowEdges(model);
  const nodes = new Set(edges.flatMap((e) => [e.from, e.to]));
  const lines = [`flowchart ${direction}`];
  if (!edges.length) {
    lines.push('  none["this model states no PROV-O workflow: no prov:used, prov:wasGeneratedBy, prov:wasAssociatedWith, prov:atLocation or prov:wasInformedBy between its classes"]');
    return lines.join("\n");
  }
  for (const n of nodes) {
    const c = map.get(n);
    lines.push(`  ${SHAPE[kindOf(model, n, map)](id(n), quote(c?.label || n))}`);
  }
  const arrow = { used: "-->", generatedBy: "==>", generated: "==>", carriedOutBy: "-.->", at: "-.-", informedBy: "-->" };
  for (const e of edges) lines.push(`  ${id(e.from)} ${arrow[e.role]}|${quote(e.label)}| ${id(e.to)}`);
  // Specialisations among the nodes, faintly: a DataAnalysis is a DataGeneratingActivity.
  for (const n of nodes) for (const sup of map.get(n)?.isA || []) if (nodes.has(sup)) lines.push(`  ${id(n)} -.-|is a| ${id(sup)}`);
  const hl = [...nodes].filter((n) => highlight.includes(n));
  lines.push("  classDef added fill:#0f3d36,stroke:#2dd4bf,color:#e8ecef");
  if (hl.length) lines.push(`  class ${hl.map(id).join(",")} added`);
  return lines.join("\n");
}
