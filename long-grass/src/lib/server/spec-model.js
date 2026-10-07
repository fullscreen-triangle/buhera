/* ============================================================================
 * A specification's model, read from what the specification itself states.
 *
 * Two sources, one shape:
 *
 *   property tables   an HTML specification (DCAT-AP and the other SEMIC
 *                     profiles) where each class has a table with Property,
 *                     Range and Card(inality) columns; the class is the
 *                     section's heading, a range linked inside the page is
 *                     another class
 *   LinkML            a LinkML schema (DCAT-AP+): classes with is_a/mixins,
 *                     slots with range/required/multivalued, slot_usage
 *                     overriding them per class
 *
 *   model { source, kind, title, classes: [{
 *     name, label, anchor, uri, description, isA: [], abstract, notes: [],
 *     properties: [{ name, label, uri, range, rangeClass, min, max, card,
 *                    definition, notes: [] }] }] }
 *
 * Nothing is inferred beyond the source: a property's obligation is its
 * minimum cardinality (`required` in LinkML), a range is a class only when
 * the source links or names one. lib/spec/ draws and compares models.
 * ========================================================================== */

import * as cheerio from "cheerio";
import yaml from "js-yaml";

const clean = (s) => String(s || "").replace(/\s+/g, " ").trim();

export function parseCard(card) {
  const c = clean(card).replace(/\s/g, "");
  const m = /^(\d+)(?:\.\.(\d+|\*|n|N))?$/.exec(c);
  if (!m) return { min: 0, max: null, card: c || "?" };
  const min = Number(m[1]);
  const max = m[2] === undefined ? min : /^\d+$/.test(m[2]) ? Number(m[2]) : "*";
  return { min, max, card: m[2] === undefined ? String(min) : `${min}..${max}` };
}

// ── HTML with property tables ────────────────────────────────────────────

function headerCols($, table) {
  const cells = $(table).find("thead th, thead td").toArray();
  const head = (cells.length ? cells : $(table).find("tr").first().find("th, td").toArray()).map((c) => clean($(c).text()).toLowerCase());
  const at = (re) => head.findIndex((h) => re.test(h));
  return { property: at(/^property/), range: at(/^range/), card: at(/^card/), definition: at(/^definition/), usage: at(/^usage/) };
}

export function fromPropertyTables(html, url) {
  const $ = cheerio.load(html);
  $("*").contents().filter((_, n) => n.type === "comment").remove();
  const title = clean($("title").first().text() || $("h1").first().text());
  const classes = new Map();

  function classFor(table) {
    // The nearest heading before the table, inside the same section.
    const section = $(table).closest("section");
    const h = section.length ? section.children("h2, h3, h4, .header-wrapper").first() : $(table).prevAll("h2, h3, h4").first();
    const head = h.is(".header-wrapper") ? h.find("h2, h3, h4").first() : h;
    const label = clean(head.text()).replace(/^[\d.]+\s+/, "").replace(/[¶§]\s*$/, "");
    const anchor = section.attr("id") || head.attr("id") || label.replace(/\W+/g, "");
    const dl = section.find("dl").first();
    const field = (name) => {
      const dt = dl.children("dt").filter((_, d) => clean($(d).text()).toLowerCase() === name).first();
      return dt.length ? clean(dt.next("dd").text()) : "";
    };
    return { name: anchor, label, anchor, uri: null, description: field("definition"), isA: field("subclass of") ? [field("subclass of")] : [], abstract: false, notes: [], properties: [], usageNote: field("usage note") };
  }

  $("table").each((_, table) => {
    const cols = headerCols($, table);
    if (cols.property < 0 || cols.range < 0 || cols.card < 0) return;
    const cls = classFor(table);
    const known = classes.get(cls.name) || cls;
    classes.set(cls.name, known);
    $(table).find("tbody tr").each((__, tr) => {
      const td = $(tr).children("td").toArray();
      if (td.length <= Math.max(cols.property, cols.range, cols.card)) return;
      const pa = $(td[cols.property]).find("a").first();
      const ra = $(td[cols.range]).find("a").first();
      const rHref = ra.attr("href") || "";
      const { min, max, card } = parseCard($(td[cols.card]).text());
      known.properties.push({
        name: ($(tr).attr("id") || "").split(".").slice(1).join(".") || clean($(td[cols.property]).text()),
        label: clean($(td[cols.property]).text()),
        uri: $(tr).attr("resource") || pa.attr("href") || null,
        anchor: $(tr).attr("id") || null,
        range: clean($(td[cols.range]).text()),
        rangeClass: rHref.startsWith("#") ? rHref.slice(1) : null,
        min, max, card,
        definition: cols.definition >= 0 ? clean($(td[cols.definition]).text()) : "",
        notes: cols.usage >= 0 && clean($(td[cols.usage]).text()) ? [clean($(td[cols.usage]).text())] : [],
      });
    });
  });

  // Sections that declare a class with no table of its own ("does not impose
  // any additional requirements") are classes too, when something ranges over them.
  // A range that names a literal type is a datatype, not a class: Literal,
  // xsd:*, or an anchor that sits in a table cell (the Datatypes table)
  // rather than on a section or heading.
  const all = [...classes.values()];
  const named = new Set(all.map((c) => c.name));
  const isDatatype = (id, el) => /^literal$/i.test(id) || /^xsd(%3A|:)/i.test(id) || el.closest("td").length > 0;
  for (const c of all) {
    for (const p of c.properties) {
      if (!p.rangeClass || named.has(p.rangeClass)) continue;
      const el = $(`[id="${p.rangeClass}"]`).first();
      if (!el.length || isDatatype(p.rangeClass, el)) { p.rangeClass = null; continue; }
      const sec = el.is("section") ? el : el.closest("section");
      const label = clean((el.is("h2, h3, h4") ? el : sec.find("h2, h3, h4").first()).text());
      all.push({ name: p.rangeClass, label: label || p.rangeClass, anchor: p.rangeClass, uri: null, description: clean(sec.find("dd").first().text()), isA: [], abstract: false, notes: [], properties: [] });
      named.add(p.rangeClass);
    }
  }
  return { source: url, kind: "property-tables", title, classes: all };
}

// ── LinkML ───────────────────────────────────────────────────────────────

function expand(curie, prefixes) {
  if (!curie || /^https?:/.test(curie)) return curie || null;
  const [p, local] = curie.split(":");
  const base = prefixes[p];
  const ns = typeof base === "string" ? base : base?.prefix_reference;
  return ns && local !== undefined ? ns + local : curie;
}

export function fromLinkML(text, url) {
  const doc = yaml.load(text);
  if (!doc || typeof doc !== "object" || !doc.classes) throw new Error("not a LinkML schema: no classes");
  const prefixes = doc.prefixes || {};
  const slots = doc.slots || {};
  const classNames = new Set(Object.keys(doc.classes));
  const defaultRange = doc.default_range || "string";
  const asList = (v) => (v == null ? [] : Array.isArray(v) ? v : [v]);

  const classes = Object.entries(doc.classes).map(([name, c]) => {
    c = c || {};
    const own = [...asList(c.slots), ...Object.keys(c.attributes || {})];
    const usage = c.slot_usage || {};
    for (const k of Object.keys(usage)) if (!own.includes(k)) own.push(k);
    const properties = own.map((sn) => {
      const s = { ...(slots[sn] || {}), ...((c.attributes || {})[sn] || {}), ...(usage[sn] || {}) };
      const range = s.range || defaultRange;
      const min = s.required ? 1 : 0;
      const max = s.multivalued ? "*" : 1;
      return {
        name: sn,
        label: sn.replace(/_/g, " "),
        uri: expand(s.slot_uri || (slots[sn] || {}).slot_uri, prefixes),
        anchor: null,
        range,
        rangeClass: classNames.has(range) ? range : null,
        min, max, card: `${min}..${max}`,
        definition: clean(s.description),
        notes: [...asList((slots[sn] || {}).notes), ...asList((usage[sn] || {}).notes)].map(clean),
      };
    });
    return {
      name,
      label: name,
      anchor: name,
      uri: expand(c.class_uri, prefixes),
      description: clean(c.description),
      isA: [...asList(c.is_a), ...asList(c.mixins)],
      abstract: !!c.abstract || !!c.mixin,
      notes: asList(c.notes).map(clean),
      properties,
    };
  });
  return { source: url, kind: "linkml", title: clean(doc.title || doc.name), version: doc.version || null, classes };
}

/** Pick by content: a LinkML schema is YAML with `classes`; anything else HTML. */
export function modelFrom(text, url) {
  const s = String(text);
  const html = /<!doctype html|<html|<head|<body/i.test(s.slice(0, 4000));
  if (!html && /^classes:\s*$/m.test(s)) return fromLinkML(s, url);
  return fromPropertyTables(s, url);
}
