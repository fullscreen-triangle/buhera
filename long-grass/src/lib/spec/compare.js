/* ============================================================================
 * What one specification's model changes about another's: the classes it
 * adds, the properties it adds to classes both have, and the obligations and
 * ranges it changes. For DCAT-AP → DCAT-AP+ this is the extension, computed.
 *
 * Classes are matched by name (letters and digits only, case ignored,
 * British "licence" read as "license"), properties by URI and else by name.
 * A match the two specifications would not agree on is possible — so every
 * difference carries where it was found in each, to be checked.
 * ========================================================================== */

import { ancestors } from "@/lib/spec/diagram";

export const norm = (s) => String(s || "").toLowerCase().replace(/licence/g, "license").replace(/[^a-z0-9]/g, "");

function propIndex(c) {
  const byUri = new Map();
  const byName = new Map();
  for (const p of c.properties) {
    if (p.uri) byUri.set(p.uri, p);
    byName.set(norm(p.label || p.name), p);
    byName.set(norm(p.name), p);
  }
  return (p) => (p.uri && byUri.get(p.uri)) || byName.get(norm(p.label || p.name)) || byName.get(norm(p.name)) || null;
}

/**
 * → { addedClasses, missingClasses, changes: [{ class, added: [p], stricter, looser, narrowed }], counts }
 * `a` is the base (DCAT-AP), `b` the profile built on it (DCAT-AP+).
 */
export function compare(a, b) {
  const aBy = new Map(a.classes.map((c) => [norm(c.name), c]));
  const bBy = new Map(b.classes.map((c) => [norm(c.name), c]));
  const addedClasses = b.classes.filter((c) => !aBy.has(norm(c.name)));
  const missingClasses = a.classes.filter((c) => !bBy.has(norm(c.name)));
  const changes = [];

  for (const cb of b.classes) {
    const ca = aBy.get(norm(cb.name));
    if (!ca) continue;
    const find = propIndex(ca);
    const ch = { class: cb.name, aAnchor: ca.anchor, added: [], stricter: [], looser: [], narrowed: [] };
    for (const pb of cb.properties) {
      const pa = find(pb);
      if (!pa) {
        // A property ranging over a class the base does not have, or one the base never lists.
        ch.added.push({ name: pb.name, range: pb.range, card: pb.card, uri: pb.uri, definition: pb.definition });
        continue;
      }
      if (pb.min > pa.min) ch.stricter.push({ name: pb.name, from: pa.card, to: pb.card, aAnchor: pa.anchor });
      if (pb.min < pa.min) ch.looser.push({ name: pb.name, from: pa.card, to: pb.card, aAnchor: pa.anchor });
      if (pa.rangeClass && pb.rangeClass && norm(pa.rangeClass) !== norm(pb.rangeClass)) {
        const sub = ancestors(b, pb.rangeClass).map(norm).includes(norm(pa.rangeClass));
        ch.narrowed.push({ name: pb.name, from: pa.rangeClass, to: pb.rangeClass, subclass: sub, aAnchor: pa.anchor });
      }
    }
    if (ch.added.length || ch.stricter.length || ch.looser.length || ch.narrowed.length) changes.push(ch);
  }

  return {
    a: { title: a.title, source: a.source, kind: a.kind },
    b: { title: b.title, source: b.source, kind: b.kind, version: b.version || null },
    addedClasses: addedClasses.map((c) => ({ name: c.name, isA: c.isA, description: c.description, uri: c.uri })),
    missingClasses: missingClasses.map((c) => ({ name: c.name, label: c.label, anchor: c.anchor })),
    changes,
    counts: {
      addedClasses: addedClasses.length,
      addedProperties: changes.reduce((s, c) => s + c.added.length, 0),
      stricter: changes.reduce((s, c) => s + c.stricter.length, 0),
      narrowed: changes.reduce((s, c) => s + c.narrowed.length, 0),
    },
  };
}
