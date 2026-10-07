/* ============================================================================
 * Edges — which modules surface at which screen edge.
 *
 * The blank surface shows nothing until the pointer reaches an edge:
 *
 *   top     where the session is: connected devices, this machine, the
 *           network, your mail, shared experiments, jobs on AppHub, and the
 *           runtime graph
 *   right   how the screen is used: preferences (text size, spacing),
 *           screen, code visibility, device connections (printer, pointer,
 *           screen → image)
 *   bottom  who the work is for: the personalised model, groups and
 *           projects, RAG settings, planning (find things, plan experiments
 *           and tasks), reports
 *   left    every registered module, with a search at the top
 *
 * This table is the single place that decision is written down, so it can be
 * reviewed and changed without touching any module. Modules not named here
 * still appear in the left column.
 *
 * `glance` is an optional read-only instruction dispatched when the module is
 * picked from an edge, so its page opens showing current state (sources,
 * machines, disk usage) instead of an empty card. A glance must never mutate
 * anything — picking a module from an edge is looking, not acting. Modules
 * without a safe read-only instruction get no glance; their page shows the
 * description and the actions.
 * ========================================================================== */

export const EDGES = {
  top: {
    title: "devices · machine · network · mail · experiments · apphub · runtime",
    entries: [
      { id: "devices", glance: "list" },
      { id: "machine", glance: "specs" },
      { id: "network", glance: { kind: "status" } },
      { id: "mail", glance: "accounts" },
      { id: "experiments", glance: "list" },
      { id: "lattice", glance: "show" },
      { id: "runtime", glance: "map" },
    ],
  },
  right: {
    title: "preferences · screen · code · devices",
    entries: [
      { id: "preferences", glance: "show" },
      { id: "screen", glance: "size" },
      { id: "code", glance: "show" },
      { id: "peripherals", glance: "show" },
    ],
  },
  bottom: {
    title: "model · projects · rag · planning · reports",
    entries: [
      { id: "model", glance: "show" },
      { id: "projects", glance: "show" },
      { id: "rag", glance: "show" },
      { id: "planning", glance: "board" },
      { id: "reports", glance: "show" },
    ],
  },
};

/** The edge a module is filed under, or null (left column only). */
export function edgeOf(moduleId) {
  for (const [edge, { entries }] of Object.entries(EDGES)) {
    if (entries.some((e) => e.id === moduleId)) return edge;
  }
  return null;
}

/** The glance instruction for a module, or undefined. */
export function glanceOf(moduleId) {
  for (const { entries } of Object.values(EDGES)) {
    const hit = entries.find((e) => e.id === moduleId);
    if (hit) return hit.glance;
  }
  return undefined;
}

/**
 * Resolve an edge's entries against the live registry listing: keep the
 * table's order, drop ids that are not registered (so a module that failed
 * to load does not leave a dead entry).
 *
 * @param {"top"|"right"|"bottom"} edge
 * @param {Array<{id: string}>} registered  listModules() output
 */
export function modulesAt(edge, registered) {
  const byId = new Map(registered.map((m) => [m.id, m]));
  return (EDGES[edge]?.entries || []).map((e) => byId.get(e.id)).filter(Boolean);
}

/**
 * Filter the full module list for the left column's search, case-insensitive.
 * An empty query returns everything, sorted by id. Otherwise matches rank by
 * where the query was found — the id exactly, then the id's start, then
 * anywhere in the id, then only in the description — and by id within a rank,
 * so typing a module's name puts that module first.
 */
export function searchModules(registered, query) {
  const q = (query || "").trim().toLowerCase();
  const sorted = [...registered].sort((a, b) => a.id.localeCompare(b.id));
  if (!q) return sorted;
  const rank = (m) => {
    const id = m.id.toLowerCase();
    if (id === q) return 0;
    if (id.startsWith(q)) return 1;
    if (id.includes(q)) return 2;
    if ((m.description || "").toLowerCase().includes(q)) return 3;
    return -1;
  };
  return sorted
    .map((m) => [rank(m), m])
    .filter(([r]) => r >= 0)
    .sort((a, b) => a[0] - b[0])
    .map(([, m]) => m);
}

/**
 * Whether an instruction example is a template the user must fill in
 * (`<name>`, `"..."`) rather than something runnable as written. Picking a
 * template from a module page puts it at the caret instead of running it.
 */
export function isTemplate(instruction) {
  return /<[^>]+>|"\.\.\."|\.\.\./.test(String(instruction));
}
