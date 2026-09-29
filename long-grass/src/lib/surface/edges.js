/* ============================================================================
 * Edges — which modules surface at which screen edge.
 *
 * The blank surface shows nothing until the pointer reaches an edge:
 *
 *   top     streams, databases, internet connection
 *   right   cluster / distributed compute, servers, VPNs
 *   bottom  configuration: disk space, setup-config, restart, update
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
    title: "streams · data · connection",
    entries: [
      { id: "network", glance: { kind: "status" } },
      { id: "spraypaint", glance: { kind: "scenes" } },
      { id: "triangle", glance: "sources" },
      { id: "hfq" },
      // group stream and database connectors land here as they are built
    ],
  },
  right: {
    title: "compute · servers · network",
    entries: [
      { id: "gateway", glance: "catalysts" },
      { id: "catalysts", glance: "list" },
      { id: "compute" },
      { id: "srn", glance: { kind: "peers" } },
      { id: "pylon" },
      { id: "sbs-core" },
    ],
  },
  bottom: {
    title: "system",
    entries: [
      { id: "disk", glance: { kind: "usage" } },
      { id: "config", glance: { kind: "show" } },
      { id: "restart" },
      { id: "update", glance: { kind: "check" } },
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
 * Filter the full module list for the left column's search. Matches the
 * query against id and description, case-insensitively; an empty query
 * returns everything, sorted by id.
 */
export function searchModules(registered, query) {
  const q = (query || "").trim().toLowerCase();
  const sorted = [...registered].sort((a, b) => a.id.localeCompare(b.id));
  if (!q) return sorted;
  return sorted.filter(
    (m) =>
      m.id.toLowerCase().includes(q) ||
      (m.description || "").toLowerCase().includes(q)
  );
}

/**
 * Whether an instruction example is a template the user must fill in
 * (`<name>`, `"..."`) rather than something runnable as written. Picking a
 * template from a module page puts it at the caret instead of running it.
 */
export function isTemplate(instruction) {
  return /<[^>]+>|"\.\.\."|\.\.\./.test(String(instruction));
}
