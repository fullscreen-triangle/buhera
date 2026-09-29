/* ============================================================================
 * The surface's own verbs — pure parsing, no dispatch.
 *
 * Only one so far: "chart" / "plot" / "graph" / "visualise". It needs the
 * page in view, which only the surface holds, so it cannot be resolved
 * anywhere else. "chart" charts that page; "chart records" picks a table on
 * it by name; "chart memory" | "chart audit" | "chart pages" chart a named
 * source and need no page.
 * ========================================================================== */

const VIS_VERB = /^(chart|plot|graph|visuali[sz]e|vis)\b\s*(?:this\b\s*)?(.*)$/i;
const VIS_SOURCES = new Set(["memory", "audit", "pages"]);

/**
 * The vis instruction an utterance asks for, or null if it is not a chart
 * request. `{ missing: true }` means it asked to chart the page but none is
 * in view.
 */
export function visInstruction(utterance, page) {
  const m = String(utterance).trim().match(VIS_VERB);
  if (!m) return null;
  const rest = m[2].trim().replace(/^["']|["']$/g, "");
  if (VIS_SOURCES.has(rest.toLowerCase())) return rest.toLowerCase();
  if (!page) return { missing: true };
  return { page, focus: rest };
}
