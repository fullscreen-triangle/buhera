/* ============================================================================
 * The surface's own verbs — pure parsing, no dispatch.
 *
 * The work verbs (mail, find, plan, apphub) are at the end of the file.
 *
 * "chart" / "plot" / "graph" / "visualise" needs the
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

// ── the work verbs: mail, find, plan, apphub ──────────────────────────────
//
// One line only: a script of several lines is a script, not a verb. Each
// verb names a module and its instruction; resolve.js dispatches it.
//
//   mail <query> | email <query> | inbox     search your mail (inbox: the newest)
//   find <words> | search <words>            find in mail, files, the web, your plans
//   plan [experiment|task] <title>           a new plan item ("plan experiment …")
//   plans | plan                             the plan board
//   apphub | lattice                         jobs on AppHub

const MAIL_VERB = /^(?:mail|email|e-mail)\b\s*(.*)$/i;
const FIND_VERB = /^(?:find|search)\s+(.+)$/i;
const PLAN_VERB = /^plan\s+(?:(?:an?\s+)?(experiment|task)\b\s*:?\s*)?(.+)$/i;

export function workVerb(utterance) {
  const s = String(utterance).trim();
  if (!s || s.includes("\n")) return null;
  if (/^inbox$/i.test(s)) return { module: "mail", instruction: { kind: "search", query: "" } };
  let m = s.match(MAIL_VERB);
  if (m) return { module: "mail", instruction: m[1].trim() ? { kind: "search", query: m[1].trim() } : "accounts" };
  m = s.match(FIND_VERB);
  if (m) return { module: "planning", instruction: { kind: "find", query: m[1].trim() } };
  if (/^plans?$/i.test(s)) return { module: "planning", instruction: "board" };
  m = s.match(PLAN_VERB);
  if (m) return { module: "planning", instruction: { kind: "new", type: (m[1] || "task").toLowerCase(), title: m[2].trim() } };
  if (/^(?:apphub|lattice)$/i.test(s)) return { module: "lattice", instruction: "show" };
  return null;
}
