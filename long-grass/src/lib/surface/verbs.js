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
//   web <words>                              a search engine's results
//   read <url> | read site <url> [n]         read a page, or a documentation site, into the library
//   library                                  every page read
//   diagram <url> [around <Class>] [depth 2] [all] [against <url>]
//                                            a class diagram of a specification
//   workflow <url> [against <url>]           the workflow its PROV-O terms describe
//   compare <url> with <url>                 what the second specification changes about the first
//   draw <what>                              a workflow drafted by your model from your open plan
//   flowchart … / classDiagram … (Mermaid)   drawn as written — the one verb of several lines

const MAIL_VERB = /^(?:mail|email|e-mail)\b\s*(.*)$/i;
const FIND_VERB = /^(?:find|search)\s+(.+)$/i;
const PLAN_VERB = /^plan\s+(?:(?:an?\s+)?(experiment|task)\b\s*:?\s*)?(.+)$/i;

const URL_PART = String.raw`(https?:\/\/\S+)`;
const MERMAID = /^(flowchart|graph|classDiagram|sequenceDiagram|stateDiagram(?:-v2)?|erDiagram|mindmap|timeline|gantt)\b/;
const READ_SITE = new RegExp(String.raw`^read\s+(?:the\s+)?(?:site|all(?:\s+of)?)\s+${URL_PART}(?:\s+(\d+))?$`, "i");
const READ = new RegExp(String.raw`^read\s+${URL_PART}$`, "i");
const COMPARE = new RegExp(String.raw`^compare\s+${URL_PART}\s+(?:with|and|to)\s+${URL_PART}$`, "i");
const WORKFLOW = new RegExp(String.raw`^workflow\s+(?:of\s+)?${URL_PART}(?:\s+against\s+${URL_PART})?$`, "i");
const DIAGRAM = new RegExp(String.raw`^diagram\s+(?:of\s+)?${URL_PART}(.*)$`, "i");
const AGAINST = new RegExp(String.raw`\bagainst\s+${URL_PART}`, "i");

function readingVerb(s) {
  let m = s.match(READ_SITE);
  if (m) return { module: "web", instruction: { kind: "site", url: m[1], limit: m[2] ? Number(m[2]) : undefined } };
  m = s.match(READ);
  if (m) return { module: "web", instruction: { kind: "read", url: m[1] } };
  m = s.match(/^web\s+(.+)$/i);
  if (m) return { module: "web", instruction: { kind: "search", query: m[1].trim() } };
  if (/^library$/i.test(s)) return { module: "web", instruction: { kind: "library" } };
  m = s.match(COMPARE);
  if (m) return { module: "spec", instruction: { kind: "compare", a: m[1], b: m[2] } };
  m = s.match(WORKFLOW);
  if (m) return { module: "spec", instruction: { kind: "diagram", view: "flow", url: m[1], against: m[2] } };
  m = s.match(DIAGRAM);
  if (m) {
    const rest = m[2];
    return {
      module: "spec",
      instruction: {
        kind: "diagram",
        view: "classes",
        url: m[1],
        focus: /\baround\s+([A-Za-z0-9_]+)/i.exec(rest)?.[1],
        depth: Number(/\bdepth\s+(\d)/i.exec(rest)?.[1]) || undefined,
        attributes: /\ball\b/i.test(rest) ? "all" : "mandatory",
        against: AGAINST.exec(rest)?.[1],
      },
    };
  }
  m = s.match(/^draw\s+(.+)$/i);
  if (m) return { module: "spec", instruction: { kind: "draft", request: m[1].trim() } };
  return null;
}

export function workVerb(utterance) {
  const s = String(utterance).trim();
  if (MERMAID.test(s)) return { module: "spec", instruction: { kind: "mermaid", text: s } };
  if (!s || s.includes("\n")) return null;
  const reading = readingVerb(s);
  if (reading) return reading;
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
