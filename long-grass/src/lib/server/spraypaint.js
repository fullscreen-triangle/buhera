/* ============================================================================
 * The pure half of the spraypaint bridge (pages/api/spraypaint.js spawns the
 * CLI; this file decides what to ask it and reads what it says).
 *
 * Contract: spraypaint 0.2.0 (graffiti/specifications.md). Every subcommand
 * takes --json; an ask carries a coverage verdict (covered | partial |
 * declined) and per passage the evidence lines and matched terms. Older
 * builds print text for identity/count/scenes/verify and have no verdict —
 * the text parsers below are kept for them, and a result without `coverage`
 * is passed through so the renderer can say it has no verdict.
 *
 * Root policy. The route runs in a browser-facing server, and spraypaint
 * prints file contents. So:
 *   - a local request (this machine, not through a proxy) may name any root
 *     and rebuild the index;
 *   - a remote request searches SPRAYPAINT_ROOT only, and may not index.
 *     With no SPRAYPAINT_ROOT set, remote search is refused rather than
 *     falling back to the server's parent directory.
 * ========================================================================== */

import path from "path";

export const DEFAULT_BUDGET = 8;

/**
 * Which tree this request may search.
 * → { root } or { error, status }
 */
export function resolveRoot({ requested, local, envRoot, fallback, action = "ask" }) {
  const asked = typeof requested === "string" && requested.trim() ? requested.trim() : null;
  if (local) return { root: asked || envRoot || fallback };
  if (action === "index") {
    return { status: 403, error: "rebuilding the search index is only allowed from the machine the server runs on" };
  }
  if (!envRoot) {
    return { status: 403, error: "search is not configured on this server (SPRAYPAINT_ROOT is unset)" };
  }
  if (asked && path.resolve(asked) !== path.resolve(envRoot)) {
    return { status: 403, error: "this server searches one tree only; a root cannot be chosen remotely" };
  }
  return { root: envRoot };
}

/** argv for one ask. `dryRun` previews: the verdict and passages, no commit. */
export function askArgs({ query, root, budget, scenes, dryRun }) {
  const k = Number.isFinite(budget) ? Math.max(1, Math.floor(budget)) : DEFAULT_BUDGET;
  const args = ["ask", String(query), "--root", root, "--json", "-k", String(k)];
  if (Array.isArray(scenes) && scenes.length > 0) args.push("--scenes", scenes.join(","));
  if (dryRun) args.push("--dry-run");
  return args;
}

// `index --json` prints a progress line before the JSON on some builds.
export function parseJsonLoose(stdout) {
  const s = String(stdout || "");
  const start = s.search(/[[{]/);
  try {
    return JSON.parse(start >= 0 ? s.slice(start) : s);
  } catch {
    return null;
  }
}

// ── identity / count / scenes / verify, JSON first, text as fallback ─────

export function readIdentity(stdout) {
  const j = parseJsonLoose(stdout);
  if (j && typeof j === "object" && "fingerprint" in j) {
    return { fingerprint: j.fingerprint, chi: j.char_invariant, floor: j.floor, vertices: j.n_vertices, edges: j.n_edges };
  }
  const t = String(stdout);
  const num = (re) => { const m = re.exec(t)?.[1]; return m != null ? Number(m) : null; };
  return {
    fingerprint: /fingerprint:\s*(\S+)/.exec(t)?.[1] ?? null,
    chi: num(/chi\):\s*([\d.eE+-]+)/),
    floor: num(/floor:\s*([\d.eE+-]+)/),
    vertices: num(/vertices:\s*(\d+)/),
    edges: num(/edges:\s*(\d+)/),
  };
}

export function readCount(stdout) {
  const j = parseJsonLoose(stdout);
  if (j && typeof j.committed_count === "number") return j.committed_count;
  const n = /committed acts:\s*(\d+)/.exec(String(stdout))?.[1];
  return n != null ? Number(n) : null;
}

export function readScenes(stdout) {
  const j = parseJsonLoose(stdout);
  if (Array.isArray(j)) return j.map((s) => ({ scene: s.name, documents: s.documents, passages: s.passages }));
  const rows = [];
  for (const line of String(stdout).split("\n")) {
    const m = /^(\S.*?)\s+(\d+) doc\(s\), (\d+) passage\(s\)/.exec(line);
    if (m) rows.push({ scene: m[1].trim(), documents: Number(m[2]), passages: Number(m[3]) });
  }
  return rows;
}

const INVARIANTS = [
  ["inv1_identity", "Inv 1 · conserved identity"],
  ["inv2_count", "Inv 2 · never-resetting count"],
  ["inv3_search_not_fetch", "Inv 3 · search, not fetch"],
  ["inv4_phases", "Inv 4 · exclusive phases"],
];

/**
 * → { overall: PASS|FAIL|N/A, degeneracies[], invariants: [{ name, status, checks: [{name,status,detail}] }] }
 * Exit codes (0.2.0): 0 all pass, 1 a breach, 2 nothing failed but a check
 * was N/A or the corpus is degenerate.
 */
export function readVerify(stdout, exitCode) {
  const j = parseJsonLoose(stdout);
  if (j && j.overall) {
    return {
      overall: j.overall,
      degeneracies: j.degeneracies || [],
      invariants: INVARIANTS.filter(([k]) => j[k]).map(([k, name]) => ({
        name,
        status: j[k].status,
        checks: (j[k].checks || []).map((c) => ({ name: c.name, status: c.status, detail: c.detail })),
      })),
    };
  }
  const invariants = [];
  for (const line of String(stdout).trim().split("\n")) {
    const m = /^(Inv \d+ \S.*?)\s+\[(PASS|FAIL|N\/A)\]\s*(.*)$/.exec(line);
    if (m) invariants.push({ name: m[1].trim(), status: m[2], checks: [{ name: "", status: m[2], detail: m[3].trim() }] });
  }
  const overall = /overall:\s*(PASS|FAIL|N\/A)/.exec(String(stdout))?.[1] ?? (exitCode === 0 ? "PASS" : exitCode === 2 ? "N/A" : "FAIL");
  return { overall, degeneracies: [], invariants };
}
