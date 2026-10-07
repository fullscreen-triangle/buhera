/* ============================================================================
 * The pure half of the lattice bridge (pages/api/lattice.js spawns the CLI).
 *
 * lattice (pylon/lattice, 0.2) wraps a repository and one of its tasks into a
 * runnable unit for AppHub, the university's Code-Server: `wrap` writes
 * .lattice/<unit>/, commits it and pushes it to Gitea; on AppHub you
 * `git pull` and run the unit; it pushes its results to the branch
 * lattice-results/<unit>, which `results` reads back.
 *
 * lattice prints text, not JSON, in fixed formats (render.rs, results.rs);
 * these parsers read those formats and keep the raw text beside what they
 * read, so a format change shows up as missing fields, not as wrong ones.
 * Its own notes ("lattice: …") go to stderr; they are kept in order.
 * ========================================================================== */

/** argv for `plan` / `wrap` from a spec. */
export function specArgs(spec = {}) {
  const a = [];
  if (spec.task) a.push(String(spec.task));
  if (spec.name) a.push("--name", String(spec.name));
  for (const m of spec.matrix || []) a.push("--matrix", String(m));
  if (spec.each) a.push("--each", String(spec.each));
  for (const o of spec.outputs || []) a.push("--output", String(o));
  for (const e of spec.env || []) a.push("--env", String(e));
  for (const s of spec.secrets || []) a.push("--secret", String(s));
  if (spec.gpu && ["none", "recommended", "required"].includes(spec.gpu)) a.push("--gpu", spec.gpu);
  if (Number.isFinite(spec.parallel)) a.push("--parallel", String(Math.max(1, Math.floor(spec.parallel))));
  if (Number.isFinite(spec.timeout)) a.push("--timeout", String(Math.max(1, Math.floor(spec.timeout))));
  const command = Array.isArray(spec.command) ? spec.command : spec.command ? [String(spec.command)] : [];
  if (command.length) a.push("--", ...command);
  return a;
}

/** lattice's own notes from stderr, and its error if it failed. */
export function readNotes(stderr) {
  const notes = [];
  let error = null;
  for (const line of String(stderr || "").split(/\r?\n/)) {
    const m = /^lattice: (error: )?(.*)$/.exec(line);
    if (!m) continue;
    if (m[1]) error = m[2];
    else notes.push(m[2]);
  }
  return { notes, error };
}

export function parseTasks(stdout) {
  return String(stdout || "")
    .split(/\r?\n/)
    .map((l) => /^(\S+)\s+(.+)$/.exec(l))
    .filter(Boolean)
    .map((m) => ({ source: m[1], name: m[2].trim() }));
}

export function parseUnits(stdout) {
  const out = [];
  for (const l of String(stdout || "").split(/\r?\n/)) {
    let m = /^(\S+)\s+(\d+) shard\(s\)\s+GPU (\S+)\s+(.*)$/.exec(l);
    if (m) { out.push({ name: m[1], shards: Number(m[2]), gpu: m[3], from: m[4].trim() }); continue; }
    m = /^(\S+)\s+unreadable: (.*)$/.exec(l);
    if (m) out.push({ name: m[1], error: m[2] });
  }
  return out;
}

const FIELDS = { unit: "unit", from: "from", runs: "runs", environment: "environment", compute: "compute", profile: "profile", parallel: "parallel", models: "models", secrets: "secrets", results: "results" };

/**
 * A plan / wrap summary: `key<12 value` lines, notes and WARNING items as
 * continuation lines with an empty key, then "On AppHub" steps.
 */
export function parseSummary(stdout) {
  const text = String(stdout || "").replace(/\r\n/g, "\n");
  const [head, ...tail] = text.split("\nOn AppHub");
  const s = { notes: [], warnings: [], steps: [], back: null };
  let list = null;
  for (const line of head.split("\n")) {
    if (!line.trim()) continue;
    const key = line.slice(0, 12).trim();
    const value = line.slice(13).trim();
    if (!key) { if (list) list.push(value.replace(/^- /, "")); continue; }
    if (key === "notes") { list = s.notes; list.push(value.replace(/^- /, "")); continue; }
    if (key === "WARNING") { list = s.warnings; list.push(value.replace(/^- /, "")); continue; }
    list = null;
    if (FIELDS[key]) s[FIELDS[key]] = value;
  }
  const u = /^(\S+) — (\d+) shard\(s\)(?: \((.*)\))?$/.exec(s.unit || "");
  if (u) { s.name = u[1]; s.shards = Number(u[2]); s.split = u[3] || null; }
  if (tail.length) {
    const rest = tail.join("\nOn AppHub").split("\n").slice(1);
    let step = null;
    for (const l of rest) {
      const back = /^Back here: (.*)$/.exec(l);
      if (back) { s.back = back[1].trim(); continue; }
      const m = /^\s+(\d+)\. ([^:]+):\s+(.*)$/.exec(l);
      if (m) { step = { n: Number(m[1]), what: m[2].trim(), command: m[3].trim(), more: [] }; s.steps.push(step); continue; }
      if (step && l.trim()) step.more.push(l.trim());
    }
  }
  return s;
}

const SPAN = /^\d+(s|m|d|h\d{2}m)…?$/;

/** `lattice results`: runs, the shard table and the totals. */
export function parseResults(stdout) {
  const lines = String(stdout || "").replace(/\r\n/g, "\n").split("\n");
  const r = { runs: [], shards: [], totals: null };
  const head = /^(\S+) — results at (\S+) \((.*) ago\)$/.exec(lines[0] || "");
  if (head) { r.unit = head[1]; r.commit = head[2]; r.age = head[3]; }
  let inTable = false;
  for (const l of lines.slice(1)) {
    const run = /^\s+part (\S+)\s+(\S+)\s+on (\S*): (\S+) shard\(s\), (\S+) at a time, (\S+) CPU\(s\), (\S+) GPU\(s\); updated (.*?) ago(\s+\[earlier version of the unit\])?$/.exec(l);
    if (run) {
      r.runs.push({ part: run[1], state: run[2], host: run[3], shards: run[4], parallel: run[5], cpus: run[6], gpus: run[7], updated: run[8], stale: !!run[9] });
      continue;
    }
    if (/^\s+shard\s+state\s+exit\s+took\s+host$/.test(l)) { inTable = true; continue; }
    const tot = /^\s+(\d+) done, (\d+) failed, (\d+) running, (\d+) not reported, of (\d+)$/.exec(l);
    if (tot) {
      inTable = false;
      r.totals = { done: +tot[1], failed: +tot[2], running: +tot[3], pending: +tot[4], of: +tot[5] };
      continue;
    }
    if (inTable && l.trim()) {
      const [id, state, ...rest] = l.trim().split(/\s+/);
      const shard = { id, state, exit: null, took: null, host: null };
      if (rest.length && /^-?\d+$/.test(rest[0])) shard.exit = Number(rest.shift());
      if (rest.length && SPAN.test(rest[0])) shard.took = rest.shift();
      if (rest.length) shard.host = rest.join(" ");
      r.shards.push(shard);
    }
  }
  r.complete = !!r.totals && r.totals.done === r.totals.of;
  return r;
}

/** What `lattice results` exit codes mean (main.rs). */
export function resultsState(code) {
  if (code === 0) return "complete";
  if (code === 2) return "nothing-pushed";
  if (code === 3) return "incomplete";
  return "failed";
}
