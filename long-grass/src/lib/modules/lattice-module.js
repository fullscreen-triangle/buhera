/* ============================================================================
 * lattice — jobs on AppHub, the university's Code-Server (server side:
 * pages/api/lattice.js, which runs the lattice CLI from pylon/lattice).
 *
 * AppHub accepts no connection from outside, so a job travels by git:
 * lattice wraps a task of a repository into a unit (`plan` shows what it
 * needs, `wrap` writes, commits and pushes it to the university's Gitea);
 * you start an AppHub session and run the unit there; it pushes its results
 * back, and `results` reads them. lattice cannot start AppHub sessions —
 * nothing outside AppHub can.
 *
 * Instruction shapes:
 *   "show"                                          → your repositories and their units
 *   { kind: "tasks" | "units", repo }
 *   { kind: "plan", repo, task?, command?, name?, matrix?, each?, outputs?, gpu?, … }
 *   { kind: "wrap", …same, confirm: true, remote?, push? }   → commits and pushes
 *   { kind: "results", repo, unit, remote? }
 *   { kind: "log", repo, unit, shard, remote? }
 *   { kind: "get", repo, unit, confirm: true }                → outputs into the working tree
 * ========================================================================== */

async function post(body) {
  try {
    const res = await fetch("/api/lattice", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!json) return { ok: false, error: `HTTP ${res.status}` };
    return json;
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });

export const latticeModule = {
  id: "lattice",

  describe() {
    return {
      id: "lattice",
      description:
        "Jobs on AppHub: wrap a repository's task into a unit (it works out the environment, GPU, models, " +
        "secrets and shards), push it to the university's Gitea, run it in an AppHub session, read the results back here.",
      instructions: [
        'dispatch("lattice", { kind: "tasks", repo: "C:/path/to/repo" })',
        'dispatch("lattice", { kind: "plan", repo: "C:/path/to/repo", task: "train", matrix: ["seed=1..5"] })',
        'dispatch("lattice", { kind: "results", repo: "C:/path/to/repo", unit: "train" })',
      ],
    };
  },

  async execute(instruction) {
    const inst = typeof instruction === "string" ? { kind: instruction.trim() || "show" } : instruction || {};
    const kind = inst.kind || "show";
    if (kind === "show") return done({ kind: "lattice_home" });
    const { kind: _k, ...rest } = inst;
    const r = await post({ action: kind, ...rest });
    if (r.ok === false && !r.kind) return done({ kind: "text", lines: [`lattice ${kind}: ${r.error}`] }, false);
    return done({ ...r, kind: r.kind || `lattice_${kind}` }, r.ok !== false);
  },

  outputCell() {
    return { kind: "lattice_cell" };
  },
};
