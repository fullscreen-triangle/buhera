// API route for lattice: jobs on AppHub, the university's Code-Server.
//
//   POST /api/lattice { action: "tasks", repo }                       → what can be wrapped
//   POST /api/lattice { action: "plan", repo, ...spec }               → what a task needs on AppHub; writes nothing
//   POST /api/lattice { action: "wrap", repo, ...spec, confirm: true, remote?, push? }
//                                                                     → write the unit, commit it, push it to Gitea
//   POST /api/lattice { action: "units", repo }                       → the units in a repository
//   POST /api/lattice { action: "results", repo, unit, remote? }      → what AppHub has pushed back
//   POST /api/lattice { action: "log", repo, unit, shard, remote? }   → one shard's output
//   POST /api/lattice { action: "get", repo, unit, confirm: true }    → copy the outputs into the working tree
//
// spec: { task?, command?: string[], name?, matrix?: ["seed=1..5"], each?, outputs?, env?, secrets?, gpu?, parallel?, timeout? }
//
// lattice works on repositories on this machine and pushes to the
// university's Gitea, so the route answers local requests only; `wrap` and
// `get` change a repository and must say `confirm: true`.

import fs from "fs";
import path from "path";
import { isLocalRequest } from "@/lib/server/rag";
import { findBinary, run } from "@/lib/server/spawn";
import { parseResults, parseSummary, parseTasks, parseUnits, readNotes, resultsState, specArgs } from "@/lib/server/lattice";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "method not allowed" });
  if (!isLocalRequest(req)) {
    return res.status(403).json({ ok: false, error: "lattice works on repositories on the machine this server runs on — run long-grass locally" });
  }
  const bin = findBinary("lattice", "LATTICE_CLI");
  if (!bin) {
    return res.status(503).json({ ok: false, error: "lattice is not installed: `cargo install --path lattice` in pylon, or set LATTICE_CLI" });
  }

  const { action, repo, unit, shard, remote, confirm, push = true, ...spec } = req.body ?? {};
  if (typeof repo !== "string" || !path.isAbsolute(repo) || !fs.existsSync(repo)) {
    return res.status(400).json({ ok: false, error: "repo must be the absolute path of a repository on this machine" });
  }
  const lattice = (args, timeoutMs = 120_000) => run(bin, ["-C", repo, ...args], { timeoutMs });
  const answer = (r, body) => {
    const { notes, error } = readNotes(r.stderr);
    const ok = body.ok ?? r.code === 0;
    return res.status(ok ? 200 : 422).json({ ok, repo, notes, error: ok ? null : error || r.stderr.trim() || `lattice exited with ${r.code}`, raw: r.stdout, elapsed_ms: r.elapsed_ms, ...body });
  };

  switch (action) {
    case "tasks": {
      const r = await lattice(["tasks"]);
      return answer(r, { kind: "lattice_tasks", tasks: parseTasks(r.stdout) });
    }
    case "units": {
      const r = await lattice(["units"]);
      return answer(r, { kind: "lattice_units", units: parseUnits(r.stdout) });
    }
    case "plan": {
      const r = await lattice(["plan", ...specArgs(spec)], 300_000);
      return answer(r, { kind: "lattice_plan", spec, summary: parseSummary(r.stdout) });
    }
    case "wrap": {
      if (confirm !== true) return res.status(400).json({ ok: false, error: "wrap commits and pushes: send confirm: true" });
      const extra = [...(remote ? ["--remote", String(remote)] : []), ...(push ? [] : ["--no-push"]), ...(spec.force ? ["--force"] : [])];
      const r = await lattice(["wrap", ...extra, ...specArgs(spec)], 600_000);
      return answer(r, { kind: "lattice_wrapped", spec, pushed: r.code === 0 && push, summary: parseSummary(r.stdout) });
    }
    case "results":
    case "get": {
      if (action === "get" && confirm !== true) return res.status(400).json({ ok: false, error: "get writes into the working tree: send confirm: true" });
      const args = ["results", ...(unit ? [String(unit)] : []), ...(remote ? ["--remote", String(remote)] : []), ...(action === "get" ? ["--get"] : [])];
      const r = await lattice(args, 300_000);
      const state = resultsState(r.code);
      // 2 (nothing pushed yet) and 3 (still running) are answers, not failures.
      return answer(r, { ok: state !== "failed", kind: "lattice_results", unit: unit || null, state, results: parseResults(r.stdout) });
    }
    case "log": {
      if (!shard) return res.status(400).json({ ok: false, error: "shard is required" });
      const r = await lattice(["results", ...(unit ? [String(unit)] : []), "--log", String(shard), ...(remote ? ["--remote", String(remote)] : [])]);
      return answer(r, { kind: "lattice_log", unit: unit || null, shard, log: r.stdout.slice(-200_000) });
    }
    default:
      return res.status(400).json({ ok: false, error: `unknown action "${action}"` });
  }
}
