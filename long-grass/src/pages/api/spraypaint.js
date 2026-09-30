// API route for the CLI-bridged spraypaint integration.
//
// spraypaint is a Rust CLI (fullscreen-triangle/graffiti/spraypaint) that
// does full-text passage retrieval over a directory tree — BM25 scored within
// "scenes" (top-level directories), a water-filling allocator splitting a
// fixed result budget across scenes, and (0.2.0) a coverage verdict saying
// whether the corpus holds what the query names at all. Local filesystem
// only; it has no network client of any kind. Spec: graffiti/specifications.md.
//
// Contract:
//   POST /api/spraypaint   body: { action: "ask", query, root?, budget?, scenes?, dry_run? }
//   -> { output_delta: { kind: "spraypaint_result", ... } }
//   POST /api/spraypaint   body: { action: "index" | "identity" | "count" | "scenes" | "verify", root? }
//   -> { output_delta: { kind: "spraypaint_<action>_result", ... } }
//   -> { ok: false, error: string, stderr?: string } on failure
//
// `dry_run` previews an ask: the verdict and the passages, with no committed
// act. Commit only the ask whose answer is used — the count never goes down.
//
// Where the binary is: SPRAYPAINT_CLI, else ~/.cargo/bin, /usr/local/bin,
// /usr/bin. Which tree it searches: see resolveRoot in lib/server/spraypaint.js
// — a remote request may only search SPRAYPAINT_ROOT, never choose a root.

import path from "path";
import os from "os";
import { spawn } from "child_process";
import { existsSync } from "fs";
import { isLocalRequest } from "@/lib/server/rag";
import { askArgs, parseJsonLoose, readCount, readIdentity, readScenes, readVerify, resolveRoot } from "@/lib/server/spraypaint";

const MAX_STDOUT_BYTES = 4 * 1024 * 1024; // 4 MiB cap
const FALLBACK_ROOT = path.resolve(process.cwd(), ".."); // the buhera repo, when run from long-grass/

function resolveBinary() {
  if (process.env.SPRAYPAINT_CLI && existsSync(process.env.SPRAYPAINT_CLI)) {
    return process.env.SPRAYPAINT_CLI;
  }
  const suffix = process.platform === "win32" ? ".exe" : "";
  const candidates = [
    path.join(os.homedir(), ".cargo", "bin", `spraypaint${suffix}`),
    "/usr/local/bin/spraypaint",
    "/usr/bin/spraypaint",
  ];
  return candidates.find((c) => existsSync(c)) || null;
}

function runSpraypaint(binary, args) {
  return new Promise((resolve) => {
    const child = spawn(binary, args, { windowsHide: true });
    const stdoutChunks = [];
    const stderrChunks = [];
    let bytes = 0;
    let truncated = false;

    child.stdout.on("data", (chunk) => {
      bytes += chunk.length;
      if (bytes > MAX_STDOUT_BYTES) {
        truncated = true;
        child.kill("SIGTERM");
        return;
      }
      stdoutChunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => stderrChunks.push(chunk));
    child.on("error", (err) =>
      resolve({ code: -1, stdout: "", stderr: err.message, truncated })
    );
    child.on("close", (code) =>
      resolve({
        code: code ?? -1,
        stdout: Buffer.concat(stdoutChunks).toString("utf8"),
        stderr: Buffer.concat(stderrChunks).toString("utf8"),
        truncated,
      })
    );
  });
}

const OUTPUTS = {
  identity: (stdout) => ({ kind: "spraypaint_identity_result", ...readIdentity(stdout) }),
  count: (stdout) => ({ kind: "spraypaint_count_result", committed_count: readCount(stdout) }),
  scenes: (stdout) => ({ kind: "spraypaint_scenes_result", scenes: readScenes(stdout) }),
  verify: (stdout, code) => ({ kind: "spraypaint_verify_result", exit_code: code, ...readVerify(stdout, code) }),
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { action = "ask", query, root, budget, scenes, dry_run } = req.body ?? {};

  const binary = resolveBinary();
  if (!binary) {
    return res.status(503).json({
      ok: false,
      error:
        "spraypaint CLI not found. Install it with `cargo install --path spraypaint` " +
        "in fullscreen-triangle/graffiti, or set SPRAYPAINT_CLI to the binary path.",
    });
  }

  const where = resolveRoot({
    requested: root,
    local: isLocalRequest(req),
    envRoot: process.env.SPRAYPAINT_ROOT || null,
    fallback: FALLBACK_ROOT,
    action,
  });
  if (where.error) return res.status(where.status).json({ ok: false, error: where.error });
  const resolvedRoot = where.root;

  const fail = (status, error, extra = {}) => res.status(status).json({ ok: false, error, ...extra });

  if (action === "index") {
    const t0 = Date.now();
    const { code, stdout, stderr, truncated } = await runSpraypaint(binary, ["index", "--root", resolvedRoot, "--json"]);
    const elapsed_ms = Date.now() - t0;
    if (code !== 0) return fail(502, `spraypaint index exited with code ${code}`, { stderr: stderr.trim(), elapsed_ms });
    if (truncated) return fail(502, "spraypaint output exceeded size limit", { elapsed_ms });
    const parsed = parseJsonLoose(stdout) || { summary: stdout.trim() };
    return res.status(200).json({
      output_delta: { kind: "spraypaint_index_result", elapsed_ms, ...parsed, root: resolvedRoot },
      residue: 1,
    });
  }

  if (OUTPUTS[action]) {
    const t0 = Date.now();
    const { code, stdout, stderr, truncated } = await runSpraypaint(binary, [action, "--root", resolvedRoot, "--json"]);
    const elapsed_ms = Date.now() - t0;
    // verify's exit code is its verdict (1 a breach, 2 degenerate or N/A) and
    // still comes with a full report; for the others nonzero is a failure.
    if (code !== 0 && action !== "verify") {
      return fail(502, `spraypaint ${action} exited with code ${code}`, { stderr: stderr.trim(), elapsed_ms });
    }
    if (truncated) return fail(502, "spraypaint output exceeded size limit", { elapsed_ms });
    const output_delta = { ...OUTPUTS[action](stdout, code), root: resolvedRoot, elapsed_ms };
    const residue = output_delta.scenes?.length ?? output_delta.invariants?.length ?? 1;
    return res.status(200).json({ output_delta, residue });
  }

  if (action !== "ask") return fail(400, `unknown action "${action}"`);
  if (typeof query !== "string" || !query.trim()) return fail(400, "query is required");

  const t0 = Date.now();
  const { code, stdout, stderr, truncated } = await runSpraypaint(
    binary,
    askArgs({ query, root: resolvedRoot, budget, scenes, dryRun: !!dry_run })
  );
  const elapsed_ms = Date.now() - t0;
  if (code !== 0) return fail(502, `spraypaint exited with code ${code}`, { stderr: stderr.trim(), elapsed_ms });
  if (truncated) return fail(502, "spraypaint output exceeded size limit", { elapsed_ms });

  const parsed = parseJsonLoose(stdout);
  if (!parsed) return fail(502, "failed to parse spraypaint stdout as JSON", { stdout: stdout.slice(0, 400), elapsed_ms });

  return res.status(200).json({
    output_delta: { kind: "spraypaint_result", query, root: resolvedRoot, elapsed_ms, ...parsed },
    residue: Array.isArray(parsed.results) ? parsed.results.length : 0,
  });
}
