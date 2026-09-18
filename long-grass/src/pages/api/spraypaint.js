// API route for the CLI-bridged spraypaint integration.
//
// spraypaint is a Rust CLI (fullscreen-triangle/graffiti/spraypaint) that
// does full-text passage retrieval over a repo — BM25 scored within
// "scenes" (top-level directories), then a water-filling allocator splits
// a fixed result budget across scenes rather than taking a flat global
// top-k. Local filesystem only; it has no network client of any kind.
//
// Contract:
//   POST /api/spraypaint   body: { action: "ask", query, root?, budget?, scenes? }
//   -> { output_delta: { kind: "spraypaint_result", ... } }
//   POST /api/spraypaint   body: { action: "index", root? }
//   -> { output_delta: { kind: "spraypaint_index_result", ... } }
//   -> { ok: false, error: string, stderr?: string } on failure
//
// Binary path is resolved via the SPRAYPAINT_CLI env var, or falls back to
// the default cargo install location (~/.cargo/bin/spraypaint[.exe]) — the
// same place `purpose` lives, since both ship from the same workspace.
// `--root` defaults to the buhera repo root (one level up from long-grass/,
// where this Next.js app itself lives) so a tutorial's queries search the
// whole codebase, not just this app.

import path from "path";
import os from "os";
import { spawn } from "child_process";
import { existsSync } from "fs";

const MAX_STDOUT_BYTES = 4 * 1024 * 1024; // 4 MiB cap
const DEFAULT_ROOT = path.resolve(process.cwd(), "..");
const DEFAULT_BUDGET = 8;

function resolveBinary() {
  if (process.env.SPRAYPAINT_CLI && existsSync(process.env.SPRAYPAINT_CLI)) {
    return process.env.SPRAYPAINT_CLI;
  }
  const suffix = process.platform === "win32" ? ".exe" : "";
  const candidates = [
    path.join(os.homedir(), ".cargo", "bin", `spraypaint${suffix}`),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return null;
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

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const {
    action = "ask",
    query,
    root,
    budget,
    scenes,
  } = req.body ?? {};

  const binary = resolveBinary();
  if (!binary) {
    return res.status(503).json({
      ok: false,
      error:
        "spraypaint CLI not found. Install it with `cargo install --path spraypaint` " +
        "in fullscreen-triangle/graffiti, or set SPRAYPAINT_CLI to the binary path.",
    });
  }

  const resolvedRoot = typeof root === "string" && root.trim() ? root : DEFAULT_ROOT;

  if (action === "index") {
    const args = ["index", "--root", resolvedRoot, "--json"];
    const t0 = Date.now();
    const { code, stdout, stderr, truncated } = await runSpraypaint(binary, args);
    const elapsed_ms = Date.now() - t0;

    if (code !== 0) {
      return res.status(502).json({
        ok: false,
        error: `spraypaint index exited with code ${code}`,
        stderr: stderr.trim(),
        elapsed_ms,
      });
    }
    if (truncated) {
      return res.status(502).json({ ok: false, error: "spraypaint output exceeded size limit", elapsed_ms });
    }

    // Unlike `ask --json`, `index --json` prints a plain-text "Indexing
    // <root> ..." progress line to stdout before the JSON object — strip
    // everything before the first `{` rather than assuming stdout is pure
    // JSON (verified against the real binary, not assumed).
    let parsed;
    try {
      const jsonStart = stdout.indexOf("{");
      parsed = JSON.parse(jsonStart >= 0 ? stdout.slice(jsonStart) : stdout);
    } catch {
      parsed = { summary: stdout.trim() };
    }

    return res.status(200).json({
      output_delta: {
        kind: "spraypaint_index_result",
        root: resolvedRoot,
        elapsed_ms,
        ...parsed,
      },
      residue: 1,
    });
  }

  if (action === "identity" || action === "count" || action === "scenes" || action === "verify") {
    const args = [action, "--root", resolvedRoot];
    const t0 = Date.now();
    const { code, stdout, stderr, truncated } = await runSpraypaint(binary, args);
    const elapsed_ms = Date.now() - t0;

    // None of these four subcommands support --json (checked against the
    // real CLI: identity/count/scenes/verify all print plain text only) —
    // parse the specific line shapes each one actually produces instead of
    // pretending a --json flag exists.
    if (code !== 0 && action !== "verify") {
      // verify's own exit code is meaningful (nonzero on invariant breach)
      // and still carries a real stdout report worth returning, so only
      // the other three treat a nonzero exit as a hard failure.
      return res.status(502).json({
        ok: false,
        error: `spraypaint ${action} exited with code ${code}`,
        stderr: stderr.trim(),
        elapsed_ms,
      });
    }
    if (truncated) {
      return res.status(502).json({ ok: false, error: "spraypaint output exceeded size limit", elapsed_ms });
    }

    if (action === "identity") {
      const text = stdout.trim();
      const fp = /fingerprint:\s*(\S+)/.exec(text)?.[1] ?? null;
      const chi = /chi\):\s*([\d.eE+-]+)/.exec(text)?.[1] ?? null;
      const floorM = /floor:\s*([\d.eE+-]+)/.exec(text)?.[1] ?? null;
      const vertices = /vertices:\s*(\d+)/.exec(text)?.[1] ?? null;
      const edges = /edges:\s*(\d+)/.exec(text)?.[1] ?? null;
      return res.status(200).json({
        output_delta: {
          kind: "spraypaint_identity_result",
          root: resolvedRoot,
          fingerprint: fp,
          chi: chi != null ? Number(chi) : null,
          floor: floorM != null ? Number(floorM) : null,
          vertices: vertices != null ? Number(vertices) : null,
          edges: edges != null ? Number(edges) : null,
          raw: text,
          elapsed_ms,
        },
        residue: 1,
      });
    }

    if (action === "count") {
      const n = /committed acts:\s*(\d+)/.exec(stdout)?.[1] ?? null;
      return res.status(200).json({
        output_delta: {
          kind: "spraypaint_count_result",
          root: resolvedRoot,
          committed_count: n != null ? Number(n) : null,
          raw: stdout.trim(),
          elapsed_ms,
        },
        residue: 1,
      });
    }

    if (action === "scenes") {
      const rows = [];
      for (const line of stdout.split("\n")) {
        const m = /^(\S.*?)\s+(\d+) doc\(s\), (\d+) passage\(s\)/.exec(line);
        if (m) rows.push({ scene: m[1].trim(), documents: Number(m[2]), passages: Number(m[3]) });
      }
      return res.status(200).json({
        output_delta: {
          kind: "spraypaint_scenes_result",
          root: resolvedRoot,
          scenes: rows,
          elapsed_ms,
        },
        residue: rows.length,
      });
    }

    // action === "verify"
    const lines = stdout.trim().split("\n").filter(Boolean);
    const invariants = [];
    for (const line of lines) {
      const m = /^(Inv \d+ \S.*?)\s+\[(PASS|FAIL)\]\s*(.*)$/.exec(line);
      if (m) invariants.push({ name: m[1].trim(), status: m[2], detail: m[3].trim() });
    }
    const overall = /overall:\s*(PASS|FAIL)/.exec(stdout)?.[1] ?? (code === 0 ? "PASS" : "FAIL");
    return res.status(200).json({
      output_delta: {
        kind: "spraypaint_verify_result",
        root: resolvedRoot,
        overall,
        invariants,
        exit_code: code,
        raw: stdout.trim(),
        elapsed_ms,
      },
      residue: invariants.length,
    });
  }

  // action === "ask"
  if (typeof query !== "string" || !query.trim()) {
    return res.status(400).json({ ok: false, error: "query is required" });
  }

  const args = ["ask", query, "--root", resolvedRoot, "--json"];
  const k = Number.isFinite(budget) ? Math.max(1, Math.floor(budget)) : DEFAULT_BUDGET;
  args.push("-k", String(k));
  if (Array.isArray(scenes) && scenes.length > 0) {
    args.push("--scenes", scenes.join(","));
  }

  const t0 = Date.now();
  const { code, stdout, stderr, truncated } = await runSpraypaint(binary, args);
  const elapsed_ms = Date.now() - t0;

  if (code !== 0) {
    return res.status(502).json({
      ok: false,
      error: `spraypaint exited with code ${code}`,
      stderr: stderr.trim(),
      elapsed_ms,
    });
  }
  if (truncated) {
    return res.status(502).json({ ok: false, error: "spraypaint output exceeded size limit", elapsed_ms });
  }

  let parsed;
  try {
    parsed = JSON.parse(stdout.trim());
  } catch {
    return res.status(502).json({
      ok: false,
      error: "failed to parse spraypaint stdout as JSON",
      stdout: stdout.slice(0, 400),
      elapsed_ms,
    });
  }

  return res.status(200).json({
    output_delta: {
      kind: "spraypaint_result",
      query,
      root: resolvedRoot,
      elapsed_ms,
      ...parsed,
    },
    residue: Array.isArray(parsed.results) ? parsed.results.length : 0,
  });
}
