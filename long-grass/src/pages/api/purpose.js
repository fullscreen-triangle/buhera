// API route for the CLI-bridged Purpose integration.
//
// Contract (integration.md §5 + §9.4):
//   POST /api/purpose   body: { utterance: string, mode?: "value" | "fragment" }
//   -> { ok: true, value?: any, fragment?: any, elapsed_ms: number }
//   -> { ok: false, error: string, stderr?: string }
//
// Spawns the Rust `purpose` binary with `--raw` (value mode) or
// `--fragment` (compiled vaHera only). Binary path is resolved via the
// PURPOSE_CLI env var, or falls back to the debug target of the
// sibling workspace at ../mechanistic-synthesis/implementation.

import path from "path";
import { spawn } from "child_process";
import { existsSync } from "fs";

const MAX_STDOUT_BYTES = 2 * 1024 * 1024; // 2 MiB cap

function resolveBinary() {
  if (process.env.PURPOSE_CLI && existsSync(process.env.PURPOSE_CLI)) {
    return process.env.PURPOSE_CLI;
  }
  const suffix = process.platform === "win32" ? ".exe" : "";
  const candidates = [
    path.resolve(
      process.cwd(),
      "..",
      "mechanistic-synthesis",
      "implementation",
      "target",
      "release",
      `purpose${suffix}`
    ),
    path.resolve(
      process.cwd(),
      "..",
      "mechanistic-synthesis",
      "implementation",
      "target",
      "debug",
      `purpose${suffix}`
    ),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return null;
}

function runPurpose(binary, args) {
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

  const { utterance, mode = "value" } = req.body ?? {};
  if (typeof utterance !== "string" || !utterance.trim()) {
    return res.status(400).json({ ok: false, error: "utterance required" });
  }

  const binary = resolveBinary();
  if (!binary) {
    return res.status(503).json({
      ok: false,
      error:
        "purpose CLI not found. Build it with `cargo build --release` in mechanistic-synthesis/implementation, or set PURPOSE_CLI to the binary path.",
    });
  }

  const args = ["query", utterance];
  if (mode === "fragment") args.push("--fragment");
  else args.push("--raw");

  const t0 = Date.now();
  const { code, stdout, stderr, truncated } = await runPurpose(binary, args);
  const elapsed_ms = Date.now() - t0;

  if (code !== 0) {
    return res.status(502).json({
      ok: false,
      error: `purpose exited with code ${code}`,
      stderr: stderr.trim(),
      elapsed_ms,
    });
  }
  if (truncated) {
    return res.status(502).json({
      ok: false,
      error: "purpose output exceeded size limit",
      elapsed_ms,
    });
  }

  let parsed;
  try {
    parsed = JSON.parse(stdout.trim());
  } catch (e) {
    return res.status(502).json({
      ok: false,
      error: "failed to parse purpose stdout as JSON",
      stdout: stdout.slice(0, 400),
      elapsed_ms,
    });
  }

  if (mode === "fragment") {
    return res.status(200).json({ ok: true, fragment: parsed, elapsed_ms });
  }
  return res.status(200).json({ ok: true, value: parsed, elapsed_ms });
}
