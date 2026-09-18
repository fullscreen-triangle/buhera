// Shared sandboxed-execution helper (server-only).
//
// Generalizes the spawn/stdout-cap/kill-on-overflow pattern already used by
// api/spraypaint.js's runSpraypaint(), so interceptor-run.js and
// interceptor-windtunnel.js don't duplicate it. Two execution paths:
//
//   runTypeScript(code, { timeoutMs })  — node --experimental-strip-types on a
//                                          temp file (Node 24's native TS type
//                                          stripping; no tsx/ts-node dependency)
//   runRust(code, { timeoutMs })        — spawns the interceptor-run Rust CLI
//                                          (crates/interceptor-run), which
//                                          itself compiles + runs + times out
//                                          and prints one JSON object
//
// Both resolve to the same envelope:
//   { ok, stdout, stderr, exit_code, elapsed_ms, timed_out, truncated, stage }

import path from "path";
import os from "os";
import fs from "fs/promises";
import { existsSync } from "fs";
import { spawn } from "child_process";

const MAX_STDOUT_BYTES = 2 * 1024 * 1024; // 2 MiB cap
const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_TIMEOUT_MS = 30_000;

function clampTimeout(timeoutMs) {
  const n = Number(timeoutMs);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.min(n, MAX_TIMEOUT_MS);
}

/**
 * Spawn a binary/argv combo, capturing stdout/stderr with a byte cap and a
 * hard kill on timeout. Never rejects — always resolves an envelope.
 */
function spawnCapped(command, args, { timeoutMs, env }) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    let child;
    try {
      child = spawn(command, args, { windowsHide: true, env: env || process.env });
    } catch (err) {
      resolve({
        ok: false,
        stdout: "",
        stderr: `exec-sandbox: failed to spawn ${command}: ${err.message || String(err)}`,
        exit_code: null,
        elapsed_ms: Date.now() - t0,
        timed_out: false,
        truncated: false,
        stage: "spawn",
      });
      return;
    }

    const stdoutChunks = [];
    const stderrChunks = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let truncated = false;
    let timedOut = false;
    let settled = false;

    const timer = setTimeout(() => {
      timedOut = true;
      try { child.kill("SIGTERM"); } catch { /* noop */ }
    }, clampTimeout(timeoutMs));

    child.stdout.on("data", (chunk) => {
      stdoutBytes += chunk.length;
      if (stdoutBytes > MAX_STDOUT_BYTES) {
        truncated = true;
        try { child.kill("SIGTERM"); } catch { /* noop */ }
        return;
      }
      stdoutChunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderrBytes += chunk.length;
      if (stderrBytes > MAX_STDOUT_BYTES) {
        truncated = true;
        try { child.kill("SIGTERM"); } catch { /* noop */ }
        return;
      }
      stderrChunks.push(chunk);
    });

    const finish = (exitCode) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        ok: !timedOut && exitCode === 0,
        stdout: Buffer.concat(stdoutChunks).toString("utf8"),
        stderr: Buffer.concat(stderrChunks).toString("utf8"),
        exit_code: exitCode,
        elapsed_ms: Date.now() - t0,
        timed_out: timedOut,
        truncated,
        stage: "run",
      });
    };

    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        ok: false,
        stdout: Buffer.concat(stdoutChunks).toString("utf8"),
        stderr: err.message || String(err),
        exit_code: null,
        elapsed_ms: Date.now() - t0,
        timed_out: timedOut,
        truncated,
        stage: "run",
      });
    });
    child.on("close", (code) => finish(code));
  });
}

async function withTempFile(prefix, extension, contents, fn) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
  const file = path.join(dir, `snippet${extension}`);
  await fs.writeFile(file, contents, "utf8");
  try {
    return await fn(file);
  } finally {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

/**
 * Run TypeScript (or plain JS) source via Node's native type-stripping.
 * Works for both .ts-shaped and .js-shaped code — `--experimental-strip-types`
 * is a no-op syntactically for code with no type annotations.
 */
export async function runTypeScript(code, { timeoutMs } = {}) {
  return withTempFile("interceptor-ts-", ".ts", code, (file) =>
    spawnCapped(
      process.execPath,
      ["--experimental-strip-types", "--disable-warning=ExperimentalWarning", file],
      // NO_COLOR/FORCE_COLOR=0: console.log() colorizes inspected values
      // (numbers, etc.) by default even when stdout is piped, which pollutes
      // captured output with ANSI escapes that have nothing to do with the
      // program's actual behavior. Disable it at the source.
      { timeoutMs, env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" } }
    )
  );
}

function resolveRustBinary() {
  if (process.env.INTERCEPTOR_RUN_CLI && existsSync(process.env.INTERCEPTOR_RUN_CLI)) {
    return process.env.INTERCEPTOR_RUN_CLI;
  }
  const suffix = process.platform === "win32" ? ".exe" : "";
  const repoRoot = path.resolve(process.cwd(), "..");
  const candidates = [
    path.join(repoRoot, "target", "debug", `interceptor-run${suffix}`),
    path.join(repoRoot, "target", "release", `interceptor-run${suffix}`),
    path.join(os.homedir(), ".cargo", "bin", `interceptor-run${suffix}`),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return null;
}

/**
 * Run Rust source via the interceptor-run CLI (crates/interceptor-run),
 * which itself compiles, runs with a timeout, and prints one JSON result.
 */
export async function runRust(code, { timeoutMs } = {}) {
  const binary = resolveRustBinary();
  if (!binary) {
    return {
      ok: false,
      stdout: "",
      stderr:
        "interceptor-run binary not found. Build it with " +
        "`cargo build -p interceptor-run` from the buhera repo root, or set " +
        "INTERCEPTOR_RUN_CLI to the binary path.",
      exit_code: null,
      elapsed_ms: 0,
      timed_out: false,
      truncated: false,
      stage: "spawn",
    };
  }

  return withTempFile("interceptor-rs-", ".rs", code, async (file) => {
    const t0 = Date.now();
    const raw = await spawnCapped(
      binary,
      ["--code-file", file, "--timeout-ms", String(clampTimeout(timeoutMs))],
      { timeoutMs: clampTimeout(timeoutMs) + 15_000 } // allow rustc compile time on top
    );
    if (!raw.ok && raw.stage === "spawn") return raw; // binary missing/failed to launch

    // interceptor-run prints one JSON object on stdout regardless of outcome;
    // parse it and use it as the authoritative result (it has its own
    // ok/stage/timed_out fields distinguishing compile vs run failures).
    try {
      const parsed = JSON.parse(raw.stdout.trim());
      return {
        ok: !!parsed.ok,
        stdout: parsed.stdout || "",
        stderr: parsed.stderr || "",
        exit_code: parsed.exit_code ?? null,
        elapsed_ms: parsed.elapsed_ms ?? Date.now() - t0,
        timed_out: !!parsed.timed_out,
        truncated: !!parsed.truncated,
        stage: parsed.stage || "run",
      };
    } catch {
      return {
        ok: false,
        stdout: "",
        stderr: `interceptor-run: failed to parse CLI output as JSON: ${raw.stdout.slice(0, 400)}`,
        exit_code: raw.exit_code,
        elapsed_ms: Date.now() - t0,
        timed_out: raw.timed_out,
        truncated: raw.truncated,
        stage: "run",
      };
    }
  });
}

/**
 * Dispatch to the right runner by language name.
 */
export async function runCode(language, code, opts = {}) {
  const lang = String(language || "").toLowerCase();
  if (lang === "rust" || lang === "rs") return runRust(code, opts);
  if (lang === "typescript" || lang === "ts" || lang === "javascript" || lang === "js") {
    return runTypeScript(code, opts);
  }
  return {
    ok: false,
    stdout: "",
    stderr: `exec-sandbox: unsupported language "${language}" (expected "typescript" or "rust")`,
    exit_code: null,
    elapsed_ms: 0,
    timed_out: false,
    truncated: false,
    stage: "spawn",
  };
}
