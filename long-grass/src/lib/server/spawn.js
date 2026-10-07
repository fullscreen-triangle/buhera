/* ============================================================================
 * Running a local CLI from an API route: spraypaint, lattice.
 *
 * findBinary looks in the env var first, then where cargo and system
 * installs put binaries. run() never throws: a missing binary, a crash or a
 * timeout comes back as { code: -1, stderr } so the route can say what
 * happened. Output past `maxBytes` kills the child and sets `truncated`.
 * ========================================================================== */

import path from "path";
import os from "os";
import { spawn } from "child_process";
import { existsSync } from "fs";

export function findBinary(name, envVar) {
  const fromEnv = envVar && process.env[envVar];
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const exe = process.platform === "win32" ? ".exe" : "";
  const candidates = [
    path.join(os.homedir(), ".cargo", "bin", `${name}${exe}`),
    `/usr/local/bin/${name}`,
    `/usr/bin/${name}`,
  ];
  return candidates.find((c) => existsSync(c)) || null;
}

export function run(binary, args, { cwd, env, maxBytes = 4 * 1024 * 1024, timeoutMs = 120_000 } = {}) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    let child;
    try {
      child = spawn(binary, args, { cwd, env: env ? { ...process.env, ...env } : process.env, windowsHide: true });
    } catch (err) {
      resolve({ code: -1, stdout: "", stderr: err.message, truncated: false, timedOut: false, elapsed_ms: 0 });
      return;
    }
    const out = [];
    const errs = [];
    let bytes = 0;
    let truncated = false;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      bytes += chunk.length;
      if (bytes > maxBytes) { truncated = true; child.kill("SIGTERM"); return; }
      out.push(chunk);
    });
    child.stderr.on("data", (chunk) => errs.push(chunk));
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ code: -1, stdout: "", stderr: err.message, truncated, timedOut, elapsed_ms: Date.now() - t0 });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({
        code: code ?? -1,
        stdout: Buffer.concat(out).toString("utf8"),
        stderr: Buffer.concat(errs).toString("utf8"),
        truncated,
        timedOut,
        elapsed_ms: Date.now() - t0,
      });
    });
  });
}
