// API route: wind-tunnel stability testing for interceptor-generated code.
//
// Runs the given code N times through the same sandboxed executor
// interceptor-run.js uses, then applies src/lib/wind-tunnel's holonomy /
// order-parameter analysis (a TS port of wind-tunnel's cycle-level emergent
// testing) to report whether the program behaves consistently.
//
// Contract:
//   POST /api/interceptor-windtunnel
//     body: { language: "typescript"|"rust", code: string, runs?: number, timeout_ms?: number }
//   -> { output_delta: { kind: "windtunnel_report", ... } }

import { runCode } from "@/lib/server/exec-sandbox";
import { computeStability } from "@/lib/wind-tunnel";

const MAX_CODE_BYTES = 64 * 1024;
const SUPPORTED_LANGUAGES = new Set(["typescript", "rust"]);
const MAX_RUNS = 7;
const DEFAULT_RUNS = 3;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { language, code, runs, timeout_ms: timeoutMs } = req.body || {};
  const lang = String(language || "typescript").toLowerCase();

  if (!SUPPORTED_LANGUAGES.has(lang)) {
    return res.status(400).json({
      ok: false,
      error: `language must be "typescript" or "rust"; got "${language}"`,
    });
  }
  if (typeof code !== "string" || !code.trim()) {
    return res.status(400).json({ ok: false, error: "code (non-empty string) is required" });
  }
  if (code.length > MAX_CODE_BYTES) {
    return res.status(413).json({ ok: false, error: "code exceeds 64 KiB" });
  }

  const n = Number.isInteger(runs) && runs > 0 ? Math.min(runs, MAX_RUNS) : DEFAULT_RUNS;

  const results = [];
  const t0 = Date.now();
  // Sequential, not parallel: a Rust run compiles fresh each time (no shared
  // build cache across processes) and concurrent rustc invocations would
  // compete for the same temp-file namespace pattern; sequential keeps the
  // sandbox's resource ceiling predictable regardless of language.
  for (let i = 0; i < n; i++) {
    // eslint-disable-next-line no-await-in-loop
    const r = await runCode(lang, code, { timeoutMs });
    results.push(r);
  }
  const elapsed_ms = Date.now() - t0;

  const stability = computeStability(results);

  return res.status(200).json({
    output_delta: {
      kind: "windtunnel_report",
      language: lang,
      code,
      elapsed_ms,
      ...stability,
    },
    // residue: 1 - order_parameter, matching the registry convention that
    // lower residue means "closer to done" (a fully stable program is done).
    residue: Math.round((1 - stability.order_parameter) * 100) / 100,
  });
}
