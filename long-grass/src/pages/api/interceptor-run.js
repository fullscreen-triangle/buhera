// API route: interceptor code execution.
//
// The "run" half of the interceptor module — executes a Rust or TypeScript
// snippet in a sandboxed subprocess and captures its console output. This is
// the piece vaHera monitors: every result this route returns is what the
// interceptor module goes on to store as vaHera memory.
//
// Contract:
//   POST /api/interceptor-run
//     body: { language: "typescript"|"rust", code: string, timeout_ms?: number }
//   -> { output_delta: { kind: "interceptor_run_result", ... } }
//   -> { ok: false, error: string } on a bad request (not a failed run —
//      a failed run is still a 200 with ok:false inside output_delta, since
//      "the program crashed" is a normal outcome to report, not an API error)

import { runCode } from "@/lib/server/exec-sandbox";

const MAX_CODE_BYTES = 64 * 1024;
const SUPPORTED_LANGUAGES = new Set(["typescript", "rust"]);

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { language, code, timeout_ms: timeoutMs } = req.body || {};
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

  const result = await runCode(lang, code, { timeoutMs });

  return res.status(200).json({
    output_delta: {
      kind: "interceptor_run_result",
      language: lang,
      code,
      ok: result.ok,
      stdout: result.stdout,
      stderr: result.stderr,
      exit_code: result.exit_code,
      elapsed_ms: result.elapsed_ms,
      timed_out: result.timed_out,
      truncated: result.truncated,
      stage: result.stage,
    },
    residue: result.ok ? 0 : 1,
  });
}
