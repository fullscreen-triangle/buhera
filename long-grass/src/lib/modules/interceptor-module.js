/* ============================================================================
 * Interceptor Module Adapter
 *
 * The AI assistant module: generates runnable Rust or TypeScript code from a
 * natural-language description, executes it in a sandboxed subprocess, feeds
 * the captured console output into vaHera memory (so it becomes searchable
 * and recallable exactly like any other vaHera memory), and — on request —
 * runs it through the wind-tunnel stability pass (does the program behave
 * the same way every time it runs?).
 *
 * Three server routes back this module (all server-only, following the
 * spraypaint/dsl-generate CLI-spawn convention):
 *   /api/interceptor-generate    — NL task -> code (llm-cascade.js)
 *   /api/interceptor-run         — code -> { stdout, stderr, exit_code, ... }
 *   /api/interceptor-windtunnel  — code -> N runs -> stability report
 *
 * Instruction shapes:
 *   • a plain string                        — sugar for { mode: "assist", language: "typescript", task: <string> }
 *   • { mode: "generate", language, task }
 *   • { mode: "run", language, code }
 *   • { mode: "test", language, code, runs? }
 *   • { mode: "assist", language, task, runs? }   — generate, run, vaHera-store, wind-tunnel-test, in one act
 * ========================================================================== */

import { dispatch as dispatchModule } from "@/lib/modules/registry";

const GENERATE_ROUTE = "/api/interceptor-generate";
const RUN_ROUTE = "/api/interceptor-run";
const WINDTUNNEL_ROUTE = "/api/interceptor-windtunnel";

let _runCounter = 0;

function normaliseInstruction(instruction) {
  if (instruction == null || instruction === "") {
    return { mode: "assist", language: "typescript", task: "" };
  }
  if (typeof instruction === "string") {
    return { mode: "assist", language: "typescript", task: instruction };
  }
  if (typeof instruction === "object") {
    return {
      mode: String(instruction.mode || "assist"),
      language: String(instruction.language || "typescript").toLowerCase(),
      task: typeof instruction.task === "string" ? instruction.task : "",
      code: typeof instruction.code === "string" ? instruction.code : "",
      runs: Number.isInteger(instruction.runs) ? instruction.runs : undefined,
      timeoutMs: Number.isInteger(instruction.timeoutMs) ? instruction.timeoutMs : undefined,
    };
  }
  return { mode: "assist", language: "typescript", task: String(instruction) };
}

async function postJson(route, body) {
  let res;
  try {
    res = await fetch(route, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { ok: false, error: `network error contacting ${route}: ${err.message || String(err)}` };
  }
  let json;
  try {
    json = await res.json();
  } catch {
    return { ok: false, error: `non-JSON response from ${route} (HTTP ${res.status})` };
  }
  if (!res.ok && json.ok === undefined) {
    return { ok: false, error: json.error || `HTTP ${res.status}` };
  }
  return json;
}

/**
 * Store a run's captured console output as a vaHera memory, so it becomes
 * recallable via `memory find nearest "..."` like any other memory. Escapes
 * embedded quotes the same way runInput's bare-search path does.
 */
async function storeInVahera(label, text) {
  const safe = String(text || "").replace(/"/g, "'").slice(0, 4000);
  const name = `interceptor-run-${++_runCounter}-${label}`;
  const script = `memory store "${name}" = "${safe || "(no output)"}"`;
  try {
    const act = await dispatchModule("vahera", script);
    return { name, act };
  } catch (err) {
    return { name, error: err.message || String(err) };
  }
}

async function doGenerate({ language, task }) {
  if (!task.trim()) {
    return { ok: false, error: "interceptor: task description is required for generate/assist." };
  }
  return postJson(GENERATE_ROUTE, { language, task });
}

async function doRun({ language, code, timeoutMs }) {
  if (!code.trim()) {
    return { ok: false, error: "interceptor: code is required for run mode." };
  }
  return postJson(RUN_ROUTE, { language, code, timeout_ms: timeoutMs });
}

async function doTest({ language, code, runs, timeoutMs }) {
  if (!code.trim()) {
    return { ok: false, error: "interceptor: code is required for test mode." };
  }
  return postJson(WINDTUNNEL_ROUTE, { language, code, runs, timeout_ms: timeoutMs });
}

export const interceptorModule = {
  id: "interceptor",

  describe() {
    return {
      id: "interceptor",
      description:
        "Interceptor: AI code-generation assistant. Generates runnable Rust " +
        "or TypeScript from a natural-language description, executes it in a " +
        "sandboxed subprocess, stores the captured console output as vaHera " +
        "memory, and can wind-tunnel-test it for run-to-run stability.",
      instructions: [
        'dispatch("interceptor", "print the first 10 fibonacci numbers")',
        'dispatch("interceptor", { mode: "generate", language: "rust", task: "..." })',
        'dispatch("interceptor", { mode: "run", language: "typescript", code: "console.log(1+1)" })',
        'dispatch("interceptor", { mode: "test", language: "typescript", code: "...", runs: 5 })',
        'dispatch("interceptor", { mode: "assist", language: "rust", task: "...", runs: 3 })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const norm = normaliseInstruction(instruction);

    if (norm.mode === "generate") {
      const gen = await doGenerate(norm);
      if (!gen.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`interceptor: generation failed — ${gen.error}`] },
          residue: 0,
          completed: true,
          error: gen.error,
        };
      }
      return {
        ok: true,
        output_delta: { kind: "interceptor_generated", language: gen.language, code: gen.code, provider: gen.provider, model: gen.model },
        residue: 0,
        completed: true,
      };
    }

    if (norm.mode === "run") {
      const run = await doRun(norm);
      if (!run.output_delta) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`interceptor: run failed — ${run.error}`] },
          residue: 0,
          completed: true,
          error: run.error,
        };
      }
      const combined = `${run.output_delta.stdout || ""}${run.output_delta.stderr || ""}`;
      const stored = await storeInVahera("run", combined);
      return {
        ok: run.output_delta.ok,
        output_delta: { ...run.output_delta, vahera_memory: stored.name },
        residue: run.residue ?? (run.output_delta.ok ? 0 : 1),
        completed: true,
      };
    }

    if (norm.mode === "test") {
      const test = await doTest(norm);
      if (!test.output_delta) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`interceptor: wind-tunnel test failed — ${test.error}`] },
          residue: 0,
          completed: true,
          error: test.error,
        };
      }
      return {
        ok: true,
        output_delta: test.output_delta,
        residue: test.residue ?? 0,
        completed: true,
      };
    }

    // mode === "assist" (default): generate -> run -> vaHera-store -> wind-tunnel-test.
    const gen = await doGenerate(norm);
    if (!gen.ok) {
      return {
        ok: false,
        output_delta: { kind: "text", lines: [`interceptor: generation failed — ${gen.error}`] },
        residue: 0,
        completed: true,
        error: gen.error,
      };
    }

    const run = await doRun({ language: gen.language, code: gen.code, timeoutMs: norm.timeoutMs });
    if (!run.output_delta) {
      return {
        ok: false,
        output_delta: {
          kind: "interceptor_generated",
          language: gen.language,
          code: gen.code,
          provider: gen.provider,
          model: gen.model,
          run_error: run.error,
        },
        residue: 1,
        completed: true,
        error: run.error,
      };
    }

    const combined = `${run.output_delta.stdout || ""}${run.output_delta.stderr || ""}`;
    const stored = await storeInVahera("assist", combined);

    let windtunnel = null;
    if (run.output_delta.ok) {
      const test = await doTest({ language: gen.language, code: gen.code, runs: norm.runs, timeoutMs: norm.timeoutMs });
      windtunnel = test.output_delta || null;
    }

    return {
      ok: run.output_delta.ok,
      output_delta: {
        kind: "interceptor_assist_result",
        language: gen.language,
        task: norm.task,
        code: gen.code,
        provider: gen.provider,
        model: gen.model,
        run: run.output_delta,
        vahera_memory: stored.name,
        windtunnel,
      },
      residue: windtunnel ? Math.round((1 - windtunnel.order_parameter) * 100) / 100 : (run.output_delta.ok ? 0 : 1),
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "interceptor_cell" };
  },
};
