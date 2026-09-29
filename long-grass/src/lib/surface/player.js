/* ============================================================================
 * The player — the default resolver on the blank surface.
 *
 * The user writes whatever they want, in whatever syntax. The player decides
 * nothing about what they meant; it only routes:
 *
 *   • text already in one of the federation's DSLs (vaHera, turbulance, a
 *     dispatch(...) call, a SCOPE cell, …) runs as written — a person's own
 *     script. Nobody has to switch DSLs on their screen.
 *   • anything else goes to /api/player, where retrieval over the user's own
 *     sources and the user's personal model write the vaHera script (validated
 *     by vaHera's parser, repaired until it parses).
 *
 * Every vaHera script — written or generated — then runs in the kernel AND
 * seeds the runtime graph (the CKG module): each `describe X with "…"` is a
 * node τ = X; each `spawn P from X` whose P is a registered module attaches a
 * chunk that runs P on X's description. The nodes' chunks run; their outputs
 * land on the nodes as values. A run is a line through the nodes it touched;
 * the same τ in a later run is the same node (it converges).
 *
 * The script is kept as a plan; the completed run is kept as a report.
 * ========================================================================== */

import { routeInput } from "@/lib/runtime/route-input";
import { runInput } from "@/lib/runtime/run-input";
import { executeVahera, parseVahera } from "@/lib/vahera";
import { listModules, getModule, dispatch as dispatchModule } from "@/lib/modules/registry";
import { EDGES } from "@/lib/surface/edges";
import { getSettings, addPlan, addReport } from "@/lib/surface/settings";

// Modules that configure or observe the surface, or are the machinery the
// player itself drives — never offered to the model as programs.
const NOT_PROGRAMS = new Set([
  ...Object.values(EDGES).flatMap((e) => e.entries.map((x) => x.id)),
  "vahera", "echo", "ckg", "desk", "vis", "disk", "config", "restart", "update", "network",
]);

/** The registered modules a script may `spawn`, with their descriptions. */
export function programs() {
  return listModules()
    .filter((m) => !NOT_PROGRAMS.has(m.id))
    .map((m) => ({ id: m.id, description: m.description || "" }));
}

// ── runs: the lines of the runtime map ───────────────────────────────────

const _runs = []; // [{ id, at, words, by, taus, spawned }]

export function getRuns() {
  return _runs.slice();
}

async function ckg(instruction) {
  const res = await dispatchModule("ckg", instruction);
  return res?.output_delta ?? null;
}

/** The runtime map's data: every run's line and the graph they built. */
export async function runtimeSnapshot() {
  const graph = await ckg({ op: "graph" });
  return { kind: "runtime_map", runs: getRuns(), graph };
}

/**
 * Run a vaHera script in the kernel and seed the runtime graph from it.
 * Errors are values, never halts (the runtime judges nothing).
 */
export async function runScript(script, { runtime, words, by }) {
  let results = [];
  let error = null;
  try {
    const out = executeVahera(script, runtime.kernel, { useProteinDb: false, rerank: true });
    results = out.results.length ? out.results : out.lastResult ? [out.lastResult] : [];
  } catch (err) {
    error = err.message || String(err);
  }

  let stmts = [];
  try { stmts = parseVahera(script); } catch { stmts = []; } // a parse error was already reported by executeVahera

  const project = getSettings().project || "default";
  const described = new Map();
  for (const s of stmts) if (s.op === "describe") described.set(s.target, s.text);
  const runId = `run-${_runs.length + 1}`;
  const taus = [];
  const spawned = [];

  for (const [tau] of described) {
    await ckg({ op: "represent", tau, address: [project, tau] });
    taus.push(tau);
  }
  const touched = new Set();
  for (const s of stmts.filter((x) => x.op === "spawn")) {
    const isProgram = !!getModule(s.program) && !NOT_PROGRAMS.has(s.program);
    if (!described.has(s.target)) {
      spawned.push({ program: s.program, target: s.target, attached: false, note: "target was never described" });
      continue;
    }
    if (!isProgram) {
      spawned.push({ program: s.program, target: s.target, attached: false, note: "a kernel process only — no module of that name" });
      continue;
    }
    await ckg({ op: "attach", tau: s.target, name: `${s.program}@${runId}`, module: s.program, instruction: described.get(s.target) });
    spawned.push({ program: s.program, target: s.target, attached: true });
    touched.add(s.target);
  }
  for (const tau of touched) await ckg({ op: "dispatch", tau });

  const run = { id: runId, at: Date.now(), words, by, taus, spawned };
  _runs.push(run);
  const graph = await ckg({ op: "graph" });
  return { results, error, run, graph };
}

async function askPlayer(utterance) {
  const s = getSettings();
  const res = await fetch("/api/player", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      utterance,
      project: s.project,
      rag: s.rag,
      model: s.model,
      programs: programs(),
    }),
  });
  if (!res.ok) return { ok: false, stage: "network", error: `player: HTTP ${res.status}` };
  return res.json();
}

/**
 * The resolver the surface installs by default. Returns an envelope.
 * @param {string} utterance
 * @param {{ runtime }} ctx
 */
export async function play(utterance, { runtime }) {
  const route = routeInput(utterance);

  // Already a DSL other than vaHera: run it as written.
  if (route.type !== "nl" && route.type !== "vahera") {
    const env = await runInput(utterance, runtime);
    if (route.type !== "meta" && route.type !== "noop") addPlan({ source: utterance, script: utterance, by: "person", dsl: route.type });
    return env;
  }

  // vaHera written by the person, or generated from free text.
  let script = route.type === "vahera" ? utterance : null;
  let by = "person";
  let retrieval = null;
  let generation = null;
  if (!script) {
    const answer = await askPlayer(utterance).catch((e) => ({ ok: false, stage: "network", error: String(e) }));
    retrieval = answer.retrieval || null;
    generation = answer.generation || null;
    if (!answer.ok) {
      return {
        kind: "artifact",
        result: { kind: "player_run", words: utterance, ok: false, stage: answer.stage, error: answer.error, script: answer.script || null, retrieval, generation },
      };
    }
    script = answer.script;
    by = "model";
  }

  const { results, error, run, graph } = await runScript(script, { runtime, words: utterance, by });
  addPlan({ source: utterance, script, by, dsl: "vahera" });
  addReport({
    source: utterance,
    script,
    by,
    run: { id: run.id, taus: run.taus, spawned: run.spawned },
    retrieval,
    summary: `${run.taus.length} node(s), ${run.spawned.filter((x) => x.attached).length} module chunk(s)${error ? ` · error: ${error}` : ""}`,
  });
  return {
    kind: "artifact",
    result: { kind: "player_run", words: utterance, ok: true, by, script, results, error, run, graph, retrieval, generation },
  };
}
