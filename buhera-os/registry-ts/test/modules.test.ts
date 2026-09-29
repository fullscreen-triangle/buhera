// Per-module behaviour against the REAL engines (specifications/specs/*.md,
// "Conformance" sections). The wasm-hosted cases assert the same facts as
// buhera-modules/tests/conformance.rs: one implementation, two hosts.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { createFederation } from "../src/modules/index.ts";
import { realEngines } from "./engines.ts";

const fed = createFederation(await realEngines());
const run = (id: string, instruction: Parameters<typeof fed.registry.dispatch>[1], budget = 1) =>
  fed.registry.dispatch(id, instruction, budget);

// ── wasm-hosted Rust modules (parity with the Rust host) ───────────────────

test("ndombolo (wasm): demo scores the manuscript proposition at 0.9", async () => {
  const r = await run("ndombolo", "demo");
  assert.equal(r.ok, true);
  const m = (r.output_delta as any).propositions[0].motions[0];
  assert.equal(m.name, "Strong");
  assert.ok(Math.abs(m.score - 0.9) < 1e-12);
  assert.equal((r.output_delta as any).stopped_at, null);
});

test("ndombolo (wasm): turbulance validator reports script-absolute lines", () => {
  const v = fed.dsls.validate("turbulance", "item a = 1\n// ---\nitem = = 2");
  assert.equal(v.ok, false);
  assert.equal(v.errors[0]?.line, 3);
});

test("windtunnel (wasm): unmeasured assertions are skipped, never passed", async () => {
  const script = 'scope S:\n    include "src/**"\n\nassert:\n    regime >= Coherent\n    r_est >= 0.75\n    no holonomy_violations\n';
  const r = await run("windtunnel", { kind: "evaluate", script, metric: { regime: "Coherent", r_est: 0.81 } });
  const d = r.output_delta as any;
  assert.equal(d.verdict, "incomplete");
  assert.equal(d.passed, 2);
  assert.equal(d.skipped, 1);
  assert.ok(Math.abs(r.residue - 1 / 3) < 1e-12);
});

test("windtunnel (wasm): measure is unavailable without a filesystem-bearing host", async () => {
  const r = await run("windtunnel", { kind: "measure", script: "scope S:\n", project: "/x" });
  assert.equal(r.ok, false);
  assert.equal(r.error, "bridge unavailable");
});

test("tracker (wasm): χ of a small index; filesystem verbs refuse on this host", async () => {
  const sym = (name: string, file: string, snippet: string) => ({ name, kind: "fn", file, line: 1, snippet });
  const index = { root: "/r", symbols: [sym("alpha_parse", "src/a.rs", "calls beta_emit"), sym("beta_emit", "src/b.rs", "calls alpha_parse"), sym("gamma_note", "docs/c.md", "prose")] };
  const r = await run("tracker", { kind: "character", index });
  const d = r.output_delta as any;
  assert.equal(d.blocks, 3);
  assert.equal(d.fragments, 2);
  assert.ok(d.chi > 0);
  const fs = await run("tracker", { kind: "list", root: "/r" });
  assert.equal(fs.error, "unavailable on this host");
});

// ── TS-native modules ───────────────────────────────────────────────────────

test("sbs: glycolysis demo on the CPU path, residue = 1 − V", async () => {
  const empty = await run("sbs", { kind: "run", source: "", preferCPU: true });
  assert.equal(empty.ok, true, "a script with no circuit compiles and observes nothing");
  assert.equal(empty.residue, 0);
  const demo = await run("sbs", "demo");
  const d = demo.output_delta as any;
  assert.equal(d.kind, "sbs_result");
  assert.equal(d.circuit.numNodes, 10);
  assert.equal(d.backend, "cpu");
  assert.ok(Math.abs(demo.residue - (1 - d.metrics.V)) < 1e-12);
  assert.ok(d.metrics.V < 0.2, "perturbation reduces visibility");
});

test("sbs: adapter warns about the engine's silent edge-prop behaviour", async () => {
  const src = 'circuit c { node A { mu: -10, concentration: 1 } node B { mu: -20, concentration: 1 } edge A -> B { rate: 1 } }\nperturb c { edge: "A->B", factor: 0.2 }';
  const r = await run("sbs", src);
  assert.ok((r.output_delta as any).adapter_warnings.some((w: string) => w.includes("edge")));
});

test("sbs: validator is the engine's own checker", () => {
  assert.equal(fed.dsls.validate("sbs", "circuit c { node A { mu: 1 } }").ok, true);
  assert.equal(fed.dsls.validate("sbs", "circuit c { node A { mu: 1 }").ok, false);
});

test("hfq: Mark Doerr's mark_q1 runs; a static refusal is ok:true with refused_statically", async () => {
  const q1 = await run("hfq", { kind: "preset", id: "mark_q1" });
  assert.equal(q1.ok, true);
  assert.equal((q1.output_delta as any).result.world, "biocat");
  const ill = await run("hfq", { kind: "preset", id: "ill_capability" });
  assert.equal(ill.ok, true);
  assert.equal((ill.output_delta as any).refused_statically, true);
  assert.ok(ill.residue >= 1, "blocked steps are residue");
});

test("hfq: empty is an answer, not residue; budget_trap is refused/budget", async () => {
  const empty = await run("hfq", { kind: "preset", id: "empty_answer" });
  assert.equal((empty.output_delta as any).verdicts.empty >= 1, true);
  const trap = await run("hfq", { kind: "preset", id: "budget_trap" });
  assert.equal((trap.output_delta as any).blockers.budget >= 1, true);
});

test("hfq: validator catches unknown sources before any request", () => {
  const v = fed.dsls.validate("hfq", "plan p {\n  budget 5 requests\n  let x = from nowhere ask record(\"a\")\n  emit x\n}");
  assert.equal(v.ok, false);
  assert.match(v.errors[0]?.message ?? "", /nowhere/);
});

test("pylon: submit allocates an agent; stepping drives residual down to retirement", async () => {
  const s = await run("pylon", { kind: "submit", source: "|t : (2,1,0,+)| not { k } do { emit self.n } to { * }", goal: [0.003, 0.004] });
  assert.equal(s.ok, true);
  const id = (s.output_delta as any).yield.agent as string;
  const before = s.residue;
  const step = await run("pylon", { kind: "step", agent: id }, 100);
  assert.ok(step.residue < before);
  assert.equal(step.completed, true, "goal 0.005 away, 100 ticks of 1e-3 reach it");
});

test("pylon: glyph without a negation boundary is rejected by the real parser", () => {
  const v = fed.dsls.validate("srn", "|x : (2,1,0,+)| do { emit self.n } to { n = 2 }");
  assert.equal(v.ok, false);
  assert.match(v.errors[0]?.message ?? "", /negation/);
});

test("pylon: clear refuses an empty slot list instead of letting the engine throw", async () => {
  const r = await run("pylon", { kind: "clear", agents: ["x"], slots: [] });
  assert.equal(r.error, "no slots");
});

test("tempus: the coolant lesson compiles; simulate is seeded and paged by the act budget", async () => {
  const c = await run("tempus", "demo");
  assert.equal(c.ok, true);
  const missing = await run("tempus", { kind: "simulate", totalEvents: 100 });
  assert.equal(missing.error, "invalid instruction", "simulate needs source text");
  const sim1 = await run("tempus", { kind: "simulate", source: DEMO_TEMPUS, totalEvents: 100, batchSize: 25, seed: 7 }, 2);
  const sim2 = await run("tempus", { kind: "simulate", source: DEMO_TEMPUS, totalEvents: 100, batchSize: 25, seed: 7 }, 2);
  assert.deepEqual(sim1.output_delta, sim2.output_delta, "deterministic under a seed");
  assert.equal((sim1.output_delta as any).synthetic, true);
  assert.equal((sim1.output_delta as any).generated, 50, "2 budget units × 25-event batches");
  assert.equal(sim1.completed, false);
  assert.ok(Math.abs(sim1.residue - 0.5) < 1e-12);
});

test("tempus: diagnostics lesson surfaces the engine's did-you-mean", () => {
  const v = fed.dsls.validate("tempus", "cell WARN bounds (0, 1) action 0\nwhen WARM do emit x");
  assert.equal(v.ok, false);
  assert.match(v.errors.map((e) => e.message).join(" "), /WARN/);
});

test("zangalewa-dsl: an unreachable broker is a result, never a throw", async () => {
  const r = await run("zangalewa-dsl", { kind: "generate", dslId: "vahera", instructions: "list memory" });
  assert.equal(r.ok, false);
  assert.equal(r.error, "remote unreachable");
});

const DEMO_TEMPUS = `sync coolant at 10.0e6 freq
cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7) action 1
compose d=1 channels coolant into coolant_traj
when NOMINAL do emit status_ok
when WARM do emit status_warn`;

test("lazy wasm modules: describe before load, run after; a failed load is a result", async () => {
  const { makeLazyWasmModules } = await import("../src/modules/rust-wasm.ts");
  const { loadWasmEngine } = await import("../src/wasm.ts");
  const bytes = await readFile(new URL("../wasm/buhera_modules.wasm", import.meta.url));
  const nd = makeLazyWasmModules(() => loadWasmEngine(bytes)).find((m) => m.id === "ndombolo");
  assert.equal(nd?.describe().dsl, "turbulance");
  const r = await nd!.execute("demo", 1);
  assert.equal(r.ok, true);
  assert.equal(nd!.describe().binding, "native");
  const [broken] = makeLazyWasmModules(() => Promise.reject(new Error("404")));
  const b = await broken!.execute("demo", 1);
  assert.equal(b.error, "engine unavailable");
});

test("remote (gateway): verbatim result with executed_on; transport failures are results", async () => {
  const { makeRemoteModule } = await import("../src/modules/remote.ts");
  const fakeFetch = (async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    assert.equal(url, "https://gw.example/api/dispatch");
    assert.equal((init.headers as Record<string, string>)["Authorization"], "Bearer t0k");
    return new Response(JSON.stringify({ executed_on: "gateway", act_id: 7, result: { ok: true, output_delta: { kind: "sbs_core_result", module: body.module }, residue: 0.88, completed: true } }), { status: 200 });
  }) as typeof fetch;
  const m = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "https://gw.example/", token: () => "t0k", fetch: fakeFetch });
  const r = await m.execute("demo", 1);
  assert.equal(r.residue, 0.88);
  assert.equal((r.output_delta as any).executed_on, "gateway");
  assert.equal((r.output_delta as any).remote_act_id, 7);
  const signedOut = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "x", token: () => null });
  assert.equal((await signedOut.execute("demo", 1)).error, "remote unauthorized");
  const down = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "http://127.0.0.1:9", token: () => "t" });
  assert.equal((await down.execute("demo", 1)).error, "remote unreachable");
});

test("smith: the canonical front end types the demo and runs it deterministically, models off", async () => {
  const chk = await run("smith", { source: SMITH_CLERK });
  assert.equal(chk.ok, true);
  const a = (chk.output_delta as any).agents[0];
  assert.equal(a.name, "clerk");
  assert.equal(a.chi, 4);
  assert.equal(a.floor, 4);
  const r1 = await run("smith", "demo");
  const r2 = await run("smith", "demo");
  const strip = (d: any) => JSON.stringify({ ...d, steps: d.steps.map((s: any) => ({ ...s, content: null })) });
  assert.equal(strip(r1.output_delta), strip(r2.output_delta), "deterministic across runs");
  assert.ok((r1.output_delta as any).steps.length > 0);
  assert.equal(r1.completed, true, "a reach task-agent halts at quiescence");
  assert.ok((r1.output_delta as any).steps.every((s: any) => s.model == null), "no model was called");
});

test("smith: typing rules come from the real checker (disconnected self-graph, unknown potential)", () => {
  const disconnected = SMITH_CLERK.replace("(patience, memory: 2)", "(memory, memory: 2)");
  assert.equal(fed.dsls.validate("smith", disconnected).ok, false);
  const v = fed.dsls.validate("smith", SMITH_CLERK.replace("minimise backlog", "minimise Threat"));
  assert.equal(v.ok, false);
  assert.match(v.errors[0]?.message ?? "", /convex/);
});

const SMITH_CLERK = `agent clerk {
  purpose minimise backlog
  scenes {
    scene counter serves backlog with serve_hook
    scene filing  serves backlog with file_hook
  }
  self {
    parts { memory, manner, patience }
    separations {
      (memory, manner: 2), (manner, patience: 3), (patience, memory: 2)
    }
  }
  budget 1.0
  floor  2.0
}`;

test("synopsis: the upstream conformance corpus — positives check, negatives refuse with the declared class", async () => {
  const engines = await realEngines();
  const { registry, dsls } = createFederation({ synopsis: engines.synopsis });
  const corpus = JSON.parse(await readFile(new URL("../../../long-grass/vendor/synopsis/corpus/corpus.json", import.meta.url), "utf8"));
  const { isSubclassOf } = engines.synopsis as unknown as { isSubclassOf: (got: string, want: string) => boolean };
  for (const p of corpus.positive) {
    const r = await registry.dispatch("synopsis", p.src);
    assert.equal(r.ok, true, `${p.name}: ${r.error}`);
    assert.equal(r.residue, 0);
    assert.equal(dsls.validate("synopsis", p.src).ok, true, p.name);
  }
  for (const n of corpus.negative) {
    const r = await registry.dispatch("synopsis", n.src);
    assert.equal(r.ok, false, n.name);
    const cls = (r.output_delta as unknown as { refusal: { className: string } }).refusal.className;
    assert.ok(isSubclassOf(cls, n.expect), `${n.name}: got ${cls}, expected ${n.expect} or a subclass`);
    const v = dsls.validate("synopsis", n.src);
    assert.equal(v.ok, false, n.name);
    assert.ok(v.errors[0]!.message.startsWith(cls), n.name);
  }
});

test("synopsis: parse/tokens are plain JSON; running is refused, never approximated", async () => {
  const engines = await realEngines();
  const { registry } = createFederation({ synopsis: engines.synopsis });
  const corpus = JSON.parse(await readFile(new URL("../../../long-grass/vendor/synopsis/corpus/corpus.json", import.meta.url), "utf8"));
  const src = corpus.positive[0].src;
  const ast = await registry.dispatch("synopsis", { kind: "parse", source: src });
  assert.equal(ast.ok, true);
  assert.deepEqual(JSON.parse(JSON.stringify(ast.output_delta)), ast.output_delta, "no Maps survive into the delta");
  const toks = await registry.dispatch("synopsis", { kind: "tokens", source: src });
  assert.ok((toks.output_delta as unknown as { count: number }).count > 10);
  const run = await registry.dispatch("synopsis", { kind: "run", source: src });
  assert.equal(run.ok, false);
  assert.equal(run.error, "no evaluator exists");
  const report = await registry.dispatch("synopsis", src);
  const params = (report.output_delta as unknown as { report: { parameters: Record<string, unknown> } }).report.parameters;
  assert.ok(Object.keys(params).length > 0, "the checker's report records the script's parameters");
});

test("cfc: the five upstream examples reproduce their statuses; verdicts carry tolerances; ERROR is the only failure", async () => {
  const engines = await realEngines();
  const { registry, dsls } = createFederation({ cfc: engines.cfc });
  const expect: Record<string, { ok: boolean; status: string; residue?: number }> = {
    "01_validity_gate.cfc": { ok: true, status: "OK", residue: 0 },
    "02_undecidable.cfc": { ok: true, status: "OK", residue: 1 },
    "03_invalid_reference.cfc": { ok: true, status: "INVALID", residue: 1 },
    "04_rejected.cfc": { ok: false, status: "ERROR" },
    "05_node_vs_edge.cfc": { ok: true, status: "OK", residue: 0 },
  };
  for (const [name, want] of Object.entries(expect)) {
    const r = await registry.dispatch("cfc", { kind: "example", name });
    const d = r.output_delta as unknown as { status: string; verdicts: Array<{ verdict: string; tolerance: { star: number } }> };
    assert.equal(r.ok, want.ok, name);
    assert.equal(d.status, want.status, name);
    if (want.residue != null) assert.equal(r.residue, want.residue, name);
    for (const v of d.verdicts) assert.equal(typeof v.tolerance.star, "number", `${name}: a verdict never exists without its tolerance`);
    const src = (engines.cfc as unknown as { EXAMPLES: Record<string, string> }).EXAMPLES[name]!;
    assert.equal(dsls.validate("cfc", src).ok, want.status !== "ERROR", name);
  }
  const demo = await registry.dispatch("cfc", "demo");
  assert.deepEqual((demo.output_delta as unknown as { witnessSet: string[] }).witnessSet, ["SHNT"]);
  assert.equal((demo.output_delta as unknown as { committedMeasurements: number }).committedMeasurements, 5);
  const bad = dsls.validate("cfc", "floor 1e-9\nadmit h yield v\n");
  assert.equal(bad.ok, false);
  assert.equal(typeof bad.errors[0]!.line, "number");
});

test("sthurbert: load a symbol index, query it with the engine's own χ; refusals carry line/column", async () => {
  const engines = await realEngines();
  const { registry, dsls } = createFederation({ sthurbert: engines.sthurbert });
  const empty = await registry.dispatch("sthurbert", "navigate * ; show chi");
  assert.equal(empty.ok, false, "no repos analysed yet is a runtime refusal");
  const demo = await registry.dispatch("sthurbert", "demo");
  assert.equal(demo.ok, true, demo.error);
  const blocks = (demo.output_delta as unknown as { blocks: Array<{ kind: string; lines: string[] }> }).blocks;
  assert.ok(blocks.some((b) => b.kind === "chi"), "show chi yields a chi block");
  const syms = blocks.find((b) => b.kind === "symbols")!;
  assert.ok(syms.lines.some((l) => l.includes("alpha")) && syms.lines.some((l) => l.includes("gamma")));
  const again = await registry.dispatch("sthurbert", "navigate demo ; find \"Delta\"");
  assert.equal(again.ok, true, "the federation persists across acts (R6)");
  const v = dsls.validate("sthurbert", "show nope");
  assert.equal(v.ok, false);
  assert.equal(v.errors[0]!.line, 1);
  assert.equal(typeof v.errors[0]!.column, "number");
  assert.equal(dsls.validate("sthurbert", "slice where line > abc").ok, false);
  const reset = await registry.dispatch("sthurbert", "reset");
  assert.equal(reset.ok, true);
  assert.equal((await registry.dispatch("sthurbert", "navigate demo ; show chi")).ok, false);
});

test("honjo: the four upstream examples run; M is honjo's clock; errors carry line/col; range errors surface at run", async () => {
  const engines = await realEngines();
  const { registry, dsls } = createFederation({ honjo: engines.honjo });
  const dir = new URL("../../../long-grass/vendor/honjo/examples/", import.meta.url);
  const M: Record<string, number> = { "carbon.hj": 1, "salt.hj": 4, "track.hj": 6, "water.hj": 5 };
  for (const [file, m] of Object.entries(M)) {
    const src = await readFile(new URL(file, dir), "utf8");
    assert.equal(dsls.validate("honjo", src).ok, true, file);
    const r = await registry.dispatch("honjo", src);
    assert.equal(r.ok, true, `${file}: ${r.error}`);
    assert.equal((r.output_delta as unknown as { cutCount: number }).cutCount, m, file);
    assert.equal(r.residue, 0);
  }
  const water = await registry.dispatch("honjo", await readFile(new URL("water.hj", dir), "utf8"));
  assert.match((water.output_delta as unknown as { values: Record<string, string> }).values["W"]!, /OH2\s+geometry=bent\s+angle=104\.5/);
  const floor0 = dsls.validate("honjo", "floor 0\nO := cut 8");
  assert.equal(floor0.ok, false);
  assert.deepEqual([floor0.errors[0]!.line, floor0.errors[0]!.column], [1, 1]);
  assert.match(floor0.errors[0]!.message, /sharp cut is not expressible/);
  assert.equal(dsls.validate("honjo", "floor 1\ny := z").errors[0]!.line, 2);
  const range = await registry.dispatch("honjo", "floor 1\nX := cut 200");
  assert.equal(range.ok, false, "Z beyond the named elements is refused when the program runs");
  const fe = await registry.dispatch("honjo", { kind: "derive", z: 26 });
  assert.equal((fe.output_delta as unknown as { atom: { symbol: string } }).atom.symbol, "Fe");
});

test("wasm: heihachi, olduvai, levinthal are the Rust modules — same numbers as the Rust tests", async () => {
  const engines = await realEngines();
  const { registry, dsls } = createFederation({ wasm: engines.wasm });
  const ex = new URL("../../vendor/heihachi/examples/", import.meta.url);
  const recall = await readFile(new URL("recall.mma", ex), "utf8");
  const reese = await readFile(new URL("reese.sgn", ex), "utf8");
  assert.equal(dsls.validate("mishima", recall).ok, true);
  assert.equal(dsls.validate("sangoma", reese).ok, true);
  const demo = await registry.dispatch("heihachi", "demo");
  const cp = (demo.output_delta as unknown as { ladders: Array<{ composite_power: number }> }).ladders[0]!.composite_power;
  assert.ok(Math.abs(cp - 0.8245) < 1e-12, `reese composite ${cp}`);
  const noNot = dsls.validate("mishima", "floor 0.02\nseek x\n  toward { region(y) }\n  via { rung a at 0.4 >> rung b at 0.4 >> rung c at 0.4 }\n  until closure\n  yield r\n");
  assert.equal(noNot.ok, false);
  assert.match(noNot.errors[0]!.message, /rule:mandatory-not/);

  const lev = await registry.dispatch("levinthal", "demo");
  const states = (lev.output_delta as unknown as { states: Array<{ n: number; l: number; m: number }> }).states.map((s) => [s.n, s.l, s.m]);
  assert.deepEqual(states, [[1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 1], [3, 2, 1]]);

  const old = await registry.dispatch("olduvai", "demo");
  assert.equal(old.ok, true);
  assert.equal(old.residue, (old.output_delta as unknown as { resolution_lost: number }).resolution_lost);
  const again = await registry.dispatch("olduvai", { kind: "nearest", coords: { s_k: 0.3, s_t: 0.6, s_e: 0.2 } });
  assert.equal(again.residue, 0, "the trie persists across acts and an exact hit loses nothing");
});
