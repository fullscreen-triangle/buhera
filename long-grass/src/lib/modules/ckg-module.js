/* ============================================================================
 * CKG Module — represent a problem as a runtime, let the federation reason it.
 *
 * This is the module the "complete CKG experiment" tutorial drives. It holds a
 * live CkgRuntime session and a catalogue of nodes, and exposes the verbs that
 * turn the three artifacts of a hand-built causal-knowledge-graph pipeline into
 * three views of ONE runtime trajectory:
 *
 *   ontology model  →  the node's tau + the chunk-bag you attach (`represent`,
 *                       `attach`) — structure handed in, before any run.
 *   reasoner        →  `dispatch` running the federation. Derived facts are
 *                       *emitted value-deltas*, not schema entailments. Which
 *                       module reads which node is decided DURING the run.
 *   SPARQL access   →  `graph` / `report` — you walk the graph the run produced
 *                       and read the account assembled from emitted values. The
 *                       original gave results and no report; this gives both.
 *
 * A chunk attached by `attach` calls a REAL registered module and folds its
 * output_delta onto the node as a value. So the experiment is genuinely live:
 * lavoisier/shapeshifter/sbs/graffiti/purpose-carry do real work, and the
 * trajectory that emerges is one no ontology you could have authored produces.
 *
 * The runtime judges nothing (CkgRuntime.dispatch has no verdict). This module
 * only *projects* — it never decides a value is right or wrong.
 *
 * Instruction shapes (vaHera cell forms — `dispatch("ckg", <instr>)`):
 *   { op: "reset" }
 *   { op: "represent", tau, address?, seed? }         — make/get a node
 *   { op: "attach", tau, name, module, instruction? } — chunk = live module call
 *   { op: "seed", tau, value }                        — set the emergence seed
 *   { op: "dispatch", tau }                           — run all chunks on a node
 *   { op: "carry", from?, magnitudeKey? }             — run the emergent carry
 *   { op: "graph" }                                   — project the trajectory
 *   { op: "report" }                                  — assemble the report
 *   { op: "fingerprint", edits? }                     — protocol hash (Prop. 8)
 * ========================================================================== */

import {
  CkgRuntime,
  makeNode,
  addChunk,
  makeReader,
  provenanceFingerprint,
} from "../ckg/runtime";
import { dispatch as dispatchModule } from "./registry";

// --- the live session -------------------------------------------------------
//
// One runtime, one node catalogue, one edge set, held for the browser session.
// `reset` rebuilds them. The tutorial runs many cells against this session, so
// state accretes exactly the way convergence (same tau accretes) predicts.

let _rt = new CkgRuntime();
let _nodes = new Map(); // tau → node
let _edges = []; // { from, to, via, magnitude } — the emergent trajectory
let _order = []; // insertion order of taus, used by the carry's reach

function reset() {
  _rt = new CkgRuntime();
  _nodes = new Map();
  _edges = [];
  _order = [];
}

function getOrMakeNode(tau, address, seed) {
  let node = _nodes.get(tau);
  if (!node) {
    node = makeNode(tau, address || ["ckg", tau]);
    _nodes.set(tau, node);
    _order.push(tau);
  } else if (address && address.length) {
    node.address = address.slice();
  }
  if (seed != null) node.values.seed = seed;
  return node;
}

// --- chunk factories --------------------------------------------------------

/**
 * A chunk that calls a real registered module and folds its output_delta onto
 * the node as a value-delta. The value `kind` is namespaced by the module id so
 * two contributors never clobber each other's channel. Errors from the module
 * dispatch are NOT swallowed here — they propagate, and CkgRuntime.dispatch
 * turns them into a recorded "error" value (Thm. 2). That is deliberate: a
 * contributor failing is itself a fact the run records, not a halt.
 */
function moduleChunk(moduleId, instruction) {
  return async function chunk(_values) {
    const res = await dispatchModule(moduleId, instruction);
    return {
      kind: `fact:${moduleId}`,
      payload: {
        module: moduleId,
        ok: res && res.ok,
        // keep the delta small and legible; the tutorial reads a summary, not
        // the whole module payload.
        delta: summariseDelta(res && res.output_delta),
      },
      source_chunk: `${moduleId}_chunk`,
    };
  };
}

/**
 * A "publish" chunk (the trajectory_emergence.py seed): emits a signal one
 * greater than the node's current seed. This is the value the emergent carry
 * reads to decide its reach, so it is what makes two seeds yield two
 * trajectories over the same nodes.
 */
function publishChunk() {
  return function chunk(values) {
    return {
      kind: "signal",
      payload: (values.seed || 0) + 1,
      source_chunk: "publish",
    };
  };
}

function summariseDelta(delta) {
  if (delta == null) return null;
  if (typeof delta !== "object") return delta;
  // Pull a few well-known summary fields the federation modules expose, so the
  // report is readable without dumping entire payloads.
  const pick = {};
  if (delta.kind) pick.kind = delta.kind;
  if (delta.summary && typeof delta.summary === "object") {
    if (delta.summary.count != null) pick.count = delta.summary.count;
    if (delta.summary.avgEntropy != null) pick.avgEntropy = delta.summary.avgEntropy;
  }
  if (delta.ambient_floor != null) pick.ambient_floor = delta.ambient_floor;
  if (delta.value != null && typeof delta.value !== "object") pick.value = delta.value;
  if (Array.isArray(delta.keep)) pick.keep = delta.keep.length;
  if (Array.isArray(delta.workspace)) pick.workspace = delta.workspace.length;
  if (delta.summary && typeof delta.summary === "string") pick.summary = delta.summary;
  return Object.keys(pick).length ? pick : { kind: delta.kind || "opaque" };
}

// --- the emergent carry (Thm. 5 / Prop. 1, ported from the validator) -------
//
// A reader that carries a node's emitted `signal` onward. The *reach* of the
// carry — how many nodes ahead in insertion order it lands — is the signal's
// magnitude, a value produced only during this run. A node that inherited a
// larger seed reaches further, so seeding a different node reshapes which reads
// happen and thus which edges exist. The edges it lands are the trajectory.

async function runCarry(magnitudeKey = "signal") {
  const carrier = makeReader(
    "carrier",
    (n) => magnitudeKey in n.values,
    (u) => ({ kind: "carried", payload: u.values[magnitudeKey] || 0, source_chunk: "fwd" })
  );

  _edges = [];
  for (let i = 0; i < _order.length; i++) {
    const u = _nodes.get(_order[i]);
    await _rt.dispatch(u); // runs the node's bag → emits signal (+ any facts)
    const carried = await carrier.readTransformEmit(u);
    if (carried == null) continue;
    const reach = Math.max(1, Math.floor(Number(carried.payload) || 0));
    const j = i + reach;
    if (j < _order.length) {
      const v = _nodes.get(_order[j]);
      _edges.push({ from: u.tau, to: v.tau, via: "carrier", magnitude: reach });
      // propagate: v inherits the carry, so its own later reach is shaped by
      // what reached it this run.
      v.values.seed = Math.max(v.values.seed || 0, Number(carried.payload) || 0);
    }
  }
  return _edges.slice();
}

// --- projections ------------------------------------------------------------

/**
 * Project the trajectory as a knowledge graph. Nodes = touched subtasks; edges
 * = carrier reads that landed (the run-induced causal relation); facts = the
 * emitted value-deltas each node carries (excluding the internal seed/signal
 * bookkeeping). This is the SPARQL replacement: you walk the graph the run
 * produced, you do not query a graph you authored.
 */
function projectGraph() {
  const nodes = _order.map((tau) => {
    const n = _nodes.get(tau);
    const facts = Object.keys(n.values)
      .filter((k) => k.startsWith("fact:") || k === "error")
      .map((k) => ({ predicate: k, object: n.values[k] }));
    return {
      tau: n.tau,
      address: n.address,
      signal: n.values.signal,
      facts,
    };
  });
  return {
    kind: "ckg_graph",
    nodes,
    edges: _edges.slice(),
    node_count: nodes.length,
    edge_count: _edges.length,
    fact_count: nodes.reduce((s, n) => s + n.facts.length, 0),
  };
}

/**
 * Assemble the report from emitted values — the account the original pipeline
 * never produced. It reads the audit (what ran) and the nodes' emitted facts
 * (what the federation asserted), and lays them out per contributing module. It
 * judges nothing: an "error" fact is reported as a fact, not a failure.
 */
function assembleReport() {
  const contributions = {}; // moduleId → [{ tau, ok, delta }]
  let errorCount = 0;
  for (const tau of _order) {
    const n = _nodes.get(tau);
    for (const k of Object.keys(n.values)) {
      if (k.startsWith("fact:")) {
        const mod = k.slice("fact:".length);
        (contributions[mod] = contributions[mod] || []).push({
          tau,
          ok: n.values[k] && n.values[k].ok,
          delta: n.values[k] && n.values[k].delta,
        });
      } else if (k === "error") {
        errorCount += 1;
      }
    }
  }
  return {
    kind: "ckg_report",
    tau_count: _order.length,
    acts: _rt.audit.length,
    edges: _edges.length,
    error_facts: errorCount,
    contributors: Object.keys(contributions).sort(),
    contributions,
    // the audit is the run-to-completion witness: every act appears, none was
    // gated on a verdict (Cor. 4).
    audit: _rt.audit.slice(),
  };
}

// --- the module -------------------------------------------------------------

export const ckgModule = {
  id: "ckg",

  describe() {
    return {
      id: "ckg",
      description:
        "CKG: represent a problem as a runtime and let the federation reason it. " +
        "Attach chunks that call real modules; dispatch (judging nothing); then " +
        "project the trajectory as a knowledge graph, a report, and a protocol " +
        "fingerprint. The knowledge graph IS the runtime trajectory.",
      instructions: [
        'dispatch("ckg", { op: "represent", tau: "assay", seed: 1 })',
        'dispatch("ckg", { op: "attach", tau: "assay", name: "spectra", module: "lavoisier", instruction: "demo" })',
        'dispatch("ckg", { op: "carry" })',
        'dispatch("ckg", { op: "graph" })',
        'dispatch("ckg", { op: "report" })',
        'dispatch("ckg", { op: "fingerprint" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const instr =
      typeof instruction === "string" ? { op: instruction } : instruction || {};
    const op = instr.op;

    try {
      if (op === "reset") {
        reset();
        return ok({ kind: "ckg_ack", op, message: "runtime reset" });
      }

      if (op === "represent") {
        const node = getOrMakeNode(instr.tau, instr.address, instr.seed);
        // every represented node gets the publish chunk so the emergent carry
        // has a signal to read — unless the caller opted out.
        if (instr.publish !== false && !node.chunks.publish) {
          addChunk(node, "publish", publishChunk());
        }
        return ok({
          kind: "ckg_ack",
          op,
          tau: node.tau,
          address: node.address,
          chunks: Object.keys(node.chunks),
          message: `node "${node.tau}" represented at ${node.address.join("/")}`,
        });
      }

      if (op === "attach") {
        const node = _nodes.get(instr.tau);
        if (!node) return fail(`attach: no node "${instr.tau}" — represent it first`);
        addChunk(
          node,
          instr.name || `${instr.module}_chunk`,
          moduleChunk(instr.module, instr.instruction ?? "demo")
        );
        return ok({
          kind: "ckg_ack",
          op,
          tau: node.tau,
          chunks: Object.keys(node.chunks),
          message: `attached ${instr.module} as "${instr.name || instr.module + "_chunk"}" on "${node.tau}"`,
        });
      }

      if (op === "seed") {
        const node = _nodes.get(instr.tau);
        if (!node) return fail(`seed: no node "${instr.tau}"`);
        node.values.seed = instr.value;
        return ok({ kind: "ckg_ack", op, tau: node.tau, message: `seed(${node.tau}) = ${instr.value}` });
      }

      if (op === "dispatch") {
        const node = _nodes.get(instr.tau);
        if (!node) return fail(`dispatch: no node "${instr.tau}"`);
        const deltas = await _rt.dispatch(node);
        return ok({
          kind: "ckg_ack",
          op,
          tau: node.tau,
          emitted: deltas.map((d) => d.kind),
          message: `dispatched "${node.tau}" — ${deltas.length} chunk(s), judged nothing`,
        });
      }

      if (op === "carry") {
        const edges = await runCarry(instr.magnitudeKey);
        return ok({
          kind: "ckg_ack",
          op,
          edges: edges.length,
          trajectory: edges.map((e) => `${e.from}→${e.to}`),
          message: `carry ran to completion — ${edges.length} edge(s) induced this run`,
        });
      }

      if (op === "graph") return ok(projectGraph());
      if (op === "report") return ok(assembleReport());

      if (op === "fingerprint") {
        const fp = provenanceFingerprint(
          Array.from(_nodes.values()),
          instr.edits || []
        );
        return ok({
          kind: "ckg_fingerprint",
          fingerprint: fp,
          nodes: _order.length,
          edits: (instr.edits || []).map((a) => a.join("/")),
          note: "hashes the protocol (tau + chunk bag + edits); excludes run values (Prop. 8)",
        });
      }

      return fail(`unknown op "${op}" — try represent | attach | carry | graph | report | fingerprint`);
    } catch (err) {
      return fail(err && err.message ? err.message : String(err));
    }
  },

  outputCell() {
    return { kind: "ckg_cell" };
  },
};

function ok(output_delta) {
  return { ok: true, output_delta, residue: 0, completed: true };
}

function fail(message) {
  return {
    ok: false,
    output_delta: { kind: "ckg_ack", op: "error", message },
    residue: 0,
    completed: true,
    error: message,
  };
}
