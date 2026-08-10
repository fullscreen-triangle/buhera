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
 * the node as a value-delta. The value `kind` is namespaced by the module id
 * AND the chunk name (`fact:<module>#<chunk>`), so a single node can carry
 * SEVERAL facts from the same module without the later chunk clobbering the
 * earlier one's channel — a P450 `measure` node can hold the module's Compound-I
 * chemistry fact and its Fe=O spectroscopy fact at once, because they arrived as
 * two distinct chunks. (The runtime folds by `delta.kind` into `node.values`;
 * two chunks sharing a kind would collide, so the chunk name disambiguates.)
 * The module id is still recoverable — `report`/`graph` split on `#`. Errors
 * from the dispatch are NOT swallowed here — they propagate, and
 * CkgRuntime.dispatch turns them into a recorded "error" value (Thm. 2): a
 * contributor failing is itself a fact the run records, not a halt.
 */
function moduleChunk(moduleId, instruction, chunkName) {
  const channel = chunkName ? `fact:${moduleId}#${chunkName}` : `fact:${moduleId}`;
  return async function chunk(_values) {
    const res = await dispatchModule(moduleId, instruction);
    const delta = res && res.output_delta;
    return {
      kind: channel,
      payload: {
        module: moduleId,
        ok: res && res.ok,
        // Carry the module's WHOLE output_delta. It is exactly what the module
        // returns when you dispatch it directly — the same charts, workspaces,
        // metrics — so the renderer can hand it back to <Artifact> and draw the
        // module's own view. The fingerprint hashes tau + chunk names only, so
        // stashing a rich value here never leaks into the protocol hash.
        delta,
        // A small, readable digest of the named properties the module asserted,
        // so the report has a headline without re-parsing the whole delta.
        findings: deriveFindings(delta),
      },
      source_chunk: chunkName || `${moduleId}_chunk`,
    };
  };
}

/** Recover the contributing module id from a `fact:<module>[#<chunk>]` key. */
function moduleIdOf(factKey) {
  return factKey.slice("fact:".length).split("#")[0];
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

/**
 * Extract a compact digest of the NAMED properties a module asserted, WITHOUT
 * discarding anything — the full delta always travels alongside this (see
 * moduleChunk). This is the report's headline: "what did shapeshifter actually
 * find here", as a few { label, value } pairs, per known module output kind. The
 * full charts render from the delta itself; this is just the caption.
 *
 * Returns { kind, headline, props: [{label, value}] } or null for an empty delta.
 */
function deriveFindings(delta) {
  if (delta == null) return null;
  if (typeof delta !== "object") return { kind: "scalar", headline: String(delta), props: [] };

  const props = [];
  const push = (label, value) => {
    if (value != null && value !== "") props.push({ label, value });
  };
  const kind = delta.kind || "opaque";

  switch (kind) {
    case "sbs_result": {
      // Systems-Biology-Shaders: a cellular circuit + its S-entropy observation.
      const c = delta.circuit || {};
      const m = delta.metrics || {};
      push("nodes", c.numNodes);
      push("edges", c.numEdges);
      if (typeof m.R === "number") push("coherence R", m.R.toFixed(3));
      if (typeof m.V === "number") push("flux visibility V", m.V.toFixed(3));
      push("backend", m.backend);
      if (delta.navigation && delta.navigation.from) push("navigated from", delta.navigation.from);
      return { kind, headline: delta.summary || "SBS circuit", props };
    }
    case "shapeshifter_run": {
      // Shape Shifter: virtual mass-spec — the produced workspace is the finding.
      const ws = Array.isArray(delta.workspace) ? delta.workspace : [];
      push("workspace values", ws.length);
      for (const w of ws) push(w.name || "value", w.kind || "");
      const errs = (delta.diagnostics || []).filter((d) => d.severity === "error").length;
      if (errs) push("diagnostics (error)", errs);
      return {
        kind,
        headline: ws.length
          ? `spectra → ${ws.map((w) => w.name).filter(Boolean).join(", ")}`
          : "compiled (no workspace produced)",
        props,
      };
    }
    case "scope_run": {
      // SCOPE: microscopy — surface whatever the run summary carries.
      const r = delta.result || {};
      push("observations", Array.isArray(r.observations) ? r.observations.length : r.count);
      push("summary", typeof r.summary === "string" ? r.summary : undefined);
      return { kind, headline: "microscopy observation", props };
    }
    case "lavoisier_run": {
      const s = delta.summary || {};
      push("records", s.count);
      if (s.avgEntropy != null) push("avg S-entropy", Number(s.avgEntropy).toFixed(3));
      return { kind, headline: "instrument run", props };
    }
    case "graffiti_result": {
      push("projects", Array.isArray(delta.projects) ? delta.projects.length : undefined);
      if (delta.ambient_floor != null) push("ambient floor β", Number(delta.ambient_floor).toFixed(3));
      return { kind, headline: "graffiti scan", props };
    }
    case "text": {
      const lines = Array.isArray(delta.lines) ? delta.lines : [];
      return { kind, headline: lines[0] || "(text)", props };
    }
    // --- cytochrome P450 monograph contributors -----------------------------
    case "cyp_electron_transfer": {
      push("d_C", delta.d_C);
      push("Marcus λ (eV)", delta.marcus_lambda_eV);
      push("rate-limiting", delta.rate_limiting);
      push("total ΔM", delta.total_dM);
      return { kind, headline: delta.summary || "electron-transfer chain", props };
    }
    case "cyp_compound_i": {
      push("aperture d_C", delta.aperture_dC);
      push("ΔM", delta.dM);
      push("mechanism", delta.mechanism);
      push("KIE", delta.kie);
      return { kind, headline: delta.summary || "Compound I formation", props };
    }
    case "cyp_pathway": {
      if (delta.ok === false) return { kind, headline: `error: ${delta.error}`, props };
      push("family", delta.family);
      push("ΔM", delta.dM);
      push("KIE", delta.kie);
      push("RDS", delta.rate_determining_step);
      return { kind, headline: delta.summary || "reaction pathway", props };
    }
    case "cyp_states": {
      push("states", Array.isArray(delta.states) ? delta.states.length : undefined);
      push("closed", delta.closed);
      push("orbit ΣΔM", delta.orbit_sum_dM);
      return { kind, headline: delta.summary || "catalytic closed orbit", props };
    }
    case "cyp_spectroscopy": {
      if (delta.soret) push("Soret shift (nm)", `${delta.soret.resting_nm}→${delta.soret.compound_i_nm}`);
      if (delta.raman) push("Fe=O Raman (cm⁻¹)", delta.raman.fe_o_cm);
      return { kind, headline: delta.summary || "spectroscopy", props };
    }
    case "cyp_soret": {
      push("resting (nm)", delta.resting_nm);
      push("Compound I (nm)", delta.compound_i_nm);
      push("shift (nm)", delta.shift_nm);
      return { kind, headline: delta.summary || "Soret band", props };
    }
    case "cyp_epr": {
      push("g_low", delta.g_low);
      push("g_mid", delta.g_mid);
      push("g_high", delta.g_high);
      return { kind, headline: delta.summary || "EPR g-tensor", props };
    }
    case "cyp_raman": {
      push("Fe=O (cm⁻¹)", delta.fe_o_cm);
      push("¹⁸O (cm⁻¹)", delta.fe_o_18O_cm);
      push("shift (cm⁻¹)", delta.shift_cm);
      return { kind, headline: delta.summary || "resonance Raman Fe=O", props };
    }
    case "cyp_isoform": {
      push("CYP", delta.cyp);
      push("family", delta.family);
      if (delta.phenotype) push("phenotype", delta.phenotype);
      if (delta.dM != null) push("ΔM", delta.dM);
      return { kind, headline: delta.summary || "isoform", props };
    }
    case "cyp_participants": {
      push("participants", Array.isArray(delta.participants) ? delta.participants.length : undefined);
      push("carriers", Array.isArray(delta.carriers) ? delta.carriers.length : undefined);
      push("cuts M", delta.M);
      if (delta.floor != null) push("floor β", Number(delta.floor).toExponential(2));
      return { kind, headline: delta.invariant || "participant/carrier cut", props };
    }
    case "cyp_floor": {
      if (delta.beta != null) push("β", Number(delta.beta).toExponential(3));
      push("dominant term", delta.dominant_term);
      return { kind, headline: delta.summary || "conditioned floor", props };
    }
    default: {
      // Unknown module output — still give a headline, still keep the delta.
      if (typeof delta.summary === "string") return { kind, headline: delta.summary, props };
      if (delta.ok === false && delta.error) return { kind, headline: `error: ${delta.error}`, props };
      return { kind, headline: kind, props };
    }
  }
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
  // A dossier, not a tally: per contributing module, the list of subtasks it
  // spoke on, each carrying that module's FULL output_delta (so the renderer can
  // draw the module's own chart) plus the extracted findings headline. This is
  // the account the original pipeline never produced — the assembled findings,
  // with the real artifacts, not a table of counts.
  const contributions = {}; // moduleId → [{ tau, ok, findings, delta }]
  let errorCount = 0;
  for (const tau of _order) {
    const n = _nodes.get(tau);
    for (const k of Object.keys(n.values)) {
      if (k.startsWith("fact:")) {
        const mod = moduleIdOf(k);
        const v = n.values[k] || {};
        (contributions[mod] = contributions[mod] || []).push({
          tau,
          channel: k,
          ok: v.ok,
          findings: v.findings || null,
          // the whole module payload — same object <Artifact> renders when you
          // dispatch the module directly.
          delta: v.delta,
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
        {
          const chunkName = instr.name || `${instr.module}_chunk`;
          addChunk(node, chunkName, moduleChunk(instr.module, instr.instruction ?? "demo", chunkName));
        }
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
