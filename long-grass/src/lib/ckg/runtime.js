/* ============================================================================
 * Causal-Knowledge-Graph runtime — browser JS twin of ckg_runtime.py.
 *
 * The web-side model the "complete CKG experiment" tutorial runs against. It is
 * a faithful port of long-grass/docs/causal-knowledge-graph-runtime/validation/
 * ckg_runtime.py, which itself deliberately mirrors registry.js: a chunk is a
 * pure function from a node's current values to a value-delta; `dispatch(node)`
 * runs EVERY chunk in the node's bag, records each emission (an error included,
 * as a value), appends an audit entry, and JUDGES NOTHING — no exit_code, no ok,
 * no verdict. What a value means is left to the modules that read it.
 *
 * Why this file exists as its own object rather than reusing registry.js: the
 * paper's runtime is a *theory of* Buhera's federation, and the tutorial needs
 * to exhibit the theory's own properties — non-judging inertia, emergent
 * trajectory, protocol-not-results reproducibility — as first-class, inspectable
 * artifacts. registry.js is the production dispatcher; this is the specimen the
 * tutorial dissects. A chunk here CAN call a real registered module (see
 * ckg-module.js `attach`), so the specimen is driven by the live federation.
 *
 * Nothing in this file compares a value to an expectation. That inertia is the
 * whole point: the theorems are properties of it.
 * ========================================================================== */

// --- values, chunks, nodes --------------------------------------------------

/**
 * What a chunk emits. `payload` is opaque to the runtime; `kind` lets a
 * downstream *module* (never the runtime) recognise a value it cares about —
 * including an "error" record, which is just another value.
 *
 * @typedef {{ kind: string, payload: any, source_chunk: string }} ValueDelta
 */

/**
 * A node: a subtask fused with its realising code (Def. 1).
 *   tau      — subtask identity; two realisations with the same tau converge.
 *   chunks   — a *bag* of realisations (name → fn); the runtime runs all of them.
 *   values   — the time-varying values the node carries.
 *   address  — hierarchical address (coarsest..finest); depth = resolution.
 */
export function makeNode(tau, address = []) {
  return {
    tau,
    chunks: {}, // name → (values) => ValueDelta   (may be async; may throw)
    values: {},
    address: Array.isArray(address) ? address.slice() : [],
  };
}

/**
 * Add a chunk to a node's bag. Convergence merges bags: adding onto an existing
 * tau accretes rather than replaces the catalogue — same-name overwrites, which
 * is how a coarse import is later refined at a finer address.
 */
export function addChunk(node, name, chunk) {
  node.chunks[name] = chunk;
  return node;
}

// --- the runtime: execute all chunks, judge nothing -------------------------

export class CkgRuntime {
  constructor() {
    this.audit = []; // AuditEntry[]
    this._act = 0;
  }

  /**
   * Run every chunk in the node's bag (Def. 3). A chunk that throws produces an
   * "error" value-delta — emitted like any other value — and the run continues
   * (Thm. 2 / Cor. 4). The runtime never branches on the *content* of an
   * emission. Chunks may be async (a real module dispatch is async); we await.
   *
   * @returns {Promise<ValueDelta[]>} the deltas emitted this dispatch
   */
  async dispatch(node) {
    const deltas = [];
    for (const name of Object.keys(node.chunks)) {
      const chunk = node.chunks[name];
      this._act += 1;
      let raised = false;
      let delta;
      try {
        delta = await chunk(node.values);
        // A chunk that returns nothing is a silent no-op — still audited, but
        // it emits no value. Model it as an explicit empty delta so the audit
        // records the act without asserting a fact onto the node.
        if (delta == null) {
          delta = { kind: "noop", payload: null, source_chunk: name };
        }
      } catch (exc) {
        raised = true;
        delta = { kind: "error", payload: String(exc && exc.message ? exc.message : exc), source_chunk: name };
      }
      // Emit onto the node. The runtime does not inspect payload meaning.
      // "noop" is recorded but does not clobber a real value channel.
      if (delta.kind !== "noop") node.values[delta.kind] = delta.payload;
      deltas.push(delta);
      this.audit.push({
        act_id: this._act,
        tau: node.tau,
        chunk: name,
        emitted_kind: delta.kind,
        raised,
      });
    }
    return deltas;
  }

  // Deliberately no exit_code, no ok, no verdict.
}

// --- modules: read / transform / emit ---------------------------------------

/**
 * A competence that reads node values, transforms internally, and emits.
 * Whether it reads a given node in a run is decided by `wants`, which consults
 * values *already present in this run* — the mechanism behind trajectory
 * emergence (Thm. 5): the causal edge relation is a product of the run, not an
 * input to it.
 */
export function makeReader(id, wants, emit) {
  return {
    id,
    wants,
    async readTransformEmit(node) {
      if (!wants(node)) return null;
      const delta = await emit(node);
      if (delta != null) node.values[delta.kind] = delta.payload;
      return delta;
    },
  };
}

// --- provenance fingerprint (reproducibility of protocol, not results) ------

/**
 * Structure of a node independent of its (run-varying) values.
 */
export function tauShape(node) {
  return node.tau + "{" + Object.keys(node.chunks).sort().join(",") + "}";
}

/**
 * A stable hash of the *protocol*: imported subtree + chunk bag + edits.
 * Deliberately EXCLUDES node values — results are not expected to repeat and
 * must not enter the fingerprint (Prop. 8). What repeats is this string.
 *
 * Uses the same FNV-1a squash the smith compiler uses for its deterministic
 * seed, so the webtool stays dependency-free (no crypto import) while remaining
 * stable across runs.
 */
export function provenanceFingerprint(nodes, editAddresses = []) {
  const parts = [];
  const sorted = nodes
    .slice()
    .sort((a, b) => a.address.join("/").localeCompare(b.address.join("/")));
  for (const n of sorted) {
    parts.push(n.address.join("/") + ":" + tauShape(n));
  }
  parts.push(
    "EDITS=" +
      editAddresses
        .map((a) => a.join("/"))
        .sort()
        .join(";")
  );
  return fnv1aHex(parts.join("|"));
}

function fnv1aHex(str) {
  // 64-bit-ish via two interleaved 32-bit FNV-1a lanes, hex-joined. Enough to
  // separate distinct protocols legibly in a tutorial; not a security hash.
  let h1 = 0x811c9dc5;
  let h2 = 0x811c9dc5 ^ 0x5bd1e995;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    h1 ^= c;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= (c + i) & 0xff;
    h2 = Math.imul(h2, 0x01000193) >>> 0;
  }
  const hex = (n) => (n >>> 0).toString(16).padStart(8, "0");
  return hex(h1) + hex(h2);
}
