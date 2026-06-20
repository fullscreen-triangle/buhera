// vahera.js — parser + interpreter for vaHera, covering all 15
// statement kinds (matching the Rust crate `buhera-vahera`).
//
// The interpreter dispatches against a Kernel (from kernel.js) and
// returns either a single artifact (for the protein-style demo where
// a NL question maps to one answer) or an array of artifacts (for
// scripts with multiple find / list / stats statements).

import { embedText, embedProtein, embedMolecule, sDistance, tokenOverlap } from "./substrate";
import { resolveProtein, PROTEINS } from "./proteins";

// ─────────────────────────────────────────────────────────────────────
//  Parser.
// ─────────────────────────────────────────────────────────────────────

const S_COORD = /S\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)/;

export function parseVahera(src) {
  const out = [];
  const aspects = [];
  let lineNo = 0;

  for (const raw of src.split("\n")) {
    lineNo++;
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("# aspect:")) {
      aspects.push(line.slice("# aspect:".length).trim());
      continue;
    }
    if (line.startsWith("#")) continue;

    let m;
    if ((m = line.match(/^describe\s+(\S+)\s+with\s+"([^"]*)"$/))) {
      out.push({ op: "describe", target: m[1], text: m[2] });
    } else if ((m = line.match(/^resolve\s+(\S+)$/))) {
      out.push({ op: "resolve", target: m[1] });
    } else if ((m = line.match(/^spawn\s+(\S+)\s+from\s+(\S+)$/))) {
      out.push({ op: "spawn", program: m[1], target: m[2] });
    } else if (line === "navigate to penultimate") {
      out.push({ op: "navigate" });
    } else if (line === "complete trajectory") {
      out.push({ op: "complete" });
    } else if (line.startsWith("memory create at")) {
      m = line.match(S_COORD);
      if (!m) throw new Error(`line ${lineNo}: expected S(k,t,e): ${line}`);
      out.push({
        op: "memory_create",
        coord: { k: parseFloat(m[1]), t: parseFloat(m[2]), e: parseFloat(m[3]) },
      });
    } else if ((m = line.match(/^memory\s+store\s+"([^"]*)"\s*=\s*"([^"]*)"$/))) {
      out.push({ op: "memory_store", name: m[1], text: m[2] });
    } else if ((m = line.match(/^memory\s+find\s+nearest\s+"([^"]*)"(?:\s+k=(\d+))?$/))) {
      out.push({ op: "memory_find", query: m[1], k: m[2] ? parseInt(m[2], 10) : 3 });
    } else if (line === "memory list") {
      out.push({ op: "memory_list" });
    } else if ((m = line.match(/^memory\s+dump\s+(\S+)$/))) {
      out.push({ op: "memory_dump", name: m[1] });
    } else if (line === "demon sort") {
      out.push({ op: "demon_sort" });
    } else if (line === "controller verify") {
      out.push({ op: "controller_verify" });
    } else if (line === "kernel stats") {
      out.push({ op: "kernel_stats" });
    } else if (line === "kernel trace") {
      out.push({ op: "kernel_trace" });
    } else if (line === "process list") {
      out.push({ op: "process_list" });
    } else {
      throw new Error(`line ${lineNo}: unknown vaHera: ${line}`);
    }
  }
  out._aspects = aspects;
  return out;
}

// ─────────────────────────────────────────────────────────────────────
//  Helpers.
// ─────────────────────────────────────────────────────────────────────

function resolveCoord(name, text, useProteinDb) {
  if (useProteinDb) {
    const key = resolveProtein(name);
    if (key) return embedProtein(key, PROTEINS[key]);
  }
  return embedText(text || name);
}

function rerankByOverlap(query, hits, boost = 0.5) {
  if (!hits.length || boost <= 0) return hits;
  const scored = hits.map((h, i) => {
    const source = h.source || (h.payload && typeof h.payload === "string" ? h.payload : "") || "";
    const overlap = tokenOverlap(query, source);
    return { i, blended: h.distance - boost * overlap };
  });
  scored.sort((a, b) => a.blended - b.blended);
  return scored.map((s) => hits[s.i]);
}

function makeProtein(obj, aspect) {
  const name = String(obj.metadata.name || "unknown");
  return { kind: "protein", name, payload: obj.payload, aspect };
}

// ─────────────────────────────────────────────────────────────────────
//  Interpreter.
// ─────────────────────────────────────────────────────────────────────

export function executeVahera(source, kernel, options = {}) {
  const {
    useProteinDb = false,
    rerank = true,
    rerankBoost = 0.5,
  } = options;

  const program = parseVahera(source);
  const aspects = program._aspects || [];
  const targets = new Map();
  const processes = new Map();
  const results = [];
  const trace = [];
  let lastResult = null;
  let activeProgram = null;
  let activeTargetName = null;

  // Aspect handling for the protein demo (compare:X, function, etc.).
  let aspect = "full";
  let compareKey = null;
  for (const a of aspects) {
    if (a.startsWith("compare:")) {
      aspect = "compare";
      compareKey = a.slice("compare:".length).trim();
    } else {
      aspect = a;
    }
  }

  for (const stmt of program) {
    switch (stmt.op) {
      case "describe": {
        const coord = resolveCoord(stmt.target, stmt.text, useProteinDb);
        targets.set(stmt.target, coord);
        trace.push(`describe ${stmt.target} -> ${coord.toString()}`);
        break;
      }

      case "resolve": {
        if (!targets.has(stmt.target)) {
          const coord = resolveCoord(stmt.target, stmt.target, useProteinDb);
          targets.set(stmt.target, coord);
        }
        const coord = targets.get(stmt.target);
        trace.push(`resolve ${stmt.target} -> ${coord.toString()}`);
        break;
      }

      case "spawn": {
        const targetCoord = targets.get(stmt.target);
        if (!targetCoord) throw new Error(`spawn: unresolved target ${stmt.target}`);
        processes.set(stmt.program, { targetCoord, state: "ready" });
        activeProgram = stmt.program;
        activeTargetName = stmt.target;
        trace.push(`spawn ${stmt.program} from ${stmt.target}`);
        break;
      }

      case "navigate": {
        if (!activeProgram) throw new Error("navigate: no active process");
        const p = processes.get(activeProgram);
        const traj = kernel.runTrajectory(p.targetCoord);
        p.state = "navigated";
        p.penultimate = traj.penultimate;
        trace.push(`navigate ${activeProgram} steps=${traj.steps}`);
        break;
      }

      case "complete": {
        if (!activeProgram) throw new Error("complete: no active process");
        const p = processes.get(activeProgram);
        const targetCoord = p.targetCoord;
        const nearest = kernel.surgicalRetrieve(targetCoord, 1);
        p.state = "completed";
        trace.push(`complete ${activeProgram}`);

        if (nearest.length > 0) {
          const [obj] = nearest[0];
          const kind = obj.metadata && obj.metadata.kind;
          if (kind === "protein" || (useProteinDb && obj.payload && obj.payload.role)) {
            if (aspect === "compare" && compareKey) {
              const key2 = resolveProtein(compareKey);
              const p2 = key2 ? PROTEINS[key2] : null;
              if (p2) {
                lastResult = {
                  kind: "protein_compare",
                  a: { name: String(obj.metadata.name || activeTargetName), payload: obj.payload },
                  b: { name: key2, payload: p2 },
                };
              } else {
                lastResult = makeProtein(obj, "full");
              }
            } else {
              lastResult = makeProtein(obj, aspect);
            }
          } else {
            const name = String(obj.metadata.name || activeTargetName || "unknown");
            const text = typeof obj.payload === "string" ? obj.payload : JSON.stringify(obj.payload);
            lastResult = {
              kind: "note",
              name,
              text,
              address: obj.address,
              coord: obj.coord,
              tier: obj.tier,
            };
          }
        } else {
          lastResult = { kind: "text", lines: ["no categorical match."] };
        }
        results.push(lastResult);
        break;
      }

      case "memory_create": {
        const obj = kernel.allocate(stmt.coord, null, {});
        trace.push(`memory_create addr=${obj.address}`);
        break;
      }

      case "memory_store": {
        const coord = embedText(stmt.text);
        const obj = kernel.allocate(coord, stmt.text, {
          name: stmt.name,
          source: stmt.text,
          kind: "note",
        });
        trace.push(`memory_store name=${stmt.name} addr=${obj.address}`);
        break;
      }

      case "memory_find": {
        const qCoord = embedText(stmt.query);
        let hitsRaw = kernel.surgicalRetrieve(qCoord, Math.max(stmt.k, 8));
        let hits = hitsRaw.map(([o, d]) => ({
          name: String(o.metadata.name || "unnamed"),
          payload: o.payload,
          source: o.metadata.source || (typeof o.payload === "string" ? o.payload : null),
          distance: d,
          address: o.address,
        }));
        if (rerank) hits = rerankByOverlap(stmt.query, hits, rerankBoost);
        hits = hits.slice(0, stmt.k);

        lastResult = {
          kind: "find",
          query: stmt.query,
          items: hits,
        };
        results.push(lastResult);
        trace.push(`memory_find query="${stmt.query}" -> ${hits.length} hits`);
        break;
      }

      case "memory_list": {
        const objs = kernel.allObjects().map((o) => ({
          name: String(o.metadata.name || "unnamed"),
          address: o.address,
          coord: o.coord,
          tier: o.tier,
          source: o.metadata.source || null,
        }));
        lastResult = { kind: "list_objects", items: objs };
        results.push(lastResult);
        trace.push(`memory_list ${objs.length} objects`);
        break;
      }

      case "memory_dump": {
        const objs = kernel.allObjects();
        const matched = objs.find((o) => o.metadata && o.metadata.name === stmt.name);
        lastResult = {
          kind: "dump",
          name: stmt.name,
          object: matched
            ? {
                name: stmt.name,
                address: matched.address,
                coord: matched.coord,
                tier: matched.tier,
                payload: matched.payload,
                metadata: matched.metadata,
              }
            : null,
        };
        results.push(lastResult);
        trace.push(`memory_dump name=${stmt.name}`);
        break;
      }

      case "demon_sort": {
        const items = kernel.allObjects().map((o) => [o.coord, o]);
        const sorted = kernel.categoricalSort(items);
        lastResult = {
          kind: "sorted_objects",
          items: sorted.map(([, o]) => ({
            name: String(o.metadata.name || "unnamed"),
            address: o.address,
            coord: o.coord,
            tier: o.tier,
          })),
        };
        results.push(lastResult);
        trace.push(`demon_sort ${sorted.length}`);
        break;
      }

      case "controller_verify": {
        const stats = kernel.stats();
        lastResult = {
          kind: "verify",
          samples: stats.tem,
          message: `triple-equivalence: ${stats.tem} samples, ${stats.pveRej} rejections`,
        };
        results.push(lastResult);
        trace.push(`controller_verify samples=${stats.tem}`);
        break;
      }

      case "kernel_stats": {
        const stats = kernel.stats();
        lastResult = { kind: "stats", stats };
        results.push(lastResult);
        trace.push(`kernel_stats objects=${stats.objects}`);
        break;
      }

      case "kernel_trace": {
        const log = kernel.events.slice();
        lastResult = { kind: "trace", log };
        results.push(lastResult);
        trace.push(`kernel_trace ${log.length} entries`);
        break;
      }

      case "process_list": {
        const procs = [];
        for (const [name, p] of processes.entries()) {
          procs.push({ name, state: p.state, target: p.targetCoord });
        }
        lastResult = { kind: "processes", items: procs };
        results.push(lastResult);
        trace.push(`process_list ${procs.length}`);
        break;
      }

      default:
        throw new Error(`unimplemented vaHera op: ${stmt.op}`);
    }
  }

  return {
    results,
    trace,
    lastResult,
    aspects,
    targets,
  };
}
