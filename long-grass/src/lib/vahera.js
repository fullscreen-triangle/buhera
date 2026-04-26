// vahera.js — parser + evaluator for vaHera statements (protein edition).

import { embedText, embedProtein } from "./substrate";
import { resolveProtein, PROTEINS } from "./proteins";

export function parseVahera(src) {
  const out = [];
  const aspects = [];
  for (const raw of src.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    // parse aspect comment
    if (line.startsWith("# aspect:")) {
      aspects.push(line.slice("# aspect:".length));
      continue;
    }
    if (line.startsWith("#")) continue;

    let m;
    if ((m = line.match(/^describe\s+(\S+)\s+with\s+"([^"]*)"/))) {
      out.push({ op: "describe", target: m[1], text: m[2] });
    } else if ((m = line.match(/^resolve\s+(\S+)/))) {
      out.push({ op: "resolve", target: m[1] });
    } else if ((m = line.match(/^spawn\s+(\S+)\s+from\s+(\S+)/))) {
      out.push({ op: "spawn", program: m[1], target: m[2] });
    } else if (line === "navigate to penultimate") {
      out.push({ op: "navigate" });
    } else if (line === "complete trajectory") {
      out.push({ op: "complete" });
    } else if ((m = line.match(/^memory\s+store\s+"([^"]*)"\s*=\s*"([^"]*)"/))) {
      out.push({ op: "memory_store", name: m[1], text: m[2] });
    } else if ((m = line.match(/^memory\s+find\s+nearest\s+"([^"]*)"(?:\s+k=(\d+))?/))) {
      out.push({ op: "memory_find", query: m[1], k: m[2] ? parseInt(m[2]) : 5 });
    } else if (line === "demon sort") {
      out.push({ op: "demon_sort" });
    } else {
      throw new Error(`unknown vaHera: ${line}`);
    }
  }
  out._aspects = aspects;
  return out;
}

function resolveCoord(name) {
  const key = resolveProtein(name);
  if (key) return embedProtein(key, PROTEINS[key]);
  return embedText(name);
}

export function executeVahera(source, kernel) {
  const program = parseVahera(source);
  const aspects = program._aspects || [];
  const targets = new Map();
  let result = null;
  let processTarget = null;
  let compareKey = null;
  let primaryKey = null;

  // Figure out aspect + compare target from the aspect list
  let aspect = "full";
  for (const a of aspects) {
    if (a.startsWith("compare:")) {
      aspect = "compare";
      compareKey = a.slice("compare:".length);
    } else {
      aspect = a;
    }
  }

  for (const stmt of program) {
    if (stmt.op === "describe") {
      targets.set(stmt.target, resolveCoord(stmt.target));
    } else if (stmt.op === "resolve") {
      if (!targets.has(stmt.target)) {
        targets.set(stmt.target, resolveCoord(stmt.target));
      }
    } else if (stmt.op === "spawn") {
      const tgt = targets.get(stmt.target);
      if (!tgt) throw new Error(`spawn: unresolved ${stmt.target}`);
      processTarget = tgt;
      primaryKey = stmt.target;
    } else if (stmt.op === "navigate" || stmt.op === "complete") {
      if (processTarget && stmt.op === "complete") {
        kernel.runTrajectory(processTarget);
        const near = kernel.surgicalRetrieve(processTarget, 1);
        if (near.length > 0) {
          const [obj] = near[0];
          const payload = obj.payload;
          const name = String(obj.metadata.name || primaryKey || "unknown");

          if (aspect === "compare" && compareKey) {
            const key2 = resolveProtein(compareKey);
            const p2 = key2 ? PROTEINS[key2] : null;
            if (p2) {
              result = {
                kind: "protein_compare",
                a: { name, payload },
                b: { name: key2, payload: p2 },
              };
            } else {
              result = {
                kind: "protein",
                name,
                payload,
                aspect: "full",
              };
            }
          } else {
            result = { kind: "protein", name, payload, aspect };
          }
        } else {
          result = { kind: "text", lines: ["no categorical match."] };
        }
      }
    } else if (stmt.op === "memory_store") {
      const coord = embedText(stmt.text);
      kernel.allocate(coord, stmt.text, { name: stmt.name, kind: "note" });
      result = { kind: "text", lines: ["stored."] };
    } else if (stmt.op === "memory_find") {
      const qCoord = embedText(stmt.query);
      const hits = kernel.surgicalRetrieve(qCoord, stmt.k);
      result = {
        kind: "list",
        title: `nearest ${hits.length}`,
        items: hits.map(([o, d]) => ({
          name: String(o.metadata.name || "unnamed"),
          payload: o.payload,
          distance: d,
        })),
      };
    }
  }

  return result;
}
