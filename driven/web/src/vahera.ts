// vahera.ts — minimal vaHera parser + evaluator for the web demo.

import { SCoord, embedText, embedMolecule } from "./substrate";
import type { Kernel } from "./kernel";

export type Stmt =
  | { op: "describe"; target: string; text: string }
  | { op: "resolve"; target: string }
  | { op: "spawn"; program: string; target: string }
  | { op: "navigate" }
  | { op: "complete" }
  | { op: "memory_store"; name: string; text: string }
  | { op: "memory_find"; query: string; k: number }
  | { op: "demon_sort" };

export interface ExecResult {
  kind: "text" | "scalar" | "list" | "molecule" | "chart";
  title?: string;
  value?: string | number;
  lines?: string[];
  items?: Array<{ name: string; payload: unknown; distance: number }>;
  compound?: { name: string; formula?: string; payload: Record<string, unknown> };
  coord?: SCoord;
  chartData?: number[];
}

export function parseVahera(src: string): Stmt[] {
  const out: Stmt[] = [];
  for (const raw of src.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;

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
      out.push({ op: "memory_find", query: m[1], k: m[2] ? parseInt(m[2]) : 3 });
    } else if (line === "demon sort") {
      out.push({ op: "demon_sort" });
    } else {
      throw new Error(`unknown vaHera: ${line}`);
    }
  }
  return out;
}

function resolveCoord(
  name: string,
  text: string,
  molData: Record<string, Record<string, unknown>>
): SCoord {
  const nameLc = name.toLowerCase();
  const textLc = text.toLowerCase();

  if (molData[nameLc]) {
    return embedMolecule(nameLc, molData[nameLc] as Record<string, number | string>);
  }
  const tokens = new Set([
    ...nameLc.split(/[\s_-]+/),
    ...textLc.split(/\s+/),
  ]);
  for (const m of Object.keys(molData)) {
    if (tokens.has(m.toLowerCase())) {
      return embedMolecule(m, molData[m] as Record<string, number | string>);
    }
  }
  return embedText(text);
}

export function executeVahera(
  source: string,
  kernel: Kernel,
  molData: Record<string, Record<string, unknown>>
): ExecResult | null {
  const program = parseVahera(source);
  const targets = new Map<string, SCoord>();
  let result: ExecResult | null = null;
  let processTarget: SCoord | null = null;

  for (const stmt of program) {
    if (stmt.op === "describe") {
      const coord = resolveCoord(stmt.target, stmt.text, molData);
      targets.set(stmt.target, coord);
    } else if (stmt.op === "resolve") {
      if (!targets.has(stmt.target)) {
        targets.set(stmt.target, resolveCoord(stmt.target, stmt.target, molData));
      }
    } else if (stmt.op === "spawn") {
      const tgt = targets.get(stmt.target);
      if (!tgt) throw new Error(`spawn: unresolved ${stmt.target}`);
      processTarget = tgt;
    } else if (stmt.op === "navigate" || stmt.op === "complete") {
      if (processTarget) {
        if (stmt.op === "complete") {
          // apply trajectory; find nearest stored compound
          kernel.runTrajectory(processTarget);
          const near = kernel.surgicalRetrieve(processTarget, 1);
          if (near.length > 0) {
            const [obj, dist] = near[0];
            const name = String(obj.metadata.name ?? "unknown");
            const payload = obj.payload as Record<string, unknown>;
            result = {
              kind: "molecule",
              compound: {
                name,
                formula: String(payload.formula ?? ""),
                payload,
              },
              coord: obj.coord,
            };
            void dist;
          } else {
            result = {
              kind: "text",
              lines: ["no categorical match at this coordinate."],
            };
          }
        }
      }
    } else if (stmt.op === "memory_store") {
      const coord = embedText(stmt.text);
      kernel.allocate(coord, stmt.text, { name: stmt.name, kind: "note" });
      result = {
        kind: "text",
        lines: [`stored at coord (${coord.k.toFixed(3)}, ${coord.t.toFixed(3)}, ${coord.e.toFixed(3)})`],
      };
    } else if (stmt.op === "memory_find") {
      const qCoord = embedText(stmt.query);
      const hits = kernel.surgicalRetrieve(qCoord, stmt.k);
      result = {
        kind: "list",
        title: `nearest ${hits.length}`,
        items: hits.map(([o, d]) => ({
          name: String(o.metadata.name ?? "unnamed"),
          payload: o.payload,
          distance: d,
        })),
      };
    } else if (stmt.op === "demon_sort") {
      const items: Array<[SCoord, unknown]> = kernel.allObjects().map(o => [o.coord, o.metadata.name ?? "unnamed"]);
      const sorted = kernel.categoricalSort(items);
      result = {
        kind: "list",
        title: "sorted",
        items: sorted.map(([c, name]) => ({
          name: String(name),
          payload: null,
          distance: Math.sqrt(c.k * c.k + c.t * c.t + c.e * c.e),
        })),
      };
    }
  }

  return result;
}
