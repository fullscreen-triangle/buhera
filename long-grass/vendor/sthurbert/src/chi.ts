/**
 * The character invariant χ — a repo's conserved sense/goal — computed in the
 * browser in pure TypeScript. This is a faithful port of the Rust tracker's
 * `chi.rs`; it is deliberately NOT linked to the Rust tool via WASM.
 *
 * Pipeline: fetched source files → symbols (regex line-scan, standing in for
 * `purpose index`) → file/section block graph (containment + reference proximity)
 * → χ = Stoer–Wagner global min-cut of the largest connected component.
 *
 * χ is positive (floor β), conserved under relabelling, and non-local (a region).
 * Because a repo's raw graph is naturally disconnected (docs vs code), χ is the
 * invariant of the *largest connected component* and the fragment count is reported.
 */

import type { SourceFile } from "./github";

/** The weight floor β > 0 (T0). */
export const BETA = 1.0;

export interface Symbol {
  name: string;
  kind: string;
  file: string;
  line: number;
  snippet: string;
}

export interface Character {
  /** χ — min cut-residual of the largest connected component. */
  chi: number;
  /** Total file/section blocks across the repo. */
  blocks: number;
  /** Connected-component count (fragmentation). */
  fragments: number;
  /** Blocks in the largest component χ was computed over. */
  coreBlocks: number;
  /** The block set most cheaply severed within the core (a region). */
  cutSide: string[];
  /** Highest-degree blocks — the sense surface. */
  salient: { file: string; weight: number }[];
}

// ── symbol extraction (browser stand-in for `purpose index`) ──────────────────

const MODIFIERS = "(?:pub |async |unsafe |export |default |public |private |protected |static |final |abstract |const )*";
const CODE_PATTERNS: { kind: string; re: RegExp }[] = [
  { kind: "fn", re: new RegExp(`^\\s*${MODIFIERS}fn\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "func", re: new RegExp(`^\\s*${MODIFIERS}func(?:tion)?\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "def", re: new RegExp(`^\\s*${MODIFIERS}def\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "struct", re: new RegExp(`^\\s*${MODIFIERS}struct\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "enum", re: new RegExp(`^\\s*${MODIFIERS}enum\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "trait", re: new RegExp(`^\\s*${MODIFIERS}trait\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "type", re: new RegExp(`^\\s*${MODIFIERS}type\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
  { kind: "class", re: new RegExp(`^\\s*${MODIFIERS}class\\s+([A-Za-z_][A-Za-z0-9_]*)`) },
];
const HEADING_MD = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/;
const HEADING_TEX = /^\s*\\(sub)*section\*?\{([^}]+)\}/;

function isProse(ext: string): boolean {
  return ext === "md" || ext === "tex";
}

/** Extract definitions (code) or headings (prose) from one file. */
export function extractSymbols(file: SourceFile): Symbol[] {
  const out: Symbol[] = [];
  const lines = file.text.split(/\r?\n/);
  const prose = isProse(file.ext);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length > 400) continue;
    if (prose) {
      const md = line.match(HEADING_MD);
      if (md) {
        out.push({ name: md[2].trim(), kind: "heading", file: file.path, line: i + 1, snippet: line.trim() });
        continue;
      }
      const tex = line.match(HEADING_TEX);
      if (tex) {
        out.push({ name: tex[2].trim(), kind: "section", file: file.path, line: i + 1, snippet: line.trim() });
      }
    } else {
      for (const { kind, re } of CODE_PATTERNS) {
        const m = line.match(re);
        if (m) {
          out.push({ name: m[1], kind, file: file.path, line: i + 1, snippet: line.trim() });
          break; // one symbol per line
        }
      }
    }
  }
  return out;
}

export function extractAllSymbols(files: SourceFile[]): Symbol[] {
  return files.flatMap(extractSymbols);
}

// ── block graph ───────────────────────────────────────────────────────────────

class BlockGraph {
  labels: string[];
  w: number[][];

  constructor(labels: string[], w: number[][]) {
    this.labels = labels;
    this.w = w;
  }

  /** Connected components as index lists, largest first. */
  components(): number[][] {
    const n = this.labels.length;
    const seen = new Array<boolean>(n).fill(false);
    const comps: number[][] = [];
    for (let start = 0; start < n; start++) {
      if (seen[start]) continue;
      const stack = [start];
      seen[start] = true;
      const comp: number[] = [];
      while (stack.length) {
        const v = stack.pop()!;
        comp.push(v);
        for (let u = 0; u < n; u++) {
          if (!seen[u] && this.w[v][u] > 0) {
            seen[u] = true;
            stack.push(u);
          }
        }
      }
      comps.push(comp);
    }
    comps.sort((a, b) => b.length - a.length);
    return comps;
  }

  largestComponentSubgraph(): BlockGraph {
    const comps = this.components();
    const core = comps[0] ?? [];
    const labels = core.map((i) => this.labels[i]);
    const k = core.length;
    const w = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    for (let ni = 0; ni < k; ni++) {
      for (let nj = 0; nj < k; nj++) {
        w[ni][nj] = this.w[core[ni]][core[nj]];
      }
    }
    return new BlockGraph(labels, w);
  }

  topByDegree(k: number): { file: string; weight: number }[] {
    const ds = this.labels.map((file, i) => ({
      file,
      weight: this.w[i].reduce((s, x) => s + x, 0),
    }));
    ds.sort((a, b) => b.weight - a.weight || (a.file < b.file ? -1 : 1)); // deterministic tie-break
    return ds.slice(0, k);
  }

  /** Stoer–Wagner global minimum cut. Deterministic, O(V³). Returns [cut, side]. */
  stoerWagnerMinCut(): [number, number[]] {
    const n = this.labels.length;
    const w = this.w.map((row) => row.slice());
    const groups: number[][] = Array.from({ length: n }, (_, i) => [i]);
    let alive = Array.from({ length: n }, (_, i) => i);
    let bestCut = Infinity;
    let bestSide: number[] = [];

    while (alive.length > 1) {
      const m = alive.length;
      const added = new Array<boolean>(m).fill(false);
      const weights = new Array<number>(m).fill(0);
      const order: number[] = [];
      for (let step = 0; step < m; step++) {
        let sel = -1;
        let best = -Infinity;
        for (let i = 0; i < m; i++) {
          if (!added[i] && weights[i] > best) {
            best = weights[i];
            sel = i;
          }
        }
        added[sel] = true;
        order.push(sel);
        for (let i = 0; i < m; i++) {
          if (!added[i]) weights[i] += w[alive[sel]][alive[i]];
        }
      }
      const tLocal = order[m - 1];
      const sLocal = order[m - 2];
      const cutOfPhase = weights[tLocal];
      const t = alive[tLocal];
      const s = alive[sLocal];
      if (cutOfPhase < bestCut) {
        bestCut = cutOfPhase;
        bestSide = groups[t].slice();
      }
      // merge t into s
      groups[s] = groups[s].concat(groups[t]);
      groups[t] = [];
      for (const x of alive) {
        if (x !== s && x !== t) {
          w[s][x] += w[t][x];
          w[x][s] += w[x][t];
        }
      }
      alive = alive.filter((x) => x !== t);
    }

    // report the smaller side (stable, non-local witness)
    let side = bestSide;
    if (side.length * 2 > n) {
      const inSide = new Set(side);
      side = Array.from({ length: n }, (_, i) => i).filter((i) => !inSide.has(i));
    }
    side.sort((a, b) => a - b);
    return [bestCut, side];
  }

  character(): Character {
    const n = this.labels.length;
    if (n < 2) {
      return {
        chi: BETA,
        blocks: n,
        fragments: 1,
        coreBlocks: n,
        cutSide: this.labels.slice(),
        salient: this.topByDegree(8),
      };
    }
    const [chi, side] = this.stoerWagnerMinCut();
    return {
      chi,
      blocks: n,
      fragments: 1,
      coreBlocks: n,
      cutSide: side.map((i) => this.labels[i]),
      salient: this.topByDegree(8),
    };
  }
}

function addEdge(w: number[][], i: number, j: number, delta: number) {
  w[i][j] += delta;
  w[j][i] += delta;
}

/** Build the file-block graph: containment (shared path depth) + reference proximity. */
function buildGraph(symbols: Symbol[]): BlockGraph {
  const labelOf = new Map<string, number>();
  const labels: string[] = [];
  const namesIn: string[][] = [];
  const snippetsIn: string[] = [];

  for (const s of symbols) {
    let id = labelOf.get(s.file);
    if (id === undefined) {
      id = labels.length;
      labelOf.set(s.file, id);
      labels.push(s.file);
      namesIn.push([]);
      snippetsIn.push("");
    }
    namesIn[id].push(s.name);
    snippetsIn[id] += s.snippet.toLowerCase() + "\n";
  }

  const n = labels.length;
  const w = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const comps = labels.map((f) => f.split(/[/\\]/));

  // containment
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let shared = 0;
      const a = comps[i], b = comps[j];
      const lim = Math.min(a.length, b.length);
      while (shared < lim && a[shared] === b[shared]) shared++;
      if (shared > 0) addEdge(w, i, j, BETA * shared);
    }
  }

  // reference proximity (distinctive names, len ≥ 4)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      let refs = 0;
      for (const name of namesIn[i]) {
        if (name.length >= 4 && snippetsIn[j].includes(name.toLowerCase())) refs++;
      }
      if (refs > 0) addEdge(w, i, j, BETA * refs);
    }
  }

  return new BlockGraph(labels, w);
}

/** Compute χ and its salient surface from a repo's extracted symbols. */
export function computeCharacter(symbols: Symbol[]): Character {
  const graph = buildGraph(symbols);
  const fragments = graph.components().length;
  const salient = graph.topByDegree(8);
  const core = graph.largestComponentSubgraph();
  const character = core.character();
  character.blocks = graph.labels.length;
  character.fragments = fragments;
  character.coreBlocks = core.labels.length;
  character.salient = salient;
  return character;
}
