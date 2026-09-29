/**
 * st-Hurbert compiler — interpreter. Evaluates a parsed program against the analysed
 * federation and returns a structured result the dashboard renders. Execution is a
 * small state machine: `navigate` sets the active scope, `slice`/`find` filter the
 * working symbol set, `show` emits a result block.
 */

import type { Character, Symbol } from "../chi";
import { latest, regimeLines, regimeSummary, versionLine } from "../lineage";
import { AnalysedRepo, Federation, findRepo, repoName } from "../model";
import { Cond, Field, Op, Program, Stmt } from "./ast";

export interface ResultBlock {
  /** The statement that produced this block, echoed for the dashboard. */
  label: string;
  kind:
    | "sense" | "chi" | "salient" | "fragments" | "files" | "symbols"
    | "lineage" | "regime" | "health"
    | "slice" | "find" | "scope";
  /** Human-readable lines. */
  lines: string[];
  /** Optional structured payload for richer rendering. */
  data?: unknown;
}

export interface RunResult {
  blocks: ResultBlock[];
}

export class RuntimeError extends Error {
  constructor(message: string) {
    super(`runtime error: ${message}`);
    this.name = "RuntimeError";
  }
}

interface Scope {
  /** Repos in scope ("*" → all). */
  repos: AnalysedRepo[];
  /** Working symbol set (after slices/finds); starts as all in-scope symbols. */
  working: { repo: AnalysedRepo; sym: Symbol }[];
  label: string;
}

export function interpret(program: Program, fed: Federation): RunResult {
  if (fed.repos.length === 0) throw new RuntimeError("no repos analysed yet");
  const blocks: ResultBlock[] = [];

  // default scope = whole federation
  let scope: Scope = scopeAll(fed);

  for (const stmt of program) {
    switch (stmt.kind) {
      case "navigate": {
        scope = navigate(fed, stmt.target);
        blocks.push({
          label: `navigate ${stmt.target}`,
          kind: "scope",
          lines: [`scope: ${scope.label} (${scope.repos.length} repo(s), ${scope.working.length} symbols)`],
        });
        break;
      }
      case "slice": {
        scope = slice(scope, stmt.symbolKind, stmt.cond);
        blocks.push({
          label: sliceLabel(stmt),
          kind: "slice",
          lines: [`${scope.working.length} symbol(s) after slice`],
          data: sample(scope),
        });
        break;
      }
      case "find": {
        const s = find(scope, stmt.text);
        blocks.push({
          label: `find "${stmt.text}"`,
          kind: "find",
          lines:
            s.working.length === 0
              ? [`no symbols matching "${stmt.text}"`]
              : s.working.slice(0, 30).map((w) => `${repoName(w.repo)}  ${w.sym.file}:${w.sym.line}  [${w.sym.kind}] ${w.sym.name}`),
          data: sample(s),
        });
        scope = s;
        break;
      }
      case "show": {
        blocks.push(show(scope, stmt.target));
        break;
      }
    }
  }

  return { blocks };
}

// ── scope construction ────────────────────────────────────────────────────────

function scopeAll(fed: Federation): Scope {
  return {
    repos: fed.repos,
    working: allSymbols(fed.repos),
    label: "* (federation)",
  };
}

function navigate(fed: Federation, target: string): Scope {
  if (target === "*") return scopeAll(fed);
  const repo = findRepo(fed, target);
  if (!repo) {
    const names = fed.repos.map(repoName).join(", ");
    throw new RuntimeError(`no repo ${JSON.stringify(target)} in scope (have: ${names})`);
  }
  return {
    repos: [repo],
    working: repo.symbols.map((sym) => ({ repo, sym })),
    label: repoName(repo),
  };
}

function allSymbols(repos: AnalysedRepo[]) {
  return repos.flatMap((repo) => repo.symbols.map((sym) => ({ repo, sym })));
}

// ── slice / find ────────────────────────────────────────────────────────────

function slice(scope: Scope, kind: string | null, cond: Cond | null): Scope {
  let working = scope.working;
  if (kind) working = working.filter((w) => w.sym.kind === kind);
  if (cond) working = working.filter((w) => evalCond(cond, w.sym));
  return { ...scope, working };
}

function find(scope: Scope, text: string): Scope {
  const needle = text.toLowerCase();
  const working = scope.working.filter(
    (w) =>
      w.sym.name.toLowerCase().includes(needle) ||
      w.sym.snippet.toLowerCase().includes(needle)
  );
  return { ...scope, working };
}

function fieldValue(field: Field, sym: Symbol): string | number {
  switch (field) {
    case "name": return sym.name;
    case "kind": return sym.kind;
    case "file": return sym.file;
    case "line": return sym.line;
  }
}

function evalCond(cond: Cond, sym: Symbol): boolean {
  switch (cond.kind) {
    case "and": return evalCond(cond.left, sym) && evalCond(cond.right, sym);
    case "or": return evalCond(cond.left, sym) || evalCond(cond.right, sym);
    case "cmp": return evalCmp(cond.field, cond.op, cond.value, sym);
  }
}

function evalCmp(field: Field, op: Op, value: string | number, sym: Symbol): boolean {
  const fv = fieldValue(field, sym);
  switch (op) {
    case "==": return String(fv) === String(value);
    case "!=": return String(fv) !== String(value);
    case "~":
    case "contains": return String(fv).toLowerCase().includes(String(value).toLowerCase());
    case ">": return Number(fv) > Number(value);
    case "<": return Number(fv) < Number(value);
    case ">=": return Number(fv) >= Number(value);
    case "<=": return Number(fv) <= Number(value);
  }
}

// ── show ──────────────────────────────────────────────────────────────────────

function show(scope: Scope, target: ResultBlock["kind"] & string): ResultBlock {
  const label = `show ${target}`;
  // Federation-wide vs single-repo aware.
  switch (target) {
    case "chi":
      return {
        label,
        kind: "chi",
        lines: scope.repos.map((r) => `${repoName(r)}: χ = ${r.character.chi.toFixed(3)}`),
        data: scope.repos.map((r) => ({ repo: repoName(r), chi: r.character.chi })),
      };
    case "sense":
      return {
        label,
        kind: "sense",
        lines: scope.repos.flatMap((r) => senseLines(r.character, repoName(r))),
      };
    case "salient":
      return {
        label,
        kind: "salient",
        lines: scope.repos.flatMap((r) => [
          `${repoName(r)}:`,
          ...r.character.salient.slice(0, 6).map((s) => `  · ${s.file}  (weight ${s.weight.toFixed(1)})`),
        ]),
      };
    case "fragments":
      return {
        label,
        kind: "fragments",
        lines: scope.repos.map(
          (r) => `${repoName(r)}: ${r.character.fragments} fragment(s), core ${r.character.coreBlocks}/${r.character.blocks} blocks`
        ),
      };
    case "files": {
      const files = Array.from(new Set(scope.working.map((w) => w.sym.file))).sort();
      return { label, kind: "files", lines: files.slice(0, 200), data: files };
    }
    case "symbols":
      return {
        label,
        kind: "symbols",
        lines: scope.working
          .slice(0, 100)
          .map((w) => `${w.sym.file}:${w.sym.line}  [${w.sym.kind}] ${w.sym.name}`),
        data: sample(scope),
      };
    case "lineage":
      return {
        label,
        kind: "lineage",
        lines: scope.repos.flatMap((r) => lineageLines(r)),
        data: scope.repos.map((r) => ({ repo: repoName(r), lineage: r.lineage ?? null })),
      };
    case "regime":
      return {
        label,
        kind: "regime",
        lines: scope.repos.flatMap((r) => regimeForRepo(r)),
        data: scope.repos.map((r) => ({ repo: repoName(r), latest: latest(r.lineage) ?? null })),
      };
    case "health":
      return {
        label,
        kind: "health",
        lines: scope.repos.flatMap((r) => healthLines(r)),
        data: scope.repos.map((r) => ({
          repo: repoName(r),
          chi: r.character.chi,
          latest: latest(r.lineage) ?? null,
        })),
      };
    default:
      throw new RuntimeError(`cannot show ${target}`);
  }
}

function senseLines(c: Character, name: string): string[] {
  return [
    `${name}: χ = ${c.chi.toFixed(3)} (core ${c.coreBlocks}/${c.blocks} blocks; ${c.fragments} fragment(s))`,
    ...c.salient.slice(0, 5).map((s) => `  · ${s.file}  (weight ${s.weight.toFixed(1)})`),
  ];
}

// ── image-knowability renderers ─────────────────────────────────────────────
// These read a repo's recorded lineage. When none is present they say so
// plainly — an unbuilt/unexercised repo has an *unknown* runtime health, and the
// tool's whole point is to make that gap visible rather than to paper over it.

function lineageLines(r: AnalysedRepo): string[] {
  const name = repoName(r);
  const lin = r.lineage;
  if (!lin || lin.versions.length === 0) {
    return [`${name}: no image lineage yet — nothing built/exercised/judged`];
  }
  return [
    `${name}: ${lin.versions.length} version(s)`,
    ...lin.versions.map((v) => `  ${versionLine(v)}`),
  ];
}

function regimeForRepo(r: AnalysedRepo): string[] {
  const name = repoName(r);
  const v = latest(r.lineage);
  if (!v) {
    return [`${name}: no image version — runtime health is unknown (build one to make it knowable)`];
  }
  return [`${name} (latest):`, ...regimeLines(v).map((l) => `  ${l}`)];
}

/**
 * The two parallel lineages side by side: sense-over-time (χ, always known) and
 * runtime-health-over-time (regime map, known only once judged).
 */
function healthLines(r: AnalysedRepo): string[] {
  const name = repoName(r);
  const c = r.character;
  const v = latest(r.lineage);
  const sense = `sense: χ = ${c.chi.toFixed(3)} (${c.fragments} fragment(s))`;
  const health = v
    ? `health: v${v.version} build:${v.build} — ${
        v.knowledge === "judged" && v.regime ? regimeSummary(v.regime) : v.knowledge
      }`
    : `health: unknown (no image version)`;
  return [`${name}:`, `  ${sense}`, `  ${health}`];
}

function sliceLabel(stmt: Extract<Stmt, { kind: "slice" }>): string {
  const k = stmt.symbolKind ? ` ${stmt.symbolKind}` : "";
  const w = stmt.cond ? " where …" : "";
  return `slice${k}${w}`;
}

function sample(scope: Scope) {
  return scope.working.slice(0, 50).map((w) => ({
    repo: repoName(w.repo),
    file: w.sym.file,
    line: w.sym.line,
    kind: w.sym.kind,
    name: w.sym.name,
  }));
}
