/* ============================================================================
 * sthurbert — st-Hurbert, the repo-query language (specification
 * specs/sthurbert.md). Wraps bloodhound thrust's repo-lens compiler (lexer,
 * parser, interpreter) and its character computation (chi.ts), vendored.
 *
 * The language navigates and slices an ANALYSED federation: repositories
 * with their symbol index and their character χ. The module holds that
 * federation as state (R6): `load` adds repositories from symbol indexes
 * (the `{name, kind, file, line, snippet}` rows a `.purpose/index.json`
 * holds), and each repository's character is computed by the engine's own
 * `computeCharacter`, never here. Queries then run against it.
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";
import type { DslEntry, Validation } from "../dsl.ts";

export interface SthSymbol {
  name: string;
  kind: string;
  file: string;
  line: number;
  snippet: string;
}

interface SthError {
  stage: string;
  message: string;
  line?: number;
  col?: number;
}

/** The subset of the vendored st-Hurbert engine this adapter uses. */
export interface SthurbertEngine {
  compile(source: string): { ok: boolean; error?: SthError };
  run(source: string, fed: { repos: unknown[] }): { ok: boolean; result?: { blocks: unknown[] }; error?: SthError };
  computeCharacter(symbols: SthSymbol[]): unknown;
}

export const STHURBERT_ID = "sthurbert";

/** A tiny, openly synthetic repository for the demo. */
const DEMO_REPO = {
  name: "demo",
  symbols: [
    { name: "alpha", kind: "fn", file: "src/lib.rs", line: 1, snippet: "fn alpha() -> u32" },
    { name: "Beta", kind: "struct", file: "src/lib.rs", line: 2, snippet: "struct Beta" },
    { name: "gamma", kind: "fn", file: "src/lib.rs", line: 3, snippet: "fn gamma(b: Beta)" },
    { name: "Delta", kind: "enum", file: "src/model.rs", line: 1, snippet: "enum Delta" },
    { name: "epsilon", kind: "fn", file: "src/model.rs", line: 4, snippet: "fn epsilon(d: Delta)" },
  ],
};
const DEMO_QUERY = "navigate demo ; show chi\nslice fn where name ~ \"a\" ; show symbols";

function toError(e: SthError | undefined): { message: string; line?: number; column?: number } {
  const message = e ? `${e.stage} error: ${e.message}` : "rejected";
  const out: { message: string; line?: number; column?: number } = { message };
  if (typeof e?.line === "number") out.line = e.line;
  if (typeof e?.col === "number") out.column = e.col;
  return out;
}

export function sthurbertValidate(engine: SthurbertEngine) {
  return (source: string): Validation => {
    const c = engine.compile(source);
    return c.ok ? { ok: true, errors: [] } : { ok: false, errors: [toError(c.error)] };
  };
}

export function sthurbertDsl(engine: SthurbertEngine): DslEntry {
  return { id: "sthurbert", label: "st-Hurbert", extension: ".sth", moduleId: STHURBERT_ID, packId: "sthurbert", validate: sthurbertValidate(engine) };
}

function isSymbol(x: Json): boolean {
  return !!x && typeof x === "object" && !Array.isArray(x) && typeof x["name"] === "string" && typeof x["kind"] === "string" && typeof x["file"] === "string" && typeof x["line"] === "number";
}

export function makeSthurbertModule(engine: SthurbertEngine): Module {
  const repos = new Map<string, unknown>();
  const expected =
    'st-Hurbert source, "demo", { kind: "query" | "compile", source }, { kind: "load", repos: [{ name, path?, symbols }] }, or "reset"';

  function load(name: string, path: string, symbols: SthSymbol[]) {
    repos.set(name, {
      origin: { kind: "local", path, name },
      // The interpreter never reads the snapshot; the repo is known by its index.
      snapshot: {},
      symbols,
      character: engine.computeCharacter(symbols),
    });
  }

  function query(source: string): ActResult {
    let out;
    try {
      out = engine.run(source, { repos: [...repos.values()] });
    } catch (err) {
      return fail([`sthurbert: ${errorText(err)}`], errorText(err));
    }
    if (!out.ok) {
      const e = toError(out.error);
      return { ok: false, output_delta: { kind: "repo_query", ok: false, blocks: [], error: e }, residue: 1, completed: true, error: e.message };
    }
    const blocks = JSON.parse(JSON.stringify(out.result?.blocks ?? []));
    return done({ kind: "repo_query", ok: true, summary: `st-Hurbert: ${blocks.length} block(s) over ${repos.size} repo(s)`, blocks }, 0);
  }

  return {
    id: STHURBERT_ID,
    describe: () => ({
      id: STHURBERT_ID,
      description:
        "st-Hurbert — navigate, slice and query an analysed federation of repositories: symbols, files, character χ " +
        "(the minimum cut of the symbol co-occurrence graph), salient files, fragments, lineage, health. " +
        "Load repositories from symbol indexes, then query.",
      instructions: [
        'dispatch("sthurbert", "demo")',
        'dispatch("sthurbert", { kind: "load", repos: [{ name: "myrepo", symbols: [...] }] })',
        'dispatch("sthurbert", "navigate myrepo ; slice fn where name contains parse ; show symbols")',
        'dispatch("sthurbert", { kind: "compile", source })',
      ],
      dsl: "sthurbert",
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      if (instruction === "demo") {
        load(DEMO_REPO.name, `/demo/${DEMO_REPO.name}`, DEMO_REPO.symbols);
        return query(DEMO_QUERY);
      }
      if (instruction === "reset") {
        repos.clear();
        return done({ kind: "repo_query", ok: true, summary: "st-Hurbert: federation cleared", blocks: [] }, 0);
      }
      if (typeof instruction === "string") return query(instruction);
      if (!instruction || typeof instruction !== "object" || Array.isArray(instruction)) return invalid(STHURBERT_ID, expected);
      const kind = instruction["kind"];
      const source = instruction["source"];
      if (kind === "query" && typeof source === "string") return query(source);
      if (kind === "compile" && typeof source === "string") {
        const v = sthurbertValidate(engine)(source);
        return v.ok
          ? done({ kind: "repo_query", ok: true, summary: "st-Hurbert: program compiles", blocks: [] }, 0)
          : { ok: false, output_delta: { kind: "repo_query", ok: false, blocks: [], error: v.errors[0] }, residue: 1, completed: true, error: v.errors[0]!.message };
      }
      if (kind === "load" && Array.isArray(instruction["repos"])) {
        const loaded: string[] = [];
        for (const r of instruction["repos"]) {
          if (!r || typeof r !== "object" || Array.isArray(r) || typeof r["name"] !== "string" || !Array.isArray(r["symbols"]) || !r["symbols"].every(isSymbol)) {
            return invalid(STHURBERT_ID, "{ kind: \"load\", repos: [{ name, path?, symbols: [{ name, kind, file, line, snippet }] }] }");
          }
          const symbols = r["symbols"].map((s) => {
            const o = s as Record<string, Json>;
            return { name: String(o["name"]), kind: String(o["kind"]), file: String(o["file"]), line: Number(o["line"]), snippet: typeof o["snippet"] === "string" ? o["snippet"] : "" };
          });
          load(r["name"], typeof r["path"] === "string" ? r["path"] : `/${r["name"]}`, symbols);
          loaded.push(r["name"]);
        }
        return done({ kind: "repo_query", ok: true, summary: `st-Hurbert: loaded ${loaded.join(", ")} (${repos.size} repo(s) in federation)`, blocks: [] }, 0);
      }
      return invalid(STHURBERT_ID, expected);
    },

    outputCell: () => ({ kind: "repo_query_cell" }),
  };
}
