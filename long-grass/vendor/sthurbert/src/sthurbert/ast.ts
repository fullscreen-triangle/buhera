/** st-Hurbert AST node types (repo-query subset). */

export type Program = Stmt[];

export type Stmt =
  | { kind: "navigate"; target: string } // repo name or "*"
  | { kind: "slice"; symbolKind: string | null; cond: Cond | null }
  | { kind: "show"; target: ShowTarget }
  | { kind: "find"; text: string };

export type ShowTarget =
  | "sense" | "chi" | "salient" | "fragments" | "files" | "symbols"
  // image-knowability lineage (runtime-health-over-time, beside χ's sense-over-time)
  | "lineage" | "regime" | "health";

export type Cond =
  | { kind: "cmp"; field: Field; op: Op; value: string | number }
  | { kind: "and"; left: Cond; right: Cond }
  | { kind: "or"; left: Cond; right: Cond };

export type Field = "name" | "kind" | "file" | "line";
export type Op = "==" | "!=" | "~" | "contains" | ">" | "<" | ">=" | "<=";

export const SHOW_TARGETS: ShowTarget[] = [
  "sense", "chi", "salient", "fragments", "files", "symbols",
  "lineage", "regime", "health",
];
export const FIELDS: Field[] = ["name", "kind", "file", "line"];
