// Everything the site shows is read from the specification tree at build time:
// the Markdown documents and the two registry JSON files. Nothing here is
// hand-copied, so the site cannot drift from what the tests check.
import catalogueJson from "../../registry/catalogue.json";
import vendorJson from "../../registry/vendor.json";

export type Binding = "native" | "remote" | "bridge" | "none";
export type Host = "rust" | "ts";

export interface Upstream {
  repo: string;
  path: string;
  language: string;
  commit: string;
  vendored_at?: string;
}

export interface ModuleRow {
  id: string;
  name: string;
  layer: string;
  summary: string;
  dsl: string | null;
  upstream: Upstream[];
  bindings: Record<Host, Binding>;
  output_kinds: string[];
  residue: string;
  side_effects: string[];
  spec: string;
}

export interface DslRow {
  id: string;
  label: string;
  extension: string;
  module_id: string;
  pack_id: string;
  validators: Host[];
}

export interface Catalogue {
  schema: string;
  layers: Record<string, string>;
  modules: ModuleRow[];
  dsls: DslRow[];
}

export interface VendorEntry {
  id: string;
  module: string | null;
  repo: string;
  from: string;
  to: string;
  mode: "tree" | "file" | "build";
  commit: string;
  exclude?: string[];
  local?: string[];
  note?: string;
}

export const catalogue = catalogueJson as Catalogue;
export const vendor = vendorJson as { schema: string; clones: Record<string, string>; entries: VendorEntry[] };

const archRaw = import.meta.glob("../../architecture/*.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
const specRaw = import.meta.glob("../../specs/*.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

export interface Doc {
  slug: string;
  title: string;
  number: string;
  body: string;
}

const titleOf = (md: string) => (/^#\s+(.+)$/m.exec(md)?.[1] ?? "Untitled").trim();

export const architecture: Doc[] = Object.entries(archRaw)
  .map(([p, body]) => {
    const file = p.split("/").pop()!.replace(/\.md$/, "");
    const [number = "", ...rest] = file.split("-");
    return { slug: file, number, title: titleOf(body).replace(/^\d+\s+—\s+/, ""), body, _rest: rest };
  })
  .sort((a, b) => a.slug.localeCompare(b.slug));

export const specs: Record<string, Doc> = Object.fromEntries(
  Object.entries(specRaw).map(([p, body]) => {
    const slug = p.split("/").pop()!.replace(/\.md$/, "");
    return [slug, { slug, number: "", title: titleOf(body), body }];
  }),
);

export const LAYER_ORDER = ["language", "science", "coordination", "observation", "generation"];

export const shortSha = (s: string) => s.slice(0, 7);
export const repoName = (r: string) => r.split("/").pop() ?? r;
