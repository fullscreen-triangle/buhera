/* ============================================================================
 * The catalogue and its conformance check — specification 05-catalogue.md.
 *
 * Twin of buhera-registry::catalogue. Rules C1–C5 are identical; a message
 * produced here reads the same as the Rust one for the same violation.
 * ========================================================================== */

import type { DslRegistry } from "./dsl.ts";
import type { Registry } from "./registry.ts";

export const SCHEMA = "buhera.catalogue/1";

export type Host = "rust" | "ts";
export type Binding = "native" | "remote" | "bridge" | "none";

export interface Upstream {
  repo: string;
  path: string;
  language: "rust" | "ts" | "js";
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
  bindings: { rust: Binding; ts: Binding };
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
  modules: ModuleRow[];
  dsls: DslRow[];
}

export function conformance(cat: Catalogue, host: Host, modules: Registry, dsls: DslRegistry): string[] {
  const v: string[] = [];
  if (cat.schema !== SCHEMA) v.push(`schema: expected ${SCHEMA}, found ${cat.schema}`);
  const H = host === "rust" ? "Rust" : "Ts";
  const descriptors = modules.list();

  for (const row of cat.modules) {
    const want = row.bindings[host];
    const got = descriptors.find((d) => d.id === row.id);
    if (want !== "none" && got) {
      if (got.binding !== want) v.push(`C1 ${row.id}: binding ${got.binding} ≠ catalogue ${want}`);
      if (got.dsl !== undefined && got.dsl !== row.dsl) v.push(`C3 ${row.id}: dsl ${got.dsl} ≠ catalogue ${row.dsl}`);
    } else if (want !== "none") {
      v.push(`C1 ${row.id}: listed for ${H} but not registered`);
    } else if (got) {
      v.push(`C2 ${row.id}: registered but catalogue says none for ${H}`);
    }
  }
  for (const d of descriptors) {
    if (!cat.modules.some((m) => m.id === d.id)) v.push(`C2 ${d.id}: registered but not in the catalogue`);
  }
  for (const row of cat.dsls) {
    const listed = row.validators.includes(host);
    const e = dsls.get(row.id);
    if (listed && e) {
      if (e.moduleId !== row.module_id || e.packId !== row.pack_id) {
        v.push(`C4 ${row.id}: routes to ${e.moduleId}/${e.packId} ≠ catalogue ${row.module_id}/${row.pack_id}`);
      }
    } else if (listed) {
      v.push(`C4 ${row.id}: validator listed for ${H} but not registered`);
    } else if (e) {
      v.push(`C5 ${row.id}: registered but catalogue lists no ${H} validator`);
    }
  }
  for (const id of dsls.ids()) {
    if (!cat.dsls.some((r) => r.id === id)) v.push(`C5 ${id}: registered language not in the catalogue`);
  }
  return v;
}
