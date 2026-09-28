#!/usr/bin/env node
// sync-registry-ts — mirror the TypeScript registry library into long-grass.
//
// Source of truth: buhera-os/registry-ts (@buhera/registry). long-grass is
// deployed from its own directory, so, like every other engine it runs, it
// consumes a copy under long-grass/vendor/registry. The wasm artifact is not
// copied: long-grass serves it from public/wasm (scripts/build-wasm.mjs).
//
//   node scripts/sync-registry-ts.mjs          # copy
//   node scripts/sync-registry-ts.mjs --check  # exit 1 if the copy has drifted
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FROM = path.join(ROOT, "buhera-os/registry-ts");
const TO = path.join(ROOT, "long-grass/vendor/registry");
const ITEMS = ["package.json", "README.md", "src"];

function files(base, rel = "") {
  const abs = path.join(base, rel);
  if (!existsSync(abs)) return [];
  if (!statSync(abs).isDirectory()) return [rel];
  return readdirSync(abs).flatMap((e) => files(base, rel ? `${rel}/${e}` : e));
}

const want = ITEMS.flatMap((i) => files(FROM, i)).sort();
const have = ITEMS.flatMap((i) => files(TO, i)).sort();

if (process.argv.includes("--check")) {
  const lf = (p) => readFileSync(p, "utf8").replace(/\r\n/g, "\n");
  const drift = [
    ...want.filter((f) => !existsSync(path.join(TO, f)) || lf(path.join(FROM, f)) !== lf(path.join(TO, f))).map((f) => `differs ${f}`),
    ...have.filter((f) => !want.includes(f)).map((f) => `extra ${f}`),
  ];
  for (const d of drift) console.log(d);
  console.log(drift.length ? `long-grass/vendor/registry has drifted (${drift.length})` : "long-grass/vendor/registry is in sync");
  process.exit(drift.length ? 1 : 0);
}

rmSync(TO, { recursive: true, force: true });
for (const f of want) {
  mkdirSync(path.dirname(path.join(TO, f)), { recursive: true });
  writeFileSync(path.join(TO, f), readFileSync(path.join(FROM, f)));
}
console.log(`synced ${want.length} file(s) → long-grass/vendor/registry`);
