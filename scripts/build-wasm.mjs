#!/usr/bin/env node
// build-wasm — compile buhera-os/crates/buhera-wasm for wasm32 and install the
// artifact where the TypeScript hosts load it (specification 07 §4):
//   buhera-os/registry-ts/wasm/buhera_modules.wasm   (library + its tests)
//   long-grass/public/wasm/buhera_modules.wasm       (served to the browser)
// Prints the artifact's size and SHA-256 so a commit can record which build
// shipped. Usage: node scripts/build-wasm.mjs
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OS = path.join(ROOT, "buhera-os");
execFileSync("cargo", ["build", "-p", "buhera-wasm", "--target", "wasm32-unknown-unknown", "--release"], {
  cwd: OS,
  stdio: "inherit",
});
const built = path.join(OS, "target/wasm32-unknown-unknown/release/buhera_wasm.wasm");
const bytes = readFileSync(built);
for (const dest of ["buhera-os/registry-ts/wasm/buhera_modules.wasm", "long-grass/public/wasm/buhera_modules.wasm"]) {
  const abs = path.join(ROOT, dest);
  mkdirSync(path.dirname(abs), { recursive: true });
  copyFileSync(built, abs);
  console.log(`installed ${dest}`);
}
console.log(`${bytes.length} bytes  sha256 ${createHash("sha256").update(bytes).digest("hex")}`);
