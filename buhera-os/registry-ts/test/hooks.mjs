// Test-only module hooks. The library's own src/ is erasable TypeScript that
// Node runs natively; the *vendored engines* the tests inject (long-grass/
// vendor) follow bundler conventions instead: extensionless relative imports
// and TypeScript that imports types without `import type`. In the app,
// webpack/SWC handles both. Here: resolve extensionless specifiers the way a
// bundler would, and transpile vendored .ts with the TypeScript compiler
// (which elides type-only imports). Registered by test/register.mjs.
import { existsSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = pathToFileURL(path.resolve(fileURLToPath(import.meta.url), "../../src") + path.sep).href;
const require = createRequire(import.meta.url);
let ts = null;
const typescript = () =>
  (ts ??= require(path.resolve(fileURLToPath(import.meta.url), "../../../../long-grass/node_modules/typescript")));

function withExtension(url) {
  if (!url.startsWith("file:")) return url;
  const p = fileURLToPath(url);
  if (existsSync(p) && !statSync(p).isDirectory()) return url;
  for (const ext of [".ts", ".js", ".mjs", "/index.ts", "/index.js"]) {
    if (existsSync(p + ext)) return pathToFileURL(p + ext).href;
  }
  return url;
}

export async function resolve(specifier, context, next) {
  if ((specifier.startsWith("./") || specifier.startsWith("../")) && context.parentURL) {
    return next(withExtension(new URL(specifier, context.parentURL).href), context);
  }
  return next(specifier, context);
}

export async function load(url, context, next) {
  if (url.endsWith(".ts") && !url.startsWith(SRC) && !url.includes("/test/")) {
    const source = readFileSync(fileURLToPath(url), "utf8");
    const out = typescript().transpileModule(source, {
      compilerOptions: { module: 99 /* ESNext */, target: 9 /* ES2022 */ },
      fileName: fileURLToPath(url),
    });
    return { format: "module", source: out.outputText, shortCircuit: true };
  }
  return next(url, context);
}
