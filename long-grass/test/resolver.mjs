// Resolve hook: makes `node --test` understand this repo's bundler-only import
// conventions — the `@/` alias (jsconfig -> ./src/) and extensionless imports.
// Registered via ./test/alias-hook.mjs.
import { pathToFileURL, fileURLToPath } from "node:url";
import { existsSync, statSync } from "node:fs";
import path from "node:path";

const SRC = pathToFileURL(path.join(process.cwd(), "src") + path.sep).href;

// Append .js / resolve /index.js the way webpack would, when the target has
// no extension. Returns a string URL. Only touches file: URLs that exist.
function withExtension(urlStr) {
  if (!urlStr.startsWith("file:")) return urlStr;
  let p;
  try {
    p = fileURLToPath(urlStr);
  } catch {
    return urlStr;
  }
  if (existsSync(p)) {
    if (statSync(p).isDirectory()) {
      const idx = path.join(p, "index.js");
      return existsSync(idx) ? pathToFileURL(idx).href : urlStr;
    }
    return urlStr;
  }
  for (const ext of [".js", ".mjs", ".cjs", ".json", ".ts"]) {
    if (existsSync(p + ext)) return pathToFileURL(p + ext).href;
  }
  return urlStr;
}

export async function resolve(specifier, context, next) {
  let spec = specifier;
  if (spec.startsWith("@/")) {
    spec = withExtension(SRC + spec.slice(2));
  } else if ((spec.startsWith("./") || spec.startsWith("../")) && context.parentURL) {
    const abs = new URL(spec, context.parentURL).href;
    spec = withExtension(abs);
  }
  return next(spec, context);
}

// Vendored TypeScript engines (vendor/tempus, vendor/zangalewa-interceptor)
// follow bundler conventions — type-only imports without `import type` — so
// Node's native type stripping cannot load them. Transpile vendored .ts with
// the TypeScript compiler (which elides type-only imports), as SWC does in
// the Next build. vendor/registry is erasable TypeScript and would load
// natively; it goes through the same path for uniformity.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const VENDOR = pathToFileURL(path.join(process.cwd(), "vendor") + path.sep).href;
let _ts = null;

export async function load(url, context, next) {
  if (url.endsWith(".ts") && url.startsWith(VENDOR)) {
    _ts ??= createRequire(import.meta.url)("typescript");
    const out = _ts.transpileModule(readFileSync(fileURLToPath(url), "utf8"), {
      compilerOptions: { module: _ts.ModuleKind.ESNext, target: _ts.ScriptTarget.ES2022 },
      fileName: fileURLToPath(url),
    });
    return { format: "module", source: out.outputText, shortCircuit: true };
  }
  return next(url, context);
}
