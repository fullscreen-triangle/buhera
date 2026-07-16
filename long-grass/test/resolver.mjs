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
  for (const ext of [".js", ".mjs", ".cjs", ".json"]) {
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
