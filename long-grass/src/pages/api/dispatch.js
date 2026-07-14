// API route: /api/dispatch
//
// The seam that turns every Buhera deployment into a remote catalyst.
//
//   GET  /api/dispatch?health=1     → { ok, name, version, modules[] }
//   POST /api/dispatch              body: { moduleId, instruction, actBudget }
//                                   → the ActResult the module returned
//
// Auth: if BUHERA_DISPATCH_TOKEN is set on the deployment, requests must send
// "Authorization: Bearer <token>". If unset, endpoint is open (fine for LAN,
// Tailscale, or a private deployment; NOT fine for a public Vercel URL that
// you don't want others to bill your API keys to).
//
// Modules that require browser-only state (window, localStorage, live
// singletons in the terminal component) cannot run through this route. The
// server-safe set is enumerated below; adding a module means confirming it
// has no browser dependency.

import { echoModule } from "@/lib/modules/echo-module";
import { lavoisierModule } from "@/lib/modules/lavoisier-module";
import { purposeCarryModule } from "@/lib/modules/purpose-carry-module";
import { graffitiModule } from "@/lib/modules/graffiti-module";

// The subset of modules that work outside a browser. Every entry MUST NOT
// touch window, document, or localStorage during execute().
const SERVER_SAFE_MODULES = {
  echo: echoModule,
  lavoisier: lavoisierModule,
  "purpose-carry": purposeCarryModule,
  graffiti: graffitiModule,
};

const NAME = "long-grass";
const VERSION = "0.1.0";
const MAX_INSTRUCTION_BYTES = 512 * 1024;

// --------------------------------------------------------------------------
// Auth check
// --------------------------------------------------------------------------

function checkAuth(req) {
  const expected = process.env.BUHERA_DISPATCH_TOKEN;
  if (!expected) return { ok: true }; // open endpoint
  const header = req.headers.authorization || "";
  if (!header.startsWith("Bearer ")) {
    return { ok: false, status: 401, error: "missing Authorization: Bearer <token>" };
  }
  if (header.slice("Bearer ".length).trim() !== expected) {
    return { ok: false, status: 403, error: "invalid dispatch token" };
  }
  return { ok: true };
}

// --------------------------------------------------------------------------
// Handler
// --------------------------------------------------------------------------

export default async function handler(req, res) {
  // Permissive CORS — a Buhera at URL A dispatches from a browser at URL B.
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.status(204).end();

  // Health check.
  if (req.method === "GET") {
    if (req.query.health === "1" || req.query.health === "true") {
      return res.status(200).json({
        ok: true,
        name: NAME,
        version: VERSION,
        modules: Object.keys(SERVER_SAFE_MODULES),
        auth_required: !!process.env.BUHERA_DISPATCH_TOKEN,
        timestamp: new Date().toISOString(),
      });
    }
    return res.status(400).json({ ok: false, error: "missing ?health=1 or use POST" });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST, OPTIONS");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const auth = checkAuth(req);
  if (!auth.ok) return res.status(auth.status).json({ ok: false, error: auth.error });

  const { moduleId, instruction, actBudget } = req.body || {};
  if (typeof moduleId !== "string" || !moduleId) {
    return res.status(400).json({ ok: false, error: "moduleId (string) required" });
  }

  // Size guard.
  try {
    const bytes = Buffer.byteLength(JSON.stringify(instruction ?? null));
    if (bytes > MAX_INSTRUCTION_BYTES) {
      return res.status(413).json({
        ok: false,
        error: `instruction exceeds ${MAX_INSTRUCTION_BYTES} bytes`,
      });
    }
  } catch {
    return res.status(400).json({ ok: false, error: "instruction is not JSON-serializable" });
  }

  const mod = SERVER_SAFE_MODULES[moduleId];
  if (!mod) {
    return res.status(404).json({
      ok: false,
      error: `moduleId "${moduleId}" is not available on this catalyst.`,
      available: Object.keys(SERVER_SAFE_MODULES),
    });
  }

  const t0 = Date.now();
  let result;
  try {
    result = await mod.execute(
      instruction,
      Number.isFinite(actBudget) ? Number(actBudget) : 1,
    );
  } catch (err) {
    return res.status(500).json({
      ok: false,
      error: err.message || String(err),
      elapsed_ms: Date.now() - t0,
    });
  }

  // The ActResult IS the response body. Pass it through verbatim, plus a
  // small envelope telling the caller which catalyst answered.
  return res.status(200).json({
    ...result,
    _catalyst: {
      name: NAME,
      version: VERSION,
      elapsed_ms: Date.now() - t0,
    },
  });
}
