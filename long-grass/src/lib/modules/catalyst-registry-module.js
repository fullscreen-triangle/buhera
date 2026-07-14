/* ============================================================================
 * Catalyst Registry Module
 *
 * The compute pool. Not the module registry (that's the federation of what can
 * be done); this is the registry of where things can run. Buhera OS is a
 * decision plane; catalysts are the resources it can route dispatches to.
 *
 * A catalyst is any Buhera-compatible endpoint the browser can reach: a Vercel
 * deployment, a Hetzner VM, a Vultr instance, a Codespaces port, a Chromebook
 * on Tailscale, another user's laptop with ngrok. Each catalyst is a URL plus
 * some metadata about what it can do and what it costs.
 *
 * Persistence: browser localStorage under buhera.catalysts. Each browser holds
 * its own pool. Export / import lets a user move the pool between devices.
 *
 * Instructions:
 *   • { kind: "add", entry: { name, url, auth?, capabilities?, cost_hint?,
 *                             availability?, notes? } }
 *   • { kind: "remove", name }
 *   • { kind: "list" }
 *   • { kind: "get", name }
 *   • { kind: "ping", name }
 *   • { kind: "export" }               → returns a JSON blob
 *   • { kind: "import", blob, mode? }  → mode "merge" | "replace"
 *   • { kind: "clear" }
 *   • "list" (bare string)             → shorthand for { kind: "list" }
 *
 * A default entry named "local" is seeded on first boot: it points at the
 * current instance itself (window.location.origin), so a compute dispatch to
 * "local" round-trips through this Buhera's own /api/dispatch route.
 * ========================================================================== */

const STORAGE_KEY = "buhera.catalysts";
const SCHEMA_VERSION = 1;

// --------------------------------------------------------------------------
// Storage layer (browser-only; guarded for SSR).
// --------------------------------------------------------------------------

function readStore() {
  if (typeof window === "undefined") return { version: SCHEMA_VERSION, entries: {} };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { version: SCHEMA_VERSION, entries: {} };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return { version: SCHEMA_VERSION, entries: {} };
    }
    return {
      version: parsed.version || SCHEMA_VERSION,
      entries: parsed.entries && typeof parsed.entries === "object" ? parsed.entries : {},
    };
  } catch {
    return { version: SCHEMA_VERSION, entries: {} };
  }
}

function writeStore(store) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // Quota exceeded or storage disabled; silently continue. The pool will
    // still work for the lifetime of the tab.
  }
}

// --------------------------------------------------------------------------
// Seeding the "local" catalyst on first boot.
// --------------------------------------------------------------------------

function seedIfEmpty(store) {
  if (Object.keys(store.entries).length > 0) return store;
  if (typeof window === "undefined") return store;

  const origin = window.location?.origin;
  if (!origin) return store;

  store.entries.local = {
    name: "local",
    url: origin,
    auth: { kind: "none" },
    capabilities: ["cpu", "in-browser"],
    cost_hint: "free",
    availability: "always-on",
    notes: "this Buhera instance itself (round-trip through /api/dispatch)",
    added_at: new Date().toISOString(),
  };
  writeStore(store);
  return store;
}

// --------------------------------------------------------------------------
// Entry validation.
// --------------------------------------------------------------------------

const VALID_AUTH_KINDS = new Set(["none", "bearer", "basic"]);
const VALID_COST_HINTS = new Set(["free", "cheap", "expensive"]);
const VALID_AVAILABILITY = new Set(["always-on", "on-demand", "ephemeral"]);

function validateEntry(entry) {
  if (!entry || typeof entry !== "object") {
    return "entry must be an object";
  }
  const name = String(entry.name || "").trim();
  if (!name) return "entry.name is required (non-empty string)";
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/i.test(name)) {
    return "entry.name must be lowercase alphanumeric or hyphen, 1-64 chars";
  }
  const url = String(entry.url || "").trim();
  if (!url) return "entry.url is required";
  try {
    // eslint-disable-next-line no-new
    new URL(url);
  } catch {
    return `entry.url is not a valid URL: ${url}`;
  }
  if (entry.auth != null) {
    if (typeof entry.auth !== "object") return "entry.auth must be an object";
    const kind = entry.auth.kind || "none";
    if (!VALID_AUTH_KINDS.has(kind)) {
      return `entry.auth.kind must be one of ${[...VALID_AUTH_KINDS].join(", ")}`;
    }
    if (kind === "bearer" && !entry.auth.token) {
      return "entry.auth.kind='bearer' requires entry.auth.token";
    }
    if (kind === "basic" && (!entry.auth.username || !entry.auth.password)) {
      return "entry.auth.kind='basic' requires username and password";
    }
  }
  if (entry.capabilities != null && !Array.isArray(entry.capabilities)) {
    return "entry.capabilities must be an array of strings";
  }
  if (entry.cost_hint != null && !VALID_COST_HINTS.has(entry.cost_hint)) {
    return `entry.cost_hint must be one of ${[...VALID_COST_HINTS].join(", ")}`;
  }
  if (entry.availability != null && !VALID_AVAILABILITY.has(entry.availability)) {
    return `entry.availability must be one of ${[...VALID_AVAILABILITY].join(", ")}`;
  }
  return null;
}

function normaliseEntry(entry) {
  return {
    name: String(entry.name).trim(),
    url: String(entry.url).trim().replace(/\/$/, ""),
    auth: entry.auth || { kind: "none" },
    capabilities: Array.isArray(entry.capabilities) ? entry.capabilities.map(String) : [],
    cost_hint: entry.cost_hint || "unknown",
    availability: entry.availability || "unknown",
    notes: entry.notes ? String(entry.notes) : "",
    added_at: entry.added_at || new Date().toISOString(),
  };
}

// --------------------------------------------------------------------------
// Public accessor used by other modules (compute-module).
// --------------------------------------------------------------------------

export function getCatalyst(name) {
  const store = seedIfEmpty(readStore());
  return store.entries[name] || null;
}

export function listCatalysts() {
  const store = seedIfEmpty(readStore());
  return Object.values(store.entries);
}

// --------------------------------------------------------------------------
// Ping: GET <url>/api/dispatch?health=1
// --------------------------------------------------------------------------

async function pingCatalyst(entry) {
  const url = entry.url.replace(/\/$/, "") + "/api/dispatch?health=1";
  const headers = {};
  if (entry.auth?.kind === "bearer") {
    headers.Authorization = `Bearer ${entry.auth.token}`;
  } else if (entry.auth?.kind === "basic") {
    headers.Authorization =
      "Basic " +
      Buffer.from(`${entry.auth.username}:${entry.auth.password}`).toString("base64");
  }
  const started = Date.now();
  try {
    const res = await fetch(url, { method: "GET", headers });
    const elapsedMs = Date.now() - started;
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    return {
      ok: res.ok,
      status: res.status,
      elapsed_ms: elapsedMs,
      body,
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      elapsed_ms: Date.now() - started,
      error: err.message || String(err),
    };
  }
}

// --------------------------------------------------------------------------
// The Module trait.
// --------------------------------------------------------------------------

export const catalystRegistryModule = {
  id: "catalysts",

  describe() {
    return {
      id: "catalysts",
      description:
        "Catalyst pool — the compute resources this Buhera instance can " +
        "route dispatches to. Each entry is a URL plus metadata. Persists " +
        "to browser localStorage.",
      instructions: [
        'dispatch("catalysts", "list")',
        'dispatch("catalysts", { kind: "add", entry: { name: "hetzner", url: "https://...", auth: { kind: "bearer", token: "..." }, capabilities: ["cpu","always-on"], cost_hint: "cheap" } })',
        'dispatch("catalysts", { kind: "ping", name: "hetzner" })',
        'dispatch("catalysts", { kind: "remove", name: "hetzner" })',
        'dispatch("catalysts", { kind: "export" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const inst = typeof instruction === "string"
      ? (instruction === "list" ? { kind: "list" } : { kind: "list" })
      : instruction || { kind: "list" };
    const kind = inst.kind || "list";

    try {
      if (kind === "list") {
        const entries = listCatalysts();
        return {
          ok: true,
          output_delta: {
            kind: "catalyst_list",
            entries,
            count: entries.length,
          },
          residue: entries.length,
          completed: true,
        };
      }

      if (kind === "get") {
        const entry = getCatalyst(String(inst.name || ""));
        if (!entry) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: [`catalysts: no entry named "${inst.name}"`] },
            residue: 0,
            completed: true,
            error: "not-found",
          };
        }
        return {
          ok: true,
          output_delta: { kind: "catalyst_entry", entry },
          residue: 1,
          completed: true,
        };
      }

      if (kind === "add") {
        const err = validateEntry(inst.entry);
        if (err) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: [`catalysts: ${err}`] },
            residue: 0,
            completed: true,
            error: err,
          };
        }
        const store = seedIfEmpty(readStore());
        const entry = normaliseEntry(inst.entry);
        const isUpdate = !!store.entries[entry.name];
        store.entries[entry.name] = entry;
        writeStore(store);
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [
              `catalysts: ${isUpdate ? "updated" : "added"} "${entry.name}" → ${entry.url}`,
              `pool now holds ${Object.keys(store.entries).length} entries.`,
            ],
          },
          residue: 1,
          completed: true,
        };
      }

      if (kind === "remove") {
        const store = seedIfEmpty(readStore());
        const name = String(inst.name || "");
        if (!store.entries[name]) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: [`catalysts: no entry named "${name}"`] },
            residue: 0,
            completed: true,
            error: "not-found",
          };
        }
        delete store.entries[name];
        writeStore(store);
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [`catalysts: removed "${name}". pool now holds ${Object.keys(store.entries).length} entries.`],
          },
          residue: 0,
          completed: true,
        };
      }

      if (kind === "ping") {
        const entry = getCatalyst(String(inst.name || ""));
        if (!entry) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: [`catalysts: no entry named "${inst.name}"`] },
            residue: 0,
            completed: true,
            error: "not-found",
          };
        }
        const result = await pingCatalyst(entry);
        const lines = [
          `ping ${entry.name} (${entry.url}):`,
          result.ok
            ? `  reachable — HTTP ${result.status} in ${result.elapsed_ms} ms`
            : `  UNREACHABLE — ${result.error || `HTTP ${result.status}`} (${result.elapsed_ms} ms)`,
        ];
        if (result.body?.modules && Array.isArray(result.body.modules)) {
          lines.push(`  reports ${result.body.modules.length} modules: ${result.body.modules.join(", ")}`);
        }
        return {
          ok: result.ok,
          output_delta: {
            kind: "catalyst_ping",
            name: entry.name,
            url: entry.url,
            result,
            lines,
          },
          residue: result.ok ? 1 : 0,
          completed: true,
        };
      }

      if (kind === "export") {
        const store = seedIfEmpty(readStore());
        const blob = JSON.stringify(store, null, 2);
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [
              `catalysts: exported ${Object.keys(store.entries).length} entries.`,
              "",
              blob,
            ],
          },
          residue: Object.keys(store.entries).length,
          completed: true,
        };
      }

      if (kind === "import") {
        let payload;
        try {
          payload = typeof inst.blob === "string" ? JSON.parse(inst.blob) : inst.blob;
        } catch (err) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: [`catalysts: import blob is not valid JSON — ${err.message}`] },
            residue: 0,
            completed: true,
            error: "invalid-blob",
          };
        }
        if (!payload || typeof payload !== "object" || !payload.entries) {
          return {
            ok: false,
            output_delta: { kind: "text", lines: ["catalysts: import blob missing 'entries' object"] },
            residue: 0,
            completed: true,
            error: "invalid-blob",
          };
        }
        const mode = inst.mode === "replace" ? "replace" : "merge";
        const store = mode === "replace"
          ? { version: SCHEMA_VERSION, entries: {} }
          : seedIfEmpty(readStore());
        let added = 0;
        for (const [name, entry] of Object.entries(payload.entries)) {
          const err = validateEntry({ ...entry, name });
          if (err) continue;
          store.entries[name] = normaliseEntry({ ...entry, name });
          added++;
        }
        writeStore(store);
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [
              `catalysts: imported ${added} entries in ${mode} mode.`,
              `pool now holds ${Object.keys(store.entries).length} entries.`,
            ],
          },
          residue: added,
          completed: true,
        };
      }

      if (kind === "clear") {
        writeStore({ version: SCHEMA_VERSION, entries: {} });
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: ["catalysts: cleared. next dispatch will re-seed 'local'."],
          },
          residue: 0,
          completed: true,
        };
      }

      return {
        ok: false,
        output_delta: { kind: "text", lines: [`catalysts: unknown kind "${kind}"`] },
        residue: 0,
        completed: true,
        error: `unknown kind "${kind}"`,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: { kind: "text", lines: [`catalysts error: ${err.message || String(err)}`] },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  outputCell() {
    return { kind: "catalyst_registry_cell" };
  },
};
