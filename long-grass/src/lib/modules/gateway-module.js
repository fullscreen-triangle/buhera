/* ============================================================================
 * Gateway Module
 *
 * The seam to the Buhera gateway (accounts, sessions, machine pairing, and
 * routed execution) at https://buhera-91-98-157-147.sslip.io. Unlike the
 * catalyst registry — a pool of URLs this browser calls directly — the
 * gateway holds identity: one account, followed between machines, with its
 * own roster of paired catalysts and its own routing decision per run.
 *
 * The gateway is a JSON API with no page of its own; this module is what
 * turns it into something the terminal can drive.
 *
 * Persistence: browser localStorage under buhera.gateway.session — just the
 * token and account id, nothing about the account's content (that lives on
 * the gateway, or on the user's own paired machine).
 *
 * Instructions:
 *   • { kind: "signup", email, password }
 *   • { kind: "login", email, password }
 *   • { kind: "logout" }
 *   • { kind: "whoami" }
 *   • { kind: "catalysts" }                                    → list machines
 *   • { kind: "pair", name, capabilities? }                    → register a machine
 *   • { kind: "unpair", name }
 *   • { kind: "run", source, capability?, prefer? }            → execute vaHera
 * ========================================================================== */

const STORAGE_KEY = "buhera.gateway.session";
const DEFAULT_BASE_URL = "https://buhera-91-98-157-147.sslip.io";

function baseUrl() {
  if (typeof window !== "undefined" && window.__BUHERA_GATEWAY_URL__) {
    return window.__BUHERA_GATEWAY_URL__;
  }
  return process.env.NEXT_PUBLIC_BUHERA_GATEWAY_URL || DEFAULT_BASE_URL;
}

// --------------------------------------------------------------------------
// Session storage (browser-only; guarded for SSR).
// --------------------------------------------------------------------------

function readSession() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.token !== "string") return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeSession(session) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Quota exceeded or storage disabled; the session still works for the
    // lifetime of the tab, it just won't survive a reload.
  }
}

function clearSession() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // noop
  }
}

// --------------------------------------------------------------------------
// Wire helper.
// --------------------------------------------------------------------------

async function call(path, { method = "GET", auth = false, body } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const session = readSession();
    if (!session) {
      return { ok: false, status: 0, error: "not logged in" };
    }
    headers.Authorization = `Bearer ${session.token}`;
  }
  let res;
  try {
    res = await fetch(baseUrl() + path, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    return { ok: false, status: 0, error: err.message || String(err) };
  }
  let json = null;
  try { json = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    // A 401 here always means "not authenticated" — the gateway deliberately
    // gives no other detail (see buhera-gateway/src/http.rs). Treat it as a
    // signal to drop whatever session we're holding, since it's clearly no
    // longer accepted.
    if (res.status === 401) clearSession();
    return { ok: false, status: res.status, error: json?.error || `HTTP ${res.status}` };
  }
  return { ok: true, status: res.status, ...json };
}

// --------------------------------------------------------------------------
// The Module trait.
// --------------------------------------------------------------------------

export const gatewayModule = {
  id: "gateway",

  describe() {
    return {
      id: "gateway",
      description:
        "Buhera gateway — accounts, machine pairing, and routed execution. " +
        `Currently pointed at ${baseUrl()}.`,
      instructions: [
        'dispatch("gateway", { kind: "signup", email: "you@example.com", password: "at least twelve chars" })',
        'dispatch("gateway", { kind: "login", email: "you@example.com", password: "..." })',
        'dispatch("gateway", "whoami")',
        'dispatch("gateway", { kind: "pair", name: "office", capabilities: ["kernel", "vahera"] })',
        'dispatch("gateway", "catalysts")',
        'dispatch("gateway", { kind: "run", source: "memory list" })',
        'dispatch("gateway", "logout")',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const inst = typeof instruction === "string" ? { kind: instruction } : instruction || {};
    const kind = inst.kind || "whoami";

    if (kind === "signup" || kind === "login") {
      const email = String(inst.email || "").trim();
      const password = String(inst.password || "");
      if (!email || !password) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway ${kind}: email and password are required`] },
          residue: 0,
          completed: true,
          error: "missing-credentials",
        };
      }
      const res = await call(`/api/auth/${kind}`, { method: "POST", body: { email, password } });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway ${kind} failed: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      writeSession({ token: res.token, account_id: res.account_id, email, expires_at: res.expires_at });
      return {
        ok: true,
        output_delta: {
          kind: "text",
          lines: [`gateway: logged in as ${email}`, `session valid until ${new Date(res.expires_at * 1000).toISOString()}`],
        },
        residue: 1,
        completed: true,
      };
    }

    if (kind === "logout") {
      clearSession();
      return {
        ok: true,
        output_delta: { kind: "text", lines: ["gateway: logged out"] },
        residue: 0,
        completed: true,
      };
    }

    if (kind === "whoami") {
      const session = readSession();
      if (!session) {
        return {
          ok: true,
          output_delta: { kind: "text", lines: ["gateway: not logged in"] },
          residue: 0,
          completed: true,
        };
      }
      return {
        ok: true,
        output_delta: {
          kind: "gateway_session",
          email: session.email,
          account_id: session.account_id,
          expires_at: session.expires_at,
          lines: [`gateway: logged in as ${session.email} (${session.account_id})`],
        },
        residue: 1,
        completed: true,
      };
    }

    if (kind === "catalysts") {
      const res = await call("/api/catalysts", { auth: true });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway catalysts: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      const items = res.catalysts || [];
      const lines = items.length
        ? items.map((c) => `  ${c.name} — ${c.live ? "live" : "asleep"} — [${c.capabilities.join(", ")}]`)
        : ["  (no machines paired)"];
      return {
        ok: true,
        output_delta: { kind: "gateway_machines", entries: items, count: items.length, lines: ["gateway machines:", ...lines] },
        residue: items.length,
        completed: true,
      };
    }

    if (kind === "pair") {
      const name = String(inst.name || "").trim();
      if (!name) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: ["gateway pair: name is required"] },
          residue: 0,
          completed: true,
          error: "no-name",
        };
      }
      const capabilities = Array.isArray(inst.capabilities) ? inst.capabilities.map(String) : [];
      const res = await call("/api/catalysts", { method: "POST", auth: true, body: { name, capabilities } });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway pair: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      return {
        ok: true,
        output_delta: {
          kind: "gateway_pair_token",
          name: res.name,
          token: res.token,
          expires_at: res.expires_at,
          lines: [
            `gateway: paired "${res.name}". this token is shown once —`,
            `paste it into the machine you're pairing so it can dial in:`,
            "",
            res.token,
          ],
        },
        residue: 1,
        completed: true,
      };
    }

    if (kind === "unpair") {
      const name = String(inst.name || "").trim();
      if (!name) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: ["gateway unpair: name is required"] },
          residue: 0,
          completed: true,
          error: "no-name",
        };
      }
      const res = await call(`/api/catalysts/${encodeURIComponent(name)}`, { method: "DELETE", auth: true });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway unpair: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      return {
        ok: true,
        output_delta: { kind: "text", lines: [`gateway: unpaired "${name}"`] },
        residue: 0,
        completed: true,
      };
    }

    if (kind === "run") {
      const source = String(inst.source || "");
      if (!source.trim()) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: ["gateway run: source is required"] },
          residue: 0,
          completed: true,
          error: "no-source",
        };
      }
      const body = { source };
      if (inst.capability) body.capability = String(inst.capability);
      if (inst.prefer) body.prefer = String(inst.prefer);
      const res = await call("/api/run", { method: "POST", auth: true, body });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`gateway run: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      const lines = [`ran on: ${res.executed_on}`];
      if (res.note) lines.push(`note: ${res.note}`);
      return {
        ok: true,
        output_delta: {
          kind: "gateway_run",
          executed_on: res.executed_on,
          note: res.note || null,
          results: res.results,
          trace: res.trace,
          lines,
        },
        residue: (res.results || []).length,
        completed: true,
      };
    }

    return {
      ok: false,
      output_delta: { kind: "text", lines: [`gateway: unknown kind "${kind}"`] },
      residue: 0,
      completed: true,
      error: `unknown kind "${kind}"`,
    };
  },

  outputCell() {
    return { kind: "gateway_cell" };
  },
};
