// API route: /api/triangle
//
// Server-side proxy to the four-corners backend (four-sided-triangle),
// deployed separately from buhera-gateway (own JWT/cookie auth, own SQLite).
// long-grass holds a single service-level four-corners account — the 5
// buhera-gateway profiles do not each need their own four-corners login —
// so this route authenticates itself, caches the session JWT in memory for
// its lifetime, and re-logs-in on expiry.
//
// POST body shapes:
//   { kind: "sources" }                                — list configured sources
//   { kind: "add_source", sourceKind, config }          — add a source
//   { kind: "remove_source", id }                       — remove a source
//
// Env:
//   FOUR_CORNERS_URL            base URL, e.g. https://<host>/  (required)
//   FOUR_CORNERS_SERVICE_EMAIL  service account email           (required)
//   FOUR_CORNERS_SERVICE_PASSWORD service account password      (required)
//
// If any of these are unset, every call returns 503 "service unavailable" —
// this route is a proxy, not a fallback implementation.

const BASE_URL = process.env.FOUR_CORNERS_URL;
const SERVICE_EMAIL = process.env.FOUR_CORNERS_SERVICE_EMAIL;
const SERVICE_PASSWORD = process.env.FOUR_CORNERS_SERVICE_PASSWORD;

// In-memory session cache. Reset on cold start (serverless), which just
// means the next call re-logs-in — cheap relative to request volume here.
let cachedSession = null; // { cookie, obtainedAt }
const SESSION_MAX_AGE_MS = 6 * 24 * 60 * 60 * 1000; // renew before the 7d JWT expires

function configured() {
  return Boolean(BASE_URL && SERVICE_EMAIL && SERVICE_PASSWORD);
}

async function login() {
  const res = await fetch(`${BASE_URL.replace(/\/$/, "")}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: SERVICE_EMAIL, password: SERVICE_PASSWORD }),
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) {
    throw Object.assign(new Error(`four-corners login failed (HTTP ${res.status})`), { status: res.status });
  }
  const setCookie = res.headers.get("set-cookie");
  if (!setCookie) {
    throw new Error("four-corners login response carried no session cookie");
  }
  // Only the session cookie's name=value is needed on the way back out.
  const cookie = setCookie.split(";")[0];
  cachedSession = { cookie, obtainedAt: Date.now() };
  return cachedSession;
}

async function session() {
  if (cachedSession && Date.now() - cachedSession.obtainedAt < SESSION_MAX_AGE_MS) {
    return cachedSession;
  }
  return login();
}

async function fcFetch(path, opts = {}) {
  const sess = await session();
  const doFetch = (cookie) =>
    fetch(`${BASE_URL.replace(/\/$/, "")}${path}`, {
      ...opts,
      headers: { ...(opts.headers || {}), Cookie: cookie },
      signal: AbortSignal.timeout(8000),
    });

  let res = await doFetch(sess.cookie);
  if (res.status === 401) {
    // Cached session rejected — log in fresh once and retry, rather than
    // looping (a persistently-401ing service should surface as an error).
    const fresh = await login();
    res = await doFetch(fresh.cookie);
  }
  return res;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }
  if (!configured()) {
    return res.status(503).json({
      ok: false,
      error:
        "four-corners is not configured (FOUR_CORNERS_URL / FOUR_CORNERS_SERVICE_EMAIL / FOUR_CORNERS_SERVICE_PASSWORD)",
    });
  }

  const { kind } = req.body ?? {};

  try {
    if (kind === "sources") {
      const r = await fcFetch("/api/sources", { method: "GET" });
      const data = await r.json();
      if (!r.ok) return res.status(r.status).json({ ok: false, error: data?.error || `HTTP ${r.status}` });
      return res.status(200).json({ ok: true, sources: data });
    }

    if (kind === "add_source") {
      const { sourceKind, config } = req.body;
      if (!sourceKind || !config) {
        return res.status(400).json({ ok: false, error: "sourceKind and config are required" });
      }
      const r = await fcFetch("/api/sources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: sourceKind, config }),
      });
      const data = await r.json();
      if (!r.ok) return res.status(r.status).json({ ok: false, error: data?.error || `HTTP ${r.status}` });
      return res.status(200).json({ ok: true, source: data });
    }

    if (kind === "remove_source") {
      const { id } = req.body;
      if (!id) return res.status(400).json({ ok: false, error: "id is required" });
      const r = await fcFetch(`/api/sources/${encodeURIComponent(id)}`, { method: "DELETE" });
      const data = await r.json();
      if (!r.ok) return res.status(r.status).json({ ok: false, error: data?.error || `HTTP ${r.status}` });
      return res.status(200).json({ ok: true });
    }

    return res.status(400).json({ ok: false, error: `unknown kind "${kind}"` });
  } catch (err) {
    return res.status(502).json({ ok: false, error: err.message || String(err) });
  }
}
