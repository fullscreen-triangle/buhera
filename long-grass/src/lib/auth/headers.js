/* The gateway session as request headers, for routes that check it
 * (lib/server/session.js). Empty when signed out or outside a browser. */

export function authHeaders() {
  try {
    const s = JSON.parse(window.localStorage.getItem("buhera.gateway.session") || "null");
    return s?.token ? { Authorization: `Bearer ${s.token}` } : {};
  } catch {
    return {};
  }
}

/** POST JSON with the session; never throws. → the JSON body, with ok:false on failure. */
export async function postJSON(path, body) {
  try {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!json) return { ok: false, error: `HTTP ${res.status}` };
    return res.ok ? { ok: true, ...json } : { ...json, ok: false, error: json.error || `HTTP ${res.status}` };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}
