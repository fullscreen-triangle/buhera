/* ============================================================================
 * Who may use a route that reaches outside: this machine, or a signed-in
 * member of the hosted site.
 *
 * long-grass's own sign-in is the gateway's (pages/login.js). A request from
 * elsewhere must carry that session as `Authorization: Bearer <token>`; it is
 * checked by asking the gateway for the account's experiments, which the
 * gateway answers only for a valid session. Answers are remembered for ten
 * minutes so reading a site does not ask the gateway once per page.
 * ========================================================================== */

import { isLocalRequest } from "@/lib/server/rag";

const GATEWAY = process.env.BUHERA_GATEWAY_URL || process.env.NEXT_PUBLIC_BUHERA_GATEWAY_URL || "https://buhera-91-98-157-147.sslip.io";
const TTL_MS = 10 * 60 * 1000;
const seen = new Map(); // token → { ok, until }

/** → { ok: true, local } or { ok: false, status, error } */
export async function allowed(req) {
  if (isLocalRequest(req)) return { ok: true, local: true };
  const m = /^Bearer\s+(.+)$/i.exec(req.headers?.authorization || "");
  if (!m) return { ok: false, status: 401, error: "sign in first: on the hosted site this needs your session" };
  const token = m[1].trim();
  const hit = seen.get(token);
  if (hit && hit.until > Date.now()) return hit.ok ? { ok: true, local: false } : { ok: false, status: 401, error: "your session has expired; sign in again" };
  let ok = false;
  try {
    const r = await fetch(`${GATEWAY}/api/experiments`, { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(10_000) });
    ok = r.ok;
  } catch {
    return { ok: false, status: 503, error: "the gateway could not be reached to check your session" };
  }
  seen.set(token, { ok, until: Date.now() + TTL_MS });
  if (seen.size > 500) seen.delete(seen.keys().next().value);
  return ok ? { ok: true, local: false } : { ok: false, status: 401, error: "your session has expired; sign in again" };
}
