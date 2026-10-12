/* ============================================================================
 * useGatewaySession
 *
 * Thin React hook over the gateway module's own session storage
 * (localStorage key "buhera.gateway.session", owned by gateway-module.js).
 * Does not reimplement login/logout — dispatches to gatewayModule.execute()
 * exactly like the terminal does, then re-reads the session state so
 * components can react to it.
 * ========================================================================== */

import { useCallback, useEffect, useState } from "react";
import { gatewayModule } from "@/lib/modules/gateway-module";

// The browser's native "storage" event only fires in *other* tabs/windows,
// never the one that made the write — so a same-tab login (e.g. the /login
// page navigating to /) would leave every other mounted useGatewaySession()
// instance (like _app.js's AuthGate) holding a stale, pre-login snapshot
// until something else happened to re-render it. This custom event is
// dispatched right after every write, in this tab, so every hook instance
// picks up the change immediately regardless of which one made it.
const SESSION_CHANGED_EVENT = "buhera:gateway-session-changed";

function readRaw() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem("buhera.gateway.session");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.token !== "string") return null;
    // The gateway's session TTL is 12 hours; a stale entry here is not a
    // hypothetical — it's what every session looks like twelve hours after
    // login. Treating mere presence as "logged in" sent every page here on
    // past its expiry straight back to `/`, which then failed against the
    // real (expired) token with no way back to the login form. Expiry is
    // already in the stored object (see `writeSession` in gateway-module.js)
    // — this is a read-time check, not a new network call.
    if (typeof parsed.expires_at === "number" && parsed.expires_at * 1000 <= Date.now()) {
      window.localStorage.removeItem("buhera.gateway.session");
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function useGatewaySession() {
  const [session, setSession] = useState(() => readRaw());
  const [checked, setChecked] = useState(false);

  const refresh = useCallback(() => {
    setSession(readRaw());
  }, []);

  useEffect(() => {
    refresh();
    setChecked(true);
    // Pick up login/logout from another tab (native event) or from another
    // mounted instance of this hook in the same tab (custom event — see
    // SESSION_CHANGED_EVENT above).
    function onStorage(e) {
      if (e.key === "buhera.gateway.session") refresh();
    }
    window.addEventListener("storage", onStorage);
    window.addEventListener(SESSION_CHANGED_EVENT, refresh);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(SESSION_CHANGED_EVENT, refresh);
    };
  }, [refresh]);

  const login = useCallback(async (email, password) => {
    const res = await gatewayModule.execute({ kind: "login", email, password });
    refresh();
    window.dispatchEvent(new Event(SESSION_CHANGED_EVENT));
    return res;
  }, [refresh]);

  const logout = useCallback(async () => {
    await gatewayModule.execute("logout");
    refresh();
    window.dispatchEvent(new Event(SESSION_CHANGED_EVENT));
  }, [refresh]);

  return {
    session,
    loggedIn: !!session,
    checked,
    email: session?.email || null,
    login,
    logout,
    refresh,
  };
}
