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
