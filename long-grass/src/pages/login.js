import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import { useGatewaySession } from "@/lib/auth/useGatewaySession";
import { gatewayModule } from "@/lib/modules/gateway-module";

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // Clipboard API unavailable (insecure context, permissions) — the
          // token is still selectable/visible in the <pre> below.
        }
      }}
      className="text-xs text-emerald-400 hover:text-emerald-300 underline"
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

// A single "stripe": a bare underline field, not a boxed input — label
// above, hairline border only on the bottom edge.
function Stripe({ label, ...inputProps }) {
  return (
    <div className="mb-7">
      <label className="block text-[11px] tracking-wide text-gray-500 uppercase mb-1.5">{label}</label>
      <input
        {...inputProps}
        className="w-full bg-transparent border-0 border-b border-gray-700 focus:border-emerald-500 outline-none py-1.5 text-gray-100 text-base"
      />
    </div>
  );
}

export default function Login() {
  const router = useRouter();
  const { loggedIn, checked, login } = useGatewaySession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [device, setDevice] = useState("");
  const [error, setError] = useState(null);
  const [pairNotice, setPairNotice] = useState(null);
  const [pairToken, setPairToken] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    // Already logged in (e.g. a reload after a device was already paired
    // this session): skip straight through, same as before. A fresh
    // device-token display in progress on this same screen is left alone
    // rather than yanked away by the redirect.
    if (checked && loggedIn && !pairToken) {
      const dest = typeof router.query.next === "string" ? router.query.next : "/";
      router.replace(dest);
    }
  }, [checked, loggedIn, router, pairToken]);

  async function onSubmit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    setPairNotice(null);
    setPairToken(null);

    const res = await login(email.trim(), password);
    if (!res.ok) {
      setBusy(false);
      setError(res.error || "login failed");
      return;
    }

    // Login succeeded. A device name, entered here, folds today's separate
    // "log in, then visit /pair" flow into one screen. A pairing failure
    // must never undo a correct login — it's surfaced, not fatal.
    const name = device.trim();
    if (name) {
      const pairRes = await gatewayModule.execute({ kind: "pair", name, capabilities: [] });
      if (pairRes.ok) {
        setPairToken(pairRes.output_delta);
        setBusy(false);
        return; // Show the one-time token; the user moves on manually (see "continue" below).
      }
      setPairNotice(`signed in, but pairing "${name}" failed: ${pairRes.error || "unknown error"}`);
    }

    setBusy(false);
    const dest = typeof router.query.next === "string" ? router.query.next : "/";
    router.replace(dest);
  }

  function continueToApp() {
    const dest = typeof router.query.next === "string" ? router.query.next : "/";
    router.replace(dest);
  }

  return (
    <>
      <Head>
        <title>buhera — sign in</title>
      </Head>
      <TransitionEffect />
      <div className="fixed inset-0 bg-black text-gray-300 flex items-center justify-center font-mono">
        {pairToken ? (
          <div className="w-full max-w-sm px-6">
            <div className="mb-8">
              <div className="text-white text-lg">paired &quot;{pairToken.name}&quot;</div>
              <div className="text-gray-500 text-xs mt-0.5">this token is shown once</div>
            </div>
            <div className="text-xs text-gray-500 mb-3">
              install the <code className="text-gray-300">buhera-pair</code> CLI on this machine, then run:
            </div>
            <div className="flex items-start gap-2 bg-black border border-gray-800 p-3 mb-4">
              <pre className="flex-1 whitespace-pre-wrap break-all text-xs text-gray-200">
                {`buhera-pair pair --token ${pairToken.token} --gateway ${
                  (typeof window !== "undefined" && window.__BUHERA_GATEWAY_URL__) ||
                  process.env.NEXT_PUBLIC_BUHERA_GATEWAY_URL ||
                  "https://buhera-91-98-157-147.sslip.io"
                }`}
              </pre>
              <CopyButton
                text={`buhera-pair pair --token ${pairToken.token} --gateway ${
                  (typeof window !== "undefined" && window.__BUHERA_GATEWAY_URL__) ||
                  process.env.NEXT_PUBLIC_BUHERA_GATEWAY_URL ||
                  "https://buhera-91-98-157-147.sslip.io"
                }`}
              />
            </div>
            <button
              onClick={continueToApp}
              className="w-full border border-emerald-600 text-emerald-400 hover:bg-emerald-950 px-3 py-2"
            >
              continue
            </button>
          </div>
        ) : (
          <form onSubmit={onSubmit} className="w-full max-w-sm px-6">
            <div className="mb-10">
              <div className="text-white text-lg">buhera OS</div>
            </div>

            <Stripe
              label="username"
              type="email"
              autoComplete="username"
              autoFocus
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <Stripe
              label="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <Stripe
              label="device (optional)"
              type="text"
              placeholder="e.g. office-laptop"
              autoComplete="off"
              value={device}
              onChange={(e) => setDevice(e.target.value)}
            />

            {error && <div className="text-xs text-red-400 mb-4">{error}</div>}
            {pairNotice && <div className="text-xs text-yellow-400 mb-4">{pairNotice}</div>}

            <button
              type="submit"
              disabled={busy}
              className="w-full border border-emerald-600 text-emerald-400 hover:bg-emerald-950 disabled:opacity-50 px-3 py-2 mt-2"
            >
              {busy ? "signing in…" : "sign in"}
            </button>
          </form>
        )}
      </div>
    </>
  );
}
