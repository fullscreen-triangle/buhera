import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import { useGatewaySession } from "@/lib/auth/useGatewaySession";

export default function Login() {
  const router = useRouter();
  const { loggedIn, checked, login } = useGatewaySession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (checked && loggedIn) {
      const dest = typeof router.query.next === "string" ? router.query.next : "/";
      router.replace(dest);
    }
  }, [checked, loggedIn, router]);

  async function onSubmit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    const res = await login(email.trim(), password);
    setBusy(false);
    if (!res.ok) {
      setError(res.error || "login failed");
      return;
    }
    const dest = typeof router.query.next === "string" ? router.query.next : "/";
    router.replace(dest);
  }

  return (
    <>
      <Head>
        <title>buhera — sign in</title>
      </Head>
      <div className="fixed inset-0 bg-black text-gray-300 flex items-center justify-center font-mono text-sm">
        <form onSubmit={onSubmit} className="w-full max-w-sm px-6">
          <div className="mb-8">
            <div className="text-white text-lg">buhera OS</div>
            <div className="text-gray-500 text-xs mt-0.5">sign in to continue</div>
          </div>

          <label className="block text-xs text-gray-500 mb-1">email</label>
          <input
            type="email"
            autoComplete="username"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-3 py-2 mb-4 text-gray-200"
            required
          />

          <label className="block text-xs text-gray-500 mb-1">password</label>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-3 py-2 mb-4 text-gray-200"
            required
          />

          {error && (
            <div className="text-xs text-red-400 mb-4">{error}</div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full border border-emerald-600 text-emerald-400 hover:bg-emerald-950 disabled:opacity-50 px-3 py-2"
          >
            {busy ? "signing in…" : "sign in"}
          </button>
        </form>
      </div>
    </>
  );
}
