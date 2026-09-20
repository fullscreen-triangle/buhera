import { useEffect, useState } from "react";
import Head from "next/head";
import Link from "next/link";
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

function PairForm({ onPaired }) {
  const [name, setName] = useState("");
  const [capabilities, setCapabilities] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    if (busy || !name.trim()) return;
    setBusy(true);
    setError(null);
    const caps = capabilities
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    const res = await gatewayModule.execute({ kind: "pair", name: name.trim(), capabilities: caps });
    setBusy(false);
    if (!res.ok) {
      setError(res.error || "pairing failed");
      return;
    }
    onPaired(res.output_delta);
    setName("");
    setCapabilities("");
  }

  return (
    <form onSubmit={onSubmit} className="mb-8 border border-gray-800 p-4">
      <div className="text-xs text-gray-500 mb-3">pair a new machine</div>
      <label className="block text-xs text-gray-500 mb-1">machine name</label>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. office"
        className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-3 py-2 mb-3 text-gray-200 text-sm"
        required
      />
      <label className="block text-xs text-gray-500 mb-1">capabilities (comma-separated, optional)</label>
      <input
        value={capabilities}
        onChange={(e) => setCapabilities(e.target.value)}
        placeholder="e.g. kernel, vahera"
        className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-3 py-2 mb-3 text-gray-200 text-sm"
      />
      {error && <div className="text-xs text-red-400 mb-3">{error}</div>}
      <button
        type="submit"
        disabled={busy}
        className="border border-emerald-600 text-emerald-400 hover:bg-emerald-950 disabled:opacity-50 px-3 py-1.5 text-sm"
      >
        {busy ? "pairing…" : "pair"}
      </button>
    </form>
  );
}

function FreshToken({ result }) {
  const gateway =
    (typeof window !== "undefined" && window.__BUHERA_GATEWAY_URL__) ||
    process.env.NEXT_PUBLIC_BUHERA_GATEWAY_URL ||
    "https://buhera-91-98-157-147.sslip.io";
  const command = `buhera-pair pair --token ${result.token} --gateway ${gateway}`;
  return (
    <div className="mb-8 border border-emerald-800 p-4">
      <div className="text-xs text-emerald-400 mb-2">
        paired &quot;{result.name}&quot; — this token is shown once
      </div>
      <div className="text-xs text-gray-500 mb-3">
        install the <code className="text-gray-300">buhera-pair</code> CLI on that machine, then run:
      </div>
      <div className="flex items-start gap-2 bg-black border border-gray-800 p-3">
        <pre className="flex-1 whitespace-pre-wrap break-all text-xs text-gray-200">{command}</pre>
        <CopyButton text={command} />
      </div>
      <div className="text-xs text-gray-600 mt-3">
        expires {new Date(result.expires_at * 1000).toLocaleString()}
      </div>
    </div>
  );
}

function MachineList({ machines, busy, onUnpair }) {
  if (!machines) return <div className="text-xs text-gray-600">loading…</div>;
  if (machines.length === 0) {
    return <div className="text-xs text-gray-600">no machines paired yet.</div>;
  }
  return (
    <ul className="space-y-2">
      {machines.map((m) => (
        <li key={m.name} className="flex items-center justify-between border border-gray-800 px-3 py-2 text-sm">
          <div>
            <span className="text-gray-200">{m.name}</span>{" "}
            <span className={m.live ? "text-emerald-400" : "text-gray-600"}>
              {m.live ? "live" : "asleep"}
            </span>
            {m.capabilities?.length > 0 && (
              <span className="text-gray-600"> — [{m.capabilities.join(", ")}]</span>
            )}
          </div>
          <button
            disabled={busy === m.name}
            onClick={() => onUnpair(m.name)}
            className="text-xs text-red-400 hover:text-red-300 underline disabled:opacity-50"
          >
            {busy === m.name ? "unpairing…" : "unpair"}
          </button>
        </li>
      ))}
    </ul>
  );
}

export default function Pair() {
  const [machines, setMachines] = useState(null);
  const [lastToken, setLastToken] = useState(null);
  const [unpairing, setUnpairing] = useState(null);

  async function reload() {
    const res = await gatewayModule.execute("catalysts");
    if (res.ok) setMachines(res.output_delta.entries);
  }

  useEffect(() => {
    reload();
  }, []);

  async function onUnpair(name) {
    setUnpairing(name);
    await gatewayModule.execute({ kind: "unpair", name });
    setUnpairing(null);
    reload();
  }

  return (
    <>
      <Head>
        <title>buhera — pair a machine</title>
      </Head>
      <div className="fixed inset-0 bg-black text-gray-300 overflow-y-auto font-mono text-sm px-16 py-10 md:px-8 md:py-6">
        <div className="fixed top-3 right-4 z-10">
          <Link href="/" className="text-xs text-green-400 hover:text-green-300 no-underline">
            ▶ back to terminal
          </Link>
        </div>
        <div className="max-w-xl mx-auto">
          <div className="mb-8">
            <div className="text-white text-lg">pair a machine</div>
            <div className="text-gray-500 text-xs mt-0.5">
              connect a local install to this account with a one-time token.
            </div>
          </div>

          <PairForm
            onPaired={(delta) => {
              setLastToken(delta);
              reload();
            }}
          />

          {lastToken && <FreshToken result={lastToken} />}

          <div className="text-xs text-gray-500 mb-3">paired machines</div>
          <MachineList machines={machines} busy={unpairing} onUnpair={onUnpair} />
        </div>
      </div>
    </>
  );
}
