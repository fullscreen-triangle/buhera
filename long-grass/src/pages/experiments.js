import { useEffect, useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { gatewayModule } from "@/lib/modules/gateway-module";

function CreateForm({ onCreated }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    if (busy || !name.trim()) return;
    setBusy(true);
    setError(null);
    const res = await gatewayModule.execute({ kind: "create_experiment", name: name.trim() });
    setBusy(false);
    if (!res.ok) {
      setError(res.error || "could not create experiment");
      return;
    }
    onCreated();
    setName("");
  }

  return (
    <form onSubmit={onSubmit} className="mb-8 border border-gray-800 p-4">
      <div className="text-xs text-gray-500 mb-3">start a new experiment</div>
      <label className="block text-xs text-gray-500 mb-1">name</label>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. nfdi4cat-catalysis-2026"
        className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-3 py-2 mb-3 text-gray-200 text-sm"
        required
      />
      {error && <div className="text-xs text-red-400 mb-3">{error}</div>}
      <button
        type="submit"
        disabled={busy}
        className="border border-emerald-600 text-emerald-400 hover:bg-emerald-950 disabled:opacity-50 px-3 py-1.5 text-sm"
      >
        {busy ? "creating…" : "create"}
      </button>
    </form>
  );
}

function GrantForm({ experimentId, onGranted }) {
  const [email, setEmail] = useState("");
  const [capabilities, setCapabilities] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    if (busy || !email.trim()) return;
    setBusy(true);
    setError(null);
    const caps = capabilities.split(",").map((c) => c.trim()).filter(Boolean);
    const res = await gatewayModule.execute({
      kind: "grant",
      experiment: experimentId,
      email: email.trim(),
      capabilities: caps,
    });
    setBusy(false);
    if (!res.ok) {
      setError(res.error || "could not grant");
      return;
    }
    onGranted();
    setEmail("");
    setCapabilities("");
  }

  return (
    <form onSubmit={onSubmit} className="mt-4 border border-gray-800 p-3">
      <div className="text-xs text-gray-500 mb-2">invite a collaborator</div>
      <input
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="their account email"
        className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-2 py-1.5 mb-2 text-gray-200 text-xs"
        required
      />
      <input
        value={capabilities}
        onChange={(e) => setCapabilities(e.target.value)}
        placeholder="capabilities, comma-separated (e.g. vahera, sbs-core)"
        className="w-full bg-transparent border border-gray-700 focus:border-emerald-500 outline-none px-2 py-1.5 mb-2 text-gray-200 text-xs"
      />
      {error && <div className="text-xs text-red-400 mb-2">{error}</div>}
      <button
        type="submit"
        disabled={busy}
        className="border border-emerald-700 text-emerald-400 hover:bg-emerald-950 disabled:opacity-50 px-2 py-1 text-xs"
      >
        {busy ? "granting…" : "grant"}
      </button>
    </form>
  );
}

function Roster({ experimentId, refreshKey }) {
  const [grants, setGrants] = useState(null);
  const [revoking, setRevoking] = useState(null);

  async function reload() {
    const res = await gatewayModule.execute({ kind: "grants", experiment: experimentId });
    if (res.ok) setGrants(res.output_delta.entries);
  }

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, refreshKey]);

  async function onRevoke(accountId) {
    setRevoking(accountId);
    await gatewayModule.execute({ kind: "revoke_grant", experiment: experimentId, account_id: accountId });
    setRevoking(null);
    reload();
  }

  if (!grants) return <div className="text-xs text-gray-600 mt-2">loading roster…</div>;
  if (grants.length === 0) return <div className="text-xs text-gray-600 mt-2">no grants yet.</div>;

  return (
    <ul className="mt-2 space-y-1">
      {grants.map((g) => (
        <li key={g.account_id} className="flex items-center justify-between text-xs border border-gray-800 px-2 py-1.5">
          <div>
            <span className="text-gray-200 font-mono">{g.account_id}</span>
            <span className="text-gray-600"> — [{g.capabilities.join(", ") || "no capabilities"}]</span>
          </div>
          <button
            disabled={revoking === g.account_id}
            onClick={() => onRevoke(g.account_id)}
            className="text-red-400 hover:text-red-300 underline disabled:opacity-50"
          >
            {revoking === g.account_id ? "revoking…" : "revoke"}
          </button>
        </li>
      ))}
    </ul>
  );
}

function ExperimentCard({ experiment, onChanged }) {
  const [expanded, setExpanded] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const isOwner = experiment.standing?.kind === "owner";

  return (
    <li className="border border-gray-800 p-3 mb-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-gray-200">{experiment.name}</span>{" "}
          <span className="text-gray-600 text-xs">({experiment.id})</span>
        </div>
        {isOwner ? (
          <span className="text-emerald-400 text-xs">owner</span>
        ) : (
          <span className="text-gray-500 text-xs">
            [{(experiment.standing?.capabilities || []).join(", ") || "no capabilities"}]
          </span>
        )}
      </div>
      {isOwner && (
        <>
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 text-xs text-gray-500 hover:text-gray-300 underline"
          >
            {expanded ? "hide roster" : "manage roster"}
          </button>
          {expanded && (
            <>
              <Roster experimentId={experiment.id} refreshKey={refreshKey} />
              <GrantForm
                experimentId={experiment.id}
                onGranted={() => {
                  setRefreshKey((k) => k + 1);
                  onChanged();
                }}
              />
            </>
          )}
        </>
      )}
    </li>
  );
}

export default function Experiments() {
  const [experiments, setExperiments] = useState(null);

  async function reload() {
    const res = await gatewayModule.execute("experiments");
    if (res.ok) setExperiments(res.output_delta.entries);
  }

  useEffect(() => {
    reload();
  }, []);

  return (
    <>
      <Head>
        <title>buhera — experiments</title>
      </Head>
      <div className="fixed inset-0 bg-black text-gray-300 overflow-y-auto font-mono text-sm px-16 py-10 md:px-8 md:py-6">
        <div className="fixed top-3 right-4 z-10">
          <Link href="/" className="text-xs text-green-400 hover:text-green-300 no-underline">
            ▶ back to terminal
          </Link>
        </div>
        <div className="max-w-xl mx-auto">
          <div className="mb-8">
            <div className="text-white text-lg">experiments</div>
            <div className="text-gray-500 text-xs mt-0.5">
              a shared dispatch scope: create one, grant collaborators a capped set of
              capabilities, and everyone granted shares one module federation and one
              audit log. Who acted is provenance on each act, never a boundary on what
              the experiment contains.
            </div>
          </div>

          <CreateForm onCreated={reload} />

          <div className="text-xs text-gray-500 mb-3">yours (owned + granted)</div>
          {!experiments ? (
            <div className="text-xs text-gray-600">loading…</div>
          ) : experiments.length === 0 ? (
            <div className="text-xs text-gray-600">no experiments yet.</div>
          ) : (
            <ul>
              {experiments.map((e) => (
                <ExperimentCard key={e.id} experiment={e} onChanged={reload} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </>
  );
}
