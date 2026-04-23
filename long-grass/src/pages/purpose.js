import Head from "next/head";
import { useState } from "react";

const EXAMPLES = [
  "Tell me about SOD1",
  "Describe the protein TP53",
  "what is BRCA1",
];

function ValueRenderer({ value, depth = 0 }) {
  if (value === null || value === undefined) {
    return <span className="text-gray-500 italic">null</span>;
  }
  if (typeof value === "boolean") {
    return <span className="text-amber-600">{String(value)}</span>;
  }
  if (typeof value === "number") {
    return <span className="text-sky-700">{value}</span>;
  }
  if (typeof value === "string") {
    return <span className="text-emerald-800">{value}</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-gray-500">[]</span>;
    return (
      <ul className="ml-4 list-disc">
        {value.map((v, i) => (
          <li key={i}>
            <ValueRenderer value={v} depth={depth + 1} />
          </li>
        ))}
      </ul>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) return <span className="text-gray-500">{`{}`}</span>;
    return (
      <dl className="ml-4">
        {entries.map(([k, v]) => (
          <div key={k} className="flex gap-3 py-0.5">
            <dt className="text-gray-600 font-medium min-w-[9rem]">{k}</dt>
            <dd className="flex-1 break-words">
              <ValueRenderer value={v} depth={depth + 1} />
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return <span>{String(value)}</span>;
}

export default function PurposePage() {
  const [utterance, setUtterance] = useState("Tell me about SOD1");
  const [mode, setMode] = useState("value");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(null);

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(null);
    try {
      const resp = await fetch("/api/purpose", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ utterance, mode }),
      });
      const body = await resp.json();
      if (!body.ok) {
        setError(body.error + (body.stderr ? `\n\n${body.stderr}` : ""));
      } else {
        setResult(mode === "fragment" ? body.fragment : body.value);
        setElapsed(body.elapsed_ms);
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Head>
        <title>Purpose · Buhera</title>
        <meta
          name="description"
          content="Compile a natural-language utterance to a typed vaHera fragment and execute it against registered providers."
        />
      </Head>
      <main className="min-h-screen bg-white text-gray-900 px-6 py-10 md:px-16">
        <div className="mx-auto max-w-4xl">
          <header className="mb-8">
            <h1 className="text-3xl font-semibold mb-2">Purpose</h1>
            <p className="text-gray-600 text-sm leading-relaxed">
              Compile a natural-language utterance into a typed vaHera fragment
              and execute it against the registered Purpose domain providers.
              The backend is the Rust <code className="text-xs bg-gray-100 px-1 rounded">purpose</code> CLI,
              spawned per request; no domain content lives in this page.
            </p>
          </header>

          <form onSubmit={submit} className="mb-6">
            <label htmlFor="utterance" className="block text-sm font-medium mb-2">
              Utterance
            </label>
            <div className="flex gap-2">
              <input
                id="utterance"
                type="text"
                value={utterance}
                onChange={(e) => setUtterance(e.target.value)}
                className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
                placeholder="Tell me about SOD1"
                disabled={loading}
              />
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="border border-gray-300 rounded px-2 py-2 text-sm"
                disabled={loading}
                aria-label="output mode"
              >
                <option value="value">execute</option>
                <option value="fragment">compile only</option>
              </select>
              <button
                type="submit"
                disabled={loading || !utterance.trim()}
                className="bg-gray-900 text-white rounded px-4 py-2 text-sm disabled:opacity-50"
              >
                {loading ? "running…" : "submit"}
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((e) => (
                <button
                  key={e}
                  type="button"
                  onClick={() => setUtterance(e)}
                  className="text-xs text-gray-600 border border-gray-200 rounded px-2 py-1 hover:bg-gray-50"
                >
                  {e}
                </button>
              ))}
            </div>
          </form>

          <section>
            {error && (
              <div className="border border-red-200 bg-red-50 rounded px-4 py-3 mb-4">
                <div className="text-red-800 text-sm font-medium mb-1">Error</div>
                <pre className="text-xs text-red-900 whitespace-pre-wrap">{error}</pre>
              </div>
            )}

            {result !== null && (
              <div className="border border-gray-200 rounded px-4 py-3">
                <div className="flex items-baseline justify-between mb-3">
                  <div className="text-xs text-gray-500 uppercase tracking-wide">
                    {mode === "fragment" ? "compiled vaHera fragment" : "executed Value"}
                  </div>
                  {elapsed !== null && (
                    <div className="text-xs text-gray-500">{elapsed} ms</div>
                  )}
                </div>
                <div className="text-sm">
                  <ValueRenderer value={result} />
                </div>
                <details className="mt-4">
                  <summary className="text-xs text-gray-500 cursor-pointer">
                    raw JSON
                  </summary>
                  <pre className="mt-2 text-xs bg-gray-50 rounded p-2 overflow-x-auto">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </details>
              </div>
            )}

            {!error && result === null && !loading && (
              <p className="text-xs text-gray-500">
                No query submitted yet. The CLI spawns on submit; responses
                are rendered above.
              </p>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
