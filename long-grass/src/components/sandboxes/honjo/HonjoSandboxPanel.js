/* ============================================================================
 * HonjoSandboxPanel — the REAL honjo (Honjo Masamune) sandbox, embedded.
 *
 * Runs the real honjo compiler in the browser (src/lib/honjo.js, a zero-dep
 * esbuild bundle: lex -> parse -> accountability check -> Cut-IR -> exact
 * interpreter). Seeded with the real `track.hj` example, whose output is the
 * causal table — a Path @ floor with converged=true and an amalgamation — plus
 * the per-entity Atom/Compound/Path detail cards.
 *
 * Renderer reproduced from borgia/honjo-masamune/src/pages/playground.js.
 * ========================================================================== */

import { useState, useCallback, useEffect } from "react";
import { evaluate, compile } from "@/lib/sandboxes/honjo/honjo";
import ValueCard, { fmt } from "./ValueCard";

const ACCENT = "#58E6D9";

// The real track.hj example — the causal table.
const TRACK_SOURCE = `-- track.hj — tracking an item through a process (the causal table)
floor 1.0
import honjo.causal

O := cut 8
H := cut 1
W := close O(H, H)

-- propagate O's uncertainty through the process; admissible iff it
-- S-entropy-converges to the observed output. representation switching allowed.
path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path              -- the amalgamation IS the result
`;

export default function HonjoSandboxPanel() {
  const [src, setSrc] = useState(TRACK_SOURCE);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runScript = useCallback((code) => {
    const source = typeof code === "string" ? code : src;
    setError(null);
    setResult(null);
    try {
      compile(source);            // front-end-only first: clean accountability errors
      setResult(evaluate(source));
    } catch (e) {
      setError(e?.message || String(e));
    }
  }, [src]);

  // Auto-run the seeded track.hj on mount.
  useEffect(() => { runScript(TRACK_SOURCE); /* eslint-disable-next-line */ }, []);

  const namedEntries = result ? Object.entries(result.named) : [];

  return (
    <div className="flex flex-col overflow-hidden rounded border border-neutral-800" style={{ height: 560, background: "#0a0a0a" }}>
      <div className="flex h-8 shrink-0 items-center gap-3 px-3" style={{ background: "#101014", borderBottom: "1px solid #262626" }}>
        <span className="font-mono text-[12px] font-bold" style={{ color: ACCENT }}>honjo</span>
        <span className="text-[11px] text-neutral-400">Honjo Masamune — track.hj (the causal table)</span>
        {result && (
          <span className="ml-auto font-mono text-[11px] text-neutral-400">
            clock M = <span style={{ color: ACCENT }}>{result.cutCount}</span>
            {" · "}floor {fmt(result.floor)}
            {" · "}
            <span style={{ color: result.ok ? "#22c55e" : "#ef4444" }}>{result.ok ? "ok" : "ABORTED"}</span>
          </span>
        )}
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-2">
        {/* Editor */}
        <div className="flex min-h-0 flex-col border-r border-neutral-800">
          <div className="flex items-center justify-between px-3 py-1.5">
            <span className="text-[10px] uppercase tracking-widest text-neutral-500">source · .hj</span>
            <button
              onClick={() => runScript(src)}
              className="rounded bg-[#58E6D9] px-3 py-0.5 text-[12px] font-medium text-[#0a0a0a] hover:brightness-110"
            >
              ▶ run
            </button>
          </div>
          <textarea
            value={src}
            onChange={(e) => setSrc(e.target.value)}
            spellCheck={false}
            className="min-h-0 flex-1 resize-none bg-neutral-950 p-3 font-mono text-[12px] leading-relaxed text-neutral-200 outline-none"
          />
        </div>

        {/* Output */}
        <div className="flex min-h-0 flex-col">
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-neutral-500">measurement</div>
          <div className="min-h-0 flex-1 space-y-3 overflow-auto bg-neutral-950 p-3">
            {error && <div className="whitespace-pre-wrap font-mono text-sm text-red-400">{error}</div>}
            {!error && !result && <div className="text-sm italic text-neutral-600">Press run to perform the cuts.</div>}
            {result && (
              <>
                {/* causal table / observation log */}
                {result.log.length > 0 && (
                  <pre className="whitespace-pre-wrap border-b border-neutral-800 pb-3 font-mono text-xs text-neutral-400">
                    {result.log.join("\n")}
                  </pre>
                )}
                {/* structured value cards */}
                <div className="grid gap-3 sm:grid-cols-2">
                  {namedEntries.map(([name, v]) => (
                    <ValueCard key={name} name={name} v={v} />
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
