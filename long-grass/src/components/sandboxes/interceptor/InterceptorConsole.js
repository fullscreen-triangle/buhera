// =====================================================================
//  InterceptorConsole — presentational renderer for the interceptor
//  module's results: generated code, captured stdout/stderr from a
//  sandboxed run, and a wind-tunnel stability readout. Pure props-in,
//  no data fetching of its own — same convention as
//  sandboxes/spraypaint/SpraypaintAllocationChart.js.
// =====================================================================
import { useState } from "react";

const OK = "#58E6D9";
const BAD = "#f87171";
const DIM = "#9ca3af";

function regimeColor(name) {
  switch (name) {
    case "Phase-locked":
    case "Synchronized":
      return OK;
    case "Partially-locked":
      return "#facc15";
    default:
      return BAD;
  }
}

export function CodeBlock({ language, code }) {
  const [expanded, setExpanded] = useState(true);
  if (!code) return null;
  return (
    <div className="rounded-md border border-neutral-700 bg-[#151515] p-2 mb-2">
      <button
        type="button"
        className="mb-1 px-1 text-[11px] uppercase tracking-wider text-neutral-400 w-full text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        generated {language} {expanded ? "▾" : "▸"}
      </button>
      {expanded && (
        <pre className="p-2 bg-black/40 border border-gray-800 rounded text-xs font-mono whitespace-pre-wrap break-all text-teal-100">
          {code}
        </pre>
      )}
    </div>
  );
}

export function ConsoleOutput({ ok, stdout, stderr, exit_code, elapsed_ms, timed_out, truncated }) {
  return (
    <div className="rounded-md border border-neutral-700 bg-[#0c0c0c] p-2 mb-2">
      <div className="mb-1 px-1 text-[11px] uppercase tracking-wider text-neutral-400 flex items-center gap-2">
        <span>console output</span>
        <span style={{ color: ok ? OK : BAD }}>{ok ? "● ok" : "● failed"}</span>
        {typeof exit_code === "number" && <span className="text-neutral-500">exit {exit_code}</span>}
        {typeof elapsed_ms === "number" && <span className="text-neutral-500">{elapsed_ms} ms</span>}
        {timed_out && <span className="text-yellow-400">timed out</span>}
        {truncated && <span className="text-yellow-400">truncated</span>}
      </div>
      {stdout ? (
        <pre className="p-2 bg-black/60 border border-gray-800 rounded text-xs font-mono whitespace-pre-wrap break-all text-gray-200">
          {stdout}
        </pre>
      ) : (
        <p className="text-gray-600 text-xs px-1">(no stdout)</p>
      )}
      {stderr && (
        <pre className="mt-1 p-2 bg-black/60 border border-red-900/50 rounded text-xs font-mono whitespace-pre-wrap break-all text-red-300">
          {stderr}
        </pre>
      )}
    </div>
  );
}

export function WindTunnelReadout({ runs, order_parameter, regime, crash_count, per_run }) {
  if (!regime) return null;
  const pct = Math.round((order_parameter ?? 0) * 100);
  return (
    <div className="rounded-md border border-neutral-700 bg-[#151515] p-2 mb-2">
      <div className="mb-1 px-1 text-[11px] uppercase tracking-wider text-neutral-400">
        wind-tunnel stability — {runs} run{runs === 1 ? "" : "s"}
      </div>
      <div className="px-1 flex items-baseline gap-3">
        <span className="text-lg font-mono" style={{ color: regimeColor(regime.name) }}>
          {regime.name}
        </span>
        <span className="text-sm text-gray-400">R = {pct}%</span>
        {crash_count > 0 && <span className="text-sm text-red-400">{crash_count} crash{crash_count === 1 ? "" : "es"}</span>}
      </div>
      <p className="px-1 text-xs text-gray-500 mt-1">{regime.note}</p>
      {Array.isArray(per_run) && per_run.some((r) => !r.matches_reference) && (
        <div className="mt-2 px-1 text-xs text-gray-500">
          <span className="text-gray-400">deviations: </span>
          {per_run
            .filter((r) => !r.matches_reference)
            .map((r) => `run ${r.index} (holonomy ${r.holonomy.toFixed(2)})`)
            .join(", ")}
        </div>
      )}
    </div>
  );
}
