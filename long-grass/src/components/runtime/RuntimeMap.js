/* ============================================================================
 * RuntimeMap — the causal knowledge graph as a transit map (a Netzplan).
 *
 * The runs are the lines: each run passes through the nodes its script
 * described, in order. The nodes are the stations — one per τ, so a τ two
 * runs touched is one interchange both lines pass through (the runtime's
 * convergence, drawn). A station's mark says what its chunks have done:
 *
 *   ● ring, filled centre   emitted values (facts)
 *   ◐ half-filled           emitted an error value (the run went on)
 *   ◌ dashed ring           described, nothing has run on it yet
 *
 * Thin dotted arcs are run-induced edges: a reader that actually read one
 * node from another during a carry. Nothing here judges a value.
 *
 * Visual grammar from the "CKG Runtime Netzplan" artifact, re-inked for the
 * black surface; line colours are the dataviz palette's dark categorical
 * slots (identity), capped at eight — older runs past that fold to grey.
 * ========================================================================== */

import { useMemo } from "react";

const SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const INK = { ink: "#e8ecef", muted: "#898781", rule: "#2a3238", stop: "#000000", old: "#4a4946" };
const COL = 150;
const ROW = 96;
const PAD = { x: 70, y: 72 }; // y leaves room for stacked line badges above the first row

function stationState(node) {
  if (!node) return "planned";
  const facts = node.facts || [];
  if (facts.some((f) => f.predicate === "error")) return "partial";
  if (facts.length > 0) return "done";
  return "planned";
}

function layout(runs, graph) {
  const byTau = new Map((graph?.nodes || []).map((n) => [n.tau, n]));
  const pos = new Map(); // tau → { x, y, lines: [] }
  let col = 0;
  runs.forEach((run, r) => {
    run.taus.forEach((tau) => {
      if (!pos.has(tau)) pos.set(tau, { x: PAD.x + col++ * COL, y: PAD.y + r * ROW, lines: [] });
      pos.get(tau).lines.push(run.id);
    });
  });
  // Nodes represented outside any player run (tutorial cells, dispatch("ckg")).
  const loose = (graph?.nodes || []).filter((n) => !pos.has(n.tau));
  loose.forEach((n) => pos.set(n.tau, { x: PAD.x + col++ * COL, y: PAD.y + runs.length * ROW, lines: [] }));
  const width = Math.max(PAD.x * 2 + Math.max(0, col - 1) * COL, 360);
  const height = PAD.y * 2 + Math.max(0, runs.length - (loose.length ? 0 : 1)) * ROW + 30;
  return { pos, byTau, width, height, hasLoose: loose.length > 0 };
}

// Transit routing: leave a station along its row, then take a 45° diagonal
// into the next one. A straight line between two stations can pass exactly
// over a third station it does not stop at; an elbow does not.
function routeTo(p, q) {
  const dy = q.y - p.y;
  if (dy === 0) return `L${q.x},${q.y}`;
  const dx = q.x - p.x;
  const diag = Math.min(Math.abs(dy), Math.abs(dx));
  const elbowX = q.x - Math.sign(dx || 1) * diag;
  return `L${elbowX},${p.y} L${q.x},${q.y}`;
}

function Station({ x, y, state, hub }) {
  if (hub) {
    return (
      <g>
        <rect x={x - 15} y={y - 11} width={30} height={22} rx={8} fill={INK.stop} stroke={INK.ink} strokeWidth={3} />
        <circle cx={x} cy={y} r={3.5} fill={state === "planned" ? "none" : INK.ink} />
      </g>
    );
  }
  const r = 8;
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={INK.stop} stroke={INK.ink} strokeWidth={3}
        strokeDasharray={state === "planned" ? "3.2 2.4" : "none"} />
      {state === "done" && <circle cx={x} cy={y} r={r * 0.42} fill={INK.ink} />}
      {state === "partial" && <path d={`M${x - r},${y} A${r},${r} 0 0 0 ${x + r},${y} Z`} fill={INK.ink} />}
    </g>
  );
}

export default function RuntimeMap({ runs = [], graph = null, highlight = null, showDetails = true }) {
  const { pos, byTau, width, height } = useMemo(() => layout(runs, graph), [runs, graph]);
  const colorOf = (i) => (runs.length - i > SLOTS.length ? INK.old : SLOTS[i % SLOTS.length]);
  const edges = graph?.edges || [];
  // Runs that start at the same station stack their badges instead of
  // hiding one another.
  const badgeOffset = useMemo(() => {
    const seen = new Map();
    return runs.map((run) => {
      const first = run.taus[0];
      const k = seen.get(first) || 0;
      seen.set(first, k + 1);
      return k;
    });
  }, [runs]);

  if (!runs.length && !(graph?.nodes || []).length) {
    return (
      <p className="text-gray-500 italic">
        the runtime is empty — write something on the blank screen, and the script it becomes will lay the first line.
      </p>
    );
  }

  return (
    <div>
      <div className="overflow-x-auto no-scrollbar">
        <svg width={width} height={height} role="img" aria-label="runtime map: runs as lines through the nodes they described">
          {/* lines */}
          {runs.map((run, i) => {
            const pts = run.taus.map((t) => pos.get(t)).filter(Boolean);
            if (!pts.length) return null;
            const dim = highlight && highlight !== run.id;
            const d = pts.map((p, k) => (k ? routeTo(pts[k - 1], p) : `M${p.x},${p.y}`)).join(" ");
            const attached = run.spawned?.some((s) => s.attached);
            return (
              <g key={run.id} opacity={dim ? 0.18 : 1} style={{ transition: "opacity 200ms" }}>
                {pts.length > 1 && (
                  <path d={d} fill="none" stroke={colorOf(i)} strokeWidth={8} strokeLinecap="round" strokeLinejoin="round"
                    strokeDasharray={attached ? "none" : "13 7"} />
                )}
                <g transform={`translate(${pts[0].x - 46},${pts[0].y - badgeOffset[i] * 26})`}>
                  <rect x={-20} y={-11} width={40} height={22} rx={5} fill={colorOf(i)} />
                  <text x={0} y={4} textAnchor="middle" fontSize="11" fontWeight="700" fill="#000">{run.id.replace("run-", "R")}</text>
                </g>
              </g>
            );
          })}
          {/* run-induced edges: a read that landed */}
          {edges.map((e, i) => {
            const a = pos.get(e.from);
            const b = pos.get(e.to);
            if (!a || !b) return null;
            const mx = (a.x + b.x) / 2;
            const my = Math.min(a.y, b.y) - 34;
            return <path key={i} d={`M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`} fill="none" stroke={INK.muted} strokeWidth={1.5} strokeDasharray="2 4" />;
          })}
          {/* stations and labels */}
          {[...pos.entries()].map(([tau, p]) => {
            const node = byTau.get(tau);
            const dim = highlight && !p.lines.includes(highlight);
            return (
              <g key={tau} opacity={dim ? 0.25 : 1}>
                <Station x={p.x} y={p.y} state={stationState(node)} hub={p.lines.length > 1} />
                <text x={p.x} y={p.y + 28} textAnchor="middle" fontSize="12" fontWeight="600" fill={INK.ink}
                  style={{ paintOrder: "stroke", stroke: "#000", strokeWidth: 4, strokeLinejoin: "round" }}>
                  {tau.length > 18 ? `${tau.slice(0, 17)}…` : tau}
                </text>
                {node?.facts?.length > 0 && (
                  <text x={p.x} y={p.y + 42} textAnchor="middle" fontSize="10" fill={INK.muted}>
                    {node.facts.length} value{node.facts.length === 1 ? "" : "s"}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[11px]" style={{ color: INK.muted }}>
        <span>● emitted values</span>
        <span>◐ emitted an error, went on</span>
        <span>◌ described, nothing ran yet</span>
        <span>▭ interchange: one τ, several runs</span>
        <span>┄ dashed line: the run spawned no module</span>
      </div>

      {showDetails && (
        <div className="mt-6 grid grid-cols-2 lg:grid-cols-1 gap-x-10 gap-y-5 text-xs">
          <div>
            <div className="mb-2" style={{ color: INK.muted }}>lines</div>
            {runs.map((run, i) => (
              <div key={run.id} className="mb-3">
                <div className="flex items-baseline gap-2">
                  <span className="inline-block px-1.5 rounded text-[10px] font-bold text-black" style={{ background: colorOf(i) }}>
                    {run.id.replace("run-", "R")}
                  </span>
                  <span className="text-gray-300">{run.words}</span>
                  <span className="text-gray-600">· {run.by === "model" ? "your model wrote it" : "you wrote it"}</span>
                </div>
                {(run.spawned || []).map((s, k) => (
                  <div key={k} className="ml-10 text-gray-500">
                    spawn {s.program} from {s.target}
                    {s.attached ? <span className="text-gray-400"> → chunk attached</span> : <span> — {s.note}</span>}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div>
            <div className="mb-2" style={{ color: INK.muted }}>stations</div>
            {[...pos.keys()].map((tau) => {
              const n = byTau.get(tau);
              return (
                <div key={tau} className="mb-2">
                  <span className="text-gray-200">{tau}</span>
                  <span className="text-gray-600"> · {(n?.address || []).join("/") || "—"}</span>
                  {(n?.facts || []).map((f, k) => (
                    <div key={k} className="ml-4 text-gray-500">
                      {f.predicate}: {f.object?.findings?.headline || f.object?.delta?.kind || (typeof f.object === "string" ? f.object : "value")}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
