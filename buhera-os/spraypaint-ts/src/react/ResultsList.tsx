// ─────────────────────────────────────────────────────────────────────────────
// ResultsList — the readable output: ranked passages grouped by scene.
//
// Read-only by nature (it is the answer), but scene headers are clickable to
// toggle that scene, so even the results list participates in the crossfilter
// loop. Honours `flat`: when the query is flat, shows one global order.
// ─────────────────────────────────────────────────────────────────────────────

import React from "react";
import type { AskQuery, AskResult, AskHit } from "../types.js";
import { invertSceneToggle, type QueryDiff } from "../crossfilter.js";

export interface ResultsListProps {
  query: AskQuery;
  result: AskResult | null;
  stale: boolean;
  onGesture: (diff: QueryDiff) => void;
}

function groupByScene(hits: AskHit[]): Map<string, AskHit[]> {
  const m = new Map<string, AskHit[]>();
  for (const h of hits) {
    const arr = m.get(h.scene) ?? [];
    arr.push(h);
    m.set(h.scene, arr);
  }
  return m;
}

function Hit({ hit }: { hit: AskHit }) {
  return (
    <div style={{ padding: "6px 12px", borderBottom: "1px solid #232323" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#858585" }}>
        <span style={{ fontFamily: "monospace", color: "#9cdcfe" }}>
          {hit.path}:{hit.start_line}-{hit.end_line}
        </span>
        <span style={{ color: "#cca700" }}>{hit.score.toFixed(2)}</span>
      </div>
      <pre style={{ margin: "3px 0 0", fontSize: 12, color: "#cccccc", whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
        {hit.snippet}
      </pre>
    </div>
  );
}

export function ResultsList({ query, result, stale, onGesture }: ResultsListProps) {
  if (!result) {
    return <div style={{ padding: 16, color: "#858585", fontSize: 13 }}>No results yet — run a query.</div>;
  }

  const wrapper: React.CSSProperties = { opacity: stale ? 0.5 : 1 };

  if (query.flat) {
    return (
      <div style={wrapper}>
        {result.results.map((h, i) => (
          <Hit key={`${h.path}:${h.start_line}:${i}`} hit={h} />
        ))}
      </div>
    );
  }

  const grouped = groupByScene(result.results);
  return (
    <div style={wrapper}>
      {[...grouped.entries()].map(([scene, hits]) => (
        <div key={scene}>
          <div
            onClick={() => {
              const diff = invertSceneToggle(query, result.allocation, scene);
              if (diff) onGesture(diff);
            }}
            style={{
              padding: "4px 12px",
              background: "#252526",
              color: "#cccccc",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              justifyContent: "space-between",
            }}
            title="click to toggle this scene"
          >
            <span>{scene}</span>
            <span style={{ color: "#858585" }}>{hits.length} passage{hits.length === 1 ? "" : "s"}</span>
          </div>
          {hits.map((h, i) => (
            <Hit key={`${h.path}:${h.start_line}:${i}`} hit={h} />
          ))}
        </div>
      ))}
    </div>
  );
}
