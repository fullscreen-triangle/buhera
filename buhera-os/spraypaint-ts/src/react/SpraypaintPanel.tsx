// ─────────────────────────────────────────────────────────────────────────────
// SpraypaintPanel — the whole loop as one component.
//
// Owns a SpraypaintSession (via useSpraypaintSession) and wires the three input
// surfaces to the one AskQuery:
//   • QueryBar        — prompt/editor surface (type text, set -k, Run)
//   • AllocationChart — crossfilter surface (click scenes, drag p*)
//   • ResultsList     — output, whose scene headers also toggle scenes
//   • IdentityBadge   — the invariants, made visible
//
// A gesture from any surface calls applyAndRun → the query advances, the binary
// re-runs, the charts redraw. Undo/redo walks the unified history where all three
// sources are peers. This is the reference wiring; embed it, or copy its wiring
// into a bespoke layout.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useCallback, useEffect, useState } from "react";
import type { AskQuery, Identity } from "../types.js";
import { DEFAULT_QUERY } from "../types.js";
import type { SpraypaintClient } from "../client.js";
import { editQueryText, setBudget, invertFlatToggle, type QueryDiff } from "../crossfilter.js";
import { useSpraypaintSession } from "./useSpraypaintSession.js";
import { AllocationChart } from "./AllocationChart.js";
import { ResultsList } from "./ResultsList.js";
import { IdentityBadge } from "./IdentityBadge.js";

export interface SpraypaintPanelProps {
  client: SpraypaintClient;
  initialQuery?: AskQuery;
  /** Run once on mount if the initial query has text. Default true. */
  autoRunOnMount?: boolean;
}

export function SpraypaintPanel({
  client,
  initialQuery = DEFAULT_QUERY,
  autoRunOnMount = true,
}: SpraypaintPanelProps) {
  const session = useSpraypaintSession(client, initialQuery);
  const { state, running, error, run, applyAndRun, applyGesture, undo, redo, canUndo, canRedo } = session;
  const { query, result, stale } = state;

  const [identity, setIdentity] = useState<Identity | null>(null);
  const [draftText, setDraftText] = useState(initialQuery.query);

  // Load identity once (and refresh after each run — count/χ can move).
  const refreshIdentity = useCallback(() => {
    client.identity().then(setIdentity).catch(() => setIdentity(null));
  }, [client]);

  useEffect(() => {
    refreshIdentity();
    if (autoRunOnMount && initialQuery.query.trim().length > 0) {
      void run();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!running && result) refreshIdentity();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, running]);

  // Keep the draft text in sync when undo/redo changes the query underneath.
  useEffect(() => {
    setDraftText(query.query);
  }, [query.query]);

  const onGesture = useCallback(
    (diff: QueryDiff) => {
      void applyAndRun(diff, "crossfilter");
    },
    [applyAndRun],
  );

  const submitText = useCallback(() => {
    if (draftText !== query.query) applyGesture(editQueryText(draftText), "editor");
    void run();
  }, [draftText, query.query, applyGesture, run]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#1e1e1e", color: "#cccccc", fontFamily: "system-ui, sans-serif" }}>
      {/* QueryBar */}
      <div style={{ display: "flex", gap: 8, padding: 8, borderBottom: "1px solid #232323", alignItems: "center" }}>
        <input
          value={draftText}
          onChange={(e) => setDraftText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitText(); }}
          placeholder="search intent…  (Ctrl+Enter to run)"
          style={{ flex: 1, background: "#2d2d2d", color: "#cccccc", border: "1px solid #3c3c3c", borderRadius: 4, padding: "6px 8px", fontSize: 13 }}
        />
        <label style={{ fontSize: 11, color: "#858585", display: "flex", alignItems: "center", gap: 4 }}>
          -k
          <input
            type="number"
            min={1}
            value={query.budget}
            onChange={(e) => applyGesture(setBudget(Number(e.target.value) || 1), "editor")}
            style={{ width: 56, background: "#2d2d2d", color: "#cccccc", border: "1px solid #3c3c3c", borderRadius: 4, padding: "4px 6px", fontSize: 12 }}
          />
        </label>
        <button onClick={() => applyGesture(invertFlatToggle(query), "editor")} style={btn(query.flat)}>
          {query.flat ? "flat" : "grouped"}
        </button>
        <button onClick={undo} disabled={!canUndo} style={btn(false, !canUndo)}>undo</button>
        <button onClick={redo} disabled={!canRedo} style={btn(false, !canRedo)}>redo</button>
        <button onClick={submitText} disabled={running} style={btn(true, running)}>
          {running ? "running…" : "Run"}
        </button>
      </div>

      {error && (
        <div style={{ padding: "6px 12px", background: "#5a1d1d", color: "#f2b8b8", fontSize: 12 }}>{error}</div>
      )}

      {/* body: chart left, results right */}
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <div style={{ width: "52%", borderRight: "1px solid #232323", overflow: "auto" }}>
          <AllocationChart query={query} result={result} stale={stale} onGesture={onGesture} />
        </div>
        <div style={{ flex: 1, overflow: "auto" }}>
          <ResultsList query={query} result={result} stale={stale} onGesture={onGesture} />
        </div>
      </div>

      <IdentityBadge result={result} identity={identity} />
    </div>
  );
}

function btn(active: boolean, disabled = false): React.CSSProperties {
  return {
    background: active ? "#094771" : "#2d2d2d",
    color: disabled ? "#666" : "#cccccc",
    border: "1px solid #3c3c3c",
    borderRadius: 4,
    padding: "6px 10px",
    fontSize: 12,
    cursor: disabled ? "default" : "pointer",
    opacity: disabled ? 0.6 : 1,
  };
}
