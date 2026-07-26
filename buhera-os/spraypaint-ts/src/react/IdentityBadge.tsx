// ─────────────────────────────────────────────────────────────────────────────
// IdentityBadge — the novel invariants, made visible.
//
// These are the quantities with no pre-existing intuition (the reason the loop
// has to be teachable). Rather than hide them, we surface them compactly:
//   • χ (char invariant) ≥ floor  — conserved identity (Inv 1)
//   • committed count             — never-resetting history (Inv 2)
//   • fingerprint                 — self-graph identity (Inv 1)
//   • p* and budget from the last ask — the water-filling price
// Fed directly from real AskResult + Identity JSON.
// ─────────────────────────────────────────────────────────────────────────────

import React from "react";
import type { AskResult, Identity } from "../types.js";

export interface IdentityBadgeProps {
  result: AskResult | null;
  identity: Identity | null;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }} title={hint}>
      <span style={{ fontSize: 10, color: "#858585", textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</span>
      <span style={{ fontSize: 12, color: "#cccccc", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {value}
      </span>
    </div>
  );
}

export function IdentityBadge({ result, identity }: IdentityBadgeProps) {
  const chiOk = identity ? identity.char_invariant >= identity.floor : false;
  return (
    <div
      style={{
        display: "flex",
        gap: 18,
        padding: "6px 12px",
        background: "#1e1e1e",
        borderTop: "1px solid #232323",
        alignItems: "center",
      }}
    >
      {identity && (
        <Stat
          label="χ identity"
          value={`${identity.char_invariant.toFixed(4)} ${chiOk ? "✓" : "✗"}`}
          hint={`Stoer–Wagner min-cut of the self-graph; must stay ≥ floor ${identity.floor}. ${chiOk ? "conserved" : "BREACH"}`}
        />
      )}
      {result && (
        <>
          <Stat label="committed M" value={String(result.committed_count)} hint="never-resetting count of committed asks (Inv 2)" />
          <Stat label="p*" value={result.price.toFixed(3)} hint="water-filling clearing price for this ask" />
          <Stat label="budget" value={`-k ${result.budget}`} hint="total passages allocated across scenes" />
        </>
      )}
      {identity && (
        <Stat
          label="fingerprint"
          value={identity.fingerprint.replace(/^b3:/, "").slice(0, 12) + "…"}
          hint={identity.fingerprint}
        />
      )}
    </div>
  );
}
