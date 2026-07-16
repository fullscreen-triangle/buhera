// API route: /api/srn
//
// Bridge between BuheraTerminal's srn-module and the SRN node HTTP API
// (crates/srn-node, axum server on :7700).
//
// POST body shapes:
//   { kind: "nl",     text }         — NL statement → LLM → SRN glyph → eval
//   { kind: "eval",   glyph }        — pre-formed SRN glyph → eval on best node
//   { kind: "peers" }                — fetch peer list from local node
//   { kind: "probe",  node }         — probe a specific node URL
//   { kind: "gossip" }               — trigger gossip round on local node
//
// SRN node base URL: SRN_NODE_URL env (default http://100.77.3.78:7700)
// Fallback nodes tried in order when primary fails.

import { pickProvider, chat, chatCascade } from "@/lib/server/llm-cascade";

const PRIMARY_NODE = process.env.SRN_NODE_URL || "http://100.77.3.78:7700";

// Nodes in partition-coordinate order; tried in sequence when primary is down.
const FALLBACK_NODES = (process.env.SRN_FALLBACK_NODES || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

// ---------------------------------------------------------------------------
// SRN node HTTP helpers
// ---------------------------------------------------------------------------

async function srnFetch(nodeBase, path, body) {
  const url = nodeBase.replace(/\/$/, "") + path;
  const opts = body != null
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };

  const t0 = Date.now();
  const res = await fetch(url, { ...opts, signal: AbortSignal.timeout(8000) });
  const elapsed_ms = Date.now() - t0;

  if (!res.ok) {
    throw Object.assign(new Error(`SRN node ${res.status}`), { status: res.status, elapsed_ms });
  }
  const data = await res.json();
  return { data, elapsed_ms };
}

async function tryNodes(path, body) {
  const nodes = [PRIMARY_NODE, ...FALLBACK_NODES];
  let lastErr;
  for (const node of nodes) {
    try {
      const result = await srnFetch(node, path, body);
      return { ...result, node };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("no SRN nodes reachable");
}

// ---------------------------------------------------------------------------
// NL → SRN glyph translation via LLM cascade
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `\
You are the SRN (Sango Rine Shumba) compiler frontend.

SRN expression syntax:
  Glyph:     ◈(n=<int>, l=<int>, m=<int>, s=<±1>)
             n = principal quantum number (shell depth, 1..∞)
             l = azimuthal  (0..n-1)
             m = magnetic   (-l..l)
             s = spin parity (+1 or -1)
  Receiver:  ¬<glyph>   — mandatory NOT boundary (who cannot receive)
  Binding:   <glyph> ⊗ <glyph>   — transmit
  Query:     <glyph> → ?          — fetch evaluation

The user has written a natural-language statement describing a task or query
over the distributed SRN network. Translate it into a single SRN expression
that best captures the intent.

Output ONLY the SRN expression. No explanation. No markdown. No punctuation
outside the expression itself.

Examples:
  "bandwidth between nodes" → ◈(n=2,l=1,m=0,s=+1) → ?
  "entropy of the mesh"     → ◈(n=3,l=0,m=0,s=-1) → ?
  "probe node health"       → ◈(n=1,l=0,m=0,s=+1) → ?
  "list peers"              → ◈(n=1,l=0,m=0,s=-1) → ?
`;

async function nlToGlyph(text) {
  const provider = pickProvider();
  if (!provider) {
    return { glyph: "◈(n=1,l=0,m=0,s=+1) → ?", provider: null, model: null };
  }

  const result = await chatCascade({
    system: SYSTEM_PROMPT,
    user: text,
    maxTokens: 64,
    temperature: 0.1,
  });

  if (!result.ok) {
    throw new Error(`LLM translation failed: ${result.error}`);
  }

  const glyph = result.content.trim().split("\n")[0].trim();
  return { glyph, provider: result.provider, model: result.model };
}

// ---------------------------------------------------------------------------
// Chart-data extraction
// ---------------------------------------------------------------------------

function extractChart(nodeResponse) {
  // SRN node eval may return numeric arrays we can surface as a line chart.
  if (!nodeResponse || typeof nodeResponse !== "object") return null;
  const vals = nodeResponse.values || nodeResponse.series || nodeResponse.data;
  if (!Array.isArray(vals) || vals.length === 0) return null;
  if (!vals.every((v) => typeof v === "number")) return null;
  return {
    kind: "line",
    labels: vals.map((_, i) => String(i)),
    values: vals,
  };
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "POST only" });
  }

  const { kind, text, glyph, node } = req.body || {};

  try {
    // ── peers ──────────────────────────────────────────────────────────────
    if (kind === "peers") {
      const { data, elapsed_ms, node: usedNode } = await tryNodes("/peers", null);
      return res.status(200).json({
        output_delta: {
          kind: "srn_peers",
          peers: Array.isArray(data) ? data : data?.peers ?? [],
          elapsed_ms,
          node: usedNode,
        },
        residue: 1,
      });
    }

    // ── probe ──────────────────────────────────────────────────────────────
    if (kind === "probe") {
      const target = node || PRIMARY_NODE;
      const { data, elapsed_ms } = await srnFetch(target, "/probe", null);
      return res.status(200).json({
        output_delta: {
          kind: "srn_probe",
          target,
          elapsed_ms,
          ok: data?.ok ?? true,
          ...data,
        },
        residue: 1,
      });
    }

    // ── gossip ─────────────────────────────────────────────────────────────
    if (kind === "gossip") {
      const { data, elapsed_ms, node: usedNode } = await tryNodes("/gossip", {});
      return res.status(200).json({
        output_delta: {
          kind: "srn_result",
          glyph: "(gossip round)",
          provider: null,
          model: null,
          node: usedNode,
          elapsed_ms,
          value: data,
          chart: null,
        },
        residue: 1,
      });
    }

    // ── eval (pre-formed glyph) ────────────────────────────────────────────
    if (kind === "eval" && glyph) {
      const { data, elapsed_ms, node: usedNode } = await tryNodes("/eval", { expression: glyph });
      return res.status(200).json({
        output_delta: {
          kind: "srn_result",
          glyph,
          provider: null,
          model: null,
          node: usedNode,
          elapsed_ms,
          value: data,
          chart: extractChart(data),
        },
        residue: 1,
      });
    }

    // ── nl → glyph → eval ─────────────────────────────────────────────────
    const statement = text || (typeof req.body === "string" ? req.body : null);
    if (!statement) {
      return res.status(400).json({ error: "missing text" });
    }

    const { glyph: compiled, provider, model } = await nlToGlyph(statement);
    const { data, elapsed_ms, node: usedNode } = await tryNodes("/eval", { expression: compiled });

    return res.status(200).json({
      output_delta: {
        kind: "srn_result",
        glyph: compiled,
        provider,
        model,
        node: usedNode,
        elapsed_ms,
        value: data,
        chart: extractChart(data),
      },
      residue: 1,
    });
  } catch (err) {
    return res.status(200).json({
      error: err.message || String(err),
    });
  }
}
