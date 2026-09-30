// =====================================================================
//  SpraypaintResult — renderers for spraypaint 0.2.0's ask and verify.
//
//  An ask is read verdict-first (graffiti/specifications.md, "procedure:
//  agent protocol"): whether the corpus covers the query at all comes
//  before any passage, because without it every search returns its best
//  look-alike with the same apparent confidence. Then, when the verdict
//  is not `covered`, the per-term table saying which word is missing and
//  whether it is absent from the corpus or merely not returned. Then each
//  passage as its evidence — at most five lines, numbered from the file,
//  matched terms marked — cited path:start-end. The BM25 score is shown
//  small: it means something only within one query.
//
//  A result from an older build has no `coverage`; it is shown with a note
//  that there is no verdict, so each passage may be a look-alike.
// =====================================================================

import SpraypaintAllocationChart from "@/components/sandboxes/spraypaint/SpraypaintAllocationChart";

const VERDICT = {
  covered: { ink: "text-teal-300", ring: "border-teal-700/60", word: "covered" },
  partial: { ink: "text-amber-300", ring: "border-amber-700/60", word: "partial" },
  declined: { ink: "text-rose-300", ring: "border-rose-800/60", word: "declined" },
};

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Marks the query's terms in one evidence line. A term is a whole word as
// the index splits it, so `water` must not light up inside `watershed`.
function Marked({ line, terms }) {
  if (!terms.length) return line;
  const re = new RegExp(`(?<![\\p{L}\\p{N}_])(${terms.map(escapeRe).join("|")})(?![\\p{L}\\p{N}_])`, "giu");
  const parts = line.split(re);
  return parts.map((p, i) =>
    i % 2 === 1 ? <mark key={i} className="bg-transparent text-teal-200 underline decoration-teal-500/60 underline-offset-2">{p}</mark> : p
  );
}

function Evidence({ r, terms }) {
  const from = r.evidence_start_line ?? r.start_line;
  const lines = String(r.snippet ?? "").split("\n");
  const width = String(from + lines.length - 1).length;
  return (
    <pre className="mt-1.5 text-xs font-mono whitespace-pre-wrap break-all text-gray-300 leading-relaxed">
      {lines.map((l, i) => (
        <div key={i} className="flex gap-3">
          <span className="text-gray-600 select-none shrink-0 text-right" style={{ width: `${width}ch` }}>{from + i}</span>
          <span className="text-gray-700 select-none">│</span>
          <span><Marked line={l} terms={terms} /></span>
        </div>
      ))}
    </pre>
  );
}

function Terms({ terms }) {
  const heaviest = Math.max(...terms.map((t) => t.weight || 0), 1e-9);
  return (
    <table className="mt-2 text-xs">
      <tbody>
        {terms.map((t) => (
          <tr key={t.term}>
            <td className="pr-4 font-mono text-gray-200">{t.term}</td>
            <td className="pr-4 w-24">
              <span className="block h-1.5 rounded-sm bg-gray-500/70" style={{ width: `${Math.max(6, (t.weight / heaviest) * 100)}%` }} />
            </td>
            <td className={t.df === 0 ? "text-rose-300/90" : t.in_results ? "text-gray-400" : "text-amber-300/90"}>
              {t.df === 0 ? "not in the corpus" : t.in_results ? `in ${t.df} file${t.df === 1 ? "" : "s"}, returned` : `in ${t.df} file${t.df === 1 ? "" : "s"}, not returned — raise the budget or widen the scenes`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SpraypaintResult({ query, results, allocation, price, budget, committed_count, dry_run, coverage, query_terms, identity_fingerprint, elapsed_ms }) {
  const items = Array.isArray(results) ? results : [];
  const terms = Array.isArray(query_terms) ? query_terms : (coverage?.terms || []).map((t) => t.term);
  const v = coverage ? VERDICT[coverage.verdict] : null;
  let lastScene = null;

  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-3">
        <span className="text-gray-400">spraypaint</span>{" "}
        <span className="text-white font-mono">&quot;{query}&quot;</span>
        {dry_run ? <> · preview, nothing committed</> : typeof committed_count === "number" && <> · committed act #{committed_count}</>}
      </div>

      {coverage ? (
        <div className={`border-l-2 ${v?.ring || "border-gray-700"} pl-3 mb-4`}>
          <div>
            <span className={`font-semibold ${v?.ink || "text-gray-300"}`}>{v?.word || coverage.verdict}</span>
            <span className="text-gray-400"> — {coverage.reason}</span>
          </div>
          {coverage.verdict !== "covered" && Array.isArray(coverage.terms) && coverage.terms.length > 0 && (
            <Terms terms={coverage.terms} />
          )}
        </div>
      ) : (
        <p className="text-xs text-amber-300/80 mb-4">
          this spraypaint build gives no coverage verdict — each passage below may be a look-alike that shares a common word with the query.
        </p>
      )}

      {items.length === 0 ? (
        <p className="text-gray-500">no matching passages.</p>
      ) : (
        <div className="space-y-3 mb-4">
          {items.map((r, i) => {
            const head = r.scene !== lastScene;
            lastScene = r.scene;
            const from = r.evidence_start_line ?? r.start_line;
            const to = r.evidence_end_line ?? r.end_line;
            return (
              <div key={i}>
                {head && <div className="text-[11px] text-gray-600 mb-1">[{r.scene}]</div>}
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="font-mono text-teal-300/90 text-xs break-all">{r.path}:{from}-{to}</span>
                  {Array.isArray(r.matched_terms) && r.matched_terms.length > 0 && (
                    <span className="text-[11px] text-gray-500">matched {r.matched_terms.join(", ")}</span>
                  )}
                  <span className="text-[11px] text-gray-700">score {typeof r.score === "number" ? r.score.toFixed(2) : r.score}</span>
                </div>
                <Evidence r={r} terms={terms} />
              </div>
            );
          })}
        </div>
      )}

      {Array.isArray(allocation) && allocation.length > 0 && <SpraypaintAllocationChart allocation={allocation} />}

      <div className="text-[11px] text-gray-600 mt-2">
        {typeof price === "number" && <>p* {price.toFixed(2)}{price === 0 ? " (budget not used up)" : ""}</>}
        {typeof budget === "number" && <> · budget {budget}</>}
        {identity_fingerprint && <> · index {identity_fingerprint.slice(0, 14)}</>}
        {typeof elapsed_ms === "number" && <> · {elapsed_ms} ms</>}
      </div>
    </div>
  );
}

const STATUS_INK = { PASS: "text-green-400", FAIL: "text-red-400", "N/A": "text-amber-300" };

export function SpraypaintVerify({ overall, degeneracies, invariants, exit_code }) {
  const rows = Array.isArray(invariants) ? invariants : [];
  return (
    <div className="text-gray-300 text-sm">
      <p className="mb-2">
        overall: <span className={STATUS_INK[overall] || "text-gray-300"}>{overall}</span>
        {typeof exit_code === "number" && <span className="text-xs text-gray-600"> · exit {exit_code}</span>}
      </p>
      {Array.isArray(degeneracies) && degeneracies.length > 0 && (
        <div className="mb-3 text-xs text-amber-300/80">
          the corpus is degenerate, so a pass here is not evidence:
          {degeneracies.map((d, i) => <div key={i} className="pl-4 text-gray-400">{d}</div>)}
        </div>
      )}
      <ul className="space-y-2">
        {rows.map((inv, i) => (
          <li key={i} className="text-xs">
            <span className={STATUS_INK[inv.status] || ""}>[{inv.status}]</span>{" "}
            <span className="text-white">{inv.name}</span>
            {(inv.checks || []).map((c, k) => (
              <div key={k} className="pl-8 text-gray-500">
                {c.name && <span className={STATUS_INK[c.status] || ""}>{c.name}</span>}
                {c.name ? " — " : ""}{c.detail}
              </div>
            ))}
          </li>
        ))}
      </ul>
    </div>
  );
}
