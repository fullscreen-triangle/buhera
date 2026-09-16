/* VerdictDecisionTree — the six-verdict output space (Definition 8.4 of the
 * ladder paper) as a clickable decision tree. Clicking a leaf highlights the
 * path from root and shows the one-line gloss, echoing the VERDICT_GLOSS
 * pattern already used in honjo-masamune's federated Cells.js.
 */
import React, { useState } from "react";

const VERDICTS = {
  answer: { color: "#4fae86", gloss: "certified non-empty answer — the question was expressible, lowerable, and the data had it." },
  empty: { color: "#4a9eff", gloss: "certified empty — expressible, lowerable, executed to completion, and genuinely nothing there." },
  unexpressed: { color: "#9a6cff", gloss: "not statable in the source's own model — refused before any request is issued." },
  unsupported: { color: "#c98a3a", gloss: "statable, but not lowerable against this source's declared capabilities." },
  starved: { color: "#e0b23a", gloss: "a predecessor step in the plan under-retrieved — the characteristic federated failure." },
  exhausted: { color: "#ff5d5d", gloss: "the budget ran out before an answer was reached." },
};

const STEPS = [
  {
    id: "expressible",
    question: "Is the question expressible in the source's own model?",
    no: "unexpressed",
  },
  {
    id: "lowerable",
    question: "Does it lower against this source's declared capabilities?",
    no: "unsupported",
  },
  {
    id: "predecessors",
    question: "Did every predecessor step in the plan retrieve usably?",
    no: "starved",
  },
  {
    id: "budget",
    question: "Did the plan complete within budget?",
    no: "exhausted",
  },
  {
    id: "nonempty",
    question: "Is the certified result set non-empty?",
    no: "empty",
    yes: "answer",
  },
];

export default function VerdictDecisionTree() {
  const [picked, setPicked] = useState(null);

  return (
    <div className="flex flex-col md:flex-row gap-6 items-start">
      <div className="flex-1 min-w-[280px]">
        {STEPS.map((s, i) => (
          <div key={s.id} className="flex items-center gap-2 mb-1">
            <div className="w-5 text-right text-[10px] text-gray-600 font-mono">{i + 1}</div>
            <div className="flex-1 border border-gray-800 rounded px-3 py-2 bg-gray-950 text-xs text-gray-300">
              {s.question}
              <div className="flex gap-2 mt-1.5">
                <button
                  onClick={() => setPicked(s.no)}
                  className="text-[10px] px-2 py-0.5 rounded border border-red-900 text-red-300 hover:bg-red-950"
                >
                  no → {s.no}
                </button>
                {s.yes && (
                  <button
                    onClick={() => setPicked(s.yes)}
                    className="text-[10px] px-2 py-0.5 rounded border border-emerald-900 text-emerald-300 hover:bg-emerald-950"
                  >
                    yes → {s.yes}
                  </button>
                )}
                {!s.yes && i < STEPS.length - 1 && (
                  <span className="text-[10px] text-gray-600 self-center">yes ↓ continue</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="w-full md:w-64 shrink-0">
        <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Verdicts</div>
        <div className="space-y-1.5">
          {Object.entries(VERDICTS).map(([name, v]) => (
            <div
              key={name}
              onClick={() => setPicked(name)}
              className={`cursor-pointer rounded px-2.5 py-1.5 text-xs font-mono border transition ${
                picked === name ? "border-2" : "border-gray-800 opacity-60 hover:opacity-100"
              }`}
              style={picked === name ? { borderColor: v.color, background: v.color + "22" } : {}}
            >
              <span style={{ color: v.color }}>{name}</span>
            </div>
          ))}
        </div>
        {picked && (
          <div className="mt-3 text-[11px] text-gray-400 leading-relaxed border-t border-gray-800 pt-2">
            {VERDICTS[picked].gloss}
          </div>
        )}
        <div className="mt-3 text-[10px] text-gray-600 leading-relaxed">
          Non-degeneracy (Theorem 8.5): no verdict but <span className="text-emerald-400">answer</span> may
          carry a payload.
        </div>
      </div>
    </div>
  );
}
