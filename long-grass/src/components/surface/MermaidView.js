/* A Mermaid diagram on the surface: drawn from its text, with the text one
 * click away, and saved as SVG. Mermaid runs with securityLevel "strict"
 * (lib/spec/mermaid.js), so the SVG it returns carries no script. */

import { useEffect, useRef, useState } from "react";
import { renderMermaid } from "@/lib/spec/mermaid";

export default function MermaidView({ text, name = "diagram", compact = false }) {
  const [svg, setSvg] = useState(null);
  const [error, setError] = useState(null);
  const [showText, setShowText] = useState(false);
  const [copied, setCopied] = useState(false);
  const box = useRef(null);

  useEffect(() => {
    let alive = true;
    setSvg(null);
    setError(null);
    renderMermaid(text).then((s) => alive && setSvg(s)).catch((e) => alive && setError(String(e?.message || e).split("\n").slice(0, 3).join(" ")));
    return () => { alive = false; };
  }, [text]);

  function saveSvg() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    a.download = `${String(name).replace(/[^\w.-]+/g, "-").slice(0, 60) || "diagram"}.svg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  return (
    <div>
      {error && <p className="text-xs text-rose-300/80 mb-2">this does not parse as Mermaid: {error}</p>}
      {!svg && !error && <p className="text-xs text-gray-600 animate-pulse">drawing…</p>}
      {svg && (
        <div ref={box} className={`overflow-auto rounded border border-gray-900 bg-black p-3 ${compact ? "max-h-96" : ""}`}
          // Mermaid's own output, rendered with securityLevel "strict".
          dangerouslySetInnerHTML={{ __html: svg }} />
      )}
      <div className="flex flex-wrap gap-3 mt-2 text-[11px] text-gray-500">
        <button type="button" className="hover:text-gray-200" onClick={() => setShowText((v) => !v)}>{showText ? "hide" : "show"} the Mermaid</button>
        <button type="button" className="hover:text-gray-200" onClick={async () => { try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { setShowText(true); } }}>
          {copied ? "copied" : "copy the Mermaid"}
        </button>
        {svg && <button type="button" className="hover:text-gray-200" onClick={saveSvg}>save as SVG</button>}
      </div>
      {showText && <pre className="mt-2 p-3 rounded border border-gray-900 bg-white/[0.02] text-xs font-mono whitespace-pre-wrap text-teal-100/80">{text}</pre>}
    </div>
  );
}
