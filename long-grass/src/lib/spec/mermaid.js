/* Mermaid, loaded once in the browser, themed for the black surface.
 * securityLevel "strict": a diagram's text can never run script or links. */

let loading = null;

export function loadMermaid() {
  if (!loading) {
    loading = import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "dark",
        themeVariables: { background: "#000000", primaryColor: "#111418", primaryBorderColor: "#4a4946", lineColor: "#898781", fontFamily: "ui-monospace, monospace", fontSize: "13px" },
        // Natural size, scrolled in its frame: shrunk to the column, a
        // specification's diagram is too small to read.
        flowchart: { htmlLabels: false, curve: "basis", useMaxWidth: false },
        class: { htmlLabels: false, useMaxWidth: false },
      });
      return mermaid;
    });
  }
  return loading;
}

/** → null when it parses, else the parser's message. */
export async function mermaidError(text) {
  const mermaid = await loadMermaid();
  try {
    await mermaid.parse(text);
    return null;
  } catch (e) {
    return String(e?.message || e).split("\n").slice(0, 3).join(" ");
  }
}

let n = 0;
export async function renderMermaid(text) {
  const mermaid = await loadMermaid();
  const { svg } = await mermaid.render(`buhera-diagram-${++n}`, text);
  return svg;
}
