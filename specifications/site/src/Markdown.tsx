import { useEffect, useMemo, useRef } from "react";
import { Marked } from "marked";

// Markdown from the specification tree, rendered as-is. Fenced `mermaid`
// blocks become diagrams; relative links between spec files are rewritten to
// the site's routes so cross-references work.
const marked = new Marked({
  gfm: true,
  renderer: {
    code({ text, lang }) {
      if (lang === "mermaid") return `<div class="mermaid-wrap"><pre class="mermaid">${escape(text)}</pre></div>`;
      return `<pre><code class="lang-${lang ?? ""}">${escape(text)}</code></pre>`;
    },
    link({ href, text }) {
      const m = /^(?:\.\.\/)?(?:specs\/)?([\w-]+)\.md$/.exec(href ?? "");
      if (m) return `<a href="#/module/${m[1]}">${text}</a>`;
      return `<a href="${href}" target="_blank" rel="noreferrer">${text}</a>`;
    },
  },
});

function escape(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

let mermaidReady: Promise<typeof import("mermaid").default> | null = null;
function mermaid() {
  mermaidReady ??= import("mermaid").then((m) => {
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    m.default.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral", securityLevel: "strict", fontFamily: "Inter, sans-serif" });
    return m.default;
  });
  return mermaidReady;
}

export function Markdown({ source, skipTitle = false }: { source: string; skipTitle?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const html = useMemo(() => {
    const body = skipTitle ? source.replace(/^#\s+.+\r?\n/, "") : source;
    return marked.parse(body) as string;
  }, [source, skipTitle]);

  useEffect(() => {
    const nodes = ref.current?.querySelectorAll<HTMLElement>("pre.mermaid");
    if (!nodes || nodes.length === 0) return;
    let cancelled = false;
    mermaid().then((m) => {
      if (!cancelled) m.run({ nodes: Array.from(nodes) }).catch(() => undefined);
    });
    return () => {
      cancelled = true;
    };
  }, [html]);

  return <div className="doc" ref={ref} dangerouslySetInnerHTML={{ __html: html }} />;
}
