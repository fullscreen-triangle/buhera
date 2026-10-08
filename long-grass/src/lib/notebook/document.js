/* ============================================================================
 * The landing document: prose and cells, as Markdown.
 *
 * A fenced block whose language is a cell kind is a cell. Running it writes
 * an ```output block directly beneath it; running it again replaces that
 * block, so the document is the latest report. Every run is also appended to
 * the record (lib/server/notebook.js), which only grows: the document
 * forgets, the record does not.
 *
 * Fences may be longer than three backticks; an output containing ``` is
 * written inside a longer fence, so an answer quoting code cannot end its
 * own block early.
 * ========================================================================== */

/** Cell kinds, and what each one does. */
export const KINDS = {
  ask: { label: "ask", hint: "Claude searches the web, your library and your files, then answers with sources" },
  web: { label: "web", hint: "a search engine's results, fresh each run" },
  read: { label: "read", hint: "read a page and keep it in your library" },
  find: { label: "find", hint: "search what you have read, with a verdict" },
  files: { label: "files", hint: "search files on your PC by name and content" },
  bash: { label: "bash", hint: "a bash script, run on your PC", script: true },
  powershell: { label: "powershell", hint: "a PowerShell script, run on your PC", script: true },
  python: { label: "python", hint: "a Python script, run on your PC", script: true },
  node: { label: "node", hint: "a JavaScript file, run with Node on your PC", script: true },
};

const ALIASES = { claude: "ask", sh: "bash", shell: "bash", ps: "powershell", pwsh: "powershell", py: "python", js: "node", javascript: "node", search: "web" };

export function kindOf(lang) {
  const l = String(lang || "").toLowerCase();
  return KINDS[l] ? l : ALIASES[l] || null;
}

export const isScript = (kind) => !!KINDS[kind]?.script;

const FENCE = /^(\s{0,3})(`{3,}|~{3,})\s*([^\s`]*)(.*)$/;

/**
 * Markdown → blocks:
 *   { type: "prose", text }
 *   { type: "cell", kind, info, source, output }   (output: string | null)
 * A fenced block that is not a cell (or an output with no cell above it) stays prose.
 */
export function parse(markdown) {
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let prose = [];
  const flush = () => {
    const text = prose.join("\n").replace(/^\n+|\n+$/g, "");
    if (text.trim()) blocks.push({ type: "prose", text });
    prose = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const m = FENCE.exec(lines[i]);
    if (!m) { prose.push(lines[i]); continue; }
    const fence = m[2];
    const lang = m[3];
    // Find the closing fence: same character, at least as long.
    let j = i + 1;
    const closes = (l) => new RegExp(`^\\s{0,3}${fence[0] === "`" ? "`" : "~"}{${fence.length},}\\s*$`).test(l);
    while (j < lines.length && !closes(lines[j])) j++;
    const body = lines.slice(i + 1, j).join("\n");
    const raw = lines.slice(i, Math.min(j + 1, lines.length));
    const kind = kindOf(lang);
    const last = blocks[blocks.length - 1];

    if (kind) {
      flush();
      blocks.push({ type: "cell", kind, info: m[4].trim(), source: body, output: null });
    } else if (lang === "output" && !prose.some((l) => l.trim()) && last?.type === "cell" && last.output === null) {
      prose = [];
      last.output = body;
    } else {
      prose.push(...raw);
    }
    i = j;
  }
  flush();
  return blocks;
}

function fenceFor(text) {
  const runs = String(text).match(/`{3,}/g) || [];
  const longest = runs.reduce((n, r) => Math.max(n, r.length), 2);
  return "`".repeat(longest + 1);
}

/** blocks → Markdown. parse(serialize(b)) gives back b. */
export function serialize(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.type === "prose") { out.push(b.text.replace(/\n+$/, "")); continue; }
    const f = fenceFor(b.source);
    out.push(`${f}${b.kind}${b.info ? ` ${b.info}` : ""}\n${b.source}\n${f}`);
    if (b.output !== null && b.output !== undefined) {
      const g = fenceFor(b.output);
      out.push(`${g}output\n${b.output}\n${g}`);
    }
  }
  return out.join("\n\n") + "\n";
}

/**
 * What the command line means. → { kind, source }
 *   "web lipid standards"   → web
 *   "$ ls -la" / "bash …"   → bash
 *   "py print(1)"           → python
 *   anything else           → ask
 */
export function command(line) {
  const s = String(line || "").trim();
  if (!s) return null;
  if (s.startsWith("$ ")) return { kind: "bash", source: s.slice(2).trim() };
  if (s.startsWith("> ")) return { kind: "powershell", source: s.slice(2).trim() };
  const m = /^([a-z]+)[:\s]\s*([\s\S]*)$/i.exec(s);
  if (m) {
    const kind = kindOf(m[1]);
    if (kind && m[2].trim()) return { kind, source: m[2].trim() };
  }
  if (/^https?:\/\/\S+$/.test(s)) return { kind: "read", source: s };
  return { kind: "ask", source: s };
}

/** `timeout=120` in a cell's info string, in seconds. */
export function infoTimeout(info) {
  const m = /\btimeout=(\d+)/.exec(info || "");
  return m ? Number(m[1]) : null;
}

/** A new document that says what it is. */
export const STARTER = `# Today

This page is a document you run. Each fenced block below is a **cell**: press **run** (or Ctrl+Enter inside it) and its result is written underneath, replacing the last one. Every run is also kept in the **record**, which only grows.

Write a command in the line at the top and press Enter to add it as a new cell and run it:

- a question — Claude searches the web, what you have read, and (on your PC) your files, then answers with its sources
- \`web <words>\` — a search engine's results
- \`read <address>\` — read a page and keep it in your library
- \`find <words>\` — search what you have read, with a verdict
- \`files <words>\` — search files on your PC
- \`$ <command>\`, \`py <code>\`, \`node <code>\`, \`> <PowerShell>\` — run a script on your PC

\`\`\`ask
What changed in the DCAT-AP 3.0.1 release, and what does DCAT-AP+ add to it?
\`\`\`

\`\`\`web
LARA lab automation SiLA2
\`\`\`

\`\`\`python
import platform, datetime
print(platform.node(), platform.python_version(), datetime.datetime.now().isoformat(timespec="seconds"))
\`\`\`
`;
