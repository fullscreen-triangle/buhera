// Buhera OS web REPL.
//
// Talks to a locally-running `buhera-server` (default: http://localhost:5599)
// over plain JSON. Renders results in a terminal-style buffer. The
// only state held in React is the input line and the scrollback; all
// kernel state lives in the server process.

import { useEffect, useRef, useState } from "react";

const DEFAULT_BASE = "http://localhost:5599";

// ─────────────────────────────────────────────────────────────────────
//  Small HTTP helpers.
// ─────────────────────────────────────────────────────────────────────

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const text = await res.text();
  let body;
  try {
    body = text ? JSON.parse(text) : null;
  } catch (_) {
    body = { raw: text };
  }
  if (!res.ok) {
    const msg = body && body.error ? body.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

async function postVahera(base, source) {
  return fetchJson(`${base}/vahera`, {
    method: "POST",
    body: JSON.stringify({ source, rerank: true }),
  });
}

async function getInfo(base) {
  return fetchJson(`${base}/info`);
}

async function deleteMemory(base) {
  return fetchJson(`${base}/memory`, { method: "DELETE" });
}

// ─────────────────────────────────────────────────────────────────────
//  Friendly shortcuts (mirror the desktop REPL).
// ─────────────────────────────────────────────────────────────────────

const VAHERA_PREFIXES = [
  "describe ",
  "resolve ",
  "spawn ",
  "navigate ",
  "complete ",
  "memory ",
  "demon ",
  "controller ",
  "kernel ",
  "process ",
];

function stripQuotes(s) {
  const t = s.trim();
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) {
    return t.slice(1, -1);
  }
  return t;
}

function translateShortcut(input) {
  const lower = input.trim().toLowerCase();
  if (VAHERA_PREFIXES.some((p) => lower.startsWith(p))) return input;

  if (lower.startsWith("store ")) {
    const rest = input.trim().slice(6);
    const eq = rest.indexOf("=");
    if (eq >= 0) {
      const name = rest.slice(0, eq).trim();
      const value = stripQuotes(rest.slice(eq + 1));
      if (name && value) return `memory store "${name}" = "${value}"`;
    }
  }

  if (lower.startsWith("find ")) {
    const rest = input.trim().slice(5);
    const k = rest.match(/\sk=(\d+)\s*$/);
    let kn = 3;
    let text = rest;
    if (k) {
      kn = parseInt(k[1], 10);
      text = rest.slice(0, k.index);
    }
    return `memory find nearest "${stripQuotes(text)}" k=${kn}`;
  }

  if (lower.startsWith("dump ")) {
    return `memory dump ${input.trim().slice(5).trim()}`;
  }

  const single = {
    list: "memory list",
    sort: "demon sort",
    stats: "kernel stats",
    trace: "kernel trace",
    procs: "process list",
    ps: "process list",
    verify: "controller verify",
  }[lower];
  if (single) return single;

  // Bare line: treat as a search query.
  const safe = input.trim().replace(/"/g, "'");
  return `memory find nearest "${safe}" k=3`;
}

// ─────────────────────────────────────────────────────────────────────
//  Render one server response into terminal lines.
// ─────────────────────────────────────────────────────────────────────

function renderResult(r) {
  if (r.kind === "find_hits") {
    const lines = [`hits for "${r.query}":`];
    if (!r.hits.length) {
      lines.push("  (no hits)");
    } else {
      r.hits.forEach((h, i) => {
        const name = h.name || "?";
        const addr = h.address.slice(0, 12);
        lines.push(`  [${i + 1}] ${name} addr=${addr} d=${h.distance.toFixed(4)}`);
        if (h.source) {
          lines.push(`      "${h.source}"`);
        }
      });
    }
    return lines;
  }
  if (r.kind === "object_list") {
    const lines = [`memory (${r.objects.length} objects):`];
    r.objects.forEach((o) => {
      const name = o.name || "?";
      const addr = o.address.slice(0, 12);
      const [k, t, e] = o.coord;
      lines.push(
        `  ${addr} ${name} tier=${o.tier} coord=S(${k.toFixed(3)},${t.toFixed(3)},${e.toFixed(3)})`
      );
    });
    return lines;
  }
  if (r.kind === "sorted_objects") {
    const lines = [`sorted (${r.objects.length} objects):`];
    r.objects.forEach((o) => {
      const name = o.name || "?";
      const addr = o.address.slice(0, 12);
      lines.push(`  ${addr} ${name} tier=${o.tier}`);
    });
    return lines;
  }
  if (r.kind === "dump") {
    if (!r.object) return [`dump ${r.name}: (not found)`];
    const o = r.object;
    return [
      `dump ${r.name}:`,
      `  address: ${o.address}`,
      `  coord:   S(${o.coord[0].toFixed(3)},${o.coord[1].toFixed(3)},${o.coord[2].toFixed(3)})`,
      `  tier:    ${o.tier}`,
      ...(o.source ? [`  source:  "${o.source}"`] : []),
    ];
  }
  if (r.kind === "stats") {
    return ["stats:", ...JSON.stringify(r.stats, null, 2).split("\n").map((l) => "  " + l)];
  }
  if (r.kind === "trace") {
    return [`trace (${r.log.length} entries):`, ...r.log.map((l) => "  " + l)];
  }
  if (r.kind === "processes") {
    const lines = [`processes (${r.processes.length}):`];
    r.processes.forEach((p) => {
      lines.push(
        `  pid=${p.pid} ${p.program_name} state=${p.state} d_traj=${p.d_traj.toFixed(3)}`
      );
    });
    return lines;
  }
  return [JSON.stringify(r)];
}

// ─────────────────────────────────────────────────────────────────────
//  Help text.
// ─────────────────────────────────────────────────────────────────────

const HELP = `
The Buhera OS files things by their categorical address — three numbers
between 0 and 1 derived from the meaning of a piece of text.

Quick-start (try these one at a time):
  store note   = "remember to send the proposal to Sarah"
  store other  = "buy milk and bread from the corner store"
  find "writing tasks"
  find "shopping"
  list
  stats

Shortcuts:
  store <name> = "<text>"     memory store "<name>" = "<text>"
  find "<text>"               memory find nearest "<text>" k=3
  list / sort / stats / trace / procs / verify
  dump <name>                 memory dump <name>
  A bare line with no keyword is treated as a search query.

Meta commands:
  :help   show this message
  :clear  reset the kernel to empty
  :info   show server info (embedder, depth, object count)
  :tour   load a guided demo
`.trim();

const TOUR = `
memory store "weekend"   = "I need to do laundry and clean the kitchen this weekend"
memory store "groceries" = "buy milk eggs bread and coffee from the supermarket"
memory store "exercise"  = "go for a run on Saturday morning before it gets hot"
memory store "travel"    = "book a flight to Munich for the conference next month"
memory store "code"      = "refactor the database connection pool to use async"
memory find nearest "shopping list" k=3
memory find nearest "morning workout" k=3
memory find nearest "flight to Germany" k=3
memory find nearest "database refactor" k=3
kernel stats
`.trim();

// ─────────────────────────────────────────────────────────────────────
//  The page.
// ─────────────────────────────────────────────────────────────────────

export default function OsRepl() {
  const [base, setBase] = useState(DEFAULT_BASE);
  const [info, setInfo] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);
  const [cursor, setCursor] = useState(-1);
  const [buffer, setBuffer] = useState([
    { kind: "system", text: "Buhera OS web REPL" },
    { kind: "system", text: "type :help for commands" },
  ]);
  const bottomRef = useRef(null);

  // Probe the server on mount and whenever base changes.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const i = await getInfo(base);
        if (!cancelled) {
          setInfo(i);
          setServerError(null);
          setBuffer((b) => [
            ...b,
            {
              kind: "system",
              text: `(connected: ${i.embedder}, depth=${i.depth}, objects=${i.objects})`,
            },
          ]);
        }
      } catch (err) {
        if (!cancelled) {
          setInfo(null);
          setServerError(err.message);
          setBuffer((b) => [
            ...b,
            { kind: "error", text: `(could not reach ${base}: ${err.message})` },
            { kind: "system", text: `Start the server: cargo run -p buhera-server --release` },
          ]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [base]);

  // Auto-scroll on new buffer lines.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [buffer]);

  function appendLines(lines, kind = "out") {
    setBuffer((b) => [...b, ...lines.map((text) => ({ kind, text }))]);
  }

  async function dispatch(raw) {
    const trimmed = raw.trim();
    if (!trimmed) return;
    setBuffer((b) => [...b, { kind: "in", text: `buhera> ${trimmed}` }]);
    setHistory((h) => (h[h.length - 1] === trimmed ? h : [...h, trimmed]));
    setCursor(-1);

    // Meta commands.
    if (trimmed === ":help") {
      appendLines(HELP.split("\n"), "help");
      return;
    }
    if (trimmed === ":info") {
      try {
        const i = await getInfo(base);
        setInfo(i);
        appendLines(
          [
            `version=${i.version}`,
            `embedder=${i.embedder}`,
            `depth=${i.depth}`,
            `objects=${i.objects}`,
          ],
          "out"
        );
      } catch (err) {
        appendLines([`error: ${err.message}`], "error");
      }
      return;
    }
    if (trimmed === ":clear") {
      try {
        const i = await deleteMemory(base);
        setInfo(i);
        appendLines([`(kernel reset; ${i.objects} objects)`], "system");
      } catch (err) {
        appendLines([`error: ${err.message}`], "error");
      }
      return;
    }
    if (trimmed === ":tour") {
      await runScript(TOUR);
      return;
    }

    // Otherwise translate the shortcut and send to the server.
    const source = translateShortcut(trimmed);
    await runScript(source);
  }

  async function runScript(source) {
    setBusy(true);
    try {
      const res = await postVahera(base, source);
      // Show trace lines compactly.
      if (res.trace.length) {
        appendLines(res.trace.map((t) => `  ${t}`), "trace");
      }
      // Render each result block.
      for (const r of res.results) {
        appendLines(renderResult(r), "out");
      }
      if (!res.results.length && !res.trace.length) {
        appendLines(["ok"], "out");
      }
    } catch (err) {
      appendLines([`error: ${err.message}`], "error");
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const value = input;
      setInput("");
      dispatch(value);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!history.length) return;
      const next = cursor < 0 ? history.length - 1 : Math.max(0, cursor - 1);
      setCursor(next);
      setInput(history[next]);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (cursor < 0) return;
      const next = cursor + 1;
      if (next >= history.length) {
        setCursor(-1);
        setInput("");
      } else {
        setCursor(next);
        setInput(history[next]);
      }
    }
  }

  return (
    <div className="min-h-screen bg-black text-gray-200 font-mono p-4">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-gray-800 pb-2 mb-3">
        <div className="text-sm">
          <span className="text-gray-500">buhera-os web repl</span>
          {info && (
            <span className="ml-4 text-gray-400">
              {info.embedder} · depth {info.depth} · {info.objects} objects
            </span>
          )}
          {serverError && <span className="ml-4 text-red-400">offline</span>}
        </div>
        <div className="text-xs text-gray-500 flex items-center gap-2">
          <span>server:</span>
          <input
            value={base}
            onChange={(e) => setBase(e.target.value)}
            className="bg-gray-900 text-gray-300 px-2 py-0.5 rounded outline-none border border-gray-800 w-64"
            spellCheck={false}
          />
        </div>
      </div>

      {/* Scrollback */}
      <div className="text-sm whitespace-pre-wrap leading-relaxed">
        {buffer.map((line, i) => (
          <div
            key={i}
            className={
              line.kind === "in"
                ? "text-white"
                : line.kind === "trace"
                ? "text-gray-500"
                : line.kind === "error"
                ? "text-red-400"
                : line.kind === "system"
                ? "text-gray-500"
                : line.kind === "help"
                ? "text-gray-400"
                : "text-gray-200"
            }
          >
            {line.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Prompt */}
      <div className="flex items-center gap-2 pt-3 border-t border-gray-800 mt-3">
        <span className="text-gray-500">buhera&gt;</span>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy || !!serverError}
          autoFocus
          placeholder={
            serverError
              ? "(server offline)"
              : busy
              ? "(working...)"
              : "type a command or ask a question"
          }
          spellCheck={false}
          className="flex-1 bg-transparent outline-none text-gray-200 placeholder-gray-600"
        />
      </div>
    </div>
  );
}
