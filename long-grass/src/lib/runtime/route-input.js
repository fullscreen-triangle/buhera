/* ============================================================================
 * route-input.js — the pure terminal-line router.
 *
 * Given a raw line the user typed, classify it into a route the runner
 * dispatches on:
 *   { type: "noop" }
 *   { type: "meta", meta }
 *   { type: "scope_ctl", ctl, url? }
 *   { type: "scope", source }
 *   { type: "srn", instruction }
 *   { type: "smith", instruction }
 *   { type: "dispatch", moduleId, instruction }
 *   { type: "turbulance", source }
 *   { type: "vahera", vahera }
 *   { type: "nl", text }
 *
 * This is pure string work with no React, no browser globals, and no module
 * dispatch — extracted out of BuheraTerminal so the router can be imported by
 * the Node-side runner (run-input.js), the harness, and unit tests. The
 * terminal re-exports these names for backwards compatibility.
 * ========================================================================== */

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

// A line starting with one of these is a SCOPE REPL cell (declaration block).
const SCOPE_PREFIXES = [
  "coordinate_space",
  "channels ",
  "channels{",
  "goal ",
  "goal{",
  "rule ",
  "dispatch ",
  "dispatch{",
];
// A morphism cell: `<ident> = observe(...` — the assignment form.
const SCOPE_MORPHISM_RE = /^[a-zA-Z_]\w*\s*=\s*observe\s*\(/;

// SRN expression prefixes — routes to the srn module.
const SRN_GLYPH_RE = /^◈\s*\(/;

// A line starting with one of these is a turbulance (kwasa-kwasa) script.
const TURBULANCE_PREFIXES = [
  "funxn ",
  "item ",
  "proposition ",
  "hypothesis ",
  "point ",
  "given ",
  "considering ",
  "within ",
  "research ",
  "for each ",
];

function stripQuotes(s) {
  const t = s.trim();
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) return t.slice(1, -1);
  return t;
}

// Parse a literal `dispatch("<module>", <instruction>)` call.
//
// The instruction argument is a JS/JSON value: a quoted string, or an object/
// array literal. We locate the module id (first quoted arg) and the raw text of
// the second argument, then evaluate that text as a value. Evaluation is done
// with a tightly-scoped `Function` returning the literal — the tutorials are
// author-controlled cells, and the alternative (a full JSON5 parser) is a
// dependency the no-install webtool avoids. A parse failure returns null so the
// caller falls through to the other routes.
//
// Returns { moduleId, instruction } or null.
export function parseDispatchCall(src) {
  const text = src.trim();
  // Must start with `dispatch(` and end with `)`. Cheap gate before the work.
  const head = text.match(/^dispatch\s*\(\s*(["'])((?:\\.|[^\\])*?)\1\s*(,|\))/s);
  if (!head) return null;
  const moduleId = head[2];

  // No second argument: `dispatch("mod")` → empty-string instruction.
  if (head[3] === ")") {
    // ensure nothing trails the close paren
    if (text.slice(head.index + head[0].length).trim() !== "") return null;
    return { moduleId, instruction: "" };
  }

  // Extract the second argument: everything between the comma and the final
  // matching close paren. Find the close paren that balances the opening one.
  const openParen = text.indexOf("(");
  let depth = 0;
  let closeParen = -1;
  let inStr = null;
  for (let i = openParen; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === "\\") { i++; continue; }
      if (ch === inStr) inStr = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") { inStr = ch; continue; }
    if (ch === "(") depth++;
    else if (ch === ")") { depth--; if (depth === 0) { closeParen = i; break; } }
  }
  if (closeParen === -1) return null;
  if (text.slice(closeParen + 1).trim() !== "") return null;

  // The comma separating the two args is at head[0]'s end minus the captured
  // comma; re-find it as the first top-level comma after the module string.
  const afterModule = head.index + head[0].length; // char after the comma
  const argText = text.slice(afterModule, closeParen).trim();
  if (!argText) return { moduleId, instruction: "" };

  let instruction;
  try {
    // eslint-disable-next-line no-new-func
    instruction = Function('"use strict"; return (' + argText + ");")();
  } catch {
    return null;
  }
  return { moduleId, instruction };
}

// Split a multi-line cell into top-level statements. A newline ends a
// statement only when it sits at bracket depth 0 and outside any string, so a
// multi-line object argument — `dispatch("x", {\n  ...\n})` — or a newline
// embedded in a quoted string — `"phase p:\n  ..."` — stays a single unit.
// Returns an array of trimmed, non-empty statement strings (>= 1 element).
//
// This is deliberately conservative: the caller only uses the split result
// when the whole cell would otherwise fall through to NL search AND every
// piece routes to a real command. Block DSLs whose statements legitimately
// span depth-0 newlines (turbulance `funxn`/`for each`, scope blocks) are
// detected by the whole-cell router first and never reach this splitter.
export function splitStatements(src) {
  const stmts = [];
  let buf = "";
  let depth = 0;
  let inStr = null;
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inStr) {
      buf += ch;
      if (ch === "\\") { if (i + 1 < src.length) { buf += src[++i]; } continue; }
      if (ch === inStr) inStr = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") { inStr = ch; buf += ch; continue; }
    if (ch === "(" || ch === "[" || ch === "{") { depth++; buf += ch; continue; }
    if (ch === ")" || ch === "]" || ch === "}") { if (depth > 0) depth--; buf += ch; continue; }
    if (ch === "\n" && depth === 0) {
      if (buf.trim()) stmts.push(buf.trim());
      buf = "";
      continue;
    }
    buf += ch;
  }
  if (buf.trim()) stmts.push(buf.trim());
  return stmts.length ? stmts : [src.trim()];
}

/**
 * Classify a raw input line. Pure — never dispatches, never throws.
 */
export function routeInput(line) {
  const trimmed = line.trim();
  if (!trimmed) return { type: "noop" };

  const lower = trimmed.toLowerCase();

  // Meta commands.
  if (trimmed === ":quit" || trimmed === ":exit") return { type: "meta", meta: "quit" };
  if (trimmed === ":help") return { type: "meta", meta: "help" };
  if (trimmed === ":clear") return { type: "meta", meta: "clear" };
  if (trimmed === ":tour") return { type: "meta", meta: "tour" };
  if (trimmed === ":proteins") return { type: "meta", meta: "proteins" };
  if (trimmed === ":modules") return { type: "meta", meta: "modules" };
  if (trimmed === ":audit") return { type: "meta", meta: "audit" };
  if (trimmed === ":tutorials") return { type: "meta", meta: "tutorials" };

  // SCOPE meta-commands: `:scope load <url>`, `:scope reset`, `:scope` (state).
  if (lower === ":scope") return { type: "scope_ctl", ctl: "state" };
  if (lower === ":scope reset") return { type: "scope_ctl", ctl: "reset" };
  if (lower.startsWith(":scope load ")) {
    return { type: "scope_ctl", ctl: "load", url: trimmed.slice(":scope load ".length).trim() };
  }

  // SCOPE REPL cell.
  if (SCOPE_PREFIXES.some((p) => lower.startsWith(p)) || SCOPE_MORPHISM_RE.test(trimmed)) {
    return { type: "scope", source: trimmed };
  }

  // SRN: pre-formed glyph (starts with ◈)
  if (SRN_GLYPH_RE.test(trimmed)) {
    return { type: "srn", instruction: { kind: "eval", glyph: trimmed } };
  }

  // SRN: control commands
  if (lower === "srn:peers") {
    return { type: "srn", instruction: { kind: "peers" } };
  }
  if (lower === "srn:gossip") {
    return { type: "srn", instruction: { kind: "gossip" } };
  }
  if (lower.startsWith("srn:probe ")) {
    return { type: "srn", instruction: { kind: "probe", node: trimmed.slice("srn:probe ".length).trim() } };
  }

  // SRN: NL statement ("srn <anything>" or "link <anything>")
  if (lower.startsWith("srn ") || lower.startsWith("link ")) {
    const text = trimmed.slice(trimmed.indexOf(" ") + 1).trim();
    return { type: "srn", instruction: { kind: "nl", text } };
  }

  // Smith (agent generation).
  if (lower.startsWith("smith run ") || lower.startsWith("smith:run ")) {
    const src = trimmed.slice(trimmed.indexOf(" ", trimmed.indexOf("run")) + 1).trim();
    return { type: "smith", instruction: { source: src, run: true } };
  }
  if (lower.startsWith("smith ")) {
    return { type: "smith", instruction: { source: trimmed.slice(6).trim(), run: false } };
  }
  if (lower.startsWith("agent ") || lower.startsWith("society ")) {
    return { type: "smith", instruction: { source: trimmed, run: false } };
  }

  // A literal `dispatch("<module>", <instruction>)` call.
  {
    const call = parseDispatchCall(trimmed);
    if (call) {
      return { type: "dispatch", moduleId: call.moduleId, instruction: call.instruction };
    }
  }

  // Turbulance script (kwasa-kwasa).
  if (TURBULANCE_PREFIXES.some((p) => lower.startsWith(p))) {
    return { type: "turbulance", source: trimmed };
  }

  // Already vaHera.
  if (VAHERA_PREFIXES.some((p) => lower.startsWith(p))) {
    return { type: "vahera", vahera: trimmed };
  }

  // store <name> = "<text>"
  if (lower.startsWith("store ")) {
    const rest = trimmed.slice(6);
    const eq = rest.indexOf("=");
    if (eq >= 0) {
      const name = rest.slice(0, eq).trim();
      const value = stripQuotes(rest.slice(eq + 1));
      if (name && value) {
        return { type: "vahera", vahera: `memory store "${name}" = "${value}"` };
      }
    }
  }

  // find "<text>" [k=N]
  if (lower.startsWith("find ")) {
    const rest = trimmed.slice(5);
    const m = rest.match(/\sk=(\d+)\s*$/);
    let k = 3;
    let text = rest;
    if (m) {
      k = parseInt(m[1], 10);
      text = rest.slice(0, m.index);
    }
    const t = stripQuotes(text.trim());
    return { type: "vahera", vahera: `memory find nearest "${t}" k=${k}` };
  }

  // dump <name>
  if (lower.startsWith("dump ")) {
    return { type: "vahera", vahera: `memory dump ${trimmed.slice(5).trim()}` };
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
  if (single) return { type: "vahera", vahera: single };

  // Otherwise leave it for the NL translator (when proteins mode is
  // on) or treat as a search query.
  return { type: "nl", text: trimmed };
}
