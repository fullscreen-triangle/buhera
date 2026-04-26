// translator.js — pattern-matching NL -> vaHera, protein-focused.

import { resolveProtein, PROTEINS } from "./proteins";

export function translate(intent) {
  const q = intent.trim();
  if (!q) return "";

  let m;

  // "tell me about X" / "what is X" / "describe X"
  m = q.match(/^(?:tell me about|what is|describe|whats?|who is)\s+(?:the\s+)?(.+?)[\s?.!]*$/i);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "full");
    return queryGeneric(q);
  }

  // "function of X" / "role of X" / "what does X do"
  m = q.match(/^(?:function|role)\s+of\s+(.+?)[\s?.!]*$/i)
    || q.match(/^what does\s+(.+?)\s+do[\s?.!]*$/i);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "function");
  }

  // "diseases? (linked to|of|associated with) X"
  m = q.match(/^diseases?(?:\s+(?:linked to|of|associated with|caused by|related to))?\s+(.+?)[\s?.!]*$/i)
    || q.match(/^what diseases? does\s+(.+?)\s+cause[\s?.!]*$/i);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "diseases");
  }

  // "interactions? of X" / "what does X bind to"
  m = q.match(/^interactions?\s+(?:of|with)\s+(.+?)[\s?.!]*$/i)
    || q.match(/^what does\s+(.+?)\s+(?:bind|interact)(?:\s+with|\s+to)?[\s?.!]*$/i);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "interacts");
  }

  // "domains of X" / "structure of X"
  m = q.match(/^(?:domains?|structure|architecture)\s+of\s+(.+?)[\s?.!]*$/i);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "domains");
  }

  // "compare X and Y" / "X vs Y"
  m = q.match(/^(?:compare\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)[\s?.!]*$/i);
  if (m) {
    const k1 = resolveProtein(m[1]);
    const k2 = resolveProtein(m[2]);
    if (k1 && k2) return queryCompare(k1, k2);
  }

  // "find proteins (in|related to|for) pathway P"
  m = q.match(/^(?:find|list)\s+proteins?\s+(?:in|related to|for|involved in)\s+(.+?)[\s?.!]*$/i);
  if (m) {
    return `memory find nearest "${m[1]}" k=5`;
  }

  // "proteins? (related to|associated with|in) X"
  m = q.match(/^proteins?\s+(?:related to|associated with|in|for)\s+(.+?)[\s?.!]*$/i);
  if (m) return `memory find nearest "${m[1]}" k=5`;

  // "remember / note: <text>"
  m = q.match(/^(?:remember|store|note)(?:\s+that)?[:\s]+(.+)/i);
  if (m) {
    const text = m[1].replace(/"/g, "'");
    const id = Math.abs(hash(text)).toString(36).slice(0, 6);
    return `memory store "n_${id}" = "${text}"`;
  }

  // "find what I (wrote|said) about X" / "find my notes on X"
  m = q.match(/^find\s+(?:what i (?:wrote|said) about|my notes? on)\s+(.+?)[\s?.!]*$/i);
  if (m) return `memory find nearest "${m[1]}" k=5`;

  // Single bare word: try as protein
  m = q.match(/^(\w[\w-]*)\s*[?.!]?$/);
  if (m) {
    const key = resolveProtein(m[1]);
    if (key) return queryProtein(key, "full");
  }

  // Fallback: treat as search
  return `memory find nearest "${q.replace(/"/g, "'")}" k=5`;
}

function queryProtein(key, aspect) {
  // Emit a vaHera program that resolves the protein and asks for a
  // specific aspect of its record.
  return [
    `describe ${key} with "${key}"`,
    `resolve ${key}`,
    `spawn query from ${key}`,
    `navigate to penultimate`,
    `complete trajectory`,
    `# aspect:${aspect}`,
  ].join("\n");
}

function queryCompare(k1, k2) {
  return [
    `describe ${k1} with "${k1}"`,
    `describe ${k2} with "${k2}"`,
    `resolve ${k1}`,
    `resolve ${k2}`,
    `spawn compare from ${k1}`,
    `navigate to penultimate`,
    `complete trajectory`,
    `# aspect:compare:${k2}`,
  ].join("\n");
}

function queryGeneric(q) {
  const safe = q.replace(/"/g, "'");
  return [
    `describe query with "${safe}"`,
    `resolve query`,
    `spawn q from query`,
    `navigate to penultimate`,
    `complete trajectory`,
  ].join("\n");
}

function hash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h;
}
