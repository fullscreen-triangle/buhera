// translator.ts — pattern matching from natural language to vaHera.

export function translate(intent: string): string {
  const q = intent.trim();
  if (!q) return "";

  let m;

  // "what is the <property> of <compound>" / "<property> of <compound>"
  m = q.match(/^(?:what is (?:the )?)?(?:boiling point|melting point|molecular weight|density)\s+of\s+(\w+)$/i);
  if (m) {
    const name = m[1].toLowerCase();
    return [
      `describe ${name} with "${q}"`,
      `resolve ${name}`,
      `spawn query from ${name}`,
      `navigate to penultimate`,
      `complete trajectory`,
    ].join("\n");
  }

  // "what is <compound>"
  m = q.match(/^what is (?:an?\s+)?(\w+)/i);
  if (m) {
    const name = m[1].toLowerCase();
    return [
      `describe ${name} with "${q}"`,
      `resolve ${name}`,
      `spawn query from ${name}`,
      `navigate to penultimate`,
      `complete trajectory`,
    ].join("\n");
  }

  // "find compounds similar to <compound>" / "similar to <compound>"
  m = q.match(/(?:compounds\s+)?similar\s+to\s+(\w+)/i);
  if (m) {
    return `memory find nearest "${m[1]}" k=5`;
  }

  // "find <anything>" or "find what I wrote about <anything>"
  m = q.match(/^find\s+(?:what i (?:wrote|said) about\s+)?(.+)/i);
  if (m) {
    return `memory find nearest "${m[1]}" k=5`;
  }

  // "remember/store/note: <text>"
  m = q.match(/^(?:remember|store|note)(?:\s+that)?[:\s]+(.+)/i);
  if (m) {
    const text = m[1].replace(/"/g, "'");
    const id = Math.abs(hash(text)).toString(36).slice(0, 6);
    return `memory store "n_${id}" = "${text}"`;
  }

  // Single-word compound name
  m = q.match(/^(\w+)$/);
  if (m) {
    const name = m[1].toLowerCase();
    return [
      `describe ${name} with "${q}"`,
      `resolve ${name}`,
      `spawn query from ${name}`,
      `navigate to penultimate`,
      `complete trajectory`,
    ].join("\n");
  }

  // Fallback: treat as general query
  const safe = q.replace(/"/g, "'");
  return [
    `describe query with "${safe}"`,
    `resolve query`,
    `spawn q from query`,
    `navigate to penultimate`,
    `complete trajectory`,
  ].join("\n");
}

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h;
}
