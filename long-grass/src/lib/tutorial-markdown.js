/* ============================================================================
 * tutorial-markdown.js
 *
 * Tiny in-repo Markdown-to-blocks converter dedicated to the tutorial pages.
 * The tutorials use a small subset (headings, paragraphs, fenced code blocks,
 * inline code, tables, bold, links, unordered lists, hr). Writing this from
 * scratch is smaller than adding a runtime markdown dependency and keeps the
 * client bundle lean.
 *
 * parseTutorial(source) → an array of "blocks" the renderer walks:
 *   { type: "h1"|"h2"|"h3"|"h4", text }
 *   { type: "p", inline }             // inline = tokens with formatting
 *   { type: "code", lang, code }
 *   { type: "table", header: [inline], rows: [[inline]] }
 *   { type: "ul", items: [inline] }
 *   { type: "hr" }
 *   { type: "blockquote", inline }
 *
 * `inline` is an array of tokens:
 *   { kind: "text", text }
 *   { kind: "code", text }
 *   { kind: "bold", text }
 *   { kind: "link", text, href }
 *
 * No external state, pure functions. Callers use this at build time via
 * getStaticProps.
 * ========================================================================== */

const INLINE_CODE = /`([^`]+)`/;
const BOLD = /\*\*([^*]+)\*\*/;
const LINK = /\[([^\]]+)\]\(([^)]+)\)/;

export function parseInline(text) {
  const tokens = [];
  let rest = text;
  while (rest.length > 0) {
    const codeM = rest.match(INLINE_CODE);
    const boldM = rest.match(BOLD);
    const linkM = rest.match(LINK);
    // Find the earliest match
    const candidates = [
      codeM && { m: codeM, kind: "code", start: codeM.index },
      boldM && { m: boldM, kind: "bold", start: boldM.index },
      linkM && { m: linkM, kind: "link", start: linkM.index },
    ].filter(Boolean);
    if (candidates.length === 0) {
      tokens.push({ kind: "text", text: rest });
      break;
    }
    candidates.sort((a, b) => a.start - b.start);
    const first = candidates[0];
    if (first.start > 0) {
      tokens.push({ kind: "text", text: rest.slice(0, first.start) });
    }
    if (first.kind === "code") {
      tokens.push({ kind: "code", text: first.m[1] });
    } else if (first.kind === "bold") {
      tokens.push({ kind: "bold", text: first.m[1] });
    } else if (first.kind === "link") {
      tokens.push({ kind: "link", text: first.m[1], href: first.m[2] });
    }
    rest = rest.slice(first.start + first.m[0].length);
  }
  return tokens;
}

export function parseTutorial(source) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line
    if (/^\s*$/.test(line)) { i++; continue; }

    // Horizontal rule
    if (/^---+\s*$/.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    // Headings
    const h1 = line.match(/^# (.*)$/);
    if (h1) { blocks.push({ type: "h1", text: h1[1] }); i++; continue; }
    const h2 = line.match(/^## (.*)$/);
    if (h2) { blocks.push({ type: "h2", text: h2[1] }); i++; continue; }
    const h3 = line.match(/^### (.*)$/);
    if (h3) { blocks.push({ type: "h3", text: h3[1] }); i++; continue; }
    const h4 = line.match(/^#### (.*)$/);
    if (h4) { blocks.push({ type: "h4", text: h4[1] }); i++; continue; }

    // Fenced code block
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const lang = fence[1] || "";
      const codeLines = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing fence
      // Look at the block immediately preceding this one. If it's a bold
      // marker like "Expected" / "Output" / "Result", the code is static
      // output, not a runnable command. This decorates the block so the
      // renderer can pick the right rendering.
      const preceding = blocks[blocks.length - 1];
      let precedingLabel = null;
      if (preceding && preceding.type === "p" && Array.isArray(preceding.inline)) {
        const boldTokens = preceding.inline.filter((t) => t.kind === "bold");
        if (boldTokens.length > 0) {
          precedingLabel = boldTokens[0].text.trim().toLowerCase();
        }
      }
      blocks.push({ type: "code", lang, code: codeLines.join("\n"), precedingLabel });
      continue;
    }

    // Table (very simple: | col | col |)
    if (/^\|/.test(line) && i + 1 < lines.length && /^\|[-|\s:]+\|/.test(lines[i + 1])) {
      const header = line.split("|").slice(1, -1).map((s) => parseInline(s.trim()));
      i += 2; // skip header and separator
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) {
        rows.push(lines[i].split("|").slice(1, -1).map((s) => parseInline(s.trim())));
        i++;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // Unordered list
    if (/^[-*] /.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        // A list item may continue on subsequent indented lines
        let text = lines[i].replace(/^[-*] /, "");
        i++;
        while (i < lines.length && /^  \S/.test(lines[i])) {
          text += " " + lines[i].trim();
          i++;
        }
        items.push(parseInline(text));
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    // Blockquote
    if (/^> /.test(line)) {
      const buf = [];
      while (i < lines.length && /^> /.test(lines[i])) {
        buf.push(lines[i].replace(/^> /, ""));
        i++;
      }
      blocks.push({ type: "blockquote", inline: parseInline(buf.join(" ")) });
      continue;
    }

    // Paragraph — consume until blank line or block-starter
    const pBuf = [line];
    i++;
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^#{1,4} /.test(lines[i]) &&
      !/^```/.test(lines[i]) &&
      !/^---+\s*$/.test(lines[i]) &&
      !/^\|/.test(lines[i]) &&
      !/^[-*] /.test(lines[i]) &&
      !/^> /.test(lines[i])
    ) {
      pBuf.push(lines[i]);
      i++;
    }
    blocks.push({ type: "p", inline: parseInline(pBuf.join(" ")) });
  }

  return blocks;
}

/**
 * Extract the first h1 as the title, and the first paragraph after it as
 * the description, from a parsed block list.
 */
export function extractMeta(blocks) {
  let title = "";
  let description = "";
  for (const b of blocks) {
    if (!title && b.type === "h1") { title = b.text; continue; }
    if (title && !description && b.type === "p") {
      description = b.inline.map((t) => t.text).join("");
      break;
    }
  }
  return { title, description };
}
