/**
 * st-Hurbert compiler — lexer.
 *
 * st-Hurbert (repo-query subset) is the *informed user's* path into the repo lens:
 * a small language that navigates and slices the analysed federation. This is a
 * genuine, self-contained TypeScript compiler (lexer → parser → interpreter); it is
 * NOT a WASM binding to the Rust tool.
 *
 * Surface (this subset):
 *   navigate <repo>              -- select a repo (or `*` for the whole federation)
 *   slice <kind> where <cond>    -- filter symbols by kind / predicate
 *   show sense | chi | salient | fragments | files | symbols
 *   show lineage | regime | health   -- image-knowability: runtime-health-over-time
 *   find "<text>"                -- search symbol names/snippets
 *   compose <stmt> ; <stmt>      -- sequencing (also plain newlines / ';')
 * Conditions: <field> <op> <value>   field ∈ {name,kind,file,line}
 *             op ∈ { == != ~ contains > < >= <= }, combfinable with `and`/`or`.
 */

export enum Tok {
  Ident = "Ident",
  String = "String",
  Number = "Number",
  Op = "Op", // == != ~ > < >= <=
  Semi = "Semi",
  Star = "Star",
  EOF = "EOF",
}

export interface Token {
  type: Tok;
  value: string;
  line: number;
  col: number;
}

export class LexError extends Error {
  constructor(message: string, public line: number, public col: number) {
    super(`lex error (${line}:${col}): ${message}`);
    this.name = "LexError";
  }
}

const KEYWORDS = new Set([
  "navigate", "slice", "where", "show", "find", "compose",
  "and", "or", "contains",
  // show targets & symbol kinds are treated as idents; parser decides meaning.
]);

export function lex(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const n = src.length;

  const push = (type: Tok, value: string, c: number) =>
    tokens.push({ type, value, line, col: c });

  while (i < n) {
    const ch = src[i];

    // whitespace
    if (ch === " " || ch === "\t" || ch === "\r") { i++; col++; continue; }
    if (ch === "\n") { i++; line++; col = 1; continue; }

    // comments: `--` or `#` to end of line
    if ((ch === "-" && src[i + 1] === "-") || ch === "#") {
      while (i < n && src[i] !== "\n") i++;
      continue;
    }

    // statement separator
    if (ch === ";") { push(Tok.Semi, ";", col); i++; col++; continue; }

    // star (federation wildcard)
    if (ch === "*") { push(Tok.Star, "*", col); i++; col++; continue; }

    // strings
    if (ch === '"' || ch === "'") {
      const quote = ch;
      const startCol = col;
      i++; col++;
      let val = "";
      while (i < n && src[i] !== quote) {
        if (src[i] === "\\" && i + 1 < n) { val += src[i + 1]; i += 2; col += 2; continue; }
        if (src[i] === "\n") throw new LexError("unterminated string", line, startCol);
        val += src[i]; i++; col++;
      }
      if (i >= n) throw new LexError("unterminated string", line, startCol);
      i++; col++; // closing quote
      push(Tok.String, val, startCol);
      continue;
    }

    // numbers
    if (ch >= "0" && ch <= "9") {
      const startCol = col;
      let val = "";
      while (i < n && /[0-9.]/.test(src[i])) { val += src[i]; i++; col++; }
      push(Tok.Number, val, startCol);
      continue;
    }

    // operators
    const two = src.slice(i, i + 2);
    if (["==", "!=", ">=", "<="].includes(two)) {
      push(Tok.Op, two, col); i += 2; col += 2; continue;
    }
    if (ch === "~" || ch === ">" || ch === "<") {
      push(Tok.Op, ch, col); i++; col++; continue;
    }

    // identifiers / keywords (allow /, ., -, _ so repo names & paths lex as one ident)
    if (/[A-Za-z_]/.test(ch)) {
      const startCol = col;
      let val = "";
      while (i < n && /[A-Za-z0-9_./-]/.test(src[i])) { val += src[i]; i++; col++; }
      push(Tok.Ident, val, startCol);
      continue;
    }

    throw new LexError(`unexpected character ${JSON.stringify(ch)}`, line, col);
  }

  tokens.push({ type: Tok.EOF, value: "", line, col });
  return tokens;
}

export { KEYWORDS };
