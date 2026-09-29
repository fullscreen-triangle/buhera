/**
 * st-Hurbert compiler — parser. Recursive-descent over the token stream, producing
 * a validated AST. Reports precise (line:col) errors, which the compiler surfaces
 * to the user the way the framework's DSL generator relies on a real compiler as
 * ground truth (empty-dictionary principle: the language is judged, not memorised).
 */

import { Tok, Token } from "./lexer";
import {
  Cond, Field, FIELDS, Op, Program, ShowTarget, SHOW_TARGETS, Stmt,
} from "./ast";

export class ParseError extends Error {
  constructor(message: string, public line: number, public col: number) {
    super(`parse error (${line}:${col}): ${message}`);
    this.name = "ParseError";
  }
}

const OPS: Op[] = ["==", "!=", "~", "contains", ">", "<", ">=", "<="];

export function parse(tokens: Token[]): Program {
  return new Parser(tokens).parseProgram();
}

class Parser {
  private pos = 0;
  constructor(private toks: Token[]) {}

  private peek(): Token {
    return this.toks[this.pos];
  }
  private next(): Token {
    return this.toks[this.pos++];
  }
  private at(type: Tok, value?: string): boolean {
    const t = this.peek();
    return t.type === type && (value === undefined || t.value === value);
  }
  private eat(type: Tok, value?: string): Token {
    const t = this.peek();
    if (t.type !== type || (value !== undefined && t.value !== value)) {
      throw new ParseError(
        `expected ${value ?? type}, found ${t.type === Tok.EOF ? "end of input" : JSON.stringify(t.value)}`,
        t.line,
        t.col
      );
    }
    return this.next();
  }

  parseProgram(): Program {
    const stmts: Stmt[] = [];
    this.skipSep();
    while (!this.at(Tok.EOF)) {
      stmts.push(this.parseStmt());
      // statements separated by `;`, newlines are already whitespace → require a
      // separator or EOF between statements.
      if (!this.at(Tok.EOF)) this.skipSep();
    }
    if (stmts.length === 0) {
      const t = this.peek();
      throw new ParseError("empty program", t.line, t.col);
    }
    return stmts;
  }

  private skipSep() {
    while (this.at(Tok.Semi)) this.next();
  }

  private parseStmt(): Stmt {
    const t = this.peek();
    if (t.type !== Tok.Ident) {
      throw new ParseError(`expected a command (navigate/slice/show/find/compose), found ${JSON.stringify(t.value)}`, t.line, t.col);
    }
    switch (t.value) {
      case "navigate": return this.parseNavigate();
      case "slice": return this.parseSlice();
      case "show": return this.parseShow();
      case "find": return this.parseFind();
      case "compose": this.next(); return this.parseStmt(); // `compose` is sugar for sequencing
      default:
        throw new ParseError(`unknown command ${JSON.stringify(t.value)} (expected navigate/slice/show/find/compose)`, t.line, t.col);
    }
  }

  private parseNavigate(): Stmt {
    this.eat(Tok.Ident, "navigate");
    if (this.at(Tok.Star)) {
      this.next();
      return { kind: "navigate", target: "*" };
    }
    const t = this.eat(Tok.Ident);
    return { kind: "navigate", target: t.value };
  }

  private parseSlice(): Stmt {
    this.eat(Tok.Ident, "slice");
    let symbolKind: string | null = null;
    // optional symbol kind (an ident that is not `where`)
    if (this.at(Tok.Ident) && this.peek().value !== "where") {
      symbolKind = this.next().value;
    }
    let cond: Cond | null = null;
    if (this.at(Tok.Ident, "where")) {
      this.next();
      cond = this.parseCond();
    }
    return { kind: "slice", symbolKind, cond };
  }

  private parseShow(): Stmt {
    this.eat(Tok.Ident, "show");
    const t = this.eat(Tok.Ident);
    if (!SHOW_TARGETS.includes(t.value as ShowTarget)) {
      throw new ParseError(
        `unknown show target ${JSON.stringify(t.value)} (expected one of ${SHOW_TARGETS.join(", ")})`,
        t.line, t.col
      );
    }
    return { kind: "show", target: t.value as ShowTarget };
  }

  private parseFind(): Stmt {
    this.eat(Tok.Ident, "find");
    const t = this.eat(Tok.String);
    return { kind: "find", text: t.value };
  }

  // cond := andCond ( "or" andCond )*
  private parseCond(): Cond {
    let left = this.parseAnd();
    while (this.at(Tok.Ident, "or")) {
      this.next();
      const right = this.parseAnd();
      left = { kind: "or", left, right };
    }
    return left;
  }

  // andCond := cmp ( "and" cmp )*
  private parseAnd(): Cond {
    let left = this.parseCmp();
    while (this.at(Tok.Ident, "and")) {
      this.next();
      const right = this.parseCmp();
      left = { kind: "and", left, right };
    }
    return left;
  }

  // cmp := field op value
  private parseCmp(): Cond {
    const fTok = this.eat(Tok.Ident);
    if (!FIELDS.includes(fTok.value as Field)) {
      throw new ParseError(
        `unknown field ${JSON.stringify(fTok.value)} (expected one of ${FIELDS.join(", ")})`,
        fTok.line, fTok.col
      );
    }
    const field = fTok.value as Field;

    // operator: either Tok.Op, or the keyword `contains`
    let op: Op;
    let opTok = this.peek();
    if (this.at(Tok.Op)) {
      op = this.next().value as Op;
    } else if (this.at(Tok.Ident, "contains")) {
      this.next();
      op = "contains";
    } else {
      throw new ParseError(`expected an operator (${OPS.join(" ")}), found ${JSON.stringify(opTok.value)}`, opTok.line, opTok.col);
    }

    // value: string, number, or bare ident
    const vTok = this.peek();
    let value: string | number;
    if (this.at(Tok.String)) value = this.next().value;
    else if (this.at(Tok.Number)) value = parseFloat(this.next().value);
    else if (this.at(Tok.Ident)) value = this.next().value;
    else throw new ParseError(`expected a value after ${op}, found ${JSON.stringify(vTok.value)}`, vTok.line, vTok.col);

    // numeric ops require a number
    if (([">", "<", ">=", "<="] as Op[]).includes(op) && typeof value !== "number") {
      throw new ParseError(`operator ${op} needs a numeric value, got ${JSON.stringify(value)}`, vTok.line, vTok.col);
    }
    return { kind: "cmp", field, op, value };
  }
}
