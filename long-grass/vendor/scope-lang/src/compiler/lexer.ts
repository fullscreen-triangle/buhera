// SCOPE Lexer

export enum TokenType {
  // Keywords
  SCOPE = 'scope',
  CHANNELS = 'channels',
  SYNC = 'sync',
  CELL = 'cell',
  AT = 'at',
  BOUNDS = 'bounds',
  ACTION = 'action',
  COORDINATE_SPACE = 'coordinate_space',
  FIELD = 'field',
  DEPTH = 'depth',
  LAMBDA_S = 'lambda_s',
  LAMBDA_T = 'lambda_t',
  GOAL = 'goal',
  RULE = 'rule',
  INVARIANT = 'invariant',
  EPSILON = 'epsilon',
  DISPATCH = 'dispatch',
  WHEN = 'when',
  DO = 'do',
  EXECUTE = 'execute',
  EMIT = 'emit',
  OBSERVE = 'observe',
  CATALYZE = 'catalyze',
  FUSE = 'fuse',
  MEASURE_DISTANCE = 'measure_distance',
  ACCESS = 'access',
  VISUALISE = 'visualise',
  LOAD = 'load',
  DB = 'db',
  DATASET = 'dataset',
  IMAGE = 'image',
  N = 'n',
  RHO = 'rho',
  CONFIDENCE = 'confidence',
  THRESHOLD = 'threshold',
  WITHIN = 'within',
  // Literals
  IDENT = 'IDENT',
  NUMBER = 'NUMBER',
  STRING = 'STRING',
  UNIT = 'UNIT',
  // Punctuation
  LBRACE = '{',
  RBRACE = '}',
  LPAREN = '(',
  RPAREN = ')',
  EQUALS = '=',
  PIPE = '|>',
  COMMA = ',',
  COLON = ':',
  LT = '<',
  LTE = '<=',
  GT = '>',
  GTE = '>=',
  EQEQ = '==',
  X = 'x',
  EOF = 'EOF',
}

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  col: number;
}

const KEYWORDS: Record<string, TokenType> = {
  scope: TokenType.SCOPE,
  channels: TokenType.CHANNELS,
  sync: TokenType.SYNC,
  cell: TokenType.CELL,
  at: TokenType.AT,
  bounds: TokenType.BOUNDS,
  action: TokenType.ACTION,
  coordinate_space: TokenType.COORDINATE_SPACE,
  field: TokenType.FIELD,
  depth: TokenType.DEPTH,
  lambda_s: TokenType.LAMBDA_S,
  lambda_t: TokenType.LAMBDA_T,
  goal: TokenType.GOAL,
  rule: TokenType.RULE,
  invariant: TokenType.INVARIANT,
  epsilon: TokenType.EPSILON,
  dispatch: TokenType.DISPATCH,
  when: TokenType.WHEN,
  do: TokenType.DO,
  execute: TokenType.EXECUTE,
  emit: TokenType.EMIT,
  observe: TokenType.OBSERVE,
  catalyze: TokenType.CATALYZE,
  fuse: TokenType.FUSE,
  measure_distance: TokenType.MEASURE_DISTANCE,
  access: TokenType.ACCESS,
  visualise: TokenType.VISUALISE,
  load: TokenType.LOAD,
  db: TokenType.DB,
  dataset: TokenType.DATASET,
  image: TokenType.IMAGE,
  n: TokenType.N,
  rho: TokenType.RHO,
  confidence: TokenType.CONFIDENCE,
  threshold: TokenType.THRESHOLD,
  within: TokenType.WITHIN,
  x: TokenType.X,
};

const UNITS = new Set(['µm', 'nm', 'px', 'freq', 's', 'bits', '%', 'µm/pixel']);

export class Lexer {
  private pos = 0;
  private line = 1;
  private col = 1;

  constructor(private src: string) {}

  tokenize(): Token[] {
    const tokens: Token[] = [];
    while (this.pos < this.src.length) {
      this.skipWS();
      if (this.pos >= this.src.length) break;

      const ch = this.ch();
      const startLine = this.line;
      const startCol = this.col;

      if (ch === '/' && this.peek() === '/') {
        while (this.pos < this.src.length && this.ch() !== '\n') this.advance();
        continue;
      }

      if (ch === '{') { tokens.push(this.tok(TokenType.LBRACE, '{', startLine, startCol)); this.advance(); continue; }
      if (ch === '}') { tokens.push(this.tok(TokenType.RBRACE, '}', startLine, startCol)); this.advance(); continue; }
      if (ch === '(') { tokens.push(this.tok(TokenType.LPAREN, '(', startLine, startCol)); this.advance(); continue; }
      if (ch === ')') { tokens.push(this.tok(TokenType.RPAREN, ')', startLine, startCol)); this.advance(); continue; }
      if (ch === ',') { tokens.push(this.tok(TokenType.COMMA, ',', startLine, startCol)); this.advance(); continue; }
      if (ch === ':') { tokens.push(this.tok(TokenType.COLON, ':', startLine, startCol)); this.advance(); continue; }

      if (ch === '|' && this.peek() === '>') {
        tokens.push(this.tok(TokenType.PIPE, '|>', startLine, startCol));
        this.advance(); this.advance();
        continue;
      }

      if (ch === '<' && this.peek() === '=') {
        tokens.push(this.tok(TokenType.LTE, '<=', startLine, startCol));
        this.advance(); this.advance();
        continue;
      }
      if (ch === '>' && this.peek() === '=') {
        tokens.push(this.tok(TokenType.GTE, '>=', startLine, startCol));
        this.advance(); this.advance();
        continue;
      }
      if (ch === '=' && this.peek() === '=') {
        tokens.push(this.tok(TokenType.EQEQ, '==', startLine, startCol));
        this.advance(); this.advance();
        continue;
      }
      if (ch === '<') { tokens.push(this.tok(TokenType.LT, '<', startLine, startCol)); this.advance(); continue; }
      if (ch === '>') { tokens.push(this.tok(TokenType.GT, '>', startLine, startCol)); this.advance(); continue; }
      if (ch === '=') { tokens.push(this.tok(TokenType.EQUALS, '=', startLine, startCol)); this.advance(); continue; }

      if (ch === '"') { tokens.push(this.readString(startLine, startCol)); continue; }

      if (ch === '-' || /\d/.test(ch)) { tokens.push(this.readNumber(startLine, startCol)); continue; }

      if (/[a-zA-Z_µ]/.test(ch)) { tokens.push(this.readWord(startLine, startCol)); continue; }

      throw new Error(`Unexpected character '${ch}' at ${this.line}:${this.col}`);
    }
    tokens.push(this.tok(TokenType.EOF, '', this.line, this.col));
    return tokens;
  }

  private ch(): string { return this.src[this.pos] ?? ''; }
  private peek(n = 1): string { return this.src[this.pos + n] ?? ''; }
  private advance(): void {
    if (this.src[this.pos] === '\n') { this.line++; this.col = 1; } else { this.col++; }
    this.pos++;
  }
  private skipWS(): void {
    while (this.pos < this.src.length && /\s/.test(this.src[this.pos])) this.advance();
  }
  private tok(type: TokenType, value: string, line: number, col: number): Token {
    return { type, value, line, col };
  }

  private readString(line: number, col: number): Token {
    this.advance(); // skip "
    let s = '';
    while (this.pos < this.src.length && this.ch() !== '"') {
      if (this.ch() === '\\') { this.advance(); s += this.ch(); } else { s += this.ch(); }
      this.advance();
    }
    if (this.ch() !== '"') throw new Error(`Unterminated string at ${line}:${col}`);
    this.advance();
    return { type: TokenType.STRING, value: s, line, col };
  }

  private readNumber(line: number, col: number): Token {
    let s = '';
    if (this.ch() === '-') { s += '-'; this.advance(); }
    while (/\d/.test(this.ch())) { s += this.ch(); this.advance(); }
    if (this.ch() === '.') { s += '.'; this.advance(); while (/\d/.test(this.ch())) { s += this.ch(); this.advance(); } }
    if (/[eE]/.test(this.ch())) {
      s += this.ch(); this.advance();
      if (/[+\-]/.test(this.ch())) { s += this.ch(); this.advance(); }
      while (/\d/.test(this.ch())) { s += this.ch(); this.advance(); }
    }
    return { type: TokenType.NUMBER, value: s, line, col };
  }

  private readWord(line: number, col: number): Token {
    let s = '';
    // µm/pixel needs special handling — read ident chars including µ and /
    while (/[a-zA-Z0-9_µ/]/.test(this.ch())) { s += this.ch(); this.advance(); }
    const kw = KEYWORDS[s];
    if (kw !== undefined) return { type: kw, value: s, line, col };
    if (UNITS.has(s)) return { type: TokenType.UNIT, value: s, line, col };
    return { type: TokenType.IDENT, value: s, line, col };
  }
}
