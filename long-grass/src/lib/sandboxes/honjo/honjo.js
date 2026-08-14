// src/lexer.ts
var KEYWORDS = /* @__PURE__ */ new Set([
  "floor",
  "cut",
  "close",
  "track",
  "until",
  "yield",
  "when",
  "do",
  "emit",
  "observe",
  "in",
  "as",
  "let",
  "medium",
  "converge",
  "diverge",
  "with",
  "by",
  "import",
  "module",
  "export",
  "assert",
  "reps"
]);
var MULTI_OPS = [":=", ">=", "<=", "==", "->"];
var SINGLE_OPS = /* @__PURE__ */ new Set([
  "~",
  "(",
  ")",
  "[",
  "]",
  "{",
  "}",
  ",",
  ":",
  ".",
  ">",
  "<",
  "#"
]);
var LexError = class extends Error {
  constructor(msg, line, col) {
    super(`lex error at ${line}:${col}: ${msg}`);
    this.line = line;
    this.col = col;
    this.name = "LexError";
  }
};
var isIdentStart = (c) => /[A-Za-z_]/.test(c);
var isIdentPart = (c) => /[A-Za-z0-9_]/.test(c);
var isDigit = (c) => c >= "0" && c <= "9";
function lex(src) {
  const toks = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const n = src.length;
  const peek = (k = 0) => src[i + k] ?? "";
  const adv = () => {
    const c = src[i++];
    if (c === "\n") {
      line++;
      col = 1;
    } else {
      col++;
    }
    return c;
  };
  while (i < n) {
    const c = peek();
    if (c === " " || c === "	" || c === "\r" || c === "\n") {
      adv();
      continue;
    }
    if (c === "-" && peek(1) === "-") {
      while (i < n && peek() !== "\n") adv();
      continue;
    }
    const startLine = line, startCol = col;
    if (c === '"') {
      adv();
      let s = "";
      while (i < n && peek() !== '"') {
        const ch = adv();
        if (ch === "\\") {
          const e = adv();
          s += e === "n" ? "\n" : e === "t" ? "	" : e;
        } else {
          s += ch;
        }
      }
      if (i >= n) throw new LexError("unterminated string", startLine, startCol);
      adv();
      toks.push({ kind: "str", value: s, line: startLine, col: startCol });
      continue;
    }
    if (isDigit(c) || c === "." && isDigit(peek(1))) {
      let s = "";
      while (isDigit(peek())) s += adv();
      if (peek() === ".") {
        s += adv();
        while (isDigit(peek())) s += adv();
      }
      if (peek() === "e" || peek() === "E") {
        s += adv();
        if (peek() === "+" || peek() === "-") s += adv();
        if (!isDigit(peek())) throw new LexError("malformed exponent", startLine, startCol);
        while (isDigit(peek())) s += adv();
      }
      toks.push({ kind: "num", value: s, line: startLine, col: startCol });
      continue;
    }
    if (isIdentStart(c)) {
      let s = "";
      while (isIdentPart(peek())) s += adv();
      toks.push({ kind: KEYWORDS.has(s) ? "kw" : "ident", value: s, line: startLine, col: startCol });
      continue;
    }
    const two = c + peek(1);
    if (MULTI_OPS.includes(two)) {
      adv();
      adv();
      toks.push({ kind: "op", value: two, line: startLine, col: startCol });
      continue;
    }
    if (SINGLE_OPS.has(c)) {
      adv();
      toks.push({ kind: "op", value: c, line: startLine, col: startCol });
      continue;
    }
    throw new LexError(`unexpected character '${c}'`, startLine, startCol);
  }
  toks.push({ kind: "eof", value: "", line, col });
  return toks;
}

// src/parser.ts
var ParseError = class extends Error {
  constructor(msg, line, col) {
    super(`parse error at ${line}:${col}: ${msg}`);
    this.line = line;
    this.col = col;
    this.name = "ParseError";
  }
};
var RELOPS = /* @__PURE__ */ new Set([">", "<", ">=", "<=", "=="]);
var Parser = class {
  constructor(toks) {
    this.toks = toks;
    this.p = 0;
  }
  peek(k = 0) {
    return this.toks[this.p + k];
  }
  at(kind, value) {
    const t = this.peek();
    if (t.kind !== kind && !(kind === "kw" && t.kind === "kw")) {
    }
    return t.kind === kind && (value === void 0 || t.value === value);
  }
  isKw(v) {
    const t = this.peek();
    return t.kind === "kw" && t.value === v;
  }
  isOp(v) {
    const t = this.peek();
    return t.kind === "op" && t.value === v;
  }
  adv() {
    return this.toks[this.p++];
  }
  pos(t = this.peek()) {
    return { line: t.line, col: t.col };
  }
  expectOp(v) {
    if (!this.isOp(v)) this.fail(`expected '${v}'`);
    return this.adv();
  }
  expectKw(v) {
    if (!this.isKw(v)) this.fail(`expected keyword '${v}'`);
    return this.adv();
  }
  expectIdent() {
    const t = this.peek();
    if (t.kind !== "ident") this.fail("expected identifier");
    return this.adv().value;
  }
  fail(msg) {
    const t = this.peek();
    throw new ParseError(`${msg}, found ${t.kind} '${t.value}'`, t.line, t.col);
  }
  // program ::= { decl }
  parseProgram() {
    const decls = [];
    while (!this.at("eof")) decls.push(this.parseDecl());
    return { decls };
  }
  parseDecl() {
    if (this.isKw("floor")) return this.parseFloor();
    if (this.isKw("import")) return this.parseImport();
    if (this.isKw("module")) return this.parseModule();
    return this.parseStmt();
  }
  parseFloor() {
    const pos = this.pos();
    this.expectKw("floor");
    const value = this.parseNumberValue();
    return { tag: "floor", value, pos };
  }
  parseImport() {
    const pos = this.pos();
    this.expectKw("import");
    return { tag: "import", name: this.parseQName(), pos };
  }
  parseModule() {
    const pos = this.pos();
    this.expectKw("module");
    const name = this.expectIdent();
    this.expectOp("{");
    const body = [];
    while (!this.isOp("}") && !this.at("eof")) body.push(this.parseDecl());
    this.expectOp("}");
    return { tag: "module", name, body, pos };
  }
  // stmt ::= bind | track | observe | assert | expr
  parseStmt() {
    if (this.isKw("track")) return this.parseTrack();
    if (this.isKw("observe")) return this.parseObserve();
    if (this.isKw("assert")) return this.parseAssert();
    if (this.isKw("let")) {
      this.adv();
      return this.parseBindAfterName();
    }
    if (this.peek().kind === "ident" && this.peek(1).kind === "op" && this.peek(1).value === ":=") {
      return this.parseBindAfterName();
    }
    const pos = this.pos();
    return { tag: "exprStmt", expr: this.parseExpr(), pos };
  }
  parseBindAfterName() {
    const pos = this.pos();
    const name = this.expectIdent();
    this.expectOp(":=");
    if (this.isKw("track")) {
      return { tag: "bind", name, value: this.parseTrackExpr(), pos };
    }
    return { tag: "bind", name, value: this.parseExpr(), pos };
  }
  // track as an expression: track ident in expr [with reps ids] until <admit> [yield ident]
  parseTrackExpr() {
    const pos = this.pos();
    this.expectKw("track");
    const item = this.expectIdent();
    this.expectKw("in");
    const process = this.parseExpr();
    let reps;
    if (this.isKw("with")) {
      this.adv();
      this.expectKw("reps");
      reps = [this.expectIdent()];
      while (this.isOp(",")) {
        this.adv();
        reps.push(this.expectIdent());
      }
    }
    this.expectKw("until");
    const admit = this.parseAdmit();
    let role;
    if (this.isKw("yield")) {
      this.adv();
      role = this.expectIdent();
    }
    return { tag: "trackExpr", item, process, reps, admit, role, pos };
  }
  parseObserve() {
    const pos = this.pos();
    this.expectKw("observe");
    const expr = this.parseExpr();
    let asName;
    if (this.isKw("as")) {
      this.adv();
      asName = this.expectIdent();
    }
    return { tag: "observe", expr, as: asName, pos };
  }
  parseAssert() {
    const pos = this.pos();
    this.expectKw("assert");
    const cond = this.parseCond();
    let emit;
    if (this.isKw("emit")) {
      this.adv();
      emit = this.parseStringValue();
    }
    return { tag: "assert", cond, emit, pos };
  }
  // track ident in expr [with reps idlist] until <admit> yield ident
  parseTrack() {
    const pos = this.pos();
    this.expectKw("track");
    const item = this.expectIdent();
    this.expectKw("in");
    const process = this.parseExpr();
    let reps;
    if (this.isKw("with")) {
      this.adv();
      this.expectKw("reps");
      reps = [this.expectIdent()];
      while (this.isOp(",")) {
        this.adv();
        reps.push(this.expectIdent());
      }
    }
    this.expectKw("until");
    const admit = this.parseAdmit();
    this.expectKw("yield");
    const yieldName = this.expectIdent();
    return { tag: "track", item, process, reps, admit, yieldName, pos };
  }
  parseAdmit() {
    if (this.isKw("converge")) {
      this.adv();
      return { kind: "converge" };
    }
    if (this.isKw("diverge")) {
      this.adv();
      return { kind: "diverge" };
    }
    return { kind: "cond", cond: this.parseCond() };
  }
  // ---- expressions ----
  parseExpr() {
    let e = this.parsePrimary();
    while (this.isOp("~")) {
      const pos = this.pos();
      this.adv();
      const right = this.parsePrimary();
      let guard;
      if (this.isKw("when")) {
        this.adv();
        guard = this.parseCond();
      }
      e = { tag: "bond", left: e, right, guard, pos };
    }
    return e;
  }
  parsePrimary() {
    const t = this.peek();
    const pos = this.pos();
    if (this.isKw("cut")) {
      this.adv();
      return { tag: "cut", arg: this.parsePrimary(), pos };
    }
    if (this.isKw("close")) {
      this.adv();
      const central = this.expectIdent();
      this.expectOp("(");
      const args = this.parseArgList();
      this.expectOp(")");
      let by;
      if (this.isKw("by")) {
        this.adv();
        by = this.expectIdent();
      }
      return { tag: "close", central, args, by, pos };
    }
    if (this.isOp("(")) {
      this.adv();
      const e = this.parseExpr();
      this.expectOp(")");
      return e;
    }
    if (t.kind === "num") {
      const value = this.parseNumberValue();
      return { tag: "num", value, floor: this.lastFloor, pos };
    }
    if (t.kind === "str") {
      return { tag: "str", value: this.parseStringValue(), pos };
    }
    if (t.kind === "ident") {
      const name = this.parseQName();
      if (this.isOp("(")) {
        this.adv();
        const args = this.parseArgList();
        this.expectOp(")");
        return { tag: "call", name, args, pos };
      }
      if (name.length === 1) return { tag: "ref", name: name[0], pos };
      return { tag: "call", name, args: [], pos };
    }
    this.fail("expected expression");
  }
  parseArgList() {
    const args = [];
    if (this.isOp(")")) return args;
    args.push(this.parseArg());
    while (this.isOp(",")) {
      this.adv();
      args.push(this.parseArg());
    }
    return args;
  }
  parseArg() {
    if (this.peek().kind === "ident" && this.peek(1).kind === "op" && this.peek(1).value === ":") {
      const label = this.adv().value;
      this.adv();
      return { label, value: this.parseExpr() };
    }
    return { value: this.parseExpr() };
  }
  parseCond() {
    const pos = this.pos();
    const left = this.parsePrimary();
    const t = this.peek();
    if (t.kind !== "op" || !RELOPS.has(t.value)) this.fail("expected relational operator");
    const op = this.adv().value;
    const right = this.parsePrimary();
    return { tag: "cond", left, op, right, pos };
  }
  parseQName() {
    const parts = [this.expectIdent()];
    while (this.isOp(".")) {
      this.adv();
      parts.push(this.expectIdent());
    }
    return parts;
  }
  parseNumberValue() {
    const t = this.peek();
    if (t.kind !== "num") this.fail("expected number");
    this.adv();
    let floorVal;
    if (this.isOp("#")) {
      this.adv();
      const ft = this.peek();
      if (ft.kind !== "num") this.fail("expected floor literal after '#'");
      this.adv();
      floorVal = parseFloat(ft.value);
    }
    this.lastFloor = floorVal;
    return parseFloat(t.value);
  }
  parseStringValue() {
    const t = this.peek();
    if (t.kind !== "str") this.fail("expected string");
    this.adv();
    return t.value;
  }
};
function parse(src) {
  return new Parser(lex(src)).parseProgram();
}

// src/types.ts
var TypeError_ = class extends Error {
  constructor(msg, line, col) {
    super(`type error at ${line}:${col}: ${msg}`);
    this.line = line;
    this.col = col;
    this.name = "TypeError";
  }
};
var CUT_LIKE = /* @__PURE__ */ new Set(["Atom", "Bond", "Compound", "Path", "Cut"]);
function check(program) {
  const env = { ambientFloor: NaN, vars: /* @__PURE__ */ new Map() };
  const bindings = /* @__PURE__ */ new Map();
  const ambientFloorAt = /* @__PURE__ */ new Map();
  function fail(msg, pos) {
    throw new TypeError_(msg, pos.line, pos.col);
  }
  function requirePositiveFloor(f, pos) {
    if (!(f > 0)) fail(`floor must be > 0 (got ${f}); the sharp cut is not expressible`, pos);
    return f;
  }
  function checkExpr(e) {
    switch (e.tag) {
      case "num": {
        const f = e.floor !== void 0 ? e.floor : env.ambientFloor;
        if (Number.isNaN(f)) fail("numeric literal used before any 'floor' declaration", e.pos);
        requirePositiveFloor(f, e.pos);
        return { name: "Scalar", floor: f };
      }
      case "str":
        return { name: "Void", floor: env.ambientFloor };
      case "ref": {
        const t = env.vars.get(e.name);
        if (!t) {
          fail(`unbound identifier '${e.name}'`, e.pos);
        }
        return t;
      }
      case "cut": {
        const at = checkExpr(e.arg);
        if (at.name !== "Scalar") fail("cut expects an atomic number (Scalar)", e.pos);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Atom", floor: f };
      }
      case "bond": {
        const lt = checkExpr(e.left);
        const rt = checkExpr(e.right);
        if (!CUT_LIKE.has(lt.name) || !CUT_LIKE.has(rt.name))
          fail("a bond (~) joins two cut-like values (Atom/Compound)", e.pos);
        if (e.guard) checkCond(e.guard);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Bond", floor: f };
      }
      case "close": {
        const ct = env.vars.get(e.central);
        if (!ct) fail(`unbound central atom '${e.central}'`, e.pos);
        if (!CUT_LIKE.has(ct.name)) fail("close expects an Atom as the central item", e.pos);
        for (const a of e.args) {
          const t = checkExpr(a.value);
          if (!CUT_LIKE.has(t.name)) fail("close ligands must be cut-like (Atom)", e.pos);
        }
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Compound", floor: f };
      }
      case "trackExpr": {
        const it = env.vars.get(e.item);
        if (!it) fail(`tracking unbound item '${e.item}'`, e.pos);
        if (!CUT_LIKE.has(it.name)) fail("track expects an Atom/Compound item", e.pos);
        checkExpr(e.process);
        if (e.admit.kind === "cond") checkCond(e.admit.cond);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Path", floor: f };
      }
      case "call": {
        for (const a of e.args) checkExpr(a.value);
        const f = Number.isNaN(env.ambientFloor) ? 1 : env.ambientFloor;
        const verb = e.name[e.name.length - 1];
        const resultName = verb === "atom" || verb === "individuate" ? "Atom" : verb === "bond" ? "Bond" : verb === "close" || verb === "compound" ? "Compound" : verb === "track" || verb === "propagate" || verb === "amalgamation" ? "Path" : "Cut";
        return { name: resultName, floor: f };
      }
    }
  }
  function checkCond(c) {
    typeCondOperand(c.left);
    typeCondOperand(c.right);
    return { name: "Bool", floor: env.ambientFloor };
  }
  function typeCondOperand(e) {
    if (e.tag === "ref" || e.tag === "call") return;
    checkExpr(e);
  }
  function checkStmt(s) {
    ambientFloorAt.set(s, env.ambientFloor);
    switch (s.tag) {
      case "bind": {
        const t = checkExpr(s.value);
        env.vars.set(s.name, t);
        bindings.set(s.name, t);
        break;
      }
      case "exprStmt":
        checkExpr(s.expr);
        break;
      case "observe":
        {
          const t = checkExpr(s.expr);
          if (s.as) env.vars.set(s.as, t);
        }
        break;
      case "assert":
        checkCond(s.cond);
        break;
      case "track": {
        const it = env.vars.get(s.item);
        if (!it) fail(`tracking unbound item '${s.item}'`, s.pos);
        if (!CUT_LIKE.has(it.name)) fail("track expects an Atom/Compound item", s.pos);
        checkExpr(s.process);
        if (s.admit.kind === "cond") checkCond(s.admit.cond);
        const f = requirePositiveFloor(env.ambientFloor, s.pos);
        env.vars.set(s.yieldName, { name: "Path", floor: f });
        bindings.set(s.yieldName, { name: "Path", floor: f });
        break;
      }
    }
  }
  function checkDecl(d) {
    switch (d.tag) {
      case "floor":
        ambientFloorAt.set(d, d.value);
        requirePositiveFloor(d.value, d.pos);
        env.ambientFloor = d.value;
        break;
      case "import":
        break;
      case "module":
        for (const inner of d.body) checkDecl(inner);
        break;
      default:
        checkStmt(d);
    }
  }
  for (const d of program.decls) checkDecl(d);
  return { program, bindings, ambientFloorAt };
}

// src/lower.ts
var LowerError = class extends Error {
  constructor(msg) {
    super(`lower error: ${msg}`);
    this.name = "LowerError";
  }
};
var Lowering = class {
  constructor() {
    this.instrs = [];
    this.nameToReg = /* @__PURE__ */ new Map();
    this.next = 0;
    this.floor = NaN;
  }
  reg() {
    return this.next++;
  }
  lower(p) {
    for (const d of p.decls) this.lowerDecl(d);
    return { instrs: this.instrs, nameToReg: this.nameToReg, floor: this.floor };
  }
  lowerDecl(d) {
    switch (d.tag) {
      case "floor":
        this.floor = d.value;
        this.instrs.push({ op: "Floor", value: d.value });
        break;
      case "import":
        break;
      // imports affect name resolution only; stdlib is always available
      case "module":
        for (const inner of d.body) this.lowerDecl(inner);
        break;
      default:
        this.lowerStmt(d);
    }
  }
  lowerStmt(s) {
    switch (s.tag) {
      case "bind": {
        const r = this.lowerExpr(s.value);
        this.nameToReg.set(s.name, r);
        break;
      }
      case "exprStmt":
        this.lowerExpr(s.expr);
        break;
      case "observe": {
        const r = this.lowerExpr(s.expr);
        this.instrs.push({ op: "Obs", reg: r, as: s.as ? this.bindName(s.as, r) : void 0 });
        break;
      }
      case "assert":
        this.instrs.push({ op: "Assert", cond: this.lowerCond(s.cond), message: s.emit });
        break;
      case "track": {
        const itemReg = this.lookup(s.item);
        const procReg = this.lowerExpr(s.process);
        const dst = this.reg();
        const admit = this.lowerAdmit(s.admit);
        this.instrs.push({
          op: "Prp",
          dst,
          item: itemReg,
          process: procReg,
          reps: s.reps,
          admit,
          ty: "Path",
          floor: this.floor
        });
        this.nameToReg.set(s.yieldName, dst);
        break;
      }
    }
  }
  bindName(name, reg) {
    this.nameToReg.set(name, reg);
    return reg;
  }
  lookup(name) {
    const r = this.nameToReg.get(name);
    if (r === void 0) throw new LowerError(`unbound name '${name}'`);
    return r;
  }
  lowerExpr(e) {
    switch (e.tag) {
      case "num": {
        const dst = this.reg();
        this.instrs.push({ op: "Mov", dst, src: { kind: "lit", value: e.value, floor: e.floor ?? this.floor } });
        return dst;
      }
      case "str": {
        const dst = this.reg();
        this.instrs.push({ op: "Mov", dst, src: { kind: "lit", value: NaN, floor: this.floor } });
        return dst;
      }
      case "ref":
        return this.lookup(e.name);
      case "cut": {
        const z = this.operand(e.arg);
        const dst = this.reg();
        this.instrs.push({ op: "Ind", dst, z, ty: "Atom", floor: this.floor });
        return dst;
      }
      case "bond": {
        const a = this.lowerExpr(e.left);
        const b = this.lowerExpr(e.right);
        const dst = this.reg();
        const guard = e.guard ? this.lowerCond(e.guard) : void 0;
        this.instrs.push({ op: "Bnd", dst, a, b, guard, ty: "Bond", floor: this.floor });
        return dst;
      }
      case "close": {
        const central = this.lookup(e.central);
        const args = e.args.map((a) => this.lowerExpr(a.value));
        const dst = this.reg();
        this.instrs.push({ op: "Cls", dst, central, args, ty: "Compound", floor: this.floor });
        return dst;
      }
      case "trackExpr": {
        const itemReg = this.lookup(e.item);
        const procReg = this.lowerExpr(e.process);
        const dst = this.reg();
        this.instrs.push({
          op: "Prp",
          dst,
          item: itemReg,
          process: procReg,
          reps: e.reps,
          admit: this.lowerAdmit(e.admit),
          ty: "Path",
          floor: this.floor
        });
        return dst;
      }
      case "call": {
        const args = e.args.map((a) => this.operand(a.value));
        const dst = this.reg();
        const verb = e.name[e.name.length - 1];
        const ty = verb === "atom" || verb === "individuate" ? "Atom" : verb === "bond" ? "Bond" : verb === "close" || verb === "compound" ? "Compound" : verb === "track" || verb === "propagate" || verb === "amalgamation" ? "Path" : "Cut";
        this.instrs.push({ op: "Call", dst, name: e.name, args, ty, floor: this.floor });
        return dst;
      }
    }
  }
  operand(e) {
    switch (e.tag) {
      case "num":
        return { kind: "lit", value: e.value, floor: e.floor ?? this.floor };
      case "ref": {
        const r = this.nameToReg.get(e.name);
        if (r !== void 0) return { kind: "reg", id: r };
        return { kind: "name", name: e.name };
      }
      case "call": {
        if (e.name.length === 2 && e.args.length === 0) {
          const base = this.nameToReg.get(e.name[0]);
          if (base !== void 0) return { kind: "field", reg: base, field: e.name[1] };
        }
        const r = this.lowerExpr(e);
        return { kind: "reg", id: r };
      }
      default: {
        const r = this.lowerExpr(e);
        return { kind: "reg", id: r };
      }
    }
  }
  lowerCond(c) {
    return { left: this.operand(c.left), op: c.op, right: this.operand(c.right) };
  }
  lowerAdmit(a) {
    if (a.kind === "converge") return { kind: "converge" };
    if (a.kind === "diverge") return { kind: "diverge" };
    return { kind: "cond", cond: this.lowerCond(a.cond) };
  }
};
function lower(p) {
  return new Lowering().lower(p);
}

// src/stdlib.ts
var ELEMENTS = [
  { sym: "H", Z: 1, qv: 1, capV: 2, config: "1s1", term: "2S1/2" },
  { sym: "He", Z: 2, qv: 2, capV: 2, config: "1s2", term: "1S0" },
  { sym: "Li", Z: 3, qv: 1, capV: 8, config: "[He] 2s1", term: "2S1/2" },
  { sym: "Be", Z: 4, qv: 2, capV: 8, config: "[He] 2s2", term: "1S0" },
  { sym: "B", Z: 5, qv: 3, capV: 8, config: "[He] 2s2 2p1", term: "2P1/2" },
  { sym: "C", Z: 6, qv: 4, capV: 8, config: "[He] 2s2 2p2", term: "3P0" },
  { sym: "N", Z: 7, qv: 5, capV: 8, config: "[He] 2s2 2p3", term: "4S3/2" },
  { sym: "O", Z: 8, qv: 6, capV: 8, config: "[He] 2s2 2p4", term: "3P2" },
  { sym: "F", Z: 9, qv: 7, capV: 8, config: "[He] 2s2 2p5", term: "2P3/2" },
  { sym: "Ne", Z: 10, qv: 8, capV: 8, config: "[He] 2s2 2p6", term: "1S0" },
  { sym: "Na", Z: 11, qv: 1, capV: 8, config: "[Ne] 3s1", term: "2S1/2" },
  { sym: "Mg", Z: 12, qv: 2, capV: 8, config: "[Ne] 3s2", term: "1S0" },
  { sym: "Al", Z: 13, qv: 3, capV: 8, config: "[Ne] 3s2 3p1", term: "2P1/2" },
  { sym: "Si", Z: 14, qv: 4, capV: 8, config: "[Ne] 3s2 3p2", term: "3P0" },
  { sym: "P", Z: 15, qv: 5, capV: 8, config: "[Ne] 3s2 3p3", term: "4S3/2" },
  { sym: "S", Z: 16, qv: 6, capV: 8, config: "[Ne] 3s2 3p4", term: "3P2" },
  { sym: "Cl", Z: 17, qv: 7, capV: 8, config: "[Ne] 3s2 3p5", term: "2P3/2" },
  { sym: "Ar", Z: 18, qv: 8, capV: 8, config: "[Ne] 3s2 3p6", term: "1S0" }
];
var BY_Z = new Map(ELEMENTS.map((e) => [e.Z, e]));
function thickness(nu, floor, kappa = 1) {
  return floor + kappa * nu;
}
function individuate(Z, floor) {
  if (!Number.isInteger(Z) || Z < 1) throw new Error(`cut: atomic number must be a positive integer (got ${Z})`);
  const e = BY_Z.get(Z);
  if (!e) throw new Error(`cut: element Z=${Z} not in the light-element table (1..18 supported)`);
  const vacancy = e.capV - e.qv;
  const valence = Math.min(vacancy, e.capV - vacancy);
  const residue = thickness(vacancy, floor);
  return {
    ty: "Atom",
    Z,
    symbol: e.sym,
    config: e.config,
    term: e.term,
    qv: e.qv,
    capV: e.capV,
    vacancy,
    valence,
    floor,
    residue
  };
}
function bond(a, b, floor, kappa = 1) {
  const shared = Math.min(a.vacancy, b.vacancy);
  const sep = thickness(a.vacancy, floor, kappa) + thickness(b.vacancy, floor, kappa);
  const joined = thickness(Math.max(a.vacancy - shared, 0), floor, kappa) + thickness(Math.max(b.vacancy - shared, 0), floor, kappa);
  const delta = sep - joined;
  const exists = delta > 1e-12;
  return {
    ty: "Bond",
    a: a.symbol,
    b: b.symbol,
    delta,
    shared,
    exists,
    floor,
    residue: Math.max(delta, floor)
  };
}
var ANGLE_TET = Math.acos(-1 / 3) * 180 / Math.PI;
function close(central, ligands, floor) {
  if (ligands.length === 0) throw new Error("close: needs at least one ligand");
  const lig = ligands[0];
  if (central.symbol === lig.symbol) {
    return {
      ty: "Compound",
      central: central.symbol,
      ligand: lig.symbol,
      formula: [2, 0],
      ligands: 1,
      geometry: "linear",
      angleDeg: 180,
      valenceClosed: true,
      floor,
      residue: thickness(central.vacancy, floor)
    };
  }
  const vC = Math.max(central.valence, 1);
  const vL = Math.max(lig.valence, 1);
  const nLig = Math.max(Math.round(vC / vL), 1);
  const bondedDomains = nLig;
  const lonePairs = central.capV === 8 ? Math.max(0, Math.floor((central.qv - nLig) / 2)) : 0;
  const k = bondedDomains + lonePairs;
  let geometry = "point";
  let angleDeg = null;
  if (k === 1) {
    geometry = "terminal";
    angleDeg = null;
  } else if (k === 2) {
    geometry = lonePairs === 0 ? "linear" : "bent";
    angleDeg = lonePairs === 0 ? 180 : ANGLE_TET;
  } else if (k === 3) {
    geometry = lonePairs === 0 ? "trigonal" : "bent";
    angleDeg = lonePairs === 0 ? 120 : 117;
  } else if (k >= 4) {
    geometry = lonePairs === 0 ? "tetrahedral" : lonePairs === 1 ? "pyramidal" : "bent";
    angleDeg = lonePairs === 0 ? round2(ANGLE_TET) : lonePairs === 1 ? 107 : 104.5;
  }
  return {
    ty: "Compound",
    central: central.symbol,
    ligand: lig.symbol,
    formula: [1, nLig],
    ligands: nLig,
    geometry,
    angleDeg,
    valenceClosed: true,
    floor,
    residue: thickness(0, floor) + nLig * floor
  };
}
function round2(x) {
  return Math.round(x * 100) / 100;
}
function propagate(item, process, reps, admit, floor) {
  const contacts = [];
  let steps = 0;
  let residue = 0;
  if (process.ty === "Compound") {
    const n = process.ligands;
    for (let i = 0; i < n; i++) {
      contacts.push(`${process.central}~${process.ligand}#${i + 1}`);
      steps += 1;
      residue += floor;
    }
  } else {
    steps = process.steps;
    residue = process.residue;
    contacts.push(...process.amalgamation);
  }
  let converged;
  if (admit === "converge") converged = residue >= floor && steps > 0;
  else if (admit === "diverge") converged = !(residue >= floor && steps > 0);
  else converged = admit.holds;
  return {
    ty: "Path",
    item: item.symbol,
    steps,
    converged,
    amalgamation: converged ? contacts : [],
    reps: reps.length ? reps : ["mass"],
    floor,
    residue
  };
}

// src/interp.ts
var RuntimeError = class extends Error {
  constructor(msg) {
    super(`runtime error: ${msg}`);
    this.name = "RuntimeError";
  }
};
function run(ir) {
  const regs = [];
  const named = {};
  const log = [];
  let floor = Number.isNaN(ir.floor) ? 1 : ir.floor;
  let M = 0;
  const regToName = /* @__PURE__ */ new Map();
  for (const [name, reg] of ir.nameToReg) regToName.set(reg, name);
  const getReg = (id) => {
    const v = regs[id];
    if (v === void 0) throw new RuntimeError(`register r${id} read before write`);
    return v;
  };
  const asAtom = (id) => {
    const v = getReg(id);
    if (v.ty !== "Atom") throw new RuntimeError(`expected Atom in r${id}, got ${v.ty}`);
    return v;
  };
  const numOperand = (o) => {
    switch (o.kind) {
      case "lit":
        return o.value;
      case "reg": {
        const v = getReg(o.id);
        return "value" in v ? v.value : v.residue ?? NaN;
      }
      case "field":
        return fieldValue(getReg(o.reg), o.field);
      case "name":
        throw new RuntimeError(`unresolved measured field '${o.name}' (no value in scope)`);
    }
  };
  const fieldValue = (v, field) => {
    const anyv = v;
    if (field === "valence" || field === "closed") {
      if (v.ty === "Compound") return v.valenceClosed ? 1 : 0;
      if (v.ty === "Atom") return v.vacancy === 0 ? 1 : 0;
    }
    if (field in anyv && typeof anyv[field] === "number") return anyv[field];
    throw new RuntimeError(`value of type ${v.ty} has no numeric field '${field}'`);
  };
  const evalCond = (c, ctx) => {
    const resolve = (o) => {
      if (o.kind === "name" && ctx) {
        try {
          return fieldValue(ctx, o.name);
        } catch {
        }
      }
      if (o.kind === "name") {
        if (o.name === "closed") return ctx ? fieldValue(ctx, "valence") : 1;
      }
      return numOperand(o);
    };
    const L = resolve(c.left), R = resolve(c.right);
    switch (c.op) {
      case ">":
        return L > R;
      case "<":
        return L < R;
      case ">=":
        return L >= R;
      case "<=":
        return L <= R;
      case "==":
        return L === R;
    }
  };
  for (const ins of ir.instrs) {
    switch (ins.op) {
      case "Floor":
        if (!(ins.value > 0)) throw new RuntimeError("floor must be > 0");
        floor = ins.value;
        break;
      case "Mov":
        regs[ins.dst] = scalarFromOperand(ins.src, floor, numOperand);
        break;
      case "Ind": {
        const z = numOperand(ins.z);
        const atom = individuate(z, floor);
        regs[ins.dst] = atom;
        M += 1;
        break;
      }
      case "Bnd": {
        const a = asAtom(ins.a), b = asAtom(ins.b);
        const bd = bond(a, b, floor);
        if (ins.guard && !evalCond(ins.guard, bd)) {
          regs[ins.dst] = { ...bd, exists: false };
          break;
        }
        regs[ins.dst] = bd;
        M += 1;
        break;
      }
      case "Cls": {
        const central = asAtom(ins.central);
        const ligs = ins.args.map((r) => asAtom(r));
        const comp = close(central, ligs, floor);
        regs[ins.dst] = comp;
        M += comp.ligands;
        break;
      }
      case "Prp": {
        const item = asAtom(ins.item);
        const proc = getReg(ins.process);
        if (proc.ty !== "Compound" && proc.ty !== "Path")
          throw new RuntimeError(`track process must be Compound or Path, got ${proc.ty}`);
        const admit = ins.admit.kind === "converge" ? "converge" : ins.admit.kind === "diverge" ? "diverge" : { holds: evalCond(ins.admit.cond, proc) };
        const path = propagate(item, proc, ins.reps ?? [], admit, floor);
        regs[ins.dst] = path;
        M += path.steps;
        break;
      }
      case "Obs": {
        const v = getReg(ins.reg);
        log.push(renderValue(v, regToName.get(ins.reg)));
        if (ins.as !== void 0) regs[ins.as] = v;
        break;
      }
      case "Assert": {
        const ctx = lastCutContext(regs);
        if (!evalCond(ins.cond, ctx)) {
          log.push(`ABORT: ${ins.message ?? "assertion failed"}`);
          return finalize(false);
        }
        break;
      }
      case "Call": {
        const verb = ins.name[ins.name.length - 1];
        if (verb === "individuate" || verb === "atom") {
          const z = numOperand(ins.args[0]);
          regs[ins.dst] = individuate(z, floor);
          M += 1;
        } else {
          throw new RuntimeError(`unknown call '${ins.name.join(".")}'`);
        }
        break;
      }
    }
  }
  return finalize(true);
  function finalize(ok) {
    for (const [name, reg] of ir.nameToReg) {
      const v = regs[reg];
      if (v !== void 0) named[name] = v;
    }
    return { cutCount: M, floor, registers: regs, named, log, ok };
  }
}
function scalarFromOperand(o, floor, num) {
  if (o.kind === "lit") return { ty: "Scalar", value: o.value, floor: o.floor || floor };
  return { ty: "Scalar", value: num(o), floor };
}
function lastCutContext(regs) {
  for (let i = regs.length - 1; i >= 0; i--) {
    const v = regs[i];
    if (v && v.ty !== "Scalar") return v;
  }
  return void 0;
}
function renderValue(v, name) {
  const tag = name ? `${name} : ` : "";
  switch (v.ty) {
    case "Atom":
      return `${tag}Atom @ ${fmt(v.floor)}  Z=${v.Z} ${v.symbol}  ${v.config}  ${v.term}  vacancy=${v.vacancy}  valence=${v.valence}  residue=${fmt(v.residue)}`;
    case "Bond":
      return `${tag}Bond @ ${fmt(v.floor)}  ${v.a}~${v.b}  exists=${v.exists}  delta=${fmt(v.delta)}  shared=${v.shared}  residue=${fmt(v.residue)}`;
    case "Compound": {
      const lig = v.formula[1] > 1 ? v.ligand + v.formula[1] : v.formula[1] === 1 ? v.ligand : "";
      const formula = v.formula[0] === 2 ? v.central + "2" : v.central + lig;
      return `${tag}Compound @ ${fmt(v.floor)}  ${formula}  geometry=${v.geometry}  angle=${v.angleDeg ?? "-"}  closed=${v.valenceClosed}  residue=${fmt(v.residue)}`;
    }
    case "Path":
      return `${tag}Path @ ${fmt(v.floor)}  item=${v.item}  steps=${v.steps}  converged=${v.converged}  reps=[${v.reps.join(",")}]  amalgamation=[${v.amalgamation.join(", ")}]  residue=${fmt(v.residue)}`;
    case "Scalar":
      return `${tag}Scalar @ ${fmt(v.floor)}  ${fmt(v.value)}`;
  }
}
function fmt(n) {
  if (Number.isNaN(n)) return "NaN";
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

// src/index.ts
function compile(src) {
  const program = parse(src);
  check(program);
  const ir = lower(program);
  return { program, ir };
}
function evaluate(src) {
  const { ir } = compile(src);
  return run(ir);
}
function exec(src) {
  const r = evaluate(src);
  return { log: r.log, cutCount: r.cutCount, ok: r.ok };
}
export {
  check,
  compile,
  evaluate,
  exec,
  lex,
  lower,
  parse,
  renderValue,
  run
};
