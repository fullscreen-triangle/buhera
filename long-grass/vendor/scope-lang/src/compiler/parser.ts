// SCOPE Parser — recursive descent, all blocks optional per spec

import { Lexer, Token, TokenType } from './lexer';
import type {
  ScopeProgram, ChannelsDecl, ChannelItem, SyncItem, CellItem,
  CoordinateSpaceDecl, GoalDecl, GoalCriterion, RuleDecl,
  MorphismDecl, MorphismExpr, ObserveExpr, FrameRef,
  MorphismStep, CatalyzeStep, FuseStep, MeasureDistanceStep,
  AccessStep, VisualiseStep, VisMode,
  DispatchDecl, WhenStmt, Action,
} from './ast';

export type { ScopeProgram } from './ast';

export class ParseError extends Error {
  constructor(msg: string, public line: number, public col: number) {
    super(msg);
    this.name = 'ParseError';
  }
}

export class Parser {
  private tokens: Token[];
  private pos = 0;

  constructor(tokens: Token[]) { this.tokens = tokens; }

  static fromSource(src: string): Parser {
    return new Parser(new Lexer(src).tokenize());
  }

  parse(): ScopeProgram {
    this.expect(TokenType.SCOPE);
    const name = this.expectIdent();
    this.expect(TokenType.LBRACE);

    const body = this.parseBody(() => this.check(TokenType.RBRACE));

    this.expect(TokenType.RBRACE);
    return { kind: 'ScopeProgram', name, ...body };
  }

  /**
   * Parse un-wrapped top-level declarations — one REPL cell. Accepts the same
   * six constructs as a scope{} body (channels, coordinate_space, goal, rule,
   * dispatch, morphism), without the `scope name { ... }` wrapper. Used by the
   * session runtime to build a program incrementally.
   */
  parseFragment(name = 'repl'): ScopeProgram {
    const body = this.parseBody(() => this.check(TokenType.EOF));
    this.expect(TokenType.EOF);
    return { kind: 'ScopeProgram', name, ...body };
  }

  private parseBody(atEnd: () => boolean): {
    channels?: ChannelsDecl;
    coordinateSpace?: CoordinateSpaceDecl;
    goal?: GoalDecl;
    rules: RuleDecl[];
    morphisms: MorphismDecl[];
    dispatch?: DispatchDecl;
  } {
    let channels: ChannelsDecl | undefined;
    let coordinateSpace: CoordinateSpaceDecl | undefined;
    let goal: GoalDecl | undefined;
    const rules: RuleDecl[] = [];
    const morphisms: MorphismDecl[] = [];
    let dispatch: DispatchDecl | undefined;

    while (!atEnd() && !this.check(TokenType.EOF)) {
      if (this.check(TokenType.CHANNELS)) {
        channels = this.parseChannels();
      } else if (this.check(TokenType.COORDINATE_SPACE)) {
        coordinateSpace = this.parseCoordinateSpace();
      } else if (this.check(TokenType.GOAL)) {
        goal = this.parseGoal();
      } else if (this.check(TokenType.RULE)) {
        rules.push(this.parseRule());
      } else if (this.check(TokenType.DISPATCH)) {
        dispatch = this.parseDispatch();
      } else if (this.check(TokenType.IDENT)) {
        morphisms.push(this.parseMorphism());
      } else {
        const t = this.cur();
        throw new ParseError(`Unexpected token '${t.value}' in scope body`, t.line, t.col);
      }
    }

    return { channels, coordinateSpace, goal, rules, morphisms, dispatch };
  }

  // ── channels ────────────────────────────────────────────────────────────────

  private parseChannels(): ChannelsDecl {
    this.expect(TokenType.CHANNELS);
    this.expect(TokenType.LBRACE);
    const items: ChannelItem[] = [];
    while (!this.check(TokenType.RBRACE) && !this.check(TokenType.EOF)) {
      if (this.check(TokenType.SYNC)) items.push(this.parseSyncItem());
      else if (this.check(TokenType.CELL)) items.push(this.parseCellItem());
      else { const t = this.cur(); throw new ParseError(`Expected sync or cell, got '${t.value}'`, t.line, t.col); }
    }
    this.expect(TokenType.RBRACE);
    return { kind: 'ChannelsDecl', items };
  }

  private parseSyncItem(): SyncItem {
    this.expect(TokenType.SYNC);
    const name = this.expectIdent();
    this.expect(TokenType.AT);
    const value = this.expectNumber();
    const unit = this.expectUnit();
    return { kind: 'SyncItem', name, value, unit };
  }

  private parseCellItem(): CellItem {
    this.expect(TokenType.CELL);
    const name = this.expectIdent();
    this.expect(TokenType.BOUNDS);
    this.expect(TokenType.LPAREN);
    const boundsLow = this.expectNumber();
    this.expect(TokenType.COMMA);
    const boundsHigh = this.expectNumber();
    this.expect(TokenType.RPAREN);
    this.expect(TokenType.ACTION);
    const action = this.expectIdent();
    return { kind: 'CellItem', name, boundsLow, boundsHigh, action };
  }

  // ── coordinate_space ─────────────────────────────────────────────────────────

  private parseCoordinateSpace(): CoordinateSpaceDecl {
    this.expect(TokenType.COORDINATE_SPACE);
    this.expect(TokenType.LBRACE);
    this.expect(TokenType.FIELD);
    const fieldX = this.expectNumber();
    this.expect(TokenType.X);
    const fieldY = this.expectNumber();
    const unit = this.expectUnit();
    this.expect(TokenType.DEPTH);
    const depth = Math.round(this.expectNumber());
    this.expect(TokenType.LAMBDA_S);
    const lambdaS = this.expectNumber();
    this.expect(TokenType.LAMBDA_T);
    const lambdaT = this.expectNumber();
    this.expect(TokenType.RBRACE);
    return { kind: 'CoordinateSpaceDecl', fieldX, fieldY, unit, depth, lambdaS, lambdaT };
  }

  // ── goal [ext] ────────────────────────────────────────────────────────────────

  private parseGoal(): GoalDecl {
    this.expect(TokenType.GOAL);
    this.expect(TokenType.LBRACE);
    const criteria: GoalCriterion[] = [];
    while (!this.check(TokenType.RBRACE) && !this.check(TokenType.EOF)) {
      criteria.push(this.parseGoalCriterion());
    }
    this.expect(TokenType.RBRACE);
    return { kind: 'GoalDecl', criteria };
  }

  private parseGoalCriterion(): GoalCriterion {
    const metric = this.expectIdent();
    const op = this.parseComparisonOp();
    const threshold = this.expectNumber();
    // optional unit
    let unit = '';
    if (this.check(TokenType.UNIT)) unit = this.advance().value;
    return { kind: 'GoalCriterion', metric, op, threshold, unit };
  }

  private parseComparisonOp(): '<' | '<=' | '>' | '>=' | '==' {
    const t = this.cur();
    if (this.check(TokenType.LTE)) { this.advance(); return '<='; }
    if (this.check(TokenType.GTE)) { this.advance(); return '>='; }
    if (this.check(TokenType.LT)) { this.advance(); return '<'; }
    if (this.check(TokenType.GT)) { this.advance(); return '>'; }
    if (this.check(TokenType.EQEQ)) { this.advance(); return '=='; }
    throw new ParseError(`Expected comparison operator, got '${t.value}'`, t.line, t.col);
  }

  // ── rule [ext] ────────────────────────────────────────────────────────────────

  private parseRule(): RuleDecl {
    this.expect(TokenType.RULE);
    const name = this.expectIdent();
    this.expect(TokenType.LPAREN);
    const argument = this.expectIdent();
    this.expect(TokenType.RPAREN);
    this.expect(TokenType.LBRACE);
    this.expect(TokenType.INVARIANT);
    this.expect(TokenType.COLON);
    const invariant = this.expectString();
    this.expect(TokenType.EPSILON);
    this.expect(TokenType.COLON);
    const epsilon = this.expectNumber();
    this.expect(TokenType.RBRACE);
    return { kind: 'RuleDecl', name, argument, invariant, epsilon };
  }

  // ── morphisms ─────────────────────────────────────────────────────────────────

  private parseMorphism(): MorphismDecl {
    const name = this.expectIdent();
    this.expect(TokenType.EQUALS);
    const expr = this.parseMorphismExpr();
    return { kind: 'MorphismDecl', name, expr };
  }

  private parseMorphismExpr(): MorphismExpr {
    const observe = this.parseObserve();
    const steps: MorphismStep[] = [];
    while (this.check(TokenType.PIPE)) {
      this.advance();
      steps.push(this.parseMorphismStep());
    }
    return { kind: 'MorphismExpr', observe, steps };
  }

  private parseObserve(): ObserveExpr {
    this.expect(TokenType.OBSERVE);
    this.expect(TokenType.LPAREN);
    const frame = this.parseFrameRef();
    this.expect(TokenType.COMMA);
    this.expect(TokenType.N);
    this.expect(TokenType.EQUALS);
    const depth = Math.round(this.expectNumber());
    this.expect(TokenType.RPAREN);
    return { kind: 'ObserveExpr', frame, depth };
  }

  private parseFrameRef(): FrameRef {
    if (this.check(TokenType.LOAD)) {
      this.advance();
      this.expect(TokenType.LPAREN);
      this.expect(TokenType.DB);
      this.expect(TokenType.EQUALS);
      const db = this.expectString();
      this.expect(TokenType.COMMA);
      this.expect(TokenType.DATASET);
      this.expect(TokenType.EQUALS);
      const dataset = this.expectString();
      this.expect(TokenType.COMMA);
      this.expect(TokenType.IMAGE);
      this.expect(TokenType.EQUALS);
      const image = this.expectString();
      this.expect(TokenType.RPAREN);
      return { kind: 'LoadRef', db, dataset, image };
    } else {
      const name = this.expectIdent();
      return { kind: 'ChannelRef', name };
    }
  }

  private parseMorphismStep(): MorphismStep {
    if (this.check(TokenType.CATALYZE)) return this.parseCatalyze();
    if (this.check(TokenType.FUSE)) return this.parseFuse();
    if (this.check(TokenType.MEASURE_DISTANCE)) return this.parseMeasureDistance();
    if (this.check(TokenType.ACCESS)) return this.parseAccess();
    if (this.check(TokenType.VISUALISE)) return this.parseVisualise();
    const t = this.cur();
    throw new ParseError(`Expected morphism step, got '${t.value}'`, t.line, t.col);
  }

  private parseCatalyze(): CatalyzeStep {
    this.expect(TokenType.CATALYZE);
    this.expect(TokenType.LPAREN);
    const constraintName = this.expectIdent();
    this.expect(TokenType.LPAREN);
    const constraintArg = this.expectIdent();
    this.expect(TokenType.RPAREN);
    let confidence = 1.0;
    if (this.check(TokenType.COMMA)) {
      this.advance();
      this.expect(TokenType.CONFIDENCE);
      this.expect(TokenType.EQUALS);
      confidence = this.expectNumber();
    }
    this.expect(TokenType.RPAREN);
    return { kind: 'CatalyzeStep', constraintName, constraintArg, epsilon: 0, confidence };
  }

  private parseFuse(): FuseStep {
    this.expect(TokenType.FUSE);
    this.expect(TokenType.LPAREN);
    const morphismRef = this.expectIdent();
    this.expect(TokenType.COMMA);
    this.expect(TokenType.RHO);
    this.expect(TokenType.EQUALS);
    const rho = this.expectNumber();
    this.expect(TokenType.RPAREN);
    return { kind: 'FuseStep', morphismRef, rho };
  }

  private parseMeasureDistance(): MeasureDistanceStep {
    this.expect(TokenType.MEASURE_DISTANCE);
    this.expect(TokenType.LPAREN);
    const target1 = this.expectIdent();
    this.expect(TokenType.COMMA);
    const target2 = this.expectIdent();
    this.expect(TokenType.RPAREN);
    return { kind: 'MeasureDistanceStep', target1, target2 };
  }

  private parseAccess(): AccessStep {
    this.expect(TokenType.ACCESS);
    this.expect(TokenType.LPAREN);
    const target = this.expectIdent();
    let threshold = 0.5;
    if (this.check(TokenType.COMMA)) {
      this.advance();
      this.expect(TokenType.THRESHOLD);
      this.expect(TokenType.EQUALS);
      threshold = this.expectNumber();
    }
    this.expect(TokenType.RPAREN);
    return { kind: 'AccessStep', target, threshold };
  }

  private parseVisualise(): VisualiseStep {
    this.expect(TokenType.VISUALISE);
    this.expect(TokenType.LPAREN);
    const t = this.cur();
    const raw = t.type === TokenType.IDENT ? this.advance().value : (() => { const v = this.advance().value; return v; })();
    const VALID_MODES: VisMode[] = [
      'raw_image', 'scale_field', 'segmentation', 'distance_map', 'geodesic',
      'point_cloud', 'entropy_sphere', 'partition_tree', 'distance_tube',
      'spectral_power', 'entropy_trajectory', 'uncertainty_bar',
      'scale_histogram',
    ];
    if (!(VALID_MODES as string[]).includes(raw)) {
      throw new ParseError(`Unknown visualise mode '${raw}'`, t.line, t.col);
    }
    this.expect(TokenType.RPAREN);
    return { kind: 'VisualiseStep', mode: raw as VisMode };
  }

  // ── dispatch ──────────────────────────────────────────────────────────────────

  private parseDispatch(): DispatchDecl {
    this.expect(TokenType.DISPATCH);
    this.expect(TokenType.LBRACE);
    const rules: WhenStmt[] = [];
    while (!this.check(TokenType.RBRACE) && !this.check(TokenType.EOF)) {
      this.expect(TokenType.WHEN);
      const cell = this.expectIdent();
      this.expect(TokenType.DO);
      const action = this.parseAction();
      rules.push({ kind: 'WhenStmt', cell, action });
    }
    this.expect(TokenType.RBRACE);
    return { kind: 'DispatchDecl', rules };
  }

  private parseAction(): Action {
    if (this.check(TokenType.EXECUTE)) {
      this.advance();
      this.expect(TokenType.LPAREN);
      const morphismRef = this.expectIdent();
      this.expect(TokenType.RPAREN);
      return { kind: 'ExecuteAction', morphismRef };
    }
    if (this.check(TokenType.EMIT)) {
      this.advance();
      const name = this.expectIdent();
      return { kind: 'EmitAction', name };
    }
    if (this.check(TokenType.LBRACE)) {
      this.advance();
      const actions: Action[] = [];
      while (!this.check(TokenType.RBRACE) && !this.check(TokenType.EOF)) {
        actions.push(this.parseAction());
      }
      this.expect(TokenType.RBRACE);
      return { kind: 'BlockAction', actions };
    }
    const t = this.cur();
    throw new ParseError(`Expected execute/emit/block, got '${t.value}'`, t.line, t.col);
  }

  // ── helpers ───────────────────────────────────────────────────────────────────

  private cur(): Token { return this.tokens[this.pos] ?? { type: TokenType.EOF, value: '', line: 0, col: 0 }; }
  private check(type: TokenType): boolean { return this.cur().type === type; }
  private advance(): Token { return this.tokens[this.pos++] ?? { type: TokenType.EOF, value: '', line: 0, col: 0 }; }

  private expect(type: TokenType): Token {
    const t = this.cur();
    if (t.type !== type) throw new ParseError(`Expected '${type}', got '${t.value}'`, t.line, t.col);
    return this.advance();
  }

  private expectIdent(): string {
    const t = this.cur();
    if (t.type !== TokenType.IDENT) throw new ParseError(`Expected identifier, got '${t.value}'`, t.line, t.col);
    return this.advance().value;
  }

  private expectNumber(): number {
    const t = this.cur();
    if (t.type !== TokenType.NUMBER) throw new ParseError(`Expected number, got '${t.value}'`, t.line, t.col);
    this.advance();
    return parseFloat(t.value);
  }

  private expectString(): string {
    const t = this.cur();
    if (t.type !== TokenType.STRING) throw new ParseError(`Expected string, got '${t.value}'`, t.line, t.col);
    this.advance();
    return t.value;
  }

  private expectUnit(): string {
    const t = this.cur();
    if (t.type === TokenType.UNIT) { this.advance(); return t.value; }
    if (t.type === TokenType.IDENT) { this.advance(); return t.value; }
    throw new ParseError(`Expected unit, got '${t.value}'`, t.line, t.col);
  }
}
