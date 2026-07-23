/**
 * SRN expression model — glyph syntax, receiver-relative evaluation.
 *
 * Grammar (simplified; full grammar in the SRN paper §3):
 *   expr  ::= glyph | composed | literal
 *   glyph ::= |name : (n,l,m,s)| not { guard } do { body } to { target } [as { alias }]
 *
 * Receiver-relative evaluation (SRN Thm 4.2): the same glyph produces different
 * output at every node, because `self` binds to the evaluating node's coordinates.
 *
 * Every glyph carries a mandatory `not` boundary — individuation by negation
 * (SRN Cor 2.1), a structural requirement of the language.
 *
 * Ported from `crates/srn-node/src/expression.rs`.
 */
import type { PartitionCoords } from "../coords.js";
/** The evaluating node's identity at evaluation time. */
export interface ReceiverFrame {
    readonly coords: PartitionCoords;
    /** Monotone committed-step count M(t) at evaluation time. */
    readonly trajectoryCount: number;
}
/** The four primitive SRN operators. */
export type Operator = "compose" | "catalyst" | "parallel" | "sequential";
/** A glyph — the transmission unit of SRN. */
export interface Glyph {
    readonly kind: "glyph";
    readonly name: string;
    /** Target partition address (which region this is addressed to). */
    readonly target: PartitionCoords;
    /** The `not` boundary — what this expression is NOT. Mandatory (SRN Cor 2.1). */
    readonly notGuard: string;
    /** The `do` clause — what to evaluate. */
    readonly body: string;
    /** The `to` clause — where the result goes (region text). */
    readonly toTarget: string;
    /** Optional `as` clause — install under this registry key on evaluation. */
    readonly alias?: string;
}
/** A composed expression — two sub-expressions joined by an operator. */
export interface ComposedExpr {
    readonly kind: "composed";
    readonly left: Expr;
    readonly op: Operator;
    readonly right: Expr;
}
/** A literal value expression. */
export interface LiteralExpr {
    readonly kind: "literal";
    readonly value: unknown;
}
/** Top-level SRN expression AST. */
export type Expr = Glyph | ComposedExpr | LiteralExpr;
/** Result of evaluating an SRN expression at a receiver frame. */
export type EvalResult = {
    kind: "value";
    value: unknown;
} | {
    kind: "rejected";
    reason: string;
} | {
    kind: "emit";
    target: PartitionCoords;
    payload: unknown;
} | {
    kind: "forward";
    target: PartitionCoords;
    expr: Expr;
} | {
    kind: "error";
    message: string;
};
/** Evaluation environment — bindings visible in this receiver frame. */
export type Env = ReadonlyMap<string, unknown>;
/** Construct a glyph. `notGuard` is mandatory (SRN Cor 2.1). */
export declare function glyph(args: {
    name: string;
    target: PartitionCoords;
    notGuard: string;
    body: string;
    toTarget: string;
    alias?: string;
}): Glyph;
/** Compose two expressions with an operator. */
export declare function composed(left: Expr, op: Operator, right: Expr): ComposedExpr;
/** A literal expression. */
export declare function literal(value: unknown): LiteralExpr;
/**
 * Receiver-relative evaluation. The result depends on the evaluating node's
 * frame — same expression, different nodes, different (all simultaneously valid)
 * outputs (SRN Thm 4.2). Deterministic per (expr, frame, env) (SRN Thm 4.1).
 */
export declare function evalExpr(expr: Expr, frame: ReceiverFrame, env: Env): EvalResult;
/** Deterministic JSON serialisation for content addressing (stable key order). */
export declare function serializeExpr(expr: Expr): string;
//# sourceMappingURL=expr.d.ts.map