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
/** Construct a glyph. `notGuard` is mandatory (SRN Cor 2.1). */
export function glyph(args) {
    return {
        kind: "glyph",
        name: args.name,
        target: args.target,
        notGuard: args.notGuard,
        body: args.body,
        toTarget: args.toTarget,
        ...(args.alias !== undefined ? { alias: args.alias } : {}),
    };
}
/** Compose two expressions with an operator. */
export function composed(left, op, right) {
    return { kind: "composed", left, op, right };
}
/** A literal expression. */
export function literal(value) {
    return { kind: "literal", value };
}
/**
 * Receiver-relative evaluation. The result depends on the evaluating node's
 * frame — same expression, different nodes, different (all simultaneously valid)
 * outputs (SRN Thm 4.2). Deterministic per (expr, frame, env) (SRN Thm 4.1).
 */
export function evalExpr(expr, frame, env) {
    switch (expr.kind) {
        case "literal":
            return { kind: "value", value: expr.value };
        case "glyph":
            return evalGlyph(expr, frame, env);
        case "composed":
            return evalComposed(expr, frame, env);
    }
}
function evalGlyph(g, frame, env) {
    // Receiver-relative guard check — the `not` boundary. In this reference
    // implementation the guard is a simple env-key presence test; a full
    // implementation would parse and evaluate an SRN guard expression against
    // the frame. If the guard key is present, the node IS in the boundary region
    // (i.e. it is what the expression is NOT) and the glyph is rejected.
    if (env.get(g.notGuard) !== undefined) {
        return {
            kind: "rejected",
            reason: `guard '${g.notGuard}' fired at ${coordStr(frame.coords)}`,
        };
    }
    // Receiver-relative body: `self` is this node's coords. We encode the frame
    // into the output so callers can observe Receiver-Relativity: same glyph,
    // different coords -> different result.
    const value = {
        glyph: g.name,
        self_n: frame.coords.n,
        self_l: frame.coords.l,
        self_m: frame.coords.m,
        self_s: frame.coords.s,
        self_M: frame.trajectoryCount,
        body: g.body,
    };
    return { kind: "value", value };
}
function evalComposed(c, frame, env) {
    const left = evalExpr(c.left, frame, env);
    const right = evalExpr(c.right, frame, env);
    switch (c.op) {
        case "sequential":
            // Left runs, then right; result is right's — unless left was rejected.
            return left.kind === "rejected" ? left : right;
        case "compose":
            // Cross-representation composition (SRN Cor 2.2 apples-and-oranges): the
            // receiver unifies both into a common representation and combines them.
            if (left.kind === "value" && right.kind === "value") {
                return { kind: "value", value: { composed: [left.value, right.value] } };
            }
            if (left.kind === "rejected")
                return left;
            if (right.kind === "rejected")
                return right;
            return right;
        case "catalyst":
            // Right acts on left's residual; catalytic power composes multiplicatively
            // (SRN Thm 5.x): kappa_comb = 1 - (1-k1)(1-k2). Here we carry the pair.
            if (left.kind === "value" && right.kind === "value") {
                return { kind: "value", value: { catalysed: { base: left.value, catalyst: right.value } } };
            }
            if (left.kind === "rejected")
                return right;
            return left;
        case "parallel":
            // Both sides run independently; both results are valid simultaneously.
            if (left.kind === "value" && right.kind === "value") {
                return { kind: "value", value: { parallel: [left.value, right.value] } };
            }
            if (left.kind === "rejected")
                return right;
            return left;
    }
}
function coordStr(c) {
    return `(n=${c.n}, l=${c.l}, m=${c.m}, s=${c.s > 0 ? "+" : "-"})`;
}
/** Deterministic JSON serialisation for content addressing (stable key order). */
export function serializeExpr(expr) {
    return JSON.stringify(expr, stableReplacer(expr));
}
// Produce a replacer that emits object keys in sorted order for stable digests.
function stableReplacer(_root) {
    return function (_key, value) {
        if (value && typeof value === "object" && !Array.isArray(value)) {
            const rec = value;
            const sorted = {};
            for (const k of Object.keys(rec).sort())
                sorted[k] = rec[k];
            return sorted;
        }
        return value;
    };
}
//# sourceMappingURL=expr.js.map