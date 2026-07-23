/**
 * SRN source parser.
 *
 * Parses the glyph grammar:
 *   |name : (n,l,m,s)| not { guard } do { body } to { target } [as { alias }]
 *
 * Enforces the Mandatory Negation Boundary (SRN Cor 2.1): a glyph with no `not`
 * clause is not individuated and is rejected. Returns a typed ParseError rather
 * than throwing (core discipline), so callers can surface
 * `no-negation-boundary` / `malformed-srn` cleanly.
 */
import { coordsFromTuple, isCoordError } from "../coords.js";
import { glyph } from "./expr.js";
/** True if the parse produced an error rather than a glyph. */
export function isParseError(x) {
    return x.kind === "no-negation-boundary" || x.kind === "malformed-srn";
}
/**
 * Parse a single SRN glyph expression from source text.
 *
 * A missing `not { ... }` clause yields { kind: "no-negation-boundary" }; any
 * other structural failure yields { kind: "malformed-srn", at, message }.
 */
export function parseSrn(source) {
    const src = source;
    // ---- header: |name : (n,l,m,s)| ----
    const header = /\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*([+-]?\d*|\+|-)\s*\)\s*\|/.exec(src);
    if (!header || header.index !== firstNonSpace(src)) {
        return { kind: "malformed-srn", at: 0, message: "expected header |name : (n,l,m,s)|" };
    }
    const name = header[1];
    const nlms = [
        Number(header[2]),
        Number(header[3]),
        Number(header[4]),
        normSpin(header[5]),
    ];
    const coords = coordsFromTuple(nlms);
    if (isCoordError(coords)) {
        return { kind: "malformed-srn", at: header.index, message: coords.message };
    }
    const rest = src.slice(header.index + header[0].length);
    // ---- clauses: not{...} do{...} to{...} [as{...}] ----
    const notClause = braceClause(rest, "not");
    if (notClause === null) {
        // No `not` clause at all -> not individuated (SRN Cor 2.1).
        return { kind: "no-negation-boundary" };
    }
    const doClause = braceClause(rest, "do");
    if (doClause === null) {
        return { kind: "malformed-srn", at: header.index, message: "missing `do { ... }` clause" };
    }
    const toClause = braceClause(rest, "to");
    if (toClause === null) {
        return { kind: "malformed-srn", at: header.index, message: "missing `to { ... }` clause" };
    }
    const asClause = braceClause(rest, "as"); // optional
    return glyph({
        name,
        target: coords,
        notGuard: notClause.trim(),
        body: doClause.trim(),
        toTarget: toClause.trim(),
        ...(asClause !== null ? { alias: asClause.trim() } : {}),
    });
}
/** Extract the balanced contents of `keyword { ... }`, or null if absent. */
function braceClause(text, keyword) {
    const re = new RegExp(`\\b${keyword}\\b\\s*\\{`);
    const m = re.exec(text);
    if (!m)
        return null;
    const open = m.index + m[0].length - 1; // index of the '{'
    let depth = 0;
    for (let i = open; i < text.length; i++) {
        const ch = text[i];
        if (ch === "{")
            depth++;
        else if (ch === "}") {
            depth--;
            if (depth === 0)
                return text.slice(open + 1, i);
        }
    }
    return null; // unbalanced braces -> treat as absent for this keyword
}
function firstNonSpace(s) {
    const m = /\S/.exec(s);
    return m ? m.index : 0;
}
/** Normalise a spin token ("+", "-", "+1", "-1", "1") to +1 or -1. */
function normSpin(tok) {
    if (tok === "+" || tok === "+1" || tok === "1")
        return 1;
    if (tok === "-" || tok === "-1")
        return -1;
    const n = Number(tok);
    return n >= 0 ? 1 : -1;
}
//# sourceMappingURL=parse.js.map